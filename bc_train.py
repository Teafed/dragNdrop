"""
bc_train.py

behavior cloning trainer for the shape manipulation agent.

--- bicameral network architecture ---
   the policy is split into two streams that each process a different
   slice of the observation, then cross-attend before producing actions.

   left stream  (cursor-local, manipulation):
      input:  obs[0:44]  — cursor state + grabbed shape + nearest shape + all shapes
      purpose: "what is the cursor doing right now, and what is it near?"

   right stream (scene-global, relational):
      input:  obs[14:108] — all shapes + goal encoding
      purpose: "where is everything and what does the goal say?"

   overlap on obs[14:43] (all shapes) is intentional — both streams see
   the full shape layout but with different contextual emphasis.

   cross-attention:
      right stream queries the left stream output.
      allows global context to read fine-grained cursor state.
      single-head attention, lightweight (no positional encoding needed).

   action head:
      movement head: (left + right) -> [dx, dy]  — MSE loss, Tanh output
      grip head:     (left + right) -> grip logit — BCE loss, no Tanh.
      at runtime the grip logit is thresholded at 0.0 to produce ±1.0.

--- BC training details ---
   loss = MSE on all three action outputs (dx, dy, grip together).
   grip is continuous during BC even though it becomes binary at runtime.
   separate loss reporting for grip vs dx/dy for diagnostics.
   cosine annealing lr schedule, gradient clipping max_norm=1.0.

--- weight transplant ---
   SB3's MlpPolicy does not support the bicameral architecture.
   build_ppo_from_bc() creates a CustomActorCriticPolicy that wraps the
   bicameral network inside SB3's actor-critic framework. this lets us
   keep SB3's PPO training loop while using our custom network.

   transplant 1: copy BicameralNetwork weights into _BicameralExtractor.net
                 inside the PPO policy. this is a direct state_dict copy
                 since the architectures match exactly.

   transplant 2: compose BC's split heads (move_head: hidden*2->hidden->2,
                 grip_head: hidden*2->hidden//2->1) into SB3's single
                 action_net (hidden*2->3) by composing each head's linear
                 layers independently and stacking the results.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.monitor import Monitor

from shape_env import ShapeEnv
from config import (
   EMBEDDING_DIM, GOAL_ENCODING_DIM, POLICY_HIDDEN_SIZE,
   LEFT_STREAM_DIM, RIGHT_STREAM_DIM,
)

# obs slice indices — must match shape_env._get_obs() layout
_LEFT_SLICE  = slice(0,  44)    # cursor state + focal shapes + all shapes
_RIGHT_SLICE = slice(14, 108)   # all shapes + goal encoding


# ---------------------------------------------------------------------------
# goal encoder MLP
# ---------------------------------------------------------------------------

# fixed seed for GoalEncoder weight init. this ensures the same prompt always
# produces the same 64-dim encoding across demo collection, BC training, and PPO
_GOAL_ENCODER_SEED = 42


class GoalEncoder(nn.Module):
   """
   projects raw EMBEDDING_DIM sentence embeddings down to GOAL_ENCODING_DIM.
   sits between get_embedding() and the policy input.

   weights are fixed at construction using _GOAL_ENCODER_SEED — the encoder
   is never trained. this guarantees the same prompt always maps to the same
   64-dim vector across all runs, processes, and call sites.
   """

   def __init__(self):
      super().__init__()
      self.net = nn.Sequential(
         nn.Linear(EMBEDDING_DIM, 128),
         nn.ReLU(),
         nn.Linear(128, GOAL_ENCODING_DIM),
         nn.Tanh(),
      )
      # apply fixed seed init immediately after construction.
      # fork_rng() ensures this doesn't disturb the global RNG state.
      with torch.random.fork_rng():
         torch.manual_seed(_GOAL_ENCODER_SEED)
         for layer in self.net:
            if isinstance(layer, nn.Linear):
               nn.init.xavier_uniform_(layer.weight)
               nn.init.zeros_(layer.bias)

   def forward(self, x: torch.Tensor) -> torch.Tensor:
      return self.net(x)


# ---------------------------------------------------------------------------
# bicameral policy network
# ---------------------------------------------------------------------------

class BicameralNetwork(nn.Module):
   """
   two-stream policy network with cross-attention.

   left stream:   obs[0:44]   -> hidden_size features  (cursor-local)
   right stream:  obs[14:108] -> hidden_size features  (scene-global)
   cross-attn:    right queries left (global reads local cursor state)
   action head:   (left + right) -> action_dim

   used directly for BC training and wrapped inside SB3 for PPO.
   """

   def __init__(self, hidden: int = POLICY_HIDDEN_SIZE):
      super().__init__()
      self.hidden = hidden

      # left stream: cursor-local encoder
      self.left_encoder = nn.Sequential(
         nn.Linear(LEFT_STREAM_DIM, hidden),
         nn.Tanh(),
         nn.Linear(hidden, hidden),
         nn.Tanh(),
      )

      # right stream: scene-global encoder
      self.right_encoder = nn.Sequential(
         nn.Linear(RIGHT_STREAM_DIM, hidden),
         nn.Tanh(),
         nn.Linear(hidden, hidden),
         nn.Tanh(),
      )

      # cross-attention: right queries left
      # Q from right, K and V from left — global reads local cursor context
      self.attn_q = nn.Linear(hidden, hidden, bias=False)
      self.attn_k = nn.Linear(hidden, hidden, bias=False)
      self.attn_v = nn.Linear(hidden, hidden, bias=False)
      self.attn_scale = hidden ** -0.5

      # movement head: combined features -> [dx, dy], squashed to [-1, 1]
      self.move_head = nn.Sequential(
         nn.Linear(hidden * 2, hidden),
         nn.Linear(hidden, 2),
         nn.Tanh(),
      )

      # grip head: combined features -> grip logit (unbounded, no Tanh).
      # BCE loss during BC training; threshold at 0.0 at runtime.
      self.grip_head = nn.Sequential(
         nn.Linear(hidden * 2, hidden // 2),
         nn.Linear(hidden // 2, 1),
      )

   def forward(self, obs: torch.Tensor) -> torch.Tensor:
      """
      obs: (batch, 108)
      returns: (batch, 3) — [dx, dy] in [-1, 1], grip as raw logit.
      during BC: grip column fed to BCE loss directly.
      at runtime (PPO / inference): threshold grip logit at 0.0.
      """
      left_in  = obs[:, _LEFT_SLICE]    # (batch, 44)
      right_in = obs[:, _RIGHT_SLICE]   # (batch, 94)

      left_feat  = self.left_encoder(left_in)    # (batch, hidden)
      right_feat = self.right_encoder(right_in)  # (batch, hidden)

      # cross-attention: right queries left
      q = self.attn_q(right_feat).unsqueeze(1)   # (batch, 1, hidden)
      k = self.attn_k(left_feat).unsqueeze(1)    # (batch, 1, hidden)
      v = self.attn_v(left_feat).unsqueeze(1)    # (batch, 1, hidden)

      scores   = torch.bmm(q, k.transpose(1, 2)) * self.attn_scale
      weights  = F.softmax(scores, dim=-1)
      attended = torch.bmm(weights, v).squeeze(1)

      right_out = right_feat + attended

      combined  = torch.cat([left_feat, right_out], dim=-1)
      move      = self.move_head(combined)             # (batch, 2)  in [-1, 1]
      grip      = self.grip_head(combined)             # (batch, 1)  raw logit
      return torch.cat([move, grip], dim=-1)           # (batch, 3)

   def features_dim(self) -> int:
      return self.hidden * 2

   def predict(self, obs: torch.Tensor) -> torch.Tensor:
      """
      inference-mode forward: returns actions with grip thresholded to ±1.0.
      use this anywhere outside of BC training (demo, debug, manual eval).
      during BC training, use forward() directly so BCE gets the raw logit.
      """
      out  = self.forward(obs)                          # (batch, 3)
      grip = torch.where(out[:, 2:3] > 0.0,
                         torch.ones_like(out[:, 2:3]),
                         -torch.ones_like(out[:, 2:3]))
      return torch.cat([out[:, 0:2], grip], dim=-1)


# ---------------------------------------------------------------------------
# SB3 custom policy wrapper
# ---------------------------------------------------------------------------

class BicameralPolicy(ActorCriticPolicy):
   """
   SB3 ActorCriticPolicy that wraps BicameralNetwork.
   """

   def __init__(self, observation_space, action_space, lr_schedule,
                **kwargs):
      kwargs.pop("net_arch", None)
      super().__init__(
         observation_space, action_space, lr_schedule,
         net_arch=[],
         **kwargs,
      )

   def _build_mlp_extractor(self):
      self.mlp_extractor = _BicameralExtractor(
         hidden=POLICY_HIDDEN_SIZE,
      )


class _BicameralExtractor(nn.Module):
   """
   adapter that makes BicameralNetwork conform to SB3's mlp_extractor interface.
   SB3 expects mlp_extractor to produce (policy_features, value_features).
   """

   def __init__(self, hidden: int = POLICY_HIDDEN_SIZE):
      super().__init__()
      self.net             = BicameralNetwork(hidden=hidden)
      self.latent_dim_pi   = hidden * 2
      self.latent_dim_vf   = hidden * 2

   def forward(self, obs: torch.Tensor):
      left_in  = obs[:, _LEFT_SLICE]
      right_in = obs[:, _RIGHT_SLICE]

      left_feat  = self.net.left_encoder(left_in)
      right_feat = self.net.right_encoder(right_in)

      q = self.net.attn_q(right_feat).unsqueeze(1)
      k = self.net.attn_k(left_feat).unsqueeze(1)
      v = self.net.attn_v(left_feat).unsqueeze(1)
      scores   = torch.bmm(q, k.transpose(1, 2)) * self.net.attn_scale
      weights  = F.softmax(scores, dim=-1)
      attended = torch.bmm(weights, v).squeeze(1)
      right_out = right_feat + attended

      features = torch.cat([left_feat, right_out], dim=-1)
      return features, features

   def forward_actor(self, obs: torch.Tensor) -> torch.Tensor:
      features, _ = self.forward(obs)
      return features

   def forward_critic(self, obs: torch.Tensor) -> torch.Tensor:
      _, features = self.forward(obs)
      return features


# ---------------------------------------------------------------------------
# BC training
# ---------------------------------------------------------------------------

def train_bc(
   dataset:    dict,
   save_path:  str,
   epochs:     int   = 30,
   batch_size: int   = 256,
   lr:         float = 3e-4,
   device:     str   = "cpu",
   verbose:    bool  = True,
) -> tuple:
   """
   train a BicameralNetwork and GoalEncoder on (obs, action) pairs.

   loss:
      move loss:  MSE on action[:, 0:2]  (dx, dy) — continuous movement
      grip loss:  BCE on action[:, 2]    (grip)   — binary on/off signal
      total     = move_loss + 0.5 * grip_loss

   grip is treated as binary (oracle outputs ±1.0) so BCE is correct here.
   MSE on grip would let the network hedge toward 0.0 and never commit —
   BCE penalises confident wrong predictions exponentially, forcing the
   network to actually learn grip timing.

   returns:
      BicameralNetwork — on cpu, eval mode.
      the GoalEncoder is owned by the caller; train_bc does not touch it.
   """
   obs_t = torch.tensor(dataset["observations"], dtype=torch.float32).to(device)
   act_t = torch.tensor(dataset["actions"],      dtype=torch.float32).to(device)

   # compute grip class weight from dataset balance.
   # oracle spends most transitions navigating (grip off), so grip-on labels
   # are a minority. without reweighting, BCE minimises loss by predicting
   # "always off" — which produces catastrophically high loss on grip-on steps.
   # pos_weight = n_off / n_on tells BCE to treat each grip-on sample as if
   # it were pos_weight samples, balancing the gradient contribution.
   grip_labels = (act_t[:, 2] > 0.0)
   n_on        = grip_labels.sum().item()
   n_off       = (~grip_labels).sum().item()
   pos_weight  = torch.tensor(
      [n_off / max(n_on, 1)], dtype=torch.float32).to(device)
   if verbose:
      print(f"\n  grip balance: {n_on:,} on / {n_off:,} off  "
            f"(pos_weight={pos_weight.item():.2f})")

   network = BicameralNetwork().to(device)

   optimizer = torch.optim.Adam(network.parameters(), lr=lr)
   scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
      optimizer, T_max=epochs, eta_min=1e-5)

   loader = DataLoader(
      TensorDataset(obs_t, act_t),
      batch_size=batch_size,
      shuffle=True,
   )

   if verbose:
      print(f"\n--- behavior cloning (bicameral network) ---")
      print(f"  obs dim     : {obs_t.shape[1]}  (left={LEFT_STREAM_DIM} right={RIGHT_STREAM_DIM})")
      print(f"  action dim  : {act_t.shape[1]}  [dx, dy, grip]")
      print(f"  samples     : {len(obs_t):,}")
      print(f"  epochs      : {epochs}")
      print(f"  batch size  : {batch_size}")
      print(f"  lr          : {lr} (cosine decay to 1e-5)")
      print(f"  loss        : MSE(dx,dy) + 0.5 * BCE(grip, weighted)")
      print(f"  device      : {device}\n")

   for epoch in range(1, epochs + 1):
      epoch_loss      = 0.0
      epoch_loss_grip = 0.0
      epoch_loss_dxy  = 0.0
      n_batches       = 0

      for obs_batch, act_batch in loader:
         pred = network(obs_batch)   # (batch, 3): [dx, dy, grip_logit]

         # movement loss — MSE on continuous dx, dy
         loss_dxy  = F.mse_loss(pred[:, 0:2], act_batch[:, 0:2])

         # grip loss — weighted BCE on binary grip signal.
         # oracle grip is ±1.0; convert to 0/1 labels for BCE.
         # pos_weight corrects for class imbalance (grip-off majority).
         grip_logit = pred[:, 2]
         grip_tgt   = (act_batch[:, 2] > 0.0).float()
         loss_grip  = F.binary_cross_entropy_with_logits(
            grip_logit, grip_tgt, pos_weight=pos_weight)

         loss = loss_dxy + 0.5 * loss_grip

         optimizer.zero_grad()
         loss.backward()
         torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
         optimizer.step()

         epoch_loss      += loss.item()
         epoch_loss_grip += loss_grip.item()
         epoch_loss_dxy  += loss_dxy.item()
         n_batches       += 1

      scheduler.step()

      avg_loss      = epoch_loss      / max(n_batches, 1)
      avg_grip_loss = epoch_loss_grip / max(n_batches, 1)
      avg_dxy_loss  = epoch_loss_dxy  / max(n_batches, 1)
      cur_lr        = scheduler.get_last_lr()[0]

      if verbose and (epoch % max(epochs // 5, 1) == 0 or epoch == 1):
         print(f"  epoch {epoch:3d}/{epochs} | "
               f"loss: {avg_loss:.4f}  "
               f"(grip: {avg_grip_loss:.4f}  dx/dy: {avg_dxy_loss:.4f})  "
               f"lr: {cur_lr:.2e}")

   if save_path is not None:
      os.makedirs(save_path, exist_ok=True)
      weights_path = os.path.join(save_path, "bc_weights.pt")
      torch.save(network.state_dict(), weights_path)
      if verbose:
         print(f"\n  bicameral weights saved to  {weights_path}")
         print(f"  goal encoder: fixed seed {_GOAL_ENCODER_SEED}, no checkpoint needed")

   return network.cpu()


# ---------------------------------------------------------------------------
# PPO from BC weights
# ---------------------------------------------------------------------------

def build_ppo_from_bc(bc_network: BicameralNetwork,
                      n_shapes: int,
                      vec_env=None,
                      goal: dict = None,
                      ent_coef: float = 0.02,
                      lr_ppo:   float = 1e-4) -> PPO:
   """
   create a PPO model using BicameralPolicy and copy BC network weights in.

   transplant 1: copy BicameralNetwork weights into _BicameralExtractor.net
                 via direct state_dict copy (architectures match).

   transplant 2: if you see this, update this description
   """
   if goal is None:
      # use a neutral default that is valid for all tasks — the real goal
      # encoding is always set via set_goal_encoding() in the env factory,
      # so this dict is only used when vec_env is None (rare / debug path).
      goal = {
         "task":         "arrange_in_sequence",
         "axis":         "x",
         "direction":    "ascending",
         "attribute":    "size",
         "region":       "none",
         "bounded":      False,
         "target_color": "none",
         "target_type":  "none",
      }

   env = vec_env if vec_env is not None else Monitor(
      ShapeEnv(n_shapes=n_shapes, goal=goal))

   model = PPO(
      BicameralPolicy,
      env,
      learning_rate=lr_ppo,
      n_steps=2048,
      batch_size=128,
      n_epochs=10,
      gamma=0.99,
      gae_lambda=0.95,
      clip_range=0.2,
      ent_coef=ent_coef,
      verbose=0,
      tensorboard_log="./logs/tensorboard/",
   )

   # --- transplant 1: bicameral extractor weights ---
   try:
      target_net = model.policy.mlp_extractor.net
      with torch.no_grad():
         target_net.load_state_dict(bc_network.state_dict())
      print("  [transplant 1] bicameral BC weights copied into PPO extractor.")
   except Exception as e:
      print(f"  [transplant 1] failed ({e}) — extractor starts from random init.")

   # --- transplant 2: compose split action heads into SB3 action_net ---
   # BC layout:   move_head: Linear(hidden*2, hidden) -> Tanh -> Linear(hidden, 2)
   #              grip_head: Linear(hidden*2, hidden//2) -> Tanh -> Linear(hidden//2, 1)
   # SB3 layout:  action_net: Linear(hidden*2, 3)  (rows 0:2 = move, row 2 = grip)
   # compose each head's two linear layers independently, then stack into action_net.
   # the intermediate Tanh is ignored — linear composition is an approximation but
   # still gives a far better initialisation direction than random weights.
   try:
      with torch.no_grad():

         # move_head is now: [0]=Linear(512,256), [1]=Linear(256,2), [2]=Tanh
         # mW1   = bc_network.move_head[0].weight   # (256, 512)
         # mb1   = bc_network.move_head[0].bias     # (256,)
         # mW2   = bc_network.move_head[1].weight   # (2, 256)
         # mb2   = bc_network.move_head[1].bias     # (2,)
         # W_move = mW2 @ mW1                       # (2, hidden*2)
         # b_move = mW2 @ mb1 + mb2                 # (2,)

         # grip_head is now: [0]=Linear(512,128), [1]=Linear(128,1)
         # gW1   = bc_network.grip_head[0].weight   # (128, 512)
         # gb1   = bc_network.grip_head[0].bias     # (128,)
         # gW2   = bc_network.grip_head[1].weight   # (1, 128)
         # gb2   = bc_network.grip_head[1].bias     # (1,)
         # W_grip = gW2 @ gW1                       # (1, hidden*2)
         # b_grip = gW2 @ gb1 + gb2                 # (1,)

         # stack into (3, hidden*2) to match SB3's action_net
         # W_eff = torch.cat([W_move, W_grip], dim=0)   # (3, hidden*2)
         # b_eff = torch.cat([b_move, b_grip], dim=0)   # (3,)

         # sb3_action_net = model.policy.action_net
         # sb3_action_net.weight.copy_(W_eff)
         # sb3_action_net.bias.copy_(b_eff)
      # print("  [transplant 2] composed BC move+grip heads into PPO action_net "
            # f"({W_eff.shape[1]}->{W_eff.shape[0]}).")

         sb3_action_net = model.policy.action_net
         # small random init centered at zero so grip can go either way
         nn.init.orthogonal_(sb3_action_net.weight, gain=0.01)
         nn.init.constant_(sb3_action_net.bias, 0.0)
      print("  [action_net] initialized near zero for unbiased grip exploration.")
   except Exception as e:
      print(f"  [transplant 2] failed ({e}) — action_net starts from random init.")

   return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
   import argparse
   from oracle import collect_demonstrations

   parser = argparse.ArgumentParser(
      description="train shape agent via behavior cloning from the oracle"
   )
   parser.add_argument("--episodes",  type=int,   default=500)
   parser.add_argument("--epochs",    type=int,   default=30)
   parser.add_argument("--save",      type=str,   default="./models/bc_agent")
   parser.add_argument("--noise",     type=float, default=0.06)
   args = parser.parse_args()

   dataset = collect_demonstrations(
      n_episodes=args.episodes,
      noise_std=args.noise,
      verbose=True,
   )

   device  = "cuda" if torch.cuda.is_available() else "cpu"
   network = train_bc(
      dataset=dataset,
      save_path=args.save,
      epochs=args.epochs,
      device=device,
   )
   print("\n--- done ---")
