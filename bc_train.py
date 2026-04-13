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
      input:  obs[14:428] — all shapes + goal embedding
      purpose: "where is everything and what does the goal say?"

   overlap on obs[14:43] (all shapes) is intentional — both streams see
   the full shape layout but with different contextual emphasis.

   cross-attention:
      right stream queries the left stream output.
      allows global context to read fine-grained cursor state.
      single-head attention, lightweight (no positional encoding needed).

   action head:
      movement head: (left + right) -> [dx, dy]  — MSE loss, Tanh output
      click head:    (left + right) -> click logit — BCE loss, no Tanh.
      at runtime the click logit is thresholded at 0.0 to produce ±1.0.

--- BC training details ---
   loss = MSE on all three action outputs (dx, dy, click together).
   click is continuous during BC even though it becomes binary at runtime.
   separate loss reporting for click vs dx/dy for diagnostics.
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
                 click_head: hidden*2->hidden//2->1) into SB3's single
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
   POLICY_HIDDEN_SIZE, LEFT_STREAM_DIM, RIGHT_STREAM_DIM,
)

# obs slice indices — must match shape_env._get_obs() layout
_LEFT_SLICE  = slice(0,  44)    # cursor state + focal shapes + all shapes
_RIGHT_SLICE = slice(14, 428)   # all shapes + goal embedding

# ---------------------------------------------------------------------------
# bicameral policy network
# ---------------------------------------------------------------------------

class BicameralNetwork(nn.Module):
   """
   two-stream policy network with cross-attention.

   left stream:   obs[0:44]   -> hidden_size features  (cursor-local)
   right stream:  obs[14:428] -> hidden_size features  (scene-global)
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

      # click head: combined features -> click logit (unbounded, no Tanh).
      # BCE loss during BC training; threshold at 0.0 at runtime.
      self.click_head = nn.Sequential(
         nn.Linear(hidden * 2, hidden // 2),
         nn.Linear(hidden // 2, 1),
      )

   def forward(self, obs: torch.Tensor) -> torch.Tensor:
      """
      obs: (batch, 428)
      returns: (batch, 3) — [dx, dy] in [-1, 1], click as raw logit.
      during BC: click column fed to BCE loss directly.
      at runtime (PPO / inference): threshold click logit at 0.0.
      """
      left_in  = obs[:, _LEFT_SLICE]    # (batch, 44)
      right_in = obs[:, _RIGHT_SLICE]   # (batch, 414)

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
      click     = self.click_head(combined)             # (batch, 1)  raw logit
      return torch.cat([move, click], dim=-1)           # (batch, 3)

   def features_dim(self) -> int:
      return self.hidden * 2

   def predict(self, obs: torch.Tensor) -> torch.Tensor:
      """
      inference-mode forward: returns actions with click thresholded to ±1.0.
      use this anywhere outside of BC training (demo, debug, manual eval).
      during BC training, use forward() directly so BCE gets the raw logit.
      """
      out   = self.forward(obs)                          # (batch, 3)
      click = torch.where(out[:, 2:3] > 0.0,
                         torch.ones_like(out[:, 2:3]),
                         -torch.ones_like(out[:, 2:3]))
      return torch.cat([out[:, 0:2], click], dim=-1)


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
   pretrained_network:  BicameralNetwork = None,
) -> BicameralNetwork:
   """
   train a BicameralNetwork on (obs, action) pairs.

   loss:
      move loss:  MSE on action[:, 0:2]  (dx, dy) — continuous movement
      click loss: BCE on action[:, 2]    (click)  — binary on/off signal
      total     = move_loss + 0.5 * click_loss

   click is treated as binary (oracle outputs ±1.0) so BCE is correct here.
   MSE on click would let the network hedge toward 0.0 and never commit —
   BCE penalises confident wrong predictions exponentially, forcing the
   network to actually learn click timing.

   returns:
      BicameralNetwork — on cpu, eval mode.
   """
   obs_t = torch.tensor(dataset["observations"], dtype=torch.float32).to(device)
   act_t = torch.tensor(dataset["actions"],      dtype=torch.float32).to(device)

   # compute click class weight from dataset balance.
   # oracle spends most transitions navigating (click off), so click-on labels
   # are a minority. without reweighting, BCE minimises loss by predicting
   # "always off" — which produces catastrophically high loss on click-on steps.
   # pos_weight = n_off / n_on tells BCE to treat each click-on sample as if
   # it were pos_weight samples, balancing the gradient contribution.
   click_labels = (act_t[:, 2] > 0.0)
   n_on         = click_labels.sum().item()
   n_off        = (~click_labels).sum().item()
   pos_weight   = torch.tensor(
      [n_off / max(n_on, 1)], dtype=torch.float32).to(device)
   if verbose:
      print(f"\n  click balance: {n_on:,} on / {n_off:,} off  "
            f"(pos_weight={pos_weight.item():.2f})")

   network = BicameralNetwork().to(device)

   if pretrained_network is not None:
      network.load_state_dict(pretrained_network.state_dict())
      print("  [bc_train] loaded prompt-trained right stream weights into BC network")   

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
      print(f"  action dim  : {act_t.shape[1]}  [dx, dy, click]")
      print(f"  samples     : {len(obs_t):,}")
      print(f"  epochs      : {epochs}")
      print(f"  batch size  : {batch_size}")
      print(f"  lr          : {lr} (cosine decay to 1e-5)")
      print(f"  loss        : MSE(dx,dy) + 0.5 * BCE(click, weighted)")
      print(f"  device      : {device}\n")

   for epoch in range(1, epochs + 1):
      epoch_loss       = 0.0
      epoch_loss_click = 0.0
      epoch_loss_dxy   = 0.0
      n_batches        = 0

      for obs_batch, act_batch in loader:
         pred = network(obs_batch)   # (batch, 3): [dx, dy, click_logit]

         # movement loss — MSE on continuous dx, dy
         loss_dxy  = F.mse_loss(pred[:, 0:2], act_batch[:, 0:2])

         # click loss — weighted BCE on binary click signal.
         # oracle click is ±1.0; convert to 0/1 labels for BCE.
         # pos_weight corrects for class imbalance (click-off majority).
         click_logit = pred[:, 2]
         click_tgt   = (act_batch[:, 2] > 0.0).float()
         loss_click  = F.binary_cross_entropy_with_logits(
            click_logit, click_tgt, pos_weight=pos_weight)

         loss = loss_dxy + 0.5 * loss_click

         optimizer.zero_grad()
         loss.backward()
         torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
         optimizer.step()

         epoch_loss       += loss.item()
         epoch_loss_click += loss_click.item()
         epoch_loss_dxy   += loss_dxy.item()
         n_batches        += 1

      scheduler.step()

      avg_loss       = epoch_loss      / max(n_batches, 1)
      avg_click_loss = epoch_loss_click / max(n_batches, 1)
      avg_dxy_loss   = epoch_loss_dxy  / max(n_batches, 1)
      cur_lr         = scheduler.get_last_lr()[0]

      if verbose and (epoch % max(epochs // 5, 1) == 0 or epoch == 1):
         print(f"  epoch {epoch:3d}/{epochs} | "
               f"loss: {avg_loss:.4f}  "
               f"(click: {avg_click_loss:.4f}  dx/dy: {avg_dxy_loss:.4f})  "
               f"lr: {cur_lr:.2e}")

   if save_path is not None:
      os.makedirs(save_path, exist_ok=True)
      weights_path = os.path.join(save_path, "bc_weights.pt")
      torch.save(network.state_dict(), weights_path)
      if verbose:
         print(f"\n  bicameral weights saved to  {weights_path}")

   return network.cpu()


# ---------------------------------------------------------------------------
# PPO from BC weights
# ---------------------------------------------------------------------------

def build_ppo_from_bc(bc_network: BicameralNetwork,
                      n_shapes: int,
                      vec_env=None,
                      goal: dict = None,
                      ent_coef: float = 0.02,
                      lr_ppo:     float = 1e-4,
                      batch_size: int   = 128) -> PPO:
   """
   create a PPO model using BicameralPolicy and copy BC network weights in.

   transplant 1: copy BicameralNetwork weights into _BicameralExtractor.net
                 via direct state_dict copy (architectures match).

   transplant 2: action_net (hidden*2 -> 3) is initialized near zero with
                 orthogonal init (gain=0.01) rather than transplanting BC's
                 action heads. this gives PPO unbiased click exploration from
                 the start — click timing is shaped by reward, not BC init.
   """
   if goal is None:
      # this dict is only used when vec_env is None (rare / debug path).
      goal = {
         "task":         "none",
         "axis":         "none",
         "direction":    "none",
         "attribute":    "none",
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
      batch_size=batch_size,
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

   # --- transplant 2: initialise action_net near zero ---
   # the lossless composition approach (composing BC's split heads into SB3's
   # single action_net linear layer) was attempted but produced a strongly
   # negative click bias due to numerical amplification through the composition.
   # near-zero orthogonal init is strictly better: it gives PPO equal probability
   # of exploring click-on and click-off, letting reward shape click timing cleanly.
   # transplant 1 (navigation weights) is the valuable warm-start; click timing
   # is best left for PPO to discover.
   try:
      with torch.no_grad():
         sb3_action_net = model.policy.action_net
         nn.init.orthogonal_(sb3_action_net.weight, gain=0.01)
         nn.init.constant_(sb3_action_net.bias, 0.0)
      print("  [action_net] initialized near zero for unbiased click exploration.")
   except Exception as e:
      print(f"  [action_net] init failed ({e}) — action_net starts from random init.")

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
