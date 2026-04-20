"""
bc_train.py

behavior cloning trainer for the single-shape manipulation agent.

--- bicameral network architecture ---
   left stream  (cursor-local):
      input:  obs[0:19]  — cursor state + grabbed shape + nearest shape + all shapes
      LEFT_STREAM_DIM = 4 + 10 + 5 = 19

   right stream (scene-global):
      input:  obs[14:403] — all shapes + goal embedding
      RIGHT_STREAM_DIM = 5 + 384 = 389

   overlap on obs[14:18] (all shapes, 5-dim) is intentional.

   cross-attention: right queries left.
   action head: (left + right) -> [dx, dy, grip_logit]

--- BC training ---
   move loss:  MSE on action[:, 0:2]
   grip loss:  BCE on action[:, 2]  (weighted for class imbalance)
   total     = move_loss + 0.5 * grip_loss

--- weight transplant ---
   build_ppo_from_bc() wraps the bicameral network in SB3's PPO framework
   via BicameralPolicy, transplanting BC extractor weights and initializing
   action_net near zero for unbiased grip exploration.
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
# left:  [0:19]   cursor state + focal shapes + all shapes
# right: [14:403] all shapes + goal embedding
_LEFT_SLICE  = slice(0,  19)
_RIGHT_SLICE = slice(14, 403)


# ---------------------------------------------------------------------------
# bicameral policy network
# ---------------------------------------------------------------------------

class BicameralNetwork(nn.Module):
   """
   two-stream policy network with cross-attention.

   left stream:   obs[0:19]   -> hidden_size features  (cursor-local)
   right stream:  obs[14:403] -> hidden_size features  (scene-global)
   cross-attn:    right queries left
   action head:   (left + right) -> [dx, dy, grip_logit]
   """

   def __init__(self, hidden: int = POLICY_HIDDEN_SIZE):
      super().__init__()
      self.hidden = hidden

      self.left_encoder = nn.Sequential(
         nn.Linear(LEFT_STREAM_DIM, hidden),
         nn.Tanh(),
         nn.Linear(hidden, hidden),
         nn.Tanh(),
      )

      self.right_encoder = nn.Sequential(
         nn.Linear(RIGHT_STREAM_DIM, hidden),
         nn.Tanh(),
         nn.Linear(hidden, hidden),
         nn.Tanh(),
      )

      # cross-attention: right queries left
      self.attn_q     = nn.Linear(hidden, hidden, bias=False)
      self.attn_k     = nn.Linear(hidden, hidden, bias=False)
      self.attn_v     = nn.Linear(hidden, hidden, bias=False)
      self.attn_scale = hidden ** -0.5

      # movement head: combined -> [dx, dy], squashed to [-1, 1]
      self.move_head = nn.Sequential(
         nn.Linear(hidden * 2, hidden),
         nn.Linear(hidden, 2),
         nn.Tanh(),
      )

      # grip head: combined -> grip logit (no Tanh; BCE loss during BC)
      self.grip_head = nn.Sequential(
         nn.Linear(hidden * 2, hidden // 2),
         nn.Linear(hidden // 2, 1),
      )

   def forward(self, obs: torch.Tensor) -> torch.Tensor:
      """
      obs: (batch, 403)
      returns: (batch, 3) — [dx, dy] in [-1, 1], grip as raw logit.
      """
      left_in  = obs[:, _LEFT_SLICE]    # (batch, 19)
      right_in = obs[:, _RIGHT_SLICE]   # (batch, 389)

      left_feat  = self.left_encoder(left_in)
      right_feat = self.right_encoder(right_in)

      q = self.attn_q(right_feat).unsqueeze(1)
      k = self.attn_k(left_feat).unsqueeze(1)
      v = self.attn_v(left_feat).unsqueeze(1)

      scores   = torch.bmm(q, k.transpose(1, 2)) * self.attn_scale
      weights  = F.softmax(scores, dim=-1)
      attended = torch.bmm(weights, v).squeeze(1)

      right_out = right_feat + attended
      combined  = torch.cat([left_feat, right_out], dim=-1)

      move = self.move_head(combined)   # (batch, 2)
      grip = self.grip_head(combined)   # (batch, 1)
      return torch.cat([move, grip], dim=-1)   # (batch, 3)

   def features_dim(self) -> int:
      return self.hidden * 2

   def predict(self, obs: torch.Tensor) -> torch.Tensor:
      """inference-mode forward: grip thresholded to ±1.0."""
      out  = self.forward(obs)
      grip = torch.where(out[:, 2:3] > 0.0,
                         torch.ones_like(out[:, 2:3]),
                         -torch.ones_like(out[:, 2:3]))
      return torch.cat([out[:, 0:2], grip], dim=-1)


# ---------------------------------------------------------------------------
# SB3 custom policy wrapper
# ---------------------------------------------------------------------------

class BicameralPolicy(ActorCriticPolicy):

   def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
      kwargs.pop("net_arch", None)
      super().__init__(
         observation_space, action_space, lr_schedule,
         net_arch=[],
         **kwargs,
      )

   def _build_mlp_extractor(self):
      self.mlp_extractor = _BicameralExtractor(hidden=POLICY_HIDDEN_SIZE)


class _BicameralExtractor(nn.Module):

   def __init__(self, hidden: int = POLICY_HIDDEN_SIZE):
      super().__init__()
      self.net           = BicameralNetwork(hidden=hidden)
      self.latent_dim_pi = hidden * 2
      self.latent_dim_vf = hidden * 2

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
   dataset:            dict,
   save_path:          str,
   epochs:             int   = 30,
   batch_size:         int   = 256,
   lr:                 float = 3e-4,
   device:             str   = "cpu",
   verbose:            bool  = True,
   pretrained_network: BicameralNetwork = None,
) -> BicameralNetwork:
   """
   train a BicameralNetwork on (obs, action) pairs.

   loss = MSE(dx, dy) + 0.5 * BCE(grip, weighted)

   grip pos_weight corrects for class imbalance (grip-off majority).
   returns BicameralNetwork on cpu, eval mode.
   """
   obs_t = torch.tensor(dataset["observations"], dtype=torch.float32).to(device)
   act_t = torch.tensor(dataset["actions"],      dtype=torch.float32).to(device)

   grip_labels = (act_t[:, 2] > 0.0)
   n_on        = grip_labels.sum().item()
   n_off       = (~grip_labels).sum().item()
   pos_weight  = torch.tensor(
      [n_off / max(n_on, 1)], dtype=torch.float32).to(device)
   if verbose:
      print(f"\n  grip balance: {n_on:,} on / {n_off:,} off  "
            f"(pos_weight={pos_weight.item():.2f})")

   network = BicameralNetwork().to(device)

   if pretrained_network is not None:
      network.load_state_dict(pretrained_network.state_dict())
      print("  [bc_train] loaded pretrained weights into BC network")

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
      print(f"  obs dim     : {obs_t.shape[1]}")
      print(f"  action dim  : {act_t.shape[1]}  [dx, dy, grip]")
      print(f"  samples     : {len(obs_t):,}")
      print(f"  epochs      : {epochs}")
      print(f"  batch size  : {batch_size}")
      print(f"  lr          : {lr} (cosine decay to 1e-5)")
      print(f"  device      : {device}\n")

   for epoch in range(1, epochs + 1):
      epoch_loss = epoch_grip = epoch_dxy = 0.0
      n_batches  = 0

      for obs_batch, act_batch in loader:
         pred = network(obs_batch)

         loss_dxy  = F.mse_loss(pred[:, 0:2], act_batch[:, 0:2])
         grip_logit = pred[:, 2]
         grip_tgt   = (act_batch[:, 2] > 0.0).float()
         loss_grip  = F.binary_cross_entropy_with_logits(
            grip_logit, grip_tgt, pos_weight=pos_weight)
         loss = loss_dxy + 0.5 * loss_grip

         optimizer.zero_grad()
         loss.backward()
         torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
         optimizer.step()

         epoch_loss += loss.item()
         epoch_grip += loss_grip.item()
         epoch_dxy  += loss_dxy.item()
         n_batches  += 1

      scheduler.step()

      avg = epoch_loss / max(n_batches, 1)
      ag  = epoch_grip / max(n_batches, 1)
      ad  = epoch_dxy  / max(n_batches, 1)
      cur_lr = scheduler.get_last_lr()[0]

      if verbose and (epoch % max(epochs // 5, 1) == 0 or epoch == 1):
         print(f"  epoch {epoch:3d}/{epochs} | "
               f"loss: {avg:.4f}  (grip: {ag:.4f}  dx/dy: {ad:.4f})  "
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
                      n_shapes:   int   = 1,
                      vec_env           = None,
                      goal:       dict  = None,
                      ent_coef:   float = 0.10,
                      lr_ppo:     float = 1e-4,
                      clip_range: float = 0.3,
                      batch_size: int   = 128) -> PPO:
   """
   create a PPO model with BicameralPolicy and copy BC network weights in.

   transplant 1: BC extractor weights → PPO mlp_extractor.net
   transplant 2: action_net initialized near zero for unbiased grip exploration.

   ent_coef=0.10: higher than typical to prevent entropy collapse when the
   task changes mid-curriculum. the BC warm-start gives a confident prior
   which PPO rapidly narrows — 0.10 keeps exploration alive across stage
   transitions. decays naturally as the policy improves.

   clip_range=0.3: looser clip allows PPO to make larger corrections when
   recovering from entropy collapse or adapting to a new task. with a strong
   BC prior, 0.2 is too tight to push the policy meaningfully when needed.
   """
   if goal is None:
      goal = {
         "task": "reach", "axis": "none", "direction": "none",
         "attribute": "none", "region": "none", "bounded": False,
         "target_color": "none", "target_type": "none",
      }

   env = vec_env if vec_env is not None else Monitor(
      ShapeEnv(n_shapes=1, goal=goal))

   model = PPO(
      BicameralPolicy,
      env,
      learning_rate=lr_ppo,
      n_steps=2048,
      batch_size=batch_size,
      n_epochs=10,
      gamma=0.99,
      gae_lambda=0.95,
      clip_range=clip_range,
      ent_coef=ent_coef,
      verbose=0,
      tensorboard_log="./logs/tensorboard/",
   )

   # transplant 1: extractor weights
   try:
      target_net = model.policy.mlp_extractor.net
      with torch.no_grad():
         target_net.load_state_dict(bc_network.state_dict())
      print("  [transplant 1] BC weights copied into PPO extractor.")
   except Exception as e:
      print(f"  [transplant 1] failed ({e}) — extractor starts from random init.")

   # transplant 2: action_net near zero
   try:
      with torch.no_grad():
         sb3_action_net = model.policy.action_net
         nn.init.orthogonal_(sb3_action_net.weight, gain=0.01)
         nn.init.constant_(sb3_action_net.bias, 0.0)
      print("  [action_net] initialized near zero for unbiased grip exploration.")
   except Exception as e:
      print(f"  [action_net] init failed ({e}).")

   return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
   import argparse
   from oracle import collect_demonstrations

   parser = argparse.ArgumentParser(
      description="train shape agent via behavior cloning")
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