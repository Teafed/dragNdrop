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
      combined left+right output -> [dx, dy, grip]
      dx, dy:  MSE loss during BC
      grip:    treated as continuous during BC (also MSE), threshold at runtime

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
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from shape_env import ShapeEnv
from config import (
   EMBEDDING_DIM, GOAL_ENCODING_DIM, POLICY_HIDDEN_SIZE,
   LEFT_STREAM_DIM, RIGHT_STREAM_DIM,
   get_obs_size,
)

# obs slice indices — must match shape_env._get_obs() layout
_LEFT_SLICE  = slice(0,  44)    # cursor state + focal shapes + all shapes
_RIGHT_SLICE = slice(14, 108)   # all shapes + goal encoding


# ---------------------------------------------------------------------------
# goal encoder MLP
# ---------------------------------------------------------------------------

class GoalEncoder(nn.Module):
   """
   projects raw EMBEDDING_DIM sentence embeddings down to GOAL_ENCODING_DIM.
   sits between get_embedding() and the policy input.
   """

   def __init__(self):
      super().__init__()
      self.net = nn.Sequential(
         nn.Linear(EMBEDDING_DIM, 128),
         nn.ReLU(),
         nn.Linear(128, GOAL_ENCODING_DIM),
         nn.Tanh(),
      )

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

   def __init__(self, action_dim: int = 3,
                hidden: int = POLICY_HIDDEN_SIZE):
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

      # action head: combined (left + right) -> action
      self.action_head = nn.Sequential(
         nn.Linear(hidden * 2, hidden),
         nn.Tanh(),
         nn.Linear(hidden, action_dim),
         nn.Tanh(),
      )

   def forward(self, obs: torch.Tensor) -> torch.Tensor:
      """
      obs: (batch, 108)
      returns: (batch, action_dim) in [-1, 1]
      """
      left_in  = obs[:, _LEFT_SLICE]    # (batch, 44)
      right_in = obs[:, _RIGHT_SLICE]   # (batch, 94)

      left_feat  = self.left_encoder(left_in)    # (batch, hidden)
      right_feat = self.right_encoder(right_in)  # (batch, hidden)

      # cross-attention: right queries left
      # add sequence dim of 1 for F.scaled_dot_product_attention
      q = self.attn_q(right_feat).unsqueeze(1)   # (batch, 1, hidden)
      k = self.attn_k(left_feat).unsqueeze(1)    # (batch, 1, hidden)
      v = self.attn_v(left_feat).unsqueeze(1)    # (batch, 1, hidden)

      # scaled dot-product attention
      scores   = torch.bmm(q, k.transpose(1, 2)) * self.attn_scale  # (batch, 1, 1)
      weights  = F.softmax(scores, dim=-1)
      attended = torch.bmm(weights, v).squeeze(1)   # (batch, hidden)

      # residual: attended right + original right
      right_out = right_feat + attended   # (batch, hidden)

      # concat and decode to action
      combined = torch.cat([left_feat, right_out], dim=-1)   # (batch, hidden*2)
      return self.action_head(combined)

   def features_dim(self) -> int:
      """combined feature dimension before action head — used by SB3 wrapper."""
      return self.hidden * 2


# ---------------------------------------------------------------------------
# SB3 custom policy wrapper
# ---------------------------------------------------------------------------

class BicameralPolicy(ActorCriticPolicy):
   """
   SB3 ActorCriticPolicy that wraps BicameralNetwork.

   SB3's PPO requires an ActorCriticPolicy. this subclass replaces the
   default MLP extractor with BicameralNetwork while keeping SB3's
   value head and action distribution unchanged.
   """

   def __init__(self, observation_space, action_space, lr_schedule,
                **kwargs):
      # disable SB3's default net_arch so we can use our own network
      kwargs.pop("net_arch", None)
      super().__init__(
         observation_space, action_space, lr_schedule,
         net_arch=[],   # empty — we override mlp_extractor below
         **kwargs,
      )

   def _build_mlp_extractor(self):
      """replace SB3's mlp_extractor with BicameralNetwork."""
      self.mlp_extractor = _BicameralExtractor(
         observation_space=self.observation_space,
         hidden=POLICY_HIDDEN_SIZE,
      )


class _BicameralExtractor(nn.Module):
   """
   adapter that makes BicameralNetwork conform to SB3's mlp_extractor interface.
   SB3 expects mlp_extractor to produce (policy_features, value_features).
   """

   def __init__(self, observation_space, hidden: int = POLICY_HIDDEN_SIZE):
      super().__init__()
      self.net             = BicameralNetwork(action_dim=3, hidden=hidden)
      self.latent_dim_pi   = hidden * 2   # policy feature dim
      self.latent_dim_vf   = hidden * 2   # value feature dim (shared)

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
      return features, features   # (policy_features, value_features)

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
   lr:         float = 1e-3,
   device:     str   = "cpu",
) -> tuple:
   """
   train a BicameralNetwork and GoalEncoder on (obs, action) pairs via MSE.

   loss breakdown:
      grip loss:  MSE on action[:, 2]  — whether to grip
      dxy loss:   MSE on action[:, 0:2] — cursor movement

   returns:
      (BicameralNetwork, GoalEncoder) — both on cpu, eval mode.
   """
   obs_t = torch.tensor(dataset["observations"], dtype=torch.float32).to(device)
   act_t = torch.tensor(dataset["actions"],      dtype=torch.float32).to(device)

   network      = BicameralNetwork().to(device)
   goal_encoder = GoalEncoder().to(device)

   optimizer = torch.optim.Adam(network.parameters(), lr=lr)
   scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
      optimizer, T_max=epochs, eta_min=1e-5)
   loss_fn = nn.MSELoss(reduction="none")

   loader = DataLoader(
      TensorDataset(obs_t, act_t),
      batch_size=batch_size,
      shuffle=True,
   )

   print(f"\n--- behavior cloning (bicameral network) ---")
   print(f"  obs dim     : {obs_t.shape[1]}  (left={LEFT_STREAM_DIM} right={RIGHT_STREAM_DIM})")
   print(f"  action dim  : {act_t.shape[1]}  [dx, dy, grip]")
   print(f"  samples     : {len(obs_t):,}")
   print(f"  epochs      : {epochs}")
   print(f"  batch size  : {batch_size}")
   print(f"  lr          : {lr} (cosine decay to 1e-5)")
   print(f"  device      : {device}\n")

   for epoch in range(1, epochs + 1):
      epoch_loss      = 0.0
      epoch_loss_grip = 0.0
      epoch_loss_dxy  = 0.0
      n_batches       = 0

      for obs_batch, act_batch in loader:
         pred       = network(obs_batch)
         per_output = loss_fn(pred, act_batch)   # (batch, 3)
         loss       = per_output.mean()

         optimizer.zero_grad()
         loss.backward()
         torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
         optimizer.step()

         epoch_loss      += loss.item()
         epoch_loss_grip += per_output[:, 2].mean().item()   # grip
         epoch_loss_dxy  += per_output[:, 0:2].mean().item() # dx, dy
         n_batches       += 1

      scheduler.step()

      avg_loss      = epoch_loss      / max(n_batches, 1)
      avg_grip_loss = epoch_loss_grip / max(n_batches, 1)
      avg_dxy_loss  = epoch_loss_dxy  / max(n_batches, 1)
      cur_lr        = scheduler.get_last_lr()[0]

      if epoch % max(epochs // 5, 1) == 0 or epoch == 1:
         print(f"  epoch {epoch:3d}/{epochs} | "
               f"loss: {avg_loss:.4f}  "
               f"(grip: {avg_grip_loss:.4f}  dx/dy: {avg_dxy_loss:.4f})  "
               f"lr: {cur_lr:.2e}")

   os.makedirs(save_path, exist_ok=True)
   weights_path = os.path.join(save_path, "bc_weights.pt")
   encoder_path = os.path.join(save_path, "goal_encoder.pt")
   torch.save(network.state_dict(),      weights_path)
   torch.save(goal_encoder.state_dict(), encoder_path)
   print(f"\n  bicameral weights saved to  {weights_path}")
   print(f"  goal encoder saved to       {encoder_path}")

   return network.cpu(), goal_encoder.cpu()


# ---------------------------------------------------------------------------
# PPO from BC weights
# ---------------------------------------------------------------------------

def build_ppo_from_bc(bc_network: BicameralNetwork,
                      n_shapes: int,
                      vec_env=None,
                      goal: dict = None) -> PPO:
   """
   create a PPO model using BicameralPolicy and copy BC network weights in.

   because BicameralPolicy uses a custom mlp_extractor, the weight copy
   targets _BicameralExtractor.net (which is a BicameralNetwork instance)
   rather than SB3's default policy_net layers.
   """
   if goal is None:
      goal = {
         "task": "arrange_in_sequence", "axis": "x",
         "direction": "ascending", "attribute": "size",
         "region": "none", "bounded": False,
      }

   env = vec_env if vec_env is not None else Monitor(
      ShapeEnv(n_shapes=n_shapes, goal=goal))

   model = PPO(
      BicameralPolicy,
      env,
      learning_rate=3e-5,
      n_steps=2048,
      batch_size=128,
      n_epochs=10,
      gamma=0.99,
      gae_lambda=0.95,
      clip_range=0.2,
      ent_coef=0.005,
      verbose=0,
      tensorboard_log="./logs/tensorboard/",
   )

   try:
      target_net = model.policy.mlp_extractor.net
      with torch.no_grad():
         target_net.load_state_dict(bc_network.state_dict())
      print("  bicameral BC weights transplanted into PPO policy.")
   except Exception as e:
      print(f"  weight transplant failed ({e}) — PPO starts from random init.")

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
   parser.add_argument("--force",     action="store_true",
                       help="force re-collection of oracle demos")
   args = parser.parse_args()

   dataset = collect_demonstrations(
      n_episodes=args.episodes,
      noise_std=args.noise,
      verbose=True,
      force=args.force,
   )

   device  = "cuda" if torch.cuda.is_available() else "cpu"
   network, goal_encoder = train_bc(
      dataset=dataset,
      save_path=args.save,
      epochs=args.epochs,
      device=device,
   )
   print("\n--- done ---")
