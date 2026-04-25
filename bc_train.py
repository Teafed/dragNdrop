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
   POLICY_HIDDEN_SIZE, LEFT_STREAM_DIM, RIGHT_STREAM_DIM, EMBEDDING_DIM
)

# obs slice indices — must match shape_env._get_obs() layout
# left:  [0:19]   cursor state + focal shapes + all shapes
# right: [14:403] all shapes + goal embedding
_LEFT_SLICE  = slice(0,  19)
_RIGHT_SLICE = slice(14, 403)
_HOLDING_IDX = 2
_EMBED_SLICE = slice(19, 403)

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

      # post-grip head: embedding -> direction bias (dx_bias, dy_bias)
      # gated on holding bit so it contributes zero during reach/touch.
      self.post_grip_head = nn.Sequential(
         nn.Linear(EMBEDDING_DIM, 128),
         nn.Tanh(),
         nn.Linear(128, 2),
      )
      nn.init.orthogonal_(self.post_grip_head[-1].weight, gain=0.01)
      nn.init.constant_(self.post_grip_head[-1].bias, 0.0)

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

      # post-grip head: direction bias from embedding, gated by holding bit
      embed   = obs[:, _EMBED_SLICE]
      holding = obs[:, _HOLDING_IDX:_HOLDING_IDX+1] # (batch, 1)
      bias    = self.post_grip_head(embed)          # (batch, 2)
      gated   = bias * holding                      # zero when not holding
      move    = torch.tanh(move + gated)            # keep in [-1, 1]
      
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
   """
   custom policy that applies the post-grip direction bias after
   SB3's action_net produces a mean action.

   the bias is read from self.mlp_extractor.net.post_grip_head and
   gated by the holding bit. this keeps BC's training path
   (BicameralNetwork.forward) consistent with PPO's action path.
   """

   def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
      kwargs.pop("net_arch", None)
      super().__init__(
         observation_space, action_space, lr_schedule,
         net_arch=[],
         **kwargs,
      )

   def _build_mlp_extractor(self):
      self.mlp_extractor = _BicameralExtractor(hidden=POLICY_HIDDEN_SIZE)

   def _apply_post_grip_bias(self, mean_actions: torch.Tensor,
                              obs: torch.Tensor) -> torch.Tensor:
      """
      apply post-grip direction bias to the action_net's mean output.
      matches the behavior of BicameralNetwork.forward for the movement
      channels while leaving the grip channel untouched.
      """
      embed   = obs[:, _EMBED_SLICE]
      holding = obs[:, _HOLDING_IDX:_HOLDING_IDX+1]
      bias    = self.mlp_extractor.net.post_grip_head(embed)
      gated   = bias * holding
      # mean_actions shape: (batch, 3) — [dx, dy, grip_logit]
      # only add bias to dx, dy; leave grip alone
      biased_xy  = torch.tanh(mean_actions[:, 0:2] + gated)
      return torch.cat([biased_xy, mean_actions[:, 2:3]], dim=-1)

   def forward(self, obs, deterministic: bool = False):
      # standard SB3 forward to get features and mean actions
      features          = self.extract_features(obs)
      latent_pi, latent_vf = self.mlp_extractor(features)
      mean_actions      = self.action_net(latent_pi)
      values            = self.value_net(latent_vf)

      # apply post-grip bias to mean actions
      mean_actions = self._apply_post_grip_bias(mean_actions, obs)

      # construct action distribution from the biased mean
      distribution = self.action_dist.proba_distribution(
         mean_actions, self.log_std)
      actions      = distribution.get_actions(deterministic=deterministic)
      log_prob     = distribution.log_prob(actions)
      return actions, values, log_prob

   def evaluate_actions(self, obs, actions):
      """
      override evaluate_actions so PPO's loss computation uses
      post-grip-biased means, consistent with action sampling.
      """
      features          = self.extract_features(obs)
      latent_pi, latent_vf = self.mlp_extractor(features)
      mean_actions      = self.action_net(latent_pi)
      mean_actions      = self._apply_post_grip_bias(mean_actions, obs)
      distribution      = self.action_dist.proba_distribution(
         mean_actions, self.log_std)
      log_prob          = distribution.log_prob(actions)
      entropy           = distribution.entropy()
      values            = self.value_net(latent_vf)
      return values, log_prob, entropy

   def _predict(self, obs, deterministic: bool = False):
      """
      inference-time action (called by model.predict()). must also
      apply post-grip bias.
      """
      features          = self.extract_features(obs)
      latent_pi, _      = self.mlp_extractor(features)
      mean_actions      = self.action_net(latent_pi)
      mean_actions      = self._apply_post_grip_bias(mean_actions, obs)
      distribution      = self.action_dist.proba_distribution(
         mean_actions, self.log_std)
      return distribution.get_actions(deterministic=deterministic)

class _BicameralExtractor(nn.Module):
   """
   bicameral extractor augmented with a region head.

   the region head reads the goal embedding independently and produces
   a 512-dim direction-bias vector that is ADDED to the bicameral
   features. near-zero init means the augmented network at step 0 is
   behaviorally identical to the unaugmented one — only training can
   shift behavior, and only for drag directions the region head can
   learn to encode.
   """

   def __init__(self, hidden: int = POLICY_HIDDEN_SIZE):
      super().__init__()
      self.net           = BicameralNetwork(hidden=hidden)
      self.latent_dim_pi = hidden * 2
      self.latent_dim_vf = hidden * 2

      # region head: embedding (384) -> hidden (256) -> features (512)
      # output projection initialized near-zero via orthogonal_(gain=0.01)
      # so the region head contributes ~0 at step 0.
      self.region_head = nn.Sequential(
         nn.Linear(EMBEDDING_DIM, hidden),
         nn.Tanh(),
         nn.Linear(hidden, hidden * 2),
      )
      nn.init.orthogonal_(self.region_head[-1].weight, gain=0.01)
      nn.init.constant_(self.region_head[-1].bias, 0.0)

   def _bicameral_features(self, obs):
      """run the original bicameral pipeline — unchanged from before."""
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

      return torch.cat([left_feat, right_out], dim=-1)

   def forward(self, obs: torch.Tensor):
      features     = self._bicameral_features(obs)
      embed        = obs[:, _EMBED_SLICE]
      region_bias  = self.region_head(embed)
      augmented    = features + region_bias
      return augmented, augmented

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
   import numpy as np
   obs_t = torch.tensor(dataset["observations"], dtype=torch.float32).to(device)
   act_t = torch.tensor(dataset["actions"],      dtype=torch.float32).to(device)
   # build per-transition direction labels for supervised post-grip head loss.
   # only drag+holding transitions get a valid label; everything else gets
   # zero direction and zero loss weight.
   REGION_DIR = {
      "left":   (-1.0,  0.0),
      "right":  (+1.0,  0.0),
      "top":    ( 0.0, -1.0),
      "bottom": ( 0.0, +1.0),
   }
   regions     = dataset.get("regions")
   tasks       = dataset.get("tasks")
   holding_bit = obs_t[:, 2].cpu().numpy()

   dir_labels = np.zeros((len(obs_t), 2), dtype=np.float32)
   dir_weight = np.zeros( len(obs_t),      dtype=np.float32)

   if regions is not None:
      for i in range(len(obs_t)):
         # only apply supervision during drag+holding
         if tasks[i] == "drag" and holding_bit[i] > 0.5:
            r = regions[i]
            if r in REGION_DIR:
               dir_labels[i] = REGION_DIR[r]
               dir_weight[i] = 1.0

   dir_labels_t = torch.tensor(dir_labels, dtype=torch.float32).to(device)
   dir_weight_t = torch.tensor(dir_weight, dtype=torch.float32).to(device)
   n_supervised  = int(dir_weight.sum())

   if verbose:
      print(f"  supervised head labels: {n_supervised:,} drag+holding transitions")

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
      TensorDataset(obs_t, act_t, dir_labels_t, dir_weight_t),
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
      epoch_loss = epoch_grip = epoch_dxy = epoch_head = 0.0
      n_batches  = 0

      for obs_batch, act_batch, dir_batch, weight_batch in loader:
         pred = network(obs_batch)

         loss_dxy   = F.mse_loss(pred[:, 0:2], act_batch[:, 0:2])
         grip_logit = pred[:, 2]
         grip_tgt   = (act_batch[:, 2] > 0.0).float()
         loss_grip  = F.binary_cross_entropy_with_logits(
            grip_logit, grip_tgt, pos_weight=pos_weight)

         # supervised head loss: MSE against unit-direction labels,
         # weighted to only fire on drag+holding transitions
         embed_batch = obs_batch[:, _EMBED_SLICE]
         head_out    = network.post_grip_head(embed_batch)       # (batch, 2)
         per_sample  = ((head_out - dir_batch) ** 2).sum(dim=1)   # (batch,)
         n_sup       = weight_batch.sum().clamp(min=1.0)
         loss_head   = (per_sample * weight_batch).sum() / n_sup

         loss = loss_dxy + 0.5 * loss_grip + 1.0 * loss_head

         optimizer.zero_grad()
         loss.backward()
         torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
         optimizer.step()

         epoch_loss += loss.item()
         epoch_grip += loss_grip.item()
         epoch_dxy  += loss_dxy.item()
         # track head loss in epoch metrics (add new accumulator — see below)
         epoch_head += loss_head.item()
         n_batches  += 1

      scheduler.step()

      avg = epoch_loss / max(n_batches, 1)
      ag  = epoch_grip / max(n_batches, 1)
      ad  = epoch_dxy  / max(n_batches, 1)
      cur_lr = scheduler.get_last_lr()[0]

      ah = epoch_head / max(n_batches, 1)
      if verbose and (epoch % max(epochs // 5, 1) == 0 or epoch == 1):
         print(f"  epoch {epoch:3d}/{epochs} | "
               f"loss: {avg:.4f}  (grip: {ag:.4f}  dx/dy: {ad:.4f}  "
               f"head: {ah:.4f})  lr: {cur_lr:.2e}")

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