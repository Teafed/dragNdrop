"""
bc_train.py

Behavior Cloning (BC) trainer for the shape manipulation agent.

Instead of learning from scratch via RL, BC:
  1. Runs the scripted oracle to collect (obs, action) pairs cheaply.
  2. Trains an MLP to imitate those demonstrations via supervised learning.
  3. Optionally fine-tunes the result with PPO — the "warm start" cuts RL
     training time because the policy already has a sensible prior.

The output is a standard SB3 PPO model saved to disk, loadable by
demo.py / train.py exactly like a fully RL-trained model.

Usage:
    # Pure BC:
    python bc_train.py

    # BC then PPO fine-tune:
    python bc_train.py --prompt "sort shapes left to right" --finetune --finetune-steps 100000

    # Group task:
    python bc_train.py --prompt "group shapes by color" --episodes 800
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from shape_env import ShapeEnv
from llm_goal_parser import parse_goal
from oracle import collect_demonstrations


# ---------------------------------------------------------------------------
# BC network
# ---------------------------------------------------------------------------

class BCPolicy(nn.Module):
    """
    Simple MLP: obs -> action.
    Architecture mirrors SB3's default MlpPolicy so weights can be transplanted.
    obs_dim -> 64 -> Tanh -> 64 -> Tanh -> action_dim -> Tanh
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, action_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_bc(
    goal:       dict,
    dataset:    dict,
    save_path:  str,
    epochs:     int   = 20,
    batch_size: int   = 256,
    lr:         float = 3e-4,
    device:     str   = "cpu",
) -> BCPolicy:
    """Train a BCPolicy on (obs, action) pairs via MSE supervised learning."""
    obs_t = torch.tensor(dataset["observations"], dtype=torch.float32).to(device)
    act_t = torch.tensor(dataset["actions"],      dtype=torch.float32).to(device)

    obs_dim    = obs_t.shape[1]
    action_dim = act_t.shape[1]

    policy    = BCPolicy(obs_dim, action_dim).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    loss_fn   = nn.MSELoss()

    loader = DataLoader(
        TensorDataset(obs_t, act_t),
        batch_size=batch_size,
        shuffle=True,
    )

    print(f"\n--- behavior cloning ---")
    print(f"  obs dim    : {obs_dim}")
    print(f"  action dim : {action_dim}")
    print(f"  samples    : {len(obs_t):,}")
    print(f"  epochs     : {epochs}")
    print(f"  batch size : {batch_size}")
    print(f"  device     : {device}\n")

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        n_batches  = 0
        for obs_batch, act_batch in loader:
            pred = policy(obs_batch)
            loss = loss_fn(pred, act_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches  += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        if epoch % max(epochs // 5, 1) == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}/{epochs} | loss: {avg_loss:.6f}")

    os.makedirs(save_path, exist_ok=True)
    weights_path = os.path.join(save_path, "bc_weights.pt")
    torch.save(policy.state_dict(), weights_path)
    print(f"\n  BC weights saved to {weights_path}")

    return policy


# ---------------------------------------------------------------------------
# Weight transplant: BC -> SB3 PPO
# ---------------------------------------------------------------------------

def build_ppo_from_bc(goal: dict, bc_policy: BCPolicy, n_shapes: int,
                      vec_env=None) -> PPO:
    """
    Create a fresh SB3 PPO model and copy BC weights into its actor network.

    KEY FIX: accepts an optional vec_env argument. When train_oracle() passes
    the 4-env vectorized env here, the PPO model is built with n_envs=4 from
    the start, so set_env() is never needed and the n_envs mismatch crash
    (4 != 1) cannot happen.

    Falls back to a single Monitor env for standalone use (bc_train.py CLI).
    """
    env = vec_env if vec_env is not None else Monitor(ShapeEnv(n_shapes=n_shapes, goal=goal))

    model = PPO(
        "MlpPolicy",
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
        sb3_pi = model.policy.mlp_extractor.policy_net
        bc_net = bc_policy.net

        with torch.no_grad():
            sb3_pi[0].weight.copy_(bc_net[0].weight)
            sb3_pi[0].bias.copy_(bc_net[0].bias)
            sb3_pi[2].weight.copy_(bc_net[2].weight)
            sb3_pi[2].bias.copy_(bc_net[2].bias)
            model.policy.action_net.weight.copy_(bc_net[4].weight)
            model.policy.action_net.bias.copy_(bc_net[4].bias)

        print("  BC weights transplanted into PPO policy.")
    except Exception as e:
        print(f"  Weight transplant failed ({e}) -- PPO starts from random init.")

    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="train shape agent via behavior cloning from the oracle"
    )
    parser.add_argument("--prompt", type=str,
        default="sort the shapes from smallest to largest left to right")
    parser.add_argument("--episodes",      type=int,   default=500)
    parser.add_argument("--epochs",        type=int,   default=20)
    parser.add_argument("--n-shapes",      type=int,   default=2)
    parser.add_argument("--save",          type=str,   default="./models/bc_agent")
    parser.add_argument("--finetune",      action="store_true")
    parser.add_argument("--finetune-steps",type=int,   default=100_000)
    parser.add_argument("--noise",         type=float, default=0.05)
    args = parser.parse_args()

    print(f"\n--- parsing goal ---")
    print(f"prompt : \"{args.prompt}\"")
    goal = parse_goal(args.prompt)
    print(f"goal   : {goal}\n")

    print(f"--- collecting {args.episodes} oracle demonstrations ---")
    dataset = collect_demonstrations(
        goal=goal, n_episodes=args.episodes,
        n_shapes=args.n_shapes, noise_std=args.noise, verbose=True,
    )

    device    = "cuda" if torch.cuda.is_available() else "cpu"
    bc_policy = train_bc(goal=goal, dataset=dataset, save_path=args.save,
                         epochs=args.epochs, device=device)

    print("\n--- building SB3 PPO model from BC weights ---")
    # No vec_env here — standalone CLI uses single env
    model = build_ppo_from_bc(goal, bc_policy, args.n_shapes)

    bc_model_path = os.path.join(args.save, "bc_model")
    model.save(bc_model_path)
    print(f"  BC model saved to {bc_model_path}")

    if args.finetune:
        from train import make_env
        print(f"\n--- fine-tuning with PPO for {args.finetune_steps:,} steps ---")
        vec_env = make_vec_env(make_env(goal), n_envs=4)
        # Use PPO.load with env instead of set_env to avoid n_envs mismatch
        model = PPO.load(bc_model_path, env=vec_env)
        model.learn(total_timesteps=args.finetune_steps)
        ft_path = os.path.join(args.save, "bc_finetuned_model")
        model.save(ft_path)
        print(f"  Fine-tuned model saved to {ft_path}")

    print("\n--- done ---")
    print(f"Load with:  from stable_baselines3 import PPO; PPO.load('{bc_model_path}')")


if __name__ == "__main__":
    main()