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
from llm_goal_parser import parse_goal, get_embedding
from oracle import collect_demonstrations
from config import (
    EMBEDDING_DIM, GOAL_ENCODING_DIM, POLICY_HIDDEN_SIZE,
    get_obs_size, TASK_POOL,
)


# ---------------------------------------------------------------------------
# goal encoder MLP
# ---------------------------------------------------------------------------

class GoalEncoder(nn.Module):
    """
    projects raw EMBEDDING_DIM embeddings down to GOAL_ENCODING_DIM.
    sits between get_embedding() and the policy input.
    kept small (two layers) so it doesn't dominate training.
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
# BC network
# ---------------------------------------------------------------------------

class BCPolicy(nn.Module):
    """
    simple MLP: obs -> action.
    architecture mirrors SB3's MlpPolicy so weights can be transplanted.
    obs_dim -> POLICY_HIDDEN_SIZE -> Tanh -> POLICY_HIDDEN_SIZE -> Tanh -> action_dim -> Tanh

    obs_dim is get_obs_size() from config — fixed regardless of n_shapes.
    """

    def __init__(self, action_dim: int = 3, hidden: int = POLICY_HIDDEN_SIZE):
        super().__init__()
        obs_dim  = get_obs_size()
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
    dataset:    dict,
    save_path:  str,
    epochs:     int   = 20,
    batch_size: int   = 256,
    lr:         float = 3e-4,
    device:     str   = "cpu",
) -> tuple:
    """
    train a BCPolicy and GoalEncoder on (obs, action) pairs via MSE.

    the dataset must already have goal encodings baked into the observations
    (done by collect_demonstrations sampling from TASK_POOL and calling
    get_embedding + GoalEncoder before storing each obs).

    returns:
        (BCPolicy, GoalEncoder) — both trained, moved to cpu for saving.
    """
    obs_t = torch.tensor(dataset["observations"], dtype=torch.float32).to(device)
    act_t = torch.tensor(dataset["actions"],      dtype=torch.float32).to(device)

    policy       = BCPolicy().to(device)
    goal_encoder = GoalEncoder().to(device)

    # train policy and goal encoder jointly — goal encoder is already
    # applied to observations before this point, so we only optimize policy here.
    # goal encoder is saved separately for use in train.py.
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    loss_fn   = nn.MSELoss()

    loader = DataLoader(
        TensorDataset(obs_t, act_t),
        batch_size=batch_size,
        shuffle=True,
    )

    obs_dim    = obs_t.shape[1]
    action_dim = act_t.shape[1]

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
    encoder_path = os.path.join(save_path, "goal_encoder.pt")
    torch.save(policy.state_dict(),       weights_path)
    torch.save(goal_encoder.state_dict(), encoder_path)
    print(f"\n  BC weights saved to      {weights_path}")
    print(f"  goal encoder saved to    {encoder_path}")

    return policy.cpu(), goal_encoder.cpu()


# ---------------------------------------------------------------------------
# Weight transplant: BC -> SB3 PPO (same stage)
# ---------------------------------------------------------------------------

def build_ppo_from_bc(bc_policy: BCPolicy, n_shapes: int,
                      vec_env=None, goal: dict = None) -> PPO:
    """
    create a fresh SB3 PPO model and copy BC weights into its actor network.
    uses POLICY_HIDDEN_SIZE from config for hidden layer dimensions.
    """
    if goal is None:
        goal = {"task": "sort_by_size", "axis": "x",
                "direction": "ascending", "attribute": "size", "region": "none"}

    env = vec_env if vec_env is not None else Monitor(
        ShapeEnv(n_shapes=n_shapes, goal=goal))

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
        policy_kwargs=dict(
            net_arch=[POLICY_HIDDEN_SIZE, POLICY_HIDDEN_SIZE]
        ),
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
        print(f"  weight transplant failed ({e}) -- PPO starts from random init.")

    return model


# ---------------------------------------------------------------------------
# Weight transplant: previous stage PPO -> new stage PPO (cross-stage)
# ---------------------------------------------------------------------------

def transplant_across_stages(prev_model_path: str, new_model: PPO,
                              prev_obs_dim: int) -> PPO:
    """
    Copy weights from a smaller-obs-space PPO model into a larger one.

    The hidden layers (64->64) and action head are obs-size-independent so
    they transfer directly. The input layer grows because the new stage has
    more shapes, so we:
      - copy the old input weights into the first prev_obs_dim columns
      - leave the remaining columns (new shape features) as small random values

    This preserves everything the previous stage learned about existing shapes
    while giving the policy fresh capacity for the new shape's features.

    Args:
        prev_model_path: path to the stage N model .zip (without extension)
        new_model:       freshly created PPO model for stage N+1
        prev_obs_dim:    obs space size of the stage N model
    """
    from stable_baselines3 import PPO as PPO_

    try:
        prev = PPO_.load(prev_model_path)
    except Exception as e:
        print(f"  cross-stage transplant: could not load {prev_model_path} ({e})")
        print("  starting stage from random init instead.")
        return new_model

    prev_pi = prev.policy.mlp_extractor.policy_net
    new_pi  = new_model.policy.mlp_extractor.policy_net

    with torch.no_grad():
        # input layer: partial copy into first prev_obs_dim columns
        new_in_w = new_pi[0].weight.clone()
        new_in_w[:, :prev_obs_dim] = prev_pi[0].weight[:, :prev_obs_dim]
        # scale new columns small so they don't dominate on first forward pass
        nn.init.normal_(new_in_w[:, prev_obs_dim:], mean=0.0, std=0.01)
        new_pi[0].weight.copy_(new_in_w)
        new_pi[0].bias.copy_(prev_pi[0].bias)

        # hidden layer: full copy
        new_pi[2].weight.copy_(prev_pi[2].weight)
        new_pi[2].bias.copy_(prev_pi[2].bias)

        # action head: full copy
        new_model.policy.action_net.weight.copy_(
            prev.policy.action_net.weight)
        new_model.policy.action_net.bias.copy_(
            prev.policy.action_net.bias)

    print(f"  cross-stage transplant: copied weights from {prev_model_path}")
    print(f"  input layer: {prev_obs_dim} cols preserved, "
          f"{new_pi[0].weight.shape[1] - prev_obs_dim} new cols initialised small.")
    return new_model


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