"""
train.py

entry point for goal-conditioned training of the shape manipulation agent.

the agent is trained on ALL tasks in TASK_POOL simultaneously. each episode
samples a random task prompt, encodes it via the goal encoder MLP, and injects
the encoding into the observation. the policy learns to condition its behavior
on the goal signal rather than being a specialist on one task.

training modes:

  oracle warm-start (recommended):
      python train.py
      python train.py --timesteps 200000 --bc-episodes 1000

  PPO from scratch (slower, useful for ablations):
      python train.py --no-oracle

  demo:
      python train.py --demo --load models/shape_agent/best_model
      python train.py --demo --load models/shape_agent/best_model --prompt "sort right to left"
"""

import argparse
import os
import random
import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env

from shape_env import ShapeEnv
from llm_goal_parser import parse_goal, get_embedding
from callbacks import ShapeTaskCallback
from config import (
    TASK_POOL, MAX_SHAPES, POLICY_HIDDEN_SIZE, get_obs_size,
)


# ---------------------------------------------------------------------------
# goal-conditioned env factory
# ---------------------------------------------------------------------------

def make_goal_conditioned_env(goal_encoder, render_mode=None):
    """
    returns a factory function for Monitor-wrapped ShapeEnvs.
    each call samples a fresh task from TASK_POOL and injects the goal
    encoding so the obs is always goal-conditioned.
    """
    def _init():
        prompt   = random.choice(TASK_POOL)
        goal     = parse_goal(prompt)
        raw_emb  = get_embedding(prompt)

        with torch.no_grad():
            emb_t    = torch.tensor(raw_emb, dtype=torch.float32).unsqueeze(0)
            encoding = goal_encoder(emb_t).squeeze(0).numpy()

        env = ShapeEnv(goal=goal)   # n_shapes=None -> sampled randomly each episode
        env.set_goal_encoding(encoding)
        return Monitor(env)

    return _init


# ---------------------------------------------------------------------------
# callbacks
# ---------------------------------------------------------------------------

def build_callbacks(goal_encoder, save_path: str, n_envs: int) -> CallbackList:
    """
    build the standard callback stack.
    eval envs sample tasks from TASK_POOL so metrics are task-averaged.
    """
    def _make_eval_env():
        prompt   = random.choice(TASK_POOL)
        goal     = parse_goal(prompt)
        raw_emb  = get_embedding(prompt)
        with torch.no_grad():
            emb_t    = torch.tensor(raw_emb, dtype=torch.float32).unsqueeze(0)
            encoding = goal_encoder(emb_t).squeeze(0).numpy()
        env = ShapeEnv(goal=goal)
        env.set_goal_encoding(encoding)
        return Monitor(env)

    eval_callback = EvalCallback(
        _make_eval_env(),
        best_model_save_path=save_path,
        log_path="./logs/",
        eval_freq=max(5000 // n_envs, 1),
        n_eval_episodes=10,
        verbose=1,
    )

    task_callback = ShapeTaskCallback(
        eval_env=_make_eval_env(),
        eval_freq=5000,
        n_eval_episodes=10,
        verbose=1,
    )

    return CallbackList([eval_callback, task_callback])


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

def train(
    timesteps:   int  = 300_000,
    save_path:   str  = "./models/shape_agent",
    bc_episodes: int  = 500,
    bc_epochs:   int  = 20,
    use_oracle:  bool = True,
):
    """
    train the goal-conditioned agent.

    if use_oracle=True (default):
        1. initialise goal encoder
        2. collect oracle demos across all TASK_POOL tasks
        3. train BCPolicy via supervised learning
        4. transplant BC weights into PPO
        5. PPO fine-tune

    if use_oracle=False:
        skip steps 1-4 and train PPO from random init.
    """
    from bc_train import GoalEncoder, BCPolicy, train_bc, build_ppo_from_bc
    from oracle import collect_demonstrations

    goal_encoder = GoalEncoder()

    if use_oracle:
        print(f"\n--- collecting {bc_episodes} oracle demonstrations "
              f"across {len(TASK_POOL)} task pool prompts ---")
        dataset = collect_demonstrations(
            goal_encoder=goal_encoder,
            n_episodes=bc_episodes,
            verbose=True,
        )

        device    = "cuda" if torch.cuda.is_available() else "cpu"
        bc_policy, goal_encoder = train_bc(
            dataset=dataset,
            save_path=save_path,
            epochs=bc_epochs,
            device=device,
        )
        goal_encoder.eval()

        n_envs  = 4
        vec_env = make_vec_env(
            make_goal_conditioned_env(goal_encoder), n_envs=n_envs)

        print("\n--- initialising PPO from BC weights ---")
        model = build_ppo_from_bc(bc_policy, n_shapes=MAX_SHAPES, vec_env=vec_env)

    else:
        goal_encoder.eval()
        n_envs  = 4
        vec_env = make_vec_env(
            make_goal_conditioned_env(goal_encoder), n_envs=n_envs)

        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=1e-4,
            n_steps=2048,
            batch_size=128,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.02,
            verbose=1,
            tensorboard_log="./logs/tensorboard/",
            policy_kwargs=dict(net_arch=[POLICY_HIDDEN_SIZE, POLICY_HIDDEN_SIZE]),
        )

    callbacks = build_callbacks(goal_encoder, save_path, n_envs)

    print(f"\n--- training with PPO for {timesteps:,} timesteps ---\n")
    model.learn(total_timesteps=timesteps, callback=callbacks)

    final_path = os.path.join(save_path, "final_model")
    model.save(final_path)
    print(f"\n--- done. model saved to {final_path} ---")
    return model, goal_encoder


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="train the goal-conditioned shape manipulation agent. "
                    "for demos use demo.py instead."
    )
    parser.add_argument(
        "--timesteps", type=int, default=300_000,
        help="total PPO training timesteps",
    )
    parser.add_argument(
        "--save", type=str, default="./models/shape_agent",
        help="directory to save model checkpoints and goal encoder",
    )
    parser.add_argument(
        "--no-oracle", action="store_true",
        help="skip oracle warm-start and train PPO from random init",
    )
    parser.add_argument(
        "--bc-episodes", type=int, default=500,
        help="oracle demo episodes to collect for BC warm-start",
    )
    parser.add_argument(
        "--bc-epochs", type=int, default=20,
        help="BC training epochs",
    )
    args = parser.parse_args()

    train(
        timesteps=args.timesteps,
        save_path=args.save,
        bc_episodes=args.bc_episodes,
        bc_epochs=args.bc_epochs,
        use_oracle=not args.no_oracle,
    )
