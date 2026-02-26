"""
train.py

Entry point for training the shape manipulation agent.

Two training modes:

  Default (PPO from scratch):
      python train.py
      python train.py --prompt "sort shapes right to left"
      python train.py --prompt "arrange top to bottom" --timesteps 200000

  Oracle warm-start (recommended — BC then PPO fine-tune):
      python train.py --oracle
      python train.py --oracle --prompt "group shapes by color" --bc-episodes 500
      python train.py --oracle --timesteps 100000 --bc-episodes 1000

  Demo mode:
      python train.py --load models/shape_agent/best_model --demo
      python train.py --load models/shape_agent/best_model --demo --prompt "sort right to left"

The --oracle flag collects oracle demonstrations, trains a BC policy on them,
transplants the BC weights into a PPO model, then fine-tunes with PPO.
This typically converges 5-10× faster than PPO from random initialization.
"""

import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

from shape_env import ShapeEnv
from llm_goal_parser import parse_goal
from callbacks import ShapeTaskCallback


def make_env(goal: dict, render_mode: str = None):
    """
    Factory that returns a callable creating a Monitor-wrapped ShapeEnv.
    SB3's make_vec_env calls this n_envs times to spin up parallel workers.
    """
    def _init():
        return Monitor(ShapeEnv(n_shapes=2, goal=goal, render_mode=render_mode))
    return _init


def build_callbacks(goal: dict, save_path: str, n_envs: int) -> CallbackList:
    """
    Build the standard callback stack used by both training modes.
    Factored out so train() and train_oracle() don't duplicate this.
    """
    eval_env = Monitor(ShapeEnv(n_shapes=2, goal=goal, render_mode=None))

    # EvalCallback saves the best model by mean episode reward
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_path,
        log_path="./logs/",
        eval_freq=max(5000 // n_envs, 1),
        n_eval_episodes=10,
        verbose=1,
    )

    # ShapeTaskCallback logs task-specific metrics to TensorBoard
    task_callback = ShapeTaskCallback(
        eval_env=Monitor(ShapeEnv(n_shapes=2, goal=goal, render_mode=None)),
        eval_freq=5000,
        n_eval_episodes=10,
        verbose=1,
    )

    return CallbackList([eval_callback, task_callback])


# ---------------------------------------------------------------------------
# PPO from scratch
# ---------------------------------------------------------------------------

def train(prompt: str, timesteps: int, save_path: str):
    """Train a PPO agent from a random initialisation."""
    print(f"\n--- parsing goal ---")
    print(f"prompt: \"{prompt}\"")
    goal = parse_goal(prompt)
    print(f"goal  : {goal}\n")

    n_envs  = 4
    vec_env = make_vec_env(make_env(goal), n_envs=n_envs)

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
        ent_coef=0.02,   # entropy bonus encourages exploration from random init
        verbose=1,
        tensorboard_log="./logs/tensorboard/",
    )

    callbacks = build_callbacks(goal, save_path, n_envs)

    print(f"--- training PPO from scratch for {timesteps:,} timesteps ---\n")
    model.learn(total_timesteps=timesteps, callback=callbacks)

    final_path = os.path.join(save_path, "final_model")
    model.save(final_path)
    print(f"\n--- done. model saved to {final_path} ---")
    return model


# ---------------------------------------------------------------------------
# Oracle warm-start: BC → PPO fine-tune
# ---------------------------------------------------------------------------

def train_oracle(
    prompt:      str,
    timesteps:   int,
    save_path:   str,
    bc_episodes: int = 500,
    bc_epochs:   int = 20,
):
    """
    1. Parse goal from prompt.
    2. Collect oracle demonstrations cheaply.
    3. Train a BC policy via supervised learning.
    4. Transplant BC weights into a fresh PPO model.
    5. Fine-tune with PPO for `timesteps` steps.

    The BC step (2-3) takes ~10-30 seconds.
    The PPO fine-tune step (5) is typically 5-10× shorter than training from scratch.
    """
    # Lazy imports so the base train.py still works without torch installed
    from bc_train import train_bc, build_ppo_from_bc
    from oracle import collect_demonstrations
    import torch

    print(f"\n--- parsing goal ---")
    print(f"prompt: \"{prompt}\"")
    goal = parse_goal(prompt)
    print(f"goal  : {goal}\n")

    # Step 1: oracle demonstrations
    print(f"--- collecting {bc_episodes} oracle demonstrations ---")
    dataset = collect_demonstrations(
        goal=goal,
        n_episodes=bc_episodes,
        n_shapes=2,
        verbose=True,
    )

    # Step 2: behavior cloning
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    bc_policy = train_bc(
        goal=goal,
        dataset=dataset,
        save_path=save_path,
        epochs=bc_epochs,
        device=device,
    )

    # Step 3: build vec_env first, then pass it into build_ppo_from_bc.
    # This ensures the PPO model is created with n_envs=4 from the start,
    # so set_env() is never needed and the (4 != 1) crash cannot happen.
    n_envs  = 4
    vec_env = make_vec_env(make_env(goal), n_envs=n_envs)

    print("\n--- initialising PPO from BC weights ---")
    model = build_ppo_from_bc(goal, bc_policy, n_shapes=2, vec_env=vec_env)

    callbacks = build_callbacks(goal, save_path, n_envs)

    print(f"\n--- fine-tuning with PPO for {timesteps:,} timesteps ---\n")
    model.learn(total_timesteps=timesteps, callback=callbacks)

    final_path = os.path.join(save_path, "oracle_final_model")
    model.save(final_path)
    print(f"\n--- done. model saved to {final_path} ---")
    return model


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo(model_path: str, prompt: str):
    """
    Load a saved model and run it in the pygame window.
    Prints episode outcome (SOLVED / timed out) and resets automatically.
    """
    import pygame

    print(f"\n--- loading model from {model_path} ---")
    goal  = parse_goal(prompt)
    print(f"goal: {goal}\n")

    env   = ShapeEnv(n_shapes=2, goal=goal, render_mode="human")
    model = PPO.load(model_path)

    obs, _       = env.reset()
    total_reward = 0.0
    steps        = 0
    episode      = 1

    # Render once so pygame's video system is initialised before event polling
    env.render()
    pygame.display.flip()

    print("running demo — close the pygame window to stop.\n")
    try:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
            if not running:
                break

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            pygame.display.flip()
            env.clock.tick(env.metadata["render_fps"])

            total_reward += reward
            steps        += 1

            if terminated or truncated:
                status = "SOLVED" if terminated else "timed out"
                print(f"episode {episode} {status} — "
                      f"steps: {steps}, total reward: {total_reward:.3f}")
                obs, _       = env.reset()
                total_reward = 0.0
                steps        = 0
                episode     += 1

    except KeyboardInterrupt:
        pass
    finally:
        env.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="train or demo the shape manipulation agent"
    )
    parser.add_argument(
        "--prompt", type=str,
        default="sort the shapes from smallest to largest left to right",
        help="natural language goal prompt",
    )
    parser.add_argument(
        "--timesteps", type=int, default=300_000,
        help="total PPO training timesteps",
    )
    parser.add_argument(
        "--save", type=str, default="./models/shape_agent",
        help="directory to save model checkpoints",
    )
    parser.add_argument(
        "--load", type=str, default=None,
        help="path to a saved model (for --demo mode)",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="run demo mode (requires --load)",
    )
    # Oracle warm-start flags
    parser.add_argument(
        "--oracle", action="store_true",
        help="use oracle warm-start (BC + PPO fine-tune) instead of PPO from scratch",
    )
    parser.add_argument(
        "--bc-episodes", type=int, default=500,
        help="oracle demo episodes to collect for BC (only with --oracle)",
    )
    parser.add_argument(
        "--bc-epochs", type=int, default=20,
        help="BC training epochs (only with --oracle)",
    )
    args = parser.parse_args()

    if args.demo:
        if args.load is None:
            print("error: --demo requires --load <model_path>")
        else:
            demo(args.load, args.prompt)
    elif args.oracle:
        train_oracle(
            prompt=args.prompt,
            timesteps=args.timesteps,
            save_path=args.save,
            bc_episodes=args.bc_episodes,
            bc_epochs=args.bc_epochs,
        )
    else:
        train(args.prompt, args.timesteps, args.save)