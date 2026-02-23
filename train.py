"""
train.py

entry point for training the shape manipulation agent.
uses stable-baselines3 PPO with the ShapeEnv gymnasium environment.
the goal is parsed from a prompt (stubbed LLM) and injected into
the environment before training begins.

usage:
   python train.py
   python train.py --prompt "sort shapes right to left"
   python train.py --prompt "arrange top to bottom" --timesteps 200000
   python train.py --load models/shape_agent --demo
"""

import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

from shape_env import ShapeEnv
from llm_goal_parser import parse_goal


def make_env(goal, render_mode=None):
   """factory function that creates an env with a specific goal."""
   def _init():
      return ShapeEnv(n_shapes=4, goal=goal, render_mode=render_mode)
   return _init


def train(prompt: str, timesteps: int, save_path: str):
   print(f"\n--- parsing goal from prompt ---")
   print(f"prompt: \"{prompt}\"")
   goal = parse_goal(prompt)
   print(f"goal:   {goal}\n")

   # vectorized env — SB3 runs multiple envs in parallel to collect
   # experience faster. 4 is a reasonable number for a laptop CPU.
   n_envs = 4
   vec_env = make_vec_env(make_env(goal), n_envs=n_envs)

   # separate eval env (single, no parallelism needed)
   eval_env = ShapeEnv(n_shapes=4, goal=goal, render_mode=None)

   # save the best model whenever eval improves
   eval_callback = EvalCallback(
      eval_env,
      best_model_save_path=save_path,
      log_path="./logs/",
      eval_freq=max(5000 // n_envs, 1),
      n_eval_episodes=10,
      verbose=1,
   )

   # PPO is a solid default for continuous action spaces.
   # these hyperparameters are a reasonable starting point —
   # expect to tune them as you iterate.
   model = PPO(
      "MlpPolicy",
      vec_env,
      learning_rate=3e-4,
      n_steps=2048,
      batch_size=64,
      n_epochs=10,
      gamma=0.99,
      gae_lambda=0.95,
      clip_range=0.2,
      ent_coef=0.01,   # small entropy bonus encourages exploration
      verbose=1,
      tensorboard_log="./logs/tensorboard/",
   )

   print(f"--- starting training for {timesteps} timesteps ---\n")
   model.learn(total_timesteps=timesteps, callback=eval_callback)

   final_path = os.path.join(save_path, "final_model")
   model.save(final_path)
   print(f"\n--- training done. model saved to {final_path} ---")
   return model


def demo(model_path: str, prompt: str):
   """
   load a saved model and run it in the pygame window so you can
   watch it interact with the environment.
   """
   print(f"\n--- loading model from {model_path} ---")
   goal  = parse_goal(prompt)
   print(f"goal: {goal}\n")

   env   = ShapeEnv(n_shapes=4, goal=goal, render_mode="human")
   model = PPO.load(model_path)

   obs, _ = env.reset()
   total_reward = 0.0
   steps = 0

   print("running demo — close the pygame window to stop.\n")
   try:
      while True:
         action, _ = model.predict(obs, deterministic=True)
         obs, reward, terminated, truncated, info = env.step(action)
         total_reward += reward
         steps += 1

         if terminated or truncated:
            print(f"episode done — steps: {steps}, "
                  f"total reward: {total_reward:.3f}")
            obs, _ = env.reset()
            total_reward = 0.0
            steps = 0
   except KeyboardInterrupt:
      pass
   finally:
      env.close()


if __name__ == "__main__":
   parser = argparse.ArgumentParser(
      description="train or demo the shape manipulation agent"
   )
   parser.add_argument(
      "--prompt",
      type=str,
      default="sort the shapes from smallest to largest left to right",
      help="natural language goal prompt (passed to LLM parser)",
   )
   parser.add_argument(
      "--timesteps",
      type=int,
      default=100_000,
      help="total training timesteps",
   )
   parser.add_argument(
      "--save",
      type=str,
      default="./models/shape_agent",
      help="directory to save model checkpoints",
   )
   parser.add_argument(
      "--load",
      type=str,
      default=None,
      help="path to a saved model to load for demo mode",
   )
   parser.add_argument(
      "--demo",
      action="store_true",
      help="run demo mode (requires --load)",
   )
   args = parser.parse_args()

   if args.demo:
      if args.load is None:
         print("error: --demo requires --load <model_path>")
      else:
         demo(args.load, args.prompt)
   else:
      train(args.prompt, args.timesteps, args.save)
