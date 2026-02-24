"""
debug.py

diagnostic script — run this before worrying about the demo.
tests three things in order:
   1. environment steps actually change shape positions
   2. a random agent produces nonzero rewards
   3. a trained model (if provided) produces different actions than random

usage:
   python debug.py                          # tests env + random agent
   python debug.py --model models/shape_agent/best_model  # also tests trained model
"""

import argparse
import numpy as np
from shape_env import ShapeEnv


def test_env_steps():
   print("=== test 1: do env steps actually move shapes? ===")
   env = ShapeEnv(n_shapes=4, render_mode=None)
   obs, _ = env.reset()

   # record starting positions
   before = [(s.x, s.y) for s in env.shapes]
   print(f"  shapes before: {before}")

   # take 5 random actions
   for i in range(5):
      action = env.action_space.sample()
      obs, reward, terminated, truncated, info = env.step(action)
      print(f"  step {i+1}: action={np.round(action, 3)}  reward={reward:.4f}")

   after = [(round(s.x, 1), round(s.y, 1)) for s in env.shapes]
   print(f"  shapes after:  {after}")

   moved = before != after
   print(f"  shapes moved: {moved}")
   if not moved:
      print("  !! WARNING: shapes did not move — check step() logic")
   print()
   return moved


def test_random_rewards():
   print("=== test 2: do rewards vary with different arrangements? ===")
   env = ShapeEnv(n_shapes=4, render_mode=None)

   rewards = []
   for episode in range(5):
      obs, _ = env.reset()
      ep_reward = 0.0
      for _ in range(20):
         action = env.action_space.sample()
         obs, reward, terminated, truncated, _ = env.step(action)
         ep_reward += reward
         if terminated or truncated:
            break
      rewards.append(ep_reward)
      print(f"  episode {episode+1}: total reward = {ep_reward:.4f}")

   reward_range = max(rewards) - min(rewards)
   print(f"  reward range across episodes: {reward_range:.4f}")
   if reward_range < 0.01:
      print("  !! WARNING: rewards are not varying — reward function may be broken")
   else:
      print("  reward function looks healthy")
   print()
   return reward_range > 0.01


def test_trained_model(model_path: str):
   print(f"=== test 3: does trained model behave differently from random? ===")
   try:
      from stable_baselines3 import PPO
   except ImportError:
      print("  stable-baselines3 not installed, skipping")
      return

   try:
      model = PPO.load(model_path)
      print(f"  loaded model from {model_path}")
   except Exception as e:
      print(f"  could not load model: {e}")
      return

   env = ShapeEnv(n_shapes=4, render_mode=None)

   # run trained model for one episode, record actions
   obs, _ = env.reset()
   trained_actions = []
   trained_rewards = []
   for _ in range(50):
      action, _ = model.predict(obs, deterministic=True)
      obs, reward, terminated, truncated, _ = env.step(action)
      trained_actions.append(action)
      trained_rewards.append(reward)
      if terminated or truncated:
         break

   # run random agent for same number of steps
   obs, _ = env.reset()
   random_actions = []
   random_rewards = []
   for _ in range(len(trained_actions)):
      action = env.action_space.sample()
      obs, reward, terminated, truncated, _ = env.step(action)
      random_actions.append(action)
      random_rewards.append(reward)
      if terminated or truncated:
         break

   trained_mean = np.mean(trained_rewards)
   random_mean  = np.mean(random_rewards)
   print(f"  trained agent mean reward: {trained_mean:.4f}")
   print(f"  random  agent mean reward: {random_mean:.4f}")

   action_std_trained = np.std(trained_actions)
   action_std_random  = np.std(random_actions)
   print(f"  trained action std: {action_std_trained:.4f}  "
         f"(low = agent is picking consistent targets)")
   print(f"  random  action std: {action_std_random:.4f}")

   if trained_mean > random_mean:
      print("  trained model outperforms random — learning is happening")
   else:
      print("  !! trained model not beating random yet — "
            "may need more training or reward tuning")
   print()


def test_render():
   print("=== test 4: does pygame render without crashing? ===")
   try:
      import pygame
      env = ShapeEnv(n_shapes=4, render_mode="human")
      obs, _ = env.reset()

      print("  pygame window should appear — running 60 frames then closing")
      for i in range(60):
         action = env.action_space.sample()
         obs, reward, terminated, truncated, _ = env.step(action)
         env.render()

         # print shape positions every 20 frames so you can see they're changing
         if i % 20 == 0:
            positions = [(round(s.x, 1), round(s.y, 1)) for s in env.shapes]
            print(f"  frame {i:3d}: positions={positions}  reward={reward:.4f}")

         if terminated or truncated:
            obs, _ = env.reset()

      env.close()
      print("  render test passed — if you saw a window with moving shapes, "
            "the env is working fine")
   except Exception as e:
      print(f"  !! render error: {e}")
   print()


if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument("--model", type=str, default=None,
                       help="path to a trained model .zip to test")
   parser.add_argument("--skip-render", action="store_true",
                       help="skip the pygame render test (useful on headless machines)")
   args = parser.parse_args()

   ok1 = test_env_steps()
   ok2 = test_random_rewards()

   if args.model:
      test_trained_model(args.model)

   if not args.skip_render:
      test_render()
   else:
      print("=== test 4: skipped (--skip-render) ===")

   print("=== summary ===")
   print(f"  env steps move shapes: {ok1}")
   print(f"  rewards vary:          {ok2}")
   print()
   print("if both are true and the render window showed moving shapes, "
         "the environment is healthy.")
   print("if shapes weren't moving in the render window, the bug is in "
         "the demo loop, not the env.")
