"""
demo.py

standalone demo — loads a trained model and runs it live in a
pygame window. owns all pygame initialization itself rather than
relying on the env's lazy init, which avoids the video system
not initialized error.

usage:
   python demo.py --model models/shape_agent/best_model
   python demo.py --model models/shape_agent/best_model --prompt "sort right to left"
   python demo.py --model models/shape_agent/best_model --random  # watch a random agent
"""

import argparse
import sys
import numpy as np
import pygame

from shape_env import ShapeEnv, WINDOW_W, WINDOW_H, FPS, BG_COLOR, COLORS
from llm_goal_parser import parse_goal


def draw_env(surface, env, font):
   """draw current env state onto an existing pygame surface."""
   surface.fill(BG_COLOR)

   for shape in env.shapes:
      shape.draw(surface, font)

   # goal + reward overlay
   goal      = env.goal
   reward    = env._compute_score()
   goal_str  = (f"goal: {goal['task']} | "
                f"axis: {goal['axis']} | "
                f"dir: {goal['direction']}   "
                f"reward: {reward:.3f}")
   label = font.render(goal_str, True, (200, 200, 200))
   surface.blit(label, (10, 10))


def run_demo(model_path, prompt, use_random):
   # --- pygame init happens here, before anything else ---
   pygame.init()
   window = pygame.display.set_mode((WINDOW_W, WINDOW_H))
   pygame.display.set_caption("shape manipulation — demo")
   clock  = pygame.time.Clock()
   font   = pygame.font.SysFont("monospace", 12)

   # --- goal ---
   goal = parse_goal(prompt)
   print(f"goal: {goal}")

   # --- env (no render_mode — we draw manually) ---
   env    = ShapeEnv(n_shapes=4, goal=goal, render_mode=None)
   obs, _ = env.reset()

   # --- model ---
   if use_random:
      print("running random agent (no model loaded)")
      model = None
   else:
      try:
         from stable_baselines3 import PPO
         model = PPO.load(model_path)
         print(f"loaded model from {model_path}")
      except Exception as e:
         print(f"could not load model: {e}")
         pygame.quit()
         sys.exit(1)

   total_reward = 0.0
   steps        = 0
   episode      = 1
   print(f"\nepisode {episode} — close window or press Q to quit\n")

   running = True
   while running:
      # --- event handling ---
      for event in pygame.event.get():
         if event.type == pygame.QUIT:
            running = False
         if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
               running = False

      if not running:
         break

      # --- agent picks action ---
      if model is not None:
         action, _ = model.predict(obs, deterministic=True)
      else:
         action = env.action_space.sample()

      obs, reward, terminated, truncated, _ = env.step(action)
      total_reward += reward
      steps        += 1

      # --- draw ---
      draw_env(window, env, font)

      # highlight which shape was just moved
      shape_idx = int(np.clip(
         round((action[0] + 1.0) / 2.0 * (env.n_shapes - 1)),
         0, env.n_shapes - 1
      ))
      s = env.shapes[shape_idx]
      pygame.draw.circle(window, (255, 255, 255),
                         (int(s.x), int(s.y)), s.radius + 4, 2)

      # step counter overlay
      step_label = font.render(
         f"episode {episode}   step {steps}   "
         f"{'SOLVED' if terminated else ''}",
         True, (180, 180, 100)
      )
      window.blit(step_label, (10, WINDOW_H - 24))

      pygame.display.flip()
      clock.tick(FPS)

      # --- episode reset ---
      if terminated or truncated:
         status = "SOLVED" if terminated else "timed out"
         print(f"episode {episode} {status} — "
               f"steps: {steps}  total reward: {total_reward:.3f}")
         obs, _ = env.reset()
         total_reward = 0.0
         steps        = 0
         episode     += 1
         print(f"episode {episode}")

   pygame.quit()
   print("demo closed")


if __name__ == "__main__":
   parser = argparse.ArgumentParser(description="run a trained agent demo")
   parser.add_argument(
      "--model", type=str, default="models/shape_agent/best_model",
      help="path to saved SB3 model (without .zip extension)"
   )
   parser.add_argument(
      "--prompt", type=str,
      default="sort the shapes from smallest to largest left to right",
      help="natural language goal prompt"
   )
   parser.add_argument(
      "--random", action="store_true",
      help="use a random agent instead of a trained model"
   )
   args = parser.parse_args()
   run_demo(args.model, args.prompt, args.random)
