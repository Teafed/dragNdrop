"""
demo.py

standalone demo — loads a trained model and runs it live in a pygame window.
owns all pygame initialisation itself rather than relying on the env's lazy
init, which avoids the "video system not initialized" error.

usage:
   python demo.py --model models/shape_agent/best_model
   python demo.py --model models/shape_agent/best_model --prompt "sort shapes smallest to largest"
   python demo.py --oracle --prompt "arrange shapes in a horizontal line evenly spaced"
   python demo.py --oracle --sequential
   python demo.py --random
   python demo.py --model models/shape_agent/best_model --headless --episodes 200
"""

import argparse
import os
import random
import sys
import numpy as np
import torch
import pygame

from shape_env import ShapeEnv, WINDOW_W, WINDOW_H, FPS, BG_COLOR
from llm_goal_parser import parse_goal, get_embedding
from config import GOAL_ENCODING_DIM, TASK_POOL


# ---------------------------------------------------------------------------
# goal encoder helpers
# ---------------------------------------------------------------------------

def load_goal_encoder(model_path: str):
   """
   try to load the goal encoder saved alongside the model.
   returns a GoalEncoder instance (random init if file not found).
   """
   from bc_train import GoalEncoder
   encoder_path = os.path.join(os.path.dirname(model_path), "goal_encoder.pt")
   encoder      = GoalEncoder()
   if os.path.exists(encoder_path):
      encoder.load_state_dict(torch.load(encoder_path, map_location="cpu"))
      print(f"goal encoder loaded from {encoder_path}")
   else:
      print(f"no goal encoder found at {encoder_path} — using zero encoding")
   encoder.eval()
   return encoder


def encode_goal(goal_encoder, prompt: str) -> np.ndarray:
   """project a prompt string to a GOAL_ENCODING_DIM vector."""
   raw_emb = get_embedding(prompt)
   with torch.no_grad():
      emb_t    = torch.tensor(raw_emb, dtype=torch.float32).unsqueeze(0)
      encoding = goal_encoder(emb_t).squeeze(0).numpy()
   return encoding


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

LEGEND_LINES = [
   "Q      quit",
   "N      next episode",
   "SPC    pause / unpause",
   "S      step (one step; pauses first)",
   "D      dump state",
   "SHFT+D episode summary",
]


def draw_scene(surface, env, font, episode, steps, prompt,
               agent_label, paused, phase=None):
   """
   draw full frame — shared by all agent types.

   layout:
      top row:    task / progress / rank / prompt
      top-right:  key legend
      bottom bar: agent label, episode, step, phase, solved/pause indicator
   """
   surface.fill(BG_COLOR)

   for shape in env.shapes:
      shape.draw(surface, font)

   # highlight the target shape for reach/touch/drag tasks so the viewer
   # can tell which shape the agent is supposed to interact with
   task = env.goal.get("task", "")
   if task in ("reach", "touch", "drag") and env.shapes:
      t = env.shapes[env.target_idx]
      pygame.draw.circle(
         surface, (255, 220, 60),
         (int(t.x), int(t.y)), int(t.radius) + 6, 2)

   # draw cursor via env's own method
   _prev      = env.window
   env.window = surface
   env._draw_cursor()
   env.window = _prev

   # top-left HUD: task info
   score   = env._compute_score()
   rank    = env._compute_rank_corr()
   hud_str = (f"task: {task}   progress: {score:.2%}   "
              f"rank/cohesion: {rank:+.2f}")
   surface.blit(font.render(hud_str, True, (200, 200, 200)), (10, 10))

   # prompt on second line — truncate if very long
   prompt_display = prompt if len(prompt) <= 80 else prompt[:77] + "..."
   surface.blit(
      font.render(f"prompt: {prompt_display}", True, (160, 200, 160)),
      (10, 26))

   # top-right legend
   for i, line in enumerate(LEGEND_LINES):
      surface.blit(
         font.render(line, True, (140, 140, 140)),
         (WINDOW_W - 200, 28 + i * 14))

   # bottom bar
   solved_str = "  SOLVED!" if env._is_solved() else ""
   pause_str  = "  [PAUSED]" if paused else ""
   phase_str  = f"  phase={phase}" if phase is not None else ""
   bottom     = (f"{agent_label}  ep {episode}  step {steps}  "
                 f"score {score:.3f}{phase_str}{solved_str}{pause_str}")
   color      = (255, 220, 80) if paused else (100, 220, 180)
   surface.blit(font.render(bottom, True, color), (10, WINDOW_H - 24))


# ---------------------------------------------------------------------------
# state and episode summary dumps
# ---------------------------------------------------------------------------

def dump_state(env, step, episode, extra=None):
   """print current env state — bound to D key."""
   score = env._compute_score()
   print(f"\n--- state dump  ep {episode}  step {step} ---")
   print(f"  task    : {env.goal.get('task')}  score={score:.4f}")
   if extra:
      print(f"  {extra}")
   print(f"  cursor  : ({env.cx:.1f}, {env.cy:.1f})  "
         f"holding={env.holding}  grabbed_idx={env.grabbed_idx}")
   for i, s in enumerate(env.shapes):
      marker = " <-- target" if i == env.target_idx else ""
      print(f"  shape {i} : ({s.x:.1f}, {s.y:.1f})  "
            f"color={getattr(s,'color_name','?')}  "
            f"size={s.size:.1f}  type={s.shape_type}{marker}")
   print()


def dump_episode_summary(history: list):
   """
   print a summary of all completed episodes — bound to Shift+D.

   history: list of dicts with keys: episode, prompt, steps, reward,
            score, solved, task.
   """
   if not history:
      print("\n[no completed episodes yet]")
      return

   n        = len(history)
   solved   = [e for e in history if e["solved"]]
   solve_r  = len(solved) / n

   # per-task breakdown
   by_task  = {}
   for e in history:
      t = e["task"]
      by_task.setdefault(t, []).append(e)

   print(f"\n{'='*60}")
   print(f"  episode summary  ({n} episodes)")
   print(f"{'='*60}")
   print(f"  overall solve rate : {solve_r:.0%}  ({len(solved)}/{n})")
   print(f"  mean steps         : {np.mean([e['steps'] for e in history]):.1f}")
   print(f"  mean final score   : {np.mean([e['score'] for e in history]):.3f}")
   print(f"  mean reward        : {np.mean([e['reward'] for e in history]):.2f}")
   if by_task:
      print(f"  per-task solve rates:")
      for task, eps in sorted(by_task.items()):
         sr  = np.mean([e["solved"] for e in eps])
         bar = "█" * int(sr * 20)
         print(f"    {task:<25} {sr:5.0%}  {bar}  (n={len(eps)})")
   print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# episode / goal helpers
# ---------------------------------------------------------------------------

def make_episode(goal_encoder, prompt, multi_task, sequential_pool=None):
   """sample a new prompt (if multi_task or sequential), parse, encode."""
   if sequential_pool is not None:
      p = sequential_pool.pop(0)
      sequential_pool.append(p)   # rotate
   elif multi_task:
      p = random.choice(TASK_POOL)
   else:
      p = prompt
   g   = parse_goal(p)
   enc = (encode_goal(goal_encoder, p) if goal_encoder is not None
          else np.zeros(GOAL_ENCODING_DIM, dtype=np.float32))
   return p, g, enc


# ---------------------------------------------------------------------------
# headless diagnostic runner
# ---------------------------------------------------------------------------

def run_headless(model_path: str, prompt: str, multi_task: bool,
                 n_episodes: int, verbose: bool):
   """
   run n_episodes with no GUI and print a performance summary.
   useful for getting solve rates and score distributions outside of training.

   example:
      python demo.py --model models/shape_agent/best_model --headless --episodes 200
   """
   goal_encoder = load_goal_encoder(model_path)
   from stable_baselines3 import PPO
   model = PPO.load(model_path)
   print(f"model loaded: {model_path}")
   print(f"running {n_episodes} episodes (headless)...\n")

   history = []
   for ep in range(1, n_episodes + 1):
      p, g, enc = make_episode(goal_encoder, prompt, multi_task)
      env       = ShapeEnv(goal=g, render_mode=None)
      env.set_goal_encoding(enc)
      obs, _    = env.reset()

      total_reward = 0.0
      steps        = 0
      terminated   = truncated = False

      while not (terminated or truncated):
         action, _ = model.predict(obs, deterministic=True)
         obs, reward, terminated, truncated, _ = env.step(action)
         total_reward += reward
         steps        += 1

      final_score = env._compute_score()
      history.append({
         "episode": ep,
         "prompt":  p,
         "task":    g.get("task", "?"),
         "steps":   steps,
         "reward":  total_reward,
         "score":   final_score,
         "solved":  terminated,
      })
      env.close()

      if verbose or ep % max(n_episodes // 10, 1) == 0:
         status = "SOLVED" if terminated else "timed out"
         print(f"  ep {ep:4d}  {status:<10}  "
               f"steps={steps:4d}  score={final_score:.3f}  {p}")

   dump_episode_summary(history)


# ---------------------------------------------------------------------------
# unified interactive loop
# ---------------------------------------------------------------------------

def run_demo(model_path, prompt, use_random, multi_task,
             use_oracle, sequential):
   """
   single render loop for all interactive agent types (model, oracle, random).
   the only difference is how `action` is computed each step.
   """
   pygame.init()
   window = pygame.display.set_mode((WINDOW_W, WINDOW_H))
   pygame.display.set_caption("shape manipulation — demo")
   clock  = pygame.time.Clock()
   font   = pygame.font.SysFont("monospace", 12)

   # --- load agent ---
   if use_oracle:
      from oracle import OraclePolicy
      from bc_train import GoalEncoder
      goal_encoder = GoalEncoder()
      goal_encoder.eval()
      model        = None
      agent_label  = "ORACLE"
      print("running oracle agent")
   elif use_random:
      goal_encoder = None
      model        = None
      agent_label  = "RANDOM"
      print("running random agent")
   else:
      goal_encoder = load_goal_encoder(model_path)
      try:
         from stable_baselines3 import PPO
         model = PPO.load(model_path)
         print(f"model loaded from {model_path}")
      except Exception as e:
         print(f"could not load model: {e}")
         pygame.quit()
         sys.exit(1)
      agent_label = "MODEL"

   # sequential pool for oracle --sequential mode
   seq_pool = list(TASK_POOL) if (use_oracle and sequential) else None

   # --- first episode ---
   cur_prompt, goal, encoding = make_episode(
      goal_encoder, prompt, multi_task, seq_pool)
   env    = ShapeEnv(goal=goal, render_mode=None)
   env.set_goal_encoding(encoding)
   oracle = ((__import__("oracle").OraclePolicy)(env, noise_std=0.0)
             if use_oracle else None)
   obs, _ = env.reset()

   episode      = 1
   steps        = 0
   total_reward = 0.0
   paused       = False
   step_once    = False   # set by S key — advance exactly one step
   history      = []      # completed episode records for Shift+D

   print(f"\nepisode {episode} — {cur_prompt}")
   print("Q quit  N next  SPC pause  S step  D dump  Shift+D summary\n")

   running = True
   while running:
      skip = False

      for event in pygame.event.get():
         if event.type == pygame.QUIT:
            running = False
         if event.type == pygame.KEYDOWN:
            mods = pygame.key.get_mods()

            if event.key == pygame.K_q:
               running = False

            elif event.key == pygame.K_n:
               skip = True

            elif event.key == pygame.K_SPACE:
               paused = not paused
               print(f"[demo] {'paused' if paused else 'resumed'}  "
                     f"ep {episode}  step {steps}")

            elif event.key == pygame.K_s:
               # pause if not already, then advance one step
               if not paused:
                  paused = True
               step_once = True

            elif event.key == pygame.K_d:
               if mods & pygame.KMOD_SHIFT:
                  dump_episode_summary(history)
               else:
                  extra = None
                  if oracle is not None:
                     extra = (f"oracle: phase={oracle.phase}  "
                              f"committed_shape={oracle.committed_shape}  "
                              f"committed_target={oracle.committed_target}")
                  dump_state(env, steps, episode, extra)

      if not running:
         break

      terminated = truncated = False
      should_step = (not paused) or step_once

      if should_step and not skip:
         if use_oracle:
            action = oracle.act(obs)
         elif model is not None:
            action, _ = model.predict(obs, deterministic=True)
         else:
            action = env.action_space.sample()

         obs, reward, terminated, truncated, _ = env.step(action)
         total_reward += reward
         steps        += 1
         step_once     = False

      phase = oracle.phase if oracle is not None else None
      draw_scene(window, env, font,
                 episode, steps, cur_prompt,
                 agent_label, paused, phase)
      pygame.display.flip()
      clock.tick(FPS)

      if should_step and (terminated or truncated or skip):
         status = "SOLVED" if terminated else ("skipped" if skip else "timed out")
         final_score = env._compute_score()
         # skipped episodes excluded — they reflect user input, not agent performance
         if not skip:
            history.append({
               "episode": episode,
               "prompt":  cur_prompt,
               "task":    goal.get("task", "?"),
               "steps":   steps,
               "reward":  total_reward,
               "score":   final_score,
               "solved":  terminated,
            })
         print(f"episode {episode} {status} — "
               f"steps: {steps}  reward: {total_reward:.2f}  "
               f"score: {final_score:.3f}  task: {cur_prompt}")

         cur_prompt, goal, encoding = make_episode(
            goal_encoder, prompt, multi_task, seq_pool)
         env.goal = goal
         env.set_goal_encoding(encoding)
         oracle   = ((__import__("oracle").OraclePolicy)(env, noise_std=0.0)
                     if use_oracle else None)
         obs, _   = env.reset()

         episode     += 1
         steps        = 0
         total_reward = 0.0
         paused       = False
         print(f"episode {episode} — {cur_prompt}")

   pygame.quit()
   print("demo closed")
   if history:
      dump_episode_summary(history)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
   parser = argparse.ArgumentParser(
      description="run a trained agent or oracle demo")
   parser.add_argument(
      "--model", type=str, default="models/shape_agent/best_model",
      help="path to saved SB3 model (without .zip extension)",
   )
   parser.add_argument(
      "--prompt", type=str,
      default="sort shapes from smallest to largest left to right",
      help="natural language goal prompt",
   )
   parser.add_argument(
      "--random", action="store_true",
      help="use a random agent instead of a trained model",
   )
   parser.add_argument(
      "--multi-task", action="store_true",
      help="sample a random task from TASK_POOL each episode",
   )
   parser.add_argument(
      "--oracle", action="store_true",
      help="watch the oracle instead of a trained model",
   )
   parser.add_argument(
      "--sequential", action="store_true",
      help="cycle through TASK_POOL in order (use with --oracle)",
   )
   parser.add_argument(
      "--headless", action="store_true",
      help="run without GUI and print a performance summary",
   )
   parser.add_argument(
      "--episodes", type=int, default=100,
      help="number of episodes for --headless mode (default: 100)",
   )
   parser.add_argument(
      "--verbose", action="store_true",
      help="print every episode result in --headless mode",
   )
   args = parser.parse_args()

   if args.headless:
      run_headless(
         model_path=args.model,
         prompt=args.prompt,
         multi_task=args.multi_task,
         n_episodes=args.episodes,
         verbose=args.verbose,
      )
   else:
      run_demo(
         model_path=args.model,
         prompt=args.prompt,
         use_random=args.random,
         multi_task=args.multi_task,
         use_oracle=args.oracle,
         sequential=args.sequential,
      )
