"""
demo_reach.py  —  isolated reach-task diagnostic demo.

loads a trained model (or falls back to oracle / random) and runs it
exclusively on the REACH task so you can watch cursor behaviour in
isolation. all other tasks are suppressed.

usage:
   python demo_reach.py                                           # oracle (no model needed)
   python demo_reach.py --model models/shape_agent/best_model    # trained model
   python demo_reach.py --random                                  # random baseline
   python demo_reach.py --model ... --headless --episodes 200    # no GUI, print stats

keybinds (interactive mode):
   Q        quit
   N        skip to next episode
   SPACE    pause / unpause
   S        step once (auto-pauses)
   D        dump env state to console
   R        reset episode immediately

HUD columns (top bar):
   dist_px   cursor distance to target in pixels
   score     _score_reach() in [0, 1]  (1.0 = inside GRIP_RADIUS)
   holding   whether grip is active
   phase     oracle sub-phase (NAVIGATE / GRIP_ON etc.), model shows '—'
"""

import argparse
import sys
import numpy as np
import torch
import pygame

from shape_env import ShapeEnv, WINDOW_W, WINDOW_H, FPS, BG_COLOR, GRIP_RADIUS
from llm_goal_parser import parse_goal, get_embedding
from config import GOAL_ENCODING_DIM


# ---------------------------------------------------------------------------
# reach task prompt — fixed for this demo
# ---------------------------------------------------------------------------

_REACH_PROMPT  = "move the cursor to the shape"
_REACH_GOAL    = parse_goal(_REACH_PROMPT)

# sanity-check: make sure parse_goal resolves this to reach
assert _REACH_GOAL["task"] == "reach", (
   f"Goal parser returned '{_REACH_GOAL['task']}' for reach prompt — "
   f"check llm_goal_parser.py")


# ---------------------------------------------------------------------------
# legend
# ---------------------------------------------------------------------------

_LEGEND = [
   "Q      quit",
   "N      next episode",
   "SPC    pause / unpause",
   "S      step (one step)",
   "D      dump state",
   "R      reset episode",
]


# ---------------------------------------------------------------------------
# goal encoder helpers
# ---------------------------------------------------------------------------

def _load_goal_encoder(model_path: str = None):
   """
   load GoalEncoder from alongside the model if path given, else fresh init.
   """
   import os
   from bc_train import GoalEncoder
   encoder = GoalEncoder()
   if model_path is not None:
      enc_path = os.path.join(os.path.dirname(model_path), "goal_encoder.pt")
      if os.path.exists(enc_path):
         encoder.load_state_dict(torch.load(enc_path, map_location="cpu"))
         print(f"[goal encoder] loaded from {enc_path}")
      else:
         print(f"[goal encoder] not found at {enc_path} — using random init")
   encoder.eval()
   return encoder


def _encode(goal_encoder, prompt: str) -> np.ndarray:
   raw = get_embedding(prompt)
   with torch.no_grad():
      t = torch.tensor(raw, dtype=torch.float32).unsqueeze(0)
      return goal_encoder(t).squeeze(0).numpy()


# ---------------------------------------------------------------------------
# env factory — reach only, always 1 shape
# ---------------------------------------------------------------------------

def _make_env(goal_encoder) -> ShapeEnv:
   enc = _encode(goal_encoder, _REACH_PROMPT)
   env = ShapeEnv(n_shapes=1, goal=_REACH_GOAL, render_mode=None)
   env.set_goal_encoding(enc)
   return env


# ---------------------------------------------------------------------------
# HUD draw
# ---------------------------------------------------------------------------

def _draw(surface, env, font, episode, steps, agent_label,
          paused, phase=None, history=None):
   surface.fill(BG_COLOR)

   # draw the single shape
   for shape in env.shapes:
      shape.draw(surface, font)

   # highlight target with a bright ring
   if env.shapes:
      t = env.shapes[env.target_idx]
      pygame.draw.circle(
         surface, (255, 220, 60),
         (int(t.x), int(t.y)), int(t.radius) + 8, 2)
      # GRIP_RADIUS indicator — dashed circle approximated with arcs
      pygame.draw.circle(
         surface, (100, 180, 255),
         (int(t.x), int(t.y)), GRIP_RADIUS, 1)

   # draw cursor
   _prev      = env.window
   env.window = surface
   env._draw_cursor()
   env.window = _prev

   # metrics
   score = env._compute_score()
   dist  = 0.0
   if env.shapes:
      t    = env.shapes[env.target_idx]
      dist = float(np.sqrt((env.cx - t.x) ** 2 + (env.cy - t.y) ** 2))

   # top HUD
   hud = (f"REACH DEMO  |  ep {episode}  step {steps}  |  "
          f"dist_px: {dist:6.1f}  score: {score:.3f}  "
          f"holding: {env.holding}  "
          f"phase: {phase if phase else '—'}")
   col = (255, 220, 60) if score >= 0.85 else (200, 200, 200)
   surface.blit(font.render(hud, True, col), (10, 10))

   # legend top-right
   for i, line in enumerate(_LEGEND):
      surface.blit(font.render(line, True, (140, 140, 140)),
                   (WINDOW_W - 190, 28 + i * 14))

   # agent label bottom-left
   pause_str = "  [PAUSED]" if paused else ""
   solved    = "  ★ SOLVED" if env._is_solved() else ""
   bottom    = f"{agent_label}{pause_str}{solved}"
   surface.blit(font.render(bottom, True, (100, 220, 180)), (10, WINDOW_H - 24))

   # recent solve rate (last 20 episodes)
   if history:
      recent  = history[-20:]
      sr      = np.mean([e["solved"] for e in recent])
      sr_str  = f"last {len(recent)} ep solve rate: {sr:.0%}"
      surface.blit(font.render(sr_str, True, (160, 200, 160)),
                   (10, WINDOW_H - 40))


# ---------------------------------------------------------------------------
# interactive loop
# ---------------------------------------------------------------------------

def run_interactive(model_path, use_oracle, use_random):
   pygame.init()
   window = pygame.display.set_mode((WINDOW_W, WINDOW_H))
   pygame.display.set_caption("reach diagnostic demo")
   clock  = pygame.time.Clock()
   font   = pygame.font.SysFont("monospace", 12)

   # --- load agent ---
   if use_oracle:
      from oracle import OraclePolicy
      goal_encoder = _load_goal_encoder()
      model        = None
      agent_label  = "ORACLE"
      print("[demo_reach] running oracle agent on reach task")
   elif use_random:
      goal_encoder = _load_goal_encoder()
      model        = None
      agent_label  = "RANDOM"
      print("[demo_reach] running random agent on reach task")
   else:
      goal_encoder = _load_goal_encoder(model_path)
      try:
         from stable_baselines3 import PPO
         model = PPO.load(model_path)
         print(f"[demo_reach] model loaded from {model_path}")
      except Exception as e:
         print(f"[demo_reach] could not load model: {e}")
         pygame.quit()
         sys.exit(1)
      agent_label = "MODEL"

   env    = _make_env(goal_encoder)
   oracle = None
   if use_oracle:
      from oracle import OraclePolicy
      oracle = OraclePolicy(env, noise_std=0.0)

   obs, _ = env.reset()
   if oracle:
      oracle.reset()

   episode      = 1
   steps        = 0
   paused       = False
   step_once    = False
   history      = []

   print(f"\nepisode {episode}  —  reach  (1 shape, move cursor into blue circle)")
   running = True
   while running:
      for event in pygame.event.get():
         if event.type == pygame.QUIT:
            running = False
         if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
               running = False
            elif event.key == pygame.K_n:
               # skip to next episode without scoring it
               obs, _ = env.reset()
               if oracle:
                  oracle.reset()
               steps = 0
               paused = False
               episode += 1
               print(f"episode {episode}  (skipped)")
            elif event.key == pygame.K_r:
               obs, _ = env.reset()
               if oracle:
                  oracle.reset()
               steps = 0
               print(f"episode {episode}  reset")
            elif event.key == pygame.K_SPACE:
               paused = not paused
            elif event.key == pygame.K_s:
               paused    = True
               step_once = True
            elif event.key == pygame.K_d:
               score = env._compute_score()
               t     = env.shapes[env.target_idx]
               dist  = np.sqrt((env.cx - t.x) ** 2 + (env.cy - t.y) ** 2)
               print(f"\n--- state dump  ep {episode}  step {steps} ---")
               print(f"  score    : {score:.4f}")
               print(f"  dist_px  : {dist:.1f}  (GRIP_RADIUS={GRIP_RADIUS})")
               print(f"  cursor   : ({env.cx:.1f}, {env.cy:.1f})")
               print(f"  holding  : {env.holding}  grabbed_idx={env.grabbed_idx}")
               print(f"  target   : ({t.x:.1f}, {t.y:.1f})  idx={env.target_idx}")
               if oracle:
                  print(f"  oracle   : phase={oracle.phase}  "
                        f"committed_shape={oracle.committed_shape}")
               print()

      if not running:
         break

      should_step = (not paused) or step_once
      terminated = truncated = False

      if should_step:
         if use_oracle and oracle:
            action = oracle.act(obs)
         elif model is not None:
            action, _ = model.predict(obs, deterministic=True)
         else:
            action = env.action_space.sample()

         obs, _, terminated, truncated, _ = env.step(action)
         steps     += 1
         step_once  = False

      phase = oracle.phase if oracle else None
      _draw(window, env, font, episode, steps, agent_label,
            paused, phase, history)
      pygame.display.flip()
      clock.tick(FPS)

      if should_step and (terminated or truncated):
         solved = terminated
         history.append({"episode": episode, "steps": steps, "solved": solved})
         status = "SOLVED" if solved else "timed out"
         sr_20  = np.mean([e["solved"] for e in history[-20:]]) if history else 0.0
         print(f"ep {episode:4d}  {status:<10}  steps={steps:4d}  "
               f"score={env._compute_score():.3f}  "
               f"last-20 sr={sr_20:.0%}")

         obs, _ = env.reset()
         if oracle:
            oracle.reset()
         episode += 1
         steps    = 0

   pygame.quit()
   _print_summary(history)


# ---------------------------------------------------------------------------
# headless mode
# ---------------------------------------------------------------------------

def run_headless(model_path, use_oracle, use_random, n_episodes, verbose):
   print(f"[demo_reach] headless — {n_episodes} episodes, reach task only\n")
   goal_encoder = _load_goal_encoder(model_path if not use_oracle else None)

   model = None
   if not use_oracle and not use_random:
      from stable_baselines3 import PPO
      model = PPO.load(model_path)
      print(f"model loaded: {model_path}")

   oracle_cls = None
   if use_oracle:
      from oracle import OraclePolicy
      oracle_cls = OraclePolicy

   history = []
   for ep in range(1, n_episodes + 1):
      env    = _make_env(goal_encoder)
      oracle = oracle_cls(env, noise_std=0.0) if oracle_cls else None
      obs, _ = env.reset()
      if oracle:
         oracle.reset()

      steps = 0
      terminated = truncated = False
      while not (terminated or truncated):
         if oracle:
            action = oracle.act(obs)
         elif model:
            action, _ = model.predict(obs, deterministic=True)
         else:
            action = env.action_space.sample()
         obs, _, terminated, truncated, _ = env.step(action)
         steps += 1

      history.append({"episode": ep, "steps": steps, "solved": terminated,
                      "score": env._compute_score()})
      env.close()

      if verbose or ep % max(n_episodes // 10, 1) == 0:
         status = "SOLVED" if terminated else "timed out"
         print(f"  ep {ep:4d}  {status:<10}  "
               f"steps={steps:4d}  score={env._compute_score():.3f}")

   _print_summary(history)


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

def _print_summary(history):
   if not history:
      print("\n[no episodes completed]")
      return
   n      = len(history)
   solved = [e for e in history if e["solved"]]
   sr     = len(solved) / n
   print(f"\n{'='*50}")
   print(f"  REACH DEMO SUMMARY  ({n} episodes)")
   print(f"{'='*50}")
   print(f"  solve rate    : {sr:.0%}  ({len(solved)}/{n})")
   print(f"  mean steps    : {np.mean([e['steps'] for e in history]):.1f}")
   if "score" in history[0]:
      print(f"  mean score    : {np.mean([e['score'] for e in history]):.3f}")
   # step distribution of solved episodes
   if solved:
      print(f"  solved steps  : "
            f"min={min(e['steps'] for e in solved)}  "
            f"mean={np.mean([e['steps'] for e in solved]):.1f}  "
            f"max={max(e['steps'] for e in solved)}")
   print(f"{'='*50}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
   parser = argparse.ArgumentParser(
      description=(
         "reach-task diagnostic demo. "
         "runs the agent exclusively on the reach task with one shape. "
         "usage: python demo_reach.py [--model PATH] [--oracle] [--random]"
      )
   )
   parser.add_argument(
      "--model", type=str, default="models/shape_agent/best_model",
      help="path to saved SB3 model (without .zip). ignored with --oracle/--random.",
   )
   parser.add_argument(
      "--oracle", action="store_true",
      help="use oracle instead of a trained model (no model file needed).",
   )
   parser.add_argument(
      "--random", action="store_true",
      help="use a random agent as a sanity baseline.",
   )
   parser.add_argument(
      "--headless", action="store_true",
      help="run without pygame window and print a stats summary.",
   )
   parser.add_argument(
      "--episodes", type=int, default=100,
      help="number of episodes for --headless mode (default: 100).",
   )
   parser.add_argument(
      "--verbose", action="store_true",
      help="print every episode result in --headless mode.",
   )
   args = parser.parse_args()

   if args.headless:
      run_headless(
         model_path=args.model,
         use_oracle=args.oracle,
         use_random=args.random,
         n_episodes=args.episodes,
         verbose=args.verbose,
      )
   else:
      run_interactive(
         model_path=args.model,
         use_oracle=args.oracle,
         use_random=args.random,
      )