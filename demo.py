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
from config import GOAL_ENCODING_DIM


# ---------------------------------------------------------------------------
# goal encoder helpers
# ---------------------------------------------------------------------------

def load_model_config(model_path: str) -> dict:
   """
   load training_config.json from alongside the model.
   returns config dict. falls back to safe defaults with a warning if
   the file isn't found (e.g. for models trained before this was added).

   note: stage checkpoints (stage_00_checkpoint.zip etc.) don't have their
   own config — if you load one, the defaults apply and n_shapes may not
   match what that stage was trained on.
   """
   import json
   from bc_train import GoalEncoder
   config_path = os.path.join(os.path.dirname(model_path), "training_config.json")
   if os.path.exists(config_path):
      with open(config_path) as f:
         config = json.load(f)
      print(f"[demo] training config loaded from {config_path}")
   else:
      print(f"[demo] no training_config.json found at {config_path} — "
            f"defaulting to n_shapes=1, tasks=reach/touch/drag")
      config = {"n_shapes": 1, "tasks": ["reach", "touch", "drag"]}
   # goal encoder is always fixed seed 42 — no file needed
   encoder = GoalEncoder()
   encoder.eval()
   config["goal_encoder"] = encoder
   return config

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
   "SPC    pause / unpause",
   "N      next episode",
   "S      step",
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
   tc   = env.goal.get("target_color", "none")
   tt   = env.goal.get("target_type",  "none")
   if task in ("reach", "touch", "drag") and env.shapes:
      has_any_spec = (tc not in ("none", "any")) or (tt not in ("none", "any"))
      matching     = env._matching_shape_indices()
      if has_any_spec:
         for i in matching:
            t = env.shapes[i]
            pygame.draw.circle(surface, (255, 220, 60),
                               (int(t.x), int(t.y)), int(t.radius) + 6, 2)
      else:
         # all shapes valid — show dim ring on all
         for t in env.shapes:
            pygame.draw.circle(surface, (120, 120, 80),
                               (int(t.x), int(t.y)), int(t.radius) + 6, 1)

   # draw cursor via env's own method
   _prev      = env.window
   env.window = surface
   env._draw_cursor()
   env.window = _prev

   # top-left HUD: task info
   score   = env._compute_score()
   hud_str = f"task: {task}   progress: {score:.2%}"
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

def dump_state(env, step, episode, prompt=None, extra=None):
   """print current env state — bound to D key."""
   score = env._compute_score()
   print(f"\n--- state dump  ep {episode}  step {step} ---")
   print(f"  task    : {env.goal.get('task')}  score={score:.4f}")
   if prompt:
      print(f"  prompt  : {prompt}")
   if extra:
      print(f"  {extra}")
   print(f"  cursor  : ({env.cx:.1f}, {env.cy:.1f})  "
         f"holding={env.holding}  grabbed_idx={env.grabbed_idx}")
   task = env.goal.get("task", "")
   # parse oracle's committed_shape from extra string if available
   oracle_target = None
   if extra and "committed_shape=" in extra:
      try:
         cs = extra.split("committed_shape=")[1].split()[0].rstrip(",")
         oracle_target = int(cs) if cs.lstrip("-").isdigit() else None
      except Exception:
         pass
   for i, s in enumerate(env.shapes):
      if task in ("reach", "touch", "drag"):
         marker = " <-- target" if i in env.target_indices else ""
      else:
         # arrangement: show which shape oracle is currently working on
         if oracle_target is not None and oracle_target >= 0:
            marker = " <-- oracle" if i == oracle_target else ""
         else:
            marker = ""
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

def make_episode(goal_encoder, prompt, multi_task, sequential_pool=None, 
                 task_filter=None, n_shapes=None, trained_tasks=None):
   """sample a new prompt (if multi_task or sequential), parse, encode."""
   if sequential_pool is not None:
      p = sequential_pool.pop(0)
      sequential_pool.append(p)
   elif multi_task or task_filter or prompt is None:
      from prompt_gen import sample_prompt
      # task_filter overrides, otherwise sample from trained tasks if known
      task = task_filter or (random.choice(trained_tasks) if trained_tasks else None)
      p = sample_prompt(task)
   else:
      p = prompt
   g   = parse_goal(p)
   enc = (encode_goal(goal_encoder, p) if goal_encoder is not None
          else np.zeros(GOAL_ENCODING_DIM, dtype=np.float32))
   return p, g, enc, n_shapes


# ---------------------------------------------------------------------------
# saliency analysis
# ---------------------------------------------------------------------------

# obs region labels for the 108-dim vector
_OBS_REGIONS = [
   ("cursor_state",   slice(0,   4),  "cx cy holding grabbed_idx"),
   ("grabbed_shape",  slice(4,   9),  "grabbed shape features"),
   ("nearest_shape",  slice(9,   14), "nearest shape features"),
   ("all_shapes",     slice(14,  44), "all 6 shapes (zero-padded)"),
   ("goal_struct",    slice(44,  76), "structured goal one-hots"),
   ("goal_semantic",  slice(76,  108),"MiniLM semantic projection"),
]


def compute_saliency(model, obs_np: np.ndarray) -> dict:
   """
   compute |d(action)/d(obs)| for each obs dimension.
   returns mean absolute gradient per obs region.

   this shows which parts of the observation the policy is actually
   sensitive to — high gradient = policy is "looking at" that region.
   near-zero gradient on goal_struct means the structured encoding
   is being ignored despite being in the obs.
   """
   import torch
   obs_t = torch.tensor(obs_np, dtype=torch.float32).unsqueeze(0)
   obs_t.requires_grad_(True)

   # use the policy's forward pass directly
   policy = model.policy
   with torch.enable_grad():
      actions, _, _ = policy.forward(obs_t)
      # sum all action dims so we get a scalar to differentiate
      actions.sum().backward()

   grad = obs_t.grad.squeeze(0).abs().detach().numpy()

   result = {}
   for name, slc, desc in _OBS_REGIONS:
      result[name] = {
         "mean_grad": float(grad[slc].mean()),
         "max_grad":  float(grad[slc].max()),
         "desc":      desc,
      }
   return result


def print_saliency(saliency: dict, prompt: str, detail: bool = False):
   """
   print saliency table with a bar chart normalised to the max region.
   if detail=True, also show per-field breakdown within goal_struct.
   """
   from config import (SUPPORTED_TASKS, COLOR_NAMES_GOAL, SHAPE_TYPES)

   max_grad = max(v["mean_grad"] for k, v in saliency.items() if k != "goal_struct_raw") + 1e-8
   print(f"\n  saliency — {prompt}")
   print(f"  {'region':<16} {'mean |grad|':>11}  bar")
   print(f"  {'-'*52}")
   for name, vals in saliency.items():
      if name == "goal_struct_raw": continue
      g   = vals["mean_grad"]
      bar = "█" * int(g / max_grad * 30)
      print(f"  {name:<16} {g:>11.5f}  {bar}")

   if detail and "goal_struct_raw" in saliency:
      # break down goal_struct by field
      raw = saliency["goal_struct_raw"]   # per-dim gradients (32,)
      fields = [
         ("task",      SUPPORTED_TASKS,   7),
         ("color",     COLOR_NAMES_GOAL + ["none"], 6),
         ("type",      SHAPE_TYPES + ["none"],       4),
         ("region",    ["left","right","top","bottom","none"], 5),
         ("axis",      ["x","y","none"],              3),
         ("direction", ["asc","desc","none"],          3),
         ("attribute", ["size","color","none"],        3),
         ("bounded",   ["bounded"],                    1),
      ]
      print(f"\n  goal_struct field breakdown:")
      offset = 0
      for fname, labels, ndim in fields:
         vals_f = raw[offset:offset+ndim]
         peak_i = int(vals_f.argmax())
         peak_v = float(vals_f[peak_i])
         bar    = "█" * int(peak_v / (max_grad + 1e-8) * 30)
         label  = labels[peak_i] if peak_i < len(labels) else str(peak_i)
         print(f"    {fname:<12} peak={label:<12} {peak_v:.5f}  {bar}")
         offset += ndim
   print()

# ---------------------------------------------------------------------------
# headless diagnostic runner
# ---------------------------------------------------------------------------

def run_headless(model_path: str, prompt: str, multi_task: bool,
                 n_episodes: int, verbose: bool,
                 task_filter: str = None):
   """
   run n_episodes with no GUI and print a performance summary.
   useful for getting solve rates and score distributions outside of training.

   example:
      python demo.py --model models/shape_agent/best_model --headless --episodes 200
   """
   config       = load_model_config(model_path)
   goal_encoder = config["goal_encoder"]
   n_shapes     = config["n_shapes"]
   trained_tasks = config["tasks"]

   if task_filter is not None and task_filter not in trained_tasks:
      print(f"[demo] warning: --task {task_filter!r} not in trained tasks "
            f"{trained_tasks} — results may be poor")
   
   from stable_baselines3 import PPO
   model = PPO.load(model_path)
   print(f"model loaded: {model_path}")
   print(f"running {n_episodes} episodes (headless)  "
         f"n_shapes={n_shapes}  tasks={trained_tasks}\n")

   history = []
   for ep in range(1, n_episodes + 1):
      p, g, enc, n_shp = make_episode(goal_encoder, prompt, multi_task,
                                      task_filter=task_filter,
                                      n_shapes=n_shapes,
                                      trained_tasks=trained_tasks)
      env    = ShapeEnv(goal=g, n_shapes=n_shp, render_mode=None)
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
             use_oracle, sequential,
             show_saliency=False, task_filter=None):
   """
   single render loop for all interactive agent types (model, oracle, random).
   the only difference is how `action` is computed each step.
   """
   # load config first so n_shapes and task guard are available
   # oracle/random paths still construct GoalEncoder the same way
   if not use_oracle and not use_random:
      config        = load_model_config(model_path)
      goal_encoder  = config["goal_encoder"]
      n_shapes      = config["n_shapes"]
      trained_tasks = config["tasks"]
      if task_filter is not None and task_filter not in trained_tasks:
         print(f"[demo] warning: --task {task_filter!r} not in trained tasks "
               f"{trained_tasks} — results may be poor")
   else:
      from bc_train import GoalEncoder
      goal_encoder  = GoalEncoder()
      goal_encoder.eval()
      n_shapes      = 1   # oracle/random default
      trained_tasks = None
   
   from prompt_gen import PromptGenerator
   _gen = PromptGenerator()
   pygame.init()
   pygame.key.set_repeat(400, 80)
   window = pygame.display.set_mode((WINDOW_W, WINDOW_H))
   pygame.display.set_caption("dragNdrop demo")
   clock  = pygame.time.Clock()
   font   = pygame.font.SysFont("monospace", 12)

   # --- load agent ---
   if use_oracle:
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
      try:
         from stable_baselines3 import PPO
         model = PPO.load(model_path)
         print(f"model loaded from {model_path}")
      except Exception as e:
         print(f"could not load model: {e}")
         pygame.quit()
         sys.exit(1)
      agent_label = "MODEL"

   _saliency        = show_saliency and (model is not None)
   if show_saliency and model is None:
      print("[demo] --saliency requires a trained model (--model). "
            "saliency is disabled for oracle/random agents.")

   # sequential pool for oracle --sequential mode
   seq_pool = list(_gen.training_pool()) if (use_oracle and sequential) else None

   # --- first episode ---
   cur_prompt, goal, encoding, n_shapes = make_episode(
      goal_encoder, prompt, multi_task, seq_pool, task_filter,
      n_shapes=n_shapes, trained_tasks=trained_tasks)
   env    = ShapeEnv(n_shapes=n_shapes, goal=goal, render_mode=None)
   env.set_goal_encoding(encoding)
   oracle = ((__import__("oracle").OraclePolicy)(env, noise_std=0.0)
             if use_oracle else None)
   obs, _ = env.reset()

   episode      = 1
   steps        = 0
   total_reward = 0.0
   paused       = False
   step_once    = False   # set by S key — advance exactly one step
   ep_terminated  = False   # true once env.terminated — persists until reset
   ep_truncated   = False   # true once env.truncated — persists until reset
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
                  dump_state(env, steps, episode, cur_prompt, extra)
                  if _saliency:
                     sal = compute_saliency(model, obs)
                     print_saliency(sal, cur_prompt)

      if not running:
         break

      should_step = (not paused) or step_once

      # don't step if episode already ended — S should advance to next episode
      terminated = ep_terminated
      truncated  = ep_truncated

      if should_step and not skip:
         if use_oracle:
            action = oracle.act(obs)
         elif model is not None:
            action, _ = model.predict(obs, deterministic=True)
         else:
            action = env.action_space.sample()

         obs, reward, terminated, truncated, _ = env.step(action)
         total_reward  += reward
         steps         += 1
         ep_terminated  = terminated
         ep_truncated   = truncated
         # auto-print saliency once at step 1 (fresh scene) and every 100 steps
         if _saliency and (steps == 1 or steps % 100 == 0):
            sal = compute_saliency(model, obs)
            print_saliency(sal, cur_prompt)

      step_once = False   # always clear after this frame
      phase = oracle.phase if oracle is not None else None
      draw_scene(window, env, font,
                 episode, steps, cur_prompt,
                 agent_label, paused, phase)
      pygame.display.flip()
      clock.tick(FPS)

      episode_done = ((not paused and (terminated or truncated))
                      or skip
                      or (ep_terminated or ep_truncated))
      if episode_done:
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
         
         stay_paused = paused or skip
         cur_prompt, goal, encoding, _ = make_episode(
            goal_encoder, prompt, multi_task, seq_pool, task_filter,
            n_shapes=n_shapes, trained_tasks=trained_tasks)
         env.goal = goal
         env.set_goal_encoding(encoding)
         oracle   = ((__import__("oracle").OraclePolicy)(env, noise_std=0.0)
                     if use_oracle else None)
         obs, _   = env.reset()

         episode     += 1
         steps        = 0
         total_reward = 0.0
         ep_terminated  = False
         ep_truncated   = False
         paused         = stay_paused
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
      "--prompt", type=str, default=None,
      help="natural language goal prompt. if omitted, samples from trained tasks",
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
   parser.add_argument(
      "--task", type=str, default=None,
      help=(
         "filter episodes to a specific task (e.g. reach, touch, drag, "
         "arrange_in_region). uses prompt_gen to sample varied prompts for "
         "that task. overrides --prompt when set."
      ),
   )
   parser.add_argument(
      "--saliency", action="store_true",
      help=(
         "show obs gradient saliency. in --headless: printed after every episode. "
         "in interactive: press D to print for current obs."
      ),
   )
   args = parser.parse_args()

   if args.headless:
      run_headless(
         model_path=args.model,
         prompt=args.prompt,
         multi_task=args.multi_task,
         n_episodes=args.episodes,
         verbose=args.verbose,
         saliency=args.saliency,
         task_filter=args.task,
      )
   else:
      run_demo(
         model_path=args.model,
         prompt=args.prompt,
         use_random=args.random,
         multi_task=args.multi_task,
         use_oracle=args.oracle,
         sequential=args.sequential,
         show_saliency=args.saliency,
         task_filter=args.task,
      )
