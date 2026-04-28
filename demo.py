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
import pygame

from shape_env import ShapeEnv, WINDOW_W, WINDOW_H, FPS, BG_COLOR
from llm_goal_parser import parse_goal, get_embedding


# ---------------------------------------------------------------------------
# model config
# ---------------------------------------------------------------------------

def load_model_config(model_path: str) -> dict:
   """
   load env_config.json from alongside the model.
   returns config dict. falls back to safe defaults with a warning if
   the file isn't found (e.g. for models trained before this was added).

   note: stage checkpoints (stage_00_checkpoint.zip etc.) don't have their
   own config — if you load one, the defaults apply and n_shapes may not
   match what that stage was trained on.
   """
   import json
   config_path = os.path.join(os.path.dirname(model_path), "env_config.json")
   if os.path.exists(config_path):
      with open(config_path) as f:
         config = json.load(f)
      print(f"[demo] env config loaded from {config_path}")
   else:
      print(f"[demo] no env_config.json found at {config_path} — "
            f"defaulting to n_shapes=1, tasks=reach/touch/drag")
      config = {"n_shapes": 1, "tasks": ["reach", "touch", "drag"]}
   return config


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

# overlay timing (in frames at FPS) — shared by SOLVED and "Timed out..."
OVERLAY_FULL_FRAMES = 6    # fully visible at 255 alpha
OVERLAY_FADE_FRAMES = 6    # then fade out over this many frames

SOLVED_COLOR  = (140, 230, 170)
TIMEOUT_COLOR = (220, 90, 80)

def _draw_drag_zone(surface, env):
   """faint shaded rectangle covering the drag target region."""
   from shape_env import REGION_INNER

   if env.goal.get("task") != "drag":
      return
   region = env.goal.get("region", "none")
   if region not in REGION_INNER:
      return

   scale_factor = float(env.goal.get("drag_region_scale", 1.0))
   base = REGION_INNER[region]

   if region == "left":
      boundary = base + (WINDOW_W - base) * (1.0 - scale_factor)
      rect     = pygame.Rect(0, 0, int(boundary), WINDOW_H)
   elif region == "right":
      boundary = base - base * (1.0 - scale_factor)
      rect     = pygame.Rect(int(boundary), 0,
                              WINDOW_W - int(boundary), WINDOW_H)
   elif region == "top":
      boundary = base + (WINDOW_H - base) * (1.0 - scale_factor)
      rect     = pygame.Rect(0, 0, WINDOW_W, int(boundary))
   else:   # bottom
      boundary = base - base * (1.0 - scale_factor)
      rect     = pygame.Rect(0, int(boundary),
                              WINDOW_W, WINDOW_H - int(boundary))

   overlay = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
   overlay.fill((90, 130, 170, 20))
   surface.blit(overlay, rect.topleft)


def _format_prompt(prompt: str) -> str:
   """capitalize first letter, lowercase the rest. preserves the prompt structure."""
   if not prompt:
      return prompt
   p = prompt.strip()
   return p[0].upper() + p[1:].lower()


def _render_definition_list(surface, fonts, x, y, items, label_w, value_w,
                              line_h=18, label_color=(140, 145, 150),
                              value_color=(200, 210, 215)):
   """
   render rows of (label, value) pairs.
   labels left-aligned at x, values right-aligned at x + label_w + value_w.
   used for both bottom status and the H help legend.
   """
   for i, (label, value) in enumerate(items):
      ly = y + i * line_h
      # label, left-aligned
      surface.blit(
         fonts["status"].render(label, True, label_color),
         (x, ly))
      # value, right-aligned within its column
      val_surf = fonts["status"].render(value, True, value_color)
      val_x    = x + label_w + value_w - val_surf.get_width()
      surface.blit(val_surf, (val_x, ly))


def _render_fade_overlay(surface, font, text, color,
                         frames_remaining, fade_frames):
   """
   render `text` centered horizontally near the bottom, fading out over the
   last `fade_frames` of the visible window. no-op when frames_remaining <= 0.
   """
   if frames_remaining <= 0:
      return
   if frames_remaining > fade_frames:
      alpha = 255
   else:
      # fade_progress goes 1.0 -> 0.0 across the fade window
      alpha = int(255 * (frames_remaining / fade_frames))
   text_surf  = font.render(text, True, color)
   text_alpha = pygame.Surface(text_surf.get_size(), pygame.SRCALPHA)
   text_alpha.blit(text_surf, (0, 0))
   text_alpha.set_alpha(alpha)
   tx = (WINDOW_W - text_alpha.get_width()) // 2
   ty = WINDOW_H - text_alpha.get_height() - 80
   surface.blit(text_alpha, (tx, ty))


def draw_scene(surface, env, fonts, episode, steps, prompt,
               agent_label, paused, phase=None, show_help=False,
               solved_fade_remaining=0, timeout_fade_remaining=0):
   """
   layout:
      top-left:    prompt (large) + " (NN%)" inline
      top-right:   "press H for help" hint, expands to definition list
      bottom-left: definition-list status (User/Episode/Score)
      center:      drag zone overlay (drag tasks only), shapes, cursor
      center-low:  large PAUSED overlay if paused
   """
   surface.fill(BG_COLOR)

   # --- center: drag zone, shapes, cursor ---
   _draw_drag_zone(surface, env)

   for shape in env.shapes:
      shape.draw(surface, fonts["status"])

   _prev      = env.window
   env.window = surface
   env._draw_cursor()
   env.window = _prev

   # --- top-left: prompt with inline progress ---
   score = env._compute_score()
   task  = env.goal.get("task", "")

   formatted = _format_prompt(prompt)
   if len(formatted) > 70:
      formatted = formatted[:67] + "..."
   prompt_str = f"{formatted} ({int(score * 100)}%)"

   surface.blit(
      fonts["prompt"].render(prompt_str, True, (220, 230, 220)),
      (16, 14))

   # --- top-right: help hint or expanded legend (definition list style) ---
   
   if show_help:
      help_items = [
         ("Quit",            "Q"),
         ("Pause / resume",  "SPACE"),
         ("Next episode",    "N"),
         ("Step once",       "S"),
         ("Dump state",      "D"),
         ("Episode summary", "SHFT+D"),
         ("Toggle help",     "H"),
      ]
      line_h    = 18
      help_x    = WINDOW_W - 230
      block_h   = line_h * len(help_items)
      help_y    = WINDOW_H - block_h - 16
      _render_definition_list(
         surface, fonts, help_x, help_y, help_items,
         label_w=150, value_w=64, line_h=line_h,
         label_color=(120, 125, 130),
         value_color=(170, 180, 190))
   else:
      hint = "Press H for help"
      hint_surf = fonts["hint"].render(hint, True, (95, 95, 100))
      hint_x = WINDOW_W - hint_surf.get_width() - 16
      hint_y = WINDOW_H - hint_surf.get_height() - 16
      surface.blit(hint_surf, (hint_x, hint_y))

   # --- bottom-left: status as definition list ---
   ep_label  = f"{episode} ({task})" if task else f"{episode}"
   status_items = [
      ("Agent",   agent_label),
      ("Episode", ep_label),
      ("Score",   f"{score:.3f}"),
      ("Steps",   f"{steps}"),
   ]
   line_h    = 20
   block_h   = line_h * len(status_items)
   status_y  = WINDOW_H - block_h - 16
   _render_definition_list(
      surface, fonts, 16, status_y, status_items,
      label_w=80, value_w=60, line_h=line_h,
      label_color=(135, 140, 145),
      value_color=(200, 210, 215))

   # --- centered PAUSED overlay ---
   if paused:
      paused_surf = fonts["prompt"].render("PAUSED", True, (240, 210, 100))
      px = (WINDOW_W - paused_surf.get_width()) // 2
      py = WINDOW_H - paused_surf.get_height() - 80
      surface.blit(paused_surf, (px, py))

   # --- centered fade overlays (tick independently of pause) ---
   # solved wins on tie: timeout overlay is suppressed if both fire
   if solved_fade_remaining > 0:
      _render_fade_overlay(surface, fonts["prompt"], "SOLVED!",
                           SOLVED_COLOR, solved_fade_remaining,
                           OVERLAY_FADE_FRAMES)
   elif timeout_fade_remaining > 0:
      _render_fade_overlay(surface, fonts["prompt"], "Timed out...",
                           TIMEOUT_COLOR, timeout_fade_remaining,
                           OVERLAY_FADE_FRAMES)


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

def make_episode(prompt, multi_task, sequential_pool=None, 
                 task_filter=None, n_shapes=None, trained_tasks=None):
   """sample a new prompt (if multi_task or sequential), parse, embed."""
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
   e = get_embedding(p)

   return p, g, e, n_shapes


# ---------------------------------------------------------------------------
# saliency analysis
# ---------------------------------------------------------------------------


def compute_saliency(model, obs_np: np.ndarray) -> dict:
   """
   compute |d(action)/d(obs)| for each obs dimension.
   returns mean absolute gradient per obs region.

   this shows which parts of the observation the policy is actually
   sensitive to — high gradient = policy is "looking at" that region.
   near-zero gradient on goal_struct means the LLM embedding is
   being ignored despite being in the obs.
   """
   import torch
   from config import OBS_REGIONS
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
   for name, slc, desc in OBS_REGIONS:
      result[name] = {
         "mean_grad": float(grad[slc].mean()),
         "max_grad":  float(grad[slc].max()),
         "desc":      desc,
      }
   return result


def print_saliency(saliency: dict, prompt: str):
   """print saliency table with a bar chart normalised to the max region."""
   max_grad = max(v["mean_grad"] for v in saliency.values()) + 1e-8
   print(f"\n  saliency — {prompt}")
   print(f"  {'region':<16} {'mean |grad|':>11}  bar")
   print(f"  {'-'*52}")
   for name, vals in saliency.items():
      g   = vals["mean_grad"]
      bar = "█" * int(g / max_grad * 30)
      print(f"  {name:<16} {g:>11.5f}  {bar}")
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
      p, g, e, n_shp = make_episode(prompt, multi_task,
                                    task_filter=task_filter,
                                    n_shapes=n_shapes,
                                    trained_tasks=trained_tasks)
      env    = ShapeEnv(goal=g, n_shapes=n_shp, render_mode=None, goal_embedding=e)
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
             use_oracle, sequential, use_human=False,
             show_saliency=False, task_filter=None):
   """
   single render loop for all interactive agent types
   (model, oracle, random, human).
   the only difference is how `action` is computed each step.
   """
   # load config first so n_shapes and task guard are available.
   # oracle and random modes also load config so --task and n_shapes
   # reflect the training run rather than hardcoded defaults.
   if not use_oracle and not use_random:
      config        = load_model_config(model_path)
      n_shapes      = config["n_shapes"]
      trained_tasks = config["tasks"]
      if task_filter is not None and task_filter not in trained_tasks:
         print(f"[demo] warning: --task {task_filter!r} not in trained tasks "
               f"{trained_tasks} — results may be poor")
   else:
      # try to load config for n_shapes / task list — fall back to defaults
      # if no model has been trained yet or path doesn't exist
      try:
         config        = load_model_config(model_path)
         n_shapes      = config["n_shapes"]
         trained_tasks = config["tasks"]
      except Exception:
         n_shapes      = 2
         trained_tasks = None
   
   from prompt_gen import PromptGenerator
   _gen = PromptGenerator()
   pygame.init()
   pygame.key.set_repeat(400, 80)
   if use_human:
      pygame.mouse.set_visible(False)
   window = pygame.display.set_mode((WINDOW_W, WINDOW_H))
   pygame.display.set_caption("dragNdrop demo")
   clock  = pygame.time.Clock()

   # font sizes for different UI roles. match_font picks the first
   # available family from the comma-separated list.
   _font_name = pygame.font.match_font(
      "helveticaneue,helvetica,arial,liberationsans,dejavusans,sans") or None
   fonts = {
      "prompt":   pygame.font.Font(_font_name, 22),
      "metadata": pygame.font.Font(_font_name, 13),
      "status":   pygame.font.Font(_font_name, 14),
      "legend":   pygame.font.Font(_font_name, 11),
      "hint":     pygame.font.Font(_font_name, 12),
   }


   # --- load agent ---
   if use_oracle:
      model       = None
      agent_label = "ORACLE"
      print("running oracle agent")
   elif use_random:
      model        = None
      agent_label  = "RANDOM"
      print("running random agent")
   elif use_human:
      model       = None
      agent_label = "HUMAN"
      print("running in human control mode")
      print("  mouse:          move cursor toward pointer")
      print("  left click:     grip / release")
      print("  Q / N / SPC / D work as normal")
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

   _saliency = show_saliency and (model is not None)
   if show_saliency and model is None:
      print("[demo] --saliency requires a trained model (--model). "
            "saliency is disabled for oracle/random/human agents.")

   # sequential pool for oracle --sequential mode
   seq_pool = list(_gen.training_pool()) if (use_oracle and sequential) else None

   # --- first episode ---
   cur_prompt, goal, emb, n_shapes = make_episode(
      prompt, multi_task, seq_pool, task_filter,
      n_shapes=n_shapes, trained_tasks=trained_tasks)
   env    = ShapeEnv(n_shapes=n_shapes, goal=goal, render_mode=None, goal_embedding=emb)
   oracle = ((__import__("oracle").OraclePolicy)(env, noise_std=0.0)
             if use_oracle else None)
   obs, _ = env.reset()

   episode      = 1
   steps        = 0
   total_reward = 0.0
   paused       = False
   step_once    = False    # set by S key — advance exactly one step
   ep_terminated = False   # true once env.terminated — persists until reset
   ep_truncated  = False   # true once env.truncated — persists until reset
   history      = []       # completed episode records for Shift+D
   show_help = False       # toggled by H key
   solved_fade_remaining  = 0      # frames left to display SOLVED overlay; 0 = hidden
   timeout_fade_remaining = 0      # frames left to display "Timed out..." overlay
   was_solved             = False  # tracks _is_solved() to detect rising edge
   was_truncated          = False  # tracks env.steps >= MAX_STEPS to detect rising edge

   print(f"\nepisode {episode} — {cur_prompt}")
   if use_human:
      print("mouse to move  left-click to grip  "
            "Q quit  N next  SPC pause  D dump  Shift+D summary\n")
   else:
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
            elif event.key == pygame.K_h:
               show_help = not show_help


      if not running:
         break

      should_step = (not paused) or step_once

      # don't step if episode already ended — S should advance to next episode
      terminated = ep_terminated
      truncated  = ep_truncated

      if should_step and not skip:
         if use_human:
            # compute action from mouse position and left button state.
            # dx/dy: unit vector from cursor toward mouse, scaled to [-1,1].
            # dead zone prevents jitter when mouse is close to cursor.
            # grip: left mouse button held down.
            mx, my    = pygame.mouse.get_pos()
            btn       = pygame.mouse.get_pressed()
            ddx       = mx - env.cx
            ddy       = my - env.cy
            dist_m    = float(np.sqrt(ddx**2 + ddy**2))
            if dist_m > 8.0:
               dx_act = float(ddx / dist_m)
               dy_act = float(ddy / dist_m)
            else:
               dx_act = 0.0
               dy_act = 0.0
            grip_act = 1.0 if btn[0] else -1.0
            action   = np.array([dx_act, dy_act, grip_act], dtype=np.float32)
         elif use_oracle:
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
         # auto-print saliency once at step 1 (fresh scene) and every 50 steps
         if _saliency and (steps == 1 or steps % 50 == 0):
            sal = compute_saliency(model, obs)
            print_saliency(sal, cur_prompt)
            print(f"reward: {reward}")

      step_once = False   # always clear after this frame
      phase = oracle.phase if oracle is not None else None

      # overlay rising-edge detection. solved wins on tie: if both fire on
      # the same frame, we trigger SOLVED and skip TIMEOUT.
      is_solved_now    = env._is_solved()
      is_truncated_now = ep_truncated
      if is_solved_now and not was_solved:
         solved_fade_remaining = OVERLAY_FULL_FRAMES + OVERLAY_FADE_FRAMES
      elif is_truncated_now and not was_truncated and not is_solved_now:
         timeout_fade_remaining = OVERLAY_FULL_FRAMES + OVERLAY_FADE_FRAMES
      was_solved    = is_solved_now
      was_truncated = is_truncated_now

      # tick down every frame regardless of pause / episode reset
      if solved_fade_remaining > 0:
         solved_fade_remaining -= 1
      if timeout_fade_remaining > 0:
         timeout_fade_remaining -= 1

      draw_scene(window, env, fonts,
                 episode=episode, steps=steps,
                 prompt=cur_prompt, agent_label=agent_label,
                 paused=paused, phase=phase,
                 show_help=show_help,
                 solved_fade_remaining=solved_fade_remaining,
                 timeout_fade_remaining=timeout_fade_remaining)
      pygame.display.flip()
      clock.tick(FPS)

      # advance to next episode if: an unpaused step ended the episode,
      # the user pressed N (skip), or the episode is already finished
      # (e.g. solved on a paused S-step — advance anyway so the SOLVED
      # overlay can play out over the new episode's scene)
      episode_done = ((not paused and (terminated or truncated))
                      or skip
                      or ep_terminated or ep_truncated)
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
         
         stay_paused = paused
         cur_prompt, goal, emb, _ = make_episode(
            prompt, multi_task, seq_pool, task_filter,
            n_shapes=n_shapes, trained_tasks=trained_tasks)
         env.goal = goal
         env._goal_embedding = emb
         oracle   = ((__import__("oracle").OraclePolicy)(env, noise_std=0.0)
                     if use_oracle else None)
         obs, _   = env.reset()

         episode     += 1
         steps        = 0
         total_reward = 0.0
         ep_terminated  = False
         ep_truncated   = False
         paused         = stay_paused
         was_solved     = False  # new episode starts unsolved; let next solve re-trigger
         was_truncated  = False  # new episode starts un-truncated; let next timeout re-trigger
         print(f"episode {episode} — {cur_prompt}")

   pygame.mouse.set_visible(True)
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
      "--human", action="store_true",
      help="control the cursor yourself with mouse (left click = grip)",
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
         task_filter=args.task,
      )
   else:
      run_demo(
         model_path=args.model,
         prompt=args.prompt,
         use_random=args.random,
         use_human=args.human,
         multi_task=args.multi_task,
         use_oracle=args.oracle,
         sequential=args.sequential,
         show_saliency=args.saliency,
         task_filter=args.task,
      )
