"""
eval_truthful.py

measure solve rates on the actual training distribution:
   - cursor persists across episodes (matches training and design intent)
   - prompt resamples across episodes (matches eval intent)

this is what the agent actually does in deployment. no distribution gap,
no inflated numbers.

usage:
   python eval_truthful.py --model models/shape_agent/best_model
   python eval_truthful.py --model models/shape_agent/best_model --episodes 200
"""

import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())

from shape_env import ShapeEnv
from llm_goal_parser import parse_goal, get_embedding
from prompt_gen import sample_prompt


def wilson_ci(k, n):
   if n == 0: return (0.0, 0.0)
   z = 1.96
   p = k / n
   denom = 1 + z**2 / n
   center = (p + z**2 / (2 * n)) / denom
   half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
   return max(0.0, center - half), min(1.0, center + half)


def run_truthful(model, task: str, n: int, verbose: bool = False) -> dict:
   """reused env + resampled prompts each episode (matches demo.run_demo)."""
   prompt = sample_prompt(task)
   goal   = parse_goal(prompt)
   emb    = get_embedding(prompt)
   env    = ShapeEnv(n_shapes=1, goal=goal, render_mode=None,
                     goal_embedding=emb)
   obs, _ = env.reset()

   solved = 0
   ep_steps = []
   per_region_drag = {}   # for drag task: track per-region solve rates

   for ep in range(n):
      term = trunc = False
      steps = 0
      while not (term or trunc):
         action, _ = model.predict(obs, deterministic=True)
         obs, _, term, trunc, _ = env.step(action)
         steps += 1

      if term:
         solved += 1
      ep_steps.append(steps)

      # track per-region for drag
      if task == "drag":
         region = goal.get("region", "none")
         per_region_drag.setdefault(region, [0, 0])
         per_region_drag[region][0] += int(term)
         per_region_drag[region][1] += 1

      if verbose:
         status = "SOLVED" if term else "  miss"
         extra = f" region={goal.get('region', '-')}" if task == "drag" else ""
         print(f"  ep {ep:3d}  {status}  steps={steps:3d}{extra}  {prompt}")

      # re-sample prompt for next episode (mirrors run_demo exactly)
      prompt = sample_prompt(task)
      goal   = parse_goal(prompt)
      emb    = get_embedding(prompt)
      env.goal = goal
      env._goal_embedding = emb
      obs, _ = env.reset()

   env.close()
   lo, hi = wilson_ci(solved, n)
   return {
      "task":       task,
      "n":          n,
      "solved":     solved,
      "solve_rate": solved / n,
      "ci_low":     lo,
      "ci_high":    hi,
      "mean_steps": float(np.mean(ep_steps)),
      "per_region": per_region_drag,
   }


if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument("--model",    type=str, required=True)
   parser.add_argument("--tasks",    type=str, nargs="+",
                       default=["reach", "touch", "drag"])
   parser.add_argument("--episodes", type=int, default=150)
   parser.add_argument("--verbose",  action="store_true")
   args = parser.parse_args()

   from stable_baselines3 import PPO
   model = PPO.load(args.model)
   print(f"model loaded: {args.model}")
   print(f"truthful eval (reused env, resampled prompts)")
   print(f"tasks={args.tasks}  n={args.episodes} per task\n")

   results = []
   for task in args.tasks:
      print(f"--- {task} ---")
      r = run_truthful(model, task, args.episodes, verbose=args.verbose)
      results.append(r)

   print()
   print("=" * 72)
   print(f"  {'task':<10} {'n':>5}  {'solved':>7}  {'rate':>7}  {'95% CI':>14}  {'steps':>7}")
   print("-" * 72)
   for r in results:
      ci = f"[{r['ci_low']:.0%}, {r['ci_high']:.0%}]"
      print(f"  {r['task']:<10} {r['n']:>5}  {r['solved']:>7}  "
            f"{r['solve_rate']:>6.1%}  {ci:>14}  {r['mean_steps']:>7.1f}")
   print("=" * 72)

   # drag per-region breakdown — reveals the bottom-right mode collapse
   for r in results:
      if r["task"] == "drag" and r["per_region"]:
         print("\n  drag per-region solve rates:")
         for region, (k, n) in sorted(r["per_region"].items()):
            if n == 0: continue
            lo, hi = wilson_ci(k, n)
            ci = f"[{lo:.0%}, {hi:.0%}]"
            print(f"    {region:<8} {k:>3}/{n:<3}  {k/n:>6.1%}  {ci}")