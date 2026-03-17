"""
sweep.py

lightweight hyperparameter sweep for short training trials.

runs a grid of (ent_coef, lr_ppo, bc_episodes) combinations for a fixed
number of PPO timesteps and records per-stage solve rates at the end of
each trial. results are saved to a CSV so you can compare runs without
having to read through walls of training output.

usage:
   python sweep.py                        # run full grid, 50k steps each
   python sweep.py --steps 30000          # shorter for quick sanity checks
   python sweep.py --out results/sweep1   # custom output prefix

output files:
   <out>.csv      — one row per trial with all params and final metrics
   <out>.txt      — human-readable summary sorted by reach solve rate

the sweep is intentionally small — it tests the variables most likely to
affect whether the agent learns at all in the starter stages. once reach
and touch are consistently solving, a deeper sweep on wave 3 params would
be more informative.
"""

import argparse
import csv
import itertools
import os
import time
import numpy as np
from stable_baselines3.common.env_util import make_vec_env

from config import MAX_SHAPES, N_ENVS
from shape_env import ShapeEnv
from curriculum import CurriculumManager


# ---------------------------------------------------------------------------
# sweep grid
# ---------------------------------------------------------------------------

SWEEP_GRID = {
   # ppo entropy coef — low risks collapse, high hurts stability
   "ent_coef":    [0.01, 0.02, 0.05],
   # ppo learning rate after bc init — too high destroys bc weights
   "lr_ppo":      [1e-5, 3e-5, 1e-4],
   # bc demo episodes — more = lower loss but slower to start
   "bc_episodes": [500, 1000],
}


# ---------------------------------------------------------------------------
# single trial
# ---------------------------------------------------------------------------

def run_trial(ent_coef: float, lr_ppo: float, bc_episodes: int,
              timesteps: int, seed: int = 42, verbose: bool = False) -> dict:
   """
   run one training trial with the given hyperparams.
   returns a dict of metrics from the end of training.
   """
   from bc_train import GoalEncoder, train_bc, build_ppo_from_bc
   from oracle import collect_demonstrations

   t_start = time.time()

   # --- bc phase ---
   goal_encoder = GoalEncoder()
   dataset      = collect_demonstrations(
      n_episodes=bc_episodes,
      verbose=verbose,
   )
   bc_network = train_bc(
      dataset=dataset,
      save_path=None,   # no save during sweep
      epochs=20,        # fixed shorter epochs for sweep speed
      device="cpu",
      verbose=verbose,
   )
   goal_encoder.eval()

   # --- ppo phase ---
   curriculum = CurriculumManager(verbose=False, start_stage=0)

   def make_env():
      from llm_goal_parser import parse_goal, get_embedding
      import torch
      prompt   = curriculum.sample_prompt()
      n_shp    = curriculum.sample_n_shapes()
      goal     = parse_goal(prompt)
      raw_emb  = get_embedding(prompt)
      with torch.no_grad():
         emb_t    = torch.tensor(raw_emb, dtype=torch.float32).unsqueeze(0)
         encoding = goal_encoder(emb_t).squeeze(0).numpy()
      env = ShapeEnv(n_shapes=n_shp, goal=goal)
      env.set_goal_encoding(encoding)
      return env

   vec_env = make_vec_env(make_env, n_envs=N_ENVS, seed=seed)
   model   = build_ppo_from_bc(
      bc_network,
      n_shapes=MAX_SHAPES,
      vec_env=vec_env,
      ent_coef=ent_coef,
      lr_ppo=lr_ppo,
   )

   model.learn(total_timesteps=timesteps, progress_bar=False)

   # --- eval phase: per-task solve rates on starter tasks ---
   metrics   = _eval_solve_rates(model, goal_encoder, curriculum,
                                 tasks=["reach", "touch", "drag"],
                                 n_episodes=20)
   elapsed   = time.time() - t_start
   final_stage = curriculum.stage_idx

   return {
      "ent_coef":    ent_coef,
      "lr_ppo":      lr_ppo,
      "bc_episodes": bc_episodes,
      "timesteps":   timesteps,
      "final_stage": final_stage,
      "elapsed_s":   round(elapsed),
      **metrics,
   }


def _eval_solve_rates(model, goal_encoder, tasks: list,
                      n_episodes: int) -> dict:
   """eval solve rate for each task; returns {task_sr_reach: 0.6, ...}."""
   import torch
   from llm_goal_parser import parse_goal, get_embedding
   from prompt_gen import PromptGenerator
   from stable_baselines3.common.monitor import Monitor
   _gen = PromptGenerator()
   results = {}
   for task in tasks:
      solved = []
      for _ in range(n_episodes):
         prompt  = _gen.sample()
         goal    = parse_goal(prompt)
         raw_emb = get_embedding(prompt)
         with torch.no_grad():
            emb_t    = torch.tensor(raw_emb, dtype=torch.float32).unsqueeze(0)
            encoding = goal_encoder(emb_t).squeeze(0).numpy()
         n_shp = 1 if task in ("reach", "touch", "drag") else 3
         env   = Monitor(ShapeEnv(n_shapes=n_shp, goal=goal))
         env.env.set_goal_encoding(encoding)
         obs, _     = env.reset()
         done       = False
         terminated = False
         while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
         solved.append(float(terminated))
         env.close()
      results[f"sr_{task}"] = round(float(np.mean(solved)), 3)
   return results


# ---------------------------------------------------------------------------
# sweep runner
# ---------------------------------------------------------------------------

def run_sweep(timesteps: int, out_prefix: str, verbose: bool):
   keys   = list(SWEEP_GRID.keys())
   values = list(SWEEP_GRID.values())
   combos = list(itertools.product(*values))

   print(f"\n=== hyperparameter sweep  ({len(combos)} trials x {timesteps:,} steps) ===\n")
   print("grid:")
   for k, v in SWEEP_GRID.items():
      print(f"  {k:<14} {v}")
   print()

   os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
   csv_path = out_prefix + ".csv"
   txt_path = out_prefix + ".txt"

   all_results = []
   fieldnames  = None

   for i, combo in enumerate(combos):
      params = dict(zip(keys, combo))
      print(f"trial {i+1}/{len(combos)}: {params}  ...", flush=True)

      try:
         result = run_trial(**params, timesteps=timesteps, verbose=verbose)
         all_results.append(result)
         sr_reach = result.get("sr_reach", 0)
         sr_touch = result.get("sr_touch", 0)
         sr_drag  = result.get("sr_drag",  0)
         print(f"  -> reach={sr_reach:.0%}  touch={sr_touch:.0%}  "
               f"drag={sr_drag:.0%}  stage={result['final_stage']}  "
               f"({result['elapsed_s']}s)")
      except Exception as e:
         print(f"  !! trial failed: {e}")
         result = {**params, "timesteps": timesteps, "error": str(e)}
         all_results.append(result)

      # write csv incrementally so partial results are saved if sweep is killed
      if fieldnames is None:
         fieldnames = list(result.keys())
      with open(csv_path, "w", newline="") as f:
         writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
         writer.writeheader()
         writer.writerows(all_results)

   # write human-readable summary sorted by reach solve rate
   successful = [r for r in all_results if "sr_reach" in r]
   successful.sort(key=lambda r: r.get("sr_reach", 0), reverse=True)

   with open(txt_path, "w") as f:
      f.write(f"sweep summary  ({len(combos)} trials, {timesteps:,} steps each)\n")
      f.write("=" * 70 + "\n\n")
      f.write(f"{'ent_coef':<12} {'lr_ppo':<10} {'bc_eps':<8} "
              f"{'reach':<8} {'touch':<8} {'drag':<8} {'stage':<7} {'time'}\n")
      f.write("-" * 70 + "\n")
      for r in successful:
         f.write(
            f"{r['ent_coef']:<12} {r['lr_ppo']:<10} {r['bc_episodes']:<8} "
            f"{r.get('sr_reach',0):<8.0%} {r.get('sr_touch',0):<8.0%} "
            f"{r.get('sr_drag',0):<8.0%} {r.get('final_stage',0):<7} "
            f"{r.get('elapsed_s',0)}s\n"
         )

   print(f"\n=== sweep complete ===")
   print(f"  results saved to {csv_path}")
   print(f"  summary saved to {txt_path}")
   print(f"\ntop 3 by reach solve rate:")
   for r in successful[:3]:
      print(f"  ent_coef={r['ent_coef']}  lr={r['lr_ppo']}  "
            f"bc_eps={r['bc_episodes']}  "
            f"reach={r.get('sr_reach',0):.0%}  "
            f"touch={r.get('sr_touch',0):.0%}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
   parser = argparse.ArgumentParser(
      description="hyperparameter sweep for short shape agent training trials.")
   parser.add_argument("--steps",   type=int, default=50_000,
                       help="ppo timesteps per trial (default 50000)")
   parser.add_argument("--out",     type=str, default="logs/sweep",
                       help="output file prefix for csv and txt (default logs/sweep)")
   parser.add_argument("--verbose", action="store_true",
                       help="show full bc and ppo output per trial")
   args = parser.parse_args()

   run_sweep(timesteps=args.steps, out_prefix=args.out, verbose=args.verbose)
