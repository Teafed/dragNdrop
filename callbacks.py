"""
callbacks.py

custom SB3 callbacks for tracking task-specific training progress and
managing curriculum advancement.

ShapeTaskCallback:
    periodically evaluates the policy on the CURRENT curriculum stage.
    re-samples a fresh eval env from the curriculum at every evaluation
    so it always measures the task the agent is actually training on.
    metrics logged under the "task/" TensorBoard prefix:
        task/mean_score      — mean per-task score (0-1)
        task/solve_rate      — fraction of eval episodes solved
        task/mean_ep_length  — average steps per episode
        task/current_stage   — curriculum stage index at eval time

CurriculumCallback:
    runs per-task evaluation every eval_freq steps and reports per-task
    solve rates back to the CurriculumManager. handles curriculum
    advancement and logs stage transitions to TensorBoard.
    also logs n_shapes range and active task count as curriculum/ metrics.
"""

import random
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from shape_env import ShapeEnv
from config import GOAL_ENCODING_DIM


class ShapeTaskCallback(BaseCallback):
   """
   periodically evaluates the current policy on the CURRENT curriculum stage.

   FIX: the original version captured a single eval env at construction time
   and never updated it, so after the curriculum advanced from stage 0 (reach)
   the callback kept reporting reach solve rates even while PPO trained
   touch/drag.  this version re-samples a fresh env from the curriculum
   (or from TASK_POOL when curriculum=None) at every evaluation.

   args:
      curriculum:      CurriculumManager instance, or None for no curriculum.
      goal_encoder:    GoalEncoder used to compute goal embeddings.
      eval_freq:       how often (in training timesteps) to run eval.
      n_eval_episodes: number of episodes to average over.
      verbose:         0 = silent, 1 = print metrics each eval.
   """

   def __init__(self, curriculum, goal_encoder,
                eval_freq: int = 5000,
                n_eval_episodes: int = 10,
                verbose: int = 1):
      super().__init__(verbose)
      self.curriculum      = curriculum
      self.goal_encoder    = goal_encoder
      self.eval_freq       = eval_freq
      self.n_eval_episodes = n_eval_episodes
      self._last_eval_step = 0

   def _make_eval_env(self):
      """build a fresh Monitor-wrapped ShapeEnv for the current stage."""
      import torch
      from llm_goal_parser import parse_goal, get_embedding
      from config import TASK_POOL

      if self.curriculum is not None:
         prompt = self.curriculum.sample_prompt()
         n_shp  = self.curriculum.sample_n_shapes()
      else:
         prompt = random.choice(TASK_POOL)
         n_shp  = None

      goal    = parse_goal(prompt)
      raw_emb = get_embedding(prompt)
      with torch.no_grad():
         emb_t    = torch.tensor(raw_emb, dtype=torch.float32).unsqueeze(0)
         encoding = self.goal_encoder(emb_t).squeeze(0).numpy()

      env = ShapeEnv(n_shapes=n_shp, goal=goal)
      env.set_goal_encoding(encoding)
      return Monitor(env)

   def _on_step(self) -> bool:
      if self.num_timesteps - self._last_eval_step < self.eval_freq:
         return True
      self._last_eval_step = self.num_timesteps
      metrics = self._run_eval()
      self._log_metrics(metrics)
      if self.verbose >= 1:
         stage = (self.curriculum.stage_idx
                  if self.curriculum is not None else -1)
         print(
            f"[task eval @ {self.num_timesteps:>8,d} steps]  "
            f"stage={stage}  "
            f"score: {metrics['mean_score']:.3f}  "
            f"rank/cohesion: {metrics['rank_corr']:+.3f}  "
            f"solve_rate: {metrics['solve_rate']:.0%}  "
            f"avg_steps: {metrics['mean_ep_length']:.1f}"
         )
      return True

   def _run_eval(self) -> dict:
      scores, rank_corrs, ep_lengths, solved = [], [], [], []
      for _ in range(self.n_eval_episodes):
         # fresh env each episode so every episode reflects current stage
         eval_env   = self._make_eval_env()
         obs, _     = eval_env.reset()
         done       = False
         length     = 0
         terminated = False
         while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = eval_env.step(action)
            done    = terminated or truncated
            length += 1
         scores.append(info.get("score",     0.0))
         rank_corrs.append(info.get("rank_corr", 0.0))
         ep_lengths.append(length)
         solved.append(float(terminated))
         eval_env.close()

      stage = (self.curriculum.stage_idx
               if self.curriculum is not None else -1)
      return {
         "mean_score":     float(np.mean(scores)),
         "rank_corr":      float(np.mean(rank_corrs)),
         "solve_rate":     float(np.mean(solved)),
         "mean_ep_length": float(np.mean(ep_lengths)),
         "current_stage":  float(stage),
      }

   def _log_metrics(self, metrics: dict):
      for name, value in metrics.items():
         self.logger.record(f"task/{name}", value)
      self.logger.dump(self.num_timesteps)


class CurriculumCallback(BaseCallback):
   """
   per-task evaluation callback that drives curriculum advancement.

   runs n_eval_episodes episodes per active task every eval_freq steps,
   reports per-task solve rates to the CurriculumManager, and advances
   the curriculum when the gate condition is met.

   args:
      curriculum:      CurriculumManager instance (from curriculum.py).
      goal_encoder:    trained GoalEncoder (for computing goal embeddings).
      eval_freq:       how often (in timesteps) to run per-task eval.
      n_eval_episodes: episodes per task per eval.
      verbose:         0 = silent, 1 = print per-task results each eval.
   """

   def __init__(self, curriculum, goal_encoder,
                eval_freq: int = 10_000, n_eval_episodes: int = 20,
                verbose: int = 1):
      super().__init__(verbose)
      self.curriculum      = curriculum
      self.goal_encoder    = goal_encoder
      self.eval_freq       = eval_freq
      self.n_eval_episodes = n_eval_episodes
      self._last_eval_step = 0

   def _on_step(self) -> bool:
      if self.num_timesteps - self._last_eval_step < self.eval_freq:
         return True
      self._last_eval_step = self.num_timesteps

      per_task_sr = self._run_per_task_eval()
      self._log_curriculum_metrics(per_task_sr)

      if self.verbose >= 1:
         print(f"\n[curriculum eval @ {self.num_timesteps:,} steps]")
         for task, sr in per_task_sr.items():
            print(f"  {task:<22} solve={sr:.0%}")
         print(f"  status: {self.curriculum.status()}")

      advanced = self.curriculum.maybe_advance(per_task_sr, self.num_timesteps)
      if advanced:
         # log new stage index so tensorboard shows the transition
         self.logger.record("curriculum/stage", self.curriculum.stage_idx)
         self.logger.dump(self.num_timesteps)

      return True

   def _run_per_task_eval(self) -> dict:
      """
      run n_eval_episodes per active task and return {task: solve_rate}.
      creates temporary envs for each task — does not affect training envs.
      """
      import torch
      from llm_goal_parser import parse_goal, get_embedding
      from curriculum import _PROMPT_POOL

      per_task_sr = {}

      for task in self.curriculum.active_tasks:
         solved = []
         for _ in range(self.n_eval_episodes):
            prompt  = random.choice(_PROMPT_POOL[task])
            goal    = parse_goal(prompt)
            raw_emb = get_embedding(prompt)
            with torch.no_grad():
               emb_t    = torch.tensor(raw_emb, dtype=torch.float32).unsqueeze(0)
               encoding = self.goal_encoder(emb_t).squeeze(0).numpy()

            lo, hi  = self.curriculum.n_shapes_range
            n_shp   = random.randint(lo, hi)
            env     = Monitor(ShapeEnv(n_shapes=n_shp, goal=goal))
            env.env.set_goal_encoding(encoding)

            obs, _     = env.reset()
            done       = False
            terminated = False
            while not done:
               action, _ = self.model.predict(obs, deterministic=True)
               obs, _, terminated, truncated, _ = env.step(action)
               done = terminated or truncated
            solved.append(float(terminated))
            env.close()

         per_task_sr[task] = float(np.mean(solved))

      return per_task_sr

   def _log_curriculum_metrics(self, per_task_sr: dict):
      """log per-task solve rates and curriculum state to TensorBoard."""
      for task, sr in per_task_sr.items():
         # shorten task name for tensorboard readability
         short = task.replace("arrange_in_", "")
         self.logger.record(f"curriculum/sr_{short}", sr)
      self.logger.record("curriculum/stage",       self.curriculum.stage_idx)
      self.logger.record("curriculum/n_shapes_max",
                         self.curriculum.n_shapes_range[1])
      self.logger.record("curriculum/n_active_tasks",
                         len(self.curriculum.active_tasks))
      self.logger.dump(self.num_timesteps)
