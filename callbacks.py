"""
callbacks.py

custom stable-baselines3 callbacks for tracking task-specific
training progress in tensorboard.

the built-in SB3 metrics (ep_rew_mean, etc.) tell you about
general RL health. these callbacks tell you whether the agent
is actually getting better at the task.

usage in train.py:
   from callbacks import ShapeTaskCallback
   task_callback = ShapeTaskCallback(eval_env, eval_freq=5000, verbose=1)
   model.learn(..., callback=[eval_callback, task_callback])
"""

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EventCallback
from stable_baselines3.common.evaluation import evaluate_policy

from shape_env import ShapeEnv


class ShapeTaskCallback(BaseCallback):
   """
   runs periodic evaluation episodes and logs task-specific metrics
   to tensorboard. these show up under a "task/" prefix so they're
   easy to find alongside SB3's built-in rollout/ and train/ groups.

   metrics logged:
      task/rank_correlation       — spearman corr between positions and sizes.
                                    1.0 = perfectly sorted, -1.0 = reversed.
      task/solve_rate             — fraction of eval episodes that hit the
                                    reward threshold (i.e. "solved").
      task/mean_y_spread          — std of y positions, normalized to [0,1].
                                    lower = shapes are more in a line.
      task/mean_episode_steps     — avg steps taken per eval episode.
                                    decreasing = agent solving faster.
      task/mean_final_reward      — avg reward at the last step of each episode.
                                    useful to track separately from ep_rew_mean.
   """

   def __init__(self, eval_env: ShapeEnv, eval_freq: int = 5000,
                n_eval_episodes: int = 10, verbose: int = 1):
      """
      args:
         eval_env:        a separate ShapeEnv instance used only for eval.
                          should not be the training env.
         eval_freq:       how often (in training timesteps) to run eval.
         n_eval_episodes: how many episodes to average metrics over.
         verbose:         0 = silent, 1 = print metrics each eval.
      """
      super().__init__(verbose)
      self.eval_env        = eval_env
      self.eval_freq       = eval_freq
      self.n_eval_episodes = n_eval_episodes

   def _on_step(self) -> bool:
      # only run eval every eval_freq timesteps
      if self.num_timesteps % self.eval_freq != 0:
         return True

      metrics = self._run_eval()
      self._log_metrics(metrics)

      if self.verbose >= 1:
         print(
            f"[task eval @ {self.num_timesteps:,} steps] "
            f"rank_corr: {metrics['rank_correlation']:+.3f}  "
            f"solve_rate: {metrics['solve_rate']:.0%}  "
            f"y_spread: {metrics['mean_y_spread']:.3f}  "
            f"avg_steps: {metrics['mean_episode_steps']:.1f}"
         )

      return True   # returning False would stop training

   def _run_eval(self) -> dict:
      """
      run n_eval_episodes and collect per-episode metrics.
      returns a dict of averaged values.
      """
      rank_correlations  = []
      y_spreads          = []
      episode_steps      = []
      final_rewards      = []
      solved             = []

      for _ in range(self.n_eval_episodes):
         obs, _   = self.eval_env.reset()
         done     = False
         steps    = 0
         last_reward = 0.0

         while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.eval_env.step(action)
            done        = terminated or truncated
            steps      += 1
            last_reward = info.get("score", reward)   # score = raw rank corr, not delta

         # pull metrics directly from env state.
         # .unwrapped pierces the Monitor wrapper to get to ShapeEnv,
         # since Monitor doesn't proxy private methods like _compute_score.
         inner  = self.eval_env.unwrapped
         score  = inner._compute_score()
         shapes = inner.shapes
         goal   = inner.goal
         axis   = goal.get("axis", "x")

         if axis == "x":
            spread = float(np.std([s.y for s in shapes]) / 600)
         else:
            spread = float(np.std([s.x for s in shapes]) / 800)

         rank_correlations.append(score)
         y_spreads.append(spread)
         episode_steps.append(steps)
         final_rewards.append(last_reward)
         solved.append(terminated)

      return {
         "rank_correlation":    float(np.mean(rank_correlations)),   # using _compute_score now
         "mean_y_spread":       float(np.mean(y_spreads)),
         "mean_episode_steps":  float(np.mean(episode_steps)),
         "mean_final_reward":   float(np.mean(final_rewards)),
         "solve_rate":          float(np.mean(solved)),
      }

   def _log_metrics(self, metrics: dict):
      """write all metrics to tensorboard under the task/ prefix."""
      for name, value in metrics.items():
         self.logger.record(f"task/{name}", value)
      # force a tensorboard write at this timestep
      self.logger.dump(self.num_timesteps)
