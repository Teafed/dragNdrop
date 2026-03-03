"""
callbacks.py

custom SB3 callbacks for tracking task-specific training progress.

the built-in SB3 metrics (ep_rew_mean etc.) tell you about general RL health.
these callbacks tell you whether the agent is actually getting better at tasks.

metrics logged under the "task/" TensorBoard prefix:
    task/mean_score      — 0-1 distance-based progress (1.0 = all shapes at targets)
    task/rank_corr       — spearman corr (sort/line) or cohesion/fraction (other tasks)
    task/solve_rate      — fraction of eval episodes that terminated as solved
    task/mean_ep_length  — average steps per episode (decreasing = faster solves)

the eval env is passed in at construction time and must already have its goal
encoding set. train.py's build_callbacks() handles this by sampling from TASK_POOL
so eval metrics are averaged across all tasks.
"""

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from shape_env import ShapeEnv


class ShapeTaskCallback(BaseCallback):
    """
    periodically evaluates the current policy and logs task-specific
    metrics to TensorBoard.

    args:
        eval_env:        a Monitor-wrapped ShapeEnv used only for evaluation.
                         must be a separate instance from any training env,
                         and must already have its goal encoding set.
        eval_freq:       how often (in training timesteps) to run evaluation.
        n_eval_episodes: number of episodes to average metrics over.
        verbose:         0 = silent, 1 = print metrics to stdout each eval.
    """

    def __init__(
        self,
        eval_env,
        eval_freq:       int = 5000,
        n_eval_episodes: int = 10,
        verbose:         int = 1,
    ):
        super().__init__(verbose)
        self.eval_env        = eval_env
        self.eval_freq       = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self._last_eval_step = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval_step < self.eval_freq:
            return True

        self._last_eval_step = self.num_timesteps
        metrics = self._run_eval()
        self._log_metrics(metrics)

        if self.verbose >= 1:
            print(
                f"[task eval @ {self.num_timesteps:>8,d} steps] "
                f"score: {metrics['mean_score']:.3f}  "
                f"rank/cohesion: {metrics['rank_corr']:+.3f}  "
                f"solve_rate: {metrics['solve_rate']:.0%}  "
                f"avg_steps: {metrics['mean_ep_length']:.1f}"
            )

        return True

    def _run_eval(self) -> dict:
        scores     = []
        rank_corrs = []
        ep_lengths = []
        solved     = []

        for _ in range(self.n_eval_episodes):
            obs, _     = self.eval_env.reset()
            done       = False
            length     = 0
            terminated = False

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = self.eval_env.step(action)
                done   = terminated or truncated
                length += 1

            scores.append(info.get("score",     0.0))
            rank_corrs.append(info.get("rank_corr", 0.0))
            ep_lengths.append(length)
            solved.append(float(terminated))

        return {
            "mean_score":     float(np.mean(scores)),
            "rank_corr":      float(np.mean(rank_corrs)),
            "solve_rate":     float(np.mean(solved)),
            "mean_ep_length": float(np.mean(ep_lengths)),
        }

    def _log_metrics(self, metrics: dict):
        """write all metrics to TensorBoard under the task/ prefix."""
        for name, value in metrics.items():
            self.logger.record(f"task/{name}", value)
        self.logger.dump(self.num_timesteps)