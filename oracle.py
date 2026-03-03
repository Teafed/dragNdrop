"""
oracle.py

Scripted oracle policy for the ShapeEnv.

The oracle knows the goal structure and produces near-optimal actions
analytically — no learning required.  Its two main uses:

  1. Generating demonstration datasets for behavior cloning (bc_train.py)
     without spending compute on RL from scratch.

  2. Providing a performance ceiling and sanity check for trained agents.

The oracle is intentionally imperfect: it adds small Gaussian noise to
its actions so the BC dataset doesn't overfit to pixel-perfect moves that
a real robot couldn't replicate anyway.

Usage:
    from oracle import OraclePolicy
    policy = OraclePolicy(env)
    action = policy.act(obs)    # obs is accepted but not used

    # generate a full dataset:
    from oracle import collect_demonstrations
    dataset = collect_demonstrations(goal, n_episodes=500, n_shapes=2)
"""

import numpy as np
import torch
from stable_baselines3.common.monitor import Monitor

from shape_env import ShapeEnv, SOLVE_TOLERANCE, MAX_NUDGE, WINDOW_W, WINDOW_H
from llm_goal_parser import parse_goal, get_embedding
from config import TASK_POOL, MAX_SHAPES, GOAL_ENCODING_DIM


# ---------------------------------------------------------------------------
# Oracle policy
# ---------------------------------------------------------------------------

class OraclePolicy:
    """
    Deterministic (+ optional noise) oracle that solves ShapeEnv analytically.

    Strategy for all tasks:
        - Find the unsolved shape that is furthest from its target (most urgent).
        - Compute the unit vector toward that target.
        - Return a nudge in that direction, scaled to MAX_NUDGE.

    This greedy nearest-target strategy works for both sort and group tasks
    because _compute_targets() handles the task-specific goal geometry.

    Args:
        env:        A ShapeEnv instance (raw or Monitor-wrapped).
        noise_std:  Std dev of Gaussian noise added to dx/dy.
                    0 = perfect oracle.  0.05 = realistic noise for BC data.
        seed:       RNG seed for reproducible datasets.
    """

    def __init__(self, env, noise_std: float = 0.05, seed: int = 42):
        self._env      = env
        self.noise_std = noise_std
        self.rng       = np.random.default_rng(seed)

    @property
    def env(self) -> ShapeEnv:
        """Unwrap Monitor wrapper if present."""
        e = self._env
        return e.env if isinstance(e, Monitor) else e

    def act(self, obs=None) -> np.ndarray:
        """
        Return an action [shape_selector, dx, dy] ∈ [-1, 1]³.

        obs is accepted for API compatibility with SB3's model.predict()
        signature but is not used — the oracle reads env state directly.
        """
        env  = self.env
        task = env.goal.get("task", "sort_by_size")

        # the greedy nearest-target strategy works for all tasks because
        # _compute_targets() encodes task-specific geometry into target positions.
        # any task not listed here gets random actions as a safe fallback.
        if task in ("sort_by_size", "group_by_color", "cluster",
                    "arrange_in_line", "arrange_in_grid", "push_to_region"):
            return self._act_greedy()
        else:
            return env.action_space.sample()

    def _act_greedy(self) -> np.ndarray:
        """
        Pick the unsolved shape furthest from its target and nudge it directly
        toward the target.  Among solved shapes, idle (zero nudge).
        """
        env   = self.env
        dists = [
            np.sqrt(
                (env.shapes[i].x - env.target_pos[i][0]) ** 2 +
                (env.shapes[i].y - env.target_pos[i][1]) ** 2
            )
            for i in range(env.n_shapes)
        ]

        unsolved = [i for i, d in enumerate(dists) if d > SOLVE_TOLERANCE]
        if not unsolved:
            # All shapes solved — return a valid idle action
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # Greedy: attend to the most displaced shape first
        shape_idx = max(unsolved, key=lambda i: dists[i])
        return self._nudge_toward_target(shape_idx)

    def _nudge_toward_target(self, shape_idx: int) -> np.ndarray:
        """
        Build the action to move shape_idx toward its target.
        Clips the nudge magnitude so we don't overshoot on the last step.
        Adds Gaussian noise so BC demonstrations aren't pixel-perfect.
        """
        env    = self.env
        s      = env.shapes[shape_idx]
        tx, ty = env.target_pos[shape_idx]

        delta_x = tx - s.x
        delta_y = ty - s.y
        dist    = np.sqrt(delta_x ** 2 + delta_y ** 2) + 1e-8

        # Normalise to [-1, 1] — env scales by MAX_NUDGE internally.
        # Clip scale to 1.0 so we don't overshoot on the final approach.
        scale = min(dist / MAX_NUDGE, 1.0)
        dx    = (delta_x / dist) * scale
        dy    = (delta_y / dist) * scale

        # Add noise for dataset diversity
        if self.noise_std > 0:
            dx += self.rng.normal(0, self.noise_std)
            dy += self.rng.normal(0, self.noise_std)

        dx = float(np.clip(dx, -1.0, 1.0))
        dy = float(np.clip(dy, -1.0, 1.0))

        # Map shape index → shape_selector ∈ [-1, 1]
        selector = (shape_idx / max(env.n_shapes - 1, 1)) * 2.0 - 1.0

        return np.array([selector, dx, dy], dtype=np.float32)


# ---------------------------------------------------------------------------
# Dataset collection
# ---------------------------------------------------------------------------

def collect_demonstrations(
    goal_encoder,
    n_episodes:  int   = 500,
    noise_std:   float = 0.05,
    seed:        int   = 0,
    verbose:     bool  = True,
) -> dict:
    """
    run the oracle across all tasks in TASK_POOL and collect (obs, action) pairs.

    each episode samples a random task prompt from TASK_POOL and a random
    n_shapes (2..MAX_SHAPES), so the dataset covers the full task and shape
    count distribution the policy will see during training.

    the goal encoder MLP is applied to the raw embedding before it's stored
    in the observation, so BC training sees the same obs format as PPO training.

    args:
        goal_encoder: GoalEncoder instance (from bc_train.py). used to project
                      EMBEDDING_DIM embeddings down to GOAL_ENCODING_DIM before
                      storing in the observation.
        n_episodes:   total episodes to collect across all tasks.
        noise_std:    gaussian noise added to oracle actions for dataset diversity.
        seed:         rng seed for reproducibility.
        verbose:      print progress every 10% of episodes.

    returns:
        {
            "observations":    np.ndarray [N, obs_size],
            "actions":         np.ndarray [N, 3],
            "episode_rewards": list[float],
            "episode_lengths": list[int],
            "solve_rate":      float,
        }
    """
    goal_encoder.eval()
    rng = np.random.default_rng(seed)

    all_obs     = []
    all_actions = []
    ep_rewards  = []
    ep_lengths  = []
    n_solved    = 0

    for ep in range(n_episodes):
        # sample task and shape count for this episode
        prompt   = TASK_POOL[rng.integers(0, len(TASK_POOL))]
        n_shapes = int(rng.integers(2, MAX_SHAPES + 1))
        goal     = parse_goal(prompt)

        # compute goal encoding for this episode
        raw_emb  = get_embedding(prompt)
        with torch.no_grad():
            emb_t    = torch.tensor(raw_emb, dtype=torch.float32).unsqueeze(0)
            encoding = goal_encoder(emb_t).squeeze(0).numpy()

        env    = ShapeEnv(n_shapes=n_shapes, goal=goal)
        env.set_goal_encoding(encoding)
        oracle = OraclePolicy(env, noise_std=noise_std,
                              seed=int(rng.integers(0, 2 ** 31)))

        obs, _     = env.reset(seed=int(rng.integers(0, 2 ** 31)))
        total_r    = 0.0
        steps      = 0
        done       = False
        terminated = False

        while not done:
            action = oracle.act(obs)
            all_obs.append(obs.copy())
            all_actions.append(action.copy())

            obs, reward, terminated, truncated, _ = env.step(action)
            total_r += reward
            steps   += 1
            done     = terminated or truncated

        env.close()
        ep_rewards.append(total_r)
        ep_lengths.append(steps)
        if terminated:
            n_solved += 1

        if verbose and (ep + 1) % max(n_episodes // 10, 1) == 0:
            recent = ep_rewards[-max(n_episodes // 10, 1):]
            print(f"  episode {ep+1:4d}/{n_episodes} | "
                  f"mean reward: {float(np.mean(recent)):7.2f} | "
                  f"solve rate so far: {n_solved / (ep + 1):.1%}")

    dataset = {
        "observations":    np.array(all_obs,     dtype=np.float32),
        "actions":         np.array(all_actions, dtype=np.float32),
        "episode_rewards": ep_rewards,
        "episode_lengths": ep_lengths,
        "solve_rate":      n_solved / n_episodes,
    }

    if verbose:
        print(f"\n--- dataset summary ---")
        print(f"  total transitions : {len(all_obs):,}")
        print(f"  solve rate        : {dataset['solve_rate']:.1%}")
        print(f"  mean ep reward    : {float(np.mean(ep_rewards)):.2f}")
        print(f"  mean ep length    : {float(np.mean(ep_lengths)):.1f}")

    return dataset