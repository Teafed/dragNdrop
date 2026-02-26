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
from stable_baselines3.common.monitor import Monitor

from shape_env import ShapeEnv, SOLVE_TOLERANCE, MAX_NUDGE, WINDOW_W, WINDOW_H


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

        # All tasks currently use the same greedy strategy because the
        # target positions encode task-specific geometry.
        # Add elif branches here as new tasks require different logic.
        if task in ("sort_by_size", "group_by_color", "cluster"):
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
    goal:        dict,
    n_episodes:  int   = 500,
    n_shapes:    int   = 2,
    noise_std:   float = 0.05,
    seed:        int   = 0,
    verbose:     bool  = True,
) -> dict:
    """
    Run the oracle for n_episodes and collect (observation, action) pairs.

    This is dramatically cheaper than RL training:
        - 500 episodes × ~100 steps/episode = 50 000 transitions
        - Takes ~10 seconds on a laptop CPU
        - Typically achieves 90%+ solve rate with noise_std=0.05

    Returns:
        {
            "observations":    np.ndarray  [N, obs_dim],
            "actions":         np.ndarray  [N, 3],
            "episode_rewards": list[float],
            "episode_lengths": list[int],
            "solve_rate":      float,    # fraction of episodes solved
        }
    """
    env    = ShapeEnv(n_shapes=n_shapes, goal=goal, render_mode=None)
    oracle = OraclePolicy(env, noise_std=noise_std, seed=seed)

    all_obs     = []
    all_actions = []
    ep_rewards  = []
    ep_lengths  = []
    n_solved    = 0

    rng = np.random.default_rng(seed)

    for ep in range(n_episodes):
        obs, _  = env.reset(seed=int(rng.integers(0, 2 ** 31)))
        total_r = 0.0
        steps   = 0
        done    = False
        terminated = False

        while not done:
            action = oracle.act(obs)
            all_obs.append(obs.copy())
            all_actions.append(action.copy())

            obs, reward, terminated, truncated, _ = env.step(action)
            total_r += reward
            steps   += 1
            done     = terminated or truncated

        ep_rewards.append(total_r)
        ep_lengths.append(steps)
        if terminated:
            n_solved += 1

        if verbose and (ep + 1) % max(n_episodes // 10, 1) == 0:
            recent = ep_rewards[-max(n_episodes // 10, 1):]
            print(f"  episode {ep+1:4d}/{n_episodes} | "
                  f"mean reward: {float(np.mean(recent)):7.2f} | "
                  f"solve rate so far: {n_solved / (ep + 1):.1%}")

    env.close()

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