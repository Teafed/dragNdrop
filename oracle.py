"""
oracle.py

Analytical oracle policy for collecting BC demonstrations.

--- explore / commit loop ---
    explore: select which shape to move next, using weighted random
             selection so the "most wrong" shape is most likely but
             not guaranteed. this produces varied demonstrations for BC.
    commit:  move the selected shape until its local completion condition
             is satisfied, then return to explore.

--- task-specific logic ---
    arrange_in_sequence / arrange_in_line:
        priority  = |current_rank - ideal_rank| (displacement from ideal rank)
        commit    = shape is in its correct rank cell (closer to ideal than
                    to any adjacent slot) AND for arrange_in_line, within
                    LINE_SPREAD_THRESHOLD of the target perpendicular position
        target    = ideal slot position computed from current sizes/attrs
        variation = random offset within COMMIT_JITTER pixels of the slot center

    arrange_in_region:
        priority  = distance past the boundary (further = more urgent)
        commit    = shape is past the boundary AND has been nudged a random
                    extra distance into the region (so shapes end up spread
                    out rather than all piled at the edge)
        target    = boundary + random depth into region

    arrange_in_groups:
        priority  = per-shape cohesion deficit (worst = most urgent)
        commit    = nearest neighbor is same attribute AND nearest different-
                    attribute shape is beyond SEPARATION_TARGET distance
        target    = group centroid (average position of same-attribute shapes
                    that are already committed), with random jitter

--- stochastic selection ---
    selection probability proportional to softmax(priority / EXPLORE_TEMP).
    EXPLORE_TEMP controls greediness: low = nearly deterministic (most wrong
    shape chosen almost always), high = nearly uniform random.
"""

import numpy as np
from shape_env import (
   ShapeEnv, WINDOW_W, WINDOW_H, MARGIN, MAX_NUDGE,
   SCORE_SOLVE_THRESHOLD, REGION_INNER, LINE_SPREAD_THRESHOLD,
   COLOR_NAMES, SHAPE_TYPE_IDX,
)
from config import TASK_POOL, MAX_SHAPES

# ---------------------------------------------------------------------------
# oracle hyperparameters
# ---------------------------------------------------------------------------

EXPLORE_TEMP       = 0.5    # softmax temperature for shape selection
COMMIT_JITTER      = 30.0   # pixels — random offset from ideal slot position
REGION_EXTRA_DEPTH = 60.0   # pixels — random extra depth into region past boundary
SEPARATION_TARGET  = 100.0  # pixels — commit condition: nearest diff-attr shape
NOISE_STD          = 0.08   # action noise std during collection (set 0 for oracle demo)


# ---------------------------------------------------------------------------
# OraclePolicy
# ---------------------------------------------------------------------------

class OraclePolicy:
   """
   oracle policy for a single ShapeEnv instance.
   call act(obs) each step to get an action.
   internally maintains committed_shape and committed_target across steps.
   """

   def __init__(self, env: ShapeEnv, noise_std: float = NOISE_STD,
                rng: np.random.Generator = None):
      self.env              = env
      self.noise_std        = noise_std
      self.rng              = rng or np.random.default_rng()
      self.committed_shape  = None   # index of shape currently being moved
      self.committed_target = None   # (x, y) target for committed shape

   def reset(self):
      """call this when the env resets to clear committed state."""
      self.committed_shape  = None
      self.committed_target = None

   def act(self, obs=None) -> np.ndarray:
      """
      return an action [shape_selector, dx, dy] in [-1, 1]^3.
      obs is accepted but unused — the oracle reads env state directly.
      """
      env  = self.env
      task = env.goal.get("task", "arrange_in_sequence")

      # --- explore: pick next shape if no commitment or current one is done ---
      if (self.committed_shape is None
            or self._local_condition_met(self.committed_shape)):
         self.committed_shape, self.committed_target = self._explore(task)

      # --- commit: move toward target ---
      action = self._commit(self.committed_shape, self.committed_target)

      if self.noise_std > 0:
         action[1] += float(self.rng.normal(0, self.noise_std))
         action[2] += float(self.rng.normal(0, self.noise_std))
         action = np.clip(action, -1.0, 1.0)

      return action.astype(np.float32)

   # -------------------------------------------------------------------------
   # explore phase — select next shape and compute its target
   # -------------------------------------------------------------------------

   def _explore(self, task: str):
      """
      select next shape and compute its committed target.
      returns (shape_idx, (target_x, target_y)).
      """
      env       = self.env
      n         = env.n_shapes
      attribute = env.goal.get("attribute", "none")

      if task in ("arrange_in_sequence", "arrange_in_line"):
         priorities = self._priorities_sequence(task)
      elif task == "arrange_in_region":
         priorities = self._priorities_region()
      elif task == "arrange_in_groups":
         priorities = self._priorities_groups(attribute)
      else:
         priorities = np.ones(n)

      # softmax weighted selection
      shape_idx = self._softmax_select(priorities)
      target    = self._compute_target(shape_idx, task)
      return shape_idx, target

   def _priorities_sequence(self, task: str) -> np.ndarray:
      """
      priority for arrange_in_sequence / arrange_in_line:
      |current_rank - ideal_rank| per shape.
      """
      env       = self.env
      n         = env.n_shapes
      axis      = env.goal.get("axis", "x")
      direction = env.goal.get("direction", "ascending")
      attribute = env.goal.get("attribute", "size")

      positions   = [s.x if axis == "x" else s.y for s in env.shapes]
      attr_vals   = env._get_attribute_values(attribute)
      ideal_ranks = np.argsort(np.argsort(attr_vals)).astype(float)
      if direction == "descending":
         ideal_ranks = (n - 1) - ideal_ranks
      current_ranks = np.argsort(np.argsort(positions)).astype(float)

      priorities = np.abs(current_ranks - ideal_ranks) + 0.1
      return priorities

   def _priorities_region(self) -> np.ndarray:
      """
      priority for arrange_in_region:
      distance outside the boundary (0 if already inside).
      """
      env      = self.env
      region   = env.goal.get("region", "left")
      boundary = REGION_INNER[region]

      def dist_outside(s):
         if region == "left":
            return max(s.x - boundary, 0)
         elif region == "right":
            return max(boundary - s.x, 0)
         elif region == "top":
            return max(s.y - boundary, 0)
         else:
            return max(boundary - s.y, 0)

      priorities = np.array([dist_outside(s) for s in env.shapes]) + 0.1
      return priorities

   def _priorities_groups(self, attribute: str) -> np.ndarray:
      """
      priority for arrange_in_groups:
      per-shape cohesion deficit = 1 - per_shape_score.
      shapes that are most out of place get highest priority.
      """
      env       = self.env
      n         = env.n_shapes
      half_diag = np.sqrt((WINDOW_W / 2) ** 2 + (WINDOW_H / 2) ** 2)

      def get_attr(s):
         return s.color_name if attribute == "color" else s.shape_type

      scores = []
      for i in range(n):
         same_d = [
            np.sqrt((env.shapes[i].x - env.shapes[j].x) ** 2 +
                    (env.shapes[i].y - env.shapes[j].y) ** 2)
            for j in range(n)
            if i != j and get_attr(env.shapes[i]) == get_attr(env.shapes[j])
         ]
         diff_d = [
            np.sqrt((env.shapes[i].x - env.shapes[j].x) ** 2 +
                    (env.shapes[i].y - env.shapes[j].y) ** 2)
            for j in range(n)
            if i != j and get_attr(env.shapes[i]) != get_attr(env.shapes[j])
         ]
         if not same_d or not diff_d:
            nn_correct = 1.0
         else:
            nn_correct = 1.0 if min(same_d) < min(diff_d) else 0.0
         sep = min(min(diff_d) / half_diag, 1.0) if diff_d else 1.0
         scores.append(0.6 * nn_correct + 0.4 * sep)

      priorities = 1.0 - np.array(scores) + 0.1
      return priorities

   def _softmax_select(self, priorities: np.ndarray) -> int:
      """weighted random selection using softmax over priorities."""
      logits = priorities / EXPLORE_TEMP
      logits = logits - logits.max()   # numerical stability
      weights = np.exp(logits)
      weights = weights / weights.sum()
      return int(self.rng.choice(len(priorities), p=weights))

   # -------------------------------------------------------------------------
   # target computation — where should the selected shape go?
   # -------------------------------------------------------------------------

   def _compute_target(self, shape_idx: int, task: str) -> tuple:
      env       = self.env
      n         = env.n_shapes
      axis      = env.goal.get("axis", "x")
      direction = env.goal.get("direction", "ascending")
      attribute = env.goal.get("attribute", "none")
      region    = env.goal.get("region", "left")
      pad       = MARGIN * 2

      if task in ("arrange_in_sequence", "arrange_in_line"):
         # compute the ideal slot position for this shape
         attr_vals   = env._get_attribute_values(attribute)
         ideal_ranks = np.argsort(np.argsort(attr_vals)).astype(float)
         if direction == "descending":
            ideal_ranks = (n - 1) - ideal_ranks
         ideal_rank = ideal_ranks[shape_idx]

         if axis == "x":
            slots = np.linspace(pad, WINDOW_W - pad, n)
            tx    = float(slots[int(round(ideal_rank))])
            # for arrange_in_line, target the perpendicular center; for
            # sequence, keep the shape's current y (unconstrained)
            if task == "arrange_in_line":
               ty = WINDOW_H / 2
            else:
               ty = env.shapes[shape_idx].y
         else:
            slots = np.linspace(pad, WINDOW_H - pad, n)
            ty    = float(slots[int(round(ideal_rank))])
            if task == "arrange_in_line":
               tx = WINDOW_W / 2
            else:
               tx = env.shapes[shape_idx].x

         # add random jitter so demos are varied
         jitter_x = float(self.rng.uniform(-COMMIT_JITTER, COMMIT_JITTER))
         jitter_y = float(self.rng.uniform(-COMMIT_JITTER, COMMIT_JITTER))
         tx = float(np.clip(tx + jitter_x, pad, WINDOW_W - pad))
         ty = float(np.clip(ty + jitter_y, pad, WINDOW_H - pad))
         return (tx, ty)

      elif task == "arrange_in_region":
         boundary = REGION_INNER[region]
         extra    = float(self.rng.uniform(0, REGION_EXTRA_DEPTH))
         s        = env.shapes[shape_idx]
         if region == "left":
            tx = float(np.clip(boundary - extra, pad, WINDOW_W - pad))
            ty = s.y   # leave perpendicular coordinate alone
         elif region == "right":
            tx = float(np.clip(boundary + extra, pad, WINDOW_W - pad))
            ty = s.y
         elif region == "top":
            tx = s.x
            ty = float(np.clip(boundary - extra, pad, WINDOW_H - pad))
         else:
            tx = s.x
            ty = float(np.clip(boundary + extra, pad, WINDOW_H - pad))
         return (tx, ty)

      elif task == "arrange_in_groups":
         return self._compute_group_target(shape_idx, attribute)

      return (env.shapes[shape_idx].x, env.shapes[shape_idx].y)

   def _compute_group_target(self, shape_idx: int, attribute: str) -> tuple:
      """
      target for arrange_in_groups: move toward the centroid of same-attribute
      shapes, or toward an assigned canvas region if no same-attr shapes exist
      yet in a good position.
      """
      env = self.env
      n   = env.n_shapes
      pad = MARGIN * 2

      def get_attr(s):
         return s.color_name if attribute == "color" else s.shape_type

      my_attr  = get_attr(env.shapes[shape_idx])
      same_pos = [
         (env.shapes[j].x, env.shapes[j].y)
         for j in range(n)
         if j != shape_idx and get_attr(env.shapes[j]) == my_attr
      ]

      if same_pos:
         # move toward centroid of same-attribute shapes
         cx = float(np.mean([p[0] for p in same_pos]))
         cy = float(np.mean([p[1] for p in same_pos]))
      else:
         # no same-attr shapes to target — assign a canvas zone by attribute value
         unique_attrs = sorted(set(get_attr(s) for s in env.shapes))
         n_groups     = len(unique_attrs)
         group_idx    = unique_attrs.index(my_attr)
         xs           = np.linspace(WINDOW_W / (n_groups + 1),
                                    WINDOW_W * n_groups / (n_groups + 1),
                                    n_groups)
         cx = float(xs[group_idx])
         cy = WINDOW_H / 2

      # add jitter for demo variety
      jitter_x = float(self.rng.uniform(-COMMIT_JITTER, COMMIT_JITTER))
      jitter_y = float(self.rng.uniform(-COMMIT_JITTER, COMMIT_JITTER))
      tx = float(np.clip(cx + jitter_x, pad, WINDOW_W - pad))
      ty = float(np.clip(cy + jitter_y, pad, WINDOW_H - pad))
      return (tx, ty)

   # -------------------------------------------------------------------------
   # local completion conditions
   # -------------------------------------------------------------------------

   def _local_condition_met(self, shape_idx: int) -> bool:
      """
      returns True when the committed shape has satisfied its local condition
      and the oracle should explore for the next shape to move.
      """
      if self.committed_target is None:
         return True

      env    = self.env
      task   = env.goal.get("task", "arrange_in_sequence")
      s      = env.shapes[shape_idx]
      tx, ty = self.committed_target

      dist_to_target = np.sqrt((s.x - tx) ** 2 + (s.y - ty) ** 2)

      if task in ("arrange_in_sequence", "arrange_in_line"):
         # close enough to the committed target (within one nudge)
         return dist_to_target < MAX_NUDGE * 1.5

      elif task == "arrange_in_region":
         region   = env.goal.get("region", "left")
         boundary = REGION_INNER[region]
         # shape is past boundary and close to committed target
         in_region = (
            (region == "left"   and s.x <= boundary) or
            (region == "right"  and s.x >= boundary) or
            (region == "top"    and s.y <= boundary) or
            (region == "bottom" and s.y >= boundary)
         )
         return in_region and dist_to_target < MAX_NUDGE * 1.5

      elif task == "arrange_in_groups":
         # committed when shape is close to its target. separation is enforced
         # by the priority function re-selecting poorly separated shapes next.
         return dist_to_target < MAX_NUDGE * 2.0

      return dist_to_target < MAX_NUDGE * 1.5

   # -------------------------------------------------------------------------
   # commit phase — nudge toward target
   # -------------------------------------------------------------------------

   def _commit(self, shape_idx: int, target: tuple) -> np.ndarray:
      """produce an action that moves shape_idx toward target."""
      env    = self.env
      n      = env.n_shapes
      s      = env.shapes[shape_idx]
      tx, ty = target

      # selector value that decodes to shape_idx
      if n == 1:
         selector = 0.0
      else:
         selector = (shape_idx / (n - 1)) * 2.0 - 1.0

      # direction toward target, normalized
      dx = tx - s.x
      dy = ty - s.y
      dist = np.sqrt(dx ** 2 + dy ** 2)
      if dist < 1e-6:
         dx_norm, dy_norm = 0.0, 0.0
      else:
         dx_norm = dx / dist
         dy_norm = dy / dist

      return np.array([selector, dx_norm, dy_norm], dtype=np.float32)


# ---------------------------------------------------------------------------
# demonstration collection
# ---------------------------------------------------------------------------

def collect_demonstrations(
   n_episodes:    int   = 500,
   noise_std:     float = NOISE_STD,
   verbose:       bool  = True,
) -> dict:
   """
   run the oracle across all tasks in TASK_POOL and collect (obs, action) pairs.
   each episode samples a random prompt from TASK_POOL and random n_shapes.

   returns a dataset dict:
       "observations": np.ndarray (N, obs_dim)
       "actions":      np.ndarray (N, 3)
   """
   from llm_goal_parser import parse_goal, get_embedding
   import torch
   from bc_train import GoalEncoder
   from config import GOAL_ENCODING_DIM

   goal_encoder = GoalEncoder()
   goal_encoder.eval()
   rng = np.random.default_rng()

   all_obs     = []
   all_actions = []
   n_solved    = 0

   if verbose:
      print(f"\n--- collecting {n_episodes} oracle demonstrations "
            f"across {len(TASK_POOL)} task pool prompts ---")

   for ep in range(1, n_episodes + 1):
      prompt = rng.choice(TASK_POOL)
      goal   = parse_goal(prompt)
      n_shp  = int(rng.integers(2, MAX_SHAPES + 1))

      raw_emb = get_embedding(prompt)
      with torch.no_grad():
         import torch as th
         emb_t    = th.tensor(raw_emb, dtype=th.float32).unsqueeze(0)
         encoding = goal_encoder(emb_t).squeeze(0).numpy()

      env    = ShapeEnv(n_shapes=n_shp, goal=goal, render_mode=None)
      env.set_goal_encoding(encoding)
      oracle = OraclePolicy(env, noise_std=noise_std, rng=rng)
      obs, _ = env.reset()

      ep_obs     = []
      ep_actions = []
      done       = False

      while not done:
         action = oracle.act(obs)
         ep_obs.append(obs.copy())
         ep_actions.append(action.copy())
         obs, _, terminated, truncated, _ = env.step(action)
         done = terminated or truncated

      if terminated:
         n_solved += 1
         all_obs.extend(ep_obs)
         all_actions.extend(ep_actions)

      env.close()

      if verbose and ep % 50 == 0:
         sr = n_solved / ep * 100
         print(f"  episode {ep:4d}/{n_episodes} | "
               f"mean reward: n/a  | solve rate so far: {sr:.1f}%")

   if verbose:
      sr = n_solved / n_episodes * 100
      print(f"\n--- dataset summary ---")
      print(f"  total transitions : {len(all_obs):,}")
      print(f"  solve rate        : {sr:.1f}%")
      print(f"  mean ep length    : "
            f"{len(all_obs)/max(n_solved,1):.1f}")

   return {
      "observations": np.array(all_obs,     dtype=np.float32),
      "actions":      np.array(all_actions, dtype=np.float32),
   }
