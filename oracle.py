"""
oracle.py

analytical oracle policy for collecting BC demonstrations via cursor control.

--- action space ---
   [dx, dy, grip] — cursor movement and grip, all in [-1, 1]

--- explore / commit loop ---
   explore: select which shape to move next using weighted random selection.
            the "most wrong" shape is most likely chosen but not guaranteed.
            if the episode score is already above IDLE_THRESHOLD, all
            priorities are set very low and the oracle idles until the
            episode terminates naturally.

   commit:  navigate cursor to the selected shape, grip it, drag it to
            the computed target, then release. returns to explore once
            the local condition is met.

   commit sub-phases (per committed shape):
      NAVIGATE  cursor moves toward shape until within GRIP_RADIUS
      GRIP_ON   grip activates (one step with grip=1)
      DRAG      cursor + shape move toward target
      GRIP_OFF  grip releases (one step with grip=-1), shape stays at target

--- navigate-then-drag oracle ---
   the oracle no longer teleports shapes. all movement goes through cursor
   physics: navigate to shape, grip, drag to target, release.
   this produces BC demonstrations that reflect physical cursor interaction
   rather than instantaneous selection.

--- high-level annotations ---
   each transition also stores:
      hl_shape_idx:  which shape the oracle selected in explore (int)
      hl_target_xy:  the committed target (x, y) in pixels
   these are saved alongside (obs, action) pairs for future HBC training.
   flat BC training only uses (obs, action).

--- starter tasks ---
   reach:  navigate cursor to target shape (env.target_idx)
   touch:  navigate to env.target_idx then activate grip
   drag:   navigate to env.target_idx, grip, drag to target region

--- wave 3 tasks ---
   arrange_in_sequence / arrange_in_line:
      priority  = |current_rank - ideal_rank|
      target    = ideal slot position + jitter

   arrange_in_region:
      priority  = distance outside boundary
      target    = boundary + random depth into region

   arrange_in_groups:
      priority  = per-shape cohesion deficit
      target    = centroid of same-attribute shapes + jitter
"""

import os
import numpy as np

from shape_env import (
   ShapeEnv, WINDOW_W, WINDOW_H, MARGIN,
   SCORE_SOLVE_THRESHOLD, REGION_INNER, LINE_SPREAD_THRESHOLD,
)
from config import (
   TASK_POOL, MAX_SHAPES, CURSOR_SPEED, GRIP_RADIUS, GRIP_THRESHOLD,
)

# ---------------------------------------------------------------------------
# oracle hyperparameters
# ---------------------------------------------------------------------------

EXPLORE_TEMP       = 0.5    # softmax temperature for shape selection
COMMIT_JITTER      = 30.0   # pixels — random offset from ideal target
REGION_EXTRA_DEPTH = 60.0   # pixels — random extra depth into region
NOISE_STD          = 0.06   # action noise std (0 for clean oracle demo)

# oracle idles only when the episode is genuinely solved — same threshold
# as the environment's own solve check so there's no gap to exploit
IDLE_THRESHOLD = SCORE_SOLVE_THRESHOLD

# cursor approach: how close before gripping (slightly larger than GRIP_RADIUS
# so the oracle reliably lands inside the grip zone)
APPROACH_DIST = GRIP_RADIUS * 0.8

# commit sub-phases
_NAVIGATE  = "navigate"
_GRIP_ON   = "grip_on"
_DRAG      = "drag"
_GRIP_OFF  = "grip_off"
_DONE      = "done"      # one-step terminal — forces re-exploration next act()


# ---------------------------------------------------------------------------
# OraclePolicy
# ---------------------------------------------------------------------------

class OraclePolicy:
   """
   oracle policy for a single ShapeEnv instance.
   call act(obs) each step to get an action [dx, dy, grip].
   maintains cursor sub-phase state across steps.
   """

   def __init__(self, env: ShapeEnv, noise_std: float = NOISE_STD,
                rng: np.random.Generator = None):
      self.env       = env
      self.noise_std = noise_std
      self.rng       = rng or np.random.default_rng()
      self._reset_commit()

   def reset(self):
      """call when the env resets to clear committed state."""
      self._reset_commit()

   def _reset_commit(self):
      self.committed_shape  = None
      self.committed_target = None
      self.phase            = _DONE   # forces _explore on first act() call
      self._group_zones     = {}      # cleared each episode — new zone assignment

   def act(self, obs=None) -> np.ndarray:
      """
      return action [dx, dy, grip] in [-1, 1]^3.
      obs is accepted but unused — oracle reads env state directly.
      """
      env  = self.env
      task = env.goal.get("task", "arrange_in_sequence")

      # --- idle if already good enough ---
      if env._compute_task_score() >= IDLE_THRESHOLD:
         return np.array([0.0, 0.0, -1.0], dtype=np.float32)

      # --- explore: pick next shape if needed ---
      # re-explore when: no committed shape, phase is _DONE (grip just released),
      # or local condition confirms the shape landed correctly
      if (self.committed_shape is None
            or self.phase == _DONE
            or self._local_condition_met(self.committed_shape)):
         idx, target = self._explore(task)
         self.committed_shape  = idx
         self.committed_target = target
         self.phase            = _NAVIGATE

      # --- execute current sub-phase ---
      action = self._execute_phase()

      if self.noise_std > 0:
         action[0] += float(self.rng.normal(0, self.noise_std))
         action[1] += float(self.rng.normal(0, self.noise_std))
         action     = np.clip(action, -1.0, 1.0)

      return action.astype(np.float32)

   # -------------------------------------------------------------------------
   # sub-phase execution
   # -------------------------------------------------------------------------

   def _execute_phase(self) -> np.ndarray:
      """
      execute the current sub-phase and advance phase when complete.
      returns [dx, dy, grip].
      """
      env     = self.env
      idx     = self.committed_shape
      tx, ty  = self.committed_target
      s       = env.shapes[idx]
      cx, cy  = env.cx, env.cy

      if self.phase == _NAVIGATE:
         task = env.goal.get("task", "arrange_in_sequence")
         if task == "reach":
            # reach: navigate until cursor is within GRIP_RADIUS, no grip needed
            dist_to_shape = np.sqrt((cx - s.x) ** 2 + (cy - s.y) ** 2)
            if dist_to_shape <= GRIP_RADIUS:
               # stay put — score should now be 1.0, idle fires next act()
               return np.array([0.0, 0.0, -1.0], dtype=np.float32)
            dx, dy = self._direction_to(cx, cy, s.x, s.y)
            return np.array([dx, dy, -1.0], dtype=np.float32)
         # all other tasks: navigate to shape then grip
         dist_to_shape = np.sqrt((cx - s.x) ** 2 + (cy - s.y) ** 2)
         if dist_to_shape <= APPROACH_DIST:
            # arrived — advance to grip_on
            self.phase = _GRIP_ON
            return np.array([0.0, 0.0, 1.0], dtype=np.float32)
         dx, dy = self._direction_to(cx, cy, s.x, s.y)
         return np.array([dx, dy, -1.0], dtype=np.float32)

      elif self.phase == _GRIP_ON:
         # activate grip for one step, then start dragging
         self.phase = _DRAG
         return np.array([0.0, 0.0, 1.0], dtype=np.float32)

      elif self.phase == _DRAG:
         # drag cursor (and grabbed shape) toward target
         dist_to_target = np.sqrt((cx - tx) ** 2 + (cy - ty) ** 2)
         if dist_to_target < CURSOR_SPEED * 0.8:
            # close enough — release
            self.phase = _GRIP_OFF
            return np.array([0.0, 0.0, -1.0], dtype=np.float32)
         dx, dy = self._direction_to(cx, cy, tx, ty)
         return np.array([dx, dy, 1.0], dtype=np.float32)

      elif self.phase == _GRIP_OFF:
         # one-step release — advance to _DONE so act() always re-explores
         # next call, whether or not the shape landed on target
         self.phase = _DONE
         return np.array([0.0, 0.0, -1.0], dtype=np.float32)

      # fallback
      return np.array([0.0, 0.0, -1.0], dtype=np.float32)

   @staticmethod
   def _direction_to(fx, fy, tx, ty) -> tuple:
      """unit vector from (fx, fy) toward (tx, ty), scaled to [-1, 1]."""
      dx   = tx - fx
      dy   = ty - fy
      dist = np.sqrt(dx ** 2 + dy ** 2)
      if dist < 1e-6:
         return 0.0, 0.0
      return float(dx / dist), float(dy / dist)

   # -------------------------------------------------------------------------
   # explore phase
   # -------------------------------------------------------------------------

   def _explore(self, task: str):
      """
      select next shape and compute its target.
      returns (shape_idx, (tx, ty)).
      """
      env       = self.env
      n         = env.n_shapes
      attribute = env.goal.get("attribute", "none")

      if task in ("reach", "touch", "drag"):
         # use the env's pre-computed target index (respects target_color/type)
         idx    = env.target_idx
         target = self._compute_target(idx, task)
         return idx, target

      if task in ("arrange_in_sequence", "arrange_in_line"):
         priorities = self._priorities_sequence(task)
      elif task == "arrange_in_region":
         priorities = self._priorities_region()
      elif task == "arrange_in_groups":
         priorities = self._priorities_groups(attribute)
      else:
         priorities = np.ones(n)

      shape_idx = self._softmax_select(priorities)
      target    = self._compute_target(shape_idx, task)
      return shape_idx, target

   def _priorities_sequence(self, task: str) -> np.ndarray:
      """
      priority = pixel distance of each shape from its ideal slot.
      shapes far from where they need to be get picked first.
      for arrange_in_line with no attribute, slot assignment uses
      nearest-available-slot rather than attribute rank so the oracle
      doesn't microadjust already-placed shapes out of index order.
      """
      env       = self.env
      n         = env.n_shapes
      axis      = env.goal.get("axis", "x")
      direction = env.goal.get("direction", "ascending")
      attribute = env.goal.get("attribute", "none")
      pad       = MARGIN * 2

      if axis == "x":
         slots = np.linspace(pad, WINDOW_W - pad, n)
      else:
         slots = np.linspace(pad, WINDOW_H - pad, n)

      if task == "arrange_in_line" and attribute == "none":
         # nearest-available-slot: assign each shape to its closest
         # unoccupied slot so already-placed shapes aren't moved again
         positions  = np.array([s.x if axis == "x" else s.y
                                 for s in env.shapes])
         unassigned = list(range(n))
         slot_for   = [-1] * n
         # greedy: repeatedly assign the (shape, slot) pair with minimum dist
         remaining_shapes = list(range(n))
         remaining_slots  = list(range(n))
         while remaining_shapes:
            best_d = float("inf")
            best_s = best_sl = -1
            for si in remaining_shapes:
               for sl in remaining_slots:
                  d = abs(positions[si] - slots[sl])
                  if d < best_d:
                     best_d, best_s, best_sl = d, si, sl
            slot_for[best_s] = best_sl
            remaining_shapes.remove(best_s)
            remaining_slots.remove(best_sl)
         ideal_pos = np.array([slots[slot_for[i]] for i in range(n)])
      else:
         attr_vals   = env._get_attribute_values(attribute
                         if attribute != "none" else "size")
         ideal_ranks = np.argsort(np.argsort(attr_vals)).astype(float)
         if direction == "descending":
            ideal_ranks = (n - 1) - ideal_ranks
         ideal_pos = np.array([slots[int(round(ideal_ranks[i]))]
                                for i in range(n)])

      # priority = distance from current position to ideal slot on primary axis
      # + perpendicular distance from the line (for arrange_in_line)
      positions = np.array([s.x if axis == "x" else s.y for s in env.shapes])
      dist_along = np.abs(positions - ideal_pos)

      if task == "arrange_in_line":
         centre   = WINDOW_H / 2 if axis == "x" else WINDOW_W / 2
         perp_pos = np.array([s.y if axis == "x" else s.x
                               for s in env.shapes])
         dist_perp = np.abs(perp_pos - centre)
         priority = dist_along + dist_perp
      else:
         priority = dist_along

      return priority + 0.1   # ensure no zero weights for softmax

   def _priorities_region(self) -> np.ndarray:
      env      = self.env
      region   = env.goal.get("region", "left")
      boundary = REGION_INNER[region]

      def dist_outside(s):
         if region == "left":   return max(s.x - boundary, 0)
         elif region == "right": return max(boundary - s.x, 0)
         elif region == "top":  return max(s.y - boundary, 0)
         else:                  return max(boundary - s.y, 0)

      return np.array([dist_outside(s) for s in env.shapes]) + 0.1

   def _priorities_groups(self, attribute: str) -> np.ndarray:
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
         nn_correct = (1.0 if (not same_d or not diff_d
                               or min(same_d) < min(diff_d)) else 0.0)
         sep = min(min(diff_d) / half_diag, 1.0) if diff_d else 1.0
         scores.append(0.6 * nn_correct + 0.4 * sep)

      return 1.0 - np.array(scores) + 0.1

   def _softmax_select(self, priorities: np.ndarray) -> int:
      logits  = priorities / EXPLORE_TEMP
      logits -= logits.max()
      weights = np.exp(logits)
      weights = weights / weights.sum()
      return int(self.rng.choice(len(priorities), p=weights))

   # -------------------------------------------------------------------------
   # target computation
   # -------------------------------------------------------------------------

   def _compute_target(self, shape_idx: int, task: str) -> tuple:
      env       = self.env
      n         = env.n_shapes
      axis      = env.goal.get("axis", "x")
      direction = env.goal.get("direction", "ascending")
      attribute = env.goal.get("attribute", "none")
      region    = env.goal.get("region", "left")
      pad       = MARGIN * 2

      if task == "reach":
         # target is the shape itself — just navigate to it
         s = env.shapes[shape_idx]
         return (s.x, s.y)

      elif task == "touch":
         s = env.shapes[shape_idx]
         return (s.x, s.y)

      elif task == "drag":
         # drag shapes[0] into the target region — aim well past the boundary
         # so the shape lands comfortably inside, not just over the threshold
         boundary = REGION_INNER[region]
         extra    = float(self.rng.uniform(REGION_EXTRA_DEPTH * 0.5,
                                           REGION_EXTRA_DEPTH * 2.0))
         s        = env.shapes[shape_idx]
         if region == "left":
            return (float(np.clip(boundary - extra, pad, WINDOW_W - pad)), s.y)
         elif region == "right":
            return (float(np.clip(boundary + extra, pad, WINDOW_W - pad)), s.y)
         elif region == "top":
            return (s.x, float(np.clip(boundary - extra, pad, WINDOW_H - pad)))
         else:
            return (s.x, float(np.clip(boundary + extra, pad, WINDOW_H - pad)))

      elif task in ("arrange_in_sequence", "arrange_in_line"):
         pad = MARGIN * 2

         if axis == "x":
            slots = np.linspace(pad, WINDOW_W - pad, n)
         else:
            slots = np.linspace(pad, WINDOW_H - pad, n)

         if task == "arrange_in_line" and attribute == "none":
            # nearest-available-slot: same greedy assignment as priorities
            # so the target is consistent with which slot was chosen
            positions        = np.array([s.x if axis == "x" else s.y
                                          for s in env.shapes])
            remaining_shapes = list(range(n))
            remaining_slots  = list(range(n))
            slot_for         = [-1] * n
            while remaining_shapes:
               best_d = float("inf")
               best_s = best_sl = -1
               for si in remaining_shapes:
                  for sl in remaining_slots:
                     d = abs(positions[si] - slots[sl])
                     if d < best_d:
                        best_d, best_s, best_sl = d, si, sl
               slot_for[best_s] = best_sl
               remaining_shapes.remove(best_s)
               remaining_slots.remove(best_sl)
            tx = float(slots[slot_for[shape_idx]])
         else:
            attr_vals   = env._get_attribute_values(attribute
                            if attribute != "none" else "size")
            ideal_ranks = np.argsort(np.argsort(attr_vals)).astype(float)
            if direction == "descending":
               ideal_ranks = (n - 1) - ideal_ranks
            ideal_rank  = ideal_ranks[shape_idx]
            tx          = float(slots[int(round(ideal_rank))])

         if axis == "x":
            ty = WINDOW_H / 2 if task == "arrange_in_line" else env.shapes[shape_idx].y
         else:
            ty = tx
            tx = WINDOW_W / 2 if task == "arrange_in_line" else env.shapes[shape_idx].x

         # for arrange_in_line, only jitter along the primary axis —
         # perpendicular jitter scatters shapes off the line and tanks
         # the spread score. for sequence, jitter both axes freely.
         if task == "arrange_in_line":
            if axis == "x":
               jx = float(self.rng.uniform(-COMMIT_JITTER, COMMIT_JITTER))
               jy = 0.0
            else:
               jx = 0.0
               jy = float(self.rng.uniform(-COMMIT_JITTER, COMMIT_JITTER))
         else:
            jx = float(self.rng.uniform(-COMMIT_JITTER, COMMIT_JITTER))
            jy = float(self.rng.uniform(-COMMIT_JITTER, COMMIT_JITTER))
         return (float(np.clip(tx + jx, pad, WINDOW_W - pad)),
                 float(np.clip(ty + jy, pad, WINDOW_H - pad)))

      elif task == "arrange_in_region":
         boundary = REGION_INNER[region]
         extra    = float(self.rng.uniform(REGION_EXTRA_DEPTH * 0.5,
                                           REGION_EXTRA_DEPTH * 2.0))
         s        = env.shapes[shape_idx]
         if region == "left":
            return (float(np.clip(boundary - extra, pad, WINDOW_W - pad)), s.y)
         elif region == "right":
            return (float(np.clip(boundary + extra, pad, WINDOW_W - pad)), s.y)
         elif region == "top":
            return (s.x, float(np.clip(boundary - extra, pad, WINDOW_H - pad)))
         else:
            return (s.x, float(np.clip(boundary + extra, pad, WINDOW_H - pad)))

      elif task == "arrange_in_groups":
         return self._compute_group_target(shape_idx, attribute)

      return (env.shapes[shape_idx].x, env.shapes[shape_idx].y)

   # candidate zone centres spread across the canvas — enough for up to 6
   # distinct attribute values. zones are picked greedily so each new group
   # lands as far as possible from already-assigned groups.
   _ZONE_CANDIDATES = [
      (200, 150), (600, 150), (200, 450), (600, 450),
      (400, 300), (100, 300), (700, 300),
   ]

   def _compute_group_target(self, shape_idx: int, attribute: str) -> tuple:
      """
      zone pre-assignment strategy:
         1. build the set of unique attribute values present in the episode.
         2. for each value not yet assigned a zone, pick the candidate zone
            that maximises the minimum distance to all already-assigned zones
            (greedy maximin). this mirrors how a human would plan: "put this
            group here, that group over there, as far apart as possible."
         3. if a new attribute value appears that has no zone yet, assign it
            the candidate furthest from every zone that is already taken.
         4. target = zone centre + small jitter (both axes — shapes within
            a group should cluster loosely, not sit on a single pixel).
      """
      env = self.env
      pad = MARGIN * 2

      def get_attr(s):
         return s.color_name if attribute == "color" else s.shape_type

      # discover all unique attribute values present right now
      unique_attrs = sorted(set(get_attr(s) for s in env.shapes))

      # lazily build / extend the zone assignment dict stored on the oracle
      if not hasattr(self, "_group_zones"):
         self._group_zones = {}   # attr_value -> (cx, cy)

      candidates = list(self._ZONE_CANDIDATES)

      # assign zones to any attribute values not yet in the dict,
      # in sorted order so assignment is deterministic given the same
      # set of attributes
      for attr in unique_attrs:
         if attr in self._group_zones:
            continue
         # find the candidate zone that is furthest from all already-assigned
         # zones (maximin distance). if no zones assigned yet, pick index 0.
         assigned = list(self._group_zones.values())
         if not assigned:
            best = candidates[0]
         else:
            best      = None
            best_dist = -1.0
            for cand in candidates:
               if cand in assigned:   # don't give two groups the same zone
                  continue
               # minimum distance from this candidate to any assigned zone
               min_d = min(
                  np.sqrt((cand[0] - az[0]) ** 2 + (cand[1] - az[1]) ** 2)
                  for az in assigned
               )
               if min_d > best_dist:
                  best_dist = min_d
                  best      = cand
         self._group_zones[attr] = best

      my_attr = get_attr(env.shapes[shape_idx])
      cx, cy  = self._group_zones[my_attr]

      jx = float(self.rng.uniform(-COMMIT_JITTER, COMMIT_JITTER))
      jy = float(self.rng.uniform(-COMMIT_JITTER, COMMIT_JITTER))
      return (float(np.clip(cx + jx, pad, WINDOW_W - pad)),
              float(np.clip(cy + jy, pad, WINDOW_H - pad)))

   # -------------------------------------------------------------------------
   # local completion conditions
   # -------------------------------------------------------------------------

   def _local_condition_met(self, shape_idx: int) -> bool:
      """
      True when the committed shape has satisfied its local condition and
      the oracle should explore for the next shape to move.
      conditions are checked only after grip has been released (_GRIP_OFF phase).
      """
      if self.committed_target is None:
         return True

      env  = self.env
      task = env.goal.get("task", "arrange_in_sequence")

      # reach never grips so it stays in _NAVIGATE — check proximity directly
      if task == "reach":
         s    = env.shapes[shape_idx]
         dist = np.sqrt((env.cx - s.x) ** 2 + (env.cy - s.y) ** 2)
         return dist <= GRIP_RADIUS

      # don't interrupt mid-drag — only check once grip is fully released
      if self.phase in (_NAVIGATE, _GRIP_ON, _DRAG):
         return False
      # _GRIP_OFF: check whether the shape landed close enough to the target
      if self.committed_target is None:
         return True

      s      = env.shapes[shape_idx]
      tx, ty = self.committed_target
      dist   = np.sqrt((s.x - tx) ** 2 + (s.y - ty) ** 2)

      if task == "touch":
         # touch: complete after grip_off (grip activated over shape)
         return True

      if task in ("arrange_in_sequence", "drag"):
         return dist < CURSOR_SPEED * 2.0

      elif task == "arrange_in_line":
         # shape must be near its rank slot AND on the line (perpendicular
         # within LINE_SPREAD_THRESHOLD / n_shapes of centre)
         axis     = env.goal.get("axis", "x")
         centre   = WINDOW_H / 2 if axis == "x" else WINDOW_W / 2
         perp     = s.y if axis == "x" else s.x
         perp_ok  = abs(perp - centre) < LINE_SPREAD_THRESHOLD / max(env.n_shapes, 1)
         return dist < CURSOR_SPEED * 2.0 and perp_ok

      elif task == "arrange_in_region":
         region   = env.goal.get("region", "left")
         boundary = REGION_INNER[region]
         in_region = (
            (region == "left"   and s.x <= boundary) or
            (region == "right"  and s.x >= boundary) or
            (region == "top"    and s.y <= boundary) or
            (region == "bottom" and s.y >= boundary)
         )
         return in_region and dist < CURSOR_SPEED * 2.0

      elif task == "arrange_in_groups":
         return dist < CURSOR_SPEED * 2.5

      return dist < CURSOR_SPEED * 2.0


# ---------------------------------------------------------------------------
# demonstration collection
# ---------------------------------------------------------------------------

DEMO_SAVE_PATH = "./logs/oracle_demos.npz"


def collect_demonstrations(
   n_episodes: int   = 500,
   noise_std:  float = NOISE_STD,
   verbose:    bool  = True,
   save_path:  str   = DEMO_SAVE_PATH,
   force:      bool  = False,
) -> dict:
   """
   run the oracle across TASK_POOL and collect (obs, action) pairs plus
   high-level annotations for future HBC training.

   if save_path exists and force=False, loads from disk instead of
   re-collecting. set force=True to always re-collect.

   dataset keys:
      "observations":  np.ndarray (N, obs_dim)
      "actions":       np.ndarray (N, 3)   [dx, dy, grip]
      "hl_shape_idx":  np.ndarray (N,)     high-level: which shape was targeted
      "hl_target_x":   np.ndarray (N,)     high-level: target x in pixels
      "hl_target_y":   np.ndarray (N,)     high-level: target y in pixels
   """
   from llm_goal_parser import parse_goal, get_embedding
   import torch
   from bc_train import GoalEncoder

   # --- load from disk if available ---
   if not force and os.path.exists(save_path):
      if verbose:
         print(f"\n--- loading oracle demos from {save_path} ---")
      data = np.load(save_path)
      dataset = {k: data[k] for k in data.files}
      if verbose:
         print(f"  loaded {len(dataset['observations']):,} transitions")
      return dataset

   goal_encoder = GoalEncoder()
   goal_encoder.eval()
   rng = np.random.default_rng()

   all_obs        = []
   all_actions    = []
   all_hl_shape   = []
   all_hl_target_x = []
   all_hl_target_y = []
   n_solved       = 0

   if verbose:
      print(f"\n--- collecting {n_episodes} oracle demonstrations "
            f"across {len(TASK_POOL)} task pool prompts ---")

   for ep in range(1, n_episodes + 1):
      prompt = rng.choice(TASK_POOL)
      goal   = parse_goal(prompt)
      n_shp  = int(rng.integers(2, MAX_SHAPES + 1))

      raw_emb = get_embedding(prompt)
      with torch.no_grad():
         emb_t    = torch.tensor(raw_emb, dtype=torch.float32).unsqueeze(0)
         encoding = goal_encoder(emb_t).squeeze(0).numpy()

      env    = ShapeEnv(n_shapes=n_shp, goal=goal)
      env.set_goal_encoding(encoding)
      oracle = OraclePolicy(env, noise_std=noise_std, rng=rng)
      obs, _ = env.reset()
      oracle.reset()

      ep_obs      = []
      ep_actions  = []
      ep_hl_shape = []
      ep_hl_tx    = []
      ep_hl_ty    = []
      done        = False

      while not done:
         # record high-level annotation before act() potentially re-explores
         hl_idx = oracle.committed_shape if oracle.committed_shape is not None else -1
         hl_tx  = oracle.committed_target[0] if oracle.committed_target else 0.0
         hl_ty  = oracle.committed_target[1] if oracle.committed_target else 0.0

         action = oracle.act(obs)

         ep_obs.append(obs.copy())
         ep_actions.append(action.copy())
         ep_hl_shape.append(hl_idx)
         ep_hl_tx.append(hl_tx)
         ep_hl_ty.append(hl_ty)

         obs, _, terminated, truncated, _ = env.step(action)
         done = terminated or truncated

      if terminated:
         n_solved += 1
         all_obs.extend(ep_obs)
         all_actions.extend(ep_actions)
         all_hl_shape.extend(ep_hl_shape)
         all_hl_target_x.extend(ep_hl_tx)
         all_hl_target_y.extend(ep_hl_ty)

      env.close()

      if verbose and ep % 50 == 0:
         sr = n_solved / ep * 100
         print(f"  episode {ep:4d}/{n_episodes}  solve rate: {sr:.1f}%")

   dataset = {
      "observations": np.array(all_obs,         dtype=np.float32),
      "actions":      np.array(all_actions,      dtype=np.float32),
      "hl_shape_idx": np.array(all_hl_shape,     dtype=np.int32),
      "hl_target_x":  np.array(all_hl_target_x,  dtype=np.float32),
      "hl_target_y":  np.array(all_hl_target_y,  dtype=np.float32),
   }

   if verbose:
      sr = n_solved / n_episodes * 100
      print(f"\n--- dataset summary ---")
      print(f"  total transitions : {len(all_obs):,}")
      print(f"  solve rate        : {sr:.1f}%")
      if n_solved > 0:
         print(f"  mean ep length    : {len(all_obs)/n_solved:.1f}")

   # save to disk for reuse
   if save_path:
      os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
      np.savez(save_path, **dataset)
      if verbose:
         print(f"  saved to {save_path}")

   return dataset
