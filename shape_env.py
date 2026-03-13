"""
shape_env.py

gymnasium environment for 2D shape manipulation via a cursor.

--- action space ---
   [dx, dy, grip]  all in [-1, 1]
   dx, dy:  cursor movement, scaled by CURSOR_SPEED pixels per step
   grip:    > GRIP_THRESHOLD -> holding = True

--- observation space (108-dim) ---
   [0-3]    cursor state: cx_norm, cy_norm, holding, grabbed_idx_norm
   [4-8]    grabbed shape features (zeros if nothing grabbed)
   [9-13]   nearest non-grabbed shape features (zeros if no shapes)
   [14-43]  all shapes zero-padded (MAX_SHAPES * OBS_VALUES_PER_SHAPE)
   [44-107] goal encoding (GOAL_ENCODING_DIM)

   left stream  (cursor-local):  indices  0-43  (44 values)
   right stream (scene-global):  indices 14-107 (94 values)

--- tasks ---
   starter:
      reach        move cursor within GRIP_RADIUS of target shape
      touch        activate grip while overlapping target shape
      drag         grip shape and move into a target region

   wave 3 (2x2x2 cube):
      arrange_in_sequence  one target space, unbounded, ordered by attribute
      arrange_in_line      one target space, bounded, ordered or unordered
      arrange_in_region    one target space, bounded, unordered
      arrange_in_groups    many target spaces, bounded, unordered

--- reward design ---
   1. task score delta        improvement in per-task score function
   2. oscillation penalty     discourages score going up then immediately down
   3. wall penalty            discourages cursor pushing against canvas border
   4. inactivity penalty      discourages zero-movement actions
   5. completion bonus        large reward when task is solved
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame

from config import (
   MAX_SHAPES, OBS_VALUES_PER_SHAPE, GOAL_ENCODING_DIM, get_obs_size,
   SHAPE_TYPES, N_SHAPE_TYPES, SHAPE_TYPE_IDX,
   CURSOR_SPEED, GRIP_THRESHOLD, GRIP_RADIUS,
   CURSOR_STATE_SIZE, FOCAL_SHAPE_SIZE,
)

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

WINDOW_W     = 800
WINDOW_H     = 600
SHAPE_RADIUS = 20
FPS          = 60
MAX_STEPS    = 500
MARGIN       = SHAPE_RADIUS * 2

SCORE_SOLVE_THRESHOLD = 0.85
STEP_PENALTY          = -0.02
COMPLETION_BONUS      = 50.0  # was 25.0 — solving must be worth more than timing out.
                               # a stuck agent loses ~30 over 500 steps; bonus was 25,
                               # meaning solve was barely better than giving up. now 50.
MOVEMENT_THRESHOLD    = 0.5   # pixels — below this cursor = not moving

COLORS = {
   "red":    (173,  46,  52),
   "green":  ( 78,  99,  30),
   "teal":   ( 87, 220, 215),
   "yellow": (199, 227,  54),
   "purple": (155,  90, 195),
}
COLOR_NAMES = list(COLORS.keys())
BG_COLOR    = (30, 30, 30)

# region boundaries (fraction of canvas) — shared by env score and oracle
REGION_INNER = {
   "left":   WINDOW_W * 0.35,
   "right":  WINDOW_W * 0.65,
   "top":    WINDOW_H * 0.35,
   "bottom": WINDOW_H * 0.65,
}

LINE_SPREAD_THRESHOLD = 120  # pixels — max allowed perpendicular spread

SUPPORTED_TASKS = [
   "reach",
   "touch",
   "drag",
   "arrange_in_sequence",
   "arrange_in_line",
   "arrange_in_region",
   "arrange_in_groups",
]

# cursor crosshair dimensions
_CURSOR_RADIUS   = 3    # circle radius (pixels)
_CURSOR_GAP      = 4    # gap between circle edge and crosshair line start
_CURSOR_ARM      = 8    # crosshair arm length (pixels)
_CURSOR_COLOR    = (220, 220, 220)
_CURSOR_COLOR_ON = (220, 220, 220)   # same color — grip shown by fill only


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------

class Shape:
   """a single movable shape. can be circle, square, or triangle."""

   def __init__(self, shape_id, x, y, size, color_name, shape_type="circle"):
      self.shape_id   = shape_id
      self.x          = float(x)
      self.y          = float(y)
      self.size       = float(size)
      self.color_name = color_name
      self.color_rgb  = COLORS[color_name]
      self.radius     = int(SHAPE_RADIUS * size)
      self.shape_type = shape_type   # "circle" | "square" | "triangle"

   def draw(self, surface, font):
      cx = int(self.x)
      cy = int(self.y)
      r  = self.radius

      if self.shape_type == "circle":
         pygame.draw.circle(surface, self.color_rgb, (cx, cy), r)
      elif self.shape_type == "square":
         rect = pygame.Rect(cx - r, cy - r, r * 2, r * 2)
         pygame.draw.rect(surface, self.color_rgb, rect)
      elif self.shape_type == "triangle":
         pts = [(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)]
         pygame.draw.polygon(surface, self.color_rgb, pts)

      label = font.render(f"{self.size:.1f}", True, (255, 255, 255))
      surface.blit(label, (cx - 10, cy - 8))

   def as_obs(self) -> np.ndarray:
      """5-value obs: x_norm, y_norm, size_norm, color_norm, shape_type_norm."""
      return np.array([
         self.x / WINDOW_W,
         self.y / WINDOW_H,
         (self.size - 0.5) / 1.5,
         COLOR_NAMES.index(self.color_name) / max(len(COLOR_NAMES) - 1, 1),
         SHAPE_TYPE_IDX.get(self.shape_type, 0) / max(N_SHAPE_TYPES - 1, 1),
      ], dtype=np.float32)


# ---------------------------------------------------------------------------
# ShapeEnv
# ---------------------------------------------------------------------------

class ShapeEnv(gym.Env):
   """
   gymnasium environment for 2D shape manipulation via a cursor.
   accepts a goal dict produced by llm_goal_parser.parse_goal().
   """

   metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

   def __init__(self, n_shapes: int = None, goal: dict = None,
                goal_embedding: np.ndarray = None, render_mode: str = None):
      super().__init__()

      self._fixed_n_shapes = n_shapes
      self.n_shapes        = n_shapes if n_shapes is not None else 2
      self.render_mode     = render_mode

      self.goal = goal or {
         "task":      "arrange_in_sequence",
         "axis":      "x",
         "direction": "ascending",
         "attribute": "size",
         "region":    "none",
         "bounded":   False,
      }

      self._goal_encoding = (
         goal_embedding[:GOAL_ENCODING_DIM].astype(np.float32)
         if goal_embedding is not None
         else np.zeros(GOAL_ENCODING_DIM, dtype=np.float32)
      )

      obs_size = get_obs_size()
      self.observation_space = spaces.Box(
         low=-2.0, high=2.0, shape=(obs_size,), dtype=np.float32)
      self.action_space = spaces.Box(
         low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

      # cursor state
      self.cx          = float(WINDOW_W / 2)
      self.cy          = float(WINDOW_H / 2)
      self.holding     = False
      self.grabbed_idx = -1   # index into self.shapes, -1 = nothing grabbed

      # episode state
      self.shapes           = []
      self.steps            = 0
      self.prev_score       = 0.0
      self.prev_score_delta = 0.0
      self.prev_rank_corr   = 0.0
      self.target_idx       = 0   # index of target shape for reach/touch/drag
      self.window           = None
      self.clock            = None
      self.font             = None

   # -------------------------------------------------------------------------
   # gymnasium interface
   # -------------------------------------------------------------------------

   def reset(self, seed=None, options=None):
      super().reset(seed=seed)

      if self._fixed_n_shapes is None:
         self.n_shapes = int(self.np_random.integers(2, MAX_SHAPES + 1))
      else:
         self.n_shapes = self._fixed_n_shapes

      # cursor starts at center each episode
      self.cx          = float(WINDOW_W / 2)
      self.cy          = float(WINDOW_H / 2)
      self.holding     = False
      self.grabbed_idx = -1

      self.steps            = 0
      self.prev_score_delta = 0.0
      self.shapes           = self._spawn_shapes()
      self.target_idx       = self._pick_target_idx()
      self.prev_score       = self._compute_task_score()
      self.prev_rank_corr   = self._compute_rank_corr()

      return self._get_obs(), {}

   def step(self, action):
      self.steps += 1

      dx_raw   = float(action[0])
      dy_raw   = float(action[1])
      grip_raw = float(action[2])

      # --- move cursor ---
      new_cx = float(np.clip(
         self.cx + dx_raw * CURSOR_SPEED, MARGIN, WINDOW_W - MARGIN))
      new_cy = float(np.clip(
         self.cy + dy_raw * CURSOR_SPEED, MARGIN, WINDOW_H - MARGIN))
      cursor_moved = np.sqrt(
         (new_cx - self.cx) ** 2 + (new_cy - self.cy) ** 2)
      self.cx = new_cx
      self.cy = new_cy

      # --- grip logic ---
      new_holding = grip_raw > GRIP_THRESHOLD

      if new_holding and not self.holding:
         # grip just activated — try to attach to a shape
         self.grabbed_idx = self._try_grab()

      elif not new_holding:
         # grip released
         self.grabbed_idx = -1

      self.holding = new_holding

      # --- drag grabbed shape ---
      if self.holding and self.grabbed_idx >= 0:
         s   = self.shapes[self.grabbed_idx]
         s.x = float(np.clip(self.cx, MARGIN, WINDOW_W - MARGIN))
         s.y = float(np.clip(self.cy, MARGIN, WINDOW_H - MARGIN))

      # --- wall penalty: cursor tried to move but hit the margin ---
      intended_move = np.sqrt(
         (dx_raw * CURSOR_SPEED) ** 2 + (dy_raw * CURSOR_SPEED) ** 2)
      wall_penalty = (
         -0.05 if intended_move > 5.0 and cursor_moved < intended_move * 0.25
         else 0.0)

      # --- inactivity penalty ---
      # was -0.04, raised to -0.10 so doing nothing is more costly than
      # trying and failing. at -0.04 a stuck agent loses ~20 over 500 steps,
      # less than the risk of oscillation from thrashing. at -0.10 the cost
      # of inaction (~50) clearly exceeds the cost of trying.
      inactivity_penalty = (
         -0.10 if cursor_moved < MOVEMENT_THRESHOLD else 0.0)

      # --- task score delta ---
      new_score    = self._compute_task_score()
      score_delta  = new_score - self.prev_score
      score_reward = score_delta * 10.0

      # --- rank / cohesion delta ---
      new_rank_corr = self._compute_rank_corr()
      rank_delta    = new_rank_corr - self.prev_rank_corr
      rank_reward   = rank_delta * 2.0

      # oscillation penalty removed — it discouraged exploratory backtracking
      # which is needed for the agent to find the grip window and recover from
      # wrong-direction moves. the step penalty and inactivity penalty are
      # sufficient to discourage genuinely wasted motion.
      oscillation_penalty = 0.0

      # --- grip-on-target bonus ---
      # reward gripping the target shape regardless of where it moves.
      # without this, touch/drag have zero reward signal until the shape
      # actually reaches its destination — the prerequisite behavior (grip)
      # is invisible to the reward function. small bonus so it does not
      # dominate the task score signal.
      task = self.goal.get("task", "reach")
      grip_bonus = 0.0
      if task in ("touch", "drag") and self.holding:
         if self.grabbed_idx == self.target_idx:
            grip_bonus = 0.10

      # bookkeeping
      self.prev_score_delta = score_delta
      self.prev_score       = new_score
      self.prev_rank_corr   = new_rank_corr

      reward = (score_reward + rank_reward + oscillation_penalty
                + wall_penalty + inactivity_penalty + grip_bonus + STEP_PENALTY)

      terminated = self._is_solved()
      if terminated:
         reward += COMPLETION_BONUS

      truncated = self.steps >= MAX_STEPS
      obs       = self._get_obs()
      info      = {
         "score":     new_score,
         "rank_corr": new_rank_corr,
         "steps":     self.steps,
         "task":      self.goal["task"],
      }

      if self.render_mode == "human":
         self._render_frame()

      return obs, reward, terminated, truncated, info

   def render(self):
      if self.render_mode in ("human", "rgb_array"):
         return self._render_frame()

   def close(self):
      if self.window is not None:
         pygame.display.quit()
         pygame.quit()
         self.window = None

   # -------------------------------------------------------------------------
   # cursor helpers
   # -------------------------------------------------------------------------

   def _try_grab(self) -> int:
      """
      attempt to grab the closest shape within GRIP_RADIUS of the cursor.
      returns the shape index if found, else -1.
      if multiple shapes overlap, picks the closest (approximates topmost).
      """
      best_idx  = -1
      best_dist = float("inf")
      for i, s in enumerate(self.shapes):
         dist = np.sqrt((self.cx - s.x) ** 2 + (self.cy - s.y) ** 2)
         if dist <= GRIP_RADIUS and dist < best_dist:
            best_dist = dist
            best_idx  = i
      return best_idx

   def _nearest_non_grabbed(self) -> int:
      """return index of nearest shape that isn't currently grabbed, or -1."""
      best_idx  = -1
      best_dist = float("inf")
      for i, s in enumerate(self.shapes):
         if i == self.grabbed_idx:
            continue
         dist = np.sqrt((self.cx - s.x) ** 2 + (self.cy - s.y) ** 2)
         if dist < best_dist:
            best_dist = dist
            best_idx  = i
      return best_idx

   # -------------------------------------------------------------------------
   # spawn
   # -------------------------------------------------------------------------

   def _spawn_shapes(self) -> list:
      rng       = self.np_random
      task      = self.goal.get("task", "arrange_in_sequence")
      attribute = self.goal.get("attribute", "none")
      tc        = self.goal.get("target_color", "none")
      tt        = self.goal.get("target_type",  "none")

      color_indices = rng.integers(0, len(COLOR_NAMES), size=self.n_shapes)
      if task == "arrange_in_groups" and attribute == "color" and self.n_shapes >= 2:
         # need at least 2 distinct colors (so the task is non-trivial) and at
         # least 2 shapes sharing a color (so grouping is learnable, not just
         # "spread everything out"). all-same is the degenerate case the old
         # code missed — it only guarded against all-different.
         n_colors = len(COLOR_NAMES)
         if len(set(color_indices.tolist())) < 2:
            # all same — force shape[1] to a different color
            color_indices[1] = (int(color_indices[0]) + 1) % n_colors
         if len(set(color_indices.tolist())) == self.n_shapes and self.n_shapes > 2:
            # all different — force shape[1] to share shape[0]'s color
            color_indices[1] = color_indices[0]

      type_indices = rng.integers(0, N_SHAPE_TYPES, size=self.n_shapes)
      if task == "arrange_in_groups" and attribute == "shape_type" and self.n_shapes >= 2:
         if len(set(type_indices.tolist())) < 2:
            # all same — force shape[1] to a different type
            type_indices[1] = (int(type_indices[0]) + 1) % N_SHAPE_TYPES
         if len(set(type_indices.tolist())) == self.n_shapes and self.n_shapes > 2:
            # all different — force shape[1] to share shape[0]'s type
            type_indices[1] = type_indices[0]
      elif self.n_shapes >= 2 and len(set(type_indices.tolist())) < 2:
         type_indices[1] = (int(type_indices[0]) + 1) % N_SHAPE_TYPES

      # for starter tasks with a specific target: force shapes[0] to match,
      # then ensure no other shape accidentally also matches (keeps the target
      # unambiguous and avoids spawning a canvas full of red squares)
      if task in ("reach", "touch", "drag"):
         if tc not in ("none", "any"):
            target_color_idx = COLOR_NAMES.index(tc)
            color_indices[0] = target_color_idx
            for i in range(1, self.n_shapes):
               while color_indices[i] == target_color_idx:
                  color_indices[i] = rng.integers(0, len(COLOR_NAMES))
         if tt not in ("none", "any"):
            target_type_idx = SHAPE_TYPES.index(tt)
            type_indices[0] = target_type_idx
            for i in range(1, self.n_shapes):
               while type_indices[i] == target_type_idx:
                  type_indices[i] = rng.integers(0, N_SHAPE_TYPES)

      shapes = []
      for i in range(self.n_shapes):
         x          = float(rng.uniform(MARGIN, WINDOW_W - MARGIN))
         y          = float(rng.uniform(MARGIN, WINDOW_H - MARGIN))
         size       = float(rng.uniform(0.5, 2.0))
         color_name = COLOR_NAMES[int(color_indices[i])]
         shape_type = SHAPE_TYPES[int(type_indices[i])]
         shapes.append(Shape(i, x, y, size, color_name, shape_type))

      MAX_SPAWN_RETRIES = 10
      for _ in range(MAX_SPAWN_RETRIES):
         if not self._initial_score_solved(shapes):
            break
         for s in shapes:
            s.x = float(rng.uniform(MARGIN, WINDOW_W - MARGIN))
            s.y = float(rng.uniform(MARGIN, WINDOW_H - MARGIN))

      return shapes

   def _initial_score_solved(self, shapes) -> bool:
      original    = self.shapes
      self.shapes = shapes
      solved      = self._compute_task_score() >= SCORE_SOLVE_THRESHOLD
      self.shapes = original
      return solved

   def _pick_target_idx(self) -> int:
      """
      for reach/touch/drag: find the index of the shape that best matches
      target_color and target_type from the goal.

      matching priority:
        1. exact match on both color and type (if both specified)
        2. match on the one specified field
        3. any shape (fallback — covers "any" and "none")

      always returns a valid index (0 if nothing else matches).
      for non-starter tasks the value is unused but kept at 0.
      """
      task = self.goal.get("task", "none")
      if task not in ("reach", "touch", "drag"):
         return 0

      tc = self.goal.get("target_color", "none")
      tt = self.goal.get("target_type",  "none")

      # "any" / "none" → just use the first shape
      if tc in ("any", "none") and tt in ("any", "none"):
         return 0

      want_color = tc not in ("any", "none")
      want_type  = tt not in ("any", "none")

      for i, s in enumerate(self.shapes):
         color_ok = (not want_color) or (s.color_name == tc)
         type_ok  = (not want_type)  or (s.shape_type == tt)
         if color_ok and type_ok:
            return i

      # shouldn't reach here if _spawn_shapes guaranteed a match, but fall back
      return 0

   # -------------------------------------------------------------------------
   # obs and goal encoding
   # -------------------------------------------------------------------------

   def _get_obs(self) -> np.ndarray:
      # [0-3] cursor state
      grabbed_norm = (self.grabbed_idx / max(self.n_shapes - 1, 1)
                      if self.grabbed_idx >= 0 else -1.0)
      cursor_state = np.array([
         self.cx / WINDOW_W * 2.0 - 1.0,   # -1 to 1
         self.cy / WINDOW_H * 2.0 - 1.0,   # -1 to 1
         1.0 if self.holding else 0.0,
         float(grabbed_norm),
      ], dtype=np.float32)

      # [4-8] grabbed shape features (zeros if nothing grabbed)
      if self.grabbed_idx >= 0:
         grabbed_feats = self.shapes[self.grabbed_idx].as_obs()
      else:
         grabbed_feats = np.zeros(OBS_VALUES_PER_SHAPE, dtype=np.float32)

      # [9-13] nearest non-grabbed shape features (zeros if no shapes)
      nearest_idx = self._nearest_non_grabbed()
      if nearest_idx >= 0:
         nearest_feats = self.shapes[nearest_idx].as_obs()
      else:
         nearest_feats = np.zeros(OBS_VALUES_PER_SHAPE, dtype=np.float32)

      # [14-43] all shapes zero-padded
      active_obs = np.concatenate(
         [self.shapes[i].as_obs() for i in range(self.n_shapes)])
      n_padding  = MAX_SHAPES - self.n_shapes
      padding    = np.zeros(n_padding * OBS_VALUES_PER_SHAPE, dtype=np.float32)
      all_shapes = np.concatenate([active_obs, padding])

      # [44-107] goal encoding
      return np.concatenate([
         cursor_state,
         grabbed_feats,
         nearest_feats,
         all_shapes,
         self._goal_encoding,
      ]).astype(np.float32)

   def set_goal_encoding(self, encoding: np.ndarray):
      assert encoding.shape == (GOAL_ENCODING_DIM,), (
         f"expected ({GOAL_ENCODING_DIM},), got {encoding.shape}")
      self._goal_encoding = encoding.astype(np.float32)

   # -------------------------------------------------------------------------
   # score dispatch
   # -------------------------------------------------------------------------

   def _compute_task_score(self) -> float:
      task = self.goal.get("task", "arrange_in_sequence")
      if task == "reach":
         return self._score_reach()
      elif task == "touch":
         return self._score_touch()
      elif task == "drag":
         return self._score_drag()
      elif task == "arrange_in_sequence":
         return self._score_arrange_in_sequence()
      elif task == "arrange_in_line":
         return self._score_arrange_in_line()
      elif task == "arrange_in_region":
         return self._score_arrange_in_region()
      elif task == "arrange_in_groups":
         return self._score_arrange_in_groups()
      return 0.0

   def _compute_rank_corr(self) -> float:
      return self._compute_task_score()

   def _compute_score(self) -> float:
      return self._compute_task_score()

   def _is_solved(self) -> bool:
      return self._compute_task_score() >= SCORE_SOLVE_THRESHOLD

   # -------------------------------------------------------------------------
   # starter task score functions
   # -------------------------------------------------------------------------

   def _score_reach(self) -> float:
      """
      score for reach: how close is the cursor to the target shape?

      uses a two-zone linear function so the reward gradient is continuous
      all the way into GRIP_RADIUS:
        - outside 2*GRIP_RADIUS: linear 0.0 -> 0.7 as dist shrinks from
          ref_dist (half canvas) to 2*GRIP_RADIUS
        - inside  2*GRIP_RADIUS: linear 0.7 -> 1.0 as dist shrinks to 0,
          so the agent always has gradient signal when approaching.

      previous version capped at 0.75 which left a dead zone just outside
      GRIP_RADIUS — no gradient to push the cursor the final few pixels.
      """
      if not self.shapes:
         return 0.0
      target = self.shapes[self.target_idx]
      dist   = np.sqrt((self.cx - target.x) ** 2 + (self.cy - target.y) ** 2)
      if dist <= GRIP_RADIUS:
         return 1.0
      ref_dist    = WINDOW_W / 2.0
      near_thresh = GRIP_RADIUS * 2.0
      if dist <= near_thresh:
         # linear 0.7 -> 1.0 as dist goes from near_thresh -> 0
         # (1.0 is only returned when dist <= GRIP_RADIUS above)
         t = (near_thresh - dist) / near_thresh
         return float(0.7 + 0.29 * t)
      # linear 0.0 -> 0.7 as dist goes from ref_dist -> near_thresh
      t = 1.0 - min((dist - near_thresh) / (ref_dist - near_thresh), 1.0)
      return float(0.7 * t)

   def _score_touch(self) -> float:
      """
      score for touch: grip must be active while overlapping the target shape.
      solved when holding and overlapping target.

      score uses the same two-zone proximity shape as reach (0->0.7 far,
      0.7->0.99 near), so the agent has gradient signal all the way in.
      the grip jump to 1.0 only fires when actually over the target AND
      holding — no discontinuous gap in the score range.

      old version: 0.5*proximity + 0.5*grip_on produced scores in [0, 0.5]
      without grip and a jump to 1.0 with grip. the range (0.5, 1.0) was
      unreachable, so there was no gradient signal connecting proximity to
      the grip action. the agent learned to hover at ~0.5 and stop.
      """
      if not self.shapes:
         return 0.0
      target      = self.shapes[self.target_idx]
      dist        = np.sqrt((self.cx - target.x) ** 2 + (self.cy - target.y) ** 2)
      if self.holding and dist <= GRIP_RADIUS:
         return 1.0
      # same two-zone proximity as reach — continuous gradient to the goal
      ref_dist    = WINDOW_W / 2.0
      near_thresh = GRIP_RADIUS * 2.0
      if dist <= near_thresh:
         t = (near_thresh - dist) / near_thresh
         return float(0.7 + 0.29 * t)
      t = 1.0 - min((dist - near_thresh) / (ref_dist - near_thresh), 1.0)
      return float(0.7 * t)

   def _score_drag(self) -> float:
      """
      score for drag: grip target shape and move it into the target region.
      solved when score >= SCORE_SOLVE_THRESHOLD.

      two-phase score to provide gradient signal for both prerequisites:

      phase 1 — not holding target: score cursor proximity to the shape,
         using same two-zone function as reach/touch (0->0.99). this guides
         the agent to navigate to the shape and grip it. without this, the
         only signal is shape position, which is flat until grip happens.

      phase 2 — holding target: score shape proximity to the region boundary
         via _per_shape_region_score, scaled to [0.4, 1.0] so the transition
         from phase 1 (max ~0.99) doesn't produce a negative reward jump when
         the agent first grips. 0.4 is the floor so the agent doesn't drop
         below phase 1 score just by gripping.
      """
      if not self.shapes:
         return 0.0
      target = self.shapes[self.target_idx]
      region = self.goal.get("region", "left")

      if self.holding and self.grabbed_idx == self.target_idx:
         # phase 2: shape is moving — score its position toward the region.
         # scale _per_shape_region_score (0->1) into (0.4->1.0) so gripping
         # never produces a reward cliff vs phase 1.
         region_score = self._per_shape_region_score(target, region)
         return float(0.4 + 0.6 * region_score)

      # phase 1: not holding — score cursor proximity to target shape.
      # same two-zone function as reach so the agent has gradient all the way in.
      dist        = np.sqrt((self.cx - target.x) ** 2 + (self.cy - target.y) ** 2)
      ref_dist    = WINDOW_W / 2.0
      near_thresh = GRIP_RADIUS * 2.0
      if dist <= near_thresh:
         t = (near_thresh - dist) / near_thresh
         return float(0.7 + 0.29 * t)
      t = 1.0 - min((dist - near_thresh) / (ref_dist - near_thresh), 1.0)
      return float(0.7 * t)

   # -------------------------------------------------------------------------
   # wave 3 score functions
   # -------------------------------------------------------------------------

   def _score_arrange_in_sequence(self) -> float:
      """
      per-shape score = 1 - |current_rank - ideal_rank| / (n - 1).
      episode score = mean per-shape score.
      """
      n         = self.n_shapes
      axis      = self.goal.get("axis", "x")
      direction = self.goal.get("direction", "ascending")
      attribute = self.goal.get("attribute", "size")

      if n <= 1:
         return 1.0

      positions     = [s.x if axis == "x" else s.y for s in self.shapes]
      attr_vals     = self._get_attribute_values(attribute)
      ideal_ranks   = np.argsort(np.argsort(attr_vals)).astype(float)
      if direction == "descending":
         ideal_ranks = (n - 1) - ideal_ranks
      current_ranks = np.argsort(np.argsort(positions)).astype(float)

      per_shape = [
         1.0 - abs(current_ranks[i] - ideal_ranks[i]) / (n - 1)
         for i in range(n)
      ]
      return float(np.mean(per_shape))

   def _score_arrange_in_line(self) -> float:
      """
      order score (0.6) + perpendicular spread score (0.4).
      """
      n         = self.n_shapes
      axis      = self.goal.get("axis", "x")
      attribute = self.goal.get("attribute", "none")

      if n <= 1:
         return 1.0

      if attribute != "none":
         order_score = self._score_arrange_in_sequence()
      else:
         positions = sorted(s.x if axis == "x" else s.y for s in self.shapes)
         gaps      = [positions[i+1] - positions[i] for i in range(n - 1)]
         if not gaps or max(gaps) == 0:
            order_score = 1.0
         else:
            cv          = np.std(gaps) / (np.mean(gaps) + 1e-6)
            order_score = max(0.0, 1.0 - cv)

      perp_vals    = [s.y if axis == "x" else s.x for s in self.shapes]
      spread       = max(perp_vals) - min(perp_vals)
      spread_score = max(0.0, 1.0 - spread / LINE_SPREAD_THRESHOLD)

      return 0.6 * order_score + 0.4 * spread_score

   def _score_arrange_in_region(self) -> float:
      """mean per-shape region score."""
      region = self.goal.get("region", "left")
      scores = [self._per_shape_region_score(s, region) for s in self.shapes]
      return float(np.mean(scores))

   def _per_shape_region_score(self, s, region: str) -> float:
      """
      0.7 * in_region + 0.3 * progress toward the boundary.
      a shape just past the boundary scores ~0.7.
      """
      boundary = REGION_INNER[region]
      if region == "left":
         inside   = s.x <= boundary
         progress = 1.0 - max(s.x - boundary, 0) / max(WINDOW_W - boundary, 1)
      elif region == "right":
         inside   = s.x >= boundary
         progress = 1.0 - max(boundary - s.x, 0) / max(boundary, 1)
      elif region == "top":
         inside   = s.y <= boundary
         progress = 1.0 - max(s.y - boundary, 0) / max(WINDOW_H - boundary, 1)
      else:   # bottom
         inside   = s.y >= boundary
         progress = 1.0 - max(boundary - s.y, 0) / max(boundary, 1)
      return 0.7 * float(inside) + 0.3 * float(np.clip(progress, 0.0, 1.0))

   def _score_arrange_in_groups(self) -> float:
      """
      grouping score: 0.5 * global_ratio + 0.5 * nn_isolation.

      global_ratio = inter_mean / (inter_mean + intra_mean)
         smooth ratio of mean cross-group distance to mean total distance.
         high when groups are far apart relative to their internal spread.

      nn_isolation = mean over shapes of:
         min_diff_dist / (min_diff_dist + min_same_dist)
         per-shape check that each shape's nearest neighbor is same-group.
         singletons (no same-group peers) use nominal same_dist=1px so
         they are scored purely on how far they sit from the nearest
         different-group shape — correctly penalising a lone shape that
         has wandered into another group's territory.

      blending the two catches cases the other misses:
         global_ratio alone: inflated when one distant outlier boosts
            inter_mean while a singleton sits inside another group.
         nn_isolation alone: misses global layout quality.
      """
      n         = self.n_shapes
      attribute = self.goal.get("attribute", "color")

      if n <= 1:
         return 1.0

      def get_attr(s):
         return s.color_name if attribute == "color" else s.shape_type

      def dist(a, b):
         return float(np.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2))

      # --- global ratio ---
      intra_dists = []
      inter_dists = []
      for i in range(n):
         for j in range(i + 1, n):
            d = dist(self.shapes[i], self.shapes[j])
            if get_attr(self.shapes[i]) == get_attr(self.shapes[j]):
               intra_dists.append(d)
            else:
               inter_dists.append(d)

      if not inter_dists:
         # all shapes share the same attribute value — only one group exists.
         # spawn logic should prevent this for arrange_in_groups, but if it
         # somehow occurs (n_shapes=1, or all same after retries) return 0.5
         # as a neutral score rather than 1.0 which would falsely signal solved.
         return 0.5

      intra_mean   = float(np.mean(intra_dists)) if intra_dists else 1.0
      inter_mean   = float(np.mean(inter_dists))
      global_score = inter_mean / (inter_mean + intra_mean)

      # --- per-shape nn isolation ---
      per_shape = []
      for i in range(n):
         same_d = [dist(self.shapes[i], self.shapes[j])
                   for j in range(n)
                   if j != i and get_attr(self.shapes[j]) == get_attr(self.shapes[i])]
         diff_d = [dist(self.shapes[i], self.shapes[j])
                   for j in range(n)
                   if j != i and get_attr(self.shapes[j]) != get_attr(self.shapes[i])]
         if not diff_d:
            per_shape.append(1.0)
            continue
         min_diff = min(diff_d)
         min_same = min(same_d) if same_d else 1.0   # singleton: nominal 1px
         per_shape.append(min_diff / (min_diff + min_same))

      nn_score = float(np.mean(per_shape))

      return 0.5 * global_score + 0.5 * nn_score

   # -------------------------------------------------------------------------
   # attribute helpers
   # -------------------------------------------------------------------------

   def _get_attribute_values(self, attribute: str) -> list:
      if attribute == "size":
         return [s.size for s in self.shapes]
      elif attribute == "color":
         return [COLOR_NAMES.index(s.color_name) for s in self.shapes]
      elif attribute == "shape_type":
         return [SHAPE_TYPE_IDX.get(s.shape_type, 0) for s in self.shapes]
      return list(range(self.n_shapes))

   # -------------------------------------------------------------------------
   # rendering
   # -------------------------------------------------------------------------

   def _render_frame(self):
      if self.window is None:
         pygame.init()
         if self.render_mode == "human":
            self.window = pygame.display.set_mode((WINDOW_W, WINDOW_H))
            pygame.display.set_caption("shape manipulation env")
         else:
            self.window = pygame.Surface((WINDOW_W, WINDOW_H))
         self.clock = pygame.time.Clock()
         self.font  = pygame.font.SysFont("monospace", 12)

      self.window.fill(BG_COLOR)

      for shape in self.shapes:
         shape.draw(self.window, self.font)

      self._draw_cursor()

      score = self._compute_score()
      goal  = self.goal
      task  = goal["task"]
      hud   = (f"task: {task} | attr: {goal.get('attribute','none')} | "
               f"axis: {goal.get('axis','none')} | "
               f"region: {goal.get('region','none')}   "
               f"progress: {score:.2%}   step: {self.steps}")
      self.window.blit(
         self.font.render(hud, True, (200, 200, 200)), (10, 10))

      if self.render_mode != "human":
         return np.transpose(
            pygame.surfarray.array3d(self.window), axes=(1, 0, 2))
      return None

   def _draw_cursor(self):
      """
      draw cursor as a small circle with crosshairs.
         |
      - o -
         |
      circle is filled when grip is active, outline when not.
      """
      cx  = int(self.cx)
      cy  = int(self.cy)
      r   = _CURSOR_RADIUS
      gap = _CURSOR_GAP
      arm = _CURSOR_ARM
      col = _CURSOR_COLOR_ON if self.holding else _CURSOR_COLOR

      if self.holding:
         pygame.draw.circle(self.window, col, (cx, cy), r)
      else:
         pygame.draw.circle(self.window, col, (cx, cy), r, 1)

      # up
      pygame.draw.line(self.window, col,
                       (cx, cy - r - gap), (cx, cy - r - gap - arm))
      # down
      pygame.draw.line(self.window, col,
                       (cx, cy + r + gap), (cx, cy + r + gap + arm))
      # left
      pygame.draw.line(self.window, col,
                       (cx - r - gap, cy), (cx - r - gap - arm, cy))
      # right
      pygame.draw.line(self.window, col,
                       (cx + r + gap, cy), (cx + r + gap + arm, cy))


# ---------------------------------------------------------------------------
# utility
# ---------------------------------------------------------------------------

def _spearman_corr(a: list, b: list) -> float:
   rank_a  = np.argsort(np.argsort(a)).astype(float)
   rank_b  = np.argsort(np.argsort(b)).astype(float)
   ra_mean = rank_a.mean()
   rb_mean = rank_b.mean()
   num     = ((rank_a - ra_mean) * (rank_b - rb_mean)).sum()
   denom   = (np.sqrt(((rank_a - ra_mean) ** 2).sum()) *
              np.sqrt(((rank_b - rb_mean) ** 2).sum()))
   if denom == 0:
      return 0.0
   return float(num / denom)
