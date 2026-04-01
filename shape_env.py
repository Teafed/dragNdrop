"""
shape_env.py

gymnasium environment for 2D shape manipulation via a cursor.

--- action space ---
   [dx, dy, grip]  all in [-1, 1]
   dx, dy:  cursor movement, scaled by CURSOR_SPEED pixels per step
   grip:    > GRIP_THRESHOLD activates grip; cursor must be within
            GRIP_RADIUS of a shape to actually grab it.
            holding is True only when grip is active AND a shape is grabbed.

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

   arrange:
      arrange_in_sequence  one target space, unbounded, ordered by attribute
      arrange_in_line      one target space, bounded, ordered or unordered
      arrange_in_region    one target space, bounded, unordered
      arrange_in_groups    many target spaces, bounded, unordered

--- target indices ---
   for starter tasks, target_indices is the list of all shape indices that
   satisfy the goal's target_color / target_type spec. solved when any one
   of them satisfies the task condition. target slots are randomized each
   episode so the agent cannot learn a positional shortcut.

--- reward design ---
   score_delta * REWARD_SCORE_SCALE     dense task progress signal
   wall_penalty                         cursor hit the canvas margin
   inactivity_penalty                   cursor barely moved
   grip_bonus                           task-specific grip incentive
   STEP_PENALTY                         constant cost per step
   COMPLETION_BONUS (in step())         one-time terminal reward on solve
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
from dataclasses import dataclass

from config import (
   MAX_SHAPES, OBS_VALUES_PER_SHAPE, GOAL_ENCODING_DIM, get_obs_size,
   get_action_size, SHAPE_TYPES, N_SHAPE_TYPES, SHAPE_TYPE_IDX,
   CURSOR_SPEED, GRIP_THRESHOLD, GRIP_RADIUS,
)

# ---------------------------------------------------------------------------
# canvas / episode constants
# ---------------------------------------------------------------------------

WINDOW_W     = 800
WINDOW_H     = 600
SHAPE_RADIUS = 20
FPS          = 60
MAX_STEPS    = 500
MARGIN       = SHAPE_RADIUS * 2

SCORE_SOLVE_THRESHOLD = 0.85
# removed: moved STEP_PENALTY, COMPLETION_BONUS to RewardConfig
# TODO: if continual movements are 0.5, they should eventually add to move
MOVEMENT_THRESHOLD    = 0.5   # cursor doesn't move if below this

COLORS = {
   "red":    (173,  46,  52),
   "green":  ( 78,  99,  30),
   "teal":   ( 87, 220, 215),
   "yellow": (199, 227,  54),
   "purple": (155,  90, 195),
}
COLOR_NAMES = list(COLORS.keys())
BG_COLOR    = (30, 30, 30)

# region boundaries: parts of canvas, shared with oracle
# TODO: add center?
REGION_INNER = {
   "left":   WINDOW_W * 0.35,
   "right":  WINDOW_W * 0.65,
   "top":    WINDOW_H * 0.35,
   "bottom": WINDOW_H * 0.65,
}

LINE_SPREAD_THRESHOLD = 120  # perpendicular speed in pixels

# TODO: add "none"? see llm_goal_parser
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


# ---------------------------------------------------------------------------
# RewardConfig
# ---------------------------------------------------------------------------

@dataclass
class RewardConfig:
   """
   all reward hyperparameters in one place.
   pass a custom instance to ShapeEnv.__init__ to override defaults.

   score_scale       multiplier on per-step score improvement
   inactivity        penalty when cursor barely moves
   wall              penalty when cursor tries to move but hits the margin
   step_penalty      constant cost per step (encourages efficiency)
   completion_bonus  one-time bonus added in step() when _is_solved() fires
   grip_grab         bonus for gripping any valid target (touch/drag)
   grip_on_target    additional bonus for gripping target within GRIP_RADIUS
                     (touch only — rewards the exact solved condition)
   """
   score_scale:      float = 10.0
   inactivity:       float = -0.10
   wall:             float = -0.05
   step_penalty:     float = -0.02
   completion_bonus: float = 50.0
   grip_grab:        float = 1.0
   grip_on_target:   float = 2.0



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
   """

   metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}
   
   def __init__(
      self,
      n_shapes:      int            = None,
      goal:          dict           = None,
      goal_embedding: np.ndarray    = None,
      render_mode:   str            = None,
      reward_config: RewardConfig   = None,
   ):
      super().__init__()

      self._fixed_n_shapes = n_shapes
      self.n_shapes        = n_shapes if n_shapes is not None else 2
      self.render_mode     = render_mode
      self.rc              = reward_config or RewardConfig()

      # TODO: in llm_goal_parser, add "none" to "task"
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
      action_size = get_action_size()
      self.observation_space = spaces.Box(
         low=-2.0, high=2.0, shape=(obs_size,), dtype=np.float32)
      # how did we choose the high and low values for these?
      self.action_space = spaces.Box(
         low=-1.0, high=1.0, shape=(action_size,), dtype=np.float32)

      # cursor state
      self.cx          = float(WINDOW_W / 2)
      self.cy          = float(WINDOW_H / 2)
      self.grip        = False
      self.prev_grip   = False
      self.holding     = False  # True only when grip active AND shape grabbed
      self.grabbed_idx = -1     # index into self.shapes, -1 = nothing grabbed

      # episode state
      self.shapes           = []
      self.steps            = 0
      self.prev_score       = 0.0
      self.target_indices = [0]  # indices of all valid target shapes
      self.window           = None
      self.clock            = None
      self.font             = None

   # -------------------------------------------------------------------------
   # gymnasium interface
   # -------------------------------------------------------------------------

   def reset(self, seed=None):
      super().reset(seed=seed)

      if self._fixed_n_shapes is None:
         self.n_shapes = int(self.np_random.integers(2, MAX_SHAPES + 1))
      else:
         self.n_shapes = self._fixed_n_shapes

      self.holding     = False
      self.grabbed_idx = -1

      self.steps            = 0
      self.shapes           = self._spawn_shapes()
      self.target_indices   = self._find_target_indices()
      self.prev_score       = self._compute_task_score()
      
      return self._get_obs(), {}

   def step(self, action):
      self.steps    += 1
      prev_score     = self.prev_score

      # TODO: rn only movement is factored into these, possibly include grip
      cursor_action, intended_action = self._apply_cursor_action(action)

      new_score  = self._compute_task_score()
      reward     = self._compute_reward(
         prev_score, new_score, cursor_action, intended_action,
         self.goal.get("task", ""))

      self.prev_score = new_score

      terminated = self._is_solved()
      if terminated:
         reward += self.rc.completion_bonus

      truncated = self.steps >= MAX_STEPS
      obs       = self._get_obs()
      info      = {
         "score":    new_score,
         "steps":    self.steps,
         "task":     self.goal["task"],
      }

      if self.render_mode == "human":
         self._render_frame()

      self.prev_grip = self.grip
      return obs, reward, terminated, truncated, info

   # TODO: in demo.py, can we use this instead of having the headless function?
   def render(self):
      if self.render_mode in ("human", "rgb_array"):
         return self._render_frame()

   def close(self):
      if self.window is not None:
         pygame.display.quit()
         pygame.quit()
         self.window = None

   # -------------------------------------------------------------------------
   # cursor mechanics
   # -------------------------------------------------------------------------

   def _apply_cursor_action(self, action) -> tuple[float, float]:
      """
      apply one action: move cursor, update grip/grab state, drag grabbed shape.
      returns (cursor_action, intended_action) in pixels, used by reward.

      holding is True only when grip action is active AND a shape is grabbed.
      gripping air (grip active, no shape nearby) leaves holding=False.
      """
      dx_raw   = float(action[0])
      dy_raw   = float(action[1])
      grip_raw = float(action[2])

      # move cursor
      new_cx = float(np.clip(
         self.cx + dx_raw * CURSOR_SPEED, MARGIN, WINDOW_W - MARGIN))
      new_cy = float(np.clip(
         self.cy + dy_raw * CURSOR_SPEED, MARGIN, WINDOW_H - MARGIN))
      cursor_action   = float(np.sqrt((new_cx - self.cx)**2 + (new_cy - self.cy)**2))
      intended_action = float(np.sqrt(
         (dx_raw * CURSOR_SPEED)**2 + (dy_raw * CURSOR_SPEED)**2))
      self.cx = new_cx
      self.cy = new_cy

      # grip logic
      self.grip = grip_raw > GRIP_THRESHOLD
      grip_edge = self.grip and not self.prev_grip   # True only on the frame grip activates
      if grip_edge:
         print("this should only print once per click")
         self.grabbed_idx = self._try_grab()
      elif not self.grip:
         self.grabbed_idx = -1
         self.holding   = self.grip and self.grabbed_idx >= 0
         self.prev_grip = self.grip
      self.holding = self.grip and self.grabbed_idx >= 0

      # drag grabbed shape with cursor
      if self.holding:
         s   = self.shapes[self.grabbed_idx]
         s.x = float(np.clip(self.cx, MARGIN, WINDOW_W - MARGIN))
         s.y = float(np.clip(self.cy, MARGIN, WINDOW_H - MARGIN))

      return cursor_action, intended_action

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
   # reward
   # -------------------------------------------------------------------------

   # TODO: use more RewardConfig items from self.rc
   def _compute_reward(
      self,
      prev_score:     float,
      new_score:      float,
      cursor_action:  float,
      intended_action:float,
      task:           str,
   ) -> float:
      score_reward = (new_score - prev_score) * self.rc.score_scale
      wall         = self._wall_penalty(cursor_action, intended_action)
      inactivity   = self.rc.inactivity if cursor_action < MOVEMENT_THRESHOLD else 0.0
      grip         = self._grip_bonus(task)
      return score_reward + wall + inactivity + grip + self.rc.step_penalty

   def _wall_penalty(self, cursor_moved: float, intended_move: float) -> float:
      """penalise when the cursor tried to move but was stopped by the margin."""
      if intended_move > 5.0 and cursor_moved < intended_move * 0.25:
         return self.rc.wall
      return 0.0

   def _grip_bonus(self, task: str) -> float:
      """
      task-specific grip incentive.

      touch / drag: reward gripping any valid target shape. for touch, add
      an extra bonus when gripping while overlapping (the exact solved state),
      so the agent has a strong signal for the specific moment that matters.

      arrangement tasks probably don't need grip bonuses, score delta already
      captures shape movement progress continuously.
      """
      if not self.holding or self.grabbed_idx < 0:
         return 0.0
      if task not in ("touch", "drag"):
         return 0.0
      if self.grabbed_idx not in self.target_indices:
         return 0.0

      bonus = self.rc.grip_grab

      if task == "touch":
         s    = self.shapes[self.grabbed_idx]
         dist = float(np.sqrt((self.cx - s.x)**2 + (self.cy - s.y)**2))
         if dist <= GRIP_RADIUS:
            bonus += self.rc.grip_on_target

      return bonus

   # -------------------------------------------------------------------------
   # spawn
   # -------------------------------------------------------------------------

   def _spawn_shapes(self) -> list:
      """
      spawn n_shapes with randomized attributes and positions.
      for starter tasks with a specific target spec, one shape is guaranteed
      to match and is placed at a random slot so agent can't exploit position.
      for drag, the target shape is spawned outside the goal region.
      positions are resampled if the episode would start already solved.
      """
      rng       = self.np_random
      task      = self.goal.get("task")
      attribute = self.goal.get("attribute", "none")
      tc        = self.goal.get("target_color", "none")
      tt        = self.goal.get("target_type",  "none")

      guaranteed_slot = self._pick_guaranteed_slot(rng, task)
      color_indices   = self._assign_colors(rng, task, attribute, tc, guaranteed_slot)
      type_indices    = self._assign_types(rng, task, attribute, tt, guaranteed_slot)
      shapes          = self._spawn_positions(rng, color_indices, type_indices)

      MAX_SPAWN_RETRIES = 10
      for _ in range(MAX_SPAWN_RETRIES):
         if not self._spawn_is_solved(shapes):
            break
         for s in shapes:
            s.x = float(rng.uniform(MARGIN, WINDOW_W - MARGIN))
            s.y = float(rng.uniform(MARGIN, WINDOW_H - MARGIN))

      # for drag, ensure target shape starts outside the goal region
      if task == "drag":
         region = self.goal.get("region")
         if region and region != "none":
            self._ensure_target_outside_region(rng, shapes, region)

      return shapes

   def _pick_guaranteed_slot(self, rng, task: str) -> int:
      """
      pick one random shape index that is guaranteed to match the goal's
      target spec. other slots may also match by chance, giving multiple
      valid targets — _find_target_indices_for discovers all of them after
      spawn. for arrangement tasks, returns 0 (unused).
      """
      if task not in ("reach", "touch", "drag"):
         return 0
      return int(rng.integers(0, self.n_shapes))

   def _assign_colors(self, rng, task, attribute, tc,
                      guaranteed_slot: int) -> np.ndarray:
      """
      assign color indices for all shapes.
      - arrange_in_groups by color: enforce at least 2 distinct colors and
        at least 2 shapes sharing a color.
      - starter tasks with a specific target color: guaranteed_slot gets the
        target color; all other slots get a different color.
      - all other cases: fully random.
      """
      n             = self.n_shapes
      color_indices = rng.integers(0, len(COLOR_NAMES), size=n)
      n_colors      = len(COLOR_NAMES)

      if task == "arrange_in_groups" and attribute == "color" and n >= 2:
         if len(set(color_indices.tolist())) < 2:
            color_indices[1] = (int(color_indices[0]) + 1) % n_colors
         if len(set(color_indices.tolist())) == n and n > 2:
            color_indices[1] = color_indices[0]

      elif task in ("reach", "touch", "drag") and tc not in ("none", "any"):
         target_color_idx               = COLOR_NAMES.index(tc)
         color_indices[guaranteed_slot] = target_color_idx

      return color_indices

   def _assign_types(self, rng, task, attribute, tt,
                     guaranteed_slot: int) -> np.ndarray:
      """
      assign shape type indices for all shapes.
      - arrange_in_groups by shape_type: enforce at least 2 distinct types
        and at least 2 shapes sharing a type.
      - general: ensure at least 2 distinct types when n >= 2.
      - starter tasks with a specific target type: guaranteed_slot gets the
        target type; all other slots get a different type.
      guaranteed_slot is shared with _assign_colors so that when both color
      and type are specified, the same shape satisfies both constraints.
      """
      n            = self.n_shapes
      type_indices = rng.integers(0, N_SHAPE_TYPES, size=n)

      if task == "arrange_in_groups" and attribute == "shape_type" and n >= 2:
         if len(set(type_indices.tolist())) < 2:
            type_indices[1] = (int(type_indices[0]) + 1) % N_SHAPE_TYPES
         if len(set(type_indices.tolist())) == n and n > 2:
            type_indices[1] = type_indices[0]
      elif n >= 2 and len(set(type_indices.tolist())) < 2:
         type_indices[1] = (int(type_indices[0]) + 1) % N_SHAPE_TYPES

      if task in ("reach", "touch", "drag") and tt not in ("none", "any"):
         target_type_idx               = SHAPE_TYPES.index(tt)
         type_indices[guaranteed_slot]  = target_type_idx

      return type_indices

   def _spawn_positions(self, rng, color_indices, type_indices) -> list:
      """build the Shape list with random positions, sizes, and assigned attributes."""
      shapes = []
      for i in range(self.n_shapes):
         x          = float(rng.uniform(MARGIN, WINDOW_W - MARGIN))
         y          = float(rng.uniform(MARGIN, WINDOW_H - MARGIN))
         size       = float(rng.uniform(0.5, 2.0))
         color_name = COLOR_NAMES[int(color_indices[i])]
         shape_type = SHAPE_TYPES[int(type_indices[i])]
         shapes.append(Shape(i, x, y, size, color_name, shape_type))
      return shapes

   # TODO: can this also be used for arrangement tasks?
   def _ensure_target_outside_region(self, rng, shapes, region: str):
      """
      for drag tasks: move any target shape that spawned inside the goal
      region to the opposite side. called after _spawn_shapes builds shapes
      and after target_indices is available via _find_target_indices.
      """
      boundary = REGION_INNER[region]
      for tidx in self._find_target_indices_for(shapes):
         target = shapes[tidx]
         for _ in range(10):
            inside = (
               (region == "left"   and target.x <= boundary) or
               (region == "right"  and target.x >= boundary) or
               (region == "top"    and target.y <= boundary) or
               (region == "bottom" and target.y >= boundary)
            )
            if not inside:
               break
            if region in ("left", "right"):
               lo       = boundary + MARGIN if region == "left" else MARGIN
               hi       = WINDOW_W - MARGIN if region == "left" else boundary - MARGIN
               target.x = float(rng.uniform(lo, hi))
            else:
               lo       = boundary + MARGIN if region == "top" else MARGIN
               hi       = WINDOW_H - MARGIN if region == "top" else boundary - MARGIN
               target.y = float(rng.uniform(lo, hi))

   def _spawn_is_solved(self, shapes) -> bool:
      """
      check if a candidate spawn would start already solved.
      temporarily swaps self.shapes and self.target_indices so the score
      and solved checkers work correctly against the candidate shapes.
      """
      orig_shapes  = self.shapes
      orig_targets = self.target_indices
      self.shapes         = shapes
      self.target_indices = self._find_target_indices_for(shapes)
      solved              = self._is_solved()
      self.shapes         = orig_shapes
      self.target_indices = orig_targets
      return solved

   # -------------------------------------------------------------------------
   # target index helpers
   # -------------------------------------------------------------------------

   def _find_target_indices(self) -> list[int]:
      """find target indices against self.shapes (called in reset)."""
      return self._find_target_indices_for(self.shapes)

   def _find_target_indices_for(self, shapes) -> list[int]:
      """
      return the list of shape indices that satisfy the goal's target spec.

      for starter tasks:
        - both color and type unspecified ("any"/"none"): all shapes are valid.
        - one or both specified: all shapes matching the spec are valid.
          (there may be multiple matching shapes, e.g. two red shapes.)

      for arrangement tasks: returns [0] as a placeholder (unused for scoring).
      """
      task = self.goal.get("task", "none")
      if task not in ("reach", "touch", "drag"):
         return [0]

      tc = self.goal.get("target_color", "none")
      tt = self.goal.get("target_type",  "none")

      want_color = tc not in ("any", "none")
      want_type  = tt not in ("any", "none")

      # all shapes are valid targets
      if not want_color and not want_type:
         return list(range(len(shapes)))

      matches = []
      for i, s in enumerate(shapes):
         color_ok = (not want_color) or (s.color_name == tc)
         type_ok  = (not want_type)  or (s.shape_type == tt)
         if color_ok and type_ok:
            matches.append(i)

      return matches if matches else [0]

   def _matching_shape_indices(self) -> list[int]:
      """public alias used by demo.py for highlight rendering."""
      return self.target_indices

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
      task = self.goal.get("task", "")
      if task == "reach":             return self._score_reach()
      elif task == "touch":           return self._score_touch()
      elif task == "drag":            return self._score_drag()
      elif task == "arrange_in_sequence": return self._score_arrange_in_sequence()
      elif task == "arrange_in_line":     return self._score_arrange_in_line()
      elif task == "arrange_in_region":   return self._score_arrange_in_region()
      elif task == "arrange_in_groups":   return self._score_arrange_in_groups()
      return 0.0

   def _compute_score(self) -> float:
      """public alias used by demo.py and callbacks."""
      return self._compute_task_score()

   # -------------------------------------------------------------------------
   # solved dispatch
   # -------------------------------------------------------------------------

   def _is_solved(self) -> bool:
      task = self.goal.get("task", "")
      if task == "reach":   return self._solved_reach()
      elif task == "touch": return self._solved_touch()
      elif task == "drag":  return self._solved_drag()
      else:
         # arrangement tasks: score threshold is a reliable proxy since the
         # score directly measures spatial arrangement quality
         return self._compute_task_score() >= SCORE_SOLVE_THRESHOLD

   def _solved_reach(self) -> bool:
      """solved when cursor is within GRIP_RADIUS of any valid target."""
      if not self.shapes:
         return False
      for idx in self.target_indices:
         s    = self.shapes[idx]
         dist = float(np.sqrt((self.cx - s.x)**2 + (self.cy - s.y)**2))
         if dist <= GRIP_RADIUS:
            return True
      return False

   def _solved_touch(self) -> bool:
      # TODO: maybe make this solved once released after holding?
      # TODO: maybe consider time held? might be new task, where must be holding for some amount of time
      """solved when gripping any valid target within GRIP_RADIUS."""
      if not self.shapes or not self.holding:
         return False
      if self.grabbed_idx not in self.target_indices:
         return False
      s    = self.shapes[self.grabbed_idx]
      dist = float(np.sqrt((self.cx - s.x)**2 + (self.cy - s.y)**2))
      return dist <= GRIP_RADIUS

   def _solved_drag(self) -> bool:
      """solved when any valid target shape is inside the goal region."""
      if not self.shapes:
         return False
      region = self.goal.get("region")
      if not region or region == "none":
         return False
      for idx in self.target_indices:
         if self._per_shape_region_score(self.shapes[idx], region) >= 0.7:
            return True
      return False

   # -------------------------------------------------------------------------
   # starter task score functions
   # -------------------------------------------------------------------------

   def _score_reach(self) -> float:
      """
      two-zone proximity score toward the nearest valid target.
      zone 1 (far):  0.0 -> 0.7  as dist shrinks from ref_dist to 2*GRIP_RADIUS
      zone 2 (near): 0.7 -> 1.0  as dist shrinks from 2*GRIP_RADIUS to 0
      returns max score across all valid targets.
      """
      if not self.shapes:
         return 0.0
      best = 0.0
      ref_dist    = WINDOW_W / 2.0
      near_thresh = GRIP_RADIUS * 2.0
      for idx in self.target_indices:
         s    = self.shapes[idx]
         dist = float(np.sqrt((self.cx - s.x)**2 + (self.cy - s.y)**2))
         if dist <= GRIP_RADIUS:
            return 1.0
         if dist <= near_thresh:
            t    = (near_thresh - dist) / near_thresh
            best = max(best, 0.7 + 0.29 * t)
         else:
            t    = 1.0 - min((dist - near_thresh) / (ref_dist - near_thresh), 1.0)
            best = max(best, 0.7 * t)
      return float(best)

   def _score_touch(self) -> float:
      """
      proximity score capped low (0.39) when not gripping, so hovering near
      the shape is not a competitive strategy vs actually gripping.
      returns 1.0 only when gripping a valid target within GRIP_RADIUS.
      returns 0.1 when gripping something but not on a valid target.
      evaluates against nearest valid target.
      """
      if not self.shapes:
         return 0.0

      # gripping a valid target; check proximity
      # TODO: do we need to check proximity? should already be at cursor
      if self.holding and self.grabbed_idx in self.target_indices:
         return 1.0

      # gripping something invalid
      if self.holding:
         return 0.1

      # not gripping; low-cap proximity toward nearest valid target
      # TODO: experiment with different scoring here or maybe even none
      ref_dist    = WINDOW_W / 2.0
      near_thresh = GRIP_RADIUS * 2.0
      best        = 0.0
      for idx in self.target_indices:
         s    = self.shapes[idx]
         dist = float(np.sqrt((self.cx - s.x)**2 + (self.cy - s.y)**2))
         if dist <= near_thresh:
            t    = (near_thresh - dist) / near_thresh
            best = max(best, 0.3 + 0.09 * t)
         else:
            t    = 1.0 - min((dist - near_thresh) / (ref_dist - near_thresh), 1.0)
            best = max(best, 0.3 * t)
      return float(best)

   # TODO: consider change in env to support phases? or maybe drag can be a goal broken up into tasks. so there would be the reach task, then once completed it must complete the touch (or new hold) task, then the next task is moving to region while holding? idk
   def _score_drag(self) -> float:
      """
      two-phase score:
        phase 1 (not holding target): cursor proximity to nearest valid target,
          capped at 0.49 so gripping always produces a positive delta.
        phase 2 (holding valid target): region progress score in [0.5, 1.0].
      """
      # TODO: experiment with different scoring here. maybe phase gap at 0.25?
      if not self.shapes:
         return 0.0
      region = self.goal.get("region")
      if not region or region == "none":
         return 0.0

      # phase 2: holding a valid target
      if self.holding and self.grabbed_idx in self.target_indices:
         region_score = self._per_shape_region_score(
            self.shapes[self.grabbed_idx], region)
         return float(0.5 + 0.5 * region_score)

      # phase 1: not holding; proximity toward nearest valid target
      ref_dist    = WINDOW_W / 2.0
      near_thresh = GRIP_RADIUS * 2.0
      best        = 0.0
      for idx in self.target_indices:
         s    = self.shapes[idx]
         dist = float(np.sqrt((self.cx - s.x)**2 + (self.cy - s.y)**2))
         if dist <= near_thresh:
            t    = (near_thresh - dist) / near_thresh
            best = max(best, 0.4 + 0.09 * t)
         else:
            t    = 1.0 - min((dist - near_thresh) / (ref_dist - near_thresh), 1.0)
            best = max(best, 0.4 * t)
      return float(best)


   # -------------------------------------------------------------------------
   # arrange score functions
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
      """order score (0.6) + perpendicular spread score (0.4)."""
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
      region = self.goal.get("region")
      if not region or region == "none":
         return 0.0
      scores = [self._per_shape_region_score(s, region) for s in self.shapes]
      return float(np.mean(scores))

   def _per_shape_region_score(self, s, region: str) -> float:
      """
      0.7 * in_region + 0.3 * progress toward boundary.
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
      nn_isolation = mean over shapes of min_diff / (min_diff + min_same)
      """
      n         = self.n_shapes
      attribute = self.goal.get("attribute", "color")

      if n <= 1:
         return 1.0

      def get_attr(s):
         return s.color_name if attribute == "color" else s.shape_type

      def dist(a, b):
         return float(np.sqrt((a.x - b.x)**2 + (a.y - b.y)**2))

      intra_dists, inter_dists = [], []
      for i in range(n):
         for j in range(i + 1, n):
            d = dist(self.shapes[i], self.shapes[j])
            if get_attr(self.shapes[i]) == get_attr(self.shapes[j]):
               intra_dists.append(d)
            else:
               inter_dists.append(d)

      if not inter_dists:
         return 0.5   # all same attribute — degenerate, return neutral score

      intra_mean   = float(np.mean(intra_dists)) if intra_dists else 1.0
      inter_mean   = float(np.mean(inter_dists))
      global_score = inter_mean / (inter_mean + intra_mean)

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
         min_same = min(same_d) if same_d else 1.0
         per_shape.append(min_diff / (min_diff + min_same))

      return 0.5 * global_score + 0.5 * float(np.mean(per_shape))

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
      circle is filled when clicking, outline when not.
      """
      cx  = int(self.cx)
      cy  = int(self.cy)
      r   = _CURSOR_RADIUS
      gap = _CURSOR_GAP
      arm = _CURSOR_ARM
      col = _CURSOR_COLOR

      if self.grip:
         pygame.draw.circle(self.window, col, (cx, cy), r)
      else:
         pygame.draw.circle(self.window, col, (cx, cy), r, 1)

      pygame.draw.line(self.window, col, (cx, cy - r - gap), (cx, cy - r - gap - arm))
      pygame.draw.line(self.window, col, (cx, cy + r + gap), (cx, cy + r + gap + arm))
      pygame.draw.line(self.window, col, (cx - r - gap, cy), (cx - r - gap - arm, cy))
      pygame.draw.line(self.window, col, (cx + r + gap, cy), (cx + r + gap + arm, cy))