"""
shape_env.py

gymnasium environment for single-shape manipulation via a cursor.

*** SINGLE-SHAPE MODE: always exactly 1 shape (or phantom zone) per episode ***

--- task taxonomy ---

rudimentary — no grip, no shape interaction:
   move_cardinal   cursor must reach a target zone placed on a cardinal axis
                   from the canvas centre (left, right, top, bottom)
   move_diagonal   cursor must reach a target zone placed near a corner
   approach        real shape exists; cursor must get within 2x GRIP_RADIUS,
                   no grip required

grip builders — phantom zone, no visible/grippable shape:
   click_at        cursor must reach zone then fire grip (one rising edge)
   hold_at         cursor must reach zone and hold grip for HOLD_AT_STEPS steps

starter tasks — real shape, grip required:
   reach           cursor within GRIP_RADIUS of shape
   touch           grip shape while within GRIP_RADIUS
   drag            grip shape and move into target region

--- phantom zone encoding ---
   for move_cardinal / move_diagonal / click_at / hold_at:
   shapes[0] is a phantom Shape placed at the target zone position.
   it is never rendered and cannot be gripped (_try_grab returns -1).
   the network sees it in obs[9-18] identically to a real shape, so
   learned navigation skills transfer directly to reach/touch/drag.

--- grip bug fix (from previous revision) ---
   _prev_grip is updated unconditionally at the END of _apply_cursor_action.

--- touch fix ---
   _solved_touch: holding implies cursor was within GRIP_RADIUS when grip
   fired, so no separate distance check needed.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
from dataclasses import dataclass, field

from config import (
   MAX_SHAPES, OBS_VALUES_PER_SHAPE, get_obs_size, get_action_size,
   SHAPE_TYPES, N_SHAPE_TYPES, SHAPE_TYPE_IDX, CURSOR_SPEED,
   GRIP_THRESHOLD, GRIP_RADIUS, EMBEDDING_DIM,
   PHANTOM_ZONE_TASKS, HOLD_AT_STEPS,
)

# ---------------------------------------------------------------------------
# canvas constants
# ---------------------------------------------------------------------------

WINDOW_W     = 800
WINDOW_H     = 600
SHAPE_RADIUS = 20
FPS          = 60
MAX_STEPS    = 500
MARGIN       = SHAPE_RADIUS * 2

SCORE_SOLVE_THRESHOLD = 0.85
MOVEMENT_THRESHOLD    = 0.5   # pixels; below = inactive

COLORS = {
   "red":    (173,  46,  52),
   "green":  ( 78,  99,  30),
   "teal":   ( 87, 220, 215),
   "yellow": (199, 227,  54),
   "purple": (155,  90, 195),
}
COLOR_NAMES = list(COLORS.keys())
BG_COLOR    = (30, 30, 30)

# target zone radius for empty-canvas tasks (pixels)
ZONE_RADIUS = 30

# region boundaries (used by drag)
REGION_INNER = {
   "left":   WINDOW_W * 0.35,
   "right":  WINDOW_W * 0.65,
   "top":    WINDOW_H * 0.35,
   "bottom": WINDOW_H * 0.65,
}

LINE_SPREAD_THRESHOLD = 120   # unused in single-shape mode, kept for imports

# cursor crosshair
_CURSOR_RADIUS = 3
_CURSOR_GAP    = 4
_CURSOR_ARM    = 8
_CURSOR_COLOR  = (220, 220, 220)

# phantom zone render colour (dim white ring)
_ZONE_COLOR    = (180, 180, 180)


# ---------------------------------------------------------------------------
# RewardConfig
# ---------------------------------------------------------------------------

@dataclass
class RewardConfig:
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
   def __init__(self, shape_id, x, y, size, color_name, shape_type="circle",
                phantom: bool = False):
      self.shape_id   = shape_id
      self.x          = float(x)
      self.y          = float(y)
      self.size       = float(size)
      self.color_name = color_name
      self.color_rgb  = COLORS.get(color_name, (100, 100, 100))
      self.radius     = int(SHAPE_RADIUS * size)
      self.shape_type = shape_type
      self.phantom    = phantom   # if True: not rendered, not grippable

   def draw(self, surface, font):
      if self.phantom:
         return   # phantom zones rendered separately by env
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
      return np.array([
         self.x / WINDOW_W,
         self.y / WINDOW_H,
         (self.size - 1.0) / 1.0,
         COLOR_NAMES.index(self.color_name) / max(len(COLOR_NAMES) - 1, 1),
         SHAPE_TYPE_IDX.get(self.shape_type, 0) / max(N_SHAPE_TYPES - 1, 1),
      ], dtype=np.float32)


# ---------------------------------------------------------------------------
# ShapeEnv
# ---------------------------------------------------------------------------

class ShapeEnv(gym.Env):
   """
   single-shape manipulation environment.
   always exactly 1 shape (or phantom zone) per episode.
   """

   metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

   def __init__(
      self,
      n_shapes:       int          = None,   # ignored — always 1
      goal:           dict         = None,
      goal_embedding: np.ndarray   = None,
      render_mode:    str          = None,
      reward_config:  RewardConfig = None,
   ):
      super().__init__()

      self.n_shapes    = 1
      self.render_mode = render_mode
      self.rc          = reward_config or RewardConfig()

      self.goal = goal or {
         "task": "reach", "axis": "none", "direction": "none",
         "attribute": "none", "region": "none", "bounded": False,
         "target_color": "none", "target_type": "none",
      }

      self._goal_embedding = (
         goal_embedding[:EMBEDDING_DIM].astype(np.float32)
         if goal_embedding is not None
         else np.zeros(EMBEDDING_DIM, dtype=np.float32)
      )

      self.observation_space = spaces.Box(
         low=-2.0, high=2.0, shape=(get_obs_size(),), dtype=np.float32)
      self.action_space = spaces.Box(
         low=-1.0, high=1.0, shape=(get_action_size(),), dtype=np.float32)

      # cursor state
      self.cx          = float(WINDOW_W / 2)
      self.cy          = float(WINDOW_H / 2)
      self.grip        = False
      self._prev_grip  = False
      self.holding     = False
      self.grabbed_idx = -1

      # hold_at counter — consecutive steps grip has been held in zone
      self._hold_steps = 0

      # episode state
      self.shapes         = []
      self.steps          = 0
      self.prev_score     = 0.0
      self.target_indices = [0]

      # rendering
      self.window = None
      self.clock  = None
      self.font   = None

   # -------------------------------------------------------------------------
   # gymnasium interface
   # -------------------------------------------------------------------------

   def reset(self, seed=None):
      super().reset(seed=seed)

      self.n_shapes    = 1
      self.holding     = False
      self.grabbed_idx = -1
      self._prev_grip  = False
      self.grip        = False
      self._hold_steps = 0
      self.steps       = 0

      self.cx = float(WINDOW_W / 2)
      self.cy = float(WINDOW_H / 2)

      self.shapes         = self._spawn_shapes()
      self.target_indices = [0]
      self.prev_score     = self._compute_task_score()

      return self._get_obs(), {}

   def step(self, action):
      self.steps += 1
      prev_score  = self.prev_score

      cursor_action, intended_action = self._apply_cursor_action(action)

      # update hold counter for hold_at.
      # use self.grip not self.holding — phantom zones are never grippable
      # so self.holding is always False for hold_at, which would mean
      # _hold_steps never increments and _solved_hold_at never fires.
      task = self.goal.get("task", "none")
      if task == "hold_at":
         if self._in_zone() and self.grip:
            self._hold_steps += 1
         else:
            self._hold_steps = 0

      new_score  = self._compute_task_score()
      reward     = self._compute_reward(
         prev_score, new_score, cursor_action, intended_action, task)
      self.prev_score = new_score

      terminated = self._is_solved()
      if terminated:
         reward += self.rc.completion_bonus

      truncated = self.steps >= MAX_STEPS
      obs       = self._get_obs()
      info      = {"score": new_score, "steps": self.steps, "task": task}

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
   # cursor mechanics
   # -------------------------------------------------------------------------

   def _apply_cursor_action(self, action) -> tuple[float, float]:
      dx_raw   = float(action[0])
      dy_raw   = float(action[1])
      grip_raw = float(action[2])

      new_cx = float(np.clip(
         self.cx + dx_raw * CURSOR_SPEED, MARGIN, WINDOW_W - MARGIN))
      new_cy = float(np.clip(
         self.cy + dy_raw * CURSOR_SPEED, MARGIN, WINDOW_H - MARGIN))
      cursor_action   = float(np.sqrt((new_cx - self.cx)**2 + (new_cy - self.cy)**2))
      intended_action = float(np.sqrt(
         (dx_raw * CURSOR_SPEED)**2 + (dy_raw * CURSOR_SPEED)**2))
      self.cx = new_cx
      self.cy = new_cy

      self.grip      = grip_raw > GRIP_THRESHOLD
      grip_edge      = self.grip and not self._prev_grip

      if grip_edge:
         self.grabbed_idx = self._try_grab()
      elif not self.grip:
         self.grabbed_idx = -1

      self.holding = self.grip and (self.grabbed_idx >= 0)

      if self.holding:
         s   = self.shapes[self.grabbed_idx]
         s.x = float(np.clip(self.cx, MARGIN, WINDOW_W - MARGIN))
         s.y = float(np.clip(self.cy, MARGIN, WINDOW_H - MARGIN))

      # authoritative prev_grip update — must be last
      self._prev_grip = self.grip

      return cursor_action, intended_action

   def _try_grab(self) -> int:
      """
      grab nearest real (non-phantom) shape within GRIP_RADIUS.
      phantom shapes are never grippable.
      """
      best_idx  = -1
      best_dist = float("inf")
      for i, s in enumerate(self.shapes):
         if s.phantom:
            continue
         dist = np.sqrt((self.cx - s.x)**2 + (self.cy - s.y)**2)
         if dist <= GRIP_RADIUS and dist < best_dist:
            best_dist = dist
            best_idx  = i
      return best_idx

   def _nearest_non_grabbed(self) -> int:
      best_idx  = -1
      best_dist = float("inf")
      for i, s in enumerate(self.shapes):
         if i == self.grabbed_idx:
            continue
         dist = np.sqrt((self.cx - s.x)**2 + (self.cy - s.y)**2)
         if dist < best_dist:
            best_dist = dist
            best_idx  = i
      return best_idx

   # -------------------------------------------------------------------------
   # zone helpers (phantom-zone tasks)
   # -------------------------------------------------------------------------

   def _zone_pos(self) -> tuple[float, float]:
      """return (x, y) of the target zone (shapes[0] position)."""
      if self.shapes:
         return self.shapes[0].x, self.shapes[0].y
      return WINDOW_W / 2, WINDOW_H / 2

   def _dist_to_zone(self) -> float:
      zx, zy = self._zone_pos()
      return float(np.sqrt((self.cx - zx)**2 + (self.cy - zy)**2))

   def _in_zone(self) -> bool:
      return self._dist_to_zone() <= ZONE_RADIUS

   # -------------------------------------------------------------------------
   # reward
   # -------------------------------------------------------------------------

   def _compute_reward(self, prev_score, new_score,
                       cursor_action, intended_action, task) -> float:
      score_reward = (new_score - prev_score) * self.rc.score_scale
      wall         = self._wall_penalty(cursor_action, intended_action)
      inactivity   = self.rc.inactivity if cursor_action < MOVEMENT_THRESHOLD else 0.0
      grip         = self._grip_bonus(task)
      return score_reward + wall + inactivity + grip + self.rc.step_penalty

   def _wall_penalty(self, cursor_moved, intended_move) -> float:
      if intended_move > 5.0 and cursor_moved < intended_move * 0.25:
         return self.rc.wall
      return 0.0

   def _grip_bonus(self, task: str) -> float:
      """
      grip bonus scaled to task:
        phantom-zone tasks: bonus for clicking inside zone
        touch/drag: bonus for gripping valid shape target
      no bonus for rudimentary navigation tasks (move_*, approach).
      """
      bonus = 0.0

      if task in ("move_cardinal", "move_diagonal", "approach"):
         return 0.0

      # rising-edge bonus
      if not self._prev_grip and self.grip:
         if task in ("click_at", "hold_at"):
            # bonus proportional to how close to zone centre
            if self._in_zone():
               bonus += 5.0
            elif self._dist_to_zone() <= ZONE_RADIUS * 3:
               bonus += 2.0
         else:
            # real-shape tasks
            if self.shapes and not self.shapes[0].phantom:
               s    = self.shapes[0]
               dist = float(np.sqrt((self.cx - s.x)**2 + (self.cy - s.y)**2))
               if dist <= GRIP_RADIUS:
                  bonus += 5.0
               elif dist <= GRIP_RADIUS * 4:
                  bonus += 2.5
               else:
                  bonus += 1.5

      if task == "hold_at":
         # continuous bonus while holding inside zone
         if self.grip and self._in_zone():
            bonus += 0.5
         return bonus

      if not self.holding or self.grabbed_idx < 0:
         return bonus
      if task not in ("touch", "drag"):
         return bonus
      if self.grabbed_idx not in self.target_indices:
         return bonus

      bonus += self.rc.grip_grab
      if task == "touch" and not self._prev_grip:
         bonus += self.rc.grip_on_target

      return bonus

   # -------------------------------------------------------------------------
   # spawn
   # -------------------------------------------------------------------------

   def _spawn_shapes(self) -> list:
      rng  = self.np_random
      task = self.goal.get("task", "none")

      if task in PHANTOM_ZONE_TASKS:
         return [self._spawn_phantom_zone(rng, task)]

      # real shape
      tc = self.goal.get("target_color", "none")
      tt = self.goal.get("target_type",  "none")

      color_idx = int(rng.integers(0, len(COLOR_NAMES)))
      type_idx  = int(rng.integers(0, N_SHAPE_TYPES))
      if tc not in ("none", "any"):
         color_idx = COLOR_NAMES.index(tc)
      if tt not in ("none", "any"):
         type_idx = SHAPE_TYPES.index(tt)

      x    = float(rng.uniform(MARGIN, WINDOW_W - MARGIN))
      y    = float(rng.uniform(MARGIN, WINDOW_H - MARGIN))
      size = float(rng.uniform(1.0, 2.0))
      s    = Shape(0, x, y, size, COLOR_NAMES[color_idx], SHAPE_TYPES[type_idx])
      shapes = [s]

      # resample if already solved
      for _ in range(10):
         if not self._spawn_is_solved(shapes):
            break
         shapes[0].x = float(rng.uniform(MARGIN, WINDOW_W - MARGIN))
         shapes[0].y = float(rng.uniform(MARGIN, WINDOW_H - MARGIN))

      if task == "drag":
         region = self.goal.get("region")
         if region and region != "none":
            self._ensure_outside_region(rng, shapes[0], region)

      return shapes

   def _spawn_phantom_zone(self, rng, task: str) -> Shape:
      """
      place a phantom zone at a task-appropriate position.

      move_cardinal: one of four axis-aligned positions (left/right/top/bottom
                     of canvas, at ~70% of the way from centre to edge).
      move_diagonal: one of four corners (~70% toward corner from centre).
      click_at / hold_at: random position anywhere on the canvas,
                          biased away from the cursor start (canvas centre).
      """
      cx0 = WINDOW_W / 2
      cy0 = WINDOW_H / 2

      if task == "move_cardinal":
         # pick one of four cardinal directions from centre
         direction = int(rng.integers(0, 4))
         if direction == 0:   # left
            x, y = cx0 * 0.3, cy0
         elif direction == 1:  # right
            x, y = cx0 + cx0 * 0.7, cy0
         elif direction == 2:  # up
            x, y = cx0, cy0 * 0.3
         else:                 # down
            x, y = cx0, cy0 + cy0 * 0.7

      elif task == "move_diagonal":
         corner = int(rng.integers(0, 4))
         pad    = MARGIN * 3
         if corner == 0:
            x, y = pad, pad                           # top-left
         elif corner == 1:
            x, y = WINDOW_W - pad, pad                # top-right
         elif corner == 2:
            x, y = pad, WINDOW_H - pad               # bottom-left
         else:
            x, y = WINDOW_W - pad, WINDOW_H - pad    # bottom-right

      else:  # click_at / hold_at: random, biased away from centre
         # sample in a ring: at least 150px from centre
         for _ in range(20):
            x = float(rng.uniform(MARGIN, WINDOW_W - MARGIN))
            y = float(rng.uniform(MARGIN, WINDOW_H - MARGIN))
            if np.sqrt((x - cx0)**2 + (y - cy0)**2) > 150:
               break

      # phantom shape — size 1.0, colour "red" (arbitrary, not rendered)
      return Shape(0, float(x), float(y), 1.0, "red", "circle", phantom=True)

   def _ensure_outside_region(self, rng, s: Shape, region: str):
      boundary = REGION_INNER[region]
      for _ in range(10):
         inside = (
            (region == "left"   and s.x <= boundary) or
            (region == "right"  and s.x >= boundary) or
            (region == "top"    and s.y <= boundary) or
            (region == "bottom" and s.y >= boundary)
         )
         if not inside:
            break
         if region in ("left", "right"):
            lo  = boundary + MARGIN if region == "left" else MARGIN
            hi  = WINDOW_W - MARGIN if region == "left" else boundary - MARGIN
            s.x = float(rng.uniform(lo, hi))
         else:
            lo  = boundary + MARGIN if region == "top" else MARGIN
            hi  = WINDOW_H - MARGIN if region == "top" else boundary - MARGIN
            s.y = float(rng.uniform(lo, hi))

   def _spawn_is_solved(self, shapes) -> bool:
      orig_shapes  = self.shapes
      orig_targets = self.target_indices
      self.shapes         = shapes
      self.target_indices = [0]
      solved              = self._is_solved()
      self.shapes         = orig_shapes
      self.target_indices = orig_targets
      return solved

   # -------------------------------------------------------------------------
   # target helpers
   # -------------------------------------------------------------------------

   def _find_target_indices(self) -> list[int]:
      return [0] if self.shapes else []

   def _matching_shape_indices(self) -> list[int]:
      return self.target_indices

   # -------------------------------------------------------------------------
   # obs
   # -------------------------------------------------------------------------

   def _get_obs(self) -> np.ndarray:
      grabbed_norm = (self.grabbed_idx / max(self.n_shapes - 1, 1)
                      if self.grabbed_idx >= 0 else -1.0)
      cursor_state = np.array([
         self.cx / WINDOW_W * 2.0 - 1.0,
         self.cy / WINDOW_H * 2.0 - 1.0,
         1.0 if self.holding else 0.0,
         float(grabbed_norm),
      ], dtype=np.float32)

      if self.grabbed_idx >= 0:
         grabbed_feats = self.shapes[self.grabbed_idx].as_obs()
      else:
         grabbed_feats = np.zeros(OBS_VALUES_PER_SHAPE, dtype=np.float32)

      nearest_idx = self._nearest_non_grabbed()
      if nearest_idx >= 0:
         nearest_feats = self.shapes[nearest_idx].as_obs()
      else:
         nearest_feats = np.zeros(OBS_VALUES_PER_SHAPE, dtype=np.float32)

      if self.shapes:
         all_shapes = self.shapes[0].as_obs()
      else:
         all_shapes = np.zeros(OBS_VALUES_PER_SHAPE, dtype=np.float32)

      return np.concatenate([
         cursor_state,
         grabbed_feats,
         nearest_feats,
         all_shapes,
         self._goal_embedding,
      ]).astype(np.float32)

   # -------------------------------------------------------------------------
   # score dispatch
   # -------------------------------------------------------------------------

   def _compute_task_score(self) -> float:
      task = self.goal.get("task", "none")
      if task == "move_cardinal":  return self._score_zone_proximity()
      elif task == "move_diagonal": return self._score_zone_proximity()
      elif task == "approach":      return self._score_approach()
      elif task == "click_at":      return self._score_click_at()
      elif task == "hold_at":       return self._score_hold_at()
      elif task == "reach":         return self._score_reach()
      elif task == "touch":         return self._score_touch()
      elif task == "drag":          return self._score_drag()
      return 0.0

   def _compute_score(self) -> float:
      return self._compute_task_score()

   # -------------------------------------------------------------------------
   # solved dispatch
   # -------------------------------------------------------------------------

   def _is_solved(self) -> bool:
      task = self.goal.get("task", "none")
      if task == "none":              return True
      elif task in ("move_cardinal",
                    "move_diagonal"): return self._in_zone()
      elif task == "approach":        return self._solved_approach()
      elif task == "click_at":        return self._solved_click_at()
      elif task == "hold_at":         return self._solved_hold_at()
      elif task == "reach":           return self._solved_reach()
      elif task == "touch":           return self._solved_touch()
      elif task == "drag":            return self._solved_drag()
      return False

   # -------------------------------------------------------------------------
   # score functions
   # -------------------------------------------------------------------------

   def _score_zone_proximity(self) -> float:
      """
      smooth proximity score toward the phantom zone.
      0.0 → 0.7 as dist shrinks from ref to 2*ZONE_RADIUS.
      0.7 → 1.0 as dist shrinks from 2*ZONE_RADIUS to 0 (inside zone).
      """
      dist        = self._dist_to_zone()
      ref_dist    = WINDOW_W / 2.0
      near_thresh = ZONE_RADIUS * 2.0
      if dist <= ZONE_RADIUS:
         return 1.0
      if dist <= near_thresh:
         t = (near_thresh - dist) / near_thresh
         return 0.7 + 0.29 * t
      t = 1.0 - min((dist - near_thresh) / (ref_dist - near_thresh), 1.0)
      return float(0.7 * t)

   def _score_approach(self) -> float:
      """
      proximity score toward the shape, solved threshold = 2x GRIP_RADIUS.
      same two-zone structure as reach but with a larger solved radius.
      """
      if not self.shapes:
         return 0.0
      s           = self.shapes[0]
      dist        = float(np.sqrt((self.cx - s.x)**2 + (self.cy - s.y)**2))
      solve_dist  = GRIP_RADIUS * 2.0
      ref_dist    = WINDOW_W / 2.0
      near_thresh = solve_dist * 2.0
      if dist <= solve_dist:
         return 1.0
      if dist <= near_thresh:
         t = (near_thresh - dist) / near_thresh
         return 0.7 + 0.29 * t
      t = 1.0 - min((dist - near_thresh) / (ref_dist - near_thresh), 1.0)
      return float(0.7 * t)

   def _score_click_at(self) -> float:
      """
      proximity [0, 0.49] when not in zone or not gripping.
      1.0 when grip is active inside the zone.
      """
      if self._in_zone() and self.grip:
         return 1.0
      # proximity component — capped at 0.49 so gripping inside beats hovering
      dist        = self._dist_to_zone()
      ref_dist    = WINDOW_W / 2.0
      near_thresh = ZONE_RADIUS * 2.0
      if dist <= near_thresh:
         t = (near_thresh - dist) / near_thresh
         return min(0.4 + 0.09 * t, 0.49)
      t = 1.0 - min((dist - near_thresh) / (ref_dist - near_thresh), 1.0)
      return float(0.4 * t)

   def _score_hold_at(self) -> float:
      """
      proximity [0, 0.49] when not gripping in zone.
      [0.5, 1.0] as hold_steps increases toward HOLD_AT_STEPS.
      uses self.grip not self.holding — phantom zones are never grippable.
      """
      if self._in_zone() and self.grip:
         progress = min(self._hold_steps / HOLD_AT_STEPS, 1.0)
         return 0.5 + 0.5 * progress
      dist        = self._dist_to_zone()
      ref_dist    = WINDOW_W / 2.0
      near_thresh = ZONE_RADIUS * 2.0
      if dist <= near_thresh:
         t = (near_thresh - dist) / near_thresh
         return min(0.4 + 0.09 * t, 0.49)
      t = 1.0 - min((dist - near_thresh) / (ref_dist - near_thresh), 1.0)
      return float(0.4 * t)

   def _score_reach(self) -> float:
      if not self.shapes or self.shapes[0].phantom:
         return 0.0
      s           = self.shapes[0]
      dist        = float(np.sqrt((self.cx - s.x)**2 + (self.cy - s.y)**2))
      if dist <= GRIP_RADIUS:
         return 1.0
      ref_dist    = WINDOW_W / 2.0
      near_thresh = GRIP_RADIUS * 2.0
      if dist <= near_thresh:
         t = (near_thresh - dist) / near_thresh
         return 0.7 + 0.29 * t
      t = 1.0 - min((dist - near_thresh) / (ref_dist - near_thresh), 1.0)
      return float(0.7 * t)

   def _score_touch(self) -> float:
      if not self.shapes or self.shapes[0].phantom:
         return 0.0
      if self.holding and self.grabbed_idx == 0:
         return 1.0
      s           = self.shapes[0]
      dist        = float(np.sqrt((self.cx - s.x)**2 + (self.cy - s.y)**2))
      ref_dist    = WINDOW_W / 2.0
      near_thresh = GRIP_RADIUS * 2.0
      if dist <= near_thresh:
         t = (near_thresh - dist) / near_thresh
         return 0.3 + 0.09 * t
      t = 1.0 - min((dist - near_thresh) / (ref_dist - near_thresh), 1.0)
      return float(0.3 * t)

   def _score_drag(self) -> float:
      if not self.shapes or self.shapes[0].phantom:
         return 0.0
      region = self.goal.get("region")
      if not region or region == "none":
         return 0.0
      if self.holding and self.grabbed_idx == 0:
         region_score = self._per_shape_region_score(self.shapes[0], region)
         return float(0.5 + 0.5 * region_score)
      s           = self.shapes[0]
      dist        = float(np.sqrt((self.cx - s.x)**2 + (self.cy - s.y)**2))
      ref_dist    = WINDOW_W / 2.0
      near_thresh = GRIP_RADIUS * 2.0
      if dist <= near_thresh:
         t = (near_thresh - dist) / near_thresh
         return 0.4 + 0.09 * t
      t = 1.0 - min((dist - near_thresh) / (ref_dist - near_thresh), 1.0)
      return float(0.4 * t)

   def _per_shape_region_score(self, s, region: str) -> float:
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
      else:
         inside   = s.y >= boundary
         progress = 1.0 - max(boundary - s.y, 0) / max(boundary, 1)
      return 0.7 * float(inside) + 0.3 * float(np.clip(progress, 0.0, 1.0))

   # -------------------------------------------------------------------------
   # solved functions
   # -------------------------------------------------------------------------

   def _solved_approach(self) -> bool:
      if not self.shapes or self.shapes[0].phantom:
         return False
      s    = self.shapes[0]
      dist = float(np.sqrt((self.cx - s.x)**2 + (self.cy - s.y)**2))
      return dist <= GRIP_RADIUS * 2.0

   def _solved_click_at(self) -> bool:
      """solved when grip is active inside the zone."""
      return self._in_zone() and self.grip

   def _solved_hold_at(self) -> bool:
      return self._hold_steps >= HOLD_AT_STEPS

   def _solved_reach(self) -> bool:
      if not self.shapes or self.shapes[0].phantom:
         return False
      s    = self.shapes[0]
      dist = float(np.sqrt((self.cx - s.x)**2 + (self.cy - s.y)**2))
      return dist <= GRIP_RADIUS

   def _solved_touch(self) -> bool:
      if not self.shapes or self.shapes[0].phantom:
         return False
      return self.holding and self.grabbed_idx == 0

   def _solved_drag(self) -> bool:
      if not self.shapes or self.shapes[0].phantom:
         return False
      region = self.goal.get("region")
      if not region or region == "none":
         return False
      return self._per_shape_region_score(self.shapes[0], region) >= 0.7

   # -------------------------------------------------------------------------
   # attribute helper (kept for oracle compatibility)
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

      # draw phantom zone as a dim ring
      task = self.goal.get("task", "none")
      if task in PHANTOM_ZONE_TASKS and self.shapes:
         zx, zy = int(self.shapes[0].x), int(self.shapes[0].y)
         pygame.draw.circle(self.window, _ZONE_COLOR, (zx, zy), ZONE_RADIUS, 2)

      for shape in self.shapes:
         shape.draw(self.window, self.font)

      # highlight target for approach/reach/touch/drag
      if task in ("approach", "reach", "touch", "drag") and self.shapes:
         s = self.shapes[0]
         r = GRIP_RADIUS * 2 if task == "approach" else GRIP_RADIUS
         pygame.draw.circle(self.window, (255, 220, 60),
                            (int(s.x), int(s.y)), r + 6, 2)

      self._draw_cursor()

      score = self._compute_score()
      hud   = (f"task: {task} | progress: {score:.2%} | step: {self.steps}")
      if task == "hold_at":
         hud += f" | hold_steps: {self._hold_steps}/{HOLD_AT_STEPS}"
      self.window.blit(
         self.font.render(hud, True, (200, 200, 200)), (10, 10))

      if self.render_mode != "human":
         return np.transpose(
            pygame.surfarray.array3d(self.window), axes=(1, 0, 2))
      return None

   def _draw_cursor(self):
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