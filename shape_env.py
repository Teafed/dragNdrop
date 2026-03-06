"""
shape_env.py

Gymnasium environment for 2D shape manipulation via relative nudges.

--- action space ---
    [shape_selector, dx, dy]
    shape_selector in [-1, 1] -> mapped to a shape index
    dx, dy         in [-1, 1] -> scaled by MAX_NUDGE pixels per step

--- observation space ---
    per shape (up to MAX_SHAPES, zero-padded): x, y, size, color, shape_type
    goal encoding: GOAL_ENCODING_DIM values from GoalEncoder MLP
    action history: last_shape_idx, steps_on_shape, last_dx, last_dy
    total: MAX_SHAPES*5 + 64 + 4 = 98

--- wave 3 tasks (2x2x2 cube) ---
    arrange_in_sequence  one target space, unbounded, ordered by attribute
    arrange_in_line      one target space, bounded (actual line), ordered or unordered
    arrange_in_region    one target space, bounded (canvas subregion), unordered
    arrange_in_groups    many target spaces (one per attribute value), bounded, unordered

--- reward design ---
    1. task score delta        - improvement in per-task score function
    2. rank/cohesion delta     - secondary signal (same as score for most tasks)
    3. oscillation penalty     - discourages score going up then immediately down
    4. wall penalty            - discourages pushing into canvas borders
    5. inactivity penalty      - discourages zero-nudge actions
    6. neglect penalty         - escalating penalty for ignoring shapes too long
    7. fixation penalty        - discourages moving only one shape all episode
    8. completion bonus        - large reward when all shapes satisfy task
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame

from config import (
   MAX_SHAPES, OBS_VALUES_PER_SHAPE, ACTION_HISTORY_SIZE,
   GOAL_ENCODING_DIM, get_obs_size,
   SHAPE_TYPES, N_SHAPE_TYPES, SHAPE_TYPE_IDX,
)

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

WINDOW_W     = 800
WINDOW_H     = 600
SHAPE_RADIUS = 20
FPS          = 60
MAX_STEPS    = 500
MAX_NUDGE    = 25
MARGIN       = SHAPE_RADIUS * 2

SCORE_SOLVE_THRESHOLD = 0.85   # episode solved when score >= this
SOLVE_TOLERANCE       = 60     # pixels (kept for legacy; unused in wave 3)

STEP_PENALTY     = -0.02
COMPLETION_BONUS = 25.0

MOVEMENT_THRESHOLD = 0.5    # pixels — below this = not moving
NEGLECT_PATIENCE   = 10     # steps before neglect penalty fires

COLORS = {
   "red":    (220,  60,  60),
   "green":  ( 60, 180,  60),
   "blue":   ( 60, 100, 220),
   "yellow": (220, 200,  50),
   "purple": (160,  60, 200),
}
COLOR_NAMES  = list(COLORS.keys())
BG_COLOR     = (30, 30, 35)

# region boundaries (fraction of canvas) — shared by env score and oracle
REGION_INNER = {
   "left":   WINDOW_W * 0.35,
   "right":  WINDOW_W * 0.65,
   "top":    WINDOW_H * 0.35,
   "bottom": WINDOW_H * 0.65,
}

# perpendicular spread threshold for arrange_in_line (solve condition)
LINE_SPREAD_THRESHOLD = 80   # pixels — max allowed spread on perpendicular axis

# supported tasks — matches config.SUPPORTED_TASKS
SUPPORTED_TASKS = [
   "arrange_in_sequence",
   "arrange_in_line",
   "arrange_in_region",
   "arrange_in_groups",
]


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
   Gymnasium environment for 2D shape manipulation.
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

      self.shapes            = []
      self.steps             = 0
      self.last_shape_idx    = -1
      self.steps_on_shape    = 0
      self.last_action_dx    = 0.0
      self.last_action_dy    = 0.0
      self.steps_since_moved = []
      self.prev_score        = 0.0
      self.prev_score_delta  = 0.0
      self.prev_rank_corr    = 0.0
      self.window            = None
      self.clock             = None
      self.font              = None

   # -------------------------------------------------------------------------
   # gymnasium interface
   # -------------------------------------------------------------------------

   def reset(self, seed=None, options=None):
      super().reset(seed=seed)

      if self._fixed_n_shapes is None:
         self.n_shapes = int(self.np_random.integers(2, MAX_SHAPES + 1))
      else:
         self.n_shapes = self._fixed_n_shapes

      self.steps             = 0
      self.last_shape_idx    = -1
      self.steps_on_shape    = 0
      self.last_action_dx    = 0.0
      self.last_action_dy    = 0.0
      self.shapes            = self._spawn_shapes()
      self.steps_since_moved = [0] * self.n_shapes
      self.prev_score        = self._compute_task_score()
      self.prev_score_delta  = 0.0
      self.prev_rank_corr    = self._compute_rank_corr()
      return self._get_obs(), {}

   def step(self, action):
      self.steps += 1

      raw_idx   = float(action[0])
      shape_idx = int(np.clip(
         round((raw_idx + 1.0) / 2.0 * (self.n_shapes - 1)),
         0, self.n_shapes - 1
      ))

      s     = self.shapes[shape_idx]
      pre_x = s.x
      pre_y = s.y

      dx  = float(action[1]) * MAX_NUDGE
      dy  = float(action[2]) * MAX_NUDGE
      s.x = float(np.clip(s.x + dx, MARGIN, WINDOW_W - MARGIN))
      s.y = float(np.clip(s.y + dy, MARGIN, WINDOW_H - MARGIN))

      actual_move   = np.sqrt((s.x - pre_x) ** 2 + (s.y - pre_y) ** 2)
      intended_move = np.sqrt(dx ** 2 + dy ** 2)

      # 1. task score delta
      new_score    = self._compute_task_score()
      score_delta  = new_score - self.prev_score
      score_reward = score_delta * 10.0

      # 2. rank / cohesion delta
      new_rank_corr = self._compute_rank_corr()
      rank_delta    = new_rank_corr - self.prev_rank_corr
      rank_reward   = rank_delta * 2.0

      # 3. oscillation penalty
      oscillation_penalty = (
         -0.06 if self.prev_score_delta > 0.01 and score_delta < -0.01 else 0.0)

      # 4. wall penalty
      wall_penalty = (
         -0.05 if intended_move > 5.0 and actual_move < intended_move * 0.25
         else 0.0)

      # 5. inactivity penalty
      inactivity_penalty = -0.04 if actual_move < MOVEMENT_THRESHOLD else 0.0

      # 6. neglect penalty — escalates linearly with overdue steps
      neglect_penalty = 0.0
      for i in range(self.n_shapes):
         if i != shape_idx and self.steps_since_moved[i] > NEGLECT_PATIENCE:
            overdue = self.steps_since_moved[i] - NEGLECT_PATIENCE
            neglect_penalty -= min(0.005 * overdue, 0.02)

      # 7. fixation penalty — discourages single-shape collapse
      fixation_penalty = (
         -0.005 * min(self.steps_on_shape - 20, 20)
         if self.steps_on_shape > 20 else 0.0)

      # bookkeeping
      for i in range(self.n_shapes):
         self.steps_since_moved[i] = (
            0 if i == shape_idx else self.steps_since_moved[i] + 1)
      self.steps_on_shape   = (
         self.steps_on_shape + 1 if shape_idx == self.last_shape_idx else 1)
      self.last_shape_idx   = shape_idx
      self.last_action_dx   = float(action[1])
      self.last_action_dy   = float(action[2])
      self.prev_score_delta = score_delta
      self.prev_score       = new_score
      self.prev_rank_corr   = new_rank_corr

      reward = (score_reward + rank_reward + oscillation_penalty
                + wall_penalty + inactivity_penalty
                + neglect_penalty + fixation_penalty
                + STEP_PENALTY)

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
   # spawn
   # -------------------------------------------------------------------------

   def _spawn_shapes(self) -> list:
      rng       = self.np_random
      task      = self.goal.get("task", "arrange_in_sequence")
      attribute = self.goal.get("attribute", "none")

      # color assignment — for group tasks guarantee at least one repeated color
      color_indices = rng.integers(0, len(COLOR_NAMES), size=self.n_shapes)
      if task == "arrange_in_groups" and attribute == "color" and self.n_shapes >= 2:
         if len(set(color_indices)) == self.n_shapes:
            color_indices[1] = color_indices[0]

      # shape type assignment — for group tasks guarantee at least one repeated type
      type_indices = rng.integers(0, N_SHAPE_TYPES, size=self.n_shapes)
      if task == "arrange_in_groups" and attribute == "shape_type" and self.n_shapes >= 2:
         if len(set(type_indices)) == self.n_shapes:
            type_indices[1] = type_indices[0]
      elif self.n_shapes >= 2 and len(set(type_indices)) < 2:
         # always at least 2 distinct shape types for visual variety
         type_indices[1] = (type_indices[0] + 1) % N_SHAPE_TYPES

      shapes = []
      for i in range(self.n_shapes):
         x = float(rng.uniform(MARGIN, WINDOW_W - MARGIN))
         y = float(rng.uniform(MARGIN, WINDOW_H - MARGIN))
         size       = float(rng.uniform(0.5, 2.0))
         color_name = COLOR_NAMES[color_indices[i]]
         shape_type = SHAPE_TYPES[type_indices[i]]
         shapes.append(Shape(i, x, y, size, color_name, shape_type))

      # ensure spawn isn't already solved (can happen for group/region tasks)
      MAX_SPAWN_RETRIES = 10
      for _ in range(MAX_SPAWN_RETRIES):
         if not self._initial_score_solved(shapes):
            break
         for s in shapes:
            s.x = float(rng.uniform(MARGIN, WINDOW_W - MARGIN))
            s.y = float(rng.uniform(MARGIN, WINDOW_H - MARGIN))

      return shapes

   def _initial_score_solved(self, shapes) -> bool:
      """temporarily swap self.shapes to reuse scoring without touching env state."""
      original    = self.shapes
      self.shapes = shapes
      solved      = self._compute_task_score() >= SCORE_SOLVE_THRESHOLD
      self.shapes = original
      return solved

   # -------------------------------------------------------------------------
   # per-shape score functions
   # -------------------------------------------------------------------------
   # each returns a value in [0, 1].
   # the episode score is the mean of per-shape scores.
   # _compute_task_score() is the single entry point used by step() and _is_solved().

   def _compute_task_score(self) -> float:
      """dispatch to the appropriate per-task score function."""
      task = self.goal.get("task", "arrange_in_sequence")
      if task == "arrange_in_sequence":
         return self._score_arrange_in_sequence()
      elif task == "arrange_in_line":
         return self._score_arrange_in_line()
      elif task == "arrange_in_region":
         return self._score_arrange_in_region()
      elif task == "arrange_in_groups":
         return self._score_arrange_in_groups()
      return 0.0

   def _score_arrange_in_sequence(self) -> float:
      """
      score for arrange_in_sequence: how well are shapes ordered by attribute
      along the axis?

      per-shape score = 1 - |current_rank - ideal_rank| / (n - 1)
      episode score   = mean per-shape score.

      1.0 = every shape in its ideal rank position.
      0.5 = random permutation on average.
      0.0 = perfectly reversed.

      perpendicular axis is unconstrained — any y position is fine for
      horizontal sorting, any x for vertical.
      """
      n         = self.n_shapes
      axis      = self.goal.get("axis", "x")
      direction = self.goal.get("direction", "ascending")
      attribute = self.goal.get("attribute", "size")

      if n <= 1:
         return 1.0

      positions  = [s.x if axis == "x" else s.y for s in self.shapes]
      attr_vals  = self._get_attribute_values(attribute)

      ideal_ranks = np.argsort(np.argsort(attr_vals)).astype(float)
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
      score for arrange_in_line: shapes must lie along a line (bounded
      perpendicular spread) and optionally be ordered by attribute.

      two components:
        - order score (0.6 weight): same as _score_arrange_in_sequence if
          attribute is set, else 1.0 (evenly spaced, no ordering constraint)
        - spread score (0.4 weight): how tightly are shapes aligned on the
          perpendicular axis? 1.0 = all at the same perpendicular coordinate.

      this rewards getting shapes into a line AND keeping them there.
      """
      n         = self.n_shapes
      axis      = self.goal.get("axis", "x")
      attribute = self.goal.get("attribute", "none")

      if n <= 1:
         return 1.0

      # --- order component ---
      if attribute != "none":
         order_score = self._score_arrange_in_sequence()
      else:
         # no ordering required — just check spacing is reasonably even
         positions = sorted(s.x if axis == "x" else s.y for s in self.shapes)
         gaps      = [positions[i+1] - positions[i] for i in range(n - 1)]
         if not gaps or max(gaps) == 0:
            order_score = 1.0
         else:
            # coefficient of variation of gaps — 0 = perfectly even
            cv = np.std(gaps) / (np.mean(gaps) + 1e-6)
            order_score = max(0.0, 1.0 - cv)

      # --- spread component (perpendicular axis) ---
      perp_vals = [s.y if axis == "x" else s.x for s in self.shapes]
      spread    = max(perp_vals) - min(perp_vals)
      # normalize: spread of 0 = 1.0, spread of LINE_SPREAD_THRESHOLD = 0.0
      spread_score = max(0.0, 1.0 - spread / LINE_SPREAD_THRESHOLD)

      return 0.6 * order_score + 0.4 * spread_score

   def _score_arrange_in_region(self) -> float:
      """
      score for arrange_in_region: all shapes inside the target canvas subregion,
      distributed across it (not just at the boundary).

      per-shape score:
        - 0.0   if on the wrong side of the region boundary
        - scales from 0.0 to 1.0 as the shape moves from the boundary to
          the far wall of the region

      episode score = mean per-shape score.
      threshold 0.85 requires most shapes to be well inside the region.
      """
      region = self.goal.get("region", "left")
      scores = [self._per_shape_region_score(s, region) for s in self.shapes]
      return float(np.mean(scores))

   def _per_shape_region_score(self, s, region: str) -> float:
      """
      per-shape score for arrange_in_region in [0, 1].

      two components:
        - in_region (0.7 weight): 1.0 if shape is inside the region, else 0.0.
          crossing the boundary is the primary event.
        - progress  (0.3 weight): smooth 0->1 gradient as shape approaches and
          crosses the boundary. provides signal before crossing and rewards
          depth after.

      a shape just past the boundary scores ~0.7.
      a shape well inside the region scores toward 1.0.
      mean across all shapes >= 0.85 requires most shapes clearly inside.
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
      score for arrange_in_groups: shapes partitioned by attribute, each group
      in its own subregion, groups separated from each other.

      per-shape score = 0.6 * nn_correct + 0.4 * separation_score
        - nn_correct:       1.0 if nearest neighbor is same attribute, else 0.0
        - separation_score: how far is the nearest different-attribute shape?
                            normalized by half-diagonal. rewards well-separated groups.

      episode score = mean per-shape score.
      """
      n         = self.n_shapes
      attribute = self.goal.get("attribute", "color")
      half_diag = np.sqrt((WINDOW_W / 2) ** 2 + (WINDOW_H / 2) ** 2)

      if n <= 1:
         return 1.0

      def get_attr(s):
         return s.color_name if attribute == "color" else s.shape_type

      per_shape = []
      for i in range(n):
         same_d = [
            np.sqrt((self.shapes[i].x - self.shapes[j].x) ** 2 +
                    (self.shapes[i].y - self.shapes[j].y) ** 2)
            for j in range(n)
            if i != j and get_attr(self.shapes[i]) == get_attr(self.shapes[j])
         ]
         diff_d = [
            np.sqrt((self.shapes[i].x - self.shapes[j].x) ** 2 +
                    (self.shapes[i].y - self.shapes[j].y) ** 2)
            for j in range(n)
            if i != j and get_attr(self.shapes[i]) != get_attr(self.shapes[j])
         ]

         # nn_correct: 1 if nearest neighbor is same attribute
         if not same_d or not diff_d:
            nn_correct = 1.0   # only one attribute value — trivially grouped
         else:
            nn_correct = 1.0 if min(same_d) < min(diff_d) else 0.0

         # separation_score: how far is nearest different-attribute shape?
         if not diff_d:
            separation_score = 1.0
         else:
            separation_score = min(min(diff_d) / half_diag, 1.0)

         per_shape.append(0.6 * nn_correct + 0.4 * separation_score)

      return float(np.mean(per_shape))

   def _compute_rank_corr(self) -> float:
      """secondary quality signal for tensorboard — mirrors task score for all tasks."""
      return self._compute_task_score()

   def _compute_score(self) -> float:
      """alias for callbacks and debug."""
      return self._compute_task_score()

   def _is_solved(self) -> bool:
      return self._compute_task_score() >= SCORE_SOLVE_THRESHOLD

   # -------------------------------------------------------------------------
   # attribute helpers
   # -------------------------------------------------------------------------

   def _get_attribute_values(self, attribute: str) -> list:
      """return the attribute value for each shape as a sortable float."""
      if attribute == "size":
         return [s.size for s in self.shapes]
      elif attribute == "color":
         return [COLOR_NAMES.index(s.color_name) for s in self.shapes]
      elif attribute == "shape_type":
         return [SHAPE_TYPE_IDX.get(s.shape_type, 0) for s in self.shapes]
      else:
         return list(range(self.n_shapes))   # no attribute — use index order

   # -------------------------------------------------------------------------
   # obs and goal encoding
   # -------------------------------------------------------------------------

   def _get_obs(self) -> np.ndarray:
      active_obs = np.concatenate([self.shapes[i].as_obs()
                                   for i in range(self.n_shapes)])
      n_padding  = MAX_SHAPES - self.n_shapes
      padding    = np.zeros(n_padding * OBS_VALUES_PER_SHAPE, dtype=np.float32)
      history    = np.array([
         self.last_shape_idx / max(self.n_shapes - 1, 1),
         min(self.steps_on_shape / 10.0, 2.0),
         self.last_action_dx,
         self.last_action_dy,
      ], dtype=np.float32)
      return np.concatenate(
         [active_obs, padding, self._goal_encoding, history]
      ).astype(np.float32)

   def set_goal_encoding(self, encoding: np.ndarray):
      assert encoding.shape == (GOAL_ENCODING_DIM,), (
         f"expected ({GOAL_ENCODING_DIM},), got {encoding.shape}")
      self._goal_encoding = encoding.astype(np.float32)

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

      score  = self._compute_score()
      rank   = self._compute_rank_corr()
      goal   = self.goal
      task   = goal["task"]
      hud = (f"task: {task} | attr: {goal['attribute']} | "
             f"axis: {goal['axis']} | dir: {goal['direction']}   "
             f"progress: {score:.2%}   rank/cohesion: {rank:+.2f}   "
             f"step: {self.steps}")
      self.window.blit(self.font.render(hud, True, (200, 200, 200)), (10, 10))

      if self.render_mode != "human":
         return np.transpose(
            pygame.surfarray.array3d(self.window), axes=(1, 0, 2))
      return None


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
