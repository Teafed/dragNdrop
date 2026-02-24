"""
shape_env.py

a gymnasium environment for a 2d shape manipulation task.
the agent manipulates shapes by applying small relative nudges,
like a hand pushing objects rather than teleporting them.

action space:
   [shape_selector, dx, dy]
   - shape_selector in [-1, 1], mapped to a shape index
   - dx, dy in [-1, 1], scaled by MAX_NUDGE pixels per step

observation space:
   for each shape: [x_norm, y_norm, size_norm, color_idx_norm,
                    dist_to_target_norm, dx_to_target, dy_to_target]
   plus goal encoding: [axis, direction]
   plus history: [last_shape_idx_norm, steps_on_shape_norm, last_dx, last_dy]

reward:
   per step: directional progress + consistency bonus + switch bonus
             + camp penalty + overshoot penalty + step penalty.
   on solve: completion bonus.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame

# --- constants ---
WINDOW_W     = 800
WINDOW_H     = 600
MAX_SHAPES   = 5
SHAPE_RADIUS = 20
FPS          = 60
MAX_STEPS    = 500
MAX_NUDGE    = 25      # max pixels a shape can move per step
MARGIN       = SHAPE_RADIUS * 2

# how close each shape must be to its target to count as solved
SOLVE_TOLERANCE = 40   # pixels

# reward scaling
STEP_PENALTY     = -0.01
COMPLETION_BONUS = 25.0

COLORS = {
   "red":    (220,  60,  60),
   "green":  ( 60, 180,  60),
   "blue":   ( 60, 100, 220),
   "yellow": (220, 200,  50),
   "purple": (160,  60, 200),
}
COLOR_NAMES  = list(COLORS.keys())
BG_COLOR     = (30, 30, 35)
TARGET_COLOR = (80, 80, 80)   # ghost markers showing target positions


class Shape:
   """one shape in the environment."""

   def __init__(self, shape_id, x, y, size, color_name):
      self.shape_id   = shape_id
      self.x          = float(x)
      self.y          = float(y)
      self.size       = float(size)
      self.color_name = color_name
      self.color_rgb  = COLORS[color_name]
      self.radius     = int(SHAPE_RADIUS * size)

   def draw(self, surface, font):
      pygame.draw.circle(surface, self.color_rgb,
                         (int(self.x), int(self.y)), self.radius)
      label = font.render(f"{self.size:.1f}", True, (255, 255, 255))
      surface.blit(label, (int(self.x) - 10, int(self.y) - 8))

   def as_obs(self, target_x, target_y):
      max_dist     = np.sqrt(WINDOW_W ** 2 + WINDOW_H ** 2)
      dx_to_target = (target_x - self.x) / WINDOW_W   # signed: + means "go right"
      dy_to_target = (target_y - self.y) / WINDOW_H   # signed: + means "go down"
      dist         = np.sqrt((self.x - target_x) ** 2 + (self.y - target_y) ** 2)
      return np.array([
         self.x / WINDOW_W,
         self.y / WINDOW_H,
         (self.size - 0.5) / 1.5,
         COLOR_NAMES.index(self.color_name) / max(len(COLOR_NAMES) - 1, 1),
         dist / max_dist,
         dx_to_target,
         dy_to_target,
      ], dtype=np.float32)


class ShapeEnv(gym.Env):
   """
   gymnasium environment for shape manipulation via relative nudges.

   goal dict (injected externally, LLM plugs in here):
      {"task": "sort_by_size", "axis": "x", "direction": "ascending"}
   """

   metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

   def __init__(self, n_shapes=4, goal=None, render_mode=None):
      super().__init__()

      self.n_shapes    = n_shapes
      self.render_mode = render_mode

      self.goal = goal or {
         "task":      "sort_by_size",
         "axis":      "x",
         "direction": "ascending",
      }

      # 7 obs values per shape + 2 for goal encoding + 4 for action history
      obs_size = self.n_shapes * 7 + 2 + 4
      self.observation_space = spaces.Box(
         low=-2.0, high=2.0,   # relaxed bounds to safely fit all normalized values
         shape=(obs_size,),
         dtype=np.float32,
      )

      # [shape_selector, dx, dy] all in [-1, 1]
      self.action_space = spaces.Box(
         low=-1.0, high=1.0,
         shape=(3,),
         dtype=np.float32,
      )

      self.shapes         = []
      self.target_pos     = []
      self.steps          = 0
      self.prev_dists     = []
      self.last_shape_idx = -1
      self.steps_on_shape = 0
      self.last_action_dx = 0.0
      self.last_action_dy = 0.0
      self.window         = None
      self.clock          = None
      self.font           = None

   # ------------------------------------------------------------------
   # gymnasium interface
   # ------------------------------------------------------------------

   def reset(self, seed=None, options=None):
      super().reset(seed=seed)
      self.steps          = 0
      self.last_shape_idx = -1
      self.steps_on_shape = 0
      self.last_action_dx = 0.0
      self.last_action_dy = 0.0
      self.shapes         = self._spawn_shapes()
      self.target_pos     = self._compute_targets()
      self.prev_dists     = self._compute_dists()
      return self._get_obs(), {}

   def step(self, action):
      self.steps += 1

      # map shape_selector from [-1, 1] to an index
      raw_idx   = float(action[0])
      shape_idx = int(np.clip(
         round((raw_idx + 1.0) / 2.0 * (self.n_shapes - 1)),
         0, self.n_shapes - 1
      ))

      # apply nudge, clamped to window bounds
      dx = float(action[1]) * MAX_NUDGE
      dy = float(action[2]) * MAX_NUDGE
      s  = self.shapes[shape_idx]
      s.x = float(np.clip(s.x + dx, MARGIN, WINDOW_W - MARGIN))
      s.y = float(np.clip(s.y + dy, MARGIN, WINDOW_H - MARGIN))

      new_dists = self._compute_dists()
      max_dist  = np.sqrt(WINDOW_W ** 2 + WINDOW_H ** 2)

      # 1. directional: did the moved shape get closer to its target?
      old_d       = self.prev_dists[shape_idx]
      new_d       = new_dists[shape_idx]
      directional = (old_d - new_d) / max_dist * 4.0

      # 2. consistency bonus: reward staying on the same shape while it still needs moving.
      #    produces the "hold and push" behavior.
      if shape_idx == self.last_shape_idx and new_d > SOLVE_TOLERANCE:
         consistency_bonus = 0.03
      else:
         consistency_bonus = 0.0

      # 3. switching bonus: reward moving to a new shape once the previous one is done.
      #    prevents camping on an already-solved shape.
      switch_bonus = 0.0
      if (self.last_shape_idx >= 0
            and shape_idx != self.last_shape_idx
            and self.prev_dists[self.last_shape_idx] <= SOLVE_TOLERANCE):
         switch_bonus = 0.15

      # 4. camp penalty: penalize staying on a shape that's already at its target.
      if shape_idx == self.last_shape_idx and old_d <= SOLVE_TOLERANCE:
         camp_penalty = -0.05
      else:
         camp_penalty = 0.0

      # 5. overshoot penalty: penalize moving away from the target.
      overshoot_penalty = min(0.0, (old_d - new_d) / max_dist) * 2.0

      # update history before building obs
      self.steps_on_shape = (self.steps_on_shape + 1
                             if shape_idx == self.last_shape_idx else 1)
      self.last_shape_idx = shape_idx
      self.last_action_dx = float(action[1])
      self.last_action_dy = float(action[2])
      self.prev_dists     = new_dists

      dist_reward = (directional + consistency_bonus + switch_bonus
                     + camp_penalty + overshoot_penalty)

      terminated = self._is_solved()
      reward     = dist_reward + STEP_PENALTY
      if terminated:
         reward += COMPLETION_BONUS

      truncated = self.steps >= MAX_STEPS
      obs       = self._get_obs()
      info      = {"score": self._compute_score(), "steps": self.steps}

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

   # ------------------------------------------------------------------
   # internal helpers
   # ------------------------------------------------------------------

   def _spawn_shapes(self):
      rng         = self.np_random
      shapes      = []
      used_colors = []
      for i in range(self.n_shapes):
         x          = rng.uniform(MARGIN, WINDOW_W - MARGIN)
         y          = rng.uniform(MARGIN, WINDOW_H - MARGIN)
         size       = rng.uniform(0.5, 2.0)
         available  = [c for c in COLOR_NAMES if c not in used_colors]
         if not available:
            available = COLOR_NAMES
         color_name = available[rng.integers(0, len(available))]
         used_colors.append(color_name)
         shapes.append(Shape(i, x, y, size, color_name))
      return shapes

   def _compute_targets(self):
      """
      compute the ideal (x, y) for each shape given the current goal.

      for sort_by_size ascending on x:
         smallest shape goes to leftmost evenly-spaced x position,
         largest goes to rightmost, all at vertical center.
      """
      task      = self.goal.get("task", "sort_by_size")
      axis      = self.goal.get("axis", "x")
      direction = self.goal.get("direction", "ascending")

      if task == "sort_by_size":
         sizes = [s.size for s in self.shapes]
         order = np.argsort(sizes)
         if direction == "descending":
            order = order[::-1]

         n   = self.n_shapes
         pad = MARGIN * 2
         if axis == "x":
            xs = np.linspace(pad, WINDOW_W - pad, n)
            ys = np.full(n, WINDOW_H / 2)
         else:
            xs = np.full(n, WINDOW_W / 2)
            ys = np.linspace(pad, WINDOW_H - pad, n)

         targets = [(0.0, 0.0)] * n
         for rank, shape_idx in enumerate(order):
            targets[shape_idx] = (float(xs[rank]), float(ys[rank]))
         return targets

      # fallback: no movement needed
      return [(s.x, s.y) for s in self.shapes]

   def _compute_dists(self):
      """euclidean distance from each shape to its target."""
      return [
         float(np.sqrt(
            (self.shapes[i].x - self.target_pos[i][0]) ** 2 +
            (self.shapes[i].y - self.target_pos[i][1]) ** 2
         ))
         for i in range(self.n_shapes)
      ]

   def _compute_score(self):
      """0-1 progress metric for logging. 1.0 = all shapes at targets."""
      max_dist = np.sqrt(WINDOW_W ** 2 + WINDOW_H ** 2)
      avg_dist = float(np.mean(self._compute_dists()))
      return 1.0 - avg_dist / max_dist

   def _is_solved(self):
      """true when every shape is within SOLVE_TOLERANCE of its target."""
      return all(d <= SOLVE_TOLERANCE for d in self._compute_dists())

   def _get_obs(self):
      shape_obs = np.concatenate([
         self.shapes[i].as_obs(*self.target_pos[i])
         for i in range(self.n_shapes)
      ])
      history = np.array([
         self.last_shape_idx / max(self.n_shapes - 1, 1),  # normalized to [0, 1]; -1 maps to negative
         self.steps_on_shape / 10.0,                        # normalized; caps meaningfully at 10
         self.last_action_dx,                               # already in [-1, 1]
         self.last_action_dy,
      ], dtype=np.float32)
      return np.concatenate([shape_obs, self._encode_goal(), history]).astype(np.float32)

   def _encode_goal(self):
      axis_val = 0.0 if self.goal.get("axis", "x") == "x" else 1.0
      dir_val  = 0.0 if self.goal.get("direction", "ascending") == "ascending" else 1.0
      return np.array([axis_val, dir_val], dtype=np.float32)

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

      # draw ghost circles at target positions
      for i, (tx, ty) in enumerate(self.target_pos):
         pygame.draw.circle(self.window, TARGET_COLOR,
                            (int(tx), int(ty)),
                            self.shapes[i].radius, 2)

      for shape in self.shapes:
         shape.draw(self.window, self.font)

      # hud overlay
      score = self._compute_score()
      goal  = self.goal
      hud   = (f"task: {goal['task']} | axis: {goal['axis']} | "
               f"dir: {goal['direction']}   "
               f"progress: {score:.2%}   step: {self.steps}")
      self.window.blit(
         self.font.render(hud, True, (200, 200, 200)), (10, 10)
      )

      if self.render_mode != "human":
         return np.transpose(
            pygame.surfarray.array3d(self.window), axes=(1, 0, 2)
         )
      return None


# ------------------------------------------------------------------
# utility
# ------------------------------------------------------------------

def _spearman_corr(a, b):
   """spearman rank correlation, returns [-1, 1]."""
   rank_a  = np.argsort(np.argsort(a)).astype(float)
   rank_b  = np.argsort(np.argsort(b)).astype(float)
   ra_mean = rank_a.mean()
   rb_mean = rank_b.mean()
   num     = ((rank_a - ra_mean) * (rank_b - rb_mean)).sum()
   denom   = (np.sqrt(((rank_a - ra_mean) ** 2).sum()) *
              np.sqrt(((rank_b - rb_mean) ** 2).sum()))
   if denom == 0:
      return 0.0
   return num / denom