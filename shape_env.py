"""
shape_env.py

a gymnasium environment for a 2d shape manipulation task.
shapes can be dragged to new positions. the agent's goal is
injected as a structured dict (where an LLM would plug in).
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame

# --- constants ---
WINDOW_W       = 800
WINDOW_H       = 600
MAX_SHAPES     = 5
SHAPE_RADIUS   = 20   # used for drawing and rough collision
FPS            = 60
MAX_STEPS      = 500

# colors used for shapes (R, G, B)
COLORS = {
   "red":    (220,  60,  60),
   "green":  ( 60, 180,  60),
   "blue":   ( 60, 100, 220),
   "yellow": (220, 200,  50),
   "purple": (160,  60, 200),
}
COLOR_NAMES = list(COLORS.keys())

# background
BG_COLOR = (30, 30, 35)


class Shape:
   """
   represents one shape in the environment.
   size is a float in [0.5, 2.0] representing relative scale.
   """
   def __init__(self, shape_id, x, y, size, color_name):
      self.shape_id   = shape_id
      self.x          = float(x)
      self.y          = float(y)
      self.size       = float(size)   # relative scale
      self.color_name = color_name
      self.color_rgb  = COLORS[color_name]
      self.radius     = int(SHAPE_RADIUS * size)

   def draw(self, surface, font):
      pygame.draw.circle(surface, self.color_rgb,
                         (int(self.x), int(self.y)), self.radius)
      # draw a subtle size label for debugging
      label = font.render(f"{self.size:.1f}", True, (255, 255, 255))
      surface.blit(label, (int(self.x) - 10, int(self.y) - 8))

   def as_obs(self):
      """
      returns a flat numpy array of normalized observations:
      [x_norm, y_norm, size_norm, color_idx_norm]
      """
      return np.array([
         self.x / WINDOW_W,
         self.y / WINDOW_H,
         (self.size - 0.5) / 1.5,                       # normalize to [0,1]
         COLOR_NAMES.index(self.color_name) / (len(COLOR_NAMES) - 1),
      ], dtype=np.float32)


class ShapeEnv(gym.Env):
   """
   gymnasium environment for shape manipulation.

   observation space:
      flat array of shape observations (4 floats per shape) plus
      the goal encoding (2 floats: axis and direction).

   action space:
      [shape_idx, target_x_norm, target_y_norm]
      - shape_idx is continuous and gets rounded to select a shape
      - target coords are normalized to [0, 1]

   goal (injected externally, where an LLM would plug in):
      a dict like:
         {"task": "sort_by_size", "axis": "x", "direction": "ascending"}
   """

   metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

   def __init__(self, n_shapes=MAX_SHAPES, goal=None, render_mode=None):
      super().__init__()

      self.n_shapes    = n_shapes
      self.render_mode = render_mode

      # default goal — this is where LLM output would be injected
      self.goal = goal or {
         "task":      "sort_by_size",
         "axis":      "x",
         "direction": "ascending",
      }

      # 4 obs values per shape + 2 for goal encoding
      obs_size = self.n_shapes * 4 + 2
      self.observation_space = spaces.Box(
         low=0.0, high=1.0,
         shape=(obs_size,),
         dtype=np.float32,
      )

      # action: [shape_selector (0-1), target_x (0-1), target_y (0-1)]
      self.action_space = spaces.Box(
         low=0.0, high=1.0,
         shape=(3,),
         dtype=np.float32,
      )

      self.shapes   = []
      self.steps    = 0
      self.window   = None
      self.clock    = None
      self.font     = None

   # ------------------------------------------------------------------
   # gymnasium interface
   # ------------------------------------------------------------------

   def reset(self, seed=None, options=None):
      super().reset(seed=seed)

      self.steps = 0
      self.shapes = self._spawn_shapes()
      obs = self._get_obs()
      info = {}
      return obs, info

   def step(self, action):
      self.steps += 1

      # decode action
      shape_idx = int(np.clip(round(action[0] * (self.n_shapes - 1)),
                               0, self.n_shapes - 1))
      target_x  = float(action[1]) * WINDOW_W
      target_y  = float(action[2]) * WINDOW_H

      # clamp to window bounds with some padding
      pad = SHAPE_RADIUS * 2
      target_x = np.clip(target_x, pad, WINDOW_W - pad)
      target_y = np.clip(target_y, pad, WINDOW_H - pad)

      # move selected shape
      self.shapes[shape_idx].x = target_x
      self.shapes[shape_idx].y = target_y

      reward      = self._compute_reward()
      terminated  = self._is_solved()
      truncated   = self.steps >= MAX_STEPS
      obs         = self._get_obs()
      info        = {"reward": reward, "steps": self.steps}

      if self.render_mode == "human":
         self._render_frame()

      return obs, reward, terminated, truncated, info

   def render(self):
      if self.render_mode == "rgb_array":
         return self._render_frame()
      elif self.render_mode == "human":
         self._render_frame()

   def close(self):
      if self.window is not None:
         pygame.display.quit()
         pygame.quit()
         self.window = None

   # ------------------------------------------------------------------
   # internal helpers
   # ------------------------------------------------------------------

   def _spawn_shapes(self):
      """spawn shapes at random positions with random sizes and colors."""
      rng     = self.np_random
      shapes  = []
      used_colors = []

      for i in range(self.n_shapes):
         x          = rng.uniform(SHAPE_RADIUS * 2, WINDOW_W - SHAPE_RADIUS * 2)
         y          = rng.uniform(SHAPE_RADIUS * 2, WINDOW_H - SHAPE_RADIUS * 2)
         size       = rng.uniform(0.5, 2.0)
         # pick a color, prefer unique ones
         available  = [c for c in COLOR_NAMES if c not in used_colors]
         if not available:
            available = COLOR_NAMES
         color_name = available[rng.integers(0, len(available))]
         used_colors.append(color_name)

         shapes.append(Shape(i, x, y, size, color_name))

      return shapes

   def _get_obs(self):
      """build flat observation array."""
      shape_obs = np.concatenate([s.as_obs() for s in self.shapes])
      goal_obs  = self._encode_goal()
      return np.concatenate([shape_obs, goal_obs]).astype(np.float32)

   def _encode_goal(self):
      """
      encode the current goal as two floats.
      axis:      x=0.0, y=1.0
      direction: ascending=0.0, descending=1.0
      """
      axis_val = 0.0 if self.goal.get("axis", "x") == "x" else 1.0
      dir_val  = 0.0 if self.goal.get("direction", "ascending") == "ascending" else 1.0
      return np.array([axis_val, dir_val], dtype=np.float32)

   def _compute_reward(self):
      """
      reward based on how well x-positions correlate with sizes.
      uses spearman-style rank correlation as a smooth signal.

      returns a float in roughly [-1, 1].
      higher is better.
      """
      task = self.goal.get("task", "sort_by_size")

      if task == "sort_by_size":
         axis      = self.goal.get("axis", "x")
         direction = self.goal.get("direction", "ascending")

         positions = np.array([s.x if axis == "x" else s.y
                                for s in self.shapes])
         sizes     = np.array([s.size for s in self.shapes])

         if direction == "descending":
            sizes = -sizes

         # rank correlation
         rank_corr = _spearman_corr(positions, sizes)

         # small penalty for y spread (encourage shapes to stay in a line)
         if axis == "x":
            y_positions = np.array([s.y for s in self.shapes])
            y_spread    = np.std(y_positions) / WINDOW_H
            line_bonus  = -0.2 * y_spread
         else:
            x_positions = np.array([s.x for s in self.shapes])
            x_spread    = np.std(x_positions) / WINDOW_W
            line_bonus  = -0.2 * x_spread

         return float(rank_corr + line_bonus)

      # fallback for unknown tasks
      return 0.0

   def _is_solved(self):
      """
      episode is solved if reward is above threshold for current task.
      threshold chosen to mean "pretty well sorted and roughly in a line."
      """
      return self._compute_reward() >= 0.85

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

      # draw goal text in corner
      goal_str  = (f"goal: {self.goal['task']} | "
                   f"axis: {self.goal['axis']} | "
                   f"dir: {self.goal['direction']}")
      reward    = self._compute_reward()
      label     = self.font.render(f"{goal_str}   reward: {reward:.3f}",
                                    True, (200, 200, 200))
      self.window.blit(label, (10, 10))

      if self.render_mode == "human":
         pygame.event.pump()
         pygame.display.flip()
         self.clock.tick(FPS)
         return None
      else:
         return np.transpose(
            pygame.surfarray.array3d(self.window), axes=(1, 0, 2)
         )


# ------------------------------------------------------------------
# utility
# ------------------------------------------------------------------

def _spearman_corr(a, b):
   """
   compute spearman rank correlation between two 1d arrays.
   returns a value in [-1, 1].
   """
   n      = len(a)
   rank_a = np.argsort(np.argsort(a)).astype(float)
   rank_b = np.argsort(np.argsort(b)).astype(float)
   # pearson on ranks
   ra_mean = rank_a.mean()
   rb_mean = rank_b.mean()
   num     = ((rank_a - ra_mean) * (rank_b - rb_mean)).sum()
   denom   = (np.sqrt(((rank_a - ra_mean) ** 2).sum()) *
              np.sqrt(((rank_b - rb_mean) ** 2).sum()))
   if denom == 0:
      return 0.0
   return num / denom
