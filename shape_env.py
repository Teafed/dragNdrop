"""
shape_env.py

Gymnasium environment for 2D shape manipulation via relative nudges.
The agent pushes shapes incrementally — like a hand, not a teleporter.

--- action space ---
    [shape_selector, dx, dy]
    shape_selector ∈ [-1, 1] → mapped to a shape index
    dx, dy         ∈ [-1, 1] → scaled by MAX_NUDGE pixels per step

--- observation space ---
    per shape : [x_norm, y_norm, size_norm, color_idx_norm,
                 dist_to_target_norm, dx_to_target, dy_to_target]
    goal      : [task_idx_norm, axis_norm, direction_norm]   ← 3 values (was 2)
    history   : [last_shape_idx_norm, steps_on_shape_norm, last_dx, last_dy]

--- supported tasks ---
    sort_by_size   – sort shapes along x or y by size, ascending or descending
    group_by_color – cluster same-color shapes into distinct canvas regions
    cluster        – general spatial clustering (currently mirrors group_by_color)

--- reward design ---
    1. weighted directional reward  – progress toward target ∝ urgency
    2. rank/group correlation delta – global ordering / cohesion signal
    3. per-shape solved bonus       – reward for staying in place once solved
    4. neglect penalty              – prevents ignoring distant shapes
    5. oscillation penalty          – discourages bouncing back and forth
    6. wall penalty                 – discourages pushing into borders
    7. inactivity penalty           – discourages stalling / zero nudges
    8. camp penalty                 – discourages sitting on already-solved shapes
    9. completion bonus             – large reward when ALL shapes solved
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame

from config import (
    MAX_SHAPES, OBS_VALUES_PER_SHAPE, ACTION_HISTORY_SIZE,
    GOAL_ENCODING_DIM, get_obs_size,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WINDOW_W     = 800
WINDOW_H     = 600
SHAPE_RADIUS = 20
FPS          = 60
MAX_STEPS    = 500
MAX_NUDGE    = 25
MARGIN       = SHAPE_RADIUS * 2

SOLVE_TOLERANCE = 60    # pixels — shape is "solved" when closer than this
GROUP_TOLERANCE = 80    # pixels — radius for same-color group clustering

STEP_PENALTY     = -0.02
COMPLETION_BONUS = 25.0

MOVEMENT_THRESHOLD = 0.5    # pixels — below this = not moving
PROGRESS_THRESHOLD = 1.0    # pixels — below this = no progress
NEGLECT_THRESHOLD  = 80.0   # pixels — shapes beyond this accumulate neglect
NEGLECT_PATIENCE   = 15     # steps before neglect penalty fires

COLORS = {
    "red":    (220,  60,  60),
    "green":  ( 60, 180,  60),
    "blue":   ( 60, 100, 220),
    "yellow": (220, 200,  50),
    "purple": (160,  60, 200),
}
COLOR_NAMES  = list(COLORS.keys())
BG_COLOR     = (30, 30, 35)
TARGET_COLOR = (80, 80, 80)

SUPPORTED_TASKS = [
    "sort_by_size",
    "group_by_color",
    "cluster",
    "arrange_in_line",
    "arrange_in_grid",
    "push_to_region",
]
TASK_IDX = {t: i for i, t in enumerate(SUPPORTED_TASKS)}


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------

class Shape:
    """A single movable circle in the environment."""

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

    def as_obs(self, target_x, target_y) -> np.ndarray:
        max_dist     = np.sqrt(WINDOW_W ** 2 + WINDOW_H ** 2)
        dx_to_target = (target_x - self.x) / WINDOW_W
        dy_to_target = (target_y - self.y) / WINDOW_H
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


# ---------------------------------------------------------------------------
# ShapeEnv
# ---------------------------------------------------------------------------

class ShapeEnv(gym.Env):
    """
    Gymnasium environment for 2D shape manipulation.

    Accepts a goal dict produced by llm_goal_parser.parse_goal():
        sort_by_size   : {"task":"sort_by_size",   "axis":"x", "direction":"ascending", "attribute":"size"}
        group_by_color : {"task":"group_by_color", "axis":"none", "direction":"none",   "attribute":"color"}
        cluster        : {"task":"cluster",         "axis":"none", "direction":"none",   "attribute":"size"}
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

    def __init__(self, n_shapes: int = None, goal: dict = None,
                 goal_embedding: np.ndarray = None, render_mode: str = None):
        super().__init__()

        # n_shapes=None means sample randomly each episode up to MAX_SHAPES.
        # passing a fixed value overrides this (useful for oracle/BC collection).
        self._fixed_n_shapes = n_shapes
        self.n_shapes        = n_shapes if n_shapes is not None else 2
        self.render_mode     = render_mode

        self.goal = goal or {
            "task":      "sort_by_size",
            "axis":      "x",
            "direction": "ascending",
            "attribute": "size",
            "region":    "none",
        }

        # goal_embedding: pre-computed EMBEDDING_DIM vector from get_embedding().
        # stored as GOAL_ENCODING_DIM zeros until set — the goal encoder MLP
        # in bc_train.py/train.py projects this down before it enters the policy.
        # shape_env stores the already-projected GOAL_ENCODING_DIM vector so the
        # obs space is always the same size regardless of raw embedding dim.
        self._goal_encoding = (
            goal_embedding[:GOAL_ENCODING_DIM].astype(np.float32)
            if goal_embedding is not None
            else np.zeros(GOAL_ENCODING_DIM, dtype=np.float32)
        )

        # fixed obs size regardless of n_shapes — unused shape slots are zero-padded.
        # formula lives in config.get_obs_size() so it's always in sync.
        obs_size = get_obs_size()
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0,
            shape=(obs_size,),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(3,),
            dtype=np.float32,
        )

        self.shapes            = []
        self.target_pos        = []
        self.steps             = 0
        self.prev_dists        = []
        self.last_shape_idx    = -1
        self.steps_on_shape    = 0
        self.last_action_dx    = 0.0
        self.last_action_dy    = 0.0
        self.dist_history      = []
        self.steps_since_moved = []
        self.prev_rank_corr    = 0.0

        self.window = None
        self.clock  = None
        self.font   = None

    # -----------------------------------------------------------------------
    # Gymnasium interface
    # -----------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # sample n_shapes for this episode if not fixed at construction time
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
        self.target_pos        = self._compute_targets()
        self.prev_dists        = self._compute_dists()
        self.dist_history      = [(d, d) for d in self.prev_dists]
        self.steps_since_moved = [0] * self.n_shapes
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

        new_dists = self._compute_dists()
        max_dist  = np.sqrt(WINDOW_W ** 2 + WINDOW_H ** 2)
        old_d     = self.prev_dists[shape_idx]
        new_d     = new_dists[shape_idx]
        improvement = old_d - new_d

        # 1. WEIGHTED DIRECTIONAL REWARD
        urgency_weight = old_d / max(max_dist, 1.0)
        directional    = improvement / max_dist * 5.0 * (1.0 + urgency_weight)

        # 2. RANK / GROUP CORRELATION DELTA
        new_rank_corr   = self._compute_rank_corr()
        rank_corr_delta = new_rank_corr - self.prev_rank_corr
        rank_reward     = rank_corr_delta * 2.0

        # 3. PER-SHAPE SOLVED BONUS
        solved_bonus = sum(0.015 for d in new_dists if d <= SOLVE_TOLERANCE)

        # 4. NEGLECT PENALTY
        neglect_penalty = 0.0
        for i in range(self.n_shapes):
            if (i != shape_idx
                    and new_dists[i] > NEGLECT_THRESHOLD
                    and self.steps_since_moved[i] > NEGLECT_PATIENCE):
                neglect_penalty -= 0.003 * (new_dists[i] / max_dist)

        # 5. OSCILLATION PENALTY
        prev_improvement = (self.dist_history[shape_idx][0]
                            - self.dist_history[shape_idx][1])
        if prev_improvement > PROGRESS_THRESHOLD and improvement < -PROGRESS_THRESHOLD:
            oscillation_penalty = -0.06
        else:
            oscillation_penalty = 0.0

        # 6. WALL PENALTY
        if intended_move > 5.0 and actual_move < intended_move * 0.25:
            wall_penalty = -0.05
        else:
            wall_penalty = 0.0

        # 7. INACTIVITY PENALTY
        inactivity_penalty = -0.04 if actual_move < MOVEMENT_THRESHOLD else 0.0

        # 8. CAMP PENALTY
        if shape_idx == self.last_shape_idx and old_d <= SOLVE_TOLERANCE:
            camp_penalty = -0.08
        else:
            camp_penalty = 0.0

        # bookkeeping
        self.dist_history = [
            (self.prev_dists[i], new_dists[i]) for i in range(self.n_shapes)
        ]
        for i in range(self.n_shapes):
            self.steps_since_moved[i] = 0 if i == shape_idx else self.steps_since_moved[i] + 1

        self.steps_on_shape = (self.steps_on_shape + 1
                               if shape_idx == self.last_shape_idx else 1)
        self.last_shape_idx = shape_idx
        self.last_action_dx = float(action[1])
        self.last_action_dy = float(action[2])
        self.prev_dists     = new_dists
        self.prev_rank_corr = new_rank_corr

        dist_reward = (directional + rank_reward + solved_bonus
                       + neglect_penalty + oscillation_penalty
                       + wall_penalty + inactivity_penalty + camp_penalty)

        terminated = self._is_solved()
        reward     = dist_reward + STEP_PENALTY
        if terminated:
            reward += COMPLETION_BONUS

        truncated = self.steps >= MAX_STEPS
        obs       = self._get_obs()
        info      = {
            "score":     self._compute_score(),
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

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _spawn_shapes(self) -> list:
        rng   = self.np_random
        task  = self.goal.get("task", "sort_by_size")

        # sample colors freely — repeats are fine and expected for group tasks.
        # the old used_colors logic tried to keep colors unique, which ran out
        # at n_shapes > len(COLOR_NAMES) and produced unpredictable assignments.
        color_indices = rng.integers(0, len(COLOR_NAMES), size=self.n_shapes)

        shapes = []
        for i in range(self.n_shapes):
            x = rng.uniform(MARGIN, WINDOW_W - MARGIN)
            if task == "sort_by_size":
                y = WINDOW_H / 2 + rng.uniform(-40, 40)
            else:
                y = rng.uniform(MARGIN, WINDOW_H - MARGIN)

            size       = rng.uniform(0.5, 2.0)
            color_name = COLOR_NAMES[color_indices[i]]
            shapes.append(Shape(i, x, y, size, color_name))

        return shapes

    def _compute_targets(self) -> list:
        task = self.goal.get("task", "sort_by_size")
        if task == "sort_by_size":
            return self._targets_sort_by_size()
        elif task in ("group_by_color", "cluster"):
            return self._targets_group_by_color()
        elif task == "arrange_in_line":
            return self._targets_arrange_in_line()
        elif task == "arrange_in_grid":
            return self._targets_arrange_in_grid()
        elif task == "push_to_region":
            return self._targets_push_to_region()
        else:
            return [(s.x, s.y) for s in self.shapes]

    def _targets_sort_by_size(self) -> list:
        axis      = self.goal.get("axis", "x")
        direction = self.goal.get("direction", "ascending")
        n         = self.n_shapes
        pad       = MARGIN * 2

        sizes = [s.size for s in self.shapes]
        order = np.argsort(sizes)
        if direction == "descending":
            order = order[::-1]

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

    def _targets_group_by_color(self) -> list:
        """
        Assign each unique colour a distinct anchor region on the canvas.
        Same-color shapes are staggered in a small grid around their anchor
        so they don't overlap even when a group has 3-4 members.
        """
        unique_colors = list(dict.fromkeys(s.color_name for s in self.shapes))
        n_colors      = len(unique_colors)
        pad           = MARGIN * 3

        if n_colors == 1:
            anchors = {unique_colors[0]: (WINDOW_W / 2, WINDOW_H / 2)}
        elif n_colors == 2:
            anchors = {
                unique_colors[0]: (WINDOW_W * 0.25, WINDOW_H / 2),
                unique_colors[1]: (WINDOW_W * 0.75, WINDOW_H / 2),
            }
        elif n_colors == 3:
            anchors = {
                unique_colors[0]: (WINDOW_W * 0.2,  WINDOW_H / 2),
                unique_colors[1]: (WINDOW_W * 0.5,  WINDOW_H / 2),
                unique_colors[2]: (WINDOW_W * 0.8,  WINDOW_H / 2),
            }
        elif n_colors == 4:
            anchors = {
                unique_colors[0]: (WINDOW_W * 0.25, WINDOW_H * 0.3),
                unique_colors[1]: (WINDOW_W * 0.75, WINDOW_H * 0.3),
                unique_colors[2]: (WINDOW_W * 0.25, WINDOW_H * 0.7),
                unique_colors[3]: (WINDOW_W * 0.75, WINDOW_H * 0.7),
            }
        else:
            xs = np.linspace(pad, WINDOW_W - pad, n_colors)
            anchors = {c: (float(xs[i]), WINDOW_H / 2)
                       for i, c in enumerate(unique_colors)}

        # per-group stagger offsets in a small grid so members don't overlap.
        # pattern: (0,0), (-1,0), (+1,0), (0,-1), (0,+1), ...
        # SPREAD controls spacing between members within a group.
        SPREAD  = 50
        OFFSETS = [
            ( 0,  0),
            (-1,  0),
            ( 1,  0),
            ( 0, -1),
            ( 0,  1),
            (-1, -1),
            ( 1,  1),
        ]

        color_counts = {c: 0 for c in unique_colors}
        targets      = []

        for shape in self.shapes:
            ax, ay = anchors[shape.color_name]
            idx    = color_counts[shape.color_name]
            ox, oy = OFFSETS[idx % len(OFFSETS)]
            targets.append((
                float(np.clip(ax + ox * SPREAD, MARGIN, WINDOW_W - MARGIN)),
                float(np.clip(ay + oy * SPREAD, MARGIN, WINDOW_H - MARGIN)),
            ))
            color_counts[shape.color_name] += 1

        return targets

    def _targets_arrange_in_line(self) -> list:
        """
        evenly spaced line across the canvas.
        axis="x" → horizontal line at vertical center.
        axis="y" → vertical line at horizontal center.
        """
        axis = self.goal.get("axis", "x")
        pad  = MARGIN * 2
        n    = self.n_shapes

        if axis == "x":
            xs = np.linspace(pad, WINDOW_W - pad, n)
            return [(float(x), float(WINDOW_H / 2)) for x in xs]
        else:
            ys = np.linspace(pad, WINDOW_H - pad, n)
            return [(float(WINDOW_W / 2), float(y)) for y in ys]

    def _targets_arrange_in_grid(self) -> list:
        """
        rectangular grid layout. finds the nearest rectangle to n_shapes,
        places a partial row at the bottom if n_shapes isn't a perfect rectangle.
        shapes are assigned to grid slots in their current left-to-right order
        so the agent doesn't have to cross shapes to reach targets.
        """
        n   = self.n_shapes
        pad = MARGIN * 2

        # find cols x rows such that cols*rows >= n and cols >= rows
        cols = max(1, int(np.ceil(np.sqrt(n))))
        rows = max(1, int(np.ceil(n / cols)))

        xs = np.linspace(pad, WINDOW_W - pad, cols)
        ys = np.linspace(pad, WINDOW_H - pad, rows)

        # assign shapes to slots left-to-right, top-to-bottom
        targets = []
        for i in range(n):
            row = i // cols
            col = i  % cols
            targets.append((float(xs[col]), float(ys[row])))
        return targets

    def _targets_push_to_region(self) -> list:
        """
        pack all shapes into one half of the canvas.
        within the region, shapes are arranged in a line so they don't overlap.
        """
        region = self.goal.get("region", "left")
        n      = self.n_shapes
        pad    = MARGIN * 2

        if region == "left":
            xs = np.linspace(pad, WINDOW_W * 0.35, n)
            ys = np.full(n, WINDOW_H / 2)
        elif region == "right":
            xs = np.linspace(WINDOW_W * 0.65, WINDOW_W - pad, n)
            ys = np.full(n, WINDOW_H / 2)
        elif region == "top":
            xs = np.full(n, WINDOW_W / 2)
            ys = np.linspace(pad, WINDOW_H * 0.35, n)
        else:   # bottom
            xs = np.full(n, WINDOW_W / 2)
            ys = np.linspace(WINDOW_H * 0.65, WINDOW_H - pad, n)

        return [(float(xs[i]), float(ys[i])) for i in range(n)]

    def _compute_dists(self) -> list:
        """distance from each active shape to its target, in pixels."""
        return [
            float(np.sqrt(
                (self.shapes[i].x - self.target_pos[i][0]) ** 2 +
                (self.shapes[i].y - self.target_pos[i][1]) ** 2
            ))
            for i in range(self.n_shapes)
        ]

    def _compute_rank_corr(self) -> float:
        """
        global ordering/grouping quality signal, task-aware.

        sort_by_size    → Spearman rank correlation ∈ [-1, 1].
        group_by_color  → combined cohesion score ∈ [0, 1].
        cluster         → same as group_by_color.
        arrange_in_line → Spearman correlation of position vs slot order ∈ [-1, 1].
        arrange_in_grid → fraction of shapes within SOLVE_TOLERANCE of their slot ∈ [0, 1].
        push_to_region  → fraction of shapes inside the target region ∈ [0, 1].
        """
        task = self.goal.get("task", "sort_by_size")

        if task == "sort_by_size":
            axis        = self.goal.get("axis", "x")
            current_pos = [s.x if axis == "x" else s.y for s in self.shapes]
            target_pos  = [self.target_pos[i][0] if axis == "x"
                           else self.target_pos[i][1]
                           for i in range(self.n_shapes)]
            return _spearman_corr(current_pos, target_pos)

        elif task in ("group_by_color", "cluster"):
            return self._group_cohesion_score()

        elif task == "arrange_in_line":
            axis        = self.goal.get("axis", "x")
            current_pos = [s.x if axis == "x" else s.y for s in self.shapes]
            target_pos  = [self.target_pos[i][0] if axis == "x"
                           else self.target_pos[i][1]
                           for i in range(self.n_shapes)]
            return _spearman_corr(current_pos, target_pos)

        elif task == "arrange_in_grid":
            dists  = self._compute_dists()
            solved = sum(1 for d in dists if d <= SOLVE_TOLERANCE)
            return solved / max(self.n_shapes, 1)

        elif task == "push_to_region":
            region    = self.goal.get("region", "left")
            in_region = 0
            for s in self.shapes:
                if   region == "left"   and s.x < WINDOW_W * 0.4:
                    in_region += 1
                elif region == "right"  and s.x > WINDOW_W * 0.6:
                    in_region += 1
                elif region == "top"    and s.y < WINDOW_H * 0.4:
                    in_region += 1
                elif region == "bottom" and s.y > WINDOW_H * 0.6:
                    in_region += 1
            return in_region / max(self.n_shapes, 1)

        return 0.0

    def _group_cohesion_score(self) -> float:
        """
        Combined grouping quality score ∈ [0, 1].

        Two components, averaged:
          - Intra-group cohesion: same-color shapes should be CLOSE.
            Score = 1 - (dist / max_dist) per same-color pair.
          - Inter-group separation: different-color shapes should be FAR.
            Score = dist / max_dist per different-color pair.

        This fixes the critical bug where 2 shapes of different colors
        produced zero same-color pairs, making the score always 0.0 and
        killing the rank_reward signal entirely for the most common setup.

        With n_shapes=2 and 2 different colors:
          - 0 same-color pairs → intra score = 0 (no pairs to average)
          - 1 different-color pair → inter score = dist/max_dist
          - Final score = inter score alone → rises as shapes separate.
        """
        max_dist      = np.sqrt(WINDOW_W ** 2 + WINDOW_H ** 2)
        intra_scores  = []
        inter_scores  = []

        for i in range(self.n_shapes):
            for j in range(i + 1, self.n_shapes):
                si = self.shapes[i]
                sj = self.shapes[j]
                d  = np.sqrt((si.x - sj.x) ** 2 + (si.y - sj.y) ** 2)

                if si.color_name == sj.color_name:
                    # Same color: reward being close
                    intra_scores.append(1.0 - d / max_dist)
                else:
                    # Different color: reward being far apart
                    inter_scores.append(d / max_dist)

        # Average whichever components are present
        components = []
        if intra_scores:
            components.append(float(np.mean(intra_scores)))
        if inter_scores:
            components.append(float(np.mean(inter_scores)))

        if not components:
            return 1.0   # single shape, trivially grouped
        return float(np.mean(components))

    def _compute_score(self) -> float:
        """0–1 distance-based progress. 1.0 = all shapes exactly at targets."""
        max_dist = np.sqrt(WINDOW_W ** 2 + WINDOW_H ** 2)
        avg_dist = float(np.mean(self._compute_dists()))
        return 1.0 - avg_dist / max_dist

    def _is_solved(self) -> bool:
        return all(d <= SOLVE_TOLERANCE for d in self._compute_dists())

    def _get_obs(self) -> np.ndarray:
        # active shape observations
        active_obs = np.concatenate([
            self.shapes[i].as_obs(*self.target_pos[i])
            for i in range(self.n_shapes)
        ])

        # zero-pad unused shape slots so obs size is always MAX_SHAPES * OBS_VALUES_PER_SHAPE
        n_padding  = MAX_SHAPES - self.n_shapes
        padding    = np.zeros(n_padding * OBS_VALUES_PER_SHAPE, dtype=np.float32)

        history = np.array([
            self.last_shape_idx / max(self.n_shapes - 1, 1),
            min(self.steps_on_shape / 10.0, 2.0),
            self.last_action_dx,
            self.last_action_dy,
        ], dtype=np.float32)

        # _goal_encoding is GOAL_ENCODING_DIM values set by set_goal_encoding()
        # or zeros if no embedding has been provided yet
        return np.concatenate(
            [active_obs, padding, self._goal_encoding, history]
        ).astype(np.float32)

    def set_goal_encoding(self, encoding: np.ndarray):
        """
        update the goal encoding used in observations.
        called by the training loop after projecting the raw embedding through
        the goal encoder MLP. encoding must be shape (GOAL_ENCODING_DIM,).
        """
        assert encoding.shape == (GOAL_ENCODING_DIM,), (
            f"expected encoding shape ({GOAL_ENCODING_DIM},), got {encoding.shape}"
        )
        self._goal_encoding = encoding.astype(np.float32)

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

        for i, (tx, ty) in enumerate(self.target_pos):
            pygame.draw.circle(self.window, TARGET_COLOR,
                               (int(tx), int(ty)),
                               self.shapes[i].radius, 2)

        for shape in self.shapes:
            shape.draw(self.window, self.font)

        score     = self._compute_score()
        rank_corr = self._compute_rank_corr()
        goal      = self.goal
        hud = (f"task: {goal['task']} | axis: {goal['axis']} | "
               f"dir: {goal['direction']}   "
               f"progress: {score:.2%}   "
               f"sort/group: {rank_corr:+.2f}   "
               f"step: {self.steps}")
        self.window.blit(
            self.font.render(hud, True, (200, 200, 200)), (10, 10)
        )

        if self.render_mode != "human":
            return np.transpose(
                pygame.surfarray.array3d(self.window), axes=(1, 0, 2)
            )
        return None


# ---------------------------------------------------------------------------
# Utility
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