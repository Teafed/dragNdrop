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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WINDOW_W     = 800
WINDOW_H     = 600
MAX_SHAPES   = 5
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

SUPPORTED_TASKS = ["sort_by_size", "group_by_color", "cluster"]
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

    def __init__(self, n_shapes: int = 4, goal: dict = None, render_mode: str = None):
        super().__init__()

        self.n_shapes    = n_shapes
        self.render_mode = render_mode

        self.goal = goal or {
            "task":      "sort_by_size",
            "axis":      "x",
            "direction": "ascending",
            "attribute": "size",
        }

        # 7 obs values per shape + 3 goal encoding + 4 action history
        obs_size = self.n_shapes * 7 + 3 + 4
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
        rng         = self.np_random
        shapes      = []
        used_colors = []
        task        = self.goal.get("task", "sort_by_size")

        for i in range(self.n_shapes):
            x = rng.uniform(MARGIN, WINDOW_W - MARGIN)
            if task == "sort_by_size":
                y = WINDOW_H / 2 + rng.uniform(-40, 40)
            else:
                y = rng.uniform(MARGIN, WINDOW_H - MARGIN)

            size       = rng.uniform(0.5, 2.0)
            available  = [c for c in COLOR_NAMES if c not in used_colors]
            if not available:
                available = COLOR_NAMES
            color_name = available[rng.integers(0, len(available))]
            used_colors.append(color_name)
            shapes.append(Shape(i, x, y, size, color_name))

        return shapes

    def _compute_targets(self) -> list:
        task = self.goal.get("task", "sort_by_size")
        if task == "sort_by_size":
            return self._targets_sort_by_size()
        elif task in ("group_by_color", "cluster"):
            return self._targets_group_by_color()
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
        Shapes within the same colour group get a small horizontal stagger
        so they don't perfectly overlap.
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

        color_counts  = {c: 0 for c in unique_colors}
        targets       = []
        OFFSET_SPREAD = 40

        for shape in self.shapes:
            ax, ay = anchors[shape.color_name]
            idx    = color_counts[shape.color_name]
            offset_x = (idx - 0.5) * OFFSET_SPREAD
            targets.append((
                float(np.clip(ax + offset_x, MARGIN, WINDOW_W - MARGIN)),
                float(np.clip(ay, MARGIN, WINDOW_H - MARGIN)),
            ))
            color_counts[shape.color_name] += 1

        return targets

    def _compute_dists(self) -> list:
        return [
            float(np.sqrt(
                (self.shapes[i].x - self.target_pos[i][0]) ** 2 +
                (self.shapes[i].y - self.target_pos[i][1]) ** 2
            ))
            for i in range(self.n_shapes)
        ]

    def _compute_rank_corr(self) -> float:
        """
        Global ordering/grouping quality signal.

        sort_by_size   → Spearman rank correlation ∈ [-1, 1].
        group_by_color → Combined cohesion score ∈ [0, 1].
                         Rewards same-color shapes being close AND
                         different-color shapes being far apart.
                         This makes the signal non-zero even with 2 shapes
                         of different colors (the most common training setup).
        """
        task = self.goal.get("task", "sort_by_size")

        if task == "sort_by_size":
            axis = self.goal.get("axis", "x")
            if axis == "x":
                current_pos = [s.x for s in self.shapes]
                target_pos  = [self.target_pos[i][0] for i in range(self.n_shapes)]
            else:
                current_pos = [s.y for s in self.shapes]
                target_pos  = [self.target_pos[i][1] for i in range(self.n_shapes)]
            return _spearman_corr(current_pos, target_pos)

        elif task in ("group_by_color", "cluster"):
            return self._group_cohesion_score()

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
        shape_obs = np.concatenate([
            self.shapes[i].as_obs(*self.target_pos[i])
            for i in range(self.n_shapes)
        ])
        history = np.array([
            self.last_shape_idx / max(self.n_shapes - 1, 1),
            min(self.steps_on_shape / 10.0, 2.0),
            self.last_action_dx,
            self.last_action_dy,
        ], dtype=np.float32)
        return np.concatenate(
            [shape_obs, self._encode_goal(), history]
        ).astype(np.float32)

    def _encode_goal(self) -> np.ndarray:
        task_val = (TASK_IDX.get(self.goal.get("task", "sort_by_size"), 0)
                    / max(len(SUPPORTED_TASKS) - 1, 1))
        axis_val = {"x": 0.0, "none": 0.5, "y": 1.0}.get(
            self.goal.get("axis", "x"), 0.0)
        dir_val  = {"ascending": 0.0, "none": 0.5, "descending": 1.0}.get(
            self.goal.get("direction", "ascending"), 0.0)
        return np.array([task_val, axis_val, dir_val], dtype=np.float32)

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