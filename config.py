# config.py
# shared constants for training, demo, callbacks, debug, and the environment.
# single source of truth for architecture dimensions.
#
# *** SINGLE-SHAPE MODE: MAX_SHAPES = 1 ***
#
# *** TASK PROGRESSION ***
# rudimentary (no grip):
#   move_cardinal   navigate cursor to a target zone (cardinal/axis-aligned)
#   move_diagonal   navigate cursor to a target zone (diagonal/corner)
#   approach        shape exists, get within 2x GRIP_RADIUS, no grip
# grip builders (phantom zone, no real shape):
#   click_at        navigate to zone then fire grip once
#   hold_at         navigate to zone, hold grip for HOLD_AT_STEPS steps
# starter tasks (shape + grip):
#   reach           cursor within GRIP_RADIUS of shape
#   touch           grip shape while within GRIP_RADIUS
#   drag            grip shape and move into target region

# ---------------------------------------------------------------------------
# goal embedding
# ---------------------------------------------------------------------------

EMBEDDING_MODEL   = "all-MiniLM-L6-v2"
EMBEDDING_DIM     = 384

# ---------------------------------------------------------------------------
# observation space
# ---------------------------------------------------------------------------

MAX_SHAPES           = 1
OBS_VALUES_PER_SHAPE = 5    # x, y, size, color, shape_type

# obs vector layout (403-dim total):
#   [0-3]    cursor state: cx_norm, cy_norm, holding, grabbed_idx_norm
#   [4-8]    grabbed shape features (zeros if nothing grabbed)
#   [9-13]   nearest non-grabbed shape features (zeros if no shapes)
#   [14-18]  all shapes zero-padded (1 * 5 = 5)
#   [19-402] goal embedding (384)
#
# for phantom-zone tasks (move_cardinal, move_diagonal, click_at, hold_at):
#   shapes[0] is a phantom target zone — same obs layout, not rendered or
#   grippable. the network sees it identically to a real shape.
#
# left stream  (cursor-local):  indices  0-18  (19 values)
# right stream (scene-global):  indices 14-402 (389 values)

CURSOR_STATE_DIM = 4
FOCAL_SHAPE_DIM  = OBS_VALUES_PER_SHAPE * 2   # grabbed + nearest = 10
ALL_SHAPES_DIM   = MAX_SHAPES * OBS_VALUES_PER_SHAPE   # 5

LEFT_STREAM_DIM  = CURSOR_STATE_DIM + FOCAL_SHAPE_DIM + ALL_SHAPES_DIM  # 19
RIGHT_STREAM_DIM = ALL_SHAPES_DIM + EMBEDDING_DIM                        # 389

OBS_REGIONS = [
   ("cursor_state",   slice(0,   4),   "cx cy holding grabbed_idx"),
   ("grabbed_shape",  slice(4,   9),   "grabbed shape features"),
   ("nearest_shape",  slice(9,   14),  "nearest shape or phantom zone"),
   ("all_shapes",     slice(14,  19),  "shape / phantom zone zero-padded"),
   ("goal_embedding", slice(19,  403), "384-dim llm goal projection"),
]

# ---------------------------------------------------------------------------
# shape types and colors
# ---------------------------------------------------------------------------

SHAPE_TYPES    = ["circle", "square"]
N_SHAPE_TYPES  = len(SHAPE_TYPES)
SHAPE_TYPE_IDX = {s: i for i, s in enumerate(SHAPE_TYPES)}
SHAPE_COLORS   = ["red", "green", "teal", "yellow", "purple"]
N_SHAPE_COLORS = len(SHAPE_COLORS)
SHAPE_COLOR_IDX = {s: i for i, s in enumerate(SHAPE_COLORS)}

# ---------------------------------------------------------------------------
# goal schema
# ---------------------------------------------------------------------------

GOAL_SCHEMA = {
   "task":         str,
   "axis":         str,
   "direction":    str,
   "attribute":    str,
   "region":       str,
   "bounded":      bool,
   "target_color": str,
   "target_type":  str,
}

VALID_COLORS = ("red", "green", "teal", "yellow", "purple", "any", "none")
VALID_TYPES  = ("circle", "square", "triangle", "any", "none")

SUPPORTED_TASKS = [
   "move_cardinal",
   "move_diagonal",
   "approach",
   "click_at",
   "hold_at",
   "reach",
   "touch",
   "drag",
   "none",
]

# tasks that use a phantom target zone encoded as shapes[0]
# (not rendered, not grippable — just a position in obs space)
PHANTOM_ZONE_TASKS = {"move_cardinal", "move_diagonal", "click_at", "hold_at"}

# how many consecutive steps grip must be held for hold_at
HOLD_AT_STEPS = 10

# ---------------------------------------------------------------------------
# policy architecture
# ---------------------------------------------------------------------------

POLICY_HIDDEN_SIZE = 256
N_ENVS             = 4

# ---------------------------------------------------------------------------
# cursor physics
# ---------------------------------------------------------------------------

CURSOR_SPEED    = 15
GRIP_THRESHOLD  = 0.0
GRIP_RADIUS     = 40

# ---------------------------------------------------------------------------
# obs / action size
# ---------------------------------------------------------------------------

def get_obs_size() -> int:
   """403 = 4 + 10 + 5 + 384"""
   return (CURSOR_STATE_DIM + FOCAL_SHAPE_DIM
           + MAX_SHAPES * OBS_VALUES_PER_SHAPE
           + EMBEDDING_DIM)

def get_action_size() -> int:
   return 3