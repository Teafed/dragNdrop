# config.py
# shared constants for training, demo, callbacks, debug, and the environment.
# single source of truth for architecture dimensions.

# ---------------------------------------------------------------------------
# goal embedding
# ---------------------------------------------------------------------------

EMBEDDING_MODEL   = "all-MiniLM-L6-v2"
EMBEDDING_DIM     = 384
GOAL_ENCODING_DIM = 64   # GoalEncoder MLP output: 384 -> 128 -> 64

# ---------------------------------------------------------------------------
# observation space
# ---------------------------------------------------------------------------

MAX_SHAPES           = 6    # maximum shapes any episode can have
OBS_VALUES_PER_SHAPE = 5    # per-shape features: x, y, size, color, shape_type

# obs vector layout (108-dim total):
#   [0-3]    cursor state: cx_norm, cy_norm, holding, grabbed_idx_norm
#   [4-8]    grabbed shape features (zeros if nothing grabbed)
#   [9-13]   nearest non-grabbed shape features (zeros if no shapes)
#   [14-43]  all shapes zero-padded (MAX_SHAPES * OBS_VALUES_PER_SHAPE)
#   [44-107] goal encoding (GOAL_ENCODING_DIM)
#
# left stream  (cursor-local):  indices  0-43  (44 values)
# right stream (scene-global):  indices 14-107 (94 values)
# overlap on [14-43] is intentional — both streams see all shape positions.

CURSOR_STATE_SIZE    = 4    # cx_norm, cy_norm, holding, grabbed_idx_norm
FOCAL_SHAPE_SIZE     = OBS_VALUES_PER_SHAPE * 2   # grabbed + nearest (5 + 5)

LEFT_STREAM_DIM      = CURSOR_STATE_SIZE + FOCAL_SHAPE_SIZE + MAX_SHAPES * OBS_VALUES_PER_SHAPE
# = 4 + 10 + 30 = 44
RIGHT_STREAM_DIM     = MAX_SHAPES * OBS_VALUES_PER_SHAPE + GOAL_ENCODING_DIM   # shapes + goal = 94

# ---------------------------------------------------------------------------
# shape types and colors
# ---------------------------------------------------------------------------

SHAPE_TYPES    = ["circle", "square", "triangle"]
N_SHAPE_TYPES  = len(SHAPE_TYPES)
SHAPE_TYPE_IDX = {s: i for i, s in enumerate(SHAPE_TYPES)}
SHAPE_COLORS   = ["red", "green", "teal", "yellow", "purple"]
N_SHAPE_COLORS = len(SHAPE_COLORS)
SHAPE_COLOR_IDX = {s: i for i, s in enumerate(SHAPE_COLORS)}

# ---------------------------------------------------------------------------
# goal parsing
# ---------------------------------------------------------------------------

GOAL_SCHEMA = {
   "task":         str,
   "axis":         str,
   "direction":    str,
   "attribute":    str,
   "region":       str,
   "bounded":      bool,
   "target_color": str,   # "none" | "any" | color name (reach/touch/drag)
   "target_type":  str,   # "none" | "any" | shape type  (reach/touch/drag)
}

VALID_COLORS = ("red", "green", "teal", "yellow", "purple", "any", "none")
VALID_TYPES  = ("circle", "square", "triangle", "any", "none")

SUPPORTED_TASKS = [
   # starter tasks
   "reach",
   "touch",
   "drag",
   # arrangement tasks
   "arrange_in_sequence",
   "arrange_in_line",
   "arrange_in_region",
   "arrange_in_groups",
   "none",
]

# ---------------------------------------------------------------------------
# policy architecture
# ---------------------------------------------------------------------------

POLICY_HIDDEN_SIZE = 256   # hidden layer width for PPO and BC MLPs
N_ENVS             = 4     # parallel envs for PPO rollout collection

# ---------------------------------------------------------------------------
# cursor physics
# ---------------------------------------------------------------------------

CURSOR_SPEED    = 15    # pixels per step — kept < GRIP_RADIUS (20) to prevent single-step overshoot
GRIP_THRESHOLD  = 0.0   # action[2] > this -> holding = True
GRIP_RADIUS     = 20    # pixels — cursor must be within this to attach to shape

# ---------------------------------------------------------------------------
# obs and action size helpers
# ---------------------------------------------------------------------------

def get_obs_size() -> int:
   """
   total flattened observation vector size: 108.

      CURSOR_STATE_SIZE                          4
    + FOCAL_SHAPE_SIZE  (grabbed + nearest)     10
    + MAX_SHAPES * OBS_VALUES_PER_SHAPE         30
    + GOAL_ENCODING_DIM                         64
                                               ---
                                               108
   """
   return (CURSOR_STATE_SIZE + FOCAL_SHAPE_SIZE
           + MAX_SHAPES * OBS_VALUES_PER_SHAPE
           + GOAL_ENCODING_DIM)

def get_action_size() -> int:
   """
   cx + cy + grip = 3
   """
   return 3