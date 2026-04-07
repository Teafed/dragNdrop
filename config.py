# config.py
# shared constants for training, demo, callbacks, debug, and the environment.
# single source of truth for architecture dimensions.

# ---------------------------------------------------------------------------
# goal embedding
# ---------------------------------------------------------------------------

EMBEDDING_MODEL   = "all-MiniLM-L6-v2"
EMBEDDING_DIM     = 384

# ---------------------------------------------------------------------------
# observation space
# ---------------------------------------------------------------------------

MAX_SHAPES           = 6    # maximum shapes any episode can have
OBS_VALUES_PER_SHAPE = 5    # per-shape features: x, y, size, color, shape_type

# obs vector layout (428-dim total):
#   [0-3]    cursor state: cx_norm, cy_norm, holding, grabbed_idx_norm
#   [4-8]    grabbed shape features (zeros if nothing grabbed)
#   [9-13]   nearest non-grabbed shape features (zeros if no shapes)
#   [14-43]  all shapes zero-padded (MAX_SHAPES * OBS_VALUES_PER_SHAPE)
#   [44-427] goal embedding (EMBEDDING_DIM)
#
# left stream  (cursor-local):  indices  0-43  (44 values)
# right stream (scene-global):  indices 14-427 (413 values)
# overlap on [14-43] is intentional — both streams see all shape positions.

CURSOR_STATE_DIM    = 4    # cx_norm, cy_norm, holding, grabbed_idx_norm
FOCAL_SHAPE_DIM     = OBS_VALUES_PER_SHAPE * 2   # grabbed + nearest (5 + 5)

ALL_SHAPES_DIM      = MAX_SHAPES * OBS_VALUES_PER_SHAPE   # 6 * 5 = 30

LEFT_STREAM_DIM     = CURSOR_STATE_DIM + FOCAL_SHAPE_DIM + ALL_SHAPES_DIM
# = 4 + 10 + 30 = 44

RIGHT_STREAM_DIM     = ALL_SHAPES_DIM + EMBEDDING_DIM   # shapes + goal = 30 + 384 = 414

OBS_REGIONS = [
   ("cursor_state",   slice(0,   4),  "cx cy holding grabbed_idx"),
   ("grabbed_shape",  slice(4,   9),  "grabbed shape features"),
   ("nearest_shape",  slice(9,   14), "nearest shape features"),
   ("all_shapes",     slice(14,  44), "all 6 shapes zero-padded"),
   ("goal_embedding", slice(44, 428), "384-dim llm goal projection"),
]

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
   total flattened observation vector size: 384.

      CURSOR_STATE_DIM                           4
    + FOCAL_SHAPE_DIM  (grabbed + nearest)      10
    + MAX_SHAPES * OBS_VALUES_PER_SHAPE         30
    + EMBEDDING_DIM                            384
                                               ---
                                               428
   """
   return (CURSOR_STATE_DIM + FOCAL_SHAPE_DIM
           + MAX_SHAPES * OBS_VALUES_PER_SHAPE
           + EMBEDDING_DIM)

def get_action_size() -> int:
   """
   cx + cy + grip = 3
   """
   return 3