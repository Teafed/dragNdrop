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
# shape types
# ---------------------------------------------------------------------------

SHAPE_TYPES    = ["circle", "square", "triangle"]
N_SHAPE_TYPES  = len(SHAPE_TYPES)
SHAPE_TYPE_IDX = {s: i for i, s in enumerate(SHAPE_TYPES)}
COLOR_NAMES_GOAL = ["red", "green", "teal", "yellow", "purple"]

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
# obs and action size helper
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

# ---------------------------------------------------------------------------
# task framework
#
# WAVE 3 TASKS DISABLED FOR STARTER TASK DEBUGGING.
# To re-enable: uncomment the wave 3 entries in SUPPORTED_TASKS and TASK_POOL,
# then restore the wave 3 stages in curriculum.py.
# ---------------------------------------------------------------------------

SUPPORTED_TASKS = [
   # starter tasks — ACTIVE
   "reach",
   "touch",
   "drag",
   # wave 3 tasks — DISABLED (comment back in when ready)
   # "arrange_in_sequence",
   # "arrange_in_line",
   # "arrange_in_region",
   # "arrange_in_groups",
]

# ---------------------------------------------------------------------------
# task pool
# ---------------------------------------------------------------------------

TASK_POOL = [
   # --- starter tasks (ACTIVE) ---
   "move the cursor to the shape",
   "move the cursor to the yellow shape",
   "move the cursor to the triangle",
   "click on the shape",
   "click on the red shape",
   "click on the square",
   "drag the shape to the left side",
   "drag the shape to the right side",
   "drag the shape to the top",
   "drag the shape to the bottom",

   # --- wave 3 tasks (DISABLED — uncomment to restore) ---
   # arrange_in_sequence
   # "sort shapes from smallest to largest left to right",
   # "sort shapes from largest to smallest left to right",
   # "sort shapes from smallest to largest top to bottom",
   # "sort shapes from largest to smallest top to bottom",

   # arrange_in_line
   # "arrange shapes in a horizontal line evenly spaced",
   # "arrange shapes in a vertical line evenly spaced",
   # "arrange shapes in a horizontal line sorted smallest to largest",
   # "arrange shapes in a vertical line sorted largest to smallest",

   # arrange_in_region
   # "move all shapes to the left side",
   # "move all shapes to the right side",
   # "move all shapes to the top",
   # "move all shapes to the bottom",

   # arrange_in_groups
   # "group shapes by color",
   # "put shapes of the same color close together",
   # "group shapes by type",
   # "put shapes of the same type close together",
   # "group the circles squares and triangles separately",
]