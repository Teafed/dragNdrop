# config.py
# shared constants for training, demo, callbacks, debug, and the environment.
# this is the single source of truth for architecture dimensions — if you
# change OBS_VALUES_PER_SHAPE or ACTION_HISTORY_SIZE, the obs space in
# shape_env.py updates automatically via get_obs_size().

# ---------------------------------------------------------------------------
# observation space constants
# ---------------------------------------------------------------------------

MAX_SHAPES           = 6   # maximum shapes any episode can have
OBS_VALUES_PER_SHAPE = 5   # per-shape features: x, y, size, color, shape_type
ACTION_HISTORY_SIZE  = 4   # last_shape_idx, steps_on_shape, last_dx, last_dy

# shape types — encoded as a normalized float in obs: index / (N_SHAPE_TYPES - 1)
# 0.0 = circle, 0.5 = square, 1.0 = triangle
SHAPE_TYPES    = ["circle", "square", "triangle"]
N_SHAPE_TYPES  = len(SHAPE_TYPES)
SHAPE_TYPE_IDX = {s: i for i, s in enumerate(SHAPE_TYPES)}

# ---------------------------------------------------------------------------
# goal embedding
# ---------------------------------------------------------------------------

# sentence-transformers model used by llm_goal_parser.get_embedding().
# all-MiniLM-L6-v2 is ~80MB, runs offline, and produces 384-dim embeddings.
# changing this requires recomputing all cached embeddings and retraining.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM   = 384

# goal encoder MLP: projects EMBEDDING_DIM -> GOAL_ENCODING_DIM before
# concatenating with shape observations. keeps the policy input manageable.
GOAL_ENCODING_DIM = 64

# ---------------------------------------------------------------------------
# policy architecture
# ---------------------------------------------------------------------------

# hidden layer size for the PPO MLP policy.
# bumped from 64 to 256 to give the network enough capacity to condition
# on the goal encoding alongside shape observations.
POLICY_HIDDEN_SIZE = 256

# ---------------------------------------------------------------------------
# observation size helper
# ---------------------------------------------------------------------------

def get_obs_size() -> int:
   """
   total observation vector size for the policy input.
   called by shape_env.py and bc_train.py to keep obs sizes in sync.

      MAX_SHAPES slots (zero-padded for unused shapes)
      + GOAL_ENCODING_DIM (projected embedding, not raw 384-dim)
      + ACTION_HISTORY_SIZE

   wave 2: OBS_VALUES_PER_SHAPE = 5 (x, y, size, color, shape_type)
   wave 1 had 7 (also included dist_to_target, dx_to_target, dy_to_target).
   those are removed because wave 2 uses scoring-based rewards with no
   fixed targets, so target-relative features are meaningless.
   net change: 6*(7-5) = 12 fewer values. obs size 110 -> 98.
   """
   return MAX_SHAPES * OBS_VALUES_PER_SHAPE + GOAL_ENCODING_DIM + ACTION_HISTORY_SIZE

# ---------------------------------------------------------------------------
# task pool
# ---------------------------------------------------------------------------
# wave 1 tasks: sort_by_size, group_by_color, arrange_in_line,
#               arrange_in_grid, push_to_region, cluster
#
# wave 2 additions:
#   - group_by_shape: cluster shapes of the same type (circle/square/triangle)
#
# reward approach by task:
#   scoring-based (any valid solution rewarded):
#       sort_by_size, group_by_color, group_by_shape, cluster
#   canonical-target (unique solution, fixed targets kept for rendering):
#       arrange_in_line, arrange_in_grid, push_to_region

TASK_POOL = [
   # sort_by_size
   "sort shapes from smallest to largest left to right",
   "sort shapes from largest to smallest left to right",
   "sort shapes from smallest to largest top to bottom",
   "sort shapes from largest to smallest top to bottom",

   # group_by_color
   "group shapes by color",
   "put shapes of the same color close together",

   # group_by_shape_type (wave 2)
   "group shapes by type",
   "put shapes of the same type close together",
   "group the circles squares and triangles separately",

   # arrange_in_line
   "arrange shapes in a horizontal line evenly spaced",
   "arrange shapes in a vertical line evenly spaced",

   # arrange_in_grid
   "arrange shapes in a grid",

   # push_to_region
   "move all shapes to the left side",
   "move all shapes to the right side",
   "move all shapes to the top",
   "move all shapes to the bottom",
]

# ---------------------------------------------------------------------------
# legacy aliases
# ---------------------------------------------------------------------------
# kept so any code that still imports these names doesn't break.
# remove in wave 3.

N_SHAPES_START = 2
N_SHAPES_MAX   = MAX_SHAPES
