# config.py
# shared constants for training, demo, callbacks, debug, and the environment.
# single source of truth for architecture dimensions — changing OBS_VALUES_PER_SHAPE
# or ACTION_HISTORY_SIZE updates get_obs_size() automatically everywhere.

# ---------------------------------------------------------------------------
# observation space
# ---------------------------------------------------------------------------

MAX_SHAPES           = 6   # maximum shapes any episode can have
OBS_VALUES_PER_SHAPE = 5   # per-shape features: x, y, size, color, shape_type
ACTION_HISTORY_SIZE  = 4   # last_shape_idx, steps_on_shape, last_dx, last_dy

# shape types — encoded as normalized float: index / (N_SHAPE_TYPES - 1)
# 0.0 = circle, 0.5 = square, 1.0 = triangle
SHAPE_TYPES    = ["circle", "square", "triangle"]
N_SHAPE_TYPES  = len(SHAPE_TYPES)
SHAPE_TYPE_IDX = {s: i for i, s in enumerate(SHAPE_TYPES)}

# ---------------------------------------------------------------------------
# goal embedding
# ---------------------------------------------------------------------------

EMBEDDING_MODEL   = "all-MiniLM-L6-v2"
EMBEDDING_DIM     = 384
GOAL_ENCODING_DIM = 64   # GoalEncoder MLP output: 384 -> 128 -> 64

# ---------------------------------------------------------------------------
# policy architecture
# ---------------------------------------------------------------------------

POLICY_HIDDEN_SIZE = 256   # hidden layer width for PPO and BC MLPs

# ---------------------------------------------------------------------------
# obs size helper
# ---------------------------------------------------------------------------

def get_obs_size() -> int:
   """
   total flattened observation vector size.
   = MAX_SHAPES * OBS_VALUES_PER_SHAPE + GOAL_ENCODING_DIM + ACTION_HISTORY_SIZE
   = 6*5 + 64 + 4 = 98
   """
   return MAX_SHAPES * OBS_VALUES_PER_SHAPE + GOAL_ENCODING_DIM + ACTION_HISTORY_SIZE

# ---------------------------------------------------------------------------
# task framework — wave 3
# ---------------------------------------------------------------------------
#
# tasks are defined by three orthogonal binary dimensions:
#
#   n_target_spaces  one | many   — all shapes share a target, or each group
#                                    gets its own
#   bounded          no  | yes    — position within the target space is
#                                    unconstrained (rank only) or spatially
#                                    contained
#   ordered          no  | yes    — shapes placed in attribute order, or just
#                                    placed
#
#   task name              n_target_spaces  bounded  ordered
#   ─────────────────────  ───────────────  ───────  ───────
#   arrange_in_sequence    one              no       yes
#   arrange_in_line        one              yes      yes/no
#   arrange_in_region      one              yes      no
#   arrange_in_groups      many             yes      no
#
# goal dict parameters:
#   axis      "x" | "y" | "none"
#   direction "ascending" | "descending" | "none"
#   attribute "size" | "color" | "shape_type" | "none"
#   region    "left" | "right" | "top" | "bottom" | "none"
#   bounded   True | False

SUPPORTED_TASKS = [
   "arrange_in_sequence",   # one target space, unbounded, ordered by attribute
   "arrange_in_line",       # one target space, bounded (actual line), ordered or unordered
   "arrange_in_region",     # one target space, bounded (canvas subregion), unordered
   "arrange_in_groups",     # many target spaces (one per attribute value), bounded, unordered
]

# ---------------------------------------------------------------------------
# task pool
# ---------------------------------------------------------------------------

TASK_POOL = [
   # arrange_in_sequence
   "sort shapes from smallest to largest left to right",
   "sort shapes from largest to smallest left to right",
   "sort shapes from smallest to largest top to bottom",
   "sort shapes from largest to smallest top to bottom",

   # arrange_in_line
   "arrange shapes in a horizontal line evenly spaced",
   "arrange shapes in a vertical line evenly spaced",
   "arrange shapes in a horizontal line sorted smallest to largest",
   "arrange shapes in a vertical line sorted largest to smallest",

   # arrange_in_region
   "move all shapes to the left side",
   "move all shapes to the right side",
   "move all shapes to the top",
   "move all shapes to the bottom",

   # arrange_in_groups
   "group shapes by color",
   "put shapes of the same color close together",
   "group shapes by type",
   "put shapes of the same type close together",
   "group the circles squares and triangles separately",
]
