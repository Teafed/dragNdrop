# config.py
# shared constants for training, demo, callbacks, debug, and the environment.
# this is the single source of truth for architecture dimensions — if you
# change OBS_VALUES_PER_SHAPE or ACTION_HISTORY_SIZE, the obs space in
# shape_env.py updates automatically via get_obs_size().

# ---------------------------------------------------------------------------
# observation space constants
# ---------------------------------------------------------------------------

MAX_SHAPES           = 6   # maximum shapes any episode can have
OBS_VALUES_PER_SHAPE = 7   # per-shape features: x, y, size, color, dist, dx, dy
ACTION_HISTORY_SIZE  = 4   # last_shape_idx, steps_on_shape, last_dx, last_dy

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
   """
   return MAX_SHAPES * OBS_VALUES_PER_SHAPE + GOAL_ENCODING_DIM + ACTION_HISTORY_SIZE

# ---------------------------------------------------------------------------
# task pool
# ---------------------------------------------------------------------------
# each entry is a natural language prompt that parse_goal() and get_embedding()
# both understand. training samples from this list randomly each episode.
#
# wave 1 tasks:
#   - sort_by_size    (4 variants covering both axes and both directions)
#   - group_by_color  (2 phrasings for prompt diversity)
#   - arrange_in_line (horizontal and vertical)
#   - arrange_in_grid (rectangular n_shapes only for now)
#   - push_to_region  (4 canvas regions)
#
# to add a task: add prompts here, add a _targets_<task>() method in
# shape_env.py, add a branch in _compute_targets(), and add oracle logic
# in oracle.py if the greedy strategy isn't sufficient.

TASK_POOL = [
   # sort_by_size
   "sort shapes from smallest to largest left to right",
   "sort shapes from largest to smallest left to right",
   "sort shapes from smallest to largest top to bottom",
   "sort shapes from largest to smallest top to bottom",

   # group_by_color
   "group shapes by color",
   "put shapes of the same color close together",

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
# kept so any code that still imports these names doesn't break during
# the wave 1 transition. remove in wave 2.

N_SHAPES_START = 2
N_SHAPES_MAX   = MAX_SHAPES
