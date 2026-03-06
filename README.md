# dragNdrop

A goal-conditioned reinforcement learning agent that arranges shapes on a 2D
canvas in response to natural language prompts.

Currently, a single policy handles all tasks simultaneously; the goal is encoded
from the prompt and injected into the observation rather than training a separate
specialist per task.

```
"sort shapes from smallest to largest"          →  agent sorts by size left to right
"arrange shapes in a horizontal line"           →  agent forms a horizontal line
"move all shapes to the left side"              →  agent pushes shapes into region
"group shapes by color"                         →  agent clusters by color
"group circles squares & triangles separately"  →  agent separates by shape type
```

---

## architecture

```
natural language prompt
        │
        ▼
sentence-transformer (all-MiniLM-L6-v2, 384-dim, offline)
        │
        ▼
GoalEncoder MLP (384 → 128 → 64)
        │
        ├──────────────────────────────────────────┐
        │                                          │
        ▼                                          ▼
per-shape obs (up to 6 shapes,              goal encoding (64-dim)
zero-padded, 5 values each):
  x, y, size, color, shape_type
        │                                          │
        └──────────────┬───────────────────────────┘
                       │  + action history (4 values)
                       ▼
              obs vector (98-dim)
                       │
                       ▼
           PPO policy (MLP 256→256)
                       │
                       ▼
         action: [shape_selector, dx, dy] ∈ [-1, 1]³
```

**obs size breakdown:** `6 shapes × 5 features + 64 goal + 4 history = 98`

**training pipeline:** `oracle warm-start -> behavior cloning -> PPO fine-tuning`

---

## task framework

tasks are defined by three orthogonal binary dimensions, forming a 2×2×2 cube:

| dimension | values | meaning |
|---|---|---|
| n_target_spaces | one / many | all shapes share one target region, or each attribute group gets its own |
| bounded | no / yes | shapes must be spatially contained within the target, or just ordered |
| ordered | no / yes | shapes placed in attribute order within the target, or just placed |

the four active tasks occupy four cells of this cube:

| task | n_target_spaces | bounded | ordered | description |
|---|---|---|---|---|
| `arrange_in_sequence` | one | no | yes | ordered along an axis by attribute; perpendicular position unconstrained |
| `arrange_in_line` | one | yes | yes/no | ordered or evenly spaced along axis AND minimising perpendicular spread |
| `arrange_in_region` | one | yes | no | all shapes inside a canvas subregion, distributed across it |
| `arrange_in_groups` | many | yes | no | shapes partitioned by attribute, each group in its own subregion |

all tasks are **scoring-based**: the reward is a score delta (0→1) and any
valid solution is accepted. the episode terminates when score ≥ 0.85. there
are no fixed target positions anywhere in the codebase.

**per-shape scores** (averaged for the episode score):

- `arrange_in_sequence`: `1 - |current_rank - ideal_rank| / (n-1)` per shape
- `arrange_in_line`: 0.6 × order score + 0.4 × perpendicular spread score
- `arrange_in_region`: 0.7 × in_region + 0.3 × progress toward boundary
- `arrange_in_groups`: 0.6 × nearest-neighbor correct + 0.4 × separation score

**goal dict schema:**

```python
{
  "task":      str,    # one of SUPPORTED_TASKS
  "axis":      str,    # "x" | "y" | "none"
  "direction": str,    # "ascending" | "descending" | "none"
  "attribute": str,    # "size" | "color" | "shape_type" | "none"
  "region":    str,    # "left" | "right" | "top" | "bottom" | "none"
  "bounded":   bool,   # True for line/region/groups tasks
}
```

---

## oracle

the oracle uses an **explore / commit** loop rather than globally greedy
shape selection:

**explore:** select the next shape to move using weighted random selection.
the "most wrong" shape is most likely to be chosen but not guaranteed —
this produces varied demonstrations for behaviour cloning.

priority functions by task:
- `arrange_in_sequence/line`: `|current_rank − ideal_rank|` per shape
- `arrange_in_region`: distance outside the boundary per shape
- `arrange_in_groups`: per-shape cohesion deficit (1 − per_shape_score)

**commit:** move the selected shape toward its computed target until the
local completion condition is satisfied, then return to explore.

local completion conditions:
- sequence/line: shape is within `MAX_NUDGE × 1.5` pixels of its ideal slot
- region: shape is past the boundary and within `MAX_NUDGE × 1.5` of target
- groups: shape is within `MAX_NUDGE × 2.0` of committed group centroid

target positions include random jitter (`COMMIT_JITTER = 30px`) so
demonstrations are varied rather than all landing on the same pixel.

---

## project structure

| file | purpose |
|---|---|
| `config.py` | single source of truth for all architecture constants and task pool |
| `shape_env.py` | gymnasium environment — shapes, scoring functions, reward, rendering |
| `llm_goal_parser.py` | prompt → structured goal dict + sentence-transformer embedding |
| `oracle.py` | explore/commit oracle policy + demonstration collection |
| `bc_train.py` | GoalEncoder, BCPolicy, behaviour cloning training loop |
| `train.py` | full training pipeline entry point |
| `demo.py` | pygame demo — trained agent, random agent, or oracle |
| `callbacks.py` | SB3 eval callback with per-task metrics |
| `debug.py` | diagnostic test suite (9 tests) |

---

## setup

```bash
pip install stable-baselines3 gymnasium pygame sentence-transformers torch numpy

# first run downloads all-MiniLM-L6-v2 (~80MB) and caches it locally.
# set this to avoid network calls on subsequent runs:
export TRANSFORMERS_OFFLINE=1
```

---

## usage

**run diagnostic tests (always do this before training):**
```bash
python debug.py --oracle --skip-render
```

all tests should pass. key checks:
- obs shape is 98
- all 4 tasks initialise and step without error
- oracle achieves 100% solve rate on sequence/line/region, ≥50% on groups
- BC loss decreases over epochs and selector loss stays below 0.2

**train:**
```bash
# full pipeline: oracle collection → BC warm-start → PPO fine-tuning
python train.py --bc-episodes 500 --bc-epochs 30 --timesteps 500000

# skip oracle warm-start (faster, worse initialisation)
python train.py --no-oracle --timesteps 500000
```

checkpoints are saved to `./models/shape_agent/`. the best model by eval
reward is saved as `best_model`, the final model as `final_model`.

**demo — trained agent:**
```bash
# single fixed task
python demo.py --model models/shape_agent/best_model \
               --prompt "group shapes by color"

# cycle random tasks each episode
python demo.py --model models/shape_agent/best_model --multi-task
```

**demo — oracle:**
```bash
# fixed task (useful for inspecting oracle behavior on one task)
python demo.py --oracle --prompt "arrange shapes in a horizontal line evenly spaced"

# random tasks each episode
python demo.py --oracle

# cycle through all TASK_POOL prompts in order
python demo.py --oracle --sequential
```

press `N` in the oracle demo to skip to the next task. press `Q` to quit.

**tensorboard:**
```bash
tensorboard --logdir logs/tensorboard
```

---

## key config constants

| constant | value | description |
|---|---|---|
| `MAX_SHAPES` | 6 | max shapes per episode |
| `OBS_VALUES_PER_SHAPE` | 5 | x, y, size, color, shape_type |
| `GOAL_ENCODING_DIM` | 64 | projected goal embedding size |
| `POLICY_HIDDEN_SIZE` | 256 | PPO/BC MLP hidden layer width |
| `EMBEDDING_DIM` | 384 | sentence-transformer output dim |
| `SCORE_SOLVE_THRESHOLD` | 0.85 | episode solved when mean per-shape score ≥ this |
| `EXPLORE_TEMP` | 0.5 | oracle softmax temperature (lower = more greedy) |
| `COMMIT_JITTER` | 30px | random offset added to oracle target positions |

to change the obs size, edit `OBS_VALUES_PER_SHAPE` or `MAX_SHAPES` in
`config.py`. `get_obs_size()` updates automatically everywhere.

---

## development notes

**wave 1 (complete):** fixed-target distance rewards, oracle covers all tasks,
multi-task training pipeline, BC warm-start, per-task eval callbacks.

**wave 2 (complete):** scoring-based rewards (any valid solution accepted),
removed target-relative obs features, added shape types (circle/square/triangle),
oracle heuristics rather than fixed targets, spawn diversity for group tasks.

**wave 3 (current):** consolidated 7 tasks into 4 using the 2×2×2 cube framework
(n_target_spaces × bounded × ordered). explore/commit oracle with stochastic shape
selection. all tasks scoring-based with per-shape scores. `bounded` field added to
goal schema.

**wave 4 (planned):** two-stream policy architecture — separate streams for the
shape being moved and the relational context (where should it go relative to others),
with cross-attention between streams. addresses the implicit nature of dual attention
in the current flat MLP.
