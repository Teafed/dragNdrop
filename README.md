# shape manipulation agent

A goal-conditioned reinforcement learning agent that arranges shapes on a 2D
canvas in response to natural language prompts. A single policy handles all
tasks simultaneously; the goal is encoded from the prompt text and injected
into the observation rather than training a separate specialist per task.

```
"sort shapes from smallest to largest"  →  agent sorts by size left to right
"group shapes by color"                 →  agent clusters by color
"arrange shapes in a grid"              →  agent positions shapes in a grid
"move all shapes to the left side"      →  agent pushes shapes into region
"group shapes by type"                  →  agent clusters circles, squares, triangles
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

**training pipeline:** oracle warm-start → behaviour cloning → PPO fine-tuning

---

## tasks

| task | reward type | description |
|---|---|---|
| `sort_by_size` | scoring | spearman rank correlation of positions vs sizes |
| `group_by_color` | scoring | nn-relative cohesion score by color attribute |
| `group_by_shape_type` | scoring | nn-relative cohesion score by shape type |
| `cluster` | scoring | alias for group_by_color |
| `arrange_in_line` | canonical | evenly spaced horizontal or vertical line |
| `arrange_in_grid` | canonical | rectangular grid layout |
| `push_to_region` | canonical | pack shapes into left/right/top/bottom half |

**scoring tasks** accept any valid solution — the reward is a score delta, not
distance to a fixed target. the episode terminates when score ≥ 0.85.

**canonical tasks** have a unique solution computed by `_compute_targets()`.
ghost circles are drawn at target positions during rendering for these tasks only.

---

## project structure

| file | purpose |
|---|---|
| `config.py` | single source of truth for all architecture constants |
| `shape_env.py` | gymnasium environment — shapes, rendering, reward functions |
| `llm_goal_parser.py` | prompt → structured goal dict + sentence embedding |
| `oracle.py` | analytical oracle policy + demo dataset collection |
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

all 9 tests should pass. key things it checks:
- obs shape is 98
- all 7 tasks initialise and step without error
- oracle achieves 80%+ solve rate on every task
- BC loss decreases over epochs

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
# single task
python demo.py --model models/shape_agent/best_model \
               --prompt "group shapes by color"

# cycle random tasks each episode
python demo.py --model models/shape_agent/best_model --multi-task

# cycle tasks in TASK_POOL order
python demo.py --model models/shape_agent/best_model --multi-task --sequential
```

**demo — oracle (useful for verifying oracle behavior):**
```bash
# random tasks
python demo.py --oracle

# sequential through all TASK_POOL prompts
python demo.py --oracle --sequential
```

press `N` in the oracle demo to skip to the next task. press `Q` to quit.

**tensorboard:**
```bash
tensorboard --logdir logs/tensorboard
```

---

## key config constants (`config.py`)

| constant | value | description |
|---|---|---|
| `MAX_SHAPES` | 6 | max shapes per episode |
| `OBS_VALUES_PER_SHAPE` | 5 | x, y, size, color, shape_type |
| `GOAL_ENCODING_DIM` | 64 | projected goal embedding size |
| `POLICY_HIDDEN_SIZE` | 256 | PPO/BC MLP hidden layer width |
| `EMBEDDING_DIM` | 384 | sentence-transformer output dim |
| `SCORE_SOLVE_THRESHOLD` | 0.85 | scoring task termination threshold |
| `SOLVE_TOLERANCE` | 60px | canonical task per-shape tolerance |

to change the obs size, edit `OBS_VALUES_PER_SHAPE` or `MAX_SHAPES` in
`config.py`. `get_obs_size()` updates automatically and propagates everywhere.

---

## development notes

**wave 1 (complete):** fixed-target distance rewards, oracle covers all tasks,
multi-task training pipeline, BC warm-start, per-task eval callbacks.

**wave 2 (current):** scoring-based rewards (any valid solution accepted),
removed target-relative obs features, added shape types (circle/square/triangle),
added `group_by_shape_type` task, oracle uses task-specific heuristics rather
than reading fixed `target_pos`, spawn diversity guarantees for group tasks.

**wave 3 (planned):** collision detection, physical constraints, or hierarchical
primitive decomposition depending on wave 2 agent weaknesses after further training.

**known issue:** trained policy collapses to moving a single shape. bc loss
plateaus at ~0.44 suggesting the warm-start isn't transferring multi-shape
behavior well. under investigation.
