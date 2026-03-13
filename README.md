# dragNdrop

A gymnasium + stable-baselines3 proof of concept for an RL agent that
manipulates objects on a canvas in response to natural language goals.

The agent controls a cursor that navigates the canvas, grabs shapes, and
drags them to satisfy goal conditions. The goal is encoded from a free-text
prompt via a sentence embedding model.

---

## install

```bash
pip install gymnasium stable-baselines3 pygame numpy torch \
            sentence-transformers tensorboard
```

---

## quickstart

**watch the oracle solve tasks (no training needed):**
```bash
python demo.py --oracle --prompt "move all shapes to the left side"
python demo.py --oracle --prompt "arrange shapes in horizontal line"
python demo.py --oracle --prompt "group shapes by color"
python demo.py --oracle --sequential   # cycle through all task pool prompts
```

**demo keybinds:**
```
space    pause / unpause
n        skip to next episode
d        dump full env + oracle state to console
q        quit
```

**full training pipeline (oracle warm-start → BC → PPO):**
```bash
python train.py
python train.py --timesteps 800000 --bc-episodes 500 --bc-epochs 30 # same as default
python train.py --force-demos          # re-collect oracle demos even if cached
python train.py --start-stage 3        # skip straight to curriculum stage 3 tasks
```

**skip oracle warm-start:**
```bash
python train.py --no-oracle
python train.py --no-curriculum        # all tasks from step 0
```

**watch a trained agent:**
```bash
python demo.py --model models/shape_agent/best_model
python demo.py --model models/shape_agent/best_model --prompt "group shapes by color"
```

**run diagnostics:**
```bash
python debug.py
python debug.py --oracle --skip-render # includes oracle solve rates + BC loss
python debug.py --oracle-episodes 40 --skip-render
```

**run short hyperparameter trials**
```bash
python sweep.py                        # default 50k steps
python sweep.py --steps 30000
python sweep.py --out logs/sweep1      # custom output prefix
```

**tensorboard:**
```bash
tensorboard --logdir logs/tensorboard
```

---

## file overview

```
config.py           — shared constants: canvas size, obs dims, task lists
shape_env.py        — gymnasium environment: cursor, shapes, obs/action spaces, rewards
llm_goal_parser.py  — rule-based goal parser (task, axis, direction, attribute, region)
oracle.py           — scripted expert policy + demo collection for BC warm-start
bc_train.py         — behavior cloning trainer + bicameral network definition
curriculum.py       — staged curriculum manager (7 stages, performance-gated)
train.py            — training entry point (oracle BC warm-start → PPO fine-tune)
callbacks.py        — SB3 callbacks: per-task solve rates, curriculum advancement
demo.py             — pygame demo: oracle or trained model, pause/dump controls
debug.py            — diagnostic script: env sanity, oracle solve rates, BC loss curve
```

---

## architecture

```
user prompt
    |
    v
llm_goal_parser.parse_goal()
    |   rule-based pattern matching -> goal dict
    |
    v
GoalEncoder (MLP)
    |   sentence embedding (384-dim, all-MiniLM-L6-v2)
    |   -> 64-dim goal encoding
    |
    v
ShapeEnv (Gymnasium)
    |   observation : 108-dim (cursor state + shape features + goal encoding)
    |   action      : [dx, dy, grip]  all in [-1, 1]
    |   reward      : score delta + step penalty + completion bonus
    |
    v
BicameralPolicy (PPO)
    |-- left encoder   : obs[0:44]   -- cursor-local stream
    |-- right encoder  : obs[14:108] -- scene-global stream
    |-- cross-attention: right queries left (global reads cursor state)
    +-- action head    : concat(left, right) -> [dx, dy, grip]
```

---

## observation space (108-dim)

```
[0-3]    cursor state         cx_norm, cy_norm, holding, grabbed_idx_norm
[4-8]    grabbed shape        features of currently held shape (zeros if none)
[9-13]   nearest free shape   features of closest non-grabbed shape
[14-43]  all shapes           6 x 5 values, zero-padded
[44-107] goal encoding        64-dim from GoalEncoder

per-shape features (5 values):
    x_norm, y_norm, size_norm, color_idx_norm, shape_type_norm

left stream  (cursor-local):  indices  0-43   (44 values)
right stream (scene-global):  indices 14-107  (94 values)
overlap on [14-43] is intentional.
```

---

## action space

```
[dx, dy, grip]   all continuous in [-1, 1]

dx, dy:   cursor displacement this step, scaled by CURSOR_SPEED (25px)
grip:     > GRIP_THRESHOLD (0.0) -> grab nearest shape within GRIP_RADIUS (20px)
          <= GRIP_THRESHOLD       -> release held shape
```

---

## supported tasks

| task                  | description                              | example prompt                              |
|-----------------------|------------------------------------------|---------------------------------------------|
| reach                 | move cursor within grip radius of shape  | "move the cursor to the shape"              |
| touch                 | activate grip while overlapping shape    | "click on the shape"                        |
| drag                  | grip and drag shape into a target region | "drag the shape to the left side"           |
| arrange_in_sequence   | order shapes along axis by attribute     | "sort shapes smallest to largest"           |
| arrange_in_line       | place shapes in an evenly spaced line    | "arrange shapes in a horizontal line"       |
| arrange_in_region     | move all shapes into a canvas region     | "move all shapes to the left side"          |
| arrange_in_groups     | cluster shapes by color or type          | "group shapes by color"                     |

---

## curriculum (7 stages)

Training builds from cursor fundamentals up to full multi-shape tasks:

| stage | tasks active                  | n_shapes | gate | step ceiling |
|-------|-------------------------------|----------|------|--------------|
| 0     | reach                         | 1        | 80%  | 50k          |
| 1     | touch                         | 1        | 80%  | 50k          |
| 2     | drag                          | 1        | 70%  | 75k          |
| 3     | arrange_in_sequence           | 2-3      | 60%  | 150k         |
| 4     | + arrange_in_region           | 2-3      | 60%  | 150k         |
| 5     | + arrange_in_line             | 2-4      | 60%  | 200k         |
| 6     | all tasks                     | 2-6      | --   | remaining    |

Advancement is performance-gated on the newest task's solve rate,
with a step ceiling so training never stalls on a hard stage.

To skip ahead: `python train.py --start-stage 3`

---

## training strategy: oracle warm-start

Training from scratch is slow because the agent must simultaneously discover
cursor mechanics and task semantics. The oracle warm-start decouples these:

1. oracle - scripted policy using navigate-then-drag sub-phases:
   navigate -> grip_on -> drag -> grip_off. solves all tasks analytically.
   demos saved to logs/oracle_demos.npz and reused across runs.

2. behavior cloning - supervised imitation of oracle demos trains the
   bicameral network. gives the policy a strong cursor-control prior before
   any RL signal is seen.

3. PPO fine-tune - refines the BC policy through environment interaction,
   handling spawn variance the deterministic oracle navigates perfectly.

---

## reward design

```
step reward  = score_delta x REWARD_SCALE
             + STEP_PENALTY (-0.02 per step)
             + COMPLETION_BONUS (+25.0 on solve)

score_delta  = current_score - previous_score
               (positive when shapes move toward goal, negative otherwise)
```

Score functions per task:
- reach / touch / drag: proximity or grip-activation score in [0, 1]
- arrange_in_sequence: spearman rank correlation of positions vs attribute values
- arrange_in_line: 0.6 x gap evenness + 0.4 x perpendicular spread score
- arrange_in_region: mean per-shape region penetration score
- arrange_in_groups: 0.5 x global inter/intra distance ratio
                   + 0.5 x per-shape nearest-neighbor isolation score

---

## oracle design

The oracle uses a zone-planning approach per task:

- sequence / line: assigns each shape to an ideal slot via greedy
  nearest-available-slot. priority is proportional to pixel distance from
  ideal position, so the most-displaced shape is always tackled first.

- region: targets a random depth 30-120px past the boundary so shapes
  land comfortably inside, not just over the threshold.

- groups: pre-assigns each unique attribute value to a canvas zone using
  greedy maximin (each new group gets the zone furthest from all existing
  groups). new attribute values discovered mid-episode get the next
  farthest unoccupied zone.

---

## adding new tasks

1. add score function \_score_<task>() in shape_env.py
2. add branch in _compute_task_score() to call it
3. add task name to SUPPORTED_TASKS and TASK_POOL in config.py
4. add goal parsing in llm_goal_parser.py
5. add oracle priority function \_priorities_<task>() in oracle.py
6. add oracle target computation in _compute_target() in oracle.py
7. add a curriculum stage in curriculum.py if needed
