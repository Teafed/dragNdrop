# dragNdrop

A gymnasium + stable-baselines3 proof of concept for an RL agent that
manipulates objects on a canvas in response to natural language goals.

The agent controls a cursor that navigates the canvas, grabs shapes, and
drags them to satisfy goal conditions. The goal is encoded from a free-text
prompt via a sentence embedding model.

---

## current status

**Wave 3 tasks are temporarily disabled while debugging starter task learning.**
Training currently runs only on `reach`, `touch`, and `drag` (1 shape, cursor
skill building). Once these solve reliably, wave 3 stages can be re-enabled by
uncommenting the relevant blocks in `config.py` and `curriculum.py`.

---

## install

```bash
pip install gymnasium stable-baselines3 pygame numpy torch \
            sentence-transformers tensorboard
```

---

## quickstart

**watch the oracle solve starter tasks (no training needed):**
```bash
python demo.py --oracle --prompt "move the cursor to the shape"
python demo.py --oracle --prompt "click on the shape"
python demo.py --oracle --prompt "drag the shape to the left side"
python demo.py --oracle --sequential   # cycle through active task pool prompts
```

**reach-task diagnostic demo (isolated, one shape only):**
```bash
python demo_reach.py --oracle                                  # oracle on reach (sanity check)
python demo_reach.py --model models/shape_agent/best_model    # trained agent on reach
python demo_reach.py --random                                  # random baseline
python demo_reach.py --oracle --headless --episodes 200       # stats only, no window
```

**demo keybinds (both demo.py and demo_reach.py):**
```
space    pause / unpause
n        skip to next episode
r        reset current episode  (demo_reach.py only)
d        dump full env + oracle state to console
s        step one frame (auto-pauses)
q        quit
```

**full training pipeline (oracle warm-start → BC → PPO):**
```bash
python train.py
python train.py --timesteps 400000 --bc-episodes 500 --bc-epochs 30
python train.py --force-demos          # re-collect oracle demos even if cached
```

**skip oracle warm-start:**
```bash
python train.py --no-oracle
python train.py --no-curriculum        # all active tasks from step 0
```

**watch a trained agent:**
```bash
python demo.py --model models/shape_agent/best_model
python demo_reach.py --model models/shape_agent/best_model    # reach only
```

**run diagnostics:**
```bash
python debug.py
python debug.py --oracle               # + oracle solve rates + BC loss
python debug.py --oracle-episodes 40
```

**run curriculum tester:**
```
# train with custom params
python curriculum_tester.py train TestCurriculum --timesteps 2000 --bc-episodes 10 --bc-epochs 2

# train without oracle (skip BC entirely)
python curriculum_tester.py train TestCurriculum --timesteps 2000 --no-oracle

# run the model
python curriculum_tester.py run TestCurriculum --episodes 5

# list all curriculums
python curriculum_tester.py list
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
curriculum.py       — staged curriculum manager (starter stages only while debugging)
train.py            — training entry point (oracle BC warm-start → PPO fine-tune)
callbacks.py        — SB3 callbacks: per-task solve rates, curriculum advancement
demo.py             — pygame demo: oracle or trained model, pause/dump controls
debug.py            — diagnostic script: env sanity, oracle solve rates, BC loss curve
sweep.py            — lightweight hyperparameter sweep over short training trials
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
ShapeEnv (Gymnasium)
    |   observation : 428-dim (cursor state + shape features + goal embedding)
    |   action      : [dx, dy, click]  all in [-1, 1]
    |   reward      : score delta + step penalty + completion bonus
    |
    v
BicameralPolicy (PPO)
    |-- left encoder   : obs[0:44]   -- cursor-local stream
    |-- right encoder  : obs[14:428] -- scene-global stream
    |-- cross-attention: right queries left (global reads cursor state)
    +-- action head    : concat(left, right) -> [dx, dy, click]
```

---

## observation space (428-dim)

```
[0-3]    cursor state         cx_norm, cy_norm, holding, grabbed_idx_norm
[4-8]    grabbed shape        features of currently held shape (zeros if none)
[9-13]   nearest free shape   features of closest non-grabbed shape
[14-43]  all shapes           6 x 5 values, zero-padded
[44-427] goal embedding       384-dim from sentence-transformers

per-shape features (5 values):
    x_norm, y_norm, size_norm, color_idx_norm, shape_type_norm

left stream  (cursor-local):  indices  0-43   (44 values)
right stream (scene-global):  indices 14-427  (414 values)
overlap on [14-43] is intentional.
```

---

## action space

```
[dx, dy, click]   all continuous in [-1, 1]

dx, dy:   cursor displacement this step, scaled by CURSOR_SPEED (15px)
click:     > CLICK_THRESHOLD (0.0) -> grab nearest shape within CLICK_RADIUS (20px)
          <= CLICK_THRESHOLD      -> release held shape
```

---

## active tasks

| task   | description                             | example prompt                   |
|--------|-----------------------------------------|----------------------------------|
| reach  | move cursor within click radius of shape | "move the cursor to the shape"   |
| touch  | activate click while overlapping shape   | "click on the shape"             |
| drag   | click and drag shape into a target region| "drag the shape to the left side"|

### disabled tasks (wave 3 — re-enable once starter tasks solve)

| task                  | description                              | example prompt                              |
|-----------------------|------------------------------------------|---------------------------------------------|
| arrange_in_sequence   | order shapes along axis by attribute     | "sort shapes smallest to largest"           |
| arrange_in_line       | place shapes in an evenly spaced line    | "arrange shapes in a horizontal line"       |
| arrange_in_region     | move all shapes into a canvas region     | "move all shapes to the left side"          |
| arrange_in_groups     | cluster shapes by color or type          | "group shapes by color"                     |

---

## curriculum (active stages)

Training builds cursor fundamentals before unlocking multi-shape tasks.
Wave 3 stages are commented out in `curriculum.py` until starter tasks
solve reliably.

| stage | tasks active              | n_shapes | gate | step ceiling |
|-------|---------------------------|----------|------|--------------|
| 0     | reach                     | 1        | 50%  | 30k          |
| 1     | touch                     | 1        | 50%  | 40k          |
| 2     | drag                      | 1        | 40%  | 60k          |
| 3     | reach + touch + drag      | 1        | —    | remaining    |

### disabled stages (arrangement)

| stage | tasks active                  | n_shapes | gate | step ceiling |
|-------|-------------------------------|----------|------|--------------|
| 3     | arrange_in_sequence           | 2-3      | 60%  | 150k         |
| 4     | + arrange_in_region           | 2-3      | 60%  | 150k         |
| 5     | + arrange_in_line             | 2-4      | 60%  | 200k         |
| 6     | + arrange_in_groups           | 2-3      | 40%  | 200k         |
| 7     | all tasks                     | 2-6      | —    | remaining    |

Advancement is performance-gated on the newest task's solve rate,
with a step ceiling so training never stalls on a hard stage.

---

## re-enabling wave 3

When starter tasks are solving reliably:

1. **`config.py`** — uncomment the wave 3 entries in `SUPPORTED_TASKS` and `TASK_POOL`.
2. **`curriculum.py`** — uncomment stages 3–7, remove the temporary "starter tasks (final)" stage,
   and change `_build_prompt_pool()` to use `TASK_POOL` instead of `_STARTER_TASK_POOL`.
3. Adjust `--timesteps` in `train.py` (800k+ recommended for the full curriculum).

---

## training strategy: oracle warm-start

Training from scratch is slow because the agent must simultaneously discover
cursor mechanics and task semantics. The oracle warm-start decouples these:

1. **oracle** — scripted policy using navigate-then-drag sub-phases:
   navigate → click_on → drag → click_off. solves all tasks analytically.
   demos saved to `logs/oracle_demos.npz` and reused across runs.

2. **behavior cloning** — supervised imitation of oracle demos trains the
   bicameral network. gives the policy a strong cursor-control prior before
   any RL signal is seen.

3. **PPO fine-tune** — refines the BC policy through environment interaction,
   handling spawn variance the deterministic oracle navigates perfectly.

---

## reward design

```
step reward  = score_delta x 10.0
             + rank_delta  x  2.0
             + STEP_PENALTY        (-0.02 per step)
             + inactivity_penalty  (-0.10 when cursor barely moves)
             + wall_penalty        (-0.05 when cursor hits margin)
             + click_bonus         (+0.10 when holding target, touch/drag only)
             + COMPLETION_BONUS    (+50.0 on solve)
```

Score functions per task:
- **reach**: two-zone proximity to target — continuous gradient from half-canvas distance all the way into CLICK_RADIUS.
- **touch**: same two-zone proximity as reach, jumps to 1.0 only when holding AND overlapping.
- **drag**: phase 1 = cursor proximity to shape (guides navigation + click); phase 2 = shape proximity to region boundary (guides dragging).

---

## oracle design

The oracle uses a navigate-then-drag loop per task:

- **reach**: navigate cursor until within CLICK_RADIUS of target. No click needed.
- **touch**: navigate to target, activate click (CLICK_ON phase), release.
- **drag**: navigate to target, click, drag to a point 30–120px past the region boundary, release.

Sub-phases per committed shape: `NAVIGATE → CLICK_ON → DRAG → CLICK_OFF → DONE`.

---

## reach diagnostic demo (demo_reach.py)

`demo_reach.py` hard-locks the environment to the reach task with one shape.
Two extra rings are drawn around the target:

- **yellow ring** — visual highlight of the target shape
- **blue ring** — exact `CLICK_RADIUS` boundary (cursor centre must enter this to score 1.0)

The HUD shows `dist_px` (raw pixel distance), `score` (0–1), `holding`, and
oracle `phase` every frame — the key signals for diagnosing whether the agent
is closing distance, oscillating, or ignoring the target entirely.

---

## adding new tasks

1. add score function `_score_<task>()` in `shape_env.py`
2. add branch in `_compute_task_score()` to call it
3. add task name to `SUPPORTED_TASKS` and `TASK_POOL` in `config.py`
4. add goal parsing in `llm_goal_parser.py`
5. add oracle priority function `_priorities_<task>()` in `oracle.py`
6. add oracle target computation in `_compute_target()` in `oracle.py`
7. add a curriculum stage in `curriculum.py` if needed