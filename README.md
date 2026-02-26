# shape manipulation agent

A Gymnasium + Stable-Baselines3 proof of concept for an RL agent that
manipulates 2D shapes in response to natural language goals.

The LLM layer uses the Anthropic API (with stub fallback) — everything else
is wired and runnable with or without an API key.

---

## install

```bash
pip install "setuptools<71" gymnasium stable-baselines3 pygame numpy matplotlib tensorboard anthropic torch
```

---

## quickstart

**Recommended — oracle warm-start (fast, 5–10× less compute than PPO from scratch):**
```bash
python train.py --oracle
python train.py --oracle --prompt "group shapes by color"
python train.py --oracle --timesteps 100000
```

**PPO from scratch (original approach):**
```bash
python train.py
python train.py --prompt "sort shapes right to left"
python train.py --prompt "arrange top to bottom, biggest at top"
python train.py --timesteps 500000
```

**Watch a trained agent:**
```bash
python demo.py --model models/shape_agent/best_model
python demo.py --model models/shape_agent/best_model --prompt "sort right to left"
python demo.py --random   # watch a random agent as a baseline
```

**Behavior cloning only (no PPO fine-tune):**
```bash
python bc_train.py --prompt "sort shapes left to right"
python bc_train.py --prompt "group shapes by color" --episodes 800
python bc_train.py --finetune --finetune-steps 100000   # BC then PPO
```

**Run diagnostics:**
```bash
python debug.py
python debug.py --model models/shape_agent/best_model
python debug.py --skip-render   # headless / CI use
```

**TensorBoard:**
```bash
tensorboard --logdir logs/tensorboard
```

---

## file overview

```
llm_goal_parser.py — LLM goal parser (Anthropic API with stub fallback)
shape_env.py       — Gymnasium environment: shapes, obs/action spaces, rewards
train.py           — Training entry point (PPO from scratch or oracle warm-start)
callbacks.py       — TensorBoard callbacks logging task-specific metrics
demo.py            — Standalone pygame demo, loads a trained model or runs random
oracle.py          — Scripted expert policy + dataset collection for BC
bc_train.py        — Behavior cloning trainer (BC → PPO warm-start)
debug.py           — Diagnostic script to verify env, reward, and model behavior
```

---

## architecture

```
user prompt
    │
    ▼
llm_goal_parser.parse_goal()
    │   tries Anthropic API → falls back to pattern-matching stub
    │
    ▼
goal dict: {task, axis, direction, attribute}
    │
    ▼
ShapeEnv (Gymnasium)
    │  observation : [shape states] + [goal encoding (3 values)] + [action history]
    │  action      : [shape_selector, dx, dy]  all in [-1, 1]
    │  reward      : directional + rank/cohesion delta + penalties + completion bonus
    │
    ▼
Policy (one of):
    ├── PPO from scratch       — train.py (default)
    ├── Oracle → BC → PPO      — train.py --oracle  (recommended)
    └── Scripted oracle        — oracle.py (perfect, not learned)
```

---

## supported tasks

| Task | Description | Goal example |
|---|---|---|
| `sort_by_size` | Arrange shapes along x or y axis ordered by size | `"sort smallest to largest left to right"` |
| `group_by_color` | Cluster same-color shapes into distinct canvas regions | `"group shapes by colour"` |
| `cluster` | General spatial grouping by attribute | `"cluster shapes by size"` |

---

## training strategy: oracle warm-start

Training PPO from scratch on manipulation tasks is slow because the agent
must discover *what to do* and *how to do it* simultaneously.

The oracle warm-start decouples these:

1. **Oracle** — a scripted policy that solves the task analytically.
   Generates 500 episodes (~50 000 transitions) in ~10 seconds.

2. **Behavior Cloning** — supervised imitation of the oracle demos.
   Gives the policy a strong prior: "move the furthest-from-target shape
   toward its target."  Takes ~30 seconds.

3. **PPO fine-tune** — refine the BC policy with RL to handle edge cases
   the oracle navigates perfectly but a noisy real agent won't.

Typical result: oracle warm-start + 100k PPO steps ≈ PPO from scratch at 300k steps.

```bash
# Full oracle pipeline:
python train.py --oracle --timesteps 100000 --bc-episodes 500
```

---

## reward design

Eight components per step:

| Component | When it fires | Effect |
|---|---|---|
| Weighted directional | Always | +reward ∝ progress × urgency |
| Rank/cohesion delta | Always | +/- reward for global ordering improvement |
| Per-shape solved bonus | Each solved shape | Small persistent + per step |
| Neglect penalty | Shape ignored >15 steps AND far from target | Small - |
| Oscillation penalty | Reverses direction after progress | -0.06 |
| Wall penalty | Large intended nudge, tiny actual move | -0.05 |
| Inactivity penalty | Shape barely moved | -0.04 |
| Camp penalty | Staying on already-solved shape | -0.08 |
| Step penalty | Every step | -0.02 |
| Completion bonus | All shapes solved | +25.0 |

---

## observation space

```
[shape_0_obs, shape_1_obs, ..., goal_encoding, action_history]

per shape (7 values):
    x_norm, y_norm              — normalised position
    size_norm                   — normalised size
    color_idx_norm              — colour identity
    dist_to_target_norm         — how far from goal
    dx_to_target, dy_to_target  — direction to goal

goal encoding (3 values):  ← was 2 values; added task_idx
    task_idx_norm   — which task (sort / group / cluster)
    axis_norm       — x=0, none=0.5, y=1
    direction_norm  — ascending=0, none=0.5, descending=1

action history (4 values):
    last_shape_idx_norm, steps_on_shape_norm, last_dx, last_dy
```

---

## tensorboard metrics

The `task/` group contains the most useful training signals:

| Metric | What it means |
|---|---|
| `task/mean_score` | 0–1 distance-based progress. 1.0 = all shapes at targets |
| `task/rank_corr` | Spearman correlation (sort tasks) or group cohesion (group tasks) |
| `task/solve_rate` | Fraction of eval episodes fully solved (terminated) |
| `task/mean_ep_length` | Average steps to finish. Decreasing = agent getting faster |

> **Note:** `rank_corr` and `mean_score` are different metrics.  `mean_score`
> is distance-based (0–1).  `rank_corr` is order/cohesion-based (−1 to +1
> for sort, 0 to 1 for group).  Watch both.

---

## adding new tasks

1. Add a `_targets_<task>()` method in `ShapeEnv` that returns target positions.
2. Add a branch in `_compute_targets()` to call it.
3. Add a branch in `_compute_rank_corr()` for the quality signal.
4. Add the task name to `SUPPORTED_TASKS` in both `shape_env.py` and `llm_goal_parser.py`.
5. Update `_SYSTEM_PROMPT` in `llm_goal_parser.py` with examples.
6. Add pattern matching to `_stub_parse()` for the fallback.
7. Add an oracle strategy in `oracle.py` if the greedy approach isn't sufficient.

---

## wiring in a real LLM

The Anthropic API is already wired in `llm_goal_parser.py`.  Set the environment
variable and it activates automatically:

```bash
export ANTHROPIC_API_KEY=your_key_here
python train.py --prompt "put the small shapes on the left"
```

If the API call fails for any reason, the system falls back to pattern matching
and continues running — you'll see a `[goal_parser] LLM parse failed` message.

---

## expanding to real robot input

The trained agent outputs `[shape_selector, dx, dy]` actions in [−1, 1].
To route those to a real actuator instead of the pygame env:

```python
# swap:
obs, reward, terminated, truncated, info = env.step(action)

# for:
target_x = int((action[1] + 1) / 2 * SCREEN_W)
target_y = int((action[2] + 1) / 2 * SCREEN_H)
robot.move_to(target_x, target_y)
robot.nudge()
```

The policy doesn't change — only the actuator does.