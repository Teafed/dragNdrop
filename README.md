# shape manipulation agent — proof of concept

a gymnasium + stable-baselines3 proof of concept for an RL agent
that manipulates 2d shapes in response to a natural language goal.
the LLM layer is stubbed out — everything else is wired and runnable.

---

## install

```
pip install "setuptools<71" gymnasium stable-baselines3 pygame numpy matplotlib tensorboard
```

---

## run

**train from scratch:**
```
python train.py
```

**train with a custom prompt:**
```
python train.py --prompt "sort shapes right to left"
python train.py --prompt "arrange top to bottom, biggest at top"
```

**train longer:**
```
python train.py --timesteps 500000
```

**watch a trained agent:**
```
python demo.py --model models/shape_agent/best_model
```

**watch a random agent:**
```
python demo.py --random
```

**run diagnostics:**
```
python debug.py
python debug.py --model models/shape_agent/best_model
```

**open tensorboard:**
```
tensorboard --logdir logs/tensorboard
```

---

## file overview

```
shape_env.py       — gymnasium environment: shapes, obs/action spaces, reward
llm_goal_parser.py — LLM goal parser stub (patterns now, real API call documented)
train.py           — PPO training loop via stable-baselines3, saves checkpoints
demo.py            — standalone pygame demo, loads a trained model or runs random
callbacks.py       — custom tensorboard callback logging task-specific metrics
debug.py           — diagnostic script to verify env, reward, and model behavior
```

---

## architecture

```
user prompt
    |
    v
llm_goal_parser.parse_goal()      <-- real LLM call goes here
    |
    v
goal dict: {task, axis, direction}
    |
    v
ShapeEnv (gymnasium)              <-- environment owns the simulation
    |  observation: [shape states] + [goal encoding]
    |  action:      [shape_selector, target_x, target_y]
    |  reward:      delta rank correlation + step penalty + completion bonus
    v
PPO agent (stable-baselines3)     <-- learns the manipulation policy
```

---

## reward design

the reward at each step has three components:

- **delta reward** — change in rank correlation since the last step.
  the agent only gains from improving the arrangement, not from
  sitting in a good state. prevents the agent parking one shape
  in a decent position and looping forever.
- **step penalty** (-0.01 per step) — constant pressure to finish
  quickly. without this, an agent that reaches a local optimum
  has no reason to keep trying.
- **completion bonus** (+5.0) — awarded when the task is solved.
  makes finishing clearly worth more than any amount of stalling.

---

## tensorboard metrics

the `task/` group contains the most useful training signals:

| metric | what it means |
|---|---|
| `task/rank_correlation` | how well sorted shapes are. 1.0 = perfect, -1.0 = reversed |
| `task/solve_rate` | fraction of eval episodes that hit the solve threshold |
| `task/mean_y_spread` | std of y positions (normalized). lower = more in a line |
| `task/mean_episode_steps` | avg steps to finish. decreasing = agent getting faster |
| `task/mean_final_reward` | raw score at episode end, independent of delta shaping |

---

## wiring in a real LLM

in `llm_goal_parser.py`, replace the body of `_stub_parse()` with:

```python
import json
from openai import OpenAI

client = OpenAI()  # reads OPENAI_API_KEY from environment

def _stub_parse(prompt: str) -> dict:
   response = client.chat.completions.create(
      model="gpt-4o",
      messages=[
         {"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user",   "content": prompt},
      ],
      response_format={"type": "json_object"},
   )
   return json.loads(response.choices[0].message.content)
```

the rest of the system doesn't change.

---

## adding new tasks

1. add a reward function branch in `shape_env._compute_reward()`
2. add the task name to `llm_goal_parser.SUPPORTED_TASKS`
3. update `SYSTEM_PROMPT` in `llm_goal_parser.py` with examples
4. add pattern matching to `_stub_parse()` if staying with the stub

---

## expanding to real desktop / game input

the trained agent outputs `[shape_selector, target_x, target_y]` actions.
to route those to a real input driver instead of the pygame env:

```python
# swap this:
obs, reward, terminated, truncated, info = env.step(action)

# for something like:
target_x_pixels = int(action[1] * SCREEN_W)
target_y_pixels = int(action[2] * SCREEN_H)
pyautogui.moveTo(target_x_pixels, target_y_pixels)
pyautogui.mouseDown(); pyautogui.mouseUp()
```

the policy doesn't change — only the actuator does.
