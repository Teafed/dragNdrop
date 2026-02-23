# shape manipulation agent — proof of concept

a gymnasium + stable-baselines3 proof of concept for an RL agent
that manipulates 2d shapes in response to a natural language goal.
the LLM layer is stubbed out — everything else is wired and runnable.

---

## install

```
pip install gymnasium stable-baselines3 pygame numpy
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
python train.py --load models/shape_agent/best_model --demo
```

---

## file overview

```
shape_env.py       — gymnasium environment (shapes, obs space, action space, reward)
llm_goal_parser.py — stub LLM parser (hardcoded patterns, real LLM wiring documented)
train.py           — training loop using PPO from stable-baselines3, plus demo mode
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
    |  reward:      spearman rank correlation + line penalty
    v
PPO agent (stable-baselines3)     <-- learns the manipulation policy
```

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

## expanding to new tasks

1. add a new reward function in `shape_env._compute_reward()`
2. add the task name to `llm_goal_parser.SUPPORTED_TASKS`
3. update the LLM system prompt with examples of the new task
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
