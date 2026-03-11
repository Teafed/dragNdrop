"""
curriculum.py

curriculum manager for staged multi-task training.

the curriculum progresses through seven stages, starting with simple
cursor skill tasks and building up to the full wave 3 task suite.
progression is performance-gated (advance when the gate task hits a
solve rate threshold) with a step ceiling so training doesn't stall.

stages:
   0  reach only              n_shapes 1     gate 80%  ceiling  50k
   1  touch only              n_shapes 1     gate 80%  ceiling  50k
   2  drag only               n_shapes 1     gate 70%  ceiling  75k
   3  sequence only           n_shapes 2-3   gate 60%  ceiling 150k
   4  + arrange_in_region     n_shapes 2-3   gate 60%  ceiling 150k
   5  + arrange_in_line       n_shapes 2-4   gate 60%  ceiling 200k
   6  all tasks               n_shapes 2-6   no gate   remaining steps

within each stage, active tasks are sampled with equal probability
regardless of how many TASK_POOL prompts map to each task.

usage:
   manager = CurriculumManager()
   prompt  = manager.sample_prompt()
   n_shp   = manager.sample_n_shapes()
   ...
   advanced = manager.maybe_advance(
      per_task_solve_rates={"reach": 0.85, ...},
      current_step=30_000,
   )
"""

import random
from config import TASK_POOL, SUPPORTED_TASKS
from llm_goal_parser import parse_goal

# ---------------------------------------------------------------------------
# stage definitions
# ---------------------------------------------------------------------------

_STAGES = [
   # --- starter stages: cursor skill building ---
   {
      "name":         "stage 0 — reach",
      "tasks":        ["reach"],
      "n_shapes_min": 1,
      "n_shapes_max": 1,
      "gate_task":    "reach",
      "gate_sr":      0.80,
      "step_ceiling": 50_000,
   },
   {
      "name":         "stage 1 — touch",
      "tasks":        ["touch"],
      "n_shapes_min": 1,
      "n_shapes_max": 1,
      "gate_task":    "touch",
      "gate_sr":      0.80,
      "step_ceiling": 50_000,
   },
   {
      "name":         "stage 2 — drag",
      "tasks":        ["drag"],
      "n_shapes_min": 1,
      "n_shapes_max": 1,
      "gate_task":    "drag",
      "gate_sr":      0.70,
      "step_ceiling": 75_000,
   },
   # --- wave 3 stages: multi-shape arrangement ---
   {
      "name":         "stage 3 — sequence",
      "tasks":        ["arrange_in_sequence"],
      "n_shapes_min": 2,
      "n_shapes_max": 3,
      "gate_task":    "arrange_in_sequence",
      "gate_sr":      0.60,
      "step_ceiling": 150_000,
   },
   {
      "name":         "stage 4 — + region",
      "tasks":        ["arrange_in_sequence", "arrange_in_region"],
      "n_shapes_min": 2,
      "n_shapes_max": 3,
      "gate_task":    "arrange_in_region",
      "gate_sr":      0.60,
      "step_ceiling": 150_000,
   },
   {
      "name":         "stage 5 — + line",
      "tasks":        ["arrange_in_sequence", "arrange_in_region",
                       "arrange_in_line"],
      "n_shapes_min": 2,
      "n_shapes_max": 4,
      "gate_task":    "arrange_in_line",
      "gate_sr":      0.60,
      "step_ceiling": 200_000,
   },
   {
      "name":         "stage 6 — all tasks",
      "tasks":        SUPPORTED_TASKS,
      "n_shapes_min": 2,
      "n_shapes_max": 6,
      "gate_task":    None,   # no gate — stay here for remaining steps
      "gate_sr":      None,
      "step_ceiling": None,
   },
]

# ---------------------------------------------------------------------------
# prompt pool — keyed by task name
# ---------------------------------------------------------------------------

_FALLBACKS = {
   "reach":               "move the cursor to the shape",
   "touch":               "click on the shape",
   "drag":                "drag the shape to the left side",
   "arrange_in_sequence": "sort shapes from smallest to largest left to right",
   "arrange_in_line":     "arrange shapes in a horizontal line evenly spaced",
   "arrange_in_region":   "move all shapes to the left side",
   "arrange_in_groups":   "group shapes by color",
}


def _build_prompt_pool() -> dict:
   """return {task_name: [prompt, ...]} by parsing TASK_POOL once at import."""
   pool = {t: [] for t in SUPPORTED_TASKS}
   for prompt in TASK_POOL:
      try:
         goal = parse_goal(prompt)
         task = goal["task"]
         if task in pool:
            pool[task].append(prompt)
      except Exception:
         pass
   for task, prompts in pool.items():
      if not prompts:
         pool[task] = [_FALLBACKS[task]]
   return pool


_PROMPT_POOL = _build_prompt_pool()


# ---------------------------------------------------------------------------
# CurriculumManager
# ---------------------------------------------------------------------------

class CurriculumManager:
   """
   manages curriculum stage and task sampling for training.

   stateful: tracks current stage and the step at which it started.
   train.py calls maybe_advance() after each per-task eval.
   """

   def __init__(self, verbose: bool = True, start_stage: int = 0):
      self._stage_idx        = start_stage
      self._stage_start_step = 0
      self.verbose           = verbose
      self._log(f"curriculum starting at {self.stage['name']}")

   @property
   def stage(self) -> dict:
      return _STAGES[self._stage_idx]

   @property
   def stage_idx(self) -> int:
      return self._stage_idx

   @property
   def n_shapes_range(self) -> tuple:
      """(min, max) n_shapes for the current stage."""
      return (self.stage["n_shapes_min"], self.stage["n_shapes_max"])

   @property
   def active_tasks(self) -> list:
      return self.stage["tasks"]

   @property
   def is_final_stage(self) -> bool:
      return self._stage_idx == len(_STAGES) - 1

   def sample_prompt(self) -> str:
      """
      sample a prompt from the current stage's active tasks.
      task-balanced: each active task gets equal probability regardless
      of how many prompts map to it in TASK_POOL.
      """
      task   = random.choice(self.active_tasks)
      prompt = random.choice(_PROMPT_POOL[task])
      return prompt

   def sample_n_shapes(self, rng=None) -> int:
      """sample n_shapes uniformly within the current stage's range."""
      lo, hi = self.n_shapes_range
      if rng is not None:
         return int(rng.integers(lo, hi + 1))
      return random.randint(lo, hi)

   def maybe_advance(self, per_task_solve_rates: dict,
                     current_step: int) -> bool:
      """
      check whether to advance to the next stage.
      returns True if advanced, False otherwise.

      per_task_solve_rates: {task_name: solve_rate} from last eval.
      current_step: total training timesteps elapsed.
      """
      if self.is_final_stage:
         return False

      stage          = self.stage
      gate_task      = stage["gate_task"]
      gate_sr        = stage["gate_sr"]
      step_ceiling   = stage["step_ceiling"]
      steps_in_stage = current_step - self._stage_start_step

      gate_met = (gate_task is not None
                  and gate_task in per_task_solve_rates
                  and per_task_solve_rates[gate_task] >= gate_sr)

      ceiling_hit = (step_ceiling is not None
                     and steps_in_stage >= step_ceiling)

      if gate_met or ceiling_hit:
         reason = "performance gate" if gate_met else "step ceiling"
         self._advance(current_step, reason)
         return True

      return False

   def _advance(self, current_step: int, reason: str):
      old_name              = self.stage["name"]
      self._stage_idx      += 1
      self._stage_start_step = current_step
      new_name              = self.stage["name"]
      self._log(
         f"advancing: {old_name} → {new_name}  "
         f"(reason: {reason}  step: {current_step:,})"
      )
      self._log(
         f"  active tasks : {self.active_tasks}\n"
         f"  n_shapes     : {self.n_shapes_range[0]}–{self.n_shapes_range[1]}"
      )

   def status(self) -> str:
      s = self.stage
      return (f"{s['name']}  tasks={self.active_tasks}  "
              f"n_shapes={self.n_shapes_range[0]}-{self.n_shapes_range[1]}")

   def _log(self, msg: str):
      if self.verbose:
         print(f"[curriculum] {msg}")
