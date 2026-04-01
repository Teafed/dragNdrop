"""
curriculum.py

staged training curriculum for the shape manipulation agent.

stages progress from simple cursor skills (reach, touch, drag) up to the
full multi-shape arrangement tasks. each stage has a performance gate —
if the agent hits the solve rate threshold before the step ceiling, it
advances early. if not, it advances at the ceiling anyway so training
doesn't stall.

gate thresholds are intentionally permissive (40-50%) for starter stages.
these build cursor skills, not mastery — advancing early is better than
stalling, since the agent continues improving on earlier tasks while
training later ones.

arrangement stages are defined but commented out. enable them once
reach/touch/drag are solving reliably.
"""

import random

# ---------------------------------------------------------------------------
# stage definitions
# ---------------------------------------------------------------------------

_STAGES = [
   {
      "name":         "reach — 1 shape",
      "tasks":        ["reach"],
      "n_shapes_min": 1,
      "n_shapes_max": 1,
      "gate_task":    "reach",
      "gate_sr":      0.50,
      "step_ceiling": 25_000,
   },
   {
      "name":         "reach — 1-2 shapes",
      "tasks":        ["reach"],
      "n_shapes_min": 1,
      "n_shapes_max": 2,
      "gate_task":    "reach",
      "gate_sr":      0.50,
      "step_ceiling": 50_000,
   },
   {
      "name":         "touch — 1 shape",
      "tasks":        ["touch"],
      "n_shapes_min": 1,
      "n_shapes_max": 1,
      "gate_task":    "touch",
      "gate_sr":      0.50,
      "step_ceiling": 75_000,
   },
   {
      "name":         "touch — 1-2 shapes",
      "tasks":        ["touch"],
      "n_shapes_min": 1,
      "n_shapes_max": 2,
      "gate_task":    "touch",
      "gate_sr":      0.50,
      "step_ceiling": 100_000,
   },
   {
      "name":         "reach and touch — 1-2 shapes",
      "tasks":        ["reach", "touch"],
      "n_shapes_min": 1,
      "n_shapes_max": 2,
      "gate_task":    "touch",
      "gate_sr":      0.50,
      "step_ceiling": 150_000,
   },
   {
      "name":         "drag — 1 shape",
      "tasks":        ["drag"],
      "n_shapes_min": 1,
      "n_shapes_max": 1,
      "gate_task":    "drag",
      "gate_sr":      0.40,
      "step_ceiling": 200_000,
   },
   {
      "name":         "reach and drag — 1 shape",
      "tasks":        ["reach", "drag"],
      "n_shapes_min": 1,
      "n_shapes_max": 1,
      "gate_task":    "drag",
      "gate_sr":      0.40,
      "step_ceiling": 250_000,
   },
   {
      "name":         "reach, touch and drag — 1 shape",
      "tasks":        ["reach", "touch", "drag"],
      "n_shapes_min": 1,
      "n_shapes_max": 1,
      "gate_task":    "drag",
      "gate_sr":      0.40,
      "step_ceiling": 300_000,
   },
   {
      "name":         "reach, touch and drag — 1-2 shapes (final)",
      "tasks":        ["reach", "touch", "drag"],
      "n_shapes_min": 1,
      "n_shapes_max": 2,
      "gate_task":    None,   # no gate — stay here for remaining budget
      "gate_sr":      None,
      "step_ceiling": None,
   },

   # --- arrangement stages (disabled) ---
   # uncomment these and remove the final stage above once starter tasks
   # are solving reliably. also update final_stage reference in train.py.
   #
   # {
   #    "name":         "arrange in sequence — 2-3 shapes",
   #    "tasks":        ["arrange_in_sequence"],
   #    "n_shapes_min": 2,
   #    "n_shapes_max": 3,
   #    "gate_task":    "arrange_in_sequence",
   #    "gate_sr":      0.60,
   #    "step_ceiling": 150_000,
   # },
   # {
   #    "name":         "sequence and region — 2-3 shapes",
   #    "tasks":        ["arrange_in_sequence", "arrange_in_region"],
   #    "n_shapes_min": 2,
   #    "n_shapes_max": 3,
   #    "gate_task":    "arrange_in_region",
   #    "gate_sr":      0.60,
   #    "step_ceiling": 150_000,
   # },
   # {
   #    "name":         "sequence, region and line — 2-4 shapes",
   #    "tasks":        ["arrange_in_sequence", "arrange_in_region",
   #                     "arrange_in_line"],
   #    "n_shapes_min": 2,
   #    "n_shapes_max": 4,
   #    "gate_task":    "arrange_in_line",
   #    "gate_sr":      0.60,
   #    "step_ceiling": 200_000,
   # },
   # {
   #    "name":         "all arrangement tasks — 2-3 shapes",
   #    "tasks":        ["arrange_in_sequence", "arrange_in_region",
   #                     "arrange_in_line", "arrange_in_groups"],
   #    "n_shapes_min": 2,
   #    "n_shapes_max": 3,
   #    "gate_task":    "arrange_in_groups",
   #    "gate_sr":      0.40,
   #    "step_ceiling": 200_000,
   # },
   # {
   #    "name":         "all tasks — 2-6 shapes (final)",
   #    "tasks":        SUPPORTED_TASKS,
   #    "n_shapes_min": 2,
   #    "n_shapes_max": 6,
   #    "gate_task":    None,
   #    "gate_sr":      None,
   #    "step_ceiling": None,
   # },
]


# ---------------------------------------------------------------------------
# CurriculumManager
# ---------------------------------------------------------------------------

class CurriculumManager:
   """
   manages curriculum stage and task sampling for training.

   stateful: tracks current stage index and the step at which it started.
   train.py calls maybe_advance() after each per-task eval.
   """

   def __init__(self, verbose: bool = True, start_stage: int = 0):
      self._stage_idx        = max(0, min(start_stage, len(_STAGES) - 1))
      self._stage_start_step = 0
      self.verbose           = verbose
      self._log(f"curriculum starting at stage {self._stage_idx} — {self.stage['name']}")

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
      """sample a prompt from the current stage's active tasks."""
      from prompt_gen import sample_prompt
      task = random.choice(self.active_tasks)
      return sample_prompt(task)

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
      returns True if the stage advanced, False otherwise.

      per_task_solve_rates: {task_name: solve_rate} from the last eval.
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
      old_name               = self.stage["name"]
      self._stage_idx       += 1
      self._stage_start_step = current_step
      new_name               = self.stage["name"]
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
      return (f"stage {self._stage_idx} — {s['name']}  "
              f"tasks={self.active_tasks}  "
              f"n_shapes={self.n_shapes_range[0]}-{self.n_shapes_range[1]}")

   def _log(self, msg: str):
      if self.verbose:
         print(f"[curriculum] {msg}")
