"""
curriculum.py

12-stage curriculum for single-shape manipulation.

stage 0:  move_cardinal    — navigate to cardinal zone (no shape, no grip)
stage 1:  move_diagonal    — navigate to corner zone (no shape, no grip)
stage 2:  click_at         — navigate to zone then click (no shape)
stage 3:  hold_at          — navigate to zone, hold grip N steps (no shape)
stage 4:  approach         — real shape, get within 2x GRIP_RADIUS, no grip
stage 5:  reach            — real shape, get within GRIP_RADIUS
stage 6:  touch            — real shape, grip it
stage 7:  reach + touch    — mix of reach and touch
stage 8:  shadow drag      — real shape, large target region (~70% canvas)
stage 9:  drag             — real shape, normal region
stage 10: reach + drag     — mix
stage 11: reach+touch+drag — final
"""

import random

# ---------------------------------------------------------------------------
# stage definitions
# ---------------------------------------------------------------------------

_STAGES = [
   # --- rudimentary: no shape, no grip ---
   {
      "name":         "move cardinal — target zone",
      "tasks":        ["move_cardinal"],
      "n_shapes_min": 1,
      "n_shapes_max": 1,
      "gate_task":    "move_cardinal",
      "gate_sr":      0.80,
      "step_ceiling": 30_000,
   },
   {
      "name":         "move diagonal — corner zones",
      "tasks":        ["move_diagonal"],
      "n_shapes_min": 1,
      "n_shapes_max": 1,
      "gate_task":    "move_diagonal",
      "gate_sr":      0.80,
      "step_ceiling": 50_000,
   },

   # --- grip builders: no shape ---
   {
      "name":         "click at position",
      "tasks":        ["click_at"],
      "n_shapes_min": 1,
      "n_shapes_max": 1,
      "gate_task":    "click_at",
      "gate_sr":      0.75,
      "step_ceiling": 80_000,
   },
   {
      "name":         "hold at position",
      "tasks":        ["hold_at"],
      "n_shapes_min": 1,
      "n_shapes_max": 1,
      "gate_task":    "hold_at",
      "gate_sr":      0.70,
      "step_ceiling": 120_000,
   },

   # --- perceptual bridge ---
   {
      "name":         "approach — real shape",
      "tasks":        ["approach"],
      "n_shapes_min": 1,
      "n_shapes_max": 1,
      "gate_task":    "approach",
      "gate_sr":      0.70,
      "step_ceiling": 200_000,
   },

   # --- starter tasks ---
   {
      "name":         "reach — 1 shape + drag exposure",
      "tasks":        ["reach", "drag"],
      "task_weights": {"reach": 0.8, "drag": 0.2},
      "n_shapes_min": 1,
      "n_shapes_max": 1,
      "gate_task":    "reach",
      "gate_sr":      0.80,
      "step_ceiling": 350_000,
   },
   {
      "name":         "touch — 1 shape + drag exposure",
      "tasks":        ["touch", "drag"],
      "task_weights": {"touch": 0.8, "drag": 0.2},
      "n_shapes_min": 1,
      "n_shapes_max": 1,
      "gate_task":    "touch",
      "gate_sr":      0.80,
      "step_ceiling": 400_000,
   },
   {
      "name":         "reach and touch + drag exposure",
      "tasks":        ["reach", "touch", "drag"],
      "task_weights": {"reach": 0.4, "touch": 0.4, "drag": 0.2},
      "n_shapes_min": 1,
      "n_shapes_max": 1,
      "gate_task":    "touch",
      "gate_sr":      0.97,
      "step_ceiling": 500_000,
   },

   # --- drag progression ---
   {
      "name":         "shadow drag — multi-task",
      "tasks":        ["touch", "drag"],
      "n_shapes_min": 1,
      "n_shapes_max": 1,
      "gate_task":    "drag",
      "gate_sr":      0.65,
      "step_ceiling": 480_000,
      "drag_region_scale": 0.70,   # target region covers 70% of canvas
   },
   {
      "name":         "drag — multi-task",
      "tasks":        ["reach", "touch", "drag"],
      "n_shapes_min": 1,
      "n_shapes_max": 1,
      "gate_task":    "drag",
      "gate_sr":      0.98,
      "step_ceiling": 570_000,
   },
   {
      "name":         "reach, touch and drag — 1 shape (final)",
      "tasks":        ["reach", "touch", "drag"],
      "n_shapes_min": 1,
      "n_shapes_max": 1,
      "gate_task":    None,
      "gate_sr":      None,
      "step_ceiling": None,
   },
]


# ---------------------------------------------------------------------------
# CurriculumManager
# ---------------------------------------------------------------------------

class CurriculumManager:

   def __init__(self, verbose: bool = True, start_stage: int = 0):
      self._stage_idx        = max(0, min(start_stage, len(_STAGES) - 1))
      self._stage_start_step = 0
      self.verbose           = verbose
      self._log(f"starting at stage {self._stage_idx} — {self.stage['name']}")

   @property
   def stage(self) -> dict:
      return _STAGES[self._stage_idx]

   @property
   def stage_idx(self) -> int:
      return self._stage_idx

   @property
   def n_shapes_range(self) -> tuple:
      return (self.stage["n_shapes_min"], self.stage["n_shapes_max"])

   @property
   def active_tasks(self) -> list:
      return self.stage["tasks"]

   @property
   def is_final_stage(self) -> bool:
      return self._stage_idx == len(_STAGES) - 1

   @property
   def drag_region_scale(self) -> float:
      """
      for shadow drag stage: fraction of canvas the target region covers.
      normal drag stages return 1.0 (standard REGION_INNER boundaries).
      """
      return self.stage.get("drag_region_scale", 1.0)

   def sample_prompt(self) -> str:
      from prompt_gen import sample_prompt
      weights = self.stage.get("task_weights")
      if weights:
         tasks  = list(weights.keys())
         probs  = [weights[t] for t in tasks]
         task   = random.choices(tasks, weights=probs, k=1)[0]
      else:
         task = random.choice(self.active_tasks)
      return sample_prompt(task)

   def sample_n_shapes(self, rng=None) -> int:
      return 1

   def maybe_advance(self, per_task_solve_rates: dict,
                     current_step: int) -> bool:
      if self.is_final_stage:
         return False

      stage          = self.stage
      gate_task      = stage["gate_task"]
      gate_sr        = stage["gate_sr"]
      step_ceiling   = stage["step_ceiling"]
      steps_in_stage = current_step - self._stage_start_step

      gate_met    = (gate_task is not None
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
      self._log(
         f"advancing: {old_name} → {self.stage['name']}  "
         f"(reason: {reason}  step: {current_step:,})"
      )

   def status(self) -> str:
      s = self.stage
      return (f"stage {self._stage_idx} — {s['name']}  "
              f"tasks={self.active_tasks}  n_shapes=1")

   def _log(self, msg: str):
      if self.verbose:
         print(f"[curriculum] {msg}")