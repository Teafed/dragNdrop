"""
prompt_gen.py

template-based prompt generator — all tasks including rudimentary stages.
"""

import random
from itertools import product
from typing import Optional

from config import SHAPE_TYPES, SHAPE_COLORS, SUPPORTED_TASKS


# ---------------------------------------------------------------------------
# target phrase vocabulary (starter tasks only)
# ---------------------------------------------------------------------------

_TARGET_PHRASES: dict[tuple, list[str]] = {
   ("any", "any"): ["the shape", "any shape", "a shape", "one of the shapes"],
   ("red",    "any"): ["the red shape",    "a red shape",    "the red one"],
   ("green",  "any"): ["the green shape",  "a green shape",  "the green one"],
   ("teal",   "any"): ["the teal shape",   "a teal shape",   "the teal one"],
   ("yellow", "any"): ["the yellow shape", "a yellow shape", "the yellow one"],
   ("purple", "any"): ["the purple shape", "a purple shape", "the purple one"],
   ("any", "circle"):   ["the circle",   "a circle",   "any circle"],
   ("any", "square"):   ["the square",   "a square",   "any square"],
   ("any", "triangle"): ["the triangle", "a triangle", "any triangle"],
   ("red",    "circle"):   ["the red circle",   "a red circle"],
   ("red",    "square"):   ["the red square",   "a red square"],
   ("red",    "triangle"): ["the red triangle", "a red triangle"],
   ("green",  "circle"):   ["the green circle",   "a green circle"],
   ("green",  "square"):   ["the green square",   "a green square"],
   ("green",  "triangle"): ["the green triangle", "a green triangle"],
   ("teal",   "circle"):   ["the teal circle",   "a teal circle"],
   ("teal",   "square"):   ["the teal square",   "a teal square"],
   ("teal",   "triangle"): ["the teal triangle", "a teal triangle"],
   ("yellow", "circle"):   ["the yellow circle",   "a yellow circle"],
   ("yellow", "square"):   ["the yellow square",   "a yellow square"],
   ("yellow", "triangle"): ["the yellow triangle", "a yellow triangle"],
   ("purple", "circle"):   ["the purple circle",   "a purple circle"],
   ("purple", "square"):   ["the purple square",   "a purple square"],
   ("purple", "triangle"): ["the purple triangle", "a purple triangle"],
}

# ---------------------------------------------------------------------------
# sentence templates
# ---------------------------------------------------------------------------

_TEMPLATES: dict[str, list[str]] = {
   # rudimentary — no {target} slot
   "move_cardinal": [
      "move the cursor to the marker",
      "navigate to the zone",
      "move to the target position",
      "go to the highlighted area",
      "bring the cursor to the marker",
   ],
   "move_diagonal": [
      "move the cursor to the corner marker",
      "navigate to the corner zone",
      "move to the corner target",
      "go to the corner area",
      "bring the cursor to the corner",
   ],
   "approach": [
      "get close to {target}",
      "move near {target}",
      "approach {target}",
      "bring the cursor close to {target}",
      "move the cursor near {target}",
   ],
   # grip builders — no {target} slot
   "click_at": [
      "click on the marker",
      "click inside the zone",
      "press at the target position",
      "click the highlighted area",
      "tap the zone",
   ],
   "hold_at": [
      "hold the click at the marker",
      "click and hold the zone",
      "press and hold at the target",
      "hold down at the highlighted area",
      "click and hold the marker",
   ],
   # starter tasks
   "reach": [
      "move the cursor to {target}",
      "navigate to {target}",
   ],
   "touch": [
      "click on {target}",
   ],
   "drag": [
      "drag {target} to the left side",
      "drag {target} to the right side",
      "drag {target} to the top",
      "drag {target} to the bottom",
      "pull {target} to the left side",
      "pull {target} to the right side",
   ],
   "none": [
      "do nothing", "no task", "wait", "hang tight",
   ],
}

# tasks where target specificity applies
_TARGET_TASKS   = {"reach", "touch", "drag", "approach"}
# tasks with no {target} slot in templates
_NO_TARGET_TASKS = {"move_cardinal", "move_diagonal", "click_at", "hold_at", "none"}


# ---------------------------------------------------------------------------
# PromptGenerator
# ---------------------------------------------------------------------------

class PromptGenerator:

   def __init__(self, specificity_weights: Optional[dict] = None,
                seed: Optional[int] = None):
      self.rng = random.Random(seed)
      self.specificity_weights = specificity_weights or {
         "any": 1.0, "color": 1.5, "type": 1.5, "both": 2.0,
      }

   def sample(self, task: Optional[str] = None) -> str:
      if task is None:
         task = self.rng.choice([t for t in SUPPORTED_TASKS if t != "none"])
      return self._sample_for_task(task)

   def sample_task(self, task: str, n: int = 1) -> list[str]:
      return [self._sample_for_task(task) for _ in range(n)]

   def _sample_for_task(self, task: str) -> str:
      templates = _TEMPLATES.get(task, [])
      if not templates:
         raise ValueError(f"no templates for task {task!r}")

      template = self.rng.choice(templates)

      if task in _TARGET_TASKS:
         color, shape_type = self._sample_target_spec()
         phrase = self._sample_target_phrase(color, shape_type)
         try:
            return template.format(target=phrase)
         except KeyError:
            return template
      else:
         return template   # no {target} slot

   def _sample_target_spec(self) -> tuple[str, str]:
      weights = self.specificity_weights
      total   = sum(weights.values())
      r       = self.rng.random() * total
      cumul   = 0.0
      level   = "both"
      for lvl, w in weights.items():
         cumul += w
         if r < cumul:
            level = lvl
            break
      if level == "any":
         return "any", "any"
      elif level == "color":
         return self.rng.choice(SHAPE_COLORS), "any"
      elif level == "type":
         return "any", self.rng.choice(SHAPE_TYPES)
      else:
         return self.rng.choice(SHAPE_COLORS), self.rng.choice(SHAPE_TYPES)

   def _sample_target_phrase(self, color: str, shape_type: str) -> str:
      key = (color, shape_type)
      if key not in _TARGET_PHRASES:
         if color != "any" and shape_type != "any":
            key = (color, "any") if (color, "any") in _TARGET_PHRASES else ("any", shape_type)
         if key not in _TARGET_PHRASES:
            key = ("any", "any")
      return self.rng.choice(_TARGET_PHRASES[key])

   def all_prompts(self, task: str, color: str = "any",
                   shape_type: str = "any") -> list[str]:
      templates = _TEMPLATES.get(task, [])
      if task in _NO_TARGET_TASKS:
         return list(templates)
      phrases = _TARGET_PHRASES.get((color, shape_type),
                                    _TARGET_PHRASES[("any", "any")])
      results = []
      for tmpl, phrase in product(templates, phrases):
         try:
            results.append(tmpl.format(target=phrase))
         except KeyError:
            results.append(tmpl)
      return results

   def training_pool(self, n_per_task: int = 8) -> list[str]:
      pool = []
      # rudimentary / grip builder tasks — no target variation
      for task in ("move_cardinal", "move_diagonal", "click_at", "hold_at"):
         templates = _TEMPLATES.get(task, [])
         for _ in range(n_per_task):
            pool.append(self.rng.choice(templates))
      # approach — has target variation
      for _ in range(n_per_task):
         pool.append(self._make_prompt("approach", "any", "any"))
      for color in SHAPE_COLORS:
         pool.append(self._make_prompt("approach", color, "any"))
      for st in SHAPE_TYPES:
         pool.append(self._make_prompt("approach", "any", st))
      # starter tasks — full color x type grid
      for task in ("reach", "touch", "drag"):
         for _ in range(n_per_task):
            pool.append(self._make_prompt(task, "any", "any"))
         for color in SHAPE_COLORS:
            for _ in range(n_per_task):
               pool.append(self._make_prompt(task, color, "any"))
         for st in SHAPE_TYPES:
            for _ in range(n_per_task):
               pool.append(self._make_prompt(task, "any", st))
         for color, st in product(SHAPE_COLORS, SHAPE_TYPES):
            for _ in range(n_per_task):
               pool.append(self._make_prompt(task, color, st))
      return pool

   def _make_prompt(self, task: str, color: str, shape_type: str) -> str:
      if task in _NO_TARGET_TASKS:
         return self.rng.choice(_TEMPLATES[task])
      phrase   = self._sample_target_phrase(color, shape_type)
      template = self.rng.choice(_TEMPLATES[task])
      try:
         return template.format(target=phrase)
      except KeyError:
         return template

   def contrastive_pair(self, task: str, n_distractors: int = 1) -> list[str]:
      if task not in _TARGET_TASKS:
         raise ValueError(f"contrastive pairs only for target tasks, got {task!r}")
      n_needed = n_distractors + 1
      specs    = set()
      attempts = 0
      while len(specs) < n_needed and attempts < 100:
         specs.add((self.rng.choice(SHAPE_COLORS), self.rng.choice(SHAPE_TYPES)))
         attempts += 1
      return [self._make_prompt(task, c, t) for c, t in list(specs)[:n_needed]]

   def coverage_report(self):
      print("\nprompt generator coverage report")
      print("=" * 55)
      for task in SUPPORTED_TASKS:
         if task in _NO_TARGET_TASKS:
            n = len(_TEMPLATES.get(task, []))
            print(f"  {task:<22} templates={n}")
         else:
            n_any   = len(self.all_prompts(task, "any", "any"))
            n_color = sum(len(self.all_prompts(task, c, "any")) for c in SHAPE_COLORS)
            n_type  = sum(len(self.all_prompts(task, "any", t)) for t in SHAPE_TYPES)
            n_both  = sum(len(self.all_prompts(task, c, t))
                          for c, t in product(SHAPE_COLORS, SHAPE_TYPES))
            total   = n_any + n_color + n_type + n_both
            print(f"  {task:<22} any={n_any:3d}  color={n_color:3d}  "
                  f"type={n_type:3d}  both={n_both:3d}  total={total:4d}")
      print()


# ---------------------------------------------------------------------------
# module-level convenience
# ---------------------------------------------------------------------------

_default_gen = PromptGenerator()

def sample_prompt(task: Optional[str] = None) -> str:
   return _default_gen.sample(task)

def training_pool(n_per_task: int = 4) -> list[str]:
   return _default_gen.training_pool(n_per_task=n_per_task)


if __name__ == "__main__":
   import argparse
   parser = argparse.ArgumentParser()
   parser.add_argument("--coverage",    action="store_true")
   parser.add_argument("--sample",      type=int, default=0)
   parser.add_argument("--task",        type=str, default=None)
   parser.add_argument("--pool",        action="store_true")
   args = parser.parse_args()
   gen = PromptGenerator(seed=42)
   if args.coverage:
      gen.coverage_report()
   if args.sample > 0:
      task    = args.task
      prompts = (gen.sample_task(task, n=args.sample) if task
                 else [gen.sample() for _ in range(args.sample)])
      for p in prompts:
         print(f"  {p}")
   if args.pool:
      pool = gen.training_pool()
      print(f"\ntraining pool: {len(pool)} prompts")