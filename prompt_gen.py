"""
prompt_gen.py

template-based prompt generator for the shape manipulation agent.

separates two concerns:
   semantic templates  — sentence structure and verb choices
   goal parameters     — task, target color, target type, region, etc.

this lets us generate the full combinatorial space of (color, type, task)
combinations without manually enumerating every phrasing. adding new tasks
or phrasings only requires adding to the relevant template list.

usage:
   from prompt_gen import PromptGenerator
   gen = PromptGenerator()

   # sample one random prompt
   prompt = gen.sample()

   # sample N prompts for a specific task
   prompts = gen.sample_task("reach", n=10)

   # generate all prompts for a (task, color, type) combo
   prompts = gen.all_prompts("touch", color="red", shape_type="square")

   # get a contrastive pair: same task, different targets
   p1, p2 = gen.contrastive_pair("reach")

   # get full training pool (replaces static TASK_POOL in config.py)
   pool = gen.training_pool()
"""

import random
from itertools import product
from typing import Optional

from config import SHAPE_TYPES, SHAPE_COLORS, SUPPORTED_TASKS


# ---------------------------------------------------------------------------
# target phrase vocabulary
# ---------------------------------------------------------------------------
#
# key: (color, shape_type) where "any" means unspecified.
# value: list of natural-language phrases that refer to that target.
#
# "any"/"any" → generic target phrases
# (color, "any") → color-only target phrases
# ("any", type) → type-only target phrases
# (color, type) → fully-specified target phrases

_TARGET_PHRASES: dict[tuple, list[str]] = {
   # generic — no color or type constraint
   ("any", "any"): [
      "the shape",
      "any shape",
      "a shape",
      "one of the shapes",
   ],

   # color-only
   ("red",    "any"): ["the red shape",    "a red shape",    "the red one"],
   ("green",  "any"): ["the green shape",  "a green shape",  "the green one"],
   ("teal",   "any"): ["the teal shape",   "a teal shape",   "the teal one"],
   ("yellow", "any"): ["the yellow shape", "a yellow shape", "the yellow one"],
   ("purple", "any"): ["the purple shape", "a purple shape", "the purple one"],

   # type-only
   ("any", "circle"):   ["the circle",   "a circle",   "any circle"],
   ("any", "square"):   ["the square",   "a square",   "any square"],
   ("any", "triangle"): ["the triangle", "a triangle", "any triangle"],

   # fully specified (color + type)
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
#
# each template has a {target} slot filled by a target phrase.
# templates are grouped by task so generation stays task-appropriate.

_TEMPLATES: dict[str, list[str]] = {
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
   "arrange_in_sequence": [
      "sort shapes from smallest to largest left to right",
      "sort shapes from largest to smallest left to right",
      "sort shapes from smallest to largest top to bottom",
      "sort shapes from largest to smallest top to bottom",
      "arrange shapes by size from left to right",
      "order shapes by size ascending left to right",
      "order shapes by size descending left to right",
   ],
   "arrange_in_line": [
      "arrange shapes in a horizontal line evenly spaced",
      "arrange shapes in a vertical line evenly spaced",
      "arrange shapes in a horizontal line sorted smallest to largest",
      "arrange shapes in a vertical line sorted largest to smallest",
      "arrange shapes in a horizontal line",
      "arrange shapes in a vertical line",
   ],
   "arrange_in_region": [
      "move all shapes to the left side",
      "move all shapes to the right side",
      "move all shapes to the top",
      "move all shapes to the bottom",
      "push all shapes to the left",
      "push all shapes to the right",
   ],
   "arrange_in_groups": [
      "group shapes by color",
      "put shapes of the same color close together",
      "group shapes by type",
      "put shapes of the same type close together",
      "group the circles squares and triangles separately",
      "sort shapes into groups by color",
      "sort shapes into groups by type",
      "cluster shapes by color",
      "cluster shapes by type",
   ],
   "none": [
      "do nothing", "no task", "be yourself", "wait for next task",
      "wait", "hang tight", "do whatever", "do anything"
   ],
}

# tasks where target specificity applies (starter tasks only)
_TARGET_TASKS = {"reach", "touch", "drag"}


# ---------------------------------------------------------------------------
# PromptGenerator
# ---------------------------------------------------------------------------

class PromptGenerator:
   """
   generates natural-language prompts by composing templates with target
   phrases. supports:

      - random sampling across all tasks and specificities
      - task-filtered sampling
      - full combinatorial enumeration for training pools
      - contrastive pair generation (same task, different targets)
      - custom specificity weighting

   specificity_weights controls how often each target specificity level
   is sampled. default weights prefer fully-specified targets slightly
   over generic ones — adjust based on training needs.
   """

   def __init__(
      self,
      specificity_weights: Optional[dict] = None,
      seed: Optional[int] = None,
   ):
      self.rng = random.Random(seed)

      # default: sample all specificity levels with slight bias toward
      # specific targets so the agent sees color/type discrimination often
      self.specificity_weights = specificity_weights or {
         "any":   1.0,
         "color": 1.5,
         "type":  1.5,
         "both":  2.0,
      }

   # -------------------------------------------------------------------------
   # core sampling
   # -------------------------------------------------------------------------

   def sample(self, task: Optional[str] = None) -> str:
      """sample one random prompt, optionally filtered by task."""
      if task is None:
         task = self.rng.choice(SUPPORTED_TASKS)
      return self._sample_for_task(task)

   def sample_task(self, task: str, n: int = 1) -> list[str]:
      """sample n prompts for a specific task."""
      return [self._sample_for_task(task) for _ in range(n)]

   def _sample_for_task(self, task: str) -> str:
      templates = _TEMPLATES.get(task, [])
      if not templates:
         raise ValueError(f"no templates for task {task!r}")

      if task in _TARGET_TASKS:
         color, shape_type = self._sample_target_spec()
      else:
         color, shape_type = "any", "any"

      phrase    = self._sample_target_phrase(color, shape_type)
      template  = self.rng.choice(templates)
      return template.format(target=phrase)

   def _sample_target_spec(self) -> tuple[str, str]:
      """sample a (color, shape_type) pair according to specificity weights."""
      weights  = self.specificity_weights
      total    = sum(weights.values())
      r        = self.rng.random() * total
      cumul    = 0.0
      level    = "both"
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
      else:   # both
         return self.rng.choice(SHAPE_COLORS), self.rng.choice(SHAPE_TYPES)

   def _sample_target_phrase(self, color: str, shape_type: str) -> str:
      """sample a phrase for the given (color, shape_type) pair."""
      key = (color, shape_type)
      if key not in _TARGET_PHRASES:
         # fall back to closest match
         if color != "any" and shape_type != "any":
            # try color-only or type-only
            key = (color, "any") if (color, "any") in _TARGET_PHRASES else ("any", shape_type)
         if key not in _TARGET_PHRASES:
            key = ("any", "any")
      return self.rng.choice(_TARGET_PHRASES[key])

   # -------------------------------------------------------------------------
   # enumeration
   # -------------------------------------------------------------------------

   def all_prompts(self, task: str,
                   color: str = "any",
                   shape_type: str = "any") -> list[str]:
      """
      generate all prompts for a (task, color, shape_type) combination.
      useful for exhaustive eval or inspecting what the generator produces.
      """
      templates = _TEMPLATES.get(task, [])
      phrases   = _TARGET_PHRASES.get((color, shape_type),
                                      _TARGET_PHRASES[("any", "any")])
      results   = []
      for tmpl, phrase in product(templates, phrases):
         try:
            results.append(tmpl.format(target=phrase))
         except KeyError:
            results.append(tmpl)   # template has no {target} slot
      return results

   def training_pool(self,
                     n_per_task: int = 8,
                     include_arrangement: bool = True) -> list[str]:
      """
      generate a balanced training pool covering all task/specificity combos.

      for starter tasks: samples n_per_task prompts per (color, type) pair,
      covering all 5*3+5+3+1 = 24 target combinations.
      for arrangement tasks: samples n_per_task prompts from templates.

      replaces the static TASK_POOL in config.py.
      """
      pool = []

      # starter tasks — full color × type grid
      for task in ("reach", "touch", "drag"):
         # generic
         for _ in range(n_per_task):
            pool.append(self._make_prompt(task, "any", "any"))
         # color-only
         for color in SHAPE_COLORS:
            for _ in range(n_per_task):
               pool.append(self._make_prompt(task, color, "any"))
         # type-only
         for shape_type in SHAPE_TYPES:
            for _ in range(n_per_task):
               pool.append(self._make_prompt(task, "any", shape_type))
         # fully specified
         for color, shape_type in product(SHAPE_COLORS, SHAPE_TYPES):
            for _ in range(n_per_task):
               pool.append(self._make_prompt(task, color, shape_type))

      # arrangement tasks
      if include_arrangement:
         for task in ("arrange_in_sequence", "arrange_in_line",
                      "arrange_in_region", "arrange_in_groups"):
            templates = _TEMPLATES.get(task, [])
            for tmpl in templates:
               pool.append(tmpl)   # no {target} slot

      return pool

   def _make_prompt(self, task: str, color: str, shape_type: str) -> str:
      """generate one prompt for a specific (task, color, type) combination."""
      phrase   = self._sample_target_phrase(color, shape_type)
      template = self.rng.choice(_TEMPLATES[task])
      return template.format(target=phrase)

   # -------------------------------------------------------------------------
   # contrastive pairs
   # -------------------------------------------------------------------------

   def contrastive_pair(
      self,
      task: str,
      n_distractors: int = 1,
   ) -> list[str]:
      """
      generate a contrastive set of prompts for the same task.
      all prompts target different (color, type) combinations so the
      oracle will navigate to different shapes in the same scene.

      n_distractors=1 → pair (2 prompts)
      n_distractors=2 → triple (3 prompts)

      example:
         ["click on the red square", "click on the teal circle"]

      the caller is responsible for resetting the env to the same spawn
      seed between episodes so the scene layout stays identical.
      """
      if task not in _TARGET_TASKS:
         raise ValueError(
            f"contrastive pairs only make sense for starter tasks, got {task!r}")

      # sample n+1 distinct (color, type) pairs
      n_needed = n_distractors + 1
      specs    = set()
      attempts = 0
      while len(specs) < n_needed and attempts < 100:
         color      = self.rng.choice(SHAPE_COLORS)
         shape_type = self.rng.choice(SHAPE_TYPES)
         specs.add((color, shape_type))
         attempts  += 1

      prompts = []
      for color, shape_type in list(specs)[:n_needed]:
         prompts.append(self._make_prompt(task, color, shape_type))
      return prompts

   def contrastive_any_vs_specific(self, task: str) -> tuple[str, str]:
      """
      generate a pair where one prompt is generic ("any shape") and one
      is specific ("the red square"). useful for testing whether the agent
      behaves differently based on goal specificity.
      """
      generic  = self._make_prompt(task, "any", "any")
      color    = self.rng.choice(SHAPE_COLORS)
      stype    = self.rng.choice(SHAPE_TYPES)
      specific = self._make_prompt(task, color, stype)
      return generic, specific

   # -------------------------------------------------------------------------
   # diagnostics
   # -------------------------------------------------------------------------

   def coverage_report(self):
      """print how many prompts exist per (task, specificity) combination."""
      print("\nprompt generator coverage report")
      print("=" * 55)
      for task in SUPPORTED_TASKS:
         if task in _TARGET_TASKS:
            n_any   = len(self.all_prompts(task, "any", "any"))
            n_color = sum(len(self.all_prompts(task, c, "any"))
                          for c in SHAPE_COLORS)
            n_type  = sum(len(self.all_prompts(task, "any", t))
                          for t in SHAPE_TYPES)
            n_both  = sum(len(self.all_prompts(task, c, t))
                          for c, t in product(SHAPE_COLORS, SHAPE_TYPES))
            total   = n_any + n_color + n_type + n_both
            print(f"  {task:<22} any={n_any:3d}  color={n_color:3d}  "
                  f"type={n_type:3d}  both={n_both:3d}  total={total:4d}")
         else:
            n = len(_TEMPLATES.get(task, []))
            print(f"  {task:<22} templates={n}")
      print()


# ---------------------------------------------------------------------------
# module-level convenience instance
# ---------------------------------------------------------------------------

_default_gen = PromptGenerator()

def sample_prompt(task: Optional[str] = None) -> str:
   """sample one prompt using the default generator."""
   return _default_gen.sample(task)

def training_pool(n_per_task: int = 4) -> list[str]:
   """get a training pool using the default generator."""
   return _default_gen.training_pool(n_per_task=n_per_task)


# ---------------------------------------------------------------------------
# CLI — inspect the generator
# ---------------------------------------------------------------------------

if __name__ == "__main__":
   import argparse
   parser = argparse.ArgumentParser(description="inspect the prompt generator")
   parser.add_argument("--coverage",   action="store_true",
                       help="print coverage report")
   parser.add_argument("--sample",     type=int, default=0,
                       help="print N random prompts")
   parser.add_argument("--task",       type=str, default=None,
                       help="filter by task")
   parser.add_argument("--contrastive",action="store_true",
                       help="print example contrastive pairs")
   parser.add_argument("--pool",       action="store_true",
                       help="print full training pool size")
   args = parser.parse_args()

   gen = PromptGenerator(seed=42)

   if args.coverage:
      gen.coverage_report()

   if args.sample > 0:
      task = args.task
      print(f"\n{args.sample} random prompts"
            + (f" (task={task})" if task else "") + ":")
      for p in gen.sample_task(task or "reach", n=args.sample) if task \
               else [gen.sample() for _ in range(args.sample)]:
         print(f"  {p}")

   if args.contrastive:
      print("\nexample contrastive pairs:")
      for task in ("reach", "touch", "drag"):
         pair = gen.contrastive_pair(task)
         print(f"  {task}: {pair}")
      print("\nexample any vs specific:")
      for task in ("reach", "touch"):
         g, s = gen.contrastive_any_vs_specific(task)
         print(f"  {task}: [{g!r}] vs [{s!r}]")

   if args.pool:
      pool = gen.training_pool()
      print(f"\ntraining pool: {len(pool)} prompts")
      by_task = {}
      for p in pool:
         from llm_goal_parser import parse_goal
         t = parse_goal(p).get("task", "?")
         by_task[t] = by_task.get(t, 0) + 1
      for t, n in sorted(by_task.items()):
         print(f"  {t:<25} {n}")
