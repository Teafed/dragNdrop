"""
llm_goal_parser.py

LLM goal parsing and embedding layer.

public functions:
    parse_goal(prompt) -> dict
    get_embedding(prompt) -> np.ndarray [EMBEDDING_DIM]

wave 3 task framework — four tasks from the 2x2x2 cube:
    arrange_in_sequence  ordered along axis, perpendicular unconstrained
    arrange_in_line      ordered or spaced along axis, perpendicular bounded
    arrange_in_region    all shapes in a canvas subregion, unordered
    arrange_in_groups    shapes partitioned by attribute into subregions
"""

import numpy as np

# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------

GOAL_SCHEMA = {
   "task":         str,
   "axis":         str,
   "direction":    str,
   "attribute":    str,
   "region":       str,
   "bounded":      bool,
   "target_color": str,   # "none" | "any" | color name (reach/touch/drag)
   "target_type":  str,   # "none" | "any" | shape type  (reach/touch/drag)
}

VALID_COLORS = ("red", "green", "teal", "yellow", "purple", "any", "none")
VALID_TYPES  = ("circle", "square", "triangle", "any", "none")

SUPPORTED_TASKS = [
   "reach",
   "touch",
   "drag",
   "arrange_in_sequence",
   "arrange_in_line",
   "arrange_in_region",
   "arrange_in_groups",
]

# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def parse_goal(prompt: str) -> dict:
   """
   parse a natural language prompt into a validated goal dict.
   """
   goal = _stub_parse(prompt.lower())
   _validate_goal(goal)
   return goal


# ---------------------------------------------------------------------------
# stub fallback
# ---------------------------------------------------------------------------

def _stub_parse(prompt: str) -> dict:
   """
   keyword-matching fallback. handles common phrasings for all seven tasks.
   ambiguous prompts default to arrange_in_sequence by size ascending.
   """

   # --- starter tasks: reach, touch, drag ---
   # detect color and type early — used by all three starter tasks
   tc = _infer_target_color(prompt)
   tt = _infer_target_type(prompt)

   # drag: any explicit drag word, OR "move/push/carry [a/any/the] shape to [direction]"
   # checked before region so "drag the red square to the left side" doesn't
   # collapse into arrange_in_region
   _drag_direction_kws = ("to the left", "to the right", "to the top",
                          "to the bottom", "leftward", "rightward")
   is_single_shape_move = (
      any(kw in prompt for kw in ("a shape", "any shape", "the shape",
                                  "one shape", "this shape"))
      and any(kw in prompt for kw in _drag_direction_kws)
   )
   is_drag = any(kw in prompt for kw in ("drag", "pull", "carry", "slide")) \
             or is_single_shape_move
   if is_drag:
      region = "left"
      if any(kw in prompt for kw in ("right",)):
         region = "right"
      elif any(kw in prompt for kw in ("top", "up")):
         region = "top"
      elif any(kw in prompt for kw in ("bottom", "down")):
         region = "bottom"
      return {**_starter_goal("drag", tc, tt), "region": region}

   # reach: cursor navigation prompts
   if any(kw in prompt for kw in (
      "move the cursor", "navigate to", "move cursor",
      "reach the", "reach shape", "go to the",
      "cursor to",
   )):
      return _starter_goal("reach", tc, tt)

   # touch: grip-activation prompts. "touch the" must come before the groups
   # check so "touch the triangle" doesn't fall through to arrange_in_sequence.
   if any(kw in prompt for kw in (
      "click on", "click the", "tap the", "tap on",
      "touch the", "touch shape",
      "press the", "press on",
      "activate the", "grab the", "pick up the",
   )):
      return _starter_goal("touch", tc, tt)

   # --- arrange_in_region ---
   if any(kw in prompt for kw in ("left side", "move left", "push left", "to the left")):
      return _region_goal("left")
   if any(kw in prompt for kw in ("right side", "move right", "push right", "to the right")):
      return _region_goal("right")
   if any(kw in prompt for kw in ("to the top", "move up", "push up", "move to the top")):
      return _region_goal("top")
   if any(kw in prompt for kw in ("to the bottom", "move down", "push down", "move to the bottom")):
      return _region_goal("bottom")

   # --- arrange_in_groups ---
   shape_type_kws = ("by type", "same type", "same shape", "group by type",
                     "circles squares", "triangles separately",
                     "group the circles", "group the squares")
   if any(kw in prompt for kw in shape_type_kws):
      return _groups_goal("shape_type")

   color_kws = ("by color", "by colour", "same color", "same colour",
                "group by color", "group by colour",
                "color together", "colour together")
   if any(kw in prompt for kw in color_kws):
      return _groups_goal("color")

   if any(kw in prompt for kw in ("group", "cluster", "together", "close together")):
      if any(kw in prompt for kw in ("type", "shape", "circle", "square", "triangle")):
         return _groups_goal("shape_type")
      return _groups_goal("color")

   # --- arrange_in_line vs arrange_in_sequence ---
   is_line = any(kw in prompt for kw in (
      "in a line", "in line",
      "in a horizontal line", "in horizontal line",
      "in a vertical line",   "in vertical line",
      "in a row", "in a column",
      "evenly spaced", "equally spaced", "equally distributed",
      "lined up", "line up", "line them up",
      "in a straight line", "in straight line",
   ))
   axis      = _infer_axis(prompt)
   direction = _infer_direction(prompt)
   attribute = _infer_attribute(prompt)

   if is_line:
      has_order = attribute != "none" or any(kw in prompt for kw in (
         "sort", "order", "ascending", "descending", "smallest", "largest"))
      return {
         "task":         "arrange_in_line",
         "axis":         axis,
         "direction":    direction if has_order else "none",
         "attribute":    attribute if has_order else "none",
         "region":       "none",
         "bounded":      True,
         "target_color": "none",
         "target_type":  "none",
      }

   # default: arrange_in_sequence
   return {
      "task":         "arrange_in_sequence",
      "axis":         axis,
      "direction":    direction,
      "attribute":    attribute if attribute != "none" else "size",
      "region":       "none",
      "bounded":      False,
      "target_color": "none",
      "target_type":  "none",
   }


def _starter_goal(task: str, target_color: str = "none",
                  target_type: str = "none") -> dict:
   return {"task": task, "axis": "none", "direction": "none",
           "attribute": "none", "region": "none", "bounded": False,
           "target_color": target_color, "target_type": target_type}


def _infer_target_color(prompt: str) -> str:
   """return color name if mentioned, 'any' if 'any'/'a shape', else 'none'."""
   for color in ("red", "green", "teal", "yellow", "purple"):
      if color in prompt:
         return color
   if any(kw in prompt for kw in ("any shape", "any", "a shape", "the shape")):
      return "any"
   return "none"


def _infer_target_type(prompt: str) -> str:
   """return shape type if mentioned, 'any' if 'any'/'a shape', else 'none'."""
   for shape_type in ("circle", "square", "triangle"):
      if shape_type in prompt:
         return shape_type
   if any(kw in prompt for kw in ("any shape", "any", "a shape", "the shape")):
      return "any"
   return "none"


def _region_goal(region: str) -> dict:
   return {"task": "arrange_in_region", "axis": "none", "direction": "none",
           "attribute": "none", "region": region, "bounded": True,
           "target_color": "none", "target_type": "none"}


def _groups_goal(attribute: str) -> dict:
   return {"task": "arrange_in_groups", "axis": "none", "direction": "none",
           "attribute": attribute, "region": "none", "bounded": True,
           "target_color": "none", "target_type": "none"}


def _infer_axis(prompt: str) -> str:
   if any(kw in prompt for kw in ("vertical", "top to bottom", "column",
                                   "up to down", "top to bottom")):
      return "y"
   return "x"


def _infer_direction(prompt: str) -> str:
   if any(kw in prompt for kw in ("largest to smallest", "biggest first",
                                   "largest first", "descend", "big to small",
                                   "biggest at top", "right to left",
                                   "bottom to top")):
      return "descending"
   return "ascending"


def _infer_attribute(prompt: str) -> str:
   if any(kw in prompt for kw in ("color", "colour")):
      return "color"
   if any(kw in prompt for kw in ("type", "shape_type")):
      return "shape_type"
   if any(kw in prompt for kw in ("size", "small", "large", "big", "tiny",
                                   "smallest", "largest", "bigger", "smaller")):
      return "size"
   return "none"


# ---------------------------------------------------------------------------
# goal embedding
# ---------------------------------------------------------------------------

_embedding_model  = None
_embedding_cache: dict = {}


def get_embedding(prompt: str) -> np.ndarray:
   """
   encode a prompt using sentence-transformers all-MiniLM-L6-v2.
   model loaded once and cached. individual embeddings also cached.
   returns np.ndarray (EMBEDDING_DIM,) = (384,), float32.
   """
   global _embedding_model

   if prompt in _embedding_cache:
      return _embedding_cache[prompt]

   if _embedding_model is None:
      try:
         from sentence_transformers import SentenceTransformer
      except ImportError:
         raise RuntimeError(
            "sentence-transformers not installed — "
            "run: pip install sentence-transformers"
         )
      from config import EMBEDDING_MODEL
      print(f"[goal_parser] loading embedding model '{EMBEDDING_MODEL}'...")
      import logging
      logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
      logging.getLogger("transformers").setLevel(logging.ERROR)
      _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
      print("[goal_parser] embedding model ready.")

   embedding = _embedding_model.encode(
      prompt, convert_to_numpy=True).astype(np.float32)
   _embedding_cache[prompt] = embedding
   return embedding


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

def _validate_goal(goal: dict):
   """raise ValueError if the goal dict is missing fields or has bad values."""
   for key, expected_type in GOAL_SCHEMA.items():
      if key not in goal:
         raise ValueError(f"goal missing required key: '{key}'")
      if not isinstance(goal[key], expected_type):
         raise ValueError(
            f"goal['{key}'] should be {expected_type.__name__}, "
            f"got {type(goal[key]).__name__} (value: {goal[key]!r})"
         )
   if goal["task"] not in SUPPORTED_TASKS:
      raise ValueError(f"unsupported task: '{goal['task']}'")
   if goal["axis"] not in ("x", "y", "none"):
      raise ValueError(f"bad axis: '{goal['axis']}'")
   if goal["direction"] not in ("ascending", "descending", "none"):
      raise ValueError(f"bad direction: '{goal['direction']}'")
   if goal["attribute"] not in ("size", "color", "shape_type", "none"):
      raise ValueError(f"bad attribute: '{goal['attribute']}'")
   if goal["region"] not in ("left", "right", "top", "bottom", "none"):
      raise ValueError(f"bad region: '{goal['region']}'")
   if goal["target_color"] not in VALID_COLORS:
      raise ValueError(f"bad target_color: '{goal['target_color']}'")
   if goal["target_type"] not in VALID_TYPES:
      raise ValueError(f"bad target_type: '{goal['target_type']}'")
