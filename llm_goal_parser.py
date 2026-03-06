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

import json
import os
import re
import numpy as np

# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------

GOAL_SCHEMA = {
   "task":      str,
   "axis":      str,
   "direction": str,
   "attribute": str,
   "region":    str,
   "bounded":   bool,
}

SUPPORTED_TASKS = [
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
   uses the Anthropic API if ANTHROPIC_API_KEY is set, else stub parser.
   """
   if os.environ.get("ANTHROPIC_API_KEY", ""):
      try:
         goal = _llm_parse(prompt)
         _validate_goal(goal)
         return goal
      except Exception as e:
         print(f"[goal_parser] LLM parse failed ({e}), using stub fallback.")

   goal = _stub_parse(prompt.lower())
   _validate_goal(goal)
   return goal


# ---------------------------------------------------------------------------
# LLM backend
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """
You are a goal parser for a 2D shape manipulation agent.
Respond ONLY with a single valid JSON object — no markdown, no extra text.

Schema (all fields required):
{
  "task":      "<arrange_in_sequence | arrange_in_line | arrange_in_region | arrange_in_groups>",
  "axis":      "<x | y | none>",
  "direction": "<ascending | descending | none>",
  "attribute": "<size | color | shape_type | none>",
  "region":    "<left | right | top | bottom | none>",
  "bounded":   <true | false>
}

Rules:
- arrange_in_sequence: ordered along axis, perpendicular unconstrained.
  axis=x/y, direction=ascending/descending, attribute=size/color/shape_type,
  region=none, bounded=false.
- arrange_in_line: ordered or evenly spaced along axis AND minimising
  perpendicular spread. axis=x/y, region=none, bounded=true.
  direction/attribute=none if just spacing, else set for ordered version.
- arrange_in_region: all shapes in a canvas subregion.
  axis=none, direction=none, attribute=none, region=left/right/top/bottom,
  bounded=true.
- arrange_in_groups: partitioned by attribute, each group in its own subregion.
  axis=none, direction=none, attribute=color/shape_type, region=none, bounded=true.

Examples:
  "sort shapes smallest to largest left to right"
  -> {"task":"arrange_in_sequence","axis":"x","direction":"ascending","attribute":"size","region":"none","bounded":false}
  "arrange shapes in a horizontal line evenly spaced"
  -> {"task":"arrange_in_line","axis":"x","direction":"none","attribute":"none","region":"none","bounded":true}
  "arrange shapes in a horizontal line sorted smallest to largest"
  -> {"task":"arrange_in_line","axis":"x","direction":"ascending","attribute":"size","region":"none","bounded":true}
  "move all shapes to the left side"
  -> {"task":"arrange_in_region","axis":"none","direction":"none","attribute":"none","region":"left","bounded":true}
  "group shapes by color"
  -> {"task":"arrange_in_groups","axis":"none","direction":"none","attribute":"color","region":"none","bounded":true}
  "group shapes by type"
  -> {"task":"arrange_in_groups","axis":"none","direction":"none","attribute":"shape_type","region":"none","bounded":true}
""".strip()


def _llm_parse(prompt: str) -> dict:
   try:
      import anthropic
   except ImportError:
      raise RuntimeError("anthropic package not installed")

   client  = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
   message = client.messages.create(
      model="claude-sonnet-4-6",
      max_tokens=256,
      system=_SYSTEM_PROMPT,
      messages=[{"role": "user", "content": prompt}],
   )
   raw = message.content[0].text.strip()
   raw = re.sub(r"^```[a-z]*\n?", "", raw)
   raw = re.sub(r"\n?```$",       "", raw)
   return json.loads(raw)


# ---------------------------------------------------------------------------
# stub fallback
# ---------------------------------------------------------------------------

def _stub_parse(prompt: str) -> dict:
   """
   keyword-matching fallback. handles common phrasings for all four tasks.
   ambiguous prompts default to arrange_in_sequence by size ascending.
   """

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
   is_line = any(kw in prompt for kw in ("in a line", "in a horizontal line",
                                          "in a vertical line", "evenly spaced",
                                          "in a row", "in a column"))
   axis      = _infer_axis(prompt)
   direction = _infer_direction(prompt)
   attribute = _infer_attribute(prompt)

   if is_line:
      has_order = attribute != "none" or any(kw in prompt for kw in (
         "sort", "order", "ascending", "descending", "smallest", "largest"))
      return {
         "task":      "arrange_in_line",
         "axis":      axis,
         "direction": direction if has_order else "none",
         "attribute": attribute if has_order else "none",
         "region":    "none",
         "bounded":   True,
      }

   # default: arrange_in_sequence
   return {
      "task":      "arrange_in_sequence",
      "axis":      axis,
      "direction": direction,
      "attribute": attribute if attribute != "none" else "size",
      "region":    "none",
      "bounded":   False,
   }


def _region_goal(region: str) -> dict:
   return {"task": "arrange_in_region", "axis": "none", "direction": "none",
           "attribute": "none", "region": region, "bounded": True}


def _groups_goal(attribute: str) -> dict:
   return {"task": "arrange_in_groups", "axis": "none", "direction": "none",
           "attribute": attribute, "region": "none", "bounded": True}


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
