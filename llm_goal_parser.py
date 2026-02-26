"""
llm_goal_parser.py

LLM goal parsing layer.

parse_goal() first tries to call the Anthropic API (claude-sonnet-4-6)
to interpret the user's prompt.  If the call fails for any reason —
missing API key, network error, malformed JSON — it falls back to the
hardcoded stub so the rest of the system keeps running without interruption.

Expanded goal schema now supports:
    sort_by_size   — arrange shapes along an axis ordered by size
    group_by_color — cluster same-color shapes into their own regions
    cluster        — general spatial grouping by a named attribute

Usage:
    from llm_goal_parser import parse_goal
    goal = parse_goal("put all the red shapes on the left side")
    # → {"task": "group_by_color", "axis": "none",
    #    "direction": "none", "attribute": "color"}
"""

import json
import os
import re

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

# Every goal dict must conform to this schema.
# The RL agent, reward function, and oracle all read from it.
GOAL_SCHEMA = {
    "task":      str,   # one of SUPPORTED_TASKS
    "axis":      str,   # "x" | "y" | "none"
    "direction": str,   # "ascending" | "descending" | "none"
    "attribute": str,   # "size" | "color" | "none"
}

SUPPORTED_TASKS = ["sort_by_size", "group_by_color", "cluster"]

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_goal(prompt: str) -> dict:
    """
    Parse a natural-language prompt into a validated goal dict.

    Tries the Anthropic API first; falls back to stub on any failure.

    Args:
        prompt: User-provided instruction string.

    Returns:
        A validated goal dict matching GOAL_SCHEMA.

    Raises:
        ValueError: If neither the LLM nor the stub can produce a valid goal.
    """
    try:
        goal = _llm_parse(prompt)
        _validate_goal(goal)
        return goal
    except Exception as llm_err:
        print(f"[goal_parser] LLM parse failed ({llm_err}), using stub fallback.")
        goal = _stub_parse(prompt.lower())
        _validate_goal(goal)
        return goal


# ---------------------------------------------------------------------------
# LLM backend — Anthropic claude-sonnet-4-6
# ---------------------------------------------------------------------------

# The system prompt is the contract between us and the model.
# It specifies the schema, valid values, and concrete examples.
# Adding new tasks here is all you need to do on the LLM side.
_SYSTEM_PROMPT = """
You are a goal parser for a 2D shape manipulation agent.
The user will describe a manipulation task in natural language.
Respond ONLY with a single valid JSON object — no markdown, no extra text.

Schema (all fields required):
{
  "task":      "<one of: sort_by_size | group_by_color | cluster>",
  "axis":      "<one of: x | y | none>",
  "direction": "<one of: ascending | descending | none>",
  "attribute": "<one of: size | color | none>"
}

Rules:
- sort_by_size   → shapes arranged along an axis ordered by size.
                   Set axis (x or y) and direction (ascending = small→large/top→bottom).
                   attribute = "size".
- group_by_color → same-color shapes moved near each other in distinct regions.
                   axis = "none", direction = "none", attribute = "color".
- cluster        → general spatial grouping. Set attribute to the relevant property.
                   axis = "none", direction = "none".

Examples:
  "sort shapes smallest to largest left to right"
  → {"task":"sort_by_size","axis":"x","direction":"ascending","attribute":"size"}

  "biggest shapes at the top, smallest at the bottom"
  → {"task":"sort_by_size","axis":"y","direction":"descending","attribute":"size"}

  "group the shapes by colour"
  → {"task":"group_by_color","axis":"none","direction":"none","attribute":"color"}

  "put all red shapes together and blue shapes together"
  → {"task":"group_by_color","axis":"none","direction":"none","attribute":"color"}

  "cluster shapes by size"
  → {"task":"cluster","axis":"none","direction":"none","attribute":"size"}

Respond with ONLY the JSON object. No commentary.
""".strip()


def _llm_parse(prompt: str) -> dict:
    """
    Call the Anthropic API and parse the returned JSON goal.
    Raises RuntimeError or json.JSONDecodeError on any failure
    so parse_goal() can catch it and fall back to the stub.
    """
    try:
        import anthropic
    except ImportError:
        raise RuntimeError("anthropic package not installed — run: pip install anthropic")

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set")

    client  = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=256,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = message.content[0].text.strip()

    # Strip accidental markdown fences the model sometimes adds despite instructions
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$",       "", raw)

    return json.loads(raw)


# ---------------------------------------------------------------------------
# Stub fallback — no API key required
# ---------------------------------------------------------------------------

def _stub_parse(prompt: str) -> dict:
    """
    Hardcoded pattern matching standing in for the LLM.
    Handles the most common phrasings for each task type.

    A real implementation replaces _llm_parse() above rather than this.
    This exists purely so the system boots without credentials.
    """

    # --- Task detection (check most-specific first) ---

    # Color / grouping keywords → group_by_color
    color_keywords = ("color", "colour", "red", "blue", "green", "yellow", "purple",
                      "same color", "by colour", "by color")
    if any(kw in prompt for kw in color_keywords):
        return {
            "task":      "group_by_color",
            "axis":      "none",
            "direction": "none",
            "attribute": "color",
        }

    # Explicit cluster keyword
    if "cluster" in prompt:
        attribute = "color" if "color" in prompt or "colour" in prompt else "size"
        return {
            "task":      "cluster",
            "axis":      "none",
            "direction": "none",
            "attribute": attribute,
        }

    # General "group" without color context → cluster by size
    if "group" in prompt and not any(kw in prompt for kw in color_keywords):
        return {
            "task":      "cluster",
            "axis":      "none",
            "direction": "none",
            "attribute": "size",
        }

    # --- Default: sort_by_size ---

    # Axis
    if any(kw in prompt for kw in ("top to bottom", "vertical", "column",
                                    "up to down", "top-to-bottom")):
        axis = "y"
    else:
        axis = "x"

    # Direction
    if any(kw in prompt for kw in ("right to left", "largest first", "descend",
                                    "big to small", "biggest first",
                                    "largest to smallest", "biggest at top",
                                    "biggest at the top")):
        direction = "descending"
    else:
        direction = "ascending"

    return {
        "task":      "sort_by_size",
        "axis":      axis,
        "direction": direction,
        "attribute": "size",
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_goal(goal: dict):
    """Raise ValueError if the goal dict is missing fields or has bad values."""
    for key, expected_type in GOAL_SCHEMA.items():
        if key not in goal:
            raise ValueError(f"goal missing required key: '{key}'")
        if not isinstance(goal[key], expected_type):
            raise ValueError(
                f"goal['{key}'] should be {expected_type.__name__}, "
                f"got {type(goal[key]).__name__}"
            )

    if goal["task"] not in SUPPORTED_TASKS:
        raise ValueError(
            f"unsupported task: '{goal['task']}'. "
            f"supported: {SUPPORTED_TASKS}"
        )
    if goal["axis"] not in ("x", "y", "none"):
        raise ValueError(f"axis must be 'x', 'y', or 'none', got '{goal['axis']}'")
    if goal["direction"] not in ("ascending", "descending", "none"):
        raise ValueError(
            f"direction must be 'ascending', 'descending', or 'none', "
            f"got '{goal['direction']}'"
        )
    if goal["attribute"] not in ("size", "color", "none"):
        raise ValueError(
            f"attribute must be 'size', 'color', or 'none', "
            f"got '{goal['attribute']}'"
        )