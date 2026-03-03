"""
llm_goal_parser.py

LLM goal parsing and embedding layer.

two public functions:

    parse_goal(prompt) -> dict
        interprets a natural language prompt into a validated goal dict.
        tries the Anthropic API first, falls back to stub on any failure.

    get_embedding(prompt) -> np.ndarray  [EMBEDDING_DIM]
        encodes a prompt as a dense vector using a local sentence-transformers
        model (all-MiniLM-L6-v2, ~80MB, runs offline).
        the embedding is cached in memory so repeated calls are free.
        used by shape_env.py and bc_train.py to condition the policy on goals.

wave 1 supported tasks:
    sort_by_size    — arrange shapes along an axis ordered by size
    group_by_color  — cluster same-color shapes into their own regions
    cluster         — general spatial grouping by a named attribute
    arrange_in_line — evenly spaced horizontal or vertical line
    arrange_in_grid — rectangular grid layout
    push_to_region  — move all shapes to a canvas region (left/right/top/bottom)

usage:
    from llm_goal_parser import parse_goal, get_embedding
    goal      = parse_goal("put all the red shapes on the left side")
    embedding = get_embedding("put all the red shapes on the left side")
"""

import json
import os
import re
import numpy as np

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
    "region":    str,   # "left" | "right" | "top" | "bottom" | "none"
}

SUPPORTED_TASKS = [
    "sort_by_size",
    "group_by_color",
    "group_by_shape_type",
    "cluster",
    "arrange_in_line",
    "arrange_in_grid",
    "push_to_region",
]

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_goal(prompt: str) -> dict:
    """
    parse a natural-language prompt into a validated goal dict.

    if ANTHROPIC_API_KEY is set, tries the API first and falls back to
    the stub on failure. if no key is set, goes straight to the stub
    without logging anything — this is the expected path during training.

    args:
        prompt: user-provided instruction string.

    returns:
        a validated goal dict matching GOAL_SCHEMA.
    """
    if os.environ.get("ANTHROPIC_API_KEY", ""):
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
  "task":      "<one of: sort_by_size | group_by_color | cluster | arrange_in_line | arrange_in_grid | push_to_region>",
  "axis":      "<one of: x | y | none>",
  "direction": "<one of: ascending | descending | none>",
  "attribute": "<one of: size | color | none>",
  "region":    "<one of: left | right | top | bottom | none>"
}

Rules:
- sort_by_size    → shapes arranged along an axis ordered by size.
                    set axis (x or y) and direction (ascending = small→large / top→bottom).
                    attribute = "size". region = "none".
- group_by_color  → same-color shapes moved near each other in distinct regions.
                    axis = "none", direction = "none", attribute = "color", region = "none".
- cluster         → general spatial grouping. set attribute to the relevant property.
                    axis = "none", direction = "none", region = "none".
- arrange_in_line → evenly spaced line. axis = "x" for horizontal, "y" for vertical.
                    direction = "none", attribute = "none", region = "none".
- arrange_in_grid → rectangular grid layout.
                    axis = "none", direction = "none", attribute = "none", region = "none".
- push_to_region  → move all shapes to a canvas region.
                    axis = "none", direction = "none", attribute = "none".
                    region = "left" | "right" | "top" | "bottom".

Examples:
  "sort shapes smallest to largest left to right"
  → {"task":"sort_by_size","axis":"x","direction":"ascending","attribute":"size","region":"none"}

  "biggest shapes at the top, smallest at the bottom"
  → {"task":"sort_by_size","axis":"y","direction":"descending","attribute":"size","region":"none"}

  "group the shapes by colour"
  → {"task":"group_by_color","axis":"none","direction":"none","attribute":"color","region":"none"}

  "arrange shapes in a horizontal line evenly spaced"
  → {"task":"arrange_in_line","axis":"x","direction":"none","attribute":"none","region":"none"}

  "arrange shapes in a grid"
  → {"task":"arrange_in_grid","axis":"none","direction":"none","attribute":"none","region":"none"}

  "move all shapes to the left side"
  → {"task":"push_to_region","axis":"none","direction":"none","attribute":"none","region":"left"}

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
    hardcoded pattern matching standing in for the LLM.
    handles the most common phrasings for each wave 1 task type.
    """

    # push_to_region — check first as it's most specific
    if any(kw in prompt for kw in ("left side", "move left", "push left")):
        return {"task": "push_to_region", "axis": "none", "direction": "none",
                "attribute": "none", "region": "left"}
    if any(kw in prompt for kw in ("right side", "move right", "push right")):
        return {"task": "push_to_region", "axis": "none", "direction": "none",
                "attribute": "none", "region": "right"}
    if any(kw in prompt for kw in ("to the top", "move up", "push up")):
        return {"task": "push_to_region", "axis": "none", "direction": "none",
                "attribute": "none", "region": "top"}
    if any(kw in prompt for kw in ("to the bottom", "move down", "push down")):
        return {"task": "push_to_region", "axis": "none", "direction": "none",
                "attribute": "none", "region": "bottom"}

    # arrange_in_grid
    if "grid" in prompt:
        return {"task": "arrange_in_grid", "axis": "none", "direction": "none",
                "attribute": "none", "region": "none"}

    # arrange_in_line
    if any(kw in prompt for kw in ("horizontal line", "vertical line",
                                    "evenly spaced", "in a line")):
        axis = "y" if "vertical" in prompt else "x"
        return {"task": "arrange_in_line", "axis": axis, "direction": "none",
                "attribute": "none", "region": "none"}

    # group_by_shape_type — check before color keywords
    shape_type_keywords = ("type", "shape", "circle", "square", "triangle",
                           "by type", "same type", "same shape")
    if any(kw in prompt for kw in shape_type_keywords):
        # avoid matching "sort shapes" — only match explicit grouping intent
        if any(kw in prompt for kw in ("group", "put", "same", "together", "separately")):
            return {"task": "group_by_shape_type", "axis": "none", "direction": "none",
                    "attribute": "shape_type", "region": "none"}

    # group_by_color
    color_keywords = ("color", "colour", "red", "blue", "green", "yellow",
                      "purple", "same color", "by colour", "by color")
    if any(kw in prompt for kw in color_keywords):
        return {"task": "group_by_color", "axis": "none", "direction": "none",
                "attribute": "color", "region": "none"}

    # cluster
    if "cluster" in prompt:
        attribute = "color" if "color" in prompt or "colour" in prompt else "size"
        return {"task": "cluster", "axis": "none", "direction": "none",
                "attribute": attribute, "region": "none"}

    # general group without color → cluster by size
    if "group" in prompt and not any(kw in prompt for kw in color_keywords):
        return {"task": "cluster", "axis": "none", "direction": "none",
                "attribute": "size", "region": "none"}

    # default: sort_by_size
    axis = "y" if any(kw in prompt for kw in (
        "top to bottom", "vertical", "column", "up to down")) else "x"

    direction = "descending" if any(kw in prompt for kw in (
        "right to left", "largest first", "descend", "big to small",
        "biggest first", "largest to smallest", "biggest at top",
        "biggest at the top")) else "ascending"

    return {"task": "sort_by_size", "axis": axis, "direction": direction,
            "attribute": "size", "region": "none"}


# ---------------------------------------------------------------------------
# goal embedding
# ---------------------------------------------------------------------------

# module-level cache: prompt string -> np.ndarray
# avoids reloading the model or re-encoding the same prompt twice.
_embedding_model  = None
_embedding_cache: dict = {}


def get_embedding(prompt: str) -> np.ndarray:
    """
    encode a natural language prompt as a dense float32 vector.

    uses sentence-transformers all-MiniLM-L6-v2 (384-dim, ~80MB, offline).
    the model is loaded once on first call and cached for the session.
    individual prompt embeddings are also cached so training loop calls
    with the same prompt string are effectively free after the first call.

    args:
        prompt: any natural language goal string.

    returns:
        np.ndarray of shape (EMBEDDING_DIM,) = (384,), dtype float32.

    raises:
        RuntimeError if sentence-transformers is not installed.
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

    embedding = _embedding_model.encode(prompt, convert_to_numpy=True).astype(np.float32)
    _embedding_cache[prompt] = embedding
    return embedding


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_goal(goal: dict):
    """raise ValueError if the goal dict is missing fields or has bad values."""
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
    if goal["attribute"] not in ("size", "color", "shape_type", "none"):
        raise ValueError(
            f"attribute must be 'size', 'color', 'shape_type', or 'none', "
            f"got '{goal['attribute']}'"
        )
    if goal["region"] not in ("left", "right", "top", "bottom", "none"):
        raise ValueError(
            f"region must be 'left', 'right', 'top', 'bottom', or 'none', "
            f"got '{goal['region']}'"
        )