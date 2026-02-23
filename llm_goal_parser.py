"""
llm_goal_parser.py

stub for the LLM goal parsing layer.
in production, parse_goal() would send the user's prompt to an LLM
(e.g. GPT-4o via the openai SDK) and return a structured goal dict.

right now it just pattern-matches a few hardcoded phrases so the
rest of the system can run without an API key.
"""

# this is the schema every goal must conform to.
# the RL agent and reward function both read from this.
GOAL_SCHEMA = {
   "task":      str,   # e.g. "sort_by_size"
   "axis":      str,   # "x" or "y"
   "direction": str,   # "ascending" or "descending"
}

# supported tasks (expand as you add reward functions)
SUPPORTED_TASKS = ["sort_by_size"]


def parse_goal(prompt: str) -> dict:
   """
   parse a natural language prompt into a structured goal dict.

   args:
      prompt: user-provided instruction string

   returns:
      a goal dict matching GOAL_SCHEMA

   raises:
      ValueError if the prompt can't be mapped to a supported task
   """
   prompt_lower = prompt.lower()

   # --- stub logic: real LLM call would replace this block ---
   goal = _stub_parse(prompt_lower)
   # ----------------------------------------------------------

   _validate_goal(goal)
   return goal


def _stub_parse(prompt: str) -> dict:
   """
   hardcoded pattern matching standing in for an LLM.
   a real implementation would call something like:

      response = openai_client.chat.completions.create(
         model="gpt-4o",
         messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
         ],
         response_format={"type": "json_object"},
      )
      return json.loads(response.choices[0].message.content)
   """
   # default goal
   goal = {
      "task":      "sort_by_size",
      "axis":      "x",
      "direction": "ascending",
   }

   # axis
   if "top to bottom" in prompt or "vertical" in prompt or "column" in prompt:
      goal["axis"] = "y"
   else:
      goal["axis"] = "x"

   # direction
   if ("right to left" in prompt or "largest first" in prompt
         or "descend" in prompt or "big to small" in prompt):
      goal["direction"] = "descending"
   else:
      goal["direction"] = "ascending"

   return goal


def _validate_goal(goal: dict):
   """sanity check that a parsed goal has the right shape."""
   for key, expected_type in GOAL_SCHEMA.items():
      if key not in goal:
         raise ValueError(f"goal is missing required key: '{key}'")
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
   if goal["axis"] not in ("x", "y"):
      raise ValueError(f"axis must be 'x' or 'y', got '{goal['axis']}'")
   if goal["direction"] not in ("ascending", "descending"):
      raise ValueError(
         f"direction must be 'ascending' or 'descending', "
         f"got '{goal['direction']}'"
      )


# --- example system prompt for when you wire in a real LLM ---
SYSTEM_PROMPT = """
you are a goal parser for a 2d shape manipulation agent.
the user will describe a task in natural language.
respond ONLY with a valid JSON object matching this schema exactly:

{
   "task":      "<one of: sort_by_size>",
   "axis":      "<one of: x, y>",
   "direction": "<one of: ascending, descending>"
}

examples:
   "sort the shapes from smallest to largest left to right"
   -> {"task": "sort_by_size", "axis": "x", "direction": "ascending"}

   "arrange shapes top to bottom, biggest at the top"
   -> {"task": "sort_by_size", "axis": "y", "direction": "descending"}

do not include any explanation, only the JSON object.
""".strip()
