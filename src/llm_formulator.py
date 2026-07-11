"""
llm_formulator.py
-----------------
Translates a natural-language scheduling request into a validated ConstraintSpec.

Supports:
  AnthropicBackend  -- claude-haiku-4-5-20251001  (default)
  OpenAIBackend     -- gpt-4o-mini

Flow:
  user NL string  ->  backend.complete(prompt)  ->  JSON string
                  ->  _extract_json()            ->  dict
                  ->  ConstraintSpec(**dict)     ->  validated spec
"""

import json
import re

from src.constraint_spec import ConstraintSpec


# ---------------------------------------------------------------------------
# System prompt with few-shot examples
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an assistant that converts natural-language scheduling
requests into a JSON object for an offshore exploration drilling campaign optimizer.

The JSON must match this schema exactly (all fields are optional except explanation):
{
  "n_wells":         <int 2-20 | null>,
  "min_flow_rate":   <int >= 0 | null>,
  "sub_blocks":      <list of "34/7" and/or "34/10" | null>,
  "n_rigs":          <int 1-3 | null>,
  "rig_cluster_lock":<{"Rig-Alpha"|"Rig-Beta"|"Rig-Gamma": "34/7"|"34/10"} | null>,
  "max_makespan_days":<int 100-1000 | null>,
  "prioritize":      <"makespan" | "deferred" | null>,
  "explanation":     "<one sentence describing what this spec does>"
}

Valid blocks:  "34/7"  (Snorre area)  and  "34/10"  (Statfjord area)
Valid rigs:    "Rig-Alpha", "Rig-Beta", "Rig-Gamma"

Examples:
---
User: "only drill Snorre wells, use 2 rigs"
JSON: {"sub_blocks": ["34/7"], "n_rigs": 2, "explanation": "Restrict campaign to Snorre (block 34/7) wells using 2 rigs."}

User: "focus on the highest-rate wells, top 8 only"
JSON: {"n_wells": 8, "explanation": "Schedule only the top 8 wells by DST oil rate."}

User: "minimum 1500 bbl/day DST rate"
JSON: {"min_flow_rate": 1500, "explanation": "Exclude wells with DST oil rate below 1500 bbl/day."}

User: "lock Rig-Alpha to Statfjord, minimize campaign length"
JSON: {"rig_cluster_lock": {"Rig-Alpha": "34/10"}, "prioritize": "makespan", "explanation": "Assign Rig-Alpha to start in the Statfjord cluster and emphasise short campaign duration."}

User: "maximum campaign length 470 days"
JSON: {"max_makespan_days": 470, "explanation": "Hard cap on campaign duration: only schedule solutions that finish within 470 days."}

User: "complete the full campaign in under 465 days, prioritize makespan"
JSON: {"max_makespan_days": 465, "prioritize": "makespan", "explanation": "Cap the campaign at 465 days and weight the search towards minimising makespan."}

User: "only Statfjord wells, deadline 480 days, use all 3 rigs"
JSON: {"sub_blocks": ["34/10"], "max_makespan_days": 480, "n_rigs": 3, "explanation": "Restrict to Statfjord (34/10) wells with a 480-day hard deadline using all 3 rigs."}
---

Respond with ONLY the JSON object. No markdown fences, no extra text.
"""


# ---------------------------------------------------------------------------
# LLM Backends
# ---------------------------------------------------------------------------

class AnthropicBackend:
    def __init__(self, api_key, model="claude-haiku-4-5-20251001"):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model  = model

    def complete(self, user_message):
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}]
        )
        return msg.content[0].text


class OpenAIBackend:
    def __init__(self, api_key, model="gpt-4o-mini"):
        import openai
        self.client = openai.OpenAI(api_key=api_key)
        self.model  = model

    def complete(self, user_message):
        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=512,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_message}
            ]
        )
        return resp.choices[0].message.content


# ---------------------------------------------------------------------------
# JSON extraction & validation
# ---------------------------------------------------------------------------

def _extract_json(text):
    """Pull the first {...} block from raw LLM output."""
    text = text.strip()
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return match.group(0)
    return text


def formulate(request, backend, max_retries=2):
    """
    Translate a natural-language request into a validated ConstraintSpec.

    Parameters
    ----------
    request   : str              -- plain-English scheduling request
    backend   : AnthropicBackend | OpenAIBackend
    max_retries: int             -- retries on validation failure

    Returns
    -------
    ConstraintSpec  on success
    raises ValueError on repeated failure
    """
    error_feedback = ""
    for attempt in range(max_retries + 1):
        prompt = request if not error_feedback else (
            "{}\n\n[Previous attempt failed: {}. Please fix and try again.]"
            .format(request, error_feedback)
        )
        raw  = backend.complete(prompt)
        json_str = _extract_json(raw)

        try:
            data = json.loads(json_str)
            spec = ConstraintSpec(**data)
            return spec
        except Exception as e:
            error_feedback = str(e)
            if attempt == max_retries:
                raise ValueError(
                    "LLM failed after {} attempts. Last error: {}\nLast output: {}"
                    .format(max_retries + 1, error_feedback, raw)
                )
