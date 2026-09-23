"""LLM decision prompts: one JSON contract for choice / score / noul."""
from __future__ import annotations

import json
from typing import Any

from .schemas import Sample

SYSTEM = """You are a calibrated decision engine. You do not chat.

Given a TEXT and a decision task, output ONLY a single JSON object. No prose, no markdown fences.

Rules:
1. Every probability must be in [0, 1].
2. Probabilities over the option set must sum to 1.0 (±0.02).
3. answer must be exactly one of the allowed options (or a level number for score tasks).
4. confidence in [0, 1] reflects how peaked your distribution is (1 = certain, 0 = uniform).
5. Judge only from TEXT. Prefer honest uncertainty over forced certainty.
"""

CHOICE_USER = """TASK: multi-class decision (choice)
QUESTION: Which option best describes the TEXT?

OPTIONS:
{options}

TEXT:
{text}

Return JSON exactly:
{{"answer": "<option_key>", "probabilities": {{"<option_key>": p, ...}}, "confidence": c}}
"""

SCORE_USER = """TASK: ordinal score
QUESTION: Rate the TEXT on the following levels (0 = lowest, {max_level} = highest).

LEVELS:
{options}

TEXT:
{text}

Return JSON exactly:
{{"answer": <expected_score_float>, "probabilities": {{"0": p0, "1": p1, ...}}, "confidence": c}}
answer must be the probability-weighted score sum(i * p_i). Keep answer within [0, {max_level}].
"""

NOUL_USER = """TASK: binary judgment (noul)
QUESTION: {question}

{options_block}

TEXT:
{text}

Return JSON exactly:
{{"answer": true, "probabilities": {{"true": p, "false": 1-p}}, "confidence": c}}
Use "answer": false when p_true < 0.5. Use "answer": true when p_true >= 0.5.
"""


def _format_options(labels: list[str], descriptions: dict[str, str]) -> str:
    lines = []
    for lab in labels:
        desc = (descriptions or {}).get(lab, str(lab).replace("_", " "))
        lines.append(f"- {lab}: {desc}")
    return "\n".join(lines)


def build_llm_messages(sample: Sample) -> list[dict[str, str]]:
    if sample.task_type == "choice":
        user = CHOICE_USER.format(
            options=_format_options(sample.labels, sample.label_descriptions),
            text=sample.text,
        )
    elif sample.task_type == "score":
        levels = [str(x) for x in (sample.levels or [0, 1, 2, 3, 4])]
        max_level = max(int(x) for x in levels)
        options = _format_options(levels, sample.level_descriptions)
        user = SCORE_USER.format(options=options, text=sample.text, max_level=max_level)
    elif sample.task_type == "noul":
        # Prefer a short question from meta or generic
        q = sample.meta.get("question") or "Is the TEXT true / affirmative for the statement in the question?"
        # BoolQ already embeds question in text; keep generic
        if "question:" in sample.text.lower():
            q = "Answer the question embedded in TEXT with true or false."
        options_block = "Interpret true=yes/affirmative, false=no/negative."
        user = NOUL_USER.format(question=q, options_block=options_block, text=sample.text)
    else:
        raise ValueError(f"unknown task_type {sample.task_type}")
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
    ]


def build_jev_request(sample: Sample, model: str = "typesafe/jev-1.13") -> dict[str, Any]:
    """System One request body (TypeSafe / OpenRouter systemone)."""
    if sample.task_type == "choice":
        criteria = {
            lab: (sample.label_descriptions or {}).get(lab, str(lab).replace("_", " "))
            for lab in sample.labels
        }
        question = {
            "type": "choice",
            "instructions": "Which option best describes the text?",
            "criteria": criteria,
        }
    elif sample.task_type == "score":
        criteria = [
            (sample.level_descriptions or {}).get(str(i), str(i))
            for i in (sample.levels or [0, 1, 2, 3, 4])
        ]
        question = {
            "type": "score",
            "instructions": "Rate the sentiment / intensity level of the text.",
            "criteria": criteria,
        }
    else:
        question = {
            "type": "noul",
            "instructions": "Is the answer to the question in the text true?",
            "criteria": {"true": "Yes / affirmative", "false": "No / negative"},
        }
    return {
        "model": model,
        "state": sample.text,
        "questions": {"q": question},
    }


def parse_llm_json(content: str) -> dict[str, Any]:
    """Parse the decision JSON. One attempt after stripping fences. Raises ValueError."""
    text = (content or "").strip()
    if text.startswith("```"):
        # strip a single markdown fence if present
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
        # if still has fences inside, take first {...}
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in model output")
    obj = json.loads(text[start : end + 1])
    if not isinstance(obj, dict):
        raise ValueError("JSON root must be object")
    return obj
