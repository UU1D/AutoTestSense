# -*- coding: utf-8 -*-
"""
Detector output parser (optional).
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, Tuple


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"(\{.*\})", re.DOTALL)


class DetectorParseError(Exception):
    pass


def _extract_json(text: str) -> str:
    text = text.strip()

    m = _JSON_FENCE_RE.search(text)
    if m:
        return m.group(1).strip()

    # Try to find the first JSON object
    m = _JSON_OBJECT_RE.search(text)
    if m:
        return m.group(1).strip()

    raise DetectorParseError("No JSON object found in LLM output.")


def parse_detector_output(llm_output: str) -> Dict[str, Any]:
    """
    Parse LLM output into JSON dict. 
    """
    js = _extract_json(llm_output)
    try:
        return json.loads(js)
    except json.JSONDecodeError as e:
        raise DetectorParseError(f"Invalid JSON: {e}") from e


def extract_bug_record(parsed: Dict[str, Any]) -> Dict[str, Any]:

    return {
        "bug_found": bool(parsed.get("bug_found", False)),
        "bug_type": parsed.get("bug_type", "unknown"),
        "bug_page_step": parsed.get("bug_page_step", None),
        "bug_description": parsed.get("bug_description", parsed.get("summary", "")),
        "confidence": parsed.get("confidence", None),
        "evidence_steps": parsed.get("evidence_steps", parsed.get("step_analysis", [])),
        "exploration_feedback": parsed.get("exploration_feedback", []),
    }
