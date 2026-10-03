from __future__ import annotations
import json
import re
from typing import Any, Dict

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)

class DetectorParseError(Exception):
    pass

def _decode_objects(text: str):
    decoder = json.JSONDecoder()
    objects = []
    cursor = 0
    while True:
        index = text.find("{", cursor)
        if index < 0:
            break
        try:
            value, consumed = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            cursor = index + 1
            continue
        if isinstance(value, dict):
            objects.append(value)
        cursor = index + max(consumed, 1)
    return objects

def parse_detector_output(llm_output: str) -> Dict[str, Any]:
    candidates = []
    for fenced in _JSON_FENCE_RE.findall(llm_output):
        candidates.extend(_decode_objects(fenced))
    candidates.extend(_decode_objects(llm_output))
    if not candidates:
        raise DetectorParseError("No valid JSON object found in LLM output.")
    return candidates[-1]

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
