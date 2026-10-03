"""Map page-indexed situations to KuiTest's action-indexed transitions.

The retrieved ``start_step`` and ``end_step`` values describe screenshot/page
interfaces.  KuiTest action step ``s`` is performed on Page ``s`` and its
response is observed on Page ``s + 1``.  Therefore a situation is assigned to
an action when the action's *source page* is inside the situation page range.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List


MAPPING_SCHEMA_VERSION = 1


def load_situation_document(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        document = json.load(file)
    situations = document.get("test_situations") if isinstance(document, dict) else None
    if not isinstance(situations, list):
        raise ValueError(f"Invalid situation document (missing test_situations): {path}")
    return document


def _required_text(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing text at {location}")
    return value.strip()


def _normalise_family(candidate: Dict[str, Any], location: str) -> Dict[str, Any]:
    if not isinstance(candidate, dict):
        raise ValueError(f"Invalid rule family at {location}")
    base = candidate.get("base_unit")
    matched = candidate.get("matched_unit")
    if not isinstance(base, dict) or not isinstance(matched, dict):
        raise ValueError(f"Missing base_unit or matched_unit at {location}")

    result = {
        "family_id": candidate.get("family_id"),
        "rerank_score": candidate.get("rerank_score"),
        "base": {
            "situation": _required_text(base.get("situation"), f"{location}.base_unit.situation"),
            "commonsense_rule": _required_text(
                base.get("commonsense_rule"),
                f"{location}.base_unit.commonsense_rule",
            ),
        },
    }
    matched_type = matched.get("type")
    if matched_type not in {"BASE", "VARIANT_MEMBER"}:
        raise ValueError(f"Invalid matched_unit.type={matched_type!r} at {location}")
    if matched_type == "VARIANT_MEMBER":
        result["matched_variant"] = {
            "unit_id": matched.get("unit_id"),
            "situation": _required_text(
                matched.get("situation"), f"{location}.matched_unit.situation"
            ),
            "commonsense_rule": _required_text(
                matched.get("commonsense_rule"),
                f"{location}.matched_unit.commonsense_rule",
            ),
        }
    return result


def _history_for_situation(
    steps: List[Dict[str, Any]],
    start_page: int,
    current_action_step: int,
) -> List[Dict[str, Any]]:
    """Return only the situation prefix visible up to the current response."""
    history = []
    for action_step in range(start_page, current_action_step + 1):
        step = steps[action_step - 1]
        history.append(
            {
                "action_step_id": action_step,
                "before_page_id": action_step,
                "observed_page_id": action_step + 1,
                "before_page_description": step["before_page_description"],
                "action_description": step["action_description"],
                "after_page_description": step["after_page_description"],
            }
        )
    return history


def build_case_mapping(
    case: Dict[str, Any],
    situation_document: Dict[str, Any],
    top_k: int,
    source_file: str | None = None,
) -> Dict[str, Any]:
    """Build a transparent per-action mapping from page-indexed situations."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    steps = case["steps"]
    page_count = len(steps) + 1
    normalised_situations = []

    for index, situation in enumerate(situation_document["test_situations"], 1):
        if not isinstance(situation, dict):
            raise ValueError(f"test_situations[{index - 1}] is not an object")
        start_page = situation.get("start_step")
        end_page = situation.get("end_step")
        if not isinstance(start_page, int) or not isinstance(end_page, int):
            raise ValueError(f"Situation {index} has non-integer page bounds")
        if start_page < 1 or end_page < start_page or end_page > page_count:
            raise ValueError(
                f"Situation {index} has invalid page range {start_page}-{end_page}; "
                f"case has Pages 1-{page_count}"
            )
        candidates = situation.get("related_rule_families")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(f"Situation {index} has no related_rule_families")
        families = [
            _normalise_family(candidate, f"situation[{index}].family[{rank}]")
            for rank, candidate in enumerate(candidates[:top_k], 1)
        ]
        normalised_situations.append(
            {
                "situation_index": index,
                "start_page_id": start_page,
                "end_page_id": end_page,
                "situation_description": _required_text(
                    situation.get("situation_description"),
                    f"situation[{index}].situation_description",
                ),
                "rule_families": families,
            }
        )

    step_mappings = []
    for step in steps:
        action_step = step["step_id"]
        matched = []
        for situation in normalised_situations:
            # Bounds are PAGE interfaces. Action s is anchored on source Page s.
            if situation["start_page_id"] <= action_step <= situation["end_page_id"]:
                item = dict(situation)
                item["history_until_current_response"] = _history_for_situation(
                    steps,
                    situation["start_page_id"],
                    action_step,
                )
                matched.append(item)
        step_mappings.append(
            {
                "action_step_id": action_step,
                "before_page_id": action_step,
                "observed_page_id": action_step + 1,
                "matched_situations": matched,
            }
        )

    mapping = {
        "schema_version": MAPPING_SCHEMA_VERSION,
        "mapping_semantics": (
            "Situation start_step/end_step are page-interface IDs. "
            "Action Step s is mapped by its source Page s; its response is "
            "observed on Page s+1."
        ),
        "app": case["app"],
        "testid": case["testid"],
        "page_count": page_count,
        "action_count": len(steps),
        "top_k": top_k,
        "source_file": source_file,
        "step_mappings": step_mappings,
    }
    mapping["fingerprint"] = mapping_fingerprint(mapping)
    return mapping


def mapping_fingerprint(mapping: Dict[str, Any]) -> str:
    value = dict(mapping)
    value.pop("fingerprint", None)
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def get_step_mapping(mapping: Dict[str, Any], action_step_id: int) -> Dict[str, Any]:
    for item in mapping.get("step_mappings", []):
        if item.get("action_step_id") == action_step_id:
            return item
    raise KeyError(f"No mapping for action step {action_step_id}")


def render_step_context(step_mapping: Dict[str, Any]) -> str:
    """Render one action's mapped situations/rules for prompt injection."""
    matched = step_mapping.get("matched_situations", [])
    if not matched:
        return "[No retrieved test situation covers the current source page.]"

    lines = [
        f"Current Action Step: {step_mapping['action_step_id']}",
        f"Source interface: Page {step_mapping['before_page_id']}",
        f"Observed response interface: Page {step_mapping['observed_page_id']}",
    ]
    for situation in matched:
        lines.extend(
            [
                "",
                (
                    f"[Situation {situation['situation_index']}] Pages "
                    f"{situation['start_page_id']}-{situation['end_page_id']}: "
                    f"{situation['situation_description']}"
                ),
                "Observed history from the situation start through the current response:",
            ]
        )
        for history in situation["history_until_current_response"]:
            lines.append(
                f"  - Action Step {history['action_step_id']} "
                f"(Page {history['before_page_id']} -> Page "
                f"{history['observed_page_id']}): "
                f"Before: {history['before_page_description']} | "
                f"Action: {history['action_description']} | "
                f"After: {history['after_page_description']}"
            )
        lines.append("Retrieved common-sense rule families:")
        for rank, family in enumerate(situation["rule_families"], 1):
            base = family["base"]
            family_id = family.get("family_id") or "unknown"
            lines.append(f"  {rank}. Family {family_id} base:")
            lines.append(f"     Situation: {base['situation']}")
            lines.append(f"     Expected behavior: {base['commonsense_rule']}")
            variant = family.get("matched_variant")
            if variant:
                lines.append("     Matched variant:")
                lines.append(f"       Situation: {variant['situation']}")
                lines.append(
                    f"       Expected behavior: {variant['commonsense_rule']}"
                )
    return "\n".join(lines)


def rule_family_ids(step_mapping: Dict[str, Any]) -> List[str]:
    ids = []
    for situation in step_mapping.get("matched_situations", []):
        for family in situation.get("rule_families", []):
            family_id = family.get("family_id")
            if family_id and family_id not in ids:
                ids.append(family_id)
    return ids
