"""Validate and render flat instance-level commonsense retrieval results."""

from __future__ import annotations

import copy
from typing import Any


def _required_text(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing text at {location}.")
    return value.strip()


def _candidate_unit(
    candidate: Any,
    *,
    situation_index: int,
    rank: int,
) -> tuple[str, str, str]:
    location = f"situation[{situation_index}].candidate[{rank}]"
    if not isinstance(candidate, dict):
        raise ValueError(f"{location} must be an object.")
    unit = candidate.get("commonsense_unit")
    if not isinstance(unit, dict):
        raise ValueError(f"{location} has no commonsense_unit object.")
    candidate_id = candidate.get("id") or unit.get("unit_id")
    unit_id = unit.get("unit_id")
    if candidate_id and unit_id and candidate_id != unit_id:
        raise ValueError(
            f"{location} ID mismatch: {candidate_id!r} != {unit_id!r}."
        )
    return (
        str(candidate_id or f"instance-{situation_index}-{rank}"),
        _required_text(unit.get("situation"), f"{location}.situation"),
        _required_text(
            unit.get("commonsense_rule"), f"{location}.commonsense_rule"
        ),
    )


def build_instance_level_context(
    document: dict[str, Any],
    *,
    top_k: int,
    image_only: bool = False,
) -> str:
    """Render Top-K flat candidates without inventing generalized structure."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    situations = document.get("test_situations") if isinstance(document, dict) else None
    if not isinstance(situations, list):
        raise ValueError("Document has no test_situations list.")

    lines: list[str] = []
    for index, situation in enumerate(situations, 1):
        if not isinstance(situation, dict):
            raise ValueError(f"test_situations[{index - 1}] must be an object.")
        description = _required_text(
            situation.get("situation_description"),
            f"situation[{index}].situation_description",
        )
        range_label = "inferred operations" if image_only else "steps"
        lines.append(
            f"[Situation {index}] {range_label} {situation.get('start_step')}-"
            f"{situation.get('end_step')}: {description}"
        )
        candidates = situation.get("related_commonsense")
        if not isinstance(candidates, list) or len(candidates) < top_k:
            count = len(candidates) if isinstance(candidates, list) else 0
            raise ValueError(
                f"Situation {index} has {count} retrieved units; expected at least {top_k}."
            )
        lines.append("  Retrieved instance-level commonsense:")
        for rank, candidate in enumerate(candidates[:top_k], 1):
            _, unit_situation, rule = _candidate_unit(
                candidate, situation_index=index, rank=rank
            )
            lines.append(f"    {rank}. Situation: {unit_situation}")
            lines.append(f"       Expected behavior: {rule}")
        lines.append("")
    return "\n".join(lines).strip()


def normalise_as_singleton_families(
    document: dict[str, Any], *, top_k: int
) -> dict[str, Any]:
    """Adapt flat units to KuiTest's existing mapping renderer.

    Each record is represented as a base-only singleton. No variant relation is
    introduced; the adapter exists only to reuse the page-to-action mapping.
    """
    situations = document.get("test_situations") if isinstance(document, dict) else None
    if not isinstance(situations, list):
        raise ValueError("Document has no test_situations list.")
    converted = copy.deepcopy(document)
    converted["test_situations"] = []
    for index, situation in enumerate(situations, 1):
        if not isinstance(situation, dict):
            raise ValueError(f"test_situations[{index - 1}] must be an object.")
        candidates = situation.get("related_commonsense")
        if not isinstance(candidates, list) or len(candidates) < top_k:
            count = len(candidates) if isinstance(candidates, list) else 0
            raise ValueError(
                f"Situation {index} has {count} retrieved units; expected at least {top_k}."
            )
        families = []
        for rank, candidate in enumerate(candidates[:top_k], 1):
            unit_id, unit_situation, rule = _candidate_unit(
                candidate, situation_index=index, rank=rank
            )
            families.append(
                {
                    "family_id": unit_id,
                    "rerank_score": candidate.get("rerank_score"),
                    "base_unit": {
                        "situation": unit_situation,
                        "commonsense_rule": rule,
                    },
                    "matched_unit": {
                        "type": "BASE",
                        "unit_id": unit_id,
                        "situation": unit_situation,
                        "commonsense_rule": rule,
                    },
                }
            )
        converted_situation = copy.deepcopy(situation)
        converted_situation.pop("related_commonsense", None)
        converted_situation["related_rule_families"] = families
        converted["test_situations"].append(converted_situation)
    return converted
