"""Build deterministic family-level generalization batches from membership results."""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any


from commonsense_repro.common.paths import PACKAGE_ROOT as PROJECT_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from commonsense_repro.common import llm_pipeline_common as llm_io


DATASET_ROOT = Path(
    "work/stage4/"
    "commonsense_consolidation"
)
DEFAULT_GROUPS_FILE = (
    DATASET_ROOT
    / "consolidation_judgment/deepseek_flash_v1_1/membership_groups.json"
)
DEFAULT_OUTPUT_FILE = (
    DATASET_ROOT / "batch_consolidation/generalization_plan_max18.json"
)
DEFAULT_MAX_PENDING_UNITS = 18

MEMBERSHIP_ORDER = {
    "EQUIVALENT_MEMBER": 0,
    "VARIANT_MEMBER": 1,
}


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def require_nonempty_string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} must be a non-empty string.")
    return value.strip()


def member_sort_key(member: dict[str, Any]) -> tuple[Any, ...]:
    membership_type = member.get("membership_type")
    score = member.get("rerank_score")
    score_value = float(score) if isinstance(score, (int, float)) else -1.0
    unit_id = str(member.get("unit_id", ""))
    return (
        MEMBERSHIP_ORDER.get(str(membership_type), 99),
        -score_value,
        llm_io.natural_sort_key(unit_id),
    )


def semantic_unit(member: dict[str, Any], location: str) -> dict[str, Any]:
    unit_id = require_nonempty_string(member.get("unit_id"), f"{location}.unit_id")
    situation = require_nonempty_string(
        member.get("situation"), f"{location}.situation"
    )
    rule = require_nonempty_string(
        member.get("violated_commonsense_rule"),
        f"{location}.violated_commonsense_rule",
    )
    conditions = member.get("applicability_conditions")
    if not isinstance(conditions, list) or any(
        not isinstance(item, str) or not item.strip() for item in conditions
    ):
        raise ValueError(
            f"{location}.applicability_conditions must be an array of strings."
        )
    return {
        "id": unit_id,
        "situation": situation,
        "violated_commonsense_rule": rule,
        "applicability_conditions": [item.strip() for item in conditions],
    }


def preliminary_judgment(member: dict[str, Any], location: str) -> dict[str, Any]:
    membership_type = require_nonempty_string(
        member.get("membership_type"), f"{location}.membership_type"
    )
    if membership_type not in MEMBERSHIP_ORDER:
        raise ValueError(f"Unsupported membership type at {location}: {membership_type}")
    return {
        "unit_id": require_nonempty_string(
            member.get("unit_id"), f"{location}.unit_id"
        ),
        "membership_type": membership_type,
        "judgment_rationale": str(member.get("judgment_rationale", "")).strip(),
        "rerank_rank": member.get("rerank_rank"),
        "rerank_score": member.get("rerank_score"),
    }


def build_plan(
    groups: list[dict[str, Any]], max_pending_units: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if max_pending_units <= 0:
        raise ValueError("max_pending_units must be positive.")

    family_plans: list[dict[str, Any]] = []
    seen_family_ids: set[str] = set()
    seen_unit_ids: set[str] = set()
    incoming_size_distribution: Counter[int] = Counter()

    for group_index, group in enumerate(groups):
        location = f"groups[{group_index}]"
        if not isinstance(group, dict):
            raise ValueError(f"{location} must be an object.")
        family_id = require_nonempty_string(
            group.get("family_id"), f"{location}.family_id"
        )
        if family_id in seen_family_ids:
            raise ValueError(f"Duplicate target family ID: {family_id}")
        seen_family_ids.add(family_id)

        initial_family = group.get("existing_family")
        if not isinstance(initial_family, dict):
            raise ValueError(f"{location}.existing_family must be an object.")
        if initial_family.get("family_id") != family_id:
            raise ValueError(
                f"{location}.existing_family.family_id does not match {family_id}."
            )

        members = group.get("incoming_members")
        if not isinstance(members, list) or not members:
            raise ValueError(f"{location}.incoming_members must not be empty.")
        if group.get("incoming_count") != len(members):
            raise ValueError(f"{location}.incoming_count is inconsistent.")

        ordered_members = sorted(members, key=member_sort_key)
        batches: list[dict[str, Any]] = []
        batch_count = math.ceil(len(ordered_members) / max_pending_units)
        for batch_offset in range(0, len(ordered_members), max_pending_units):
            batch_index = batch_offset // max_pending_units + 1
            batch_id = f"{family_id}.B{batch_index:02d}"
            previous_batch_id = (
                None if batch_index == 1 else f"{family_id}.B{batch_index - 1:02d}"
            )
            chunk = ordered_members[
                batch_offset : batch_offset + max_pending_units
            ]

            pending_units: list[dict[str, Any]] = []
            judgments: list[dict[str, Any]] = []
            for member_index, member in enumerate(chunk):
                member_location = (
                    f"{location}.incoming_members[{batch_offset + member_index}]"
                )
                unit = semantic_unit(member, member_location)
                if unit["id"] in seen_unit_ids:
                    raise ValueError(
                        f"Incoming unit {unit['id']} appears in multiple families."
                    )
                seen_unit_ids.add(unit["id"])
                pending_units.append(unit)
                judgments.append(preliminary_judgment(member, member_location))

            batches.append(
                {
                    "batch_id": batch_id,
                    "batch_index": batch_index,
                    "batch_count": batch_count,
                    "depends_on_batch_id": previous_batch_id,
                    "family_state_source": {
                        "type": "INITIAL_FAMILY"
                        if previous_batch_id is None
                        else "PREVIOUS_BATCH_OUTPUT",
                        "reference": family_id
                        if previous_batch_id is None
                        else previous_batch_id,
                    },
                    "pending_unit_count": len(pending_units),
                    "pending_units": pending_units,
                    "preliminary_judgments": judgments,
                }
            )

        incoming_size_distribution[len(ordered_members)] += 1
        family_plans.append(
            {
                "plan_id": family_id,
                "family_id": family_id,
                "initial_family": initial_family,
                "pending_unit_count": len(ordered_members),
                "batch_count": len(batches),
                "requires_serial_execution": len(batches) > 1,
                "batches": batches,
            }
        )

    family_plans.sort(
        key=lambda plan: (
            -plan["pending_unit_count"],
            llm_io.natural_sort_key(plan["family_id"]),
        )
    )
    batch_count = sum(plan["batch_count"] for plan in family_plans)
    serial_family_count = sum(
        plan["requires_serial_execution"] for plan in family_plans
    )
    pending_unit_count = sum(
        plan["pending_unit_count"] for plan in family_plans
    )
    summary = {
        "target_family_count": len(family_plans),
        "pending_unit_count": pending_unit_count,
        "max_pending_units_per_batch": max_pending_units,
        "generalization_batch_count": batch_count,
        "single_batch_family_count": len(family_plans) - serial_family_count,
        "serial_family_count": serial_family_count,
        "maximum_batches_per_family": max(
            (plan["batch_count"] for plan in family_plans), default=0
        ),
        "calls_avoided_vs_one_unit_per_call": pending_unit_count - batch_count,
        "call_reduction_ratio": round(
            1 - batch_count / pending_unit_count, 6
        )
        if pending_unit_count
        else 0.0,
        "incoming_size_distribution": {
            str(size): count
            for size, count in sorted(incoming_size_distribution.items())
        },
    }
    return family_plans, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build family-level generalization batches with explicit serial dependencies."
        )
    )
    parser.add_argument("--groups-file", type=Path, default=DEFAULT_GROUPS_FILE)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument(
        "--max-pending-units",
        type=int,
        default=DEFAULT_MAX_PENDING_UNITS,
        help="Maximum new units submitted with a family in one LLM call.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    groups_path = resolve_path(args.groups_file)
    output_path = resolve_path(args.output_file)

    root = llm_io.read_json(groups_path)
    groups = root.get("groups") if isinstance(root, dict) else None
    if not isinstance(groups, list):
        raise ValueError(f"{groups_path} must contain a groups array.")

    family_plans, summary = build_plan(groups, args.max_pending_units)
    output = {
        "schema_version": "1.0",
        "groups_file": str(groups_path),
        "execution_model": "PARALLEL_ACROSS_FAMILIES_SERIAL_WITHIN_FAMILY",
        "ordering_policy": (
            "EQUIVALENT_MEMBER first, then VARIANT_MEMBER; "
            "within each type, rerank_score descending"
        ),
        "summary": summary,
        "family_plans": family_plans,
    }
    llm_io.write_json(output_path, output)

    print(
        "Done. "
        f"families={summary['target_family_count']}, "
        f"pending_units={summary['pending_unit_count']}, "
        f"batches={summary['generalization_batch_count']}, "
        f"serial_families={summary['serial_family_count']}"
    )
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()

