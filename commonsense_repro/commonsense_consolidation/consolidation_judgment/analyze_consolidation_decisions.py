"""Group accepted membership judgments by their selected commonsense generalization record."""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


from commonsense_repro.common.paths import PACKAGE_ROOT as PROJECT_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from commonsense_repro.common import llm_pipeline_common as llm_io
from commonsense_repro.commonsense_generalization.local.generalize_local_commonsense import load_source_units


DATASET_ROOT = Path(
    "work/stage4/"
    "commonsense_consolidation"
)
DEFAULT_RESULTS_FILE = (
    DATASET_ROOT
    / "consolidation_judgment/deepseek_flash_v1_1/membership_results.jsonl"
)
DEFAULT_CANDIDATES_FILE = (
    DATASET_ROOT / "rerank/reranked_consolidation_candidates_k10.jsonl"
)
DEFAULT_CATALOG_FILE = Path(
    "work/stage4/"
    "catalog_assembly/final_commonsense_generalization_records.json"
)
DEFAULT_SOURCE_FILES = (
    Path("data/checkpoints/instance_level_commonsense/gemini_v1_2_extract.json"),
    Path("data/checkpoints/instance_level_commonsense/gemini_v1_2_github.json"),
    Path("data/checkpoints/instance_level_commonsense/gemini_v1_2_negative.json"),
)
DEFAULT_OUTPUT_DIR = DATASET_ROOT / "consolidation_judgment/deepseek_flash_v1_1"
DEFAULT_OUTPUT_JSON = DEFAULT_OUTPUT_DIR / "membership_groups.json"
DEFAULT_OUTPUT_MARKDOWN = DEFAULT_OUTPUT_DIR / "membership_groups.md"
DEFAULT_MAX_NEW_UNITS_PER_CALL = 25


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def require_nonempty_string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} must be a non-empty string.")
    return value.strip()


def load_rerank_candidates(path: Path) -> dict[str, list[dict[str, Any]]]:
    records = llm_io.read_jsonl(path)
    candidates_by_unit: dict[str, list[dict[str, Any]]] = {}
    for index, record in enumerate(records):
        unit_id = require_nonempty_string(
            record.get("unit_id"), f"{path}:{index + 1}.unit_id"
        )
        neighbors = record.get("neighbors")
        if not isinstance(neighbors, list):
            raise ValueError(f"{path}:{index + 1}.neighbors must be an array.")
        if unit_id in candidates_by_unit:
            raise ValueError(f"Duplicate rerank unit ID: {unit_id}")
        candidates_by_unit[unit_id] = neighbors
    return candidates_by_unit


def load_full_catalog(path: Path) -> dict[str, dict[str, Any]]:
    root = llm_io.read_json(path)
    families = root.get("rule_families") if isinstance(root, dict) else None
    if not isinstance(families, list) or not families:
        raise ValueError(f"{path} must contain a non-empty rule_families array.")

    catalog: dict[str, dict[str, Any]] = {}
    for index, family in enumerate(families):
        location = f"{path}.rule_families[{index}]"
        if not isinstance(family, dict):
            raise ValueError(f"{location} must be an object.")
        family_id = require_nonempty_string(
            family.get("family_id"), f"{location}.family_id"
        )
        if family_id in catalog:
            raise ValueError(f"Duplicate family ID: {family_id}")
        required_fields = {
            "base_unit",
            "equivalent_member_ids",
            "variant_groups",
            "family_rationale",
        }
        missing = required_fields - set(family)
        if missing:
            raise ValueError(
                f"{location} is missing fields: {sorted(missing)}"
            )
        catalog[family_id] = family
    return catalog


def selected_candidate(
    unit_id: str,
    family_id: str,
    candidates_by_unit: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    candidates = candidates_by_unit.get(unit_id)
    if candidates is None:
        raise ValueError(f"No rerank candidates found for unit {unit_id}.")
    matches = [item for item in candidates if item.get("family_id") == family_id]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one rerank candidate for {unit_id} -> {family_id}; "
            f"found {len(matches)}."
        )
    return matches[0]


def build_groups(
    results: list[dict[str, Any]],
    candidates_by_unit: dict[str, list[dict[str, Any]]],
    catalog: dict[str, dict[str, Any]],
    source_units: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_unit_ids: set[str] = set()
    decision_counts: Counter[str] = Counter()

    for index, result in enumerate(results):
        unit_id = require_nonempty_string(
            result.get("unit_id"), f"results[{index}].unit_id"
        )
        if unit_id in seen_unit_ids:
            raise ValueError(f"Duplicate membership result for unit {unit_id}.")
        seen_unit_ids.add(unit_id)

        decision = require_nonempty_string(
            result.get("decision"), f"results[{index}].decision"
        )
        decision_counts[decision] += 1
        if decision != "ADD_TO_FAMILY":
            continue

        family_id = require_nonempty_string(
            result.get("selected_family_id"),
            f"results[{index}].selected_family_id",
        )
        membership_type = require_nonempty_string(
            result.get("membership_type"), f"results[{index}].membership_type"
        )
        if membership_type not in {"EQUIVALENT_MEMBER", "VARIANT_MEMBER"}:
            raise ValueError(
                f"Unsupported membership type for {unit_id}: {membership_type}"
            )
        if family_id not in catalog:
            raise ValueError(f"Selected family {family_id} is absent from the catalog.")
        source = source_units.get(unit_id)
        if source is None:
            raise ValueError(f"Source content is missing for unit {unit_id}.")

        candidate = selected_candidate(unit_id, family_id, candidates_by_unit)
        grouped[family_id].append(
            {
                "unit_id": unit_id,
                "membership_type": membership_type,
                "rerank_rank": candidate.get("rank"),
                "rerank_score": candidate.get("rerank_score"),
                "matched_representation": candidate.get("matched_representation"),
                "situation": source["situation"],
                "violated_commonsense_rule": source[
                    "violated_commonsense_rule"
                ],
                "applicability_conditions": source[
                    "applicability_conditions"
                ],
                "judgment_rationale": result.get("rationale", ""),
            }
        )

    groups: list[dict[str, Any]] = []
    for family_id, members in grouped.items():
        members.sort(key=lambda item: llm_io.natural_sort_key(item["unit_id"]))
        family = catalog[family_id]
        groups.append(
            {
                "family_id": family_id,
                "existing_family": family,
                "incoming_count": len(members),
                "equivalent_member_count": sum(
                    item["membership_type"] == "EQUIVALENT_MEMBER"
                    for item in members
                ),
                "variant_member_count": sum(
                    item["membership_type"] == "VARIANT_MEMBER"
                    for item in members
                ),
                "incoming_member_ids": [item["unit_id"] for item in members],
                "incoming_members": members,
            }
        )

    groups.sort(
        key=lambda item: (
            -item["incoming_count"],
            llm_io.natural_sort_key(item["family_id"]),
        )
    )
    return groups, dict(decision_counts)


def build_summary(
    groups: list[dict[str, Any]],
    decision_counts: dict[str, int],
    result_count: int,
    max_new_units_per_call: int,
) -> dict[str, Any]:
    incoming_count = sum(group["incoming_count"] for group in groups)
    single_groups = sum(group["incoming_count"] == 1 for group in groups)
    multi_groups = len(groups) - single_groups
    units_in_multi_groups = sum(
        group["incoming_count"]
        for group in groups
        if group["incoming_count"] >= 2
    )
    size_distribution = Counter(group["incoming_count"] for group in groups)
    grouped_calls = len(groups)
    chunked_calls = sum(
        math.ceil(group["incoming_count"] / max_new_units_per_call)
        for group in groups
    )

    return {
        "membership_result_count": result_count,
        "decision_counts": decision_counts,
        "add_to_family_count": incoming_count,
        "target_family_count": len(groups),
        "families_receiving_one_new_unit": single_groups,
        "families_receiving_multiple_new_units": multi_groups,
        "new_units_in_single_incoming_groups": single_groups,
        "new_units_in_multi_incoming_groups": units_in_multi_groups,
        "equivalent_member_count": sum(
            group["equivalent_member_count"] for group in groups
        ),
        "variant_member_count": sum(
            group["variant_member_count"] for group in groups
        ),
        "incoming_size_distribution": {
            str(size): count for size, count in sorted(size_distribution.items())
        },
        "merge_call_estimates": {
            "one_call_per_new_unit": incoming_count,
            "one_call_per_target_family": grouped_calls,
            "calls_avoided_by_family_grouping": incoming_count - grouped_calls,
            "family_grouping_reduction_ratio": round(
                1 - grouped_calls / incoming_count, 6
            )
            if incoming_count
            else 0.0,
            "max_new_units_per_call": max_new_units_per_call,
            "calls_with_size_limit": chunked_calls,
            "calls_avoided_with_size_limit": incoming_count - chunked_calls,
            "size_limited_reduction_ratio": round(
                1 - chunked_calls / incoming_count, 6
            )
            if incoming_count
            else 0.0,
        },
    }


def markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def render_markdown(summary: dict[str, Any], groups: list[dict[str, Any]]) -> str:
    calls = summary["merge_call_estimates"]
    lines = [
        "# Consolidation Decisions by Target Generalization Record",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Membership results | {summary['membership_result_count']} |",
        f"| Accepted new units | {summary['add_to_family_count']} |",
        f"| Target commonsense generalization records | {summary['target_family_count']} |",
        "| Families receiving multiple units | "
        f"{summary['families_receiving_multiple_new_units']} |",
        "| Units sharing a target family with another new unit | "
        f"{summary['new_units_in_multi_incoming_groups']} |",
        f"| Equivalent members | {summary['equivalent_member_count']} |",
        f"| Variant members | {summary['variant_member_count']} |",
        f"| Calls when processing one unit at a time | {calls['one_call_per_new_unit']} |",
        f"| Calls with one call per target family | {calls['one_call_per_target_family']} |",
        f"| Calls avoided by family grouping | {calls['calls_avoided_by_family_grouping']} |",
        "| Call reduction by family grouping | "
        f"{calls['family_grouping_reduction_ratio']:.2%} |",
        f"| Calls with at most {calls['max_new_units_per_call']} new units | "
        f"{calls['calls_with_size_limit']} |",
        "",
        "## Incoming Size Distribution",
        "",
        "| New units assigned to one family | Family count |",
        "|---:|---:|",
    ]
    lines.extend(
        f"| {size} | {count} |"
        for size, count in summary["incoming_size_distribution"].items()
    )

    lines.extend(["", "## Families Receiving Multiple Units", ""])
    for group in groups:
        if group["incoming_count"] < 2:
            continue
        family = group["existing_family"]
        base = family.get("base_unit", {})
        lines.extend(
            [
                f"### `{group['family_id']}`: {group['incoming_count']} new units",
                "",
                f"**Existing members:** {family.get('member_count', 'unknown')}",
                "",
                f"**Base situation:** {markdown_cell(base.get('situation', ''))}",
                "",
                f"**Base rule:** {markdown_cell(base.get('commonsense_rule', ''))}",
                "",
                "| Unit ID | Judgment | Rerank rank | Rerank score |",
                "|---|---|---:|---:|",
            ]
        )
        for member in group["incoming_members"]:
            score = member["rerank_score"]
            score_text = f"{score:.6f}" if isinstance(score, (int, float)) else ""
            lines.append(
                f"| `{member['unit_id']}` | {member['membership_type']} | "
                f"{member['rerank_rank']} | {score_text} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Families Receiving One Unit",
            "",
            "| Family ID | Unit ID | Judgment | Rerank score |",
            "|---|---|---|---:|",
        ]
    )
    for group in groups:
        if group["incoming_count"] != 1:
            continue
        member = group["incoming_members"][0]
        score = member["rerank_score"]
        score_text = f"{score:.6f}" if isinstance(score, (int, float)) else ""
        lines.append(
            f"| `{group['family_id']}` | `{member['unit_id']}` | "
            f"{member['membership_type']} | {score_text} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Group ADD_TO_FAMILY judgments by selected commonsense generalization record."
    )
    parser.add_argument("--results-file", type=Path, default=DEFAULT_RESULTS_FILE)
    parser.add_argument(
        "--candidates-file", type=Path, default=DEFAULT_CANDIDATES_FILE
    )
    parser.add_argument("--catalog-file", type=Path, default=DEFAULT_CATALOG_FILE)
    parser.add_argument(
        "--source-file",
        action="append",
        type=Path,
        dest="source_files",
        help="Source unit JSON file; repeat for multiple files.",
    )
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument(
        "--output-markdown", type=Path, default=DEFAULT_OUTPUT_MARKDOWN
    )
    parser.add_argument(
        "--max-new-units-per-call",
        type=int,
        default=DEFAULT_MAX_NEW_UNITS_PER_CALL,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_new_units_per_call <= 0:
        raise ValueError("--max-new-units-per-call must be positive.")

    results_path = resolve_path(args.results_file)
    candidates_path = resolve_path(args.candidates_file)
    catalog_path = resolve_path(args.catalog_file)
    source_paths = [
        resolve_path(path)
        for path in (args.source_files or list(DEFAULT_SOURCE_FILES))
    ]
    output_json_path = resolve_path(args.output_json)
    output_markdown_path = resolve_path(args.output_markdown)

    results = llm_io.read_jsonl(results_path)
    candidates_by_unit = load_rerank_candidates(candidates_path)
    catalog = load_full_catalog(catalog_path)
    source_units = load_source_units(source_paths)
    groups, decision_counts = build_groups(
        results, candidates_by_unit, catalog, source_units
    )
    summary = build_summary(
        groups,
        decision_counts,
        len(results),
        args.max_new_units_per_call,
    )

    output = {
        "schema_version": "1.0",
        "results_file": str(results_path),
        "candidates_file": str(candidates_path),
        "catalog_file": str(catalog_path),
        "source_files": [str(path) for path in source_paths],
        "summary": summary,
        "groups": groups,
    }
    llm_io.write_json(output_json_path, output)
    llm_io.write_text(output_markdown_path, render_markdown(summary, groups))

    calls = summary["merge_call_estimates"]
    print(
        "Done. "
        f"accepted_units={summary['add_to_family_count']}, "
        f"target_families={summary['target_family_count']}, "
        f"multi_unit_families={summary['families_receiving_multiple_new_units']}, "
        f"grouped_calls={calls['one_call_per_target_family']}, "
        f"size_limited_calls={calls['calls_with_size_limit']}"
    )
    print(f"JSON: {output_json_path}")
    print(f"Markdown: {output_markdown_path}")


if __name__ == "__main__":
    main()

