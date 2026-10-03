"""Assemble the final commonsense generalization record catalog without additional LLM calls."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable


from commonsense_repro.common.paths import PACKAGE_ROOT as PROJECT_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from commonsense_repro.common import llm_pipeline_common as llm_io
_global_generalization = importlib.import_module(
    "commonsense_repro.commonsense_generalization.global.generalization.generalize_global_commonsense"
)
family_member_ids = _global_generalization.family_member_ids
from commonsense_repro.commonsense_generalization.local.generalize_local_commonsense import load_source_units


DATASET_ROOT = Path("work/stage4")
DEFAULT_CATALOG_FILE = Path(
    "work/stage3/"
    "gemini_v1_2_non_mutual_s15_leiden_max25_deepseek_flash_v1_4/"
    "commonsense_generalization_records.json"
)
DEFAULT_UNMATCHED_FILE = DATASET_ROOT / (
    "candidate_graph/top3_mutual_s070/unmatched_generalization_records.json"
)
DEFAULT_STAGE1_RESULTS_FILE = DATASET_ROOT / (
    "candidate_grouping/deepseek_flash_v1_0/grouping_results.jsonl"
)
DEFAULT_STAGE2_RESULTS_FILE = DATASET_ROOT / (
    "generalization/deepseek_flash_v1_2/final_generalization_results.jsonl"
)
DEFAULT_SOURCE_FILES = (
    Path("data/checkpoints/instance_level_commonsense/gemini_v1_2_extract.json"),
    Path("data/checkpoints/instance_level_commonsense/gemini_v1_2_github.json"),
    Path("data/checkpoints/instance_level_commonsense/gemini_v1_2_negative.json"),
)
DEFAULT_OUTPUT_DIR = DATASET_ROOT / "catalog_assembly"

CATALOG_FILENAME = "final_commonsense_generalization_records.json"
SINGLETONS_FILENAME = "stage2_singleton_units.json"
SUMMARY_FILENAME = "catalog_assembly_summary.json"


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} must be a non-empty string.")
    return value.strip()


def require_object(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be an object.")
    return value


def sorted_ids(values: Iterable[str]) -> list[str]:
    return sorted(values, key=llm_io.natural_sort_key)


def index_catalog(
    catalog_root: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    families = catalog_root.get("rule_families")
    if not isinstance(families, list):
        raise ValueError("The source catalog must contain rule_families.")
    family_by_id: dict[str, dict[str, Any]] = {}
    family_by_member: dict[str, str] = {}
    for index, value in enumerate(families):
        family = require_object(value, f"rule_families[{index}]")
        family_id = require_string(family.get("family_id"), f"rule_families[{index}].family_id")
        if family_id in family_by_id:
            raise ValueError(f"Duplicate source family ID: {family_id}")
        members = family.get("member_ids")
        if not isinstance(members, list) or len(members) < 2:
            raise ValueError(f"Source family {family_id} must contain at least two members.")
        if family.get("member_count") != len(members):
            raise ValueError(f"Source family {family_id} has an inconsistent member_count.")
        for member_id in members:
            source_id = require_string(member_id, f"{family_id}.member_ids")
            if source_id in family_by_member:
                raise ValueError(
                    f"Source unit {source_id} occurs in both {family_by_member[source_id]} "
                    f"and {family_id}."
                )
            family_by_member[source_id] = family_id
        family_by_id[family_id] = family
    declared_count = catalog_root.get("family_count")
    if declared_count is not None and declared_count != len(family_by_id):
        raise ValueError("The source catalog family_count is inconsistent.")
    return family_by_id, family_by_member


def load_unmatched_ids(root: dict[str, Any]) -> set[str]:
    families = root.get("families")
    if not isinstance(families, list):
        raise ValueError("The graph unmatched file must contain families.")
    ids = {
        require_string(
            require_object(value, "unmatched family").get("family_id"),
            "unmatched family.family_id",
        )
        for value in families
    }
    if root.get("unmatched_count") != len(ids):
        raise ValueError("The graph unmatched_count is inconsistent.")
    return ids


def partition_stage1(
    rows: list[dict[str, Any]],
) -> tuple[set[str], set[str], dict[str, str]]:
    grouped: set[str] = set()
    singletons: set[str] = set()
    group_by_family: dict[str, str] = {}
    for row in rows:
        batch_id = require_string(row.get("batch_id"), "stage1.batch_id")
        groups = row.get("groups")
        singleton_ids = row.get("singleton_family_ids")
        if not isinstance(groups, list) or not isinstance(singleton_ids, list):
            raise ValueError(f"Stage-1 batch {batch_id} is incomplete.")
        for group in groups:
            group = require_object(group, f"{batch_id}.groups")
            group_id = require_string(group.get("group_id"), f"{batch_id}.group_id")
            member_ids = group.get("member_family_ids")
            if not isinstance(member_ids, list) or len(member_ids) < 2:
                raise ValueError(f"Stage-1 group {group_id} must contain at least two families.")
            for family_id in member_ids:
                family_id = require_string(family_id, f"{group_id}.member_family_ids")
                if family_id in grouped or family_id in singletons:
                    raise ValueError(f"Stage-1 family {family_id} is assigned more than once.")
                grouped.add(family_id)
                group_by_family[family_id] = group_id
        for family_id in singleton_ids:
            family_id = require_string(family_id, f"{batch_id}.singleton_family_ids")
            if family_id in grouped or family_id in singletons:
                raise ValueError(f"Stage-1 family {family_id} is assigned more than once.")
            singletons.add(family_id)
    return grouped, singletons, group_by_family


def copy_retained_family(
    family: dict[str, Any],
    *,
    family_origin: str,
) -> dict[str, Any]:
    copied = dict(family)
    copied["family_origin"] = family_origin
    copied["provenance"] = {
        "source_catalog_family_id": family["family_id"],
        "retained_without_stage2_regeneralization": True,
    }
    return copied


def normalize_stage2_family(
    family: dict[str, Any],
    *,
    result: dict[str, Any],
) -> dict[str, Any]:
    members = family_member_ids(family)
    if len(members) < 2:
        raise ValueError(f"Final family {family.get('family_id')} has fewer than two members.")
    return {
        "family_id": require_string(family.get("family_id"), "stage2.family_id"),
        "family_origin": "STAGE2_INDUCED",
        "base_origin": require_string(family.get("base_origin"), "stage2.base_origin"),
        "base_unit": require_object(family.get("base_unit"), "stage2.base_unit"),
        "member_count": len(members),
        "member_ids": members,
        "equivalent_member_ids": list(family["equivalent_member_ids"]),
        "variant_groups": list(family["variant_groups"]),
        "family_rationale": require_string(
            family.get("family_rationale"), "stage2.family_rationale"
        ),
        "provenance": {
            "stage2_batch_id": result["batch_id"],
            "stage1_component_id": result["stage1_component_id"],
            "stage1_group_signature": result["stage1_group_signature"],
            "candidate_family_ids": list(result["candidate_family_ids"]),
            "model": result["model"],
            "prompt_sha256": result["prompt_sha256"],
            "input_sha256": result["input_sha256"],
        },
    }


def assemble_catalog_assembly(
    *,
    catalog_root: dict[str, Any],
    graph_unmatched_ids: set[str],
    stage1_rows: list[dict[str, Any]],
    stage2_rows: list[dict[str, Any]],
    source_units: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    family_by_id, family_by_member = index_catalog(catalog_root)
    catalog_ids = set(family_by_id)
    stage1_grouped, stage1_singletons, _ = partition_stage1(stage1_rows)

    if graph_unmatched_ids & (stage1_grouped | stage1_singletons):
        raise ValueError("Graph-unmatched families overlap stage-1 assignments.")
    if graph_unmatched_ids | stage1_grouped | stage1_singletons != catalog_ids:
        missing = catalog_ids - (graph_unmatched_ids | stage1_grouped | stage1_singletons)
        extra = (graph_unmatched_ids | stage1_grouped | stage1_singletons) - catalog_ids
        raise ValueError(f"The 552-family partition is incomplete: missing={missing}, extra={extra}")

    stage2_candidate_ids: list[str] = []
    stage2_families: list[dict[str, Any]] = []
    stage2_singleton_context: dict[str, dict[str, Any]] = {}
    stage2_assigned_ids: set[str] = set()
    for result in stage2_rows:
        if result.get("status") != "success":
            raise ValueError(f"Stage-2 batch {result.get('batch_id')} did not succeed.")
        candidate_ids = result.get("candidate_family_ids")
        if not isinstance(candidate_ids, list):
            raise ValueError(f"Stage-2 batch {result.get('batch_id')} lacks candidates.")
        stage2_candidate_ids.extend(candidate_ids)
        generated_ids: list[str] = []
        for family in result.get("rule_families", []):
            normalized = normalize_stage2_family(family, result=result)
            generated_ids.append(normalized["family_id"])
            for source_id in normalized["member_ids"]:
                if source_id in stage2_assigned_ids or source_id in stage2_singleton_context:
                    raise ValueError(f"Stage-2 source unit {source_id} is assigned more than once.")
                stage2_assigned_ids.add(source_id)
            stage2_families.append(normalized)
        for singleton in result.get("singleton_units", []):
            singleton = require_object(singleton, "stage2.singleton_units")
            source_id = require_string(singleton.get("id"), "stage2.singleton_units.id")
            if source_id in stage2_assigned_ids or source_id in stage2_singleton_context:
                raise ValueError(f"Stage-2 source unit {source_id} is assigned more than once.")
            stage2_singleton_context[source_id] = {
                "result": result,
                "co_generated_final_family_ids": generated_ids,
            }

    if len(stage2_candidate_ids) != len(set(stage2_candidate_ids)):
        raise ValueError("A source catalog family occurs in multiple stage-2 groups.")
    if set(stage2_candidate_ids) != stage1_grouped:
        raise ValueError("Stage-2 candidate families do not match stage-1 grouped families.")

    expected_stage2_units = {
        source_id
        for family_id in stage1_grouped
        for source_id in family_by_id[family_id]["member_ids"]
    }
    observed_stage2_units = stage2_assigned_ids | set(stage2_singleton_context)
    if observed_stage2_units != expected_stage2_units:
        raise ValueError(
            "Stage-2 output does not partition its source units: "
            f"missing={expected_stage2_units - observed_stage2_units}, "
            f"extra={observed_stage2_units - expected_stage2_units}"
        )

    retained_families = [
        copy_retained_family(
            family_by_id[family_id], family_origin="GRAPH_UNMATCHED_RETAINED"
        )
        for family_id in sorted_ids(graph_unmatched_ids)
    ]
    retained_families.extend(
        copy_retained_family(
            family_by_id[family_id], family_origin="STAGE1_SINGLETON_RETAINED"
        )
        for family_id in sorted_ids(stage1_singletons)
    )
    final_families = retained_families + stage2_families
    final_families.sort(key=lambda family: llm_io.natural_sort_key(family["family_id"]))

    final_family_ids = [family["family_id"] for family in final_families]
    if len(final_family_ids) != len(set(final_family_ids)):
        raise ValueError("Final family IDs are not unique.")
    final_member_ids: list[str] = []
    for family in final_families:
        final_member_ids.extend(family["member_ids"])
    if len(final_member_ids) != len(set(final_member_ids)):
        raise ValueError("A source unit occurs in multiple final families.")

    singleton_records: list[dict[str, Any]] = []
    for source_id in sorted_ids(stage2_singleton_context):
        if source_id not in source_units:
            raise ValueError(f"Missing source text for stage-2 singleton {source_id}.")
        prior_family_id = family_by_member[source_id]
        prior_member_ids = set(family_by_id[prior_family_id]["member_ids"])
        assigned_siblings = sorted_ids((prior_member_ids - {source_id}) & stage2_assigned_ids)
        context = stage2_singleton_context[source_id]
        result = context["result"]
        singleton_records.append(
            {
                "source_unit_id": source_id,
                "status": "STAGE2_UNASSIGNED",
                "structural_reason_code": (
                    "ORPHANED_AFTER_SPLIT"
                    if assigned_siblings
                    else "WHOLE_PRIOR_FAMILY_UNASSIGNED"
                ),
                "semantic_reason": None,
                "source_unit": {
                    "situation": source_units[source_id]["situation"],
                    "commonsense_rule": source_units[source_id][
                        "violated_commonsense_rule"
                    ],
                    "applicability_conditions": source_units[source_id][
                        "applicability_conditions"
                    ],
                },
                "provenance": {
                    "prior_family_id": prior_family_id,
                    "prior_family_member_ids": sorted_ids(prior_member_ids),
                    "assigned_sibling_ids": assigned_siblings,
                    "stage2_batch_id": result["batch_id"],
                    "candidate_family_ids": list(result["candidate_family_ids"]),
                    "co_generated_final_family_ids": context[
                        "co_generated_final_family_ids"
                    ],
                    "model": result["model"],
                    "prompt_sha256": result["prompt_sha256"],
                    "input_sha256": result["input_sha256"],
                },
                "future_integration": {
                    "candidate_type": "REVIEWED_UNASSIGNED",
                    "eligible_for_rematching": True,
                },
            }
        )

    represented_unit_count = len(final_member_ids)
    singleton_count = len(singleton_records)
    source_scope_count = len(family_by_member)
    if represented_unit_count + singleton_count != source_scope_count:
        raise ValueError(
            "Final families and singleton units do not cover the source catalog scope."
        )
    counts = {
        "source_catalog_family_count": len(catalog_ids),
        "source_scope_unit_count": source_scope_count,
        "graph_unmatched_retained_family_count": len(graph_unmatched_ids),
        "stage1_singleton_retained_family_count": len(stage1_singletons),
        "stage2_input_family_count": len(stage1_grouped),
        "stage2_input_unit_count": len(expected_stage2_units),
        "stage2_generalized_family_count": len(stage2_families),
        "final_rule_family_count": len(final_families),
        "family_covered_unit_count": represented_unit_count,
        "stage2_singleton_unit_count": singleton_count,
    }
    return final_families, singleton_records, counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-file", type=Path, default=DEFAULT_CATALOG_FILE)
    parser.add_argument("--unmatched-file", type=Path, default=DEFAULT_UNMATCHED_FILE)
    parser.add_argument(
        "--stage1-results-file", type=Path, default=DEFAULT_STAGE1_RESULTS_FILE
    )
    parser.add_argument(
        "--stage2-results-file", type=Path, default=DEFAULT_STAGE2_RESULTS_FILE
    )
    parser.add_argument(
        "--source-file", type=Path, action="append", dest="source_files"
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    catalog_path = resolve_path(args.catalog_file)
    unmatched_path = resolve_path(args.unmatched_file)
    stage1_path = resolve_path(args.stage1_results_file)
    stage2_path = resolve_path(args.stage2_results_file)
    source_paths = [
        resolve_path(path) for path in (args.source_files or DEFAULT_SOURCE_FILES)
    ]
    output_dir = resolve_path(args.output_dir)

    catalog_root = require_object(llm_io.read_json(catalog_path), "catalog")
    unmatched_root = require_object(llm_io.read_json(unmatched_path), "unmatched")
    stage1_rows = llm_io.read_jsonl(stage1_path)
    stage2_rows = llm_io.read_jsonl(stage2_path)
    source_units = load_source_units(source_paths)
    families, singletons, counts = assemble_catalog_assembly(
        catalog_root=catalog_root,
        graph_unmatched_ids=load_unmatched_ids(unmatched_root),
        stage1_rows=stage1_rows,
        stage2_rows=stage2_rows,
        source_units=source_units,
    )

    source_files = {
        "source_catalog": {"path": str(catalog_path), "sha256": sha256_file(catalog_path)},
        "graph_unmatched": {
            "path": str(unmatched_path),
            "sha256": sha256_file(unmatched_path),
        },
        "stage1_results": {"path": str(stage1_path), "sha256": sha256_file(stage1_path)},
        "stage2_results": {"path": str(stage2_path), "sha256": sha256_file(stage2_path)},
        "source_units": [
            {"path": str(path), "sha256": sha256_file(path)} for path in source_paths
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    catalog_output = output_dir / CATALOG_FILENAME
    singletons_output = output_dir / SINGLETONS_FILENAME
    summary_output = output_dir / SUMMARY_FILENAME
    llm_io.write_json(
        catalog_output,
        {
            "schema_version": "1.0",
            **counts,
            "source_files": source_files,
            "rule_families": families,
        },
    )
    llm_io.write_json(
        singletons_output,
        {
            "schema_version": "1.0",
            "singleton_definition": (
                "A source instance-level commonsense item not assigned to any final commonsense generalization record."
            ),
            "singleton_unit_count": len(singletons),
            "source_files": source_files,
            "singleton_units": singletons,
        },
    )
    llm_io.write_json(
        summary_output,
        {
            **counts,
            "coverage_check": (
                f"{counts['family_covered_unit_count']} family-covered units + "
                f"{counts['stage2_singleton_unit_count']} singleton units = "
                f"{counts['source_scope_unit_count']} source units"
            ),
            "outputs": {
                "final_rule_family_catalog": str(catalog_output),
                "stage2_singleton_units": str(singletons_output),
            },
        },
    )
    print(
        "Done. "
        f"families={counts['final_rule_family_count']}, "
        f"covered_units={counts['family_covered_unit_count']}, "
        f"singleton_units={counts['stage2_singleton_unit_count']}, "
        f"scope_units={counts['source_scope_unit_count']}, "
        f"output_dir={output_dir}"
    )


if __name__ == "__main__":
    main()

