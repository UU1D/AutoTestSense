"""Add complete base and matched instance-level commonsense items to retrieval results."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
from commonsense_repro.common.paths import PACKAGE_ROOT as PROJECT_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from commonsense_repro.retrieval.batch_retrieve_commonsense_library import (
    CONFIG_FIELD,
    RESULT_FIELD,
    SITUATIONS_FIELD,
    read_json,
    require_string,
    resolve_dataset_paths,
    resolve_path,
    write_json_atomic,
)


# -----------------------------------------------------------------------------
# Editable configuration
# -----------------------------------------------------------------------------
DEFAULT_DATASET_DIR = SCRIPT_DIR / "visiondroid_situation"
DEFAULT_INPUT_SUBDIR = "retrieval_output"
DEFAULT_OUTPUT_SUBDIR = "retrieval_output_with_commonsense"
DEFAULT_CATALOG_FILE = Path(
    "work/stage4/"
    "commonsense_consolidation/serial_consolidation/deepseek_flash_v1_0/"
    "current_commonsense_generalization_records.json"
)
DEFAULT_SOURCE_FILES = (
    Path("data/checkpoints/instance_level_commonsense/gemini_v1_2_extract.json"),
    Path("data/checkpoints/instance_level_commonsense/gemini_v1_2_github.json"),
    Path("data/checkpoints/instance_level_commonsense/gemini_v1_2_negative.json"),
)
DEFAULT_TOP_K = 30
DEFAULT_INPUT_PATTERN = "*.json"
METADATA_FILENAMES = {"retrieval_summary.json", "enrichment_summary.json"}


def normalized_unit(situation: str, commonsense_rule: str) -> dict[str, str]:
    return {
        "situation": situation,
        "commonsense_rule": commonsense_rule,
    }


def load_catalog(path: Path) -> dict[str, dict[str, Any]]:
    data = read_json(path)
    families = data.get("rule_families") if isinstance(data, dict) else None
    if not isinstance(families, list):
        raise ValueError(f"{path} must contain a rule_families array.")

    by_id: dict[str, dict[str, Any]] = {}
    for index, family in enumerate(families):
        location = f"{path}.rule_families[{index}]"
        if not isinstance(family, dict):
            raise ValueError(f"{location} must be an object.")
        family_id = require_string(family.get("family_id"), f"{location}.family_id")
        if family_id in by_id:
            raise ValueError(f"Duplicate family_id: {family_id}")
        base = family.get("base_unit")
        if not isinstance(base, dict):
            raise ValueError(f"{location}.base_unit must be an object.")
        situation = require_string(base.get("situation"), f"{location}.base_unit.situation")
        rule = require_string(
            base.get("commonsense_rule"), f"{location}.base_unit.commonsense_rule"
        )
        normalized = copy.deepcopy(family)
        normalized["base_unit"] = normalized_unit(situation, rule)
        by_id[family_id] = normalized
    return by_id


def load_source_units(paths: list[Path]) -> dict[str, dict[str, str]]:
    units: dict[str, dict[str, str]] = {}
    for path in paths:
        data = read_json(path)
        if not isinstance(data, list):
            raise ValueError(f"{path} must contain a JSON array.")
        for index, row in enumerate(data):
            location = f"{path}[{index}]"
            if not isinstance(row, dict):
                raise ValueError(f"{location} must be an object.")
            unit_id = require_string(row.get("id"), f"{location}.id")
            if unit_id in units:
                raise ValueError(f"Duplicate source unit id: {unit_id}")
            units[unit_id] = normalized_unit(
                require_string(row.get("situation"), f"{location}.situation"),
                require_string(
                    row.get("violated_commonsense_rule"),
                    f"{location}.violated_commonsense_rule",
                ),
            )
    return units


def build_matched_unit(
    *,
    family: dict[str, Any],
    matched: dict[str, Any],
    source_units: dict[str, dict[str, str]],
    location: str,
) -> dict[str, Any]:
    representation_type = matched.get("type")
    if representation_type == "BASE":
        return {
            "type": "BASE",
            "unit_id": None,
            **copy.deepcopy(family["base_unit"]),
        }
    if representation_type != "VARIANT_MEMBER":
        raise ValueError(f"{location}.type is invalid: {representation_type!r}")

    member_id = require_string(matched.get("member_id"), f"{location}.member_id")
    if member_id not in source_units:
        raise ValueError(f"No source instance-level commonsense item found for {member_id}.")
    unit = source_units[member_id]
    matched_situation = require_string(
        matched.get("situation"), f"{location}.situation"
    )
    if matched_situation != unit["situation"]:
        raise ValueError(
            f"Matched situation differs from source unit {member_id}: "
            f"{matched_situation!r} != {unit['situation']!r}"
        )
    return {
        "type": "VARIANT_MEMBER",
        "unit_id": member_id,
        **copy.deepcopy(unit),
    }


def enrich_document(
    path: Path,
    data: Any,
    *,
    catalog: dict[str, dict[str, Any]],
    source_units: dict[str, dict[str, str]],
    top_k: int,
) -> tuple[dict[str, Any], int]:
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    situations = data.get(SITUATIONS_FIELD)
    if not isinstance(situations, list):
        raise ValueError(f"{path} has no {SITUATIONS_FIELD!r} list.")

    output = copy.deepcopy(data)
    enriched_count = 0
    for situation_index, situation in enumerate(output[SITUATIONS_FIELD]):
        location = f"{path}.{SITUATIONS_FIELD}[{situation_index}]"
        if not isinstance(situation, dict):
            raise ValueError(f"{location} must be an object.")
        results = situation.get(RESULT_FIELD)
        if not isinstance(results, list) or len(results) < top_k:
            raise ValueError(
                f"{location}.{RESULT_FIELD} must contain at least {top_k} results."
            )

        enriched_results: list[dict[str, Any]] = []
        seen_family_ids: set[str] = set()
        for result_index, result in enumerate(results[:top_k]):
            result_location = f"{location}.{RESULT_FIELD}[{result_index}]"
            if not isinstance(result, dict):
                raise ValueError(f"{result_location} must be an object.")
            family_id = require_string(
                result.get("family_id"), f"{result_location}.family_id"
            )
            if family_id in seen_family_ids:
                raise ValueError(f"{location} repeats family_id={family_id}.")
            if family_id not in catalog:
                raise ValueError(f"No commonsense generalization record found for {family_id}.")
            seen_family_ids.add(family_id)

            matched = result.get("matched_representation")
            if not isinstance(matched, dict):
                raise ValueError(
                    f"{result_location}.matched_representation must be an object."
                )
            family = catalog[family_id]
            enriched = copy.deepcopy(result)
            enriched["base_unit"] = copy.deepcopy(family["base_unit"])
            enriched["matched_unit"] = build_matched_unit(
                family=family,
                matched=matched,
                source_units=source_units,
                location=f"{result_location}.matched_representation",
            )
            enriched_results.append(enriched)
            enriched_count += 1
        situation[RESULT_FIELD] = enriched_results

    retrieval_config = output.get(CONFIG_FIELD)
    if isinstance(retrieval_config, dict):
        retrieval_config = copy.deepcopy(retrieval_config)
        retrieval_config["enriched_top_k"] = top_k
        retrieval_config["enriched_fields"] = ["base_unit", "matched_unit"]
        output[CONFIG_FIELD] = retrieval_config
    return output, enriched_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Keep the reranked Top K commonsense generalization records and add each family base unit "
            "and the exact base/variant unit used for reranking."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help=(
            "Dataset root. Unless explicitly overridden, input is read from "
            "<dataset-dir>/retrieval_output and output is written to "
            "<dataset-dir>/retrieval_output_with_commonsense."
        ),
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Explicit retrieval input directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Explicit enriched output directory.",
    )
    parser.add_argument("--catalog-file", type=Path, default=DEFAULT_CATALOG_FILE)
    parser.add_argument(
        "--source-files",
        type=Path,
        nargs="+",
        default=list(DEFAULT_SOURCE_FILES),
        help="Instance-Level Commonsense item JSON files used to resolve variant members.",
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--input-pattern",
        default=DEFAULT_INPUT_PATTERN,
        help="Glob used to select retrieval JSON files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be positive.")

    dataset_dir, input_dir, output_dir = resolve_dataset_paths(
        dataset_dir=args.dataset_dir,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        default_input_subdir=DEFAULT_INPUT_SUBDIR,
        default_output_subdir=DEFAULT_OUTPUT_SUBDIR,
    )
    catalog_file = resolve_path(args.catalog_file)
    source_files = [resolve_path(path) for path in args.source_files]
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if input_dir.resolve() == output_dir.resolve():
        raise ValueError("Input and output directories must be different.")
    if not catalog_file.is_file():
        raise FileNotFoundError(f"Catalog file does not exist: {catalog_file}")
    for path in source_files:
        if not path.is_file():
            raise FileNotFoundError(f"Source unit file does not exist: {path}")

    input_paths = sorted(
        path
        for path in input_dir.glob(args.input_pattern)
        if path.is_file() and path.name not in METADATA_FILENAMES
    )
    if not input_paths:
        raise FileNotFoundError(
            f"No files matching {args.input_pattern!r} found in {input_dir}."
        )

    catalog = load_catalog(catalog_file)
    source_units = load_source_units(source_files)
    total_enriched = 0
    for file_index, input_path in enumerate(input_paths, start=1):
        output, enriched_count = enrich_document(
            input_path,
            read_json(input_path),
            catalog=catalog,
            source_units=source_units,
            top_k=args.top_k,
        )
        write_json_atomic(output_dir / input_path.name, output)
        total_enriched += enriched_count
        print(
            f"[{file_index}/{len(input_paths)}] file={input_path.name} "
            f"enriched={enriched_count}"
        )

    summary = {
        "dataset_dir": str(dataset_dir),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "catalog_file": str(catalog_file),
        "source_files": [str(path) for path in source_files],
        "file_count": len(input_paths),
        "situation_count": total_enriched // args.top_k,
        "top_k": args.top_k,
        "enriched_candidate_count": total_enriched,
        "family_count": len(catalog),
        "source_unit_count": len(source_units),
    }
    write_json_atomic(output_dir / "enrichment_summary.json", summary)
    print(
        f"Done. files={len(input_paths)}, enriched={total_enriched}, "
        f"output_dir={output_dir}"
    )


if __name__ == "__main__":
    main()

