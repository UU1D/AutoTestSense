"""Add complete instance-level commonsense items to ablation retrieval results."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
from commonsense_repro.common.paths import PACKAGE_ROOT as PROJECT_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from commonsense_repro.retrieval.batch_retrieve_commonsense_library import (
    SITUATIONS_FIELD,
    read_json,
    require_string,
    resolve_dataset_paths,
    resolve_path,
    write_json_atomic,
)
from commonsense_repro.retrieval.enrich_retrieved_commonsense_library import (
    load_source_units,
)


# -----------------------------------------------------------------------------
# Editable configuration
# -----------------------------------------------------------------------------
DEFAULT_DATASET_DIR = SCRIPT_DIR
DEFAULT_INPUT_SUBDIR = "retrieval_output_instance_level"
DEFAULT_OUTPUT_SUBDIR = "retrieval_output_instance_level_with_commonsense"
DEFAULT_SOURCE_FILES = (
    Path("data/checkpoints/instance_level_commonsense/gemini_v1_2_extract.json"),
    Path("data/checkpoints/instance_level_commonsense/gemini_v1_2_github.json"),
    Path("data/checkpoints/instance_level_commonsense/gemini_v1_2_negative.json"),
)

RESULT_FIELD = "related_commonsense"
DEFAULT_TOP_K = 30
DEFAULT_INPUT_PATTERN = "*.json"
METADATA_FILENAMES = {"retrieval_summary.json", "enrichment_summary.json"}
EXPECTED_SOURCE_UNIT_COUNT = 3708


def enrich_document(
    path: Path,
    data: Any,
    *,
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
        seen_ids: set[str] = set()
        for result_index, result in enumerate(results[:top_k]):
            result_location = f"{location}.{RESULT_FIELD}[{result_index}]"
            if not isinstance(result, dict):
                raise ValueError(f"{result_location} must be an object.")
            unit_id = require_string(result.get("id"), f"{result_location}.id")
            if unit_id in seen_ids:
                raise ValueError(f"{location} repeats unit id={unit_id}.")
            if unit_id not in source_units:
                raise ValueError(f"No instance-level commonsense item found for {unit_id}.")
            seen_ids.add(unit_id)

            source_unit = source_units[unit_id]
            retrieved_situation = result.get("situation")
            if (
                isinstance(retrieved_situation, str)
                and retrieved_situation != source_unit["situation"]
            ):
                raise ValueError(
                    f"Retrieved situation differs from source unit {unit_id}: "
                    f"{retrieved_situation!r} != {source_unit['situation']!r}"
                )

            enriched = copy.deepcopy(result)
            enriched["commonsense_unit"] = {
                "unit_id": unit_id,
                **copy.deepcopy(source_unit),
            }
            enriched_results.append(enriched)
            enriched_count += 1

        situation[RESULT_FIELD] = enriched_results

    return output, enriched_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Keep the reranked Top K instance-level commonsense items and add each unit's "
            "situation and common-sense rule."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help=(
            "Dataset directory containing retrieval_output_instance_level. "
            "Relative paths are resolved from the project root."
        ),
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help=(
            "Optional explicit input directory; overrides "
            "dataset-dir/retrieval_output_instance_level."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Optional explicit output directory; overrides "
            "dataset-dir/retrieval_output_instance_level_with_commonsense."
        ),
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--source-files",
        type=Path,
        nargs="+",
        default=list(DEFAULT_SOURCE_FILES),
        help="Instance-Level Commonsense item JSON files.",
    )
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
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")
    source_files = [resolve_path(path) for path in args.source_files]
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if input_dir.resolve() == output_dir.resolve():
        raise ValueError("Input and output directories must be different.")
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

    source_units = load_source_units(source_files)
    if len(source_units) != EXPECTED_SOURCE_UNIT_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_SOURCE_UNIT_COUNT} source units, "
            f"got {len(source_units)}."
        )

    total_enriched = 0
    for file_index, input_path in enumerate(input_paths, start=1):
        output, enriched_count = enrich_document(
            input_path,
            read_json(input_path),
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
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "source_files": [str(path) for path in source_files],
        "file_count": len(input_paths),
        "situation_count": total_enriched // args.top_k,
        "top_k": args.top_k,
        "enriched_candidate_count": total_enriched,
        "source_unit_count": len(source_units),
    }
    write_json_atomic(output_dir / "enrichment_summary.json", summary)
    print(
        f"Done. files={len(input_paths)}, enriched={total_enriched}, "
        f"output_dir={output_dir}"
    )


if __name__ == "__main__":
    main()

