"""Filter positive common-sense violations and flatten their structured fields."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


# Edit these two paths for the dataset currently being processed.
DEFAULT_INPUT_FILE = Path("work/gemini_v1_2_negative.json")
DEFAULT_OUTPUT_FILE = Path("data/checkpoints/instance_level_commonsense/gemini_v1_2_negative.json")

VIOLATION_FIELDS = [
    "target_population",
    "domain",
    "situation",
    "violated_commonsense_rule",
    "applicability_conditions",
    "contextual_explanation",
]


def natural_sort_key(value: str) -> tuple[tuple[int, Any], ...]:
    parts = re.split(r"(\d+)", value.casefold())
    return tuple(
        (1, int(part)) if part.isdigit() else (0, part)
        for part in parts
        if part
    )


def load_json_array(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}.")

    records: list[dict[str, Any]] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(
                f"Expected a JSON object at index {index}, got {type(item).__name__}."
            )
        records.append(item)
    return records


def flatten_positive_record(item: dict[str, Any]) -> dict[str, Any]:
    unique_id = item.get("id")
    if not isinstance(unique_id, str) or not unique_id.strip():
        raise ValueError("Positive record is missing a non-empty string id.")

    violation = item.get("violated_common_sense")
    if not isinstance(violation, dict):
        raise ValueError(f"Record {unique_id} has no violated_common_sense object.")

    missing_fields = [field for field in VIOLATION_FIELDS if field not in violation]
    if missing_fields:
        raise ValueError(
            f"Record {unique_id} is missing violated_common_sense fields: "
            f"{', '.join(missing_fields)}"
        )

    return {
        "id": unique_id,
        "confidence_score": item.get("confidence_score"),
        **{field: violation[field] for field in VIOLATION_FIELDS},
    }


def filter_positive_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    positives = [
        flatten_positive_record(item)
        for item in records
        if item.get("is_candidate_common_sense_violation") is True
    ]
    positives.sort(key=lambda item: natural_sort_key(item["id"]))
    return positives


def validate_unique_ids(records: list[dict[str, Any]]) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for record in records:
        unique_id = record["id"]
        if unique_id in seen:
            duplicates.add(unique_id)
        seen.add(unique_id)

    if duplicates:
        duplicate_text = ", ".join(sorted(duplicates, key=natural_sort_key))
        raise ValueError(f"Duplicate positive record ids: {duplicate_text}")


def write_json(path: Path, data: Any, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists. Use --overwrite to replace it.")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Keep candidate common-sense violations and flatten violated_common_sense "
            "fields into a JSON array."
        )
    )
    parser.add_argument("--input-file", type=Path, default=DEFAULT_INPUT_FILE)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.input_file.is_file():
        raise FileNotFoundError(f"Input file does not exist: {args.input_file}")

    source_records = load_json_array(args.input_file)
    positive_records = filter_positive_records(source_records)
    validate_unique_ids(positive_records)
    write_json(args.output_file, positive_records, args.overwrite)

    print(
        "Done. "
        f"total={len(source_records)}, "
        f"positive={len(positive_records)}, "
        f"filtered_out={len(source_records) - len(positive_records)}, "
        f"output={args.output_file}"
    )


if __name__ == "__main__":
    main()

