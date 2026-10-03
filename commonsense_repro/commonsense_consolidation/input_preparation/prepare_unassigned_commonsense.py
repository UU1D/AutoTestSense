"""Prepare a catalog and queue containing every currently unassigned unit."""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path
from typing import Any

from commonsense_repro.common.paths import DATA_ROOT, WORK_ROOT


DEFAULT_BASE_CATALOG = WORK_ROOT / "stage4" / "catalog_assembly" / "final_commonsense_generalization_records.json"
DEFAULT_SOURCE_FILES = (
    DATA_ROOT / "checkpoints" / "instance_level_commonsense" / "gemini_v1_2_extract.json",
    DATA_ROOT / "checkpoints" / "instance_level_commonsense" / "gemini_v1_2_github.json",
    DATA_ROOT / "checkpoints" / "instance_level_commonsense" / "gemini_v1_2_negative.json",
)
DEFAULT_OUTPUT_DIR = WORK_ROOT / "stage5" / "serial_inputs"


def natural_key(value: str) -> tuple[Any, ...]:
    return tuple(
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", value)
    )


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            json.dump(row, file, ensure_ascii=False)
            file.write("\n")
    temporary.replace(path)


def load_units(paths: list[Path]) -> dict[str, dict[str, Any]]:
    units: dict[str, dict[str, Any]] = {}
    for path in paths:
        data = read_json(path)
        if not isinstance(data, list):
            raise ValueError(f"{path} must contain a JSON array.")
        for item in data:
            if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                raise ValueError(f"Invalid unit in {path}.")
            unit_id = item["id"]
            if unit_id in units:
                raise ValueError(f"Duplicate unit id: {unit_id}")
            units[unit_id] = item
    return units


def covered_ids(catalog: dict[str, Any]) -> set[str]:
    families = catalog.get("rule_families")
    if not isinstance(families, list):
        raise ValueError("Catalog must contain rule_families.")
    covered: set[str] = set()
    for family in families:
        member_ids = family.get("member_ids") if isinstance(family, dict) else None
        if not isinstance(member_ids, list):
            raise ValueError("Every family must contain member_ids.")
        for member_id in member_ids:
            if not isinstance(member_id, str) or member_id in covered:
                raise ValueError(f"Invalid or repeated catalog member: {member_id!r}")
            covered.add(member_id)
    return covered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-catalog", type=Path, default=DEFAULT_BASE_CATALOG)
    parser.add_argument("--source-file", type=Path, action="append", dest="source_files")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    source_files = args.source_files or list(DEFAULT_SOURCE_FILES)
    catalog = read_json(args.base_catalog)
    if not isinstance(catalog, dict):
        raise ValueError("Catalog root must be an object.")
    units = load_units(source_files)
    covered = covered_ids(catalog)
    unknown = covered - set(units)
    if unknown:
        raise ValueError(f"Catalog contains unknown member IDs: {sorted(unknown)[:10]}")

    pending_ids = sorted(set(units) - covered, key=natural_key)
    queue = [
        {
            "serial_order": index,
            "id": unit_id,
            **{
                key: copy.deepcopy(value)
                for key, value in units[unit_id].items()
                if key != "id"
            },
        }
        for index, unit_id in enumerate(pending_ids)
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "initial_commonsense_generalization_records.json", catalog)
    write_jsonl(args.output_dir / "no_match_units.jsonl", queue)
    write_json(
        args.output_dir / "input_preparation_inputs_summary.json",
        {
            "source_unit_count": len(units),
            "initial_family_count": len(catalog["rule_families"]),
            "initial_covered_unit_count": len(covered),
            "pending_unit_count": len(queue),
        },
    )
    print(
        f"Done. families={len(catalog['rule_families'])}, covered={len(covered)}, "
        f"pending={len(queue)}, output_dir={args.output_dir}"
    )


if __name__ == "__main__":
    main()
