"""Export commonsense generalization record generalization results as a reusable JSON catalog."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


from commonsense_repro.common.paths import PACKAGE_ROOT as PROJECT_ROOT
DEFAULT_OUTPUT_DIR = Path(
    "work/stage3/"
    "gemini_v1_2_non_mutual_s15_leiden_max25_deepseek_flash_v1_4"
)
DEFAULT_RESULTS_FILE = DEFAULT_OUTPUT_DIR / "commonsense_generalization_results.jsonl"
DEFAULT_OUTPUT_FILE = DEFAULT_OUTPUT_DIR / "commonsense_generalization_records.json"


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def natural_sort_key(value: str) -> tuple[tuple[int, Any], ...]:
    parts = re.split(r"(\d+)", value.casefold())
    return tuple(
        (1, int(part)) if part.isdigit() else (0, part)
        for part in parts
        if part
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}.")
            records.append(record)
    return records


def require_string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} must be a non-empty string.")
    return value.strip()


def collect_member_ids(family: dict[str, Any], location: str) -> list[str]:
    equivalent_ids = family.get("equivalent_member_ids")
    variant_groups = family.get("variant_groups")
    if not isinstance(equivalent_ids, list):
        raise ValueError(f"{location}.equivalent_member_ids must be an array.")
    if not isinstance(variant_groups, list):
        raise ValueError(f"{location}.variant_groups must be an array.")

    member_ids = [
        require_string(value, f"{location}.equivalent_member_ids")
        for value in equivalent_ids
    ]
    for group_index, group in enumerate(variant_groups):
        group_location = f"{location}.variant_groups[{group_index}]"
        if not isinstance(group, dict) or not isinstance(group.get("member_ids"), list):
            raise ValueError(f"{group_location}.member_ids must be an array.")
        member_ids.extend(
            require_string(value, f"{group_location}.member_ids")
            for value in group["member_ids"]
        )

    if len(member_ids) != len(set(member_ids)):
        raise ValueError(f"{location} contains duplicate member IDs.")
    return sorted(member_ids, key=natural_sort_key)


def build_catalog(
    results: list[dict[str, Any]], *, source_results_file: str
) -> dict[str, Any]:
    catalog_families: list[dict[str, Any]] = []
    seen_family_ids: set[str] = set()
    seen_member_ids: set[str] = set()
    models: set[str] = set()
    prompt_hashes: set[str] = set()

    for result_index, result in enumerate(results):
        if result.get("status") != "success":
            continue
        batch_id = require_string(
            result.get("batch_id"), f"results[{result_index}].batch_id"
        )
        model = result.get("model")
        prompt_hash = result.get("prompt_sha256")
        if isinstance(model, str) and model.strip():
            models.add(model.strip())
        if isinstance(prompt_hash, str) and prompt_hash.strip():
            prompt_hashes.add(prompt_hash.strip())

        families = result.get("rule_families")
        if not isinstance(families, list):
            raise ValueError(f"Batch {batch_id}.rule_families must be an array.")
        for family_index, family in enumerate(families):
            location = f"batch {batch_id}.rule_families[{family_index}]"
            if not isinstance(family, dict):
                raise ValueError(f"{location} must be an object.")
            family_id = require_string(family.get("family_id"), f"{location}.family_id")
            if family_id in seen_family_ids:
                raise ValueError(f"Duplicate family ID: {family_id}")

            base_unit = family.get("base_unit")
            if not isinstance(base_unit, dict):
                raise ValueError(f"{location}.base_unit must be an object.")
            member_ids = collect_member_ids(family, location)
            overlap = seen_member_ids.intersection(member_ids)
            if overlap:
                duplicate = sorted(overlap, key=natural_sort_key)[0]
                raise ValueError(f"Source unit {duplicate} appears in multiple families.")

            catalog_families.append(
                {
                    "family_id": family_id,
                    "base_origin": require_string(
                        family.get("base_origin"), f"{location}.base_origin"
                    ),
                    "base_unit": base_unit,
                    "member_count": len(member_ids),
                    "member_ids": member_ids,
                    "equivalent_member_ids": family.get("equivalent_member_ids", []),
                    "variant_groups": family.get("variant_groups", []),
                    "family_rationale": require_string(
                        family.get("family_rationale"),
                        f"{location}.family_rationale",
                    ),
                    "source_batch": {
                        "batch_id": batch_id,
                        "parent_community_id": result.get("parent_community_id"),
                        "parent_size": result.get("parent_size"),
                        "input_size": result.get("input_size"),
                        "was_refined": result.get("was_refined"),
                    },
                }
            )
            seen_family_ids.add(family_id)
            seen_member_ids.update(member_ids)

    catalog_families.sort(key=lambda family: natural_sort_key(family["family_id"]))
    anchored_count = sum(
        family["base_origin"] == "ANCHORED" for family in catalog_families
    )
    synthesized_count = sum(
        family["base_origin"] == "SYNTHESIZED" for family in catalog_families
    )
    return {
        "source_results_file": source_results_file,
        "models": sorted(models),
        "prompt_sha256_values": sorted(prompt_hashes),
        "family_count": len(catalog_families),
        "anchored_family_count": anchored_count,
        "synthesized_family_count": synthesized_count,
        "represented_unit_count": len(seen_member_ids),
        "rule_families": catalog_families,
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export parsed commonsense generalization record results to one JSON catalog."
    )
    parser.add_argument("--results-file", type=Path, default=DEFAULT_RESULTS_FILE)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_path = resolve_path(args.results_file)
    output_path = resolve_path(args.output_file)
    results = read_jsonl(results_path)
    catalog = build_catalog(
        results,
        source_results_file=str(args.results_file),
    )
    write_json(output_path, catalog)
    print(
        "Done. "
        f"families={catalog['family_count']}, "
        f"represented_units={catalog['represented_unit_count']}, "
        f"output={output_path}"
    )


if __name__ == "__main__":
    main()

