"""Build complete base-and-variant situation embeddings for the final catalog."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Any


from commonsense_repro.common.paths import PACKAGE_ROOT as PROJECT_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_embedding_api = importlib.import_module(
    "commonsense_repro.commonsense_generalization.global.build_generalized_rule_embeddings"
)
DASHSCOPE_API_KEY = _embedding_api.DASHSCOPE_API_KEY
DASHSCOPE_URL = _embedding_api.DASHSCOPE_URL
EMBEDDING_MODEL = _embedding_api.EMBEDDING_MODEL
accumulate_usage = _embedding_api.accumulate_usage
append_jsonl = _embedding_api.append_jsonl
extract_embeddings = _embedding_api.extract_embeddings
iter_batches = _embedding_api.iter_batches
request_embeddings = _embedding_api.request_embeddings
resolve_endpoint = _embedding_api.resolve_endpoint
write_json = _embedding_api.write_json


# -----------------------------------------------------------------------------
# Editable configuration
# -----------------------------------------------------------------------------
DATA_ROOT = Path("work/stage4")
DEFAULT_INPUT_FILE = DATA_ROOT / (
    "commonsense_consolidation/serial_consolidation/deepseek_flash_v1_0/"
    "current_commonsense_generalization_records.json"
)
DEFAULT_OUTPUT_DIR = DATA_ROOT / "final_incremental_rule_library"
DEFAULT_BASE_OUTPUT_FILE = DEFAULT_OUTPUT_DIR / "generalized_rule_situation_embeddings.jsonl"
DEFAULT_OUTPUT_FILE = DEFAULT_OUTPUT_DIR / "commonsense_library_situation_embeddings.jsonl"
DEFAULT_SUMMARY_FILE = DEFAULT_OUTPUT_DIR / "commonsense_library_situation_embeddings_summary.json"

DEFAULT_SOURCE_EMBEDDING_FILES = (
    Path("data/reference/instance_level_commonsense_situation_embeddings/gemini_v1_2_extract_embeddings.jsonl"),
    Path("data/reference/instance_level_commonsense_situation_embeddings/gemini_v1_2_github_embeddings.jsonl"),
    Path("data/reference/instance_level_commonsense_situation_embeddings/gemini_v1_2_negative_embeddings.jsonl"),
)

EMBEDDING_FIELDS = ["situation"]
EMBEDDING_TEXT_TEMPLATE = "Situation: {situation}"
DEFAULT_BATCH_SIZE = 10
DEFAULT_TIMEOUT = 120
DEFAULT_MAX_RETRIES = 3


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def require_string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} must be a non-empty string.")
    return value.strip()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_jsonl(path: Path, rows: Iterator[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            json.dump(row, file, ensure_ascii=False)
            file.write("\n")
    temporary.replace(path)


def load_catalog_records(
    path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as file:
        catalog = json.load(file)
    families = catalog.get("rule_families") if isinstance(catalog, dict) else None
    if not isinstance(families, list):
        raise ValueError(f"{path} must contain a rule_families array.")

    records: list[dict[str, Any]] = []
    variants: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_member_ids: set[str] = set()
    for index, family in enumerate(families):
        location = f"{path}.rule_families[{index}]"
        if not isinstance(family, dict):
            raise ValueError(f"{location} must be an object.")
        family_id = require_string(family.get("family_id"), f"{location}.family_id")
        if family_id in seen_ids:
            raise ValueError(f"Duplicate family_id: {family_id}")
        seen_ids.add(family_id)

        base = family.get("base_unit")
        if not isinstance(base, dict):
            raise ValueError(f"{location}.base_unit must be an object.")
        situation = require_string(
            base.get("situation"), f"{location}.base_unit.situation"
        )
        embedding_text = EMBEDDING_TEXT_TEMPLATE.format(situation=situation)
        records.append(
            {
                "family_id": family_id,
                "family_origin": require_string(
                    family.get("family_origin"), f"{location}.family_origin"
                ),
                "base_origin": require_string(
                    family.get("base_origin"), f"{location}.base_origin"
                ),
                "situation": situation,
                "situation_sha256": sha256_text(situation),
                "embedding_text": embedding_text,
                "embedding_text_sha256": sha256_text(embedding_text),
            }
        )

        member_ids = family.get("member_ids")
        equivalent_ids = family.get("equivalent_member_ids")
        variant_groups = family.get("variant_groups")
        if not isinstance(member_ids, list) or not isinstance(equivalent_ids, list):
            raise ValueError(f"{location} has invalid member arrays.")
        if not isinstance(variant_groups, list):
            raise ValueError(f"{location}.variant_groups must be an array.")
        normalized_members = {
            require_string(value, f"{location}.member_ids") for value in member_ids
        }
        if len(normalized_members) != len(member_ids):
            raise ValueError(f"{family_id} contains duplicate member IDs.")
        if family.get("member_count") != len(normalized_members):
            raise ValueError(f"{family_id}.member_count is inconsistent.")
        assigned = {
            require_string(value, f"{location}.equivalent_member_ids")
            for value in equivalent_ids
        }
        for group_index, group in enumerate(variant_groups):
            if not isinstance(group, dict) or not isinstance(group.get("member_ids"), list):
                raise ValueError(f"{location}.variant_groups[{group_index}] is invalid.")
            for value in group["member_ids"]:
                member_id = require_string(
                    value, f"{location}.variant_groups[{group_index}].member_ids"
                )
                if member_id in assigned:
                    raise ValueError(f"{member_id} is assigned twice in {family_id}.")
                assigned.add(member_id)
                variants.append(
                    {
                        "family_id": family_id,
                        "member_id": member_id,
                        "variant_group_index": group_index,
                    }
                )
        if assigned != normalized_members:
            raise ValueError(f"{family_id} member structure is inconsistent.")
        overlap = seen_member_ids.intersection(normalized_members)
        if overlap:
            raise ValueError(f"Members occur in multiple families: {sorted(overlap)[:10]}")
        seen_member_ids.update(normalized_members)

    declared_count = catalog.get("family_count")
    if declared_count is not None and declared_count != len(records):
        raise ValueError(
            f"Catalog declares {declared_count} families but contains {len(records)}."
        )
    return records, variants


def load_variant_embeddings(
    paths: list[Path], variant_refs: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], str, int]:
    refs_by_id = {row["member_id"]: row for row in variant_refs}
    if len(refs_by_id) != len(variant_refs):
        raise ValueError("A variant member occurs in multiple variant groups.")
    found: dict[str, dict[str, Any]] = {}
    all_source_ids: set[str] = set()
    models: set[str] = set()
    dimension_values: set[int] = set()
    prefix = "Situation: "

    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Source embedding file does not exist: {path}")
        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"Expected an object at {path}:{line_number}.")
                unit_id = require_string(row.get("id"), f"{path}:{line_number}.id")
                if unit_id in all_source_ids:
                    raise ValueError(f"Duplicate source embedding ID: {unit_id}")
                all_source_ids.add(unit_id)
                model = require_string(row.get("model"), f"{path}:{line_number}.model")
                dimensions = row.get("dimensions")
                vector = row.get("embedding")
                if row.get("embedding_fields") != EMBEDDING_FIELDS:
                    raise ValueError(f"{path}:{line_number} is not situation-only.")
                if not isinstance(dimensions, int) or dimensions < 1:
                    raise ValueError(f"Invalid dimensions at {path}:{line_number}.")
                if not isinstance(vector, list) or len(vector) != dimensions:
                    raise ValueError(f"Invalid embedding at {path}:{line_number}.")
                models.add(model)
                dimension_values.add(dimensions)
                ref = refs_by_id.get(unit_id)
                if ref is None:
                    continue
                embedding_text = require_string(
                    row.get("embedding_text"), f"{path}:{line_number}.embedding_text"
                )
                if not embedding_text.startswith(prefix):
                    raise ValueError(f"Unexpected embedding text at {path}:{line_number}.")
                found[unit_id] = {
                    "representation_id": f"{ref['family_id']}::VARIANT::{unit_id}",
                    "representation_type": "VARIANT_MEMBER",
                    "family_id": ref["family_id"],
                    "member_id": unit_id,
                    "variant_group_index": ref["variant_group_index"],
                    "situation": embedding_text[len(prefix) :],
                    "embedding_text": embedding_text,
                    "model": model,
                    "dimensions": dimensions,
                    "embedding_fields": EMBEDDING_FIELDS,
                    "embedding": vector,
                }

    missing = sorted(set(refs_by_id) - set(found))
    if missing:
        raise ValueError(
            f"Missing situation embeddings for {len(missing)} variants: {missing[:20]}"
        )
    if len(models) != 1 or len(dimension_values) != 1:
        raise ValueError("Source embeddings must use one model and dimension.")
    ordered = [found[row["member_id"]] for row in variant_refs]
    return ordered, next(iter(models)), next(iter(dimension_values))


def complete_base_record(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "representation_id": f"{row['family_id']}::BASE",
        "representation_type": "BASE",
        "family_id": row["family_id"],
        "member_id": None,
        "variant_group_index": None,
        **{key: value for key, value in row.items() if key != "family_id"},
    }


def load_completed(
    path: Path,
    *,
    records_by_id: dict[str, dict[str, Any]],
    model: str,
    dimensions: int,
) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    completed: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            family_id = require_string(
                row.get("family_id"), f"{path}:{line_number}.family_id"
            )
            source = records_by_id.get(family_id)
            if source is None:
                raise ValueError(f"Unknown family {family_id} in existing output.")
            if family_id in completed:
                raise ValueError(f"Duplicate family {family_id} in existing output.")
            if row.get("model") != model or row.get("dimensions") != dimensions:
                raise ValueError(f"Embedding configuration changed for {family_id}.")
            if row.get("embedding_fields") != EMBEDDING_FIELDS:
                raise ValueError(f"Embedding fields changed for {family_id}.")
            if row.get("embedding_text_sha256") != source["embedding_text_sha256"]:
                raise ValueError(f"Base situation changed for {family_id}.")
            vector = row.get("embedding")
            if not isinstance(vector, list) or len(vector) != dimensions:
                raise ValueError(f"Invalid existing vector for {family_id}.")
            completed[family_id] = row
    return completed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, default=DEFAULT_INPUT_FILE)
    parser.add_argument(
        "--source-embedding-file",
        type=Path,
        action="append",
        dest="source_embedding_files",
        help="Repeat for each existing situation embedding JSONL file.",
    )
    parser.add_argument("--base-output-file", type=Path, default=DEFAULT_BASE_OUTPUT_FILE)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--summary-file", type=Path, default=DEFAULT_SUMMARY_FILE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.batch_size <= 10:
        raise ValueError("batch_size must be between 1 and 10.")
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive.")

    input_path = resolve_path(args.input_file)
    source_paths = [
        resolve_path(path)
        for path in (args.source_embedding_files or DEFAULT_SOURCE_EMBEDDING_FILES)
    ]
    base_output_path = resolve_path(args.base_output_file)
    output_path = resolve_path(args.output_file)
    summary_path = resolve_path(args.summary_file)

    all_records, all_variant_refs = load_catalog_records(input_path)
    records = all_records[: args.limit] if args.limit is not None else all_records
    selected_ids = {row["family_id"] for row in records}
    selected_variant_refs = [
        row for row in all_variant_refs if row["family_id"] in selected_ids
    ]
    variant_records, model, dimensions = load_variant_embeddings(
        source_paths, selected_variant_refs
    )
    if EMBEDDING_MODEL and EMBEDDING_MODEL != model:
        raise ValueError(
            f"EMBEDDING_MODEL={EMBEDDING_MODEL!r} differs from existing situation "
            f"embedding model {model!r}."
        )

    if args.dry_run:
        print(
            json.dumps(
                {
                    "base_count": len(records),
                    "variant_count": len(variant_records),
                    "complete_embedding_count": len(records) + len(variant_records),
                    "model": model,
                    "dimensions": dimensions,
                    "embedding_fields": EMBEDDING_FIELDS,
                    "sample_inputs": [row["embedding_text"] for row in records[:3]],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    missing_env = [
        name
        for name, value in (
            ("DASHSCOPE_API_KEY", DASHSCOPE_API_KEY),
            ("DASHSCOPE_URL", DASHSCOPE_URL),
            ("EMBEDDING_MODEL", EMBEDDING_MODEL),
        )
        if not value
    ]
    if missing_env:
        raise ValueError(f"Missing .env values: {', '.join(missing_env)}")

    if args.overwrite:
        for path in (base_output_path, output_path, summary_path):
            if path.exists():
                path.unlink()

    records_by_id = {row["family_id"]: row for row in all_records}
    existing = load_completed(
        base_output_path,
        records_by_id=records_by_id,
        model=model,
        dimensions=dimensions,
    )
    completed = {key: value for key, value in existing.items() if key in selected_ids}
    pending = [row for row in records if row["family_id"] not in completed]

    endpoint = resolve_endpoint(DASHSCOPE_URL)
    usage: dict[str, int] = {}
    request_count = 0
    for batch_number, batch in enumerate(iter_batches(pending, args.batch_size), start=1):
        response = request_embeddings(
            endpoint=endpoint,
            api_key=DASHSCOPE_API_KEY,
            model=model,
            texts=[row["embedding_text"] for row in batch],
            timeout=args.timeout,
            max_retries=args.max_retries,
        )
        vectors = extract_embeddings(response, len(batch))
        output_rows: list[dict[str, Any]] = []
        for source, vector in zip(batch, vectors):
            if len(vector) != dimensions:
                raise ValueError(
                    f"Expected {dimensions} dimensions for {source['family_id']}, "
                    f"got {len(vector)}."
                )
            output_rows.append(
                {
                    **source,
                    "model": model,
                    "dimensions": dimensions,
                    "embedding_fields": EMBEDDING_FIELDS,
                    "embedding": vector,
                }
            )
        append_jsonl(base_output_path, output_rows)
        accumulate_usage(usage, response)
        request_count += 1
        print(
            f"Wrote batch={batch_number} size={len(batch)} "
            f"first_family_id={batch[0]['family_id']}"
        )

    completed_bases = load_completed(
        base_output_path,
        records_by_id=records_by_id,
        model=model,
        dimensions=dimensions,
    )
    missing_bases = sorted(selected_ids - set(completed_bases))
    if missing_bases:
        raise ValueError(f"Base embeddings are incomplete: {missing_bases[:20]}")
    ordered_bases = [complete_base_record(completed_bases[row["family_id"]]) for row in records]
    write_jsonl(output_path, iter([*ordered_bases, *variant_records]))

    summary = {
        "input_file": str(input_path),
        "source_embedding_files": [str(path) for path in source_paths],
        "base_output_file": str(base_output_path),
        "output_file": str(output_path),
        "catalog_base_count": len(all_records),
        "selected_base_count": len(records),
        "selected_variant_count": len(variant_records),
        "complete_embedding_count": len(records) + len(variant_records),
        "already_completed_count": len(completed),
        "new_embedding_count": len(pending),
        "model": model,
        "dimensions": dimensions,
        "embedding_fields": EMBEDDING_FIELDS,
        "embedding_text_template": EMBEDDING_TEXT_TEMPLATE,
        "request_count": request_count,
        "usage": usage,
    }
    write_json(summary_path, summary)
    print(
        f"Done. bases={len(records)}, variants={len(variant_records)}, "
        f"complete={len(records) + len(variant_records)}, new_bases={len(pending)}, "
        f"skipped_bases={len(completed)}, output={output_path}"
    )


if __name__ == "__main__":
    main()

