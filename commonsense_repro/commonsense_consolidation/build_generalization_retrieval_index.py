"""Build the reusable vector index inputs for incremental family matching.

Only final family bases are embedded here. Existing source-unit vectors are
validated and referenced through a member-to-family mapping instead of copied.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Iterator


from commonsense_repro.common.paths import PACKAGE_ROOT as PROJECT_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from commonsense_repro.clustering.build_instance_level_commonsense_embeddings import build_embedding_text
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
# Configuration
# -----------------------------------------------------------------------------
DEFAULT_FINAL_CATALOG = Path(
    "work/stage4/"
    "catalog_assembly/final_commonsense_generalization_records.json"
)
DEFAULT_SOURCE_EMBEDDING_FILES = (
    Path("work/stage2/embeddings/gemini_v1_2_extract_embeddings.jsonl"),
    Path("work/stage2/embeddings/gemini_v1_2_github_embeddings.jsonl"),
    Path("work/stage2/embeddings/gemini_v1_2_negative_embeddings.jsonl"),
)
DEFAULT_OUTPUT_DIR = Path(
    "work/stage4/"
    "commonsense_consolidation"
)
BASE_EMBEDDINGS_FILENAME = "generalized_rule_embeddings.jsonl"
MEMBER_MAPPING_FILENAME = "generalization_member_mapping.jsonl"
SUMMARY_FILENAME = "generalization_retrieval_index_summary.json"

EMBEDDING_FIELDS = ["violated_commonsense_rule", "situation"]
DEFAULT_BATCH_SIZE = 10
DEFAULT_TIMEOUT = 120
DEFAULT_MAX_RETRIES = 3


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def require_string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} must be a non-empty string.")
    return value.strip()


def sha256_json(value: Any) -> str:
    serialized = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def base_embedding_record(family: dict[str, Any]) -> dict[str, Any]:
    base = family.get("base_unit")
    if not isinstance(base, dict):
        raise ValueError(f"Family {family.get('family_id')} has no base_unit object.")
    canonical_base = {
        "situation": require_string(base.get("situation"), "base_unit.situation"),
        "commonsense_rule": require_string(
            base.get("commonsense_rule"), "base_unit.commonsense_rule"
        ),
    }
    embedding_input = {
        "violated_commonsense_rule": canonical_base["commonsense_rule"],
        "situation": canonical_base["situation"],
    }
    return {
        "family_id": require_string(family.get("family_id"), "family_id"),
        "family_origin": require_string(family.get("family_origin"), "family_origin"),
        "base_origin": require_string(family.get("base_origin"), "base_origin"),
        "member_count": family.get("member_count"),
        "base_unit": canonical_base,
        "base_unit_sha256": sha256_json(canonical_base),
        "embedding_text": build_embedding_text(embedding_input, EMBEDDING_FIELDS),
    }


def build_member_mapping(family: dict[str, Any]) -> list[dict[str, Any]]:
    family_id = require_string(family.get("family_id"), "family_id")
    member_ids = family.get("member_ids")
    equivalent_ids = family.get("equivalent_member_ids")
    variant_groups = family.get("variant_groups")
    if not isinstance(member_ids, list) or not member_ids:
        raise ValueError(f"Family {family_id} must contain member_ids.")
    if not isinstance(equivalent_ids, list) or not isinstance(variant_groups, list):
        raise ValueError(
            f"Family {family_id} must contain equivalent_member_ids and variant_groups."
        )

    expected = [require_string(value, f"{family_id}.member_ids") for value in member_ids]
    if len(expected) != len(set(expected)):
        raise ValueError(f"Family {family_id} contains duplicate member_ids.")
    declared_count = family.get("member_count")
    if declared_count != len(expected):
        raise ValueError(
            f"Family {family_id} member_count={declared_count} but has {len(expected)} IDs."
        )

    rows: list[dict[str, Any]] = []
    assigned: set[str] = set()
    for member_id in equivalent_ids:
        member_id = require_string(member_id, f"{family_id}.equivalent_member_ids")
        if member_id in assigned:
            raise ValueError(f"Member {member_id} is assigned twice in {family_id}.")
        assigned.add(member_id)
        rows.append(
            {
                "member_id": member_id,
                "family_id": family_id,
                "member_type": "EQUIVALENT",
                "variant_group_index": None,
            }
        )

    for group_index, group in enumerate(variant_groups):
        if not isinstance(group, dict) or not isinstance(group.get("member_ids"), list):
            raise ValueError(f"Family {family_id} variant group {group_index} is invalid.")
        for member_id in group["member_ids"]:
            member_id = require_string(
                member_id, f"{family_id}.variant_groups[{group_index}].member_ids"
            )
            if member_id in assigned:
                raise ValueError(f"Member {member_id} is assigned twice in {family_id}.")
            assigned.add(member_id)
            rows.append(
                {
                    "member_id": member_id,
                    "family_id": family_id,
                    "member_type": "VARIANT",
                    "variant_group_index": group_index,
                }
            )

    expected_set = set(expected)
    if assigned != expected_set:
        raise ValueError(
            f"Family {family_id} member structure is inconsistent: "
            f"missing={sorted(expected_set - assigned)}, "
            f"extra={sorted(assigned - expected_set)}"
        )
    position = {member_id: index for index, member_id in enumerate(expected)}
    rows.sort(key=lambda row: position[row["member_id"]])
    return rows


def load_catalog(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as file:
        root = json.load(file)
    families = root.get("rule_families") if isinstance(root, dict) else None
    if not isinstance(families, list):
        raise ValueError(f"{path} must contain a rule_families array.")

    bases: list[dict[str, Any]] = []
    mappings: list[dict[str, Any]] = []
    seen_family_ids: set[str] = set()
    seen_member_ids: set[str] = set()
    for family in families:
        if not isinstance(family, dict):
            raise ValueError("Every commonsense generalization record must be an object.")
        base = base_embedding_record(family)
        family_id = base["family_id"]
        if family_id in seen_family_ids:
            raise ValueError(f"Duplicate family_id: {family_id}")
        seen_family_ids.add(family_id)
        family_mappings = build_member_mapping(family)
        for row in family_mappings:
            member_id = row["member_id"]
            if member_id in seen_member_ids:
                raise ValueError(f"Member {member_id} occurs in multiple families.")
            seen_member_ids.add(member_id)
        bases.append(base)
        mappings.extend(family_mappings)
    return bases, mappings


def load_source_embedding_metadata(
    paths: list[Path],
) -> tuple[dict[str, dict[str, Any]], str, int]:
    metadata: dict[str, dict[str, Any]] = {}
    models: set[str] = set()
    dimensions: set[int] = set()
    field_configs: set[tuple[str, ...]] = set()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Source embedding file does not exist: {path}")
        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"Expected an object at {path}:{line_number}.")
                unit_id = require_string(record.get("id"), f"{path}:{line_number}.id")
                if unit_id in metadata:
                    raise ValueError(f"Duplicate source embedding ID: {unit_id}")
                model = require_string(record.get("model"), f"{path}:{line_number}.model")
                dimension = record.get("dimensions")
                fields = record.get("embedding_fields")
                embedding = record.get("embedding")
                if not isinstance(dimension, int) or dimension < 1:
                    raise ValueError(f"Invalid dimensions at {path}:{line_number}.")
                if fields != EMBEDDING_FIELDS:
                    raise ValueError(
                        f"Unexpected embedding_fields at {path}:{line_number}: {fields}"
                    )
                if not isinstance(embedding, list) or len(embedding) != dimension:
                    raise ValueError(f"Invalid embedding vector at {path}:{line_number}.")
                models.add(model)
                dimensions.add(dimension)
                field_configs.add(tuple(fields))
                metadata[unit_id] = {
                    "source_embedding_file": str(path),
                    "model": model,
                    "dimensions": dimension,
                }
    if len(models) != 1 or len(dimensions) != 1 or len(field_configs) != 1:
        raise ValueError("Source embedding files do not use one consistent configuration.")
    return metadata, next(iter(models)), next(iter(dimensions))


def validate_member_embeddings(
    mappings: list[dict[str, Any]], metadata: dict[str, dict[str, Any]]
) -> None:
    member_ids = {row["member_id"] for row in mappings}
    missing = sorted(member_ids - set(metadata))
    if missing:
        preview = missing[:20]
        raise ValueError(
            f"Missing source embeddings for {len(missing)} family members: {preview}"
        )
    for row in mappings:
        row["source_embedding_file"] = metadata[row["member_id"]][
            "source_embedding_file"
        ]


def read_existing_base_embeddings(
    path: Path,
    *,
    model: str,
    bases_by_id: dict[str, dict[str, Any]],
    expected_dimensions: int,
) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    completed: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            family_id = require_string(
                record.get("family_id"), f"{path}:{line_number}.family_id"
            )
            source = bases_by_id.get(family_id)
            if source is None:
                raise ValueError(f"Unknown existing family {family_id}; use --overwrite.")
            if family_id in completed:
                raise ValueError(f"Duplicate existing family embedding: {family_id}")
            if record.get("model") != model:
                raise ValueError(f"Model changed for {family_id}; use --overwrite.")
            if record.get("dimensions") != expected_dimensions:
                raise ValueError(f"Dimensions changed for {family_id}; use --overwrite.")
            if record.get("embedding_fields") != EMBEDDING_FIELDS:
                raise ValueError(f"Fields changed for {family_id}; use --overwrite.")
            if record.get("base_unit_sha256") != source["base_unit_sha256"]:
                raise ValueError(f"Base unit changed for {family_id}; use --overwrite.")
            if record.get("embedding_text") != source["embedding_text"]:
                raise ValueError(f"Embedding text changed for {family_id}; use --overwrite.")
            embedding = record.get("embedding")
            if not isinstance(embedding, list) or len(embedding) != expected_dimensions:
                raise ValueError(f"Invalid existing vector for {family_id}.")
            completed[family_id] = record
    return completed


def write_jsonl(path: Path, rows: Iterator[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            json.dump(row, file, ensure_ascii=False)
            file.write("\n")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-file", type=Path, default=DEFAULT_FINAL_CATALOG)
    parser.add_argument(
        "--source-embedding-file",
        type=Path,
        action="append",
        dest="source_embedding_files",
        help="Repeat for each source-unit embedding JSONL file.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
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

    catalog_path = resolve_path(args.catalog_file)
    source_paths = [
        resolve_path(path)
        for path in (args.source_embedding_files or DEFAULT_SOURCE_EMBEDDING_FILES)
    ]
    output_dir = resolve_path(args.output_dir)
    base_output = output_dir / BASE_EMBEDDINGS_FILENAME
    mapping_output = output_dir / MEMBER_MAPPING_FILENAME
    summary_output = output_dir / SUMMARY_FILENAME

    if not catalog_path.is_file():
        raise FileNotFoundError(f"Catalog file does not exist: {catalog_path}")
    all_bases, mappings = load_catalog(catalog_path)
    source_metadata, source_model, source_dimensions = load_source_embedding_metadata(
        source_paths
    )
    validate_member_embeddings(mappings, source_metadata)

    if EMBEDDING_MODEL and EMBEDDING_MODEL != source_model:
        raise ValueError(
            f"EMBEDDING_MODEL={EMBEDDING_MODEL!r} differs from existing source "
            f"embedding model {source_model!r}."
        )

    selected_bases = all_bases[: args.limit] if args.limit is not None else all_bases
    selected_ids = {base["family_id"] for base in selected_bases}
    selected_mappings = [row for row in mappings if row["family_id"] in selected_ids]

    if args.dry_run:
        preview = [
            {
                "family_id": base["family_id"],
                "embedding_text": base["embedding_text"],
            }
            for base in selected_bases[:3]
        ]
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        print(
            f"Dry run OK. families={len(selected_bases)}, "
            f"mapped_members={len(selected_mappings)}, "
            f"source_embeddings={len(source_metadata)}"
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
        for path in (base_output, mapping_output, summary_output):
            if path.exists():
                path.unlink()

    bases_by_id = {base["family_id"]: base for base in all_bases}
    all_completed = read_existing_base_embeddings(
        base_output,
        model=source_model,
        bases_by_id=bases_by_id,
        expected_dimensions=source_dimensions,
    )
    completed = {
        family_id: record
        for family_id, record in all_completed.items()
        if family_id in selected_ids
    }
    pending = [base for base in selected_bases if base["family_id"] not in completed]

    endpoint = resolve_endpoint(DASHSCOPE_URL)
    usage: dict[str, int] = {}
    request_count = 0
    for batch_number, batch in enumerate(
        iter_batches(pending, args.batch_size), start=1
    ):
        response = request_embeddings(
            endpoint=endpoint,
            api_key=DASHSCOPE_API_KEY,
            model=source_model,
            texts=[base["embedding_text"] for base in batch],
            timeout=args.timeout,
            max_retries=args.max_retries,
        )
        embeddings = extract_embeddings(response, len(batch))
        output_rows: list[dict[str, Any]] = []
        for base, embedding in zip(batch, embeddings):
            if len(embedding) != source_dimensions:
                raise ValueError(
                    f"Expected {source_dimensions} dimensions for {base['family_id']}, "
                    f"got {len(embedding)}."
                )
            output_rows.append(
                {
                    **base,
                    "model": source_model,
                    "dimensions": source_dimensions,
                    "embedding_fields": EMBEDDING_FIELDS,
                    "embedding": embedding,
                }
            )
        append_jsonl(base_output, output_rows)
        accumulate_usage(usage, response)
        request_count += 1
        print(
            f"Wrote batch={batch_number} size={len(batch)} "
            f"first_family_id={batch[0]['family_id']}"
        )

    write_jsonl(mapping_output, iter(selected_mappings))
    family_member_ids = {row["member_id"] for row in mappings}
    summary = {
        "catalog_file": str(catalog_path),
        "source_embedding_files": [str(path) for path in source_paths],
        "outputs": {
            "generalized_rule_embeddings": str(base_output),
            "generalization_member_mapping": str(mapping_output),
        },
        "model": source_model,
        "dimensions": source_dimensions,
        "embedding_fields": EMBEDDING_FIELDS,
        "embedding_text_template": (
            "Violated common-sense rule: {commonsense_rule}\\n"
            "Situation: {situation}"
        ),
        "catalog_family_count": len(all_bases),
        "selected_family_count": len(selected_bases),
        "catalog_member_count": len(mappings),
        "selected_member_count": len(selected_mappings),
        "source_embedding_count": len(source_metadata),
        "unassigned_source_embedding_count": len(set(source_metadata) - family_member_ids),
        "already_completed_base_count": len(completed),
        "new_base_count": len(pending),
        "request_count": request_count,
        "usage": usage,
    }
    write_json(summary_output, summary)
    print(
        f"Done. families={len(selected_bases)}, members={len(selected_mappings)}, "
        f"new_bases={len(pending)}, skipped_bases={len(completed)}, "
        f"unassigned_units={summary['unassigned_source_embedding_count']}, "
        f"output_dir={output_dir}"
    )


if __name__ == "__main__":
    main()

