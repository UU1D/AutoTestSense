"""Synchronize base embeddings with the current serial-consolidation catalog."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


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
extract_embeddings = _embedding_api.extract_embeddings
iter_batches = _embedding_api.iter_batches
request_embeddings = _embedding_api.request_embeddings
resolve_endpoint = _embedding_api.resolve_endpoint
from commonsense_repro.commonsense_consolidation.generalization_embedding_index.generalization_embedding_index import (
    EMBEDDING_FIELDS,
    EMBEDDING_TEMPLATE_VERSION,
    SyncAction,
    catalog_bases,
    file_sha256,
    manifest_active_records,
    plan_sync_actions,
    require_string,
    sha256_text,
    validate_vector_record,
)


# -----------------------------------------------------------------------------
# Editable configuration
# -----------------------------------------------------------------------------
DATA_ROOT = Path(
    "work/stage4"
)
DEFAULT_CATALOG_FILE = (
    DATA_ROOT / "commonsense_consolidation/input_preparation/initial_commonsense_generalization_records.json"
)
DEFAULT_BASELINE_CATALOG_FILE = DATA_ROOT / "catalog_assembly/final_commonsense_generalization_records.json"
DEFAULT_BASELINE_EMBEDDINGS_FILE = (
    DATA_ROOT / "commonsense_consolidation/generalized_rule_embeddings.jsonl"
)
DEFAULT_SOURCE_EMBEDDING_FILES = (
    Path("work/stage2/embeddings/gemini_v1_2_extract_embeddings.jsonl"),
    Path("work/stage2/embeddings/gemini_v1_2_github_embeddings.jsonl"),
    Path("work/stage2/embeddings/gemini_v1_2_negative_embeddings.jsonl"),
)
DEFAULT_OUTPUT_DIR = DATA_ROOT / "serial_family_embeddings"

EVENTS_FILENAME = "family_base_embedding_events.jsonl"
MANIFEST_FILENAME = "family_embedding_manifest.json"
MAPPING_FILENAME = "generalization_member_mapping.jsonl"
UPDATE_LOG_FILENAME = "family_embedding_update_log.jsonl"
SUMMARY_FILENAME = "family_embedding_summary.json"

DEFAULT_BATCH_SIZE = 10
DEFAULT_TIMEOUT = 120
DEFAULT_MAX_RETRIES = 3


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}.")
            rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterator[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            json.dump(row, file, ensure_ascii=False)
            file.write("\n")
    temporary.replace(path)


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as file:
        for row in rows:
            json.dump(row, file, ensure_ascii=False)
            file.write("\n")
        file.flush()


def indexed_events(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        event_id = require_string(row.get("event_id"), f"events[{index}].event_id")
        if event_id in result:
            raise ValueError(f"Duplicate event_id: {event_id}")
        result[event_id] = row
    return result


def load_baseline_embeddings(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(read_jsonl(path)):
        family_id = require_string(row.get("family_id"), f"baseline[{index}].family_id")
        if family_id in result:
            raise ValueError(f"Duplicate baseline family_id: {family_id}")
        result[family_id] = row
    return result


def load_source_embeddings(
    paths: list[Path], *, model: str, dimensions: int
) -> dict[str, dict[str, Any]]:
    by_text: dict[str, dict[str, Any]] = {}
    seen_ids: set[str] = set()
    for path in paths:
        for line_number, row in enumerate(read_jsonl(path), start=1):
            unit_id = require_string(row.get("id"), f"{path}:{line_number}.id")
            if unit_id in seen_ids:
                raise ValueError(f"Duplicate source embedding ID: {unit_id}")
            seen_ids.add(unit_id)
            validate_vector_record(
                row,
                model=model,
                dimensions=dimensions,
                location=f"{path}:{line_number}",
            )
            text = require_string(
                row.get("embedding_text"), f"{path}:{line_number}.embedding_text"
            )
            current = by_text.get(text)
            if current is None or unit_id < current["id"]:
                by_text[text] = row
    return by_text


def event_from_vector(
    *,
    event_id: str,
    action: SyncAction,
    catalog_version: int,
    model: str,
    dimensions: int,
    vector: list[float],
    vector_origin: str,
    source_reference: str | None,
) -> dict[str, Any]:
    base = action.family
    return {
        "event_id": event_id,
        "family_id": base["family_id"],
        "family_version": base["family_version"],
        "catalog_version": catalog_version,
        "update_type": action.action,
        "previous_base_unit_sha256": action.previous_base_unit_sha256,
        "base_unit": base["base_unit"],
        "base_unit_sha256": base["base_unit_sha256"],
        "embedding_text": base["embedding_text"],
        "embedding_text_sha256": base["embedding_text_sha256"],
        "embedding_template_version": EMBEDDING_TEMPLATE_VERSION,
        "embedding_fields": EMBEDDING_FIELDS,
        "model": model,
        "dimensions": dimensions,
        "vector_origin": vector_origin,
        "source_reference": source_reference,
        "embedding": vector,
    }


def next_event_ids(existing_count: int, count: int) -> list[str]:
    return [f"E{number:08d}" for number in range(existing_count + 1, existing_count + count + 1)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-file", type=Path, default=DEFAULT_CATALOG_FILE)
    parser.add_argument(
        "--baseline-catalog-file", type=Path, default=DEFAULT_BASELINE_CATALOG_FILE
    )
    parser.add_argument(
        "--baseline-embeddings-file",
        type=Path,
        default=DEFAULT_BASELINE_EMBEDDINGS_FILE,
    )
    parser.add_argument(
        "--source-embedding-file", type=Path, action="append", dest="source_files"
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.batch_size <= 10:
        raise ValueError("batch_size must be between 1 and 10.")

    catalog_path = resolve_path(args.catalog_file)
    baseline_catalog_path = resolve_path(args.baseline_catalog_file)
    baseline_embeddings_path = resolve_path(args.baseline_embeddings_file)
    source_paths = [
        resolve_path(path)
        for path in (args.source_files or DEFAULT_SOURCE_EMBEDDING_FILES)
    ]
    output_dir = resolve_path(args.output_dir)
    events_path = output_dir / EVENTS_FILENAME
    manifest_path = output_dir / MANIFEST_FILENAME
    mapping_path = output_dir / MAPPING_FILENAME
    update_log_path = output_dir / UPDATE_LOG_FILENAME
    summary_path = output_dir / SUMMARY_FILENAME

    catalog = read_json(catalog_path)
    if not isinstance(catalog, dict):
        raise ValueError("Catalog root must be an object.")
    bases, mappings = catalog_bases(catalog)
    catalog_version = catalog.get("catalog_version", 0)
    if not isinstance(catalog_version, int) or catalog_version < 0:
        raise ValueError("catalog_version must be a non-negative integer.")

    baseline_catalog = read_json(baseline_catalog_path)
    if not isinstance(baseline_catalog, dict):
        raise ValueError("Baseline catalog root must be an object.")
    baseline_bases, _ = catalog_bases(baseline_catalog)
    baseline_records = load_baseline_embeddings(baseline_embeddings_path)

    if not baseline_records:
        raise ValueError("Baseline embeddings are empty.")
    sample = next(iter(baseline_records.values()))
    model = require_string(sample.get("model"), "baseline model")
    dimensions = sample.get("dimensions")
    if not isinstance(dimensions, int) or dimensions < 1:
        raise ValueError("Baseline embedding dimensions are invalid.")
    if EMBEDDING_MODEL and EMBEDDING_MODEL != model:
        raise ValueError(
            f"EMBEDDING_MODEL={EMBEDDING_MODEL!r} differs from baseline model {model!r}."
        )
    for family_id, record in baseline_records.items():
        validate_vector_record(
            record, model=model, dimensions=dimensions, location=f"baseline {family_id}"
        )
    baseline_expected = {base["family_id"]: base for base in baseline_bases}
    if set(baseline_records) != set(baseline_expected):
        raise ValueError("Baseline catalog and baseline embeddings have different families.")
    for family_id, base in baseline_expected.items():
        record = baseline_records[family_id]
        if record.get("base_unit_sha256") != base["base_unit_sha256"]:
            raise ValueError(f"Baseline base hash mismatch for {family_id}.")
        if record.get("embedding_text") != base["embedding_text"]:
            raise ValueError(f"Baseline embedding text mismatch for {family_id}.")

    if args.overwrite:
        previous_manifest = None
        event_rows: list[dict[str, Any]] = []
    else:
        previous_manifest = read_json(manifest_path) if manifest_path.exists() else None
        event_rows = read_jsonl(events_path)
        if previous_manifest is None and event_rows:
            raise ValueError("Event log exists without a manifest; use --overwrite.")

    events_by_id = indexed_events(event_rows)
    active = (
        manifest_active_records(previous_manifest, events_by_id)
        if isinstance(previous_manifest, dict)
        else {}
    )
    for family_id, record in active.items():
        validate_vector_record(
            record, model=model, dimensions=dimensions, location=f"active {family_id}"
        )

    sources_by_text = load_source_embeddings(
        source_paths, model=model, dimensions=dimensions
    )
    actions = plan_sync_actions(
        bases,
        active_by_family=active,
        baseline_by_family=baseline_records,
        source_by_text=sources_by_text,
    )
    action_counts = Counter(action.action for action in actions)
    generation_actions = [
        action for action in actions if action.action.startswith("GENERATE_")
    ]

    preview = {
        "catalog_family_count": len(bases),
        "catalog_version": catalog_version,
        "action_counts": dict(sorted(action_counts.items())),
        "api_generation_count": len(generation_actions),
        "api_generation_family_ids": [a.family["family_id"] for a in generation_actions],
    }
    if args.dry_run:
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        return

    if generation_actions and not all((DASHSCOPE_API_KEY, DASHSCOPE_URL, model)):
        raise ValueError("DASHSCOPE_API_KEY, DASHSCOPE_URL, and model are required.")

    generated_vectors: dict[str, list[float]] = {}
    usage: dict[str, int] = {}
    request_count = 0
    endpoint = resolve_endpoint(DASHSCOPE_URL) if generation_actions else ""
    for batch in iter_batches(generation_actions, args.batch_size):
        response = request_embeddings(
            endpoint=endpoint,
            api_key=DASHSCOPE_API_KEY,
            model=model,
            texts=[action.family["embedding_text"] for action in batch],
            timeout=args.timeout,
            max_retries=args.max_retries,
        )
        vectors = extract_embeddings(response, len(batch))
        for action, vector in zip(batch, vectors):
            if len(vector) != dimensions:
                raise ValueError(
                    f"Unexpected dimensions for {action.family['family_id']}."
                )
            generated_vectors[action.family["family_id"]] = vector
        accumulate_usage(usage, response)
        request_count += 1

    new_event_count = (
        len(actions)
        if not active
        else len(actions) - action_counts["KEEP_EXISTING"]
    )
    event_ids = iter(next_event_ids(len(event_rows), new_event_count))
    new_events: list[dict[str, Any]] = []
    active_after: dict[str, dict[str, Any]] = {}
    for action in actions:
        family_id = action.family["family_id"]
        if action.action == "KEEP_EXISTING":
            if active:
                active_after[family_id] = action.source_record  # type: ignore[assignment]
                continue
            source = action.source_record
            vector_origin = "BASELINE_REUSED"
            source_reference = str(baseline_embeddings_path)
        elif action.action == "REUSE_SOURCE_FOR_NEW_FAMILY":
            source = action.source_record
            vector_origin = "SOURCE_UNIT_REUSED"
            source_reference = source.get("id") if source else None
        else:
            source = None
            vector_origin = "API_GENERATED"
            source_reference = None

        vector = (
            generated_vectors[family_id]
            if source is None
            else source["embedding"]
        )
        event = event_from_vector(
            event_id=next(event_ids),
            action=action,
            catalog_version=catalog_version,
            model=model,
            dimensions=dimensions,
            vector=vector,
            vector_origin=vector_origin,
            source_reference=source_reference,
        )
        new_events.append(event)
        active_after[family_id] = event

    expected_ids = {base["family_id"] for base in bases}
    if set(active_after) != expected_ids:
        raise ValueError("Active embedding coverage does not match the catalog.")
    for base in bases:
        event = active_after[base["family_id"]]
        if event.get("base_unit_sha256") != base["base_unit_sha256"]:
            raise ValueError(f"Active base hash mismatch for {base['family_id']}.")

    # Events are durable first. The manifest is the final activation pointer.
    if args.overwrite:
        write_jsonl(events_path, iter(new_events))
    else:
        append_jsonl(events_path, new_events)
    write_jsonl(mapping_path, iter(mappings))

    active_pointers = [
        {
            "family_id": base["family_id"],
            "family_version": base["family_version"],
            "event_id": active_after[base["family_id"]]["event_id"],
            "base_unit_sha256": base["base_unit_sha256"],
            "embedding_text_sha256": base["embedding_text_sha256"],
        }
        for base in bases
    ]
    manifest = {
        "schema_version": "1.0",
        "catalog_file": str(catalog_path),
        "catalog_sha256": file_sha256(catalog_path),
        "catalog_version": catalog_version,
        "model": model,
        "dimensions": dimensions,
        "embedding_fields": EMBEDDING_FIELDS,
        "embedding_template_version": EMBEDDING_TEMPLATE_VERSION,
        "event_log": str(events_path),
        "family_count": len(bases),
        "active_embeddings": active_pointers,
    }
    write_json(manifest_path, manifest)

    update_record = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "catalog_version": catalog_version,
        "catalog_sha256": manifest["catalog_sha256"],
        "action_counts": dict(sorted(action_counts.items())),
        "request_count": request_count,
        "usage": usage,
    }
    append_jsonl(update_log_path, [update_record])
    summary = {
        **preview,
        "model": model,
        "dimensions": dimensions,
        "embedding_fields": EMBEDDING_FIELDS,
        "embedding_template_version": EMBEDDING_TEMPLATE_VERSION,
        "source_embedding_count": len(sources_by_text),
        "member_mapping_count": len(mappings),
        "event_count": len(event_rows) + len(new_events),
        "new_event_count": len(new_events),
        "request_count": request_count,
        "usage": usage,
        "outputs": {
            "events": str(events_path),
            "manifest": str(manifest_path),
            "member_mapping": str(mapping_path),
            "update_log": str(update_log_path),
        },
    }
    write_json(summary_path, summary)
    print(
        "Done. "
        f"families={len(bases)}, actions={dict(sorted(action_counts.items()))}, "
        f"api_requests={request_count}, output_dir={output_dir}"
    )


if __name__ == "__main__":
    main()

