"""Recall Top-K commonsense generalization records for instance-level commonsense items not yet in the catalog.

Each family is represented by its combined base embedding and the existing
combined embeddings of its variant members. Its recall score is the maximum
cosine similarity over those representations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np


from commonsense_repro.common.paths import PACKAGE_ROOT as PROJECT_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DEFAULT_INDEX_DIR = Path(
    "work/stage4/"
    "commonsense_consolidation"
)
DEFAULT_BASE_EMBEDDINGS_FILE = DEFAULT_INDEX_DIR / "generalized_rule_embeddings.jsonl"
DEFAULT_MEMBER_MAPPING_FILE = DEFAULT_INDEX_DIR / "generalization_member_mapping.jsonl"
DEFAULT_SOURCE_EMBEDDING_FILES = (
    Path("work/stage2/embeddings/gemini_v1_2_extract_embeddings.jsonl"),
    Path("work/stage2/embeddings/gemini_v1_2_github_embeddings.jsonl"),
    Path("work/stage2/embeddings/gemini_v1_2_negative_embeddings.jsonl"),
)
DEFAULT_OUTPUT_DIR = DEFAULT_INDEX_DIR / "recall"

DEFAULT_TOP_K = 20
DEFAULT_CHUNK_SIZE = 256
EXPECTED_EMBEDDING_FIELDS = ["violated_commonsense_rule", "situation"]


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def natural_sort_key(value: str) -> tuple[tuple[int, Any], ...]:
    parts = re.split(r"(\d+)", value.casefold())
    return tuple(
        (1, int(part)) if part.isdigit() else (0, part)
        for part in parts
        if part
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} must be a non-empty string.")
    return value.strip()


def require_vector(value: Any, location: str) -> list[float]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{location} must be a non-empty array.")
    vector = np.asarray(value, dtype=np.float32)
    if vector.ndim != 1 or not np.isfinite(vector).all():
        raise ValueError(f"{location} must be a finite one-dimensional vector.")
    if np.linalg.norm(vector) == 0:
        raise ValueError(f"{location} must not be a zero vector.")
    return vector.tolist()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}.")
            rows.append(row)
    return rows


def load_base_embeddings(path: Path) -> list[dict[str, Any]]:
    bases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, row in enumerate(read_jsonl(path), start=1):
        location = f"{path}:{line_number}"
        family_id = require_string(row.get("family_id"), f"{location}.family_id")
        if family_id in seen:
            raise ValueError(f"Duplicate base family_id: {family_id}")
        vector = require_vector(row.get("embedding"), f"{location}.embedding")
        dimensions = row.get("dimensions")
        if dimensions != len(vector):
            raise ValueError(f"Incorrect dimensions for base family {family_id}.")
        if row.get("embedding_fields") != EXPECTED_EMBEDDING_FIELDS:
            raise ValueError(f"Unexpected embedding fields for base family {family_id}.")
        bases.append(
            {
                "family_id": family_id,
                "model": require_string(row.get("model"), f"{location}.model"),
                "dimensions": dimensions,
                "embedding": vector,
            }
        )
        seen.add(family_id)
    if not bases:
        raise ValueError(f"No base embeddings found in {path}.")
    bases.sort(key=lambda row: natural_sort_key(row["family_id"]))
    return bases


def load_source_embeddings(paths: list[Path]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in paths:
        for line_number, row in enumerate(read_jsonl(path), start=1):
            location = f"{path}:{line_number}"
            unit_id = require_string(row.get("id"), f"{location}.id")
            if unit_id in records:
                raise ValueError(f"Duplicate source embedding ID: {unit_id}")
            vector = require_vector(row.get("embedding"), f"{location}.embedding")
            dimensions = row.get("dimensions")
            if dimensions != len(vector):
                raise ValueError(f"Incorrect dimensions for source unit {unit_id}.")
            if row.get("embedding_fields") != EXPECTED_EMBEDDING_FIELDS:
                raise ValueError(f"Unexpected embedding fields for source unit {unit_id}.")
            records[unit_id] = {
                "id": unit_id,
                "model": require_string(row.get("model"), f"{location}.model"),
                "dimensions": dimensions,
                "embedding_text": require_string(
                    row.get("embedding_text"), f"{location}.embedding_text"
                ),
                "embedding": vector,
            }
    if not records:
        raise ValueError("No source-unit embeddings were loaded.")
    return records


def load_member_mapping(
    path: Path, known_family_ids: set[str]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_members: set[str] = set()
    for line_number, row in enumerate(read_jsonl(path), start=1):
        location = f"{path}:{line_number}"
        member_id = require_string(row.get("member_id"), f"{location}.member_id")
        family_id = require_string(row.get("family_id"), f"{location}.family_id")
        if member_id in seen_members:
            raise ValueError(f"Duplicate mapped member_id: {member_id}")
        if family_id not in known_family_ids:
            raise ValueError(f"Mapping refers to unknown family_id: {family_id}")
        member_type = require_string(row.get("member_type"), f"{location}.member_type")
        if member_type not in {"EQUIVALENT", "VARIANT"}:
            raise ValueError(f"Unknown member_type for {member_id}: {member_type}")
        variant_group_index = row.get("variant_group_index")
        if member_type == "VARIANT" and not isinstance(variant_group_index, int):
            raise ValueError(f"Variant member {member_id} lacks variant_group_index.")
        if member_type == "EQUIVALENT" and variant_group_index is not None:
            raise ValueError(f"Equivalent member {member_id} has a variant group.")
        rows.append(
            {
                "member_id": member_id,
                "family_id": family_id,
                "member_type": member_type,
                "variant_group_index": variant_group_index,
            }
        )
        seen_members.add(member_id)
    return rows


def validate_embedding_configuration(
    bases: list[dict[str, Any]], source: dict[str, dict[str, Any]]
) -> tuple[str, int]:
    models = {row["model"] for row in bases} | {row["model"] for row in source.values()}
    dimensions = {row["dimensions"] for row in bases} | {
        row["dimensions"] for row in source.values()
    }
    if len(models) != 1:
        raise ValueError(f"Embeddings use multiple models: {sorted(models)}")
    if len(dimensions) != 1:
        raise ValueError(f"Embeddings use multiple dimensions: {sorted(dimensions)}")
    return next(iter(models)), next(iter(dimensions))


def normalize_rows(matrix: np.ndarray, location: str) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError(f"{location} contains a zero vector.")
    return matrix / norms


def round_score(value: float) -> float:
    return round(float(value), 8)


def score_distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float32)
    return {
        "min": round_score(np.min(array)),
        "p10": round_score(np.quantile(array, 0.10)),
        "median": round_score(np.quantile(array, 0.50)),
        "mean": round_score(np.mean(array)),
        "p90": round_score(np.quantile(array, 0.90)),
        "max": round_score(np.max(array)),
    }


def build_recall_records(
    *,
    bases: list[dict[str, Any]],
    mappings: list[dict[str, Any]],
    source: dict[str, dict[str, Any]],
    query_ids: list[str],
    top_k: int,
    chunk_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    family_ids = [row["family_id"] for row in bases]
    family_index = {family_id: index for index, family_id in enumerate(family_ids)}
    if not 1 <= top_k <= len(family_ids):
        raise ValueError(f"top_k must be between 1 and {len(family_ids)}.")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")

    representations: list[dict[str, Any]] = []
    for base in bases:
        representations.append(
            {
                "family_id": base["family_id"],
                "representation_type": "BASE",
                "member_id": None,
                "variant_group_index": None,
                "embedding": base["embedding"],
            }
        )
    for mapping in mappings:
        if mapping["member_type"] != "VARIANT":
            continue
        member_id = mapping["member_id"]
        if member_id not in source:
            raise ValueError(f"Missing embedding for variant member {member_id}.")
        representations.append(
            {
                "family_id": mapping["family_id"],
                "representation_type": "VARIANT_MEMBER",
                "member_id": member_id,
                "variant_group_index": mapping["variant_group_index"],
                "embedding": source[member_id]["embedding"],
            }
        )

    representation_matrix = normalize_rows(
        np.asarray([row["embedding"] for row in representations], dtype=np.float32),
        "family representations",
    )
    representation_indices_by_family: list[np.ndarray] = []
    for family_id in family_ids:
        representation_indices_by_family.append(
            np.asarray(
                [
                    index
                    for index, representation in enumerate(representations)
                    if representation["family_id"] == family_id
                ],
                dtype=np.int64,
            )
        )

    output: list[dict[str, Any]] = []
    top_one_scores: list[float] = []
    for start in range(0, len(query_ids), chunk_size):
        chunk_ids = query_ids[start : start + chunk_size]
        query_matrix = normalize_rows(
            np.asarray([source[unit_id]["embedding"] for unit_id in chunk_ids], dtype=np.float32),
            "query embeddings",
        )
        similarities = np.clip(query_matrix @ representation_matrix.T, -1.0, 1.0)
        family_scores = np.empty((len(chunk_ids), len(family_ids)), dtype=np.float32)
        best_representation_indices = np.empty_like(family_scores, dtype=np.int64)

        for target_family_index, representation_indices in enumerate(
            representation_indices_by_family
        ):
            local_scores = similarities[:, representation_indices]
            local_best = np.argmax(local_scores, axis=1)
            family_scores[:, target_family_index] = local_scores[
                np.arange(len(chunk_ids)), local_best
            ]
            best_representation_indices[:, target_family_index] = representation_indices[
                local_best
            ]

        for query_index, unit_id in enumerate(chunk_ids):
            ranked_family_indices = sorted(
                range(len(family_ids)),
                key=lambda index: (
                    -float(family_scores[query_index, index]),
                    natural_sort_key(family_ids[index]),
                ),
            )[:top_k]
            candidates: list[dict[str, Any]] = []
            for rank, target_index in enumerate(ranked_family_indices, start=1):
                representation = representations[
                    int(best_representation_indices[query_index, target_index])
                ]
                base_representation_index = int(
                    representation_indices_by_family[target_index][0]
                )
                variant_indices = representation_indices_by_family[target_index][1:]
                best_variant_score = (
                    round_score(np.max(similarities[query_index, variant_indices]))
                    if len(variant_indices)
                    else None
                )
                candidates.append(
                    {
                        "rank": rank,
                        "family_id": family_ids[target_index],
                        "family_score": round_score(
                            family_scores[query_index, target_index]
                        ),
                        "base_unit_score": round_score(
                            similarities[query_index, base_representation_index]
                        ),
                        "best_variant_score": best_variant_score,
                        "matched_representation": {
                            "type": representation["representation_type"],
                            "member_id": representation["member_id"],
                            "variant_group_index": representation[
                                "variant_group_index"
                            ],
                        },
                    }
                )
            top_one_scores.append(candidates[0]["family_score"])
            output.append(
                {
                    "unit_id": unit_id,
                    "recall_k": top_k,
                    "candidates": candidates,
                }
            )

    statistics = {
        "query_count": len(query_ids),
        "family_count": len(family_ids),
        "base_representation_count": len(bases),
        "variant_representation_count": len(representations) - len(bases),
        "total_representation_count": len(representations),
        "candidate_entry_count": len(query_ids) * top_k,
        "top_1_family_score_distribution": score_distribution(top_one_scores),
    }
    return output, statistics


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            json.dump(row, file, ensure_ascii=False)
            file.write("\n")
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-embeddings-file", type=Path, default=DEFAULT_BASE_EMBEDDINGS_FILE
    )
    parser.add_argument(
        "--member-mapping-file", type=Path, default=DEFAULT_MEMBER_MAPPING_FILE
    )
    parser.add_argument(
        "--source-embedding-file",
        type=Path,
        action="append",
        dest="source_embedding_files",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--query-id", action="append", dest="query_ids")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive.")
    base_path = resolve_path(args.base_embeddings_file)
    mapping_path = resolve_path(args.member_mapping_file)
    source_paths = [
        resolve_path(path)
        for path in (args.source_embedding_files or DEFAULT_SOURCE_EMBEDDING_FILES)
    ]
    output_dir = resolve_path(args.output_dir)
    for path in (base_path, mapping_path, *source_paths):
        if not path.is_file():
            raise FileNotFoundError(f"Required input does not exist: {path}")

    bases = load_base_embeddings(base_path)
    source = load_source_embeddings(source_paths)
    mappings = load_member_mapping(mapping_path, {row["family_id"] for row in bases})
    model, dimensions = validate_embedding_configuration(bases, source)
    mapped_ids = {row["member_id"] for row in mappings}
    missing_mapped = sorted(mapped_ids - set(source))
    if missing_mapped:
        raise ValueError(f"Mapped members lack embeddings: {missing_mapped[:20]}")
    all_query_ids = sorted(set(source) - mapped_ids, key=natural_sort_key)
    if args.query_ids:
        requested = set(args.query_ids)
        invalid = sorted(requested - set(all_query_ids), key=natural_sort_key)
        if invalid:
            raise ValueError(f"Requested IDs are not unassigned source units: {invalid}")
        query_ids = [unit_id for unit_id in all_query_ids if unit_id in requested]
    else:
        query_ids = all_query_ids
    if args.limit is not None:
        query_ids = query_ids[: args.limit]
    if not query_ids:
        raise ValueError("No unassigned source units were selected.")

    records, statistics = build_recall_records(
        bases=bases,
        mappings=mappings,
        source=source,
        query_ids=query_ids,
        top_k=args.top_k,
        chunk_size=args.chunk_size,
    )
    output_path = output_dir / f"instance_to_generalization_recall_k{args.top_k}.jsonl"
    summary_path = output_dir / f"instance_to_generalization_recall_k{args.top_k}_summary.json"
    write_jsonl(output_path, records)
    write_json(
        summary_path,
        {
            "base_embeddings_file": str(base_path),
            "base_embeddings_sha256": file_sha256(base_path),
            "member_mapping_file": str(mapping_path),
            "member_mapping_sha256": file_sha256(mapping_path),
            "source_embedding_files": [
                {"path": str(path), "sha256": file_sha256(path)}
                for path in source_paths
            ],
            "output_file": str(output_path),
            "model": model,
            "dimensions": dimensions,
            "embedding_fields": EXPECTED_EMBEDDING_FIELDS,
            "family_score_definition": (
                "max(base_unit_score, variant_member_score_1, ...)"
            ),
            "top_k": args.top_k,
            **statistics,
        },
    )
    print(
        f"Done. queries={len(query_ids)}, families={len(bases)}, top_k={args.top_k}, "
        f"representations={statistics['total_representation_count']}, "
        f"output={output_path}"
    )


if __name__ == "__main__":
    main()

