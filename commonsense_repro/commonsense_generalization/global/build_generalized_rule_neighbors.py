"""Recall nearest commonsense generalization records from separate base-unit embeddings."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
from commonsense_repro.common.paths import PACKAGE_ROOT as PROJECT_ROOT
DEFAULT_DATA_DIR = Path(
    "work/stage4"
)
DEFAULT_INPUT_FILE = DEFAULT_DATA_DIR / "generalized_rule_embeddings.jsonl"
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_DIR / "recall"

DEFAULT_TOP_K = 20
DEFAULT_SITUATION_WEIGHT = 0.35
DEFAULT_RULE_WEIGHT = 0.65


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


def require_nonempty_string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} must be a non-empty string.")
    return value.strip()


def require_vector(value: Any, location: str) -> list[float]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{location} must be a non-empty array.")
    if not all(isinstance(item, (int, float)) for item in value):
        raise ValueError(f"{location} must contain only numbers.")
    vector = [float(item) for item in value]
    if not np.isfinite(vector).all():
        raise ValueError(f"{location} contains a non-finite value.")
    if np.linalg.norm(vector) == 0:
        raise ValueError(f"{location} must not be a zero vector.")
    return vector


def load_embedding_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    models: set[str] = set()
    dimensions: set[int] = set()
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            location = f"{path}:{line_number}"
            if not isinstance(item, dict):
                raise ValueError(f"{location} must contain a JSON object.")
            family_id = require_nonempty_string(
                item.get("family_id"), f"{location}.family_id"
            )
            if family_id in seen_ids:
                raise ValueError(f"Duplicate family_id: {family_id}")
            model = require_nonempty_string(item.get("model"), f"{location}.model")
            situation_embedding = require_vector(
                item.get("situation_embedding"),
                f"{location}.situation_embedding",
            )
            rule_embedding = require_vector(
                item.get("commonsense_rule_embedding"),
                f"{location}.commonsense_rule_embedding",
            )
            if len(situation_embedding) != len(rule_embedding):
                raise ValueError(f"Embedding dimensions differ for {family_id}.")
            declared_dimensions = item.get("dimensions")
            if declared_dimensions != len(situation_embedding):
                raise ValueError(f"Incorrect dimensions field for {family_id}.")
            records.append(
                {
                    "family_id": family_id,
                    "model": model,
                    "dimensions": declared_dimensions,
                    "situation_embedding": situation_embedding,
                    "commonsense_rule_embedding": rule_embedding,
                }
            )
            seen_ids.add(family_id)
            models.add(model)
            dimensions.add(declared_dimensions)

    if len(records) < 2:
        raise ValueError("At least two embedding records are required.")
    if len(models) != 1:
        raise ValueError(f"Input contains multiple embedding models: {sorted(models)}")
    if len(dimensions) != 1:
        raise ValueError(
            f"Input contains multiple embedding dimensions: {sorted(dimensions)}"
        )
    records.sort(key=lambda item: natural_sort_key(item["family_id"]))
    return records


def normalize_rows(matrix: np.ndarray, field_name: str) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError(f"{field_name} contains a zero vector.")
    return matrix / norms


def round_score(value: float) -> float:
    return round(float(value), 8)


def score_distribution(values: np.ndarray) -> dict[str, float]:
    return {
        "min": round_score(np.min(values)),
        "p10": round_score(np.quantile(values, 0.10)),
        "p25": round_score(np.quantile(values, 0.25)),
        "median": round_score(np.quantile(values, 0.50)),
        "mean": round_score(np.mean(values)),
        "p75": round_score(np.quantile(values, 0.75)),
        "p90": round_score(np.quantile(values, 0.90)),
        "p95": round_score(np.quantile(values, 0.95)),
        "p99": round_score(np.quantile(values, 0.99)),
        "max": round_score(np.max(values)),
    }


def build_neighbor_records(
    records: list[dict[str, Any]],
    *,
    top_k: int,
    situation_weight: float,
    rule_weight: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sample_count = len(records)
    if not 1 <= top_k < sample_count:
        raise ValueError(f"top_k must be between 1 and {sample_count - 1}.")
    if situation_weight < 0 or rule_weight < 0:
        raise ValueError("Similarity weights must be non-negative.")
    if not np.isclose(situation_weight + rule_weight, 1.0, atol=1e-9):
        raise ValueError("situation_weight and rule_weight must sum to 1.0.")

    situation_matrix = normalize_rows(
        np.asarray(
            [record["situation_embedding"] for record in records],
            dtype=np.float32,
        ),
        "situation_embedding",
    )
    rule_matrix = normalize_rows(
        np.asarray(
            [record["commonsense_rule_embedding"] for record in records],
            dtype=np.float32,
        ),
        "commonsense_rule_embedding",
    )
    situation_similarity = np.clip(
        situation_matrix @ situation_matrix.T, -1.0, 1.0
    )
    rule_similarity = np.clip(rule_matrix @ rule_matrix.T, -1.0, 1.0)
    combined_similarity = (
        situation_weight * situation_similarity
        + rule_weight * rule_similarity
    )

    ranked_similarity = combined_similarity.copy()
    np.fill_diagonal(ranked_similarity, -np.inf)
    neighbor_records: list[dict[str, Any]] = []
    for source_index, source in enumerate(records):
        # Stable sorting preserves natural family-ID order when scores tie.
        neighbor_indices = np.argsort(
            -ranked_similarity[source_index], kind="stable"
        )[:top_k]
        neighbors = []
        for rank, target_index in enumerate(neighbor_indices, start=1):
            neighbors.append(
                {
                    "rank": rank,
                    "family_id": records[int(target_index)]["family_id"],
                    "situation_similarity": round_score(
                        situation_similarity[source_index, target_index]
                    ),
                    "commonsense_rule_similarity": round_score(
                        rule_similarity[source_index, target_index]
                    ),
                    "combined_similarity": round_score(
                        combined_similarity[source_index, target_index]
                    ),
                }
            )
        neighbor_records.append(
            {
                "family_id": source["family_id"],
                "neighbors": neighbors,
            }
        )

    upper_triangle = np.triu_indices(sample_count, k=1)
    statistics = {
        "unordered_pair_count": sample_count * (sample_count - 1) // 2,
        "neighbor_entry_count": sample_count * top_k,
        "all_pair_score_distributions": {
            "situation_similarity": score_distribution(
                situation_similarity[upper_triangle]
            ),
            "commonsense_rule_similarity": score_distribution(
                rule_similarity[upper_triangle]
            ),
            "combined_similarity": score_distribution(
                combined_similarity[upper_triangle]
            ),
        },
    }
    return neighbor_records, statistics


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            json.dump(record, file, ensure_ascii=False)
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
    parser = argparse.ArgumentParser(
        description="Recall Top-K commonsense generalization records from dual base-unit embeddings."
    )
    parser.add_argument("--input-file", type=Path, default=DEFAULT_INPUT_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--situation-weight", type=float, default=DEFAULT_SITUATION_WEIGHT
    )
    parser.add_argument("--rule-weight", type=float, default=DEFAULT_RULE_WEIGHT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = resolve_path(args.input_file)
    output_dir = resolve_path(args.output_dir)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    records = load_embedding_records(input_path)
    neighbors, statistics = build_neighbor_records(
        records,
        top_k=args.top_k,
        situation_weight=args.situation_weight,
        rule_weight=args.rule_weight,
    )
    output_path = output_dir / f"generalized_rule_neighbors_k{args.top_k}.jsonl"
    summary_path = output_dir / f"generalized_rule_neighbors_k{args.top_k}_summary.json"
    summary = {
        "input_file": str(args.input_file),
        "input_sha256": file_sha256(input_path),
        "output_file": str(output_path.relative_to(PROJECT_ROOT)),
        "model": records[0]["model"],
        "dimensions": records[0]["dimensions"],
        "family_count": len(records),
        "top_k": args.top_k,
        "situation_weight": args.situation_weight,
        "commonsense_rule_weight": args.rule_weight,
        **statistics,
    }
    write_jsonl(output_path, neighbors)
    write_json(summary_path, summary)
    print(
        "Done. "
        f"families={len(records)}, top_k={args.top_k}, "
        f"neighbor_entries={statistics['neighbor_entry_count']}, "
        f"output={output_path}"
    )


if __name__ == "__main__":
    main()

