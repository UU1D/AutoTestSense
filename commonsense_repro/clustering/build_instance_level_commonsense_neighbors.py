"""Build exact cosine Top-K neighbor files from precomputed embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
EMBEDDINGS_FILES = [
    Path("work/stage2/embeddings/gemini_v1_2_extract_embeddings.jsonl"),
    Path("work/stage2/embeddings/gemini_v1_2_github_embeddings.jsonl"),
    Path("work/stage2/embeddings/gemini_v1_2_negative_embeddings.jsonl"),
]
OUTPUT_DIR = Path("work/stage2/neighbors")
K_VALUES = [30, 60]

SUMMARY_FILENAME = "embedding_neighbors_summary.json"


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


def load_embeddings(
    paths: list[Path],
) -> tuple[list[str], np.ndarray, str, list[str], dict[str, int]]:
    records_with_sources = [
        (record, path)
        for path in paths
        for record in read_jsonl(path)
    ]
    if len(records_with_sources) < 2:
        raise ValueError("At least two embedding records are required.")

    ids: list[str] = []
    vectors: list[list[float]] = []
    seen_ids: set[str] = set()
    dimensions: set[int] = set()
    models: set[str] = set()
    field_sets: set[tuple[str, ...]] = set()
    source_sample_counts = {str(path): 0 for path in paths}

    for index, (record, source_path) in enumerate(records_with_sources):
        unique_id = record.get("id")
        vector = record.get("embedding")
        if not isinstance(unique_id, str) or not unique_id.strip():
            raise ValueError(
                f"Missing id in embedding record {index} from {source_path}."
            )
        if unique_id in seen_ids:
            raise ValueError(
                f"Duplicate embedding id across input files: {unique_id} "
                f"(found again in {source_path})"
            )
        if not isinstance(vector, list) or not vector:
            raise ValueError(f"Missing embedding vector for {unique_id}.")

        seen_ids.add(unique_id)
        ids.append(unique_id)
        vectors.append(vector)
        dimensions.add(len(vector))
        models.add(str(record.get("model", "")))
        fields = record.get("embedding_fields")
        field_sets.add(tuple(fields) if isinstance(fields, list) else tuple())
        source_sample_counts[str(source_path)] += 1

    if len(dimensions) != 1:
        raise ValueError(f"Embedding dimensions are inconsistent: {dimensions}")
    if len(models) != 1:
        raise ValueError(f"Embedding models are inconsistent: {models}")
    if len(field_sets) != 1:
        raise ValueError("Embedding field configurations are inconsistent.")

    matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        zero_ids = [ids[index] for index in np.flatnonzero(norms[:, 0] == 0)]
        raise ValueError(f"Zero-length embedding vectors: {zero_ids}")

    return (
        ids,
        matrix / norms,
        next(iter(models)),
        list(next(iter(field_sets))),
        source_sample_counts,
    )


def compute_top_neighbors(
    matrix: np.ndarray,
    max_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    similarities = matrix @ matrix.T
    np.fill_diagonal(similarities, -np.inf)

    indices = np.argpartition(-similarities, kth=max_k - 1, axis=1)[:, :max_k]
    scores = np.take_along_axis(similarities, indices, axis=1)
    order = np.argsort(-scores, axis=1)
    return (
        np.take_along_axis(indices, order, axis=1),
        np.take_along_axis(scores, order, axis=1),
    )


def build_neighbor_records(
    ids: list[str],
    neighbor_indices: np.ndarray,
    neighbor_scores: np.ndarray,
    k: int,
) -> Iterable[dict[str, Any]]:
    for row, unique_id in enumerate(ids):
        yield {
            "id": unique_id,
            "neighbors": [
                {
                    "rank": rank,
                    "id": ids[int(neighbor_index)],
                    "cosine_similarity": float(score),
                }
                for rank, (neighbor_index, score) in enumerate(
                    zip(neighbor_indices[row, :k], neighbor_scores[row, :k]),
                    start=1,
                )
            ],
        }


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        for record in records:
            json.dump(record, file, ensure_ascii=False)
            file.write("\n")
    temporary_path.replace(path)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")
    temporary_path.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build exact cosine Top-K neighbors from embeddings."
    )
    parser.add_argument(
        "--embeddings-files",
        "--embeddings-file",
        dest="embeddings_files",
        type=Path,
        nargs="+",
        default=EMBEDDINGS_FILES,
        help=(
            "One or more embedding JSONL files. All records are combined before "
            "the shared Top-K neighbor search."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--k-values", type=int, nargs="+", default=K_VALUES)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.embeddings_files:
        raise ValueError("At least one embedding file is required.")
    missing_files = [path for path in args.embeddings_files if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(f"Embedding file does not exist: {missing_files[0]}")
    if len(args.embeddings_files) != len(set(args.embeddings_files)):
        raise ValueError("Embedding file arguments contain duplicate paths.")

    k_values = sorted(set(args.k_values))
    if not k_values or k_values[0] < 1:
        raise ValueError("k-values must contain positive integers.")

    (
        ids,
        matrix,
        embedding_model,
        embedding_fields,
        source_sample_counts,
    ) = load_embeddings(args.embeddings_files)
    if k_values[-1] >= len(ids):
        raise ValueError(f"Every k must be less than sample_count={len(ids)}.")

    output_files = {
        k: args.output_dir / f"embedding_neighbors_k{k}.jsonl" for k in k_values
    }
    summary_path = args.output_dir / SUMMARY_FILENAME
    existing = [path for path in [*output_files.values(), summary_path] if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {existing[0]}. Use --overwrite to rebuild."
        )

    neighbor_indices, neighbor_scores = compute_top_neighbors(matrix, k_values[-1])
    for k, output_path in output_files.items():
        write_jsonl(
            output_path,
            build_neighbor_records(ids, neighbor_indices, neighbor_scores, k),
        )
        print(f"Wrote k={k}: {output_path}")

    summary = {
        "embeddings_files": [str(path) for path in args.embeddings_files],
        "source_sample_counts": source_sample_counts,
        "output_files": {str(k): str(path) for k, path in output_files.items()},
        "sample_count": len(ids),
        "embedding_model": embedding_model,
        "embedding_fields": embedding_fields,
        "embedding_dimensions": int(matrix.shape[1]),
        "metric": "cosine_similarity",
        "k_values": k_values,
    }
    write_json(summary_path, summary)
    print(f"Done. samples={len(ids)}, k_values={k_values}, output={args.output_dir}")


if __name__ == "__main__":
    main()

