"""Ablation: retrieve dataset situations from 3,708 instance-level commonsense items."""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
from dotenv import load_dotenv


# -----------------------------------------------------------------------------
# Editable configuration
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
from commonsense_repro.common.paths import PACKAGE_ROOT as PROJECT_ROOT
load_dotenv(PROJECT_ROOT / ".env")

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "").strip()
DASHSCOPE_URL = os.getenv("DASHSCOPE_URL", "").strip()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "").strip()
DASHSCOPE_WORKSPACE_ID = os.getenv("DASHSCOPE_WORKSPACE_ID", "").strip()
RERANK_MODEL = os.getenv("RERANK_MODEL", "").strip()
RERANK_URL = (
    f"https://{DASHSCOPE_WORKSPACE_ID}.cn-beijing.maas.aliyuncs.com/"
    "compatible-api/v1/reranks"
)

DEFAULT_DATASET_DIR = SCRIPT_DIR
DEFAULT_INPUT_SUBDIR = "all_situation"
DEFAULT_OUTPUT_SUBDIR = "retrieval_output_instance_level"
EMBEDDING_FILES = [
    PROJECT_ROOT
    / "data/reference/instance_level_commonsense_situation_embeddings/gemini_v1_2_extract_embeddings.jsonl",
    PROJECT_ROOT
    / "data/reference/instance_level_commonsense_situation_embeddings/gemini_v1_2_github_embeddings.jsonl",
    PROJECT_ROOT
    / "data/reference/instance_level_commonsense_situation_embeddings/gemini_v1_2_negative_embeddings.jsonl",
]
METADATA_FILES = [
    PROJECT_ROOT / "data/checkpoints/instance_level_commonsense/gemini_v1_2_extract.json",
    PROJECT_ROOT / "data/checkpoints/instance_level_commonsense/gemini_v1_2_github.json",
    PROJECT_ROOT / "data/checkpoints/instance_level_commonsense/gemini_v1_2_negative.json",
]

SITUATIONS_FIELD = "test_situations"
DESCRIPTION_FIELD = "situation_description"
RESULT_FIELD = "related_commonsense"
EXPECTED_EMBEDDING_FIELDS = ["situation"]
QUERY_FIELD_LABEL = "Situation"

DEFAULT_RECALL_K = 60
DEFAULT_RERANK_TOP_K = 40
DEFAULT_EMBEDDING_BATCH_SIZE = 10
EXPECTED_UNIT_COUNT = 3708
REQUEST_TIMEOUT = 180
MAX_RETRIES = 5

RERANK_FIELDS = [
    "situation",
]
FIELD_LABELS = {
    "situation": "Situation",
}
RERANK_INSTRUCT = (
    "Given a short GUI operation situation, rank candidate situations by how "
    "directly they describe the same abstract and recurring operation, trigger, "
    "and UI state relationship. Focus only on situation similarity. Do not "
    "consider whether their common-sense rules could be merged, and do not rank "
    "a candidate highly based only on shared software topics or keywords."
)


@dataclass
class PendingSituation:
    source_name: str
    output_path: Path
    output_document: dict[str, Any]
    situation_index: int
    situation: dict[str, Any]
    description: str


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")
    temporary_path.replace(path)


def resolve_embedding_endpoint(base_url: str) -> str:
    endpoint = base_url.rstrip("/")
    if endpoint.endswith("/embeddings"):
        return endpoint
    return f"{endpoint}/embeddings"


def post_json_once(
    *,
    url: str,
    payload: dict[str, Any],
    timeout: int,
) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    if not isinstance(body, dict):
        raise ValueError("API response must be a JSON object.")
    return body


def post_json_with_retries(
    *,
    url: str,
    payload: dict[str, Any],
    operation: str,
) -> dict[str, Any]:
    for attempt in range(MAX_RETRIES + 1):
        try:
            return post_json_once(
                url=url,
                payload=payload,
                timeout=REQUEST_TIMEOUT,
            )
        except HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            retryable = error.code == 429 or error.code >= 500
            if not retryable or attempt == MAX_RETRIES:
                raise RuntimeError(
                    f"{operation} HTTP error: status={error.code}, body={body}"
                ) from error
            description = f"HTTP {error.code}: {body}"
        except (URLError, TimeoutError) as error:
            if attempt == MAX_RETRIES:
                raise RuntimeError(f"{operation} request failed: {error}") from error
            description = str(error)

        delay = 2**attempt
        print(
            f"{operation} failed; retrying in {delay}s "
            f"({attempt + 1}/{MAX_RETRIES}): {description}"
        )
        time.sleep(delay)

    raise RuntimeError(f"{operation} request failed after retries.")


def load_stored_embeddings(
    paths: list[Path],
) -> tuple[list[str], np.ndarray, str, list[str]]:
    missing_paths = [path for path in paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(f"Missing embedding files: {missing_paths}")

    ids: list[str] = []
    vectors: list[list[float]] = []
    seen_ids: set[str] = set()
    models: set[str] = set()
    dimensions: set[int] = set()
    field_sets: set[tuple[str, ...]] = set()

    for path in paths:
        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"Expected an object at {path}:{line_number}.")

                unique_id = record.get("id")
                vector = record.get("embedding")
                model = record.get("model")
                fields = record.get("embedding_fields")
                if not isinstance(unique_id, str) or not unique_id:
                    raise ValueError(f"Missing id at {path}:{line_number}.")
                if unique_id in seen_ids:
                    raise ValueError(f"Duplicate embedding id: {unique_id}")
                if not isinstance(vector, list) or not vector:
                    raise ValueError(f"Missing embedding at {path}:{line_number}.")
                if not isinstance(model, str) or not model:
                    raise ValueError(f"Missing model at {path}:{line_number}.")

                seen_ids.add(unique_id)
                ids.append(unique_id)
                vectors.append(vector)
                models.add(model)
                dimensions.add(len(vector))
                field_sets.add(tuple(fields) if isinstance(fields, list) else tuple())

    if len(models) != 1:
        raise ValueError(f"Embedding models are inconsistent: {sorted(models)}")
    if len(dimensions) != 1:
        raise ValueError(f"Embedding dimensions are inconsistent: {dimensions}")
    if len(field_sets) != 1:
        raise ValueError(f"Embedding fields are inconsistent: {field_sets}")

    matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Stored embeddings contain zero-length vectors.")

    return ids, matrix / norms, next(iter(models)), list(next(iter(field_sets)))


def load_metadata(paths: list[Path]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in paths:
        data = read_json(path)
        if not isinstance(data, list):
            raise ValueError(f"{path} must contain a JSON array.")
        for index, record in enumerate(data):
            if not isinstance(record, dict):
                raise ValueError(f"{path}[{index}] must be a JSON object.")
            unique_id = record.get("id")
            if not isinstance(unique_id, str) or not unique_id.strip():
                raise ValueError(f"{path}[{index}] has no valid id.")
            if unique_id in records:
                raise ValueError(f"Duplicate metadata id: {unique_id}")
            records[unique_id] = record
    return records


def build_query_text(description: str) -> str:
    prefix = f"{QUERY_FIELD_LABEL}:"
    if description.casefold().startswith(prefix.casefold()):
        return description
    return f"{prefix} {description}"


def build_rerank_document(record: dict[str, Any]) -> str:
    parts: list[str] = []
    for field in RERANK_FIELDS:
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            parts.append(f"{FIELD_LABELS[field]}: {value.strip()}")
    if not parts:
        raise ValueError(f"Record {record.get('id')} has no rerank text.")
    return "\n".join(parts)


def iter_batches(items: list[Any], size: int) -> Iterator[list[Any]]:
    for index in range(0, len(items), size):
        yield items[index : index + size]


def request_query_embeddings(texts: list[str]) -> list[np.ndarray]:
    response = post_json_with_retries(
        url=resolve_embedding_endpoint(DASHSCOPE_URL),
        payload={"model": EMBEDDING_MODEL, "input": texts},
        operation="Embedding",
    )
    data = response.get("data")
    if not isinstance(data, list):
        raise ValueError("Embedding response has no data list.")

    ordered = sorted(data, key=lambda item: item.get("index", 0))
    vectors: list[np.ndarray] = []
    for item in ordered:
        vector = item.get("embedding") if isinstance(item, dict) else None
        if not isinstance(vector, list) or not vector:
            raise ValueError("Embedding response item has no vector.")
        vectors.append(np.asarray(vector, dtype=np.float32))
    if len(vectors) != len(texts):
        raise ValueError(f"Expected {len(texts)} embeddings, got {len(vectors)}.")
    return vectors


def retrieve_top_k(
    query_vector: np.ndarray,
    matrix: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    if query_vector.ndim != 1 or query_vector.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"Query dimension {query_vector.shape} does not match "
            f"stored dimension {matrix.shape[1]}."
        )
    norm = np.linalg.norm(query_vector)
    if norm == 0:
        raise ValueError("Query embedding has zero length.")

    similarities = matrix @ (query_vector / norm)
    candidate_indices = np.argpartition(-similarities, kth=top_k - 1)[:top_k]
    order = np.argsort(-similarities[candidate_indices])
    indices = candidate_indices[order]
    return indices, similarities[indices]


def request_rerank(
    *,
    query: str,
    documents: list[str],
    top_n: int,
) -> dict[str, Any]:
    return post_json_with_retries(
        url=RERANK_URL,
        payload={
            "model": RERANK_MODEL,
            "query": query,
            "documents": documents,
            "top_n": top_n,
            "instruct": RERANK_INSTRUCT,
        },
        operation="Rerank",
    )


def parse_rerank_results(
    response: dict[str, Any],
    *,
    candidate_count: int,
    expected_count: int,
) -> list[tuple[int, float]]:
    results = response.get("results")
    if not isinstance(results, list):
        raise ValueError(
            "Rerank response has no results list: "
            f"code={response.get('code')!r}, message={response.get('message')!r}"
        )
    if len(results) != expected_count:
        raise ValueError(
            f"Expected {expected_count} rerank results, got {len(results)}."
        )

    parsed: list[tuple[int, float]] = []
    seen_indices: set[int] = set()
    for result in results:
        if not isinstance(result, dict):
            raise ValueError("A rerank result is not an object.")
        index = result.get("index")
        score = result.get("relevance_score")
        if not isinstance(index, int) or not 0 <= index < candidate_count:
            raise ValueError(f"Invalid rerank result index: {index!r}")
        if index in seen_indices:
            raise ValueError(f"Duplicate rerank result index: {index}")
        if not isinstance(score, (int, float)):
            raise ValueError(f"Invalid rerank score at index {index}: {score!r}")
        seen_indices.add(index)
        parsed.append((index, float(score)))

    return sorted(parsed, key=lambda item: (-item[1], item[0]))


def validate_input_document(path: Path, data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    situations = data.get(SITUATIONS_FIELD)
    if not isinstance(situations, list):
        raise ValueError(f"{path} has no {SITUATIONS_FIELD!r} list.")
    for index, situation in enumerate(situations):
        if not isinstance(situation, dict):
            raise ValueError(f"{path} {SITUATIONS_FIELD}[{index}] is not an object.")
        description = situation.get(DESCRIPTION_FIELD)
        if not isinstance(description, str) or not description.strip():
            raise ValueError(
                f"{path} {SITUATIONS_FIELD}[{index}] has no valid "
                f"{DESCRIPTION_FIELD!r}."
            )
        if RESULT_FIELD in situation:
            raise ValueError(
                f"Input {path} already contains reserved field {RESULT_FIELD!r}."
            )
    return data


def has_complete_results(value: Any, expected_count: int) -> bool:
    if not isinstance(value, list) or len(value) != expected_count:
        return False
    seen_ids: set[str] = set()
    for item in value:
        if not isinstance(item, dict) or set(item) != {
            "id",
            "cosine_score",
            "recall_rank",
            "rerank_score",
            "situation",
        }:
            return False
        unique_id = item.get("id")
        if (
            not isinstance(unique_id, str)
            or not unique_id
            or unique_id in seen_ids
            or not isinstance(item.get("cosine_score"), (int, float))
            or not isinstance(item.get("recall_rank"), int)
            or not isinstance(item.get("rerank_score"), (int, float))
            or not isinstance(item.get("situation"), str)
        ):
            return False
        seen_ids.add(unique_id)
    return True


def merge_existing_results(
    *,
    input_document: dict[str, Any],
    existing_output: Any,
    output_path: Path,
    rerank_top_k: int,
) -> tuple[dict[str, Any], int]:
    if not isinstance(existing_output, dict):
        raise ValueError(f"Existing output {output_path} is not an object.")
    existing_situations = existing_output.get(SITUATIONS_FIELD)
    if not isinstance(existing_situations, list):
        raise ValueError(
            f"Existing output {output_path} has no {SITUATIONS_FIELD!r} list."
        )

    merged = copy.deepcopy(input_document)
    invalidated_count = 0
    for index, input_situation in enumerate(merged[SITUATIONS_FIELD]):
        if index >= len(existing_situations):
            continue
        existing_situation = existing_situations[index]
        if not isinstance(existing_situation, dict):
            continue

        existing_base = copy.deepcopy(existing_situation)
        existing_results = existing_base.pop(RESULT_FIELD, None)
        if existing_base == input_situation and has_complete_results(
            existing_results,
            rerank_top_k,
        ):
            input_situation[RESULT_FIELD] = copy.deepcopy(existing_results)
        elif has_complete_results(existing_results, rerank_top_k):
            invalidated_count += 1

    return merged, invalidated_count


def prepare_pending_situations(
    *,
    input_dir: Path,
    output_dir: Path,
    rerank_top_k: int,
    overwrite: bool,
    limit: int | None,
) -> tuple[list[PendingSituation], int]:
    input_paths = sorted(path for path in input_dir.glob("*.json") if path.is_file())
    if not input_paths:
        raise FileNotFoundError(f"No JSON files found in {input_dir}.")

    pending: list[PendingSituation] = []
    completed_count = 0
    output_dir.mkdir(parents=True, exist_ok=True)

    for input_path in input_paths:
        if limit is not None and len(pending) >= limit:
            break
        input_document = validate_input_document(input_path, read_json(input_path))
        output_path = output_dir / input_path.name
        if output_path.exists() and not overwrite:
            existing_output = read_json(output_path)
            output_document, invalidated_count = merge_existing_results(
                input_document=input_document,
                existing_output=existing_output,
                output_path=output_path,
                rerank_top_k=rerank_top_k,
            )
            if output_document != existing_output:
                write_json_atomic(output_path, output_document)
            if invalidated_count:
                print(
                    f"Input changed for {input_path.name}; invalidated "
                    f"{invalidated_count} cached situation(s)."
                )
        else:
            output_document = copy.deepcopy(input_document)
            if overwrite:
                write_json_atomic(output_path, output_document)

        situations = output_document[SITUATIONS_FIELD]
        for index, situation in enumerate(situations):
            if has_complete_results(situation.get(RESULT_FIELD), rerank_top_k):
                completed_count += 1
                continue
            if limit is not None and len(pending) >= limit:
                break
            description = str(situation[DESCRIPTION_FIELD]).strip()
            pending.append(
                PendingSituation(
                    source_name=input_path.name,
                    output_path=output_path,
                    output_document=output_document,
                    situation_index=index,
                    situation=situation,
                    description=description,
                )
            )

    return pending, completed_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ablation retrieval over 3,708 instance-level commonsense items for "
            "a dataset's situation_description fields."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help=(
            "Dataset directory containing all_situation. Relative paths are "
            "resolved from the project root."
        ),
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Optional explicit input directory; overrides dataset-dir/all_situation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Optional explicit output directory; overrides "
            "dataset-dir/retrieval_output_instance_level."
        ),
    )
    parser.add_argument(
        "--embeddings-files",
        type=Path,
        nargs="+",
        default=EMBEDDING_FILES,
        help="Situation-only embedding JSONL files for the instance-level commonsense items.",
    )
    parser.add_argument(
        "--metadata-files",
        type=Path,
        nargs="+",
        default=METADATA_FILES,
        help="Instance-Level Commonsense item JSON files.",
    )
    parser.add_argument("--recall-k", type=int, default=DEFAULT_RECALL_K)
    parser.add_argument(
        "--rerank-top-k",
        type=int,
        default=DEFAULT_RERANK_TOP_K,
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=DEFAULT_EMBEDDING_BATCH_SIZE,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N pending situations for testing.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Discard existing enriched outputs and start again.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.recall_k < 1:
        raise ValueError("--recall-k must be positive.")
    if args.rerank_top_k < 1 or args.rerank_top_k > args.recall_k:
        raise ValueError("--rerank-top-k must be between 1 and recall-k.")
    if not 1 <= args.embedding_batch_size <= 10:
        raise ValueError("--embedding-batch-size must be between 1 and 10.")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive.")


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def resolve_dataset_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    dataset_dir = resolve_path(args.dataset_dir)
    input_dir = (
        resolve_path(args.input_dir)
        if args.input_dir is not None
        else dataset_dir / DEFAULT_INPUT_SUBDIR
    )
    output_dir = (
        resolve_path(args.output_dir)
        if args.output_dir is not None
        else dataset_dir / DEFAULT_OUTPUT_SUBDIR
    )
    return dataset_dir, input_dir, output_dir


def main() -> None:
    args = parse_args()
    validate_args(args)

    missing = [
        name
        for name, value in [
            ("DASHSCOPE_API_KEY", DASHSCOPE_API_KEY),
            ("DASHSCOPE_URL", DASHSCOPE_URL),
            ("EMBEDDING_MODEL", EMBEDDING_MODEL),
            ("DASHSCOPE_WORKSPACE_ID", DASHSCOPE_WORKSPACE_ID),
            ("RERANK_MODEL", RERANK_MODEL),
        ]
        if not value
    ]
    if missing:
        raise ValueError(f"Missing .env values: {', '.join(missing)}")

    dataset_dir, input_dir, output_dir = resolve_dataset_paths(args)
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    embedding_files = [resolve_path(path) for path in args.embeddings_files]
    metadata_files = [resolve_path(path) for path in args.metadata_files]
    ids, matrix, stored_model, stored_fields = load_stored_embeddings(embedding_files)
    if len(ids) != EXPECTED_UNIT_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_UNIT_COUNT} instance-level commonsense embeddings, got {len(ids)}."
        )
    if args.recall_k > len(ids):
        raise ValueError(f"recall-k exceeds stored record count={len(ids)}.")
    if stored_model != EMBEDDING_MODEL:
        raise ValueError(
            f"Configured embedding model {EMBEDDING_MODEL!r} differs from "
            f"stored model {stored_model!r}."
        )
    if stored_fields != EXPECTED_EMBEDDING_FIELDS:
        raise ValueError(
            f"Stored fields {stored_fields!r} differ from expected fields "
            f"{EXPECTED_EMBEDDING_FIELDS!r}."
        )

    metadata = load_metadata(metadata_files)
    if len(metadata) != EXPECTED_UNIT_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_UNIT_COUNT} metadata records, got {len(metadata)}."
        )
    missing_metadata_ids = sorted(set(ids) - set(metadata))
    missing_embedding_ids = sorted(set(metadata) - set(ids))
    if missing_metadata_ids or missing_embedding_ids:
        raise ValueError(
            "Embedding/metadata IDs differ: "
            f"missing_metadata={missing_metadata_ids[:10]}, "
            f"missing_embeddings={missing_embedding_ids[:10]}."
        )

    pending, completed_count = prepare_pending_situations(
        input_dir=input_dir,
        output_dir=output_dir,
        rerank_top_k=args.rerank_top_k,
        overwrite=args.overwrite,
        limit=args.limit,
    )

    print(
        f"Loaded embeddings={len(ids)}, pending={len(pending)}, "
        f"skipped_completed={completed_count}, recall_k={args.recall_k}, "
        f"rerank_top_k={args.rerank_top_k}."
    )

    processed = 0
    for batch in iter_batches(pending, args.embedding_batch_size):
        query_texts = [build_query_text(item.description) for item in batch]
        query_vectors = request_query_embeddings(query_texts)

        for item, query_text, query_vector in zip(
            batch,
            query_texts,
            query_vectors,
        ):
            recall_indices, recall_scores = retrieve_top_k(
                query_vector,
                matrix,
                args.recall_k,
            )
            candidate_ids = [ids[int(index)] for index in recall_indices]
            candidate_cosine_scores = [float(score) for score in recall_scores]
            documents = [
                build_rerank_document(metadata[unique_id])
                for unique_id in candidate_ids
            ]
            response = request_rerank(
                query=query_text,
                documents=documents,
                top_n=args.rerank_top_k,
            )
            reranked = parse_rerank_results(
                response,
                candidate_count=len(candidate_ids),
                expected_count=args.rerank_top_k,
            )
            related = [
                {
                    "id": candidate_ids[candidate_index],
                    "cosine_score": candidate_cosine_scores[candidate_index],
                    "recall_rank": candidate_index + 1,
                    "rerank_score": rerank_score,
                    "situation": str(
                        metadata[candidate_ids[candidate_index]]["situation"]
                    ),
                }
                for candidate_index, rerank_score in reranked
            ]
            item.situation[RESULT_FIELD] = related
            write_json_atomic(item.output_path, item.output_document)

            processed += 1
            top = related[0]
            print(
                f"[{processed}/{len(pending)}] file={item.source_name} "
                f"situation={item.situation_index + 1} top={top['id']} "
                f"score={top['rerank_score']:.6f}"
            )

    print(
        f"Done. processed={processed}, skipped_completed={completed_count}, "
        f"output_dir={output_dir}"
    )


if __name__ == "__main__":
    main()

