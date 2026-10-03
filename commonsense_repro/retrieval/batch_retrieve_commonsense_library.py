"""Retrieve and rerank final common-sense commonsense generalization records by situation."""

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

DEFAULT_DATASET_DIR = SCRIPT_DIR / "vanillaMLLM_situation"
DEFAULT_INPUT_SUBDIR = "all_situation"
DEFAULT_OUTPUT_SUBDIR = "retrieval_output"
DEFAULT_EMBEDDINGS_FILE = Path(
    "work/stage4/"
    "final_incremental_rule_library/commonsense_library_situation_embeddings.jsonl"
)

SITUATIONS_FIELD = "test_situations"
DESCRIPTION_FIELD = "situation_description"
RESULT_FIELD = "related_rule_families"
CONFIG_FIELD = "rule_family_retrieval"
QUERY_LABEL = "Situation"

DEFAULT_RECALL_K = 60
DEFAULT_RERANK_TOP_K = 40
DEFAULT_EMBEDDING_BATCH_SIZE = 10
REQUEST_TIMEOUT = 180
MAX_RETRIES = 5
WRITE_REPLACE_RETRIES = 8

EXPECTED_EMBEDDING_FIELDS = ["situation"]
RERANK_INSTRUCT = (
    "Given a short GUI operation situation, rank candidate situations by how "
    "directly they describe the same abstract and recurring operation, trigger, "
    "and UI state relationship. Focus only on situation similarity. Do not "
    "consider whether their common-sense rules could be merged, and do not rank "
    "a candidate highly based only on shared software topics or keywords."
)


@dataclass(frozen=True)
class FamilyEmbeddingStore:
    representations: list[dict[str, Any]]
    matrix: np.ndarray
    family_ids: list[str]
    indices_by_family: dict[str, np.ndarray]
    model: str
    dimensions: int


@dataclass
class PendingSituation:
    source_name: str
    output_path: Path
    output_document: dict[str, Any]
    situation_index: int
    situation: dict[str, Any]
    description: str


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def resolve_dataset_paths(
    *,
    dataset_dir: Path,
    input_dir: Path | None,
    output_dir: Path | None,
    default_input_subdir: str,
    default_output_subdir: str,
) -> tuple[Path, Path, Path]:
    resolved_dataset_dir = resolve_path(dataset_dir)
    resolved_input_dir = (
        resolve_path(input_dir)
        if input_dir is not None
        else resolved_dataset_dir / default_input_subdir
    )
    resolved_output_dir = (
        resolve_path(output_dir)
        if output_dir is not None
        else resolved_dataset_dir / default_output_subdir
    )
    return resolved_dataset_dir, resolved_input_dir, resolved_output_dir


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")
    for attempt in range(WRITE_REPLACE_RETRIES + 1):
        try:
            temporary.replace(path)
            return
        except PermissionError as error:
            if attempt == WRITE_REPLACE_RETRIES:
                raise PermissionError(
                    f"Could not replace {path} after "
                    f"{WRITE_REPLACE_RETRIES + 1} attempts. The complete "
                    f"temporary output remains at {temporary}."
                ) from error
            delay = min(0.1 * (2**attempt), 2.0)
            print(
                "Output file is temporarily locked; retrying replace in "
                f"{delay:.1f}s ({attempt + 1}/{WRITE_REPLACE_RETRIES}): {path}"
            )
            time.sleep(delay)


def require_string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} must be a non-empty string.")
    return value.strip()


def resolve_embedding_endpoint(base_url: str) -> str:
    endpoint = base_url.rstrip("/")
    return endpoint if endpoint.endswith("/embeddings") else f"{endpoint}/embeddings"


def post_json_once(*, url: str, payload: dict[str, Any]) -> dict[str, Any]:
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
    with urlopen(request, timeout=REQUEST_TIMEOUT) as response:
        body = json.loads(response.read().decode("utf-8"))
    if not isinstance(body, dict):
        raise ValueError("API response must be a JSON object.")
    return body


def post_json_with_retries(
    *, url: str, payload: dict[str, Any], operation: str
) -> dict[str, Any]:
    for attempt in range(MAX_RETRIES + 1):
        try:
            return post_json_once(url=url, payload=payload)
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


def load_family_embeddings(path: Path) -> FamilyEmbeddingStore:
    representations: list[dict[str, Any]] = []
    vectors: list[list[float]] = []
    indices_by_family_list: dict[str, list[int]] = {}
    seen_representation_ids: set[str] = set()
    models: set[str] = set()
    dimensions: set[int] = set()
    field_sets: set[tuple[str, ...]] = set()

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            location = f"{path}:{line_number}"
            if not isinstance(row, dict):
                raise ValueError(f"{location} must be a JSON object.")

            representation_id = require_string(
                row.get("representation_id"), f"{location}.representation_id"
            )
            if representation_id in seen_representation_ids:
                raise ValueError(f"Duplicate representation_id: {representation_id}")
            seen_representation_ids.add(representation_id)

            family_id = require_string(row.get("family_id"), f"{location}.family_id")
            representation_type = row.get("representation_type")
            if representation_type not in {"BASE", "VARIANT_MEMBER"}:
                raise ValueError(
                    f"{location}.representation_type is invalid: "
                    f"{representation_type!r}"
                )
            member_id = row.get("member_id")
            if representation_type == "VARIANT_MEMBER":
                member_id = require_string(member_id, f"{location}.member_id")
            elif member_id is not None:
                raise ValueError(f"{location}.member_id must be null for a base.")

            situation = require_string(row.get("situation"), f"{location}.situation")
            model = require_string(row.get("model"), f"{location}.model")
            vector = row.get("embedding")
            if not isinstance(vector, list) or not vector:
                raise ValueError(f"{location}.embedding must be a non-empty array.")
            if not all(isinstance(value, (int, float)) for value in vector):
                raise ValueError(f"{location}.embedding contains a non-number.")
            fields = row.get("embedding_fields")
            if not isinstance(fields, list) or not all(
                isinstance(field, str) for field in fields
            ):
                raise ValueError(f"{location}.embedding_fields must be a string array.")

            index = len(representations)
            representations.append(
                {
                    "representation_id": representation_id,
                    "representation_type": representation_type,
                    "family_id": family_id,
                    "member_id": member_id,
                    "variant_group_index": row.get("variant_group_index"),
                    "situation": situation,
                }
            )
            vectors.append([float(value) for value in vector])
            indices_by_family_list.setdefault(family_id, []).append(index)
            models.add(model)
            dimensions.add(len(vector))
            field_sets.add(tuple(fields))

    if not representations:
        raise ValueError(f"No embeddings found in {path}.")
    if len(models) != 1 or len(dimensions) != 1 or len(field_sets) != 1:
        raise ValueError(
            "Stored embeddings do not use one consistent model, dimension, and "
            "field set."
        )
    stored_fields = list(next(iter(field_sets)))
    if stored_fields != EXPECTED_EMBEDDING_FIELDS:
        raise ValueError(
            f"Stored fields {stored_fields!r} differ from expected "
            f"{EXPECTED_EMBEDDING_FIELDS!r}."
        )

    for family_id, indices in indices_by_family_list.items():
        base_count = sum(
            representations[index]["representation_type"] == "BASE"
            for index in indices
        )
        if base_count != 1:
            raise ValueError(
                f"Family {family_id} must have exactly one base representation; "
                f"found {base_count}."
            )

    matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Stored embeddings contain zero-length vectors.")

    return FamilyEmbeddingStore(
        representations=representations,
        matrix=matrix / norms,
        family_ids=sorted(indices_by_family_list),
        indices_by_family={
            family_id: np.asarray(indices, dtype=np.int64)
            for family_id, indices in indices_by_family_list.items()
        },
        model=next(iter(models)),
        dimensions=next(iter(dimensions)),
    )


def build_query_text(description: str) -> str:
    prefix = f"{QUERY_LABEL}:"
    return (
        description
        if description.casefold().startswith(prefix.casefold())
        else f"{prefix} {description}"
    )


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


def retrieve_top_families(
    query_vector: np.ndarray,
    store: FamilyEmbeddingStore,
    top_k: int,
) -> list[dict[str, Any]]:
    if query_vector.ndim != 1 or query_vector.shape[0] != store.matrix.shape[1]:
        raise ValueError(
            f"Query dimension {query_vector.shape} does not match stored "
            f"dimension {store.matrix.shape[1]}."
        )
    norm = float(np.linalg.norm(query_vector))
    if norm == 0:
        raise ValueError("Query embedding has zero length.")
    if not 1 <= top_k <= len(store.family_ids):
        raise ValueError(f"top_k must be between 1 and {len(store.family_ids)}.")

    similarities = np.clip(store.matrix @ (query_vector / norm), -1.0, 1.0)
    candidates: list[dict[str, Any]] = []
    for family_id in store.family_ids:
        indices = store.indices_by_family[family_id]
        local_scores = similarities[indices]
        local_best = int(np.argmax(local_scores))
        best_index = int(indices[local_best])
        best = store.representations[best_index]

        base_indices = [
            int(index)
            for index in indices
            if store.representations[int(index)]["representation_type"] == "BASE"
        ]
        variant_indices = [
            int(index)
            for index in indices
            if store.representations[int(index)]["representation_type"]
            == "VARIANT_MEMBER"
        ]
        candidates.append(
            {
                "family_id": family_id,
                "family_score": float(similarities[best_index]),
                "base_unit_score": float(similarities[base_indices[0]]),
                "best_variant_score": (
                    float(np.max(similarities[variant_indices]))
                    if variant_indices
                    else None
                ),
                "matched_representation": {
                    "representation_id": best["representation_id"],
                    "type": best["representation_type"],
                    "member_id": best["member_id"],
                    "variant_group_index": best["variant_group_index"],
                    "situation": best["situation"],
                },
            }
        )

    ranked = sorted(
        candidates,
        key=lambda candidate: (-candidate["family_score"], candidate["family_id"]),
    )[:top_k]
    for rank, candidate in enumerate(ranked, start=1):
        candidate["recall_rank"] = rank
    return ranked


def build_rerank_document(candidate: dict[str, Any]) -> str:
    matched = candidate.get("matched_representation")
    if not isinstance(matched, dict):
        raise ValueError("Candidate has no matched_representation.")
    situation = require_string(matched.get("situation"), "matched situation")
    return f"{QUERY_LABEL}: {situation}"


def request_rerank(*, query: str, documents: list[str], top_n: int) -> dict[str, Any]:
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
    response: dict[str, Any], *, candidate_count: int, expected_count: int
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
    seen: set[int] = set()
    for result in results:
        if not isinstance(result, dict):
            raise ValueError("A rerank result is not an object.")
        index = result.get("index")
        score = result.get("relevance_score")
        if not isinstance(index, int) or not 0 <= index < candidate_count:
            raise ValueError(f"Invalid rerank result index: {index!r}")
        if index in seen:
            raise ValueError(f"Duplicate rerank result index: {index}")
        if not isinstance(score, (int, float)):
            raise ValueError(f"Invalid rerank score at index {index}: {score!r}")
        seen.add(index)
        parsed.append((index, float(score)))
    return sorted(parsed, key=lambda item: (-item[1], item[0]))


def build_final_results(
    recalled: list[dict[str, Any]], reranked: list[tuple[int, float]]
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for input_index, rerank_score in reranked:
        candidate = recalled[input_index]
        results.append(
            {
                "family_id": candidate["family_id"],
                "rerank_score": rerank_score,
                "recall_rank": candidate["recall_rank"],
                "family_score": candidate["family_score"],
                "base_unit_score": candidate["base_unit_score"],
                "best_variant_score": candidate["best_variant_score"],
                "matched_representation": candidate["matched_representation"],
            }
        )
    return results


def iter_batches(items: list[Any], size: int) -> Iterator[list[Any]]:
    for index in range(0, len(items), size):
        yield items[index : index + size]


def validate_input_document(path: Path, data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    situations = data.get(SITUATIONS_FIELD)
    if not isinstance(situations, list):
        raise ValueError(f"{path} has no {SITUATIONS_FIELD!r} list.")
    if CONFIG_FIELD in data:
        raise ValueError(f"Input {path} contains reserved field {CONFIG_FIELD!r}.")
    for index, situation in enumerate(situations):
        location = f"{path}.{SITUATIONS_FIELD}[{index}]"
        if not isinstance(situation, dict):
            raise ValueError(f"{location} must be an object.")
        require_string(situation.get(DESCRIPTION_FIELD), f"{location}.{DESCRIPTION_FIELD}")
        if RESULT_FIELD in situation:
            raise ValueError(f"{location} contains reserved field {RESULT_FIELD!r}.")
    return data


def has_complete_results(value: Any, expected_count: int) -> bool:
    if not isinstance(value, list) or len(value) != expected_count:
        return False
    family_ids: set[str] = set()
    for item in value:
        if not isinstance(item, dict):
            return False
        family_id = item.get("family_id")
        matched = item.get("matched_representation")
        if (
            not isinstance(family_id, str)
            or not family_id
            or family_id in family_ids
            or not isinstance(item.get("rerank_score"), (int, float))
            or not isinstance(item.get("family_score"), (int, float))
            or not isinstance(matched, dict)
            or matched.get("type") not in {"BASE", "VARIANT_MEMBER"}
            or not isinstance(matched.get("situation"), str)
        ):
            return False
        family_ids.add(family_id)
    return True


def build_run_config(
    *, embeddings_file: Path, store: FamilyEmbeddingStore, recall_k: int, rerank_top_k: int
) -> dict[str, Any]:
    stat = embeddings_file.stat()
    return {
        "schema_version": "1.0",
        "embedding_model": store.model,
        "embedding_fields": EXPECTED_EMBEDDING_FIELDS,
        "embedding_file": str(embeddings_file),
        "embedding_file_size": stat.st_size,
        "embedding_file_mtime_ns": stat.st_mtime_ns,
        "representation_count": len(store.representations),
        "family_count": len(store.family_ids),
        "family_score": "max(base_unit_score, variant_member_scores)",
        "recall_k": recall_k,
        "rerank_model": RERANK_MODEL,
        "rerank_top_k": rerank_top_k,
        "rerank_document": "the situation of the highest-scoring family representation",
        "rerank_instruct": RERANK_INSTRUCT,
    }


def prepare_pending_situations(
    *,
    input_dir: Path,
    output_dir: Path,
    run_config: dict[str, Any],
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

        source = validate_input_document(input_path, read_json(input_path))
        output_path = output_dir / input_path.name
        output_document = copy.deepcopy(source)
        output_document[CONFIG_FIELD] = copy.deepcopy(run_config)

        if output_path.exists() and not overwrite:
            existing = read_json(output_path)
            if isinstance(existing, dict) and existing.get(CONFIG_FIELD) == run_config:
                existing_situations = existing.get(SITUATIONS_FIELD)
                if isinstance(existing_situations, list):
                    for index, current in enumerate(output_document[SITUATIONS_FIELD]):
                        if index >= len(existing_situations):
                            continue
                        previous = existing_situations[index]
                        if not isinstance(previous, dict):
                            continue
                        previous_base = copy.deepcopy(previous)
                        previous_results = previous_base.pop(RESULT_FIELD, None)
                        if previous_base == current and has_complete_results(
                            previous_results, rerank_top_k
                        ):
                            current[RESULT_FIELD] = previous_results

        selected_from_file = False
        for index, situation in enumerate(output_document[SITUATIONS_FIELD]):
            if has_complete_results(situation.get(RESULT_FIELD), rerank_top_k):
                completed_count += 1
                continue
            if limit is not None and len(pending) >= limit:
                break
            pending.append(
                PendingSituation(
                    source_name=input_path.name,
                    output_path=output_path,
                    output_document=output_document,
                    situation_index=index,
                    situation=situation,
                    description=str(situation[DESCRIPTION_FIELD]).strip(),
                )
            )
            selected_from_file = True

        if selected_from_file:
            write_json_atomic(output_path, output_document)
    return pending, completed_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Retrieve final commonsense generalization records for situation datasets, aggregate "
            "base/variant cosine scores by maximum, and rerank each family using "
            "its highest-scoring representation."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help=(
            "Dataset root. Unless explicitly overridden, input is read from "
            "<dataset-dir>/all_situation and output is written to "
            "<dataset-dir>/retrieval_output."
        ),
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Explicit input directory; overrides dataset-dir/all_situation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Explicit output directory; overrides dataset-dir/retrieval_output.",
    )
    parser.add_argument("--embeddings-file", type=Path, default=DEFAULT_EMBEDDINGS_FILE)
    parser.add_argument("--recall-k", type=int, default=DEFAULT_RECALL_K)
    parser.add_argument("--rerank-top-k", type=int, default=DEFAULT_RERANK_TOP_K)
    parser.add_argument(
        "--embedding-batch-size", type=int, default=DEFAULT_EMBEDDING_BATCH_SIZE
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Process only the first N pending situations."
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Discard cached retrieval results."
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.recall_k < 1:
        raise ValueError("--recall-k must be positive.")
    if not 1 <= args.rerank_top_k <= args.recall_k:
        raise ValueError("--rerank-top-k must be between 1 and recall-k.")
    if not 1 <= args.embedding_batch_size <= 10:
        raise ValueError("--embedding-batch-size must be between 1 and 10.")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive.")


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

    dataset_dir, input_dir, output_dir = resolve_dataset_paths(
        dataset_dir=args.dataset_dir,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        default_input_subdir=DEFAULT_INPUT_SUBDIR,
        default_output_subdir=DEFAULT_OUTPUT_SUBDIR,
    )
    embeddings_file = resolve_path(args.embeddings_file)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not embeddings_file.is_file():
        raise FileNotFoundError(f"Embedding file does not exist: {embeddings_file}")

    store = load_family_embeddings(embeddings_file)
    if store.model != EMBEDDING_MODEL:
        raise ValueError(
            f"Configured embedding model {EMBEDDING_MODEL!r} differs from stored "
            f"model {store.model!r}."
        )
    if args.recall_k > len(store.family_ids):
        raise ValueError(
            f"recall-k={args.recall_k} exceeds family count={len(store.family_ids)}."
        )

    run_config = build_run_config(
        embeddings_file=embeddings_file,
        store=store,
        recall_k=args.recall_k,
        rerank_top_k=args.rerank_top_k,
    )
    pending, completed_count = prepare_pending_situations(
        input_dir=input_dir,
        output_dir=output_dir,
        run_config=run_config,
        rerank_top_k=args.rerank_top_k,
        overwrite=args.overwrite,
        limit=args.limit,
    )

    print(
        f"Loaded families={len(store.family_ids)}, "
        f"representations={len(store.representations)}, pending={len(pending)}, "
        f"skipped_completed={completed_count}, recall_k={args.recall_k}, "
        f"rerank_top_k={args.rerank_top_k}."
    )

    processed = 0
    for batch in iter_batches(pending, args.embedding_batch_size):
        query_texts = [build_query_text(item.description) for item in batch]
        query_vectors = request_query_embeddings(query_texts)
        for item, query_text, query_vector in zip(batch, query_texts, query_vectors):
            recalled = retrieve_top_families(query_vector, store, args.recall_k)
            documents = [build_rerank_document(candidate) for candidate in recalled]
            response = request_rerank(
                query=query_text,
                documents=documents,
                top_n=args.rerank_top_k,
            )
            reranked = parse_rerank_results(
                response,
                candidate_count=len(recalled),
                expected_count=args.rerank_top_k,
            )
            results = build_final_results(recalled, reranked)
            item.situation[RESULT_FIELD] = results
            write_json_atomic(item.output_path, item.output_document)

            processed += 1
            top = results[0]
            matched = top["matched_representation"]
            print(
                f"[{processed}/{len(pending)}] file={item.source_name} "
                f"situation={item.situation_index + 1} family={top['family_id']} "
                f"rerank={top['rerank_score']:.6f} "
                f"matched={matched['type']}:{matched['member_id'] or 'BASE'}"
            )

    summary = {
        **run_config,
        "dataset_dir": str(dataset_dir),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "processed_count": processed,
        "skipped_completed_count": completed_count,
        "selected_pending_count": len(pending),
    }
    write_json_atomic(output_dir / "retrieval_summary.json", summary)
    print(
        f"Done. processed={processed}, skipped_completed={completed_count}, "
        f"output_dir={output_dir}"
    )


if __name__ == "__main__":
    main()

