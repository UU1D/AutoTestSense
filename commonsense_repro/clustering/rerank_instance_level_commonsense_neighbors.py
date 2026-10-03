"""Recall embedding neighbors and rerank them with qwen3-rerank."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
from commonsense_repro.common.paths import PACKAGE_ROOT as PROJECT_ROOT
load_dotenv(PROJECT_ROOT / ".env")

# API configuration is read once when the script starts.
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "").strip()
DASHSCOPE_WORKSPACE_ID = os.getenv("DASHSCOPE_WORKSPACE_ID", "").strip()
RERANK_MODEL = os.getenv("RERANK_MODEL", "").strip()
RERANK_URL = (
    f"https://{DASHSCOPE_WORKSPACE_ID}.cn-beijing.maas.aliyuncs.com/"
    "compatible-api/v1/reranks"
)

# Experiment parameters. Command-line arguments can override these values.
RECALL_NEIGHBORS_FILE = Path(
    "work/stage2/neighbors/embedding_neighbors_k60.jsonl"
)
METADATA_FILES = [
    Path("data/checkpoints/instance_level_commonsense/gemini_v1_2_extract.json"),
    Path("data/checkpoints/instance_level_commonsense/gemini_v1_2_github.json"),
    Path("data/checkpoints/instance_level_commonsense/gemini_v1_2_negative.json"),
]
OUTPUT_DIR = Path("work/stage2/rerank")
FINAL_K = 30
REQUEST_TIMEOUT = 180
MAX_RETRIES = 5
REQUEST_INTERVAL = 0.0

RERANK_FIELDS = [
    "violated_commonsense_rule",
    "situation",
]

FIELD_LABELS = {
    "violated_commonsense_rule": "Violated commonsense rule",
    "situation": "Situation",
}

RERANK_INSTRUCT = (
    "Rank candidate software issue cases by whether they violate the same "
    "underlying commonsense principle as the query case. Focus primarily on "
    "the violated commonsense rule and the underlying causal or behavioral "
    "pattern. Use the situation only to disambiguate. Do not rank cases highly "
    "merely because they mention the same software domain, project, entity, "
    "component, or technical terminology."
)

CANDIDATES_FILENAME = "rerank_candidates.jsonl"
NEIGHBORS_FILENAME = "reranked_neighbors.jsonl"
SUMMARY_FILENAME = "rerank_summary.json"


# -----------------------------------------------------------------------------
# Input
# -----------------------------------------------------------------------------
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


def read_metadata(
    paths: list[Path],
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    records: dict[str, dict[str, Any]] = {}
    source_sample_counts: dict[str, int] = {}
    for path in paths:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON array in {path}.")

        source_sample_counts[str(path)] = len(data)
        for index, record in enumerate(data):
            if not isinstance(record, dict):
                raise ValueError(f"Expected an object at {path} index {index}.")
            unique_id = record.get("id")
            if not isinstance(unique_id, str) or not unique_id.strip():
                raise ValueError(f"Missing id at {path} index {index}.")
            if unique_id in records:
                raise ValueError(
                    f"Duplicate metadata id across input files: {unique_id} "
                    f"(found again in {path})"
                )
            records[unique_id] = record
    return records, source_sample_counts


def load_recall_neighbors(
    path: Path,
) -> tuple[list[str], dict[str, list[dict[str, Any]]], int]:
    records = read_jsonl(path)
    if len(records) < 2:
        raise ValueError("At least two neighbor records are required.")

    ids: list[str] = []
    seen_ids: set[str] = set()
    for index, record in enumerate(records):
        unique_id = record.get("id")
        if not isinstance(unique_id, str) or not unique_id.strip():
            raise ValueError(f"Missing id in neighbor record {index}.")
        if unique_id in seen_ids:
            raise ValueError(f"Duplicate neighbor record id: {unique_id}")
        seen_ids.add(unique_id)
        ids.append(unique_id)

    known_ids = set(ids)
    neighbors_by_id: dict[str, list[dict[str, Any]]] = {}
    neighbor_counts: set[int] = set()
    for record in records:
        unique_id = str(record["id"])
        neighbors = record.get("neighbors")
        if not isinstance(neighbors, list) or not neighbors:
            raise ValueError(f"Record {unique_id} has no neighbors list.")

        row_ids: set[str] = set()
        validated: list[dict[str, Any]] = []
        for expected_rank, neighbor in enumerate(neighbors, start=1):
            if not isinstance(neighbor, dict):
                raise ValueError(f"Invalid neighbor for {unique_id} at rank {expected_rank}.")
            neighbor_id = neighbor.get("id")
            score = neighbor.get("cosine_similarity")
            if neighbor.get("rank") != expected_rank:
                raise ValueError(f"Invalid rank for {unique_id} at position {expected_rank}.")
            if not isinstance(neighbor_id, str) or neighbor_id not in known_ids:
                raise ValueError(f"Unknown neighbor id for {unique_id}: {neighbor_id!r}")
            if neighbor_id == unique_id or neighbor_id in row_ids:
                raise ValueError(f"Invalid repeated/self neighbor {neighbor_id} for {unique_id}.")
            if not isinstance(score, (int, float)):
                raise ValueError(f"Missing cosine score for {unique_id} -> {neighbor_id}.")
            row_ids.add(neighbor_id)
            validated.append(neighbor)

        neighbor_counts.add(len(validated))
        neighbors_by_id[unique_id] = validated

    if len(neighbor_counts) != 1:
        raise ValueError(f"Neighbor counts are inconsistent: {sorted(neighbor_counts)}")
    return ids, neighbors_by_id, next(iter(neighbor_counts))


# -----------------------------------------------------------------------------
# Rerank text and API
# -----------------------------------------------------------------------------
def format_field(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "; ".join(
            str(item).strip() for item in value if str(item).strip()
        )
    if value is None:
        return ""
    return str(value).strip()


def build_rerank_text(record: dict[str, Any]) -> str:
    parts = []
    for field in RERANK_FIELDS:
        value = format_field(record.get(field))
        if value:
            parts.append(f"{FIELD_LABELS[field]}: {value}")
    if not parts:
        raise ValueError(f"Record {record.get('id')} has no rerank text.")
    return "\n".join(parts)


def call_rerank_once(
    query: str,
    documents: list[str],
    timeout: int,
) -> dict[str, Any]:
    payload = {
        "model": RERANK_MODEL,
        "query": query,
        "documents": documents,
        "top_n": len(documents),
        "instruct": RERANK_INSTRUCT,
    }
    request = Request(
        RERANK_URL,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def call_rerank(
    query: str,
    documents: list[str],
    timeout: int,
    max_retries: int,
) -> dict[str, Any]:
    for attempt in range(max_retries + 1):
        try:
            return call_rerank_once(query, documents, timeout)
        except HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            retryable = error.code == 429 or error.code >= 500
            if not retryable or attempt == max_retries:
                raise RuntimeError(
                    f"Rerank HTTP error: status={error.code}, body={body}"
                ) from error
        except (URLError, TimeoutError) as error:
            if attempt == max_retries:
                raise RuntimeError(f"Rerank request failed: {error}") from error

        delay = 2**attempt
        print(f"Request failed; retrying in {delay}s ({attempt + 1}/{max_retries}).")
        time.sleep(delay)

    raise RuntimeError("Rerank request failed after retries.")


def parse_rerank_results(
    response: dict[str, Any],
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
        if not isinstance(index, int) or not 0 <= index < expected_count:
            raise ValueError(f"Invalid rerank result index: {index!r}")
        if index in seen_indices:
            raise ValueError(f"Duplicate rerank result index: {index}")
        if not isinstance(score, (int, float)):
            raise ValueError(f"Invalid rerank score at index {index}: {score!r}")
        seen_indices.add(index)
        parsed.append((index, float(score)))

    return sorted(parsed, key=lambda item: (-item[1], item[0]))


# -----------------------------------------------------------------------------
# Cache and output
# -----------------------------------------------------------------------------
def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        json.dump(record, file, ensure_ascii=False)
        file.write("\n")
        file.flush()


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


def load_cache(path: Path, recall_k: int) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}

    cached: dict[str, dict[str, Any]] = {}
    for record in read_jsonl(path):
        unique_id = record.get("id")
        if not isinstance(unique_id, str) or not unique_id:
            raise ValueError(f"Cached record has no id in {path}.")
        if unique_id in cached:
            raise ValueError(f"Duplicate cached id: {unique_id}")
        if (
            record.get("model") != RERANK_MODEL
            or record.get("recall_k") != recall_k
            or record.get("rerank_fields") != RERANK_FIELDS
            or record.get("instruct") != RERANK_INSTRUCT
        ):
            raise ValueError(
                f"Cached configuration differs for {unique_id}; use --overwrite."
            )
        candidates = record.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != recall_k:
            raise ValueError(f"Invalid cached candidates for {unique_id}.")
        cached[unique_id] = record
    return cached


def build_candidate_record(
    query_id: str,
    candidate_ids: list[str],
    cosine_scores: list[float],
    response: dict[str, Any],
) -> dict[str, Any]:
    ranked_results = parse_rerank_results(response, len(candidate_ids))
    candidates = [
        {
            "rerank_rank": rerank_rank,
            "id": candidate_ids[input_index],
            "rerank_score": rerank_score,
            "recall_rank": input_index + 1,
            "cosine_similarity": float(cosine_scores[input_index]),
        }
        for rerank_rank, (input_index, rerank_score) in enumerate(
            ranked_results, start=1
        )
    ]
    return {
        "id": query_id,
        "model": RERANK_MODEL,
        "recall_k": len(candidate_ids),
        "rerank_fields": RERANK_FIELDS,
        "instruct": RERANK_INSTRUCT,
        "request_id": response.get("id"),
        "usage": response.get("usage", {}),
        "candidates": candidates,
    }


def export_neighbors(
    path: Path,
    ids: list[str],
    cached: dict[str, dict[str, Any]],
    final_k: int,
) -> None:
    records = []
    for unique_id in ids:
        candidates = cached[unique_id]["candidates"][:final_k]
        records.append(
            {
                "id": unique_id,
                "neighbors": [
                    {
                        "rank": rank,
                        "id": candidate["id"],
                        "rerank_score": candidate["rerank_score"],
                        "recall_rank": candidate["recall_rank"],
                        "cosine_similarity": candidate["cosine_similarity"],
                    }
                    for rank, candidate in enumerate(candidates, start=1)
                ],
            }
        )
    write_jsonl(path, records)


def sum_usage(records: Iterable[dict[str, Any]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for record in records:
        usage = record.get("usage")
        if not isinstance(usage, dict):
            continue
        for key, value in usage.items():
            if isinstance(value, int):
                totals[key] = totals.get(key, 0) + value
    return totals


# -----------------------------------------------------------------------------
# Command line and orchestration
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recall embedding neighbors and rerank them with qwen3-rerank."
    )
    parser.add_argument(
        "--neighbors-file", type=Path, default=RECALL_NEIGHBORS_FILE
    )
    parser.add_argument(
        "--metadata-files",
        "--metadata-file",
        dest="metadata_files",
        type=Path,
        nargs="+",
        default=METADATA_FILES,
        help="One or more metadata JSON files whose IDs share one neighbor space.",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--final-k", type=int, default=FINAL_K)
    parser.add_argument("--timeout", type=int, default=REQUEST_TIMEOUT)
    parser.add_argument("--max-retries", type=int, default=MAX_RETRIES)
    parser.add_argument("--request-interval", type=float, default=REQUEST_INTERVAL)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.neighbors_file.is_file():
        raise FileNotFoundError(f"Neighbor file does not exist: {args.neighbors_file}")
    if not args.metadata_files:
        raise ValueError("At least one metadata file is required.")
    missing_files = [path for path in args.metadata_files if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(f"Metadata file does not exist: {missing_files[0]}")
    if len(args.metadata_files) != len(set(args.metadata_files)):
        raise ValueError("Metadata file arguments contain duplicate paths.")
    if args.final_k < 1:
        raise ValueError("final_k must be positive.")
    if args.timeout < 1:
        raise ValueError("timeout must be positive.")
    if args.max_retries < 0 or args.request_interval < 0:
        raise ValueError("max_retries and request_interval cannot be negative.")
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive.")


def validate_api_config() -> None:
    missing = [
        name
        for name, value in [
            ("DASHSCOPE_API_KEY", DASHSCOPE_API_KEY),
            ("DASHSCOPE_WORKSPACE_ID", DASHSCOPE_WORKSPACE_ID),
            ("RERANK_MODEL", RERANK_MODEL),
        ]
        if not value
    ]
    if missing:
        raise ValueError(f"Missing .env values: {', '.join(missing)}")


def print_dry_run(
    query_id: str,
    query_text: str,
    neighbors: list[dict[str, Any]],
    final_k: int,
) -> None:
    preview = neighbors[:3]
    print(
        json.dumps(
            {
                "rerank_url": RERANK_URL,
                "rerank_model": RERANK_MODEL,
                "query_id": query_id,
                "query": query_text,
                "instruct": RERANK_INSTRUCT,
                "recall_k": len(neighbors),
                "final_k": final_k,
                "first_three_candidates": preview,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    args = parse_args()
    validate_args(args)

    ids, recall_neighbors, recall_k = load_recall_neighbors(args.neighbors_file)
    if args.final_k > recall_k:
        raise ValueError(f"final_k={args.final_k} exceeds recall neighbor k={recall_k}.")
    metadata, source_sample_counts = read_metadata(args.metadata_files)

    missing_metadata = [unique_id for unique_id in ids if unique_id not in metadata]
    if missing_metadata:
        raise ValueError(
            f"Metadata is missing {len(missing_metadata)} neighbor ids; "
            f"first={missing_metadata[0]}"
        )

    selected_ids = ids[: args.limit] if args.limit is not None else ids

    if args.dry_run:
        query_id = selected_ids[0]
        print_dry_run(
            query_id=query_id,
            query_text=build_rerank_text(metadata[query_id]),
            neighbors=recall_neighbors[query_id],
            final_k=args.final_k,
        )
        print(f"Dry run OK. selected_queries={len(selected_ids)}")
        return

    validate_api_config()
    candidates_path = args.output_dir / CANDIDATES_FILENAME
    neighbors_path = args.output_dir / NEIGHBORS_FILENAME
    summary_path = args.output_dir / SUMMARY_FILENAME

    if args.overwrite:
        for path in (candidates_path, neighbors_path, summary_path):
            if path.exists():
                path.unlink()

    cached = load_cache(candidates_path, recall_k)
    unknown_cached_ids = set(cached) - set(ids)
    if unknown_cached_ids:
        first = sorted(unknown_cached_ids)[0]
        raise ValueError(f"Cached id is absent from the neighbor input: {first}")

    pending_ids = [unique_id for unique_id in selected_ids if unique_id not in cached]
    for pending_index, query_id in enumerate(pending_ids, start=1):
        recalled = recall_neighbors[query_id]
        candidate_ids = [str(neighbor["id"]) for neighbor in recalled]
        cosine_scores = [float(neighbor["cosine_similarity"]) for neighbor in recalled]
        documents = [build_rerank_text(metadata[item_id]) for item_id in candidate_ids]
        response = call_rerank(
            query=build_rerank_text(metadata[query_id]),
            documents=documents,
            timeout=args.timeout,
            max_retries=args.max_retries,
        )
        record = build_candidate_record(
            query_id=query_id,
            candidate_ids=candidate_ids,
            cosine_scores=cosine_scores,
            response=response,
        )
        append_jsonl(candidates_path, record)
        cached[query_id] = record

        top = record["candidates"][0]
        completed = len(selected_ids) - len(pending_ids) + pending_index
        print(
            f"[{completed}/{len(selected_ids)}] query={query_id} "
            f"top={top['id']} score={top['rerank_score']:.6f}"
        )
        if args.request_interval > 0 and pending_index < len(pending_ids):
            time.sleep(args.request_interval)

    export_neighbors(neighbors_path, selected_ids, cached, args.final_k)
    selected_cache = [cached[unique_id] for unique_id in selected_ids]
    summary = {
        "recall_neighbors_file": str(args.neighbors_file),
        "metadata_files": [str(path) for path in args.metadata_files],
        "metadata_source_sample_counts": source_sample_counts,
        "candidates_file": str(candidates_path),
        "neighbors_file": str(neighbors_path),
        "rerank_model": RERANK_MODEL,
        "rerank_fields": RERANK_FIELDS,
        "instruct": RERANK_INSTRUCT,
        "query_count": len(selected_ids),
        "recall_k": recall_k,
        "final_k": args.final_k,
        "cached_count": len(selected_ids) - len(pending_ids),
        "new_count": len(pending_ids),
        "usage": sum_usage(selected_cache),
    }
    write_json(summary_path, summary)

    print(
        f"Done. queries={len(selected_ids)}, new={len(pending_ids)}, "
        f"recall_k={recall_k}, final_k={args.final_k}, "
        f"output={args.output_dir}"
    )


if __name__ == "__main__":
    main()

