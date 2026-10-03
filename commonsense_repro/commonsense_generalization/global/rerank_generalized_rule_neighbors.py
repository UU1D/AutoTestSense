"""Rerank recalled commonsense generalization record neighbors with qwen3-rerank."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
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

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "").strip()
DASHSCOPE_WORKSPACE_ID = os.getenv("DASHSCOPE_WORKSPACE_ID", "").strip()
RERANK_MODEL = os.getenv("RERANK_MODEL", "").strip()
RERANK_URL = (
    f"https://{DASHSCOPE_WORKSPACE_ID}.cn-beijing.maas.aliyuncs.com/"
    "compatible-api/v1/reranks"
)

DEFAULT_DATA_DIR = Path(
    "work/stage4"
)
DEFAULT_NEIGHBORS_FILE = (
    DEFAULT_DATA_DIR / "recall/generalized_rule_neighbors_k20.jsonl"
)
DEFAULT_CATALOG_FILE = Path(
    "work/stage3/"
    "gemini_v1_2_non_mutual_s15_leiden_max25_deepseek_flash_v1_4/"
    "commonsense_generalization_records.json"
)
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_DIR / "rerank"

DEFAULT_FINAL_K = 10
DEFAULT_WORKERS = 5
DEFAULT_TIMEOUT = 180
DEFAULT_MAX_RETRIES = 5

RERANK_FIELDS = ["situation", "commonsense_rule"]
RERANK_INSTRUCT = (
    "Rank candidate common-sense rules by semantic similarity to the query. "
    "Compare the situation in terms of the relevant operations, triggers, and "
    "state relationships, and compare the common-sense rule in terms of the "
    "expected outcome and the failure it seeks to prevent. Do not rank a "
    "candidate highly based only on shared topics or keywords."
)

CANDIDATES_FILENAME = "rerank_candidates.jsonl"
SUMMARY_FILENAME = "rerank_summary.json"


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256_json(value: Any) -> str:
    serialized = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


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


def require_nonempty_string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} must be a non-empty string.")
    return value.strip()


def load_catalog(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        catalog = json.load(file)
    if not isinstance(catalog, dict) or not isinstance(
        catalog.get("rule_families"), list
    ):
        raise ValueError(f"{path} must contain a rule_families array.")

    families: dict[str, dict[str, Any]] = {}
    for index, family in enumerate(catalog["rule_families"]):
        location = f"{path}.rule_families[{index}]"
        if not isinstance(family, dict):
            raise ValueError(f"{location} must be an object.")
        family_id = require_nonempty_string(
            family.get("family_id"), f"{location}.family_id"
        )
        if family_id in families:
            raise ValueError(f"Duplicate family_id in catalog: {family_id}")
        base = family.get("base_unit")
        if not isinstance(base, dict):
            raise ValueError(f"{location}.base_unit must be an object.")
        families[family_id] = {
            "situation": require_nonempty_string(
                base.get("situation"), f"{location}.base_unit.situation"
            ),
            "commonsense_rule": require_nonempty_string(
                base.get("commonsense_rule"),
                f"{location}.base_unit.commonsense_rule",
            ),
        }
    return families


def require_score(value: Any, location: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{location} must be a finite number.")
    return float(value)


def load_recall_neighbors(
    path: Path,
    known_ids: set[str],
) -> tuple[list[str], dict[str, list[dict[str, Any]]], int]:
    rows = read_jsonl(path)
    if len(rows) != len(known_ids):
        raise ValueError(
            f"Expected {len(known_ids)} neighbor rows, got {len(rows)}."
        )
    ids: list[str] = []
    neighbors_by_id: dict[str, list[dict[str, Any]]] = {}
    recall_counts: set[int] = set()
    for row_index, row in enumerate(rows):
        family_id = require_nonempty_string(
            row.get("family_id"), f"{path}:{row_index + 1}.family_id"
        )
        if family_id not in known_ids:
            raise ValueError(f"Unknown query family_id: {family_id}")
        if family_id in neighbors_by_id:
            raise ValueError(f"Duplicate query family_id: {family_id}")
        neighbors = row.get("neighbors")
        if not isinstance(neighbors, list) or not neighbors:
            raise ValueError(f"{family_id} has no recalled neighbors.")
        seen_neighbor_ids: set[str] = set()
        validated: list[dict[str, Any]] = []
        for expected_rank, neighbor in enumerate(neighbors, start=1):
            location = f"{family_id}.neighbors[{expected_rank - 1}]"
            if not isinstance(neighbor, dict):
                raise ValueError(f"{location} must be an object.")
            candidate_id = require_nonempty_string(
                neighbor.get("family_id"), f"{location}.family_id"
            )
            if candidate_id not in known_ids:
                raise ValueError(f"{location} contains unknown ID {candidate_id}.")
            if candidate_id == family_id or candidate_id in seen_neighbor_ids:
                raise ValueError(f"Repeated/self candidate {candidate_id} for {family_id}.")
            if neighbor.get("rank") != expected_rank:
                raise ValueError(f"Invalid recall rank in {location}.")
            validated.append(
                {
                    "rank": expected_rank,
                    "family_id": candidate_id,
                    "situation_similarity": require_score(
                        neighbor.get("situation_similarity"),
                        f"{location}.situation_similarity",
                    ),
                    "commonsense_rule_similarity": require_score(
                        neighbor.get("commonsense_rule_similarity"),
                        f"{location}.commonsense_rule_similarity",
                    ),
                    "combined_similarity": require_score(
                        neighbor.get("combined_similarity"),
                        f"{location}.combined_similarity",
                    ),
                }
            )
            seen_neighbor_ids.add(candidate_id)
        ids.append(family_id)
        neighbors_by_id[family_id] = validated
        recall_counts.add(len(validated))
    if len(recall_counts) != 1:
        raise ValueError(f"Inconsistent recall counts: {sorted(recall_counts)}")
    return ids, neighbors_by_id, next(iter(recall_counts))


def build_rerank_text(base_unit: dict[str, str]) -> str:
    return (
        f"Situation: {base_unit['situation']}\n"
        f"Common-sense rule: {base_unit['commonsense_rule']}"
    )


def call_rerank_once(
    *, query: str, documents: list[str], timeout: int
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
    *,
    query: str,
    documents: list[str],
    timeout: int,
    max_retries: int,
) -> dict[str, Any]:
    for attempt in range(max_retries + 1):
        try:
            return call_rerank_once(
                query=query, documents=documents, timeout=timeout
            )
        except HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            retryable = error.code == 429 or error.code >= 500
            if not retryable or attempt >= max_retries:
                raise RuntimeError(
                    f"Rerank HTTP error status={error.code} body={body}"
                ) from error
        except (URLError, TimeoutError) as error:
            if attempt >= max_retries:
                raise RuntimeError(f"Rerank request failed: {error}") from error
        time.sleep(2**attempt)
    raise RuntimeError("Rerank request failed after retries.")


def parse_rerank_results(
    response: dict[str, Any], expected_count: int
) -> list[tuple[int, float]]:
    results = response.get("results")
    if not isinstance(results, list):
        raise ValueError(
            "Rerank response is missing results: "
            f"code={response.get('code')!r}, message={response.get('message')!r}"
        )
    if len(results) != expected_count:
        raise ValueError(f"Expected {expected_count} results, got {len(results)}.")
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
        score = require_score(score, f"rerank result {index}.relevance_score")
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"Rerank score is outside [0, 1]: {score}")
        parsed.append((index, score))
        seen_indices.add(index)
    return sorted(parsed, key=lambda item: (-item[1], item[0]))


def input_hashes(
    family_id: str,
    recalled: list[dict[str, Any]],
    catalog: dict[str, dict[str, Any]],
) -> tuple[str, str]:
    query_hash = sha256_json(catalog[family_id])
    candidates_hash = sha256_json(
        [
            {
                "recall": neighbor,
                "base_unit": catalog[neighbor["family_id"]],
            }
            for neighbor in recalled
        ]
    )
    return query_hash, candidates_hash


def build_candidate_record(
    *,
    family_id: str,
    recalled: list[dict[str, Any]],
    response: dict[str, Any],
    query_sha256: str,
    candidates_sha256: str,
) -> dict[str, Any]:
    ranked = parse_rerank_results(response, len(recalled))
    candidates: list[dict[str, Any]] = []
    for rerank_rank, (input_index, rerank_score) in enumerate(ranked, start=1):
        original = recalled[input_index]
        candidates.append(
            {
                "rerank_rank": rerank_rank,
                "family_id": original["family_id"],
                "rerank_score": rerank_score,
                "recall_rank": original["rank"],
                "situation_similarity": original["situation_similarity"],
                "commonsense_rule_similarity": original[
                    "commonsense_rule_similarity"
                ],
                "combined_similarity": original["combined_similarity"],
            }
        )
    return {
        "family_id": family_id,
        "model": RERANK_MODEL,
        "recall_k": len(recalled),
        "rerank_fields": RERANK_FIELDS,
        "instruct": RERANK_INSTRUCT,
        "query_sha256": query_sha256,
        "candidates_sha256": candidates_sha256,
        "request_id": response.get("id"),
        "usage": response.get("usage", {}),
        "candidates": candidates,
    }


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as file:
        json.dump(record, file, ensure_ascii=False)
        file.write("\n")
        file.flush()


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
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


def load_cache(
    path: Path,
    *,
    recall_k: int,
    neighbors_by_id: dict[str, list[dict[str, Any]]],
    catalog: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    cached: dict[str, dict[str, Any]] = {}
    for line_number, record in enumerate(read_jsonl(path), start=1):
        family_id = require_nonempty_string(
            record.get("family_id"), f"{path}:{line_number}.family_id"
        )
        if family_id in cached:
            raise ValueError(f"Duplicate cached family_id: {family_id}")
        if family_id not in neighbors_by_id:
            raise ValueError(f"Cached family_id is not in recall input: {family_id}")
        query_hash, candidates_hash = input_hashes(
            family_id, neighbors_by_id[family_id], catalog
        )
        if (
            record.get("model") != RERANK_MODEL
            or record.get("recall_k") != recall_k
            or record.get("rerank_fields") != RERANK_FIELDS
            or record.get("instruct") != RERANK_INSTRUCT
            or record.get("query_sha256") != query_hash
            or record.get("candidates_sha256") != candidates_hash
        ):
            raise ValueError(
                f"Cached configuration/input differs for {family_id}; "
                "use --overwrite."
            )
        candidates = record.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != recall_k:
            raise ValueError(f"Invalid cached candidates for {family_id}.")
        cached[family_id] = record
    return cached


def rerank_one(
    family_id: str,
    *,
    neighbors_by_id: dict[str, list[dict[str, Any]]],
    catalog: dict[str, dict[str, Any]],
    timeout: int,
    max_retries: int,
) -> dict[str, Any]:
    recalled = neighbors_by_id[family_id]
    query_hash, candidates_hash = input_hashes(family_id, recalled, catalog)
    response = call_rerank(
        query=build_rerank_text(catalog[family_id]),
        documents=[
            build_rerank_text(catalog[neighbor["family_id"]])
            for neighbor in recalled
        ],
        timeout=timeout,
        max_retries=max_retries,
    )
    return build_candidate_record(
        family_id=family_id,
        recalled=recalled,
        response=response,
        query_sha256=query_hash,
        candidates_sha256=candidates_hash,
    )


def export_final_neighbors(
    path: Path,
    ids: list[str],
    cached: dict[str, dict[str, Any]],
    final_k: int,
) -> None:
    output: list[dict[str, Any]] = []
    for family_id in ids:
        candidates = cached[family_id]["candidates"][:final_k]
        output.append(
            {
                "family_id": family_id,
                "neighbors": [
                    {
                        "rank": rank,
                        **{
                            key: candidate[key]
                            for key in (
                                "family_id",
                                "rerank_score",
                                "recall_rank",
                                "situation_similarity",
                                "commonsense_rule_similarity",
                                "combined_similarity",
                            )
                        },
                    }
                    for rank, candidate in enumerate(candidates, start=1)
                ],
            }
        )
    write_jsonl(path, output)


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


def validate_api_config() -> None:
    missing = [
        name
        for name, value in (
            ("DASHSCOPE_API_KEY", DASHSCOPE_API_KEY),
            ("DASHSCOPE_WORKSPACE_ID", DASHSCOPE_WORKSPACE_ID),
            ("RERANK_MODEL", RERANK_MODEL),
        )
        if not value
    ]
    if missing:
        raise ValueError(f"Missing .env values: {', '.join(missing)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rerank recalled commonsense generalization record neighbors with qwen3-rerank."
    )
    parser.add_argument(
        "--neighbors-file", type=Path, default=DEFAULT_NEIGHBORS_FILE
    )
    parser.add_argument("--catalog-file", type=Path, default=DEFAULT_CATALOG_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--final-k", type=int, default=DEFAULT_FINAL_K)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    neighbors_path = resolve_path(args.neighbors_file)
    catalog_path = resolve_path(args.catalog_file)
    output_dir = resolve_path(args.output_dir)
    if not neighbors_path.is_file():
        raise FileNotFoundError(f"Neighbors file does not exist: {neighbors_path}")
    if not catalog_path.is_file():
        raise FileNotFoundError(f"Catalog file does not exist: {catalog_path}")
    if args.final_k < 1 or args.workers < 1 or args.timeout < 1:
        raise ValueError("final_k, workers, and timeout must be positive.")
    if args.max_retries < 0:
        raise ValueError("max_retries cannot be negative.")
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive.")

    catalog = load_catalog(catalog_path)
    ids, neighbors_by_id, recall_k = load_recall_neighbors(
        neighbors_path, set(catalog)
    )
    if args.final_k > recall_k:
        raise ValueError(f"final_k={args.final_k} exceeds recall_k={recall_k}.")
    selected_ids = ids[: args.limit] if args.limit is not None else ids

    if args.dry_run:
        family_id = selected_ids[0]
        print(
            json.dumps(
                {
                    "rerank_url": RERANK_URL,
                    "rerank_model": RERANK_MODEL,
                    "instruct": RERANK_INSTRUCT,
                    "query_family_id": family_id,
                    "query": build_rerank_text(catalog[family_id]),
                    "recall_k": recall_k,
                    "final_k": args.final_k,
                    "first_three_candidate_ids": [
                        item["family_id"]
                        for item in neighbors_by_id[family_id][:3]
                    ],
                    "selected_query_count": len(selected_ids),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    validate_api_config()
    candidates_path = output_dir / CANDIDATES_FILENAME
    final_neighbors_path = output_dir / f"reranked_neighbors_k{args.final_k}.jsonl"
    summary_path = output_dir / SUMMARY_FILENAME
    if args.overwrite:
        for path in (candidates_path, final_neighbors_path, summary_path):
            if path.exists():
                path.unlink()

    cached = load_cache(
        candidates_path,
        recall_k=recall_k,
        neighbors_by_id=neighbors_by_id,
        catalog=catalog,
    )
    pending_ids = [family_id for family_id in selected_ids if family_id not in cached]
    completed_count = len(selected_ids) - len(pending_ids)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                rerank_one,
                family_id,
                neighbors_by_id=neighbors_by_id,
                catalog=catalog,
                timeout=args.timeout,
                max_retries=args.max_retries,
            ): family_id
            for family_id in pending_ids
        }
        for future in concurrent.futures.as_completed(futures):
            record = future.result()
            append_jsonl(candidates_path, record)
            cached[record["family_id"]] = record
            completed_count += 1
            top = record["candidates"][0]
            print(
                f"[{completed_count}/{len(selected_ids)}] "
                f"query={record['family_id']} top={top['family_id']} "
                f"score={top['rerank_score']:.6f}"
            )

    export_final_neighbors(
        final_neighbors_path, selected_ids, cached, args.final_k
    )
    selected_records = [cached[family_id] for family_id in selected_ids]
    summary = {
        "recall_neighbors_file": str(args.neighbors_file),
        "catalog_file": str(args.catalog_file),
        "candidates_file": str(candidates_path.relative_to(PROJECT_ROOT)),
        "neighbors_file": str(final_neighbors_path.relative_to(PROJECT_ROOT)),
        "rerank_url": RERANK_URL,
        "rerank_model": RERANK_MODEL,
        "rerank_fields": RERANK_FIELDS,
        "instruct": RERANK_INSTRUCT,
        "query_count": len(selected_ids),
        "recall_k": recall_k,
        "final_k": args.final_k,
        "workers": args.workers,
        "cached_count": len(selected_ids) - len(pending_ids),
        "new_count": len(pending_ids),
        "usage": sum_usage(selected_records),
    }
    write_json(summary_path, summary)
    print(
        f"Done. queries={len(selected_ids)}, new={len(pending_ids)}, "
        f"recall_k={recall_k}, final_k={args.final_k}, "
        f"output={output_dir}"
    )


if __name__ == "__main__":
    main()

