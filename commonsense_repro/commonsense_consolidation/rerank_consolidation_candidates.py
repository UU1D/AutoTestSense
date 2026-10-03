"""Rerank recalled commonsense generalization records for unassigned instance-level commonsense items."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable


from commonsense_repro.common.paths import PACKAGE_ROOT as PROJECT_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_rerank_api = importlib.import_module(
    "commonsense_repro.commonsense_generalization.global.rerank_generalized_rule_neighbors"
)
DASHSCOPE_API_KEY = _rerank_api.DASHSCOPE_API_KEY
DASHSCOPE_WORKSPACE_ID = _rerank_api.DASHSCOPE_WORKSPACE_ID
RERANK_INSTRUCT = _rerank_api.RERANK_INSTRUCT
RERANK_MODEL = _rerank_api.RERANK_MODEL
RERANK_URL = _rerank_api.RERANK_URL
call_rerank = _rerank_api.call_rerank
parse_rerank_results = _rerank_api.parse_rerank_results
validate_api_config = _rerank_api.validate_api_config
from commonsense_repro.commonsense_consolidation.recall_generalization_records import (
    DEFAULT_SOURCE_EMBEDDING_FILES,
    load_source_embeddings,
    natural_sort_key,
    read_jsonl,
    require_string,
    resolve_path,
)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DEFAULT_INDEX_DIR = Path(
    "work/stage4/"
    "commonsense_consolidation"
)
DEFAULT_RECALL_FILE = DEFAULT_INDEX_DIR / "recall/instance_to_generalization_recall_k20.jsonl"
DEFAULT_CATALOG_FILE = Path(
    "work/stage4/"
    "catalog_assembly/final_commonsense_generalization_records.json"
)
DEFAULT_OUTPUT_DIR = DEFAULT_INDEX_DIR / "rerank"

DEFAULT_FINAL_K = 10
DEFAULT_WORKERS = 5
DEFAULT_TIMEOUT = 180
DEFAULT_MAX_RETRIES = 5

CANDIDATES_FILENAME = "rerank_candidates.jsonl"
SUMMARY_FILENAME = "rerank_summary.json"


def sha256_json(value: Any) -> str:
    serialized = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def load_catalog(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8") as file:
        root = json.load(file)
    families = root.get("rule_families") if isinstance(root, dict) else None
    if not isinstance(families, list):
        raise ValueError(f"{path} must contain a rule_families array.")
    output: dict[str, dict[str, str]] = {}
    for index, family in enumerate(families):
        if not isinstance(family, dict) or not isinstance(family.get("base_unit"), dict):
            raise ValueError(f"Invalid family at {path}.rule_families[{index}].")
        family_id = require_string(family.get("family_id"), f"family[{index}].family_id")
        if family_id in output:
            raise ValueError(f"Duplicate catalog family_id: {family_id}")
        output[family_id] = {
            "situation": require_string(
                family["base_unit"].get("situation"), f"{family_id}.situation"
            ),
            "commonsense_rule": require_string(
                family["base_unit"].get("commonsense_rule"),
                f"{family_id}.commonsense_rule",
            ),
        }
    return output


def load_recall(
    path: Path, known_family_ids: set[str]
) -> tuple[list[str], dict[str, list[dict[str, Any]]], int]:
    ids: list[str] = []
    by_id: dict[str, list[dict[str, Any]]] = {}
    recall_counts: set[int] = set()
    for line_number, row in enumerate(read_jsonl(path), start=1):
        unit_id = require_string(row.get("unit_id"), f"{path}:{line_number}.unit_id")
        if unit_id in by_id:
            raise ValueError(f"Duplicate recall unit_id: {unit_id}")
        candidates = row.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(f"Recall unit {unit_id} has no candidates.")
        seen: set[str] = set()
        for expected_rank, candidate in enumerate(candidates, start=1):
            if not isinstance(candidate, dict) or candidate.get("rank") != expected_rank:
                raise ValueError(f"Invalid candidate rank for {unit_id}.")
            family_id = require_string(candidate.get("family_id"), "candidate.family_id")
            if family_id not in known_family_ids or family_id in seen:
                raise ValueError(f"Invalid/repeated candidate family {family_id} for {unit_id}.")
            matched = candidate.get("matched_representation")
            if not isinstance(matched, dict) or matched.get("type") not in {
                "BASE",
                "VARIANT_MEMBER",
            }:
                raise ValueError(f"Invalid matched representation for {unit_id}/{family_id}.")
            seen.add(family_id)
        ids.append(unit_id)
        by_id[unit_id] = candidates
        recall_counts.add(len(candidates))
    if not ids or len(recall_counts) != 1:
        raise ValueError("Recall input is empty or has inconsistent candidate counts.")
    return ids, by_id, next(iter(recall_counts))


def base_text(base: dict[str, str]) -> str:
    return (
        f"Situation: {base['situation']}\n"
        f"Common-sense rule: {base['commonsense_rule']}"
    )


def candidate_document(
    candidate: dict[str, Any],
    *,
    catalog: dict[str, dict[str, str]],
    source: dict[str, dict[str, Any]],
) -> str:
    family_id = candidate["family_id"]
    matched = candidate["matched_representation"]
    if matched["type"] == "BASE":
        return base_text(catalog[family_id])
    if matched["type"] == "VARIANT_MEMBER":
        member_id = require_string(matched.get("member_id"), "matched member_id")
        if member_id not in source:
            raise ValueError(f"Missing source text for matched variant {member_id}.")
        return source[member_id]["embedding_text"]
    raise ValueError(
        f"Unknown matched representation type for {family_id}: {matched.get('type')!r}"
    )


def input_hashes(
    unit_id: str,
    candidates: list[dict[str, Any]],
    *,
    catalog: dict[str, dict[str, str]],
    source: dict[str, dict[str, Any]],
) -> tuple[str, str]:
    return (
        sha256_json(source[unit_id]["embedding_text"]),
        sha256_json(
            [
                {
                    "recall": candidate,
                    "document": candidate_document(
                        candidate, catalog=catalog, source=source
                    ),
                }
                for candidate in candidates
            ]
        ),
    )


def build_rerank_record(
    *,
    unit_id: str,
    recalled: list[dict[str, Any]],
    response: dict[str, Any],
    query_sha256: str,
    candidates_sha256: str,
) -> dict[str, Any]:
    ranked = parse_rerank_results(response, len(recalled))
    candidates: list[dict[str, Any]] = []
    for rank, (input_index, score) in enumerate(ranked, start=1):
        original = recalled[input_index]
        candidates.append(
            {
                "rerank_rank": rank,
                "family_id": original["family_id"],
                "rerank_score": score,
                "recall_rank": original["rank"],
                "family_score": original["family_score"],
                "base_unit_score": original["base_unit_score"],
                "best_variant_score": original["best_variant_score"],
                "matched_representation": original["matched_representation"],
            }
        )
    return {
        "unit_id": unit_id,
        "model": RERANK_MODEL,
        "recall_k": len(recalled),
        "instruct": RERANK_INSTRUCT,
        "query_sha256": query_sha256,
        "candidates_sha256": candidates_sha256,
        "request_id": response.get("id"),
        "usage": response.get("usage", {}),
        "candidates": candidates,
    }


def rerank_one(
    unit_id: str,
    *,
    recalled_by_id: dict[str, list[dict[str, Any]]],
    catalog: dict[str, dict[str, str]],
    source: dict[str, dict[str, Any]],
    timeout: int,
    max_retries: int,
) -> dict[str, Any]:
    recalled = recalled_by_id[unit_id]
    query_sha256, candidates_sha256 = input_hashes(
        unit_id, recalled, catalog=catalog, source=source
    )
    response = call_rerank(
        query=source[unit_id]["embedding_text"],
        documents=[
            candidate_document(candidate, catalog=catalog, source=source)
            for candidate in recalled
        ],
        timeout=timeout,
        max_retries=max_retries,
    )
    return build_rerank_record(
        unit_id=unit_id,
        recalled=recalled,
        response=response,
        query_sha256=query_sha256,
        candidates_sha256=candidates_sha256,
    )


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as file:
        json.dump(row, file, ensure_ascii=False)
        file.write("\n")
        file.flush()


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


def load_cache(
    path: Path,
    *,
    recall_k: int,
    recalled_by_id: dict[str, list[dict[str, Any]]],
    catalog: dict[str, dict[str, str]],
    source: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    cache: dict[str, dict[str, Any]] = {}
    for line_number, row in enumerate(read_jsonl(path), start=1):
        unit_id = require_string(row.get("unit_id"), f"{path}:{line_number}.unit_id")
        if unit_id in cache or unit_id not in recalled_by_id:
            raise ValueError(f"Invalid cached unit_id: {unit_id}")
        query_hash, candidate_hash = input_hashes(
            unit_id,
            recalled_by_id[unit_id],
            catalog=catalog,
            source=source,
        )
        if (
            row.get("model") != RERANK_MODEL
            or row.get("recall_k") != recall_k
            or row.get("instruct") != RERANK_INSTRUCT
            or row.get("query_sha256") != query_hash
            or row.get("candidates_sha256") != candidate_hash
        ):
            raise ValueError(f"Cached input/configuration changed for {unit_id}; use --overwrite.")
        cache[unit_id] = row
    return cache


def sum_usage(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for row in rows:
        usage = row.get("usage")
        if not isinstance(usage, dict):
            continue
        for key, value in usage.items():
            if isinstance(value, int):
                totals[key] = totals.get(key, 0) + value
    return totals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recall-file", type=Path, default=DEFAULT_RECALL_FILE)
    parser.add_argument("--catalog-file", type=Path, default=DEFAULT_CATALOG_FILE)
    parser.add_argument(
        "--source-embedding-file",
        type=Path,
        action="append",
        dest="source_embedding_files",
    )
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
    if min(args.final_k, args.workers, args.timeout) < 1:
        raise ValueError("final-k, workers, and timeout must be positive.")
    if args.max_retries < 0:
        raise ValueError("max-retries cannot be negative.")
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive.")

    recall_path = resolve_path(args.recall_file)
    catalog_path = resolve_path(args.catalog_file)
    source_paths = [
        resolve_path(path)
        for path in (args.source_embedding_files or DEFAULT_SOURCE_EMBEDDING_FILES)
    ]
    output_dir = resolve_path(args.output_dir)
    for path in (recall_path, catalog_path, *source_paths):
        if not path.is_file():
            raise FileNotFoundError(f"Required input does not exist: {path}")

    catalog = load_catalog(catalog_path)
    source = load_source_embeddings(source_paths)
    ids, recalled_by_id, recall_k = load_recall(recall_path, set(catalog))
    missing_queries = sorted(set(ids) - set(source), key=natural_sort_key)
    if missing_queries:
        raise ValueError(f"Recall queries lack source text: {missing_queries[:20]}")
    if args.final_k > recall_k:
        raise ValueError(f"final_k={args.final_k} exceeds recall_k={recall_k}.")
    selected_ids = ids[: args.limit] if args.limit is not None else ids

    if args.dry_run:
        unit_id = selected_ids[0]
        print(
            json.dumps(
                {
                    "rerank_url": RERANK_URL,
                    "rerank_model": RERANK_MODEL,
                    "instruct": RERANK_INSTRUCT,
                    "unit_id": unit_id,
                    "query": source[unit_id]["embedding_text"],
                    "first_candidate_document": candidate_document(
                        recalled_by_id[unit_id][0], catalog=catalog, source=source
                    ),
                    "recall_k": recall_k,
                    "final_k": args.final_k,
                    "selected_query_count": len(selected_ids),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    validate_api_config()
    candidates_path = output_dir / CANDIDATES_FILENAME
    final_path = output_dir / f"reranked_consolidation_candidates_k{args.final_k}.jsonl"
    summary_path = output_dir / SUMMARY_FILENAME
    if args.overwrite:
        for path in (candidates_path, final_path, summary_path):
            if path.exists():
                path.unlink()

    cache = load_cache(
        candidates_path,
        recall_k=recall_k,
        recalled_by_id=recalled_by_id,
        catalog=catalog,
        source=source,
    )
    pending_ids = [unit_id for unit_id in selected_ids if unit_id not in cache]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                rerank_one,
                unit_id,
                recalled_by_id=recalled_by_id,
                catalog=catalog,
                source=source,
                timeout=args.timeout,
                max_retries=args.max_retries,
            ): unit_id
            for unit_id in pending_ids
        }
        for future in concurrent.futures.as_completed(futures):
            unit_id = futures[future]
            try:
                row = future.result()
            except Exception as error:
                raise RuntimeError(f"Rerank failed for unit {unit_id}: {error}") from error
            append_jsonl(candidates_path, row)
            cache[unit_id] = row
            top = row["candidates"][0]
            print(
                f"Reranked unit_id={unit_id} top_family={top['family_id']} "
                f"score={top['rerank_score']:.6f}"
            )

    output_rows = [
        {
            "unit_id": unit_id,
            "neighbors": [
                {
                    "rank": rank,
                    **{
                        key: candidate[key]
                        for key in (
                            "family_id",
                            "rerank_score",
                            "recall_rank",
                            "family_score",
                            "base_unit_score",
                            "best_variant_score",
                            "matched_representation",
                        )
                    },
                }
                for rank, candidate in enumerate(
                    cache[unit_id]["candidates"][: args.final_k], start=1
                )
            ],
        }
        for unit_id in selected_ids
    ]
    write_jsonl(final_path, output_rows)
    selected_cache = [cache[unit_id] for unit_id in selected_ids]
    write_json(
        summary_path,
        {
            "recall_file": str(recall_path),
            "catalog_file": str(catalog_path),
            "output_file": str(final_path),
            "model": RERANK_MODEL,
            "rerank_url": RERANK_URL,
            "instruct": RERANK_INSTRUCT,
            "recall_k": recall_k,
            "final_k": args.final_k,
            "query_count": len(selected_ids),
            "already_completed_count": len(selected_ids) - len(pending_ids),
            "new_count": len(pending_ids),
            "workers": args.workers,
            "usage": sum_usage(selected_cache),
        },
    )
    print(
        f"Done. queries={len(selected_ids)}, new={len(pending_ids)}, "
        f"recall_k={recall_k}, final_k={args.final_k}, output={final_path}"
    )


if __name__ == "__main__":
    main()

