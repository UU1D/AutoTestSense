"""Export reproducible candidate components from reranked commonsense generalization record edges."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterable


from commonsense_repro.common.paths import PACKAGE_ROOT as PROJECT_ROOT
CATALOG_FILE = PROJECT_ROOT / (
    "work/stage3/"
    "gemini_v1_2_non_mutual_s15_leiden_max25_deepseek_flash_v1_4/"
    "commonsense_generalization_records.json"
)
RERANK_FILE = PROJECT_ROOT / (
    "work/stage4/"
    "rerank/reranked_neighbors_k10.jsonl"
)
OUTPUT_DIR = PROJECT_ROOT / (
    "work/stage4/"
    "candidate_graph/top3_mutual_s070"
)

TOP_K = 3
MIN_RERANK_SCORE = 0.70
MUTUAL_ONLY = True


def natural_key(value: str) -> tuple[Any, ...]:
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", value)
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_directed_neighbors(
    rerank_rows: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    directed: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rerank_rows:
        source = str(row["family_id"])
        for neighbor in row.get("neighbors", []):
            target = str(neighbor["family_id"])
            if source == target:
                continue
            key = (source, target)
            if key in directed:
                raise ValueError(f"Duplicate directed neighbor: {source} -> {target}")
            directed[key] = {
                "rank": int(neighbor["rank"]),
                "rerank_score": float(neighbor["rerank_score"]),
                "recall_rank": int(neighbor["recall_rank"]),
                "embedding_similarity": float(neighbor["combined_similarity"]),
            }
    return directed


def select_candidate_edges(
    rerank_rows: list[dict[str, Any]],
    top_k: int,
    min_score: float,
    mutual_only: bool,
) -> list[dict[str, Any]]:
    directed = build_directed_neighbors(rerank_rows)
    pair_keys = {
        tuple(sorted((source, target), key=natural_key))
        for source, target in directed
    }
    selected: list[dict[str, Any]] = []

    for source, target in sorted(
        pair_keys, key=lambda pair: (natural_key(pair[0]), natural_key(pair[1]))
    ):
        forward = directed.get((source, target))
        reverse = directed.get((target, source))
        forward_passes = bool(
            forward
            and forward["rank"] <= top_k
            and forward["rerank_score"] >= min_score
        )
        reverse_passes = bool(
            reverse
            and reverse["rank"] <= top_k
            and reverse["rerank_score"] >= min_score
        )
        if mutual_only and not (forward_passes and reverse_passes):
            continue
        if not mutual_only and not (forward_passes or reverse_passes):
            continue

        qualifying = [
            item
            for item, passes in ((forward, forward_passes), (reverse, reverse_passes))
            if item and passes
        ]
        selected.append(
            {
                "edge_id": f"{source}--{target}",
                "source": source,
                "target": target,
                "mutual": forward_passes and reverse_passes,
                "source_to_target": forward,
                "target_to_source": reverse,
                "mean_rerank_score": round(
                    sum(item["rerank_score"] for item in qualifying) / len(qualifying),
                    8,
                ),
                "min_rerank_score": round(
                    min(item["rerank_score"] for item in qualifying), 8
                ),
                "max_rerank_score": round(
                    max(item["rerank_score"] for item in qualifying), 8
                ),
            }
        )
    return selected


def connected_components(
    edges: list[dict[str, Any]],
) -> list[tuple[list[str], list[dict[str, Any]]]]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    edges_by_node: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        source = edge["source"]
        target = edge["target"]
        adjacency[source].add(target)
        adjacency[target].add(source)
        edges_by_node[source].append(edge)
        edges_by_node[target].append(edge)

    components: list[tuple[list[str], list[dict[str, Any]]]] = []
    visited: set[str] = set()
    for start in sorted(adjacency, key=natural_key):
        if start in visited:
            continue
        queue = deque([start])
        visited.add(start)
        members: list[str] = []
        component_edge_ids: set[str] = set()
        while queue:
            node_id = queue.popleft()
            members.append(node_id)
            for edge in edges_by_node[node_id]:
                component_edge_ids.add(edge["edge_id"])
            for neighbor in sorted(adjacency[node_id], key=natural_key):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        members.sort(key=natural_key)
        component_edges = [
            edge for edge in edges if edge["edge_id"] in component_edge_ids
        ]
        components.append((members, component_edges))

    components.sort(key=lambda item: (-len(item[0]), natural_key(item[0][0])))
    return components


def family_record(family: dict[str, Any]) -> dict[str, Any]:
    base_unit = family.get("base_unit", {})
    return {
        "family_id": str(family["family_id"]),
        "situation": base_unit.get("situation", ""),
        "commonsense_rule": base_unit.get("commonsense_rule", ""),
        "member_count": int(family.get("member_count", 0)),
        "base_origin": family.get("base_origin", "UNKNOWN"),
        "source_batch": family.get("source_batch"),
    }


def component_signature(member_ids: list[str]) -> str:
    content = "\n".join(member_ids).encode("utf-8")
    return hashlib.sha256(content).hexdigest()[:16]


def prepare_exports(
    catalog: dict[str, Any],
    rerank_rows: list[dict[str, Any]],
    top_k: int,
    min_score: float,
    mutual_only: bool,
) -> dict[str, Any]:
    family_by_id = {
        str(family["family_id"]): family
        for family in catalog.get("rule_families", [])
    }
    if not family_by_id:
        raise ValueError("The catalog does not contain any commonsense generalization records")

    query_ids = {str(row["family_id"]) for row in rerank_rows}
    unknown_query_ids = query_ids - set(family_by_id)
    if unknown_query_ids:
        raise ValueError(f"Rerank output contains unknown family IDs: {unknown_query_ids}")

    edges = select_candidate_edges(
        rerank_rows,
        top_k=top_k,
        min_score=min_score,
        mutual_only=mutual_only,
    )
    raw_components = connected_components(edges)
    components: list[dict[str, Any]] = []
    llm_batches: list[dict[str, Any]] = []
    assignments: list[dict[str, Any]] = []
    matched_ids: set[str] = set()

    for index, (member_ids, component_edges) in enumerate(raw_components, start=1):
        component_id = f"C{index:04d}"
        matched_ids.update(member_ids)
        density = 2 * len(component_edges) / (len(member_ids) * (len(member_ids) - 1))
        families = [family_record(family_by_id[family_id]) for family_id in member_ids]
        component = {
            "component_id": component_id,
            "component_signature": component_signature(member_ids),
            "node_count": len(member_ids),
            "edge_count": len(component_edges),
            "density": round(density, 8),
            "families": families,
            "edges": component_edges,
        }
        components.append(component)
        llm_batches.append(
            {
                "batch_id": component_id,
                "component_signature": component["component_signature"],
                "unit_count": len(member_ids),
                "units": [
                    {
                        "id": family["family_id"],
                        "situation": family["situation"],
                        "commonsense_rule": family["commonsense_rule"],
                    }
                    for family in families
                ],
                "candidate_edges": [
                    {
                        "source": edge["source"],
                        "target": edge["target"],
                        "score_source_to_target": edge["source_to_target"][
                            "rerank_score"
                        ]
                        if edge["source_to_target"]
                        else None,
                        "score_target_to_source": edge["target_to_source"][
                            "rerank_score"
                        ]
                        if edge["target_to_source"]
                        else None,
                        "mean_score": edge["mean_rerank_score"],
                    }
                    for edge in component_edges
                ],
            }
        )
        assignments.extend(
            {
                "family_id": family_id,
                "status": "candidate_component",
                "component_id": component_id,
                "component_signature": component["component_signature"],
                "component_size": len(member_ids),
            }
            for family_id in member_ids
        )

    unmatched_ids = sorted(set(family_by_id) - matched_ids, key=natural_key)
    assignments.extend(
        {
            "family_id": family_id,
            "status": "unmatched",
            "component_id": None,
            "component_signature": None,
            "component_size": 1,
        }
        for family_id in unmatched_ids
    )
    assignments.sort(key=lambda item: natural_key(item["family_id"]))

    return {
        "edges": edges,
        "components": components,
        "llm_batches": llm_batches,
        "unmatched_generalization_records": [
            family_record(family_by_id[family_id]) for family_id in unmatched_ids
        ],
        "assignments": assignments,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=CATALOG_FILE)
    parser.add_argument("--rerank", type=Path, default=RERANK_FILE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--min-score", type=float, default=MIN_RERANK_SCORE)
    parser.add_argument(
        "--non-mutual",
        action="store_true",
        help="Keep an edge when either direction passes instead of requiring both.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1")
    if not 0 <= args.min_score <= 1:
        raise ValueError("--min-score must be between 0 and 1")

    catalog_path = args.catalog.resolve()
    rerank_path = args.rerank.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    mutual_only = not args.non_mutual

    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    rerank_rows = load_jsonl(rerank_path)
    exports = prepare_exports(
        catalog,
        rerank_rows,
        top_k=args.top_k,
        min_score=args.min_score,
        mutual_only=mutual_only,
    )

    write_jsonl(output_dir / "graph_edges.jsonl", exports["edges"])
    write_jsonl(output_dir / "candidate_components.jsonl", exports["components"])
    write_jsonl(output_dir / "llm_batches.jsonl", exports["llm_batches"])
    write_jsonl(
        output_dir / "generalization_component_assignments.jsonl", exports["assignments"]
    )
    write_json(
        output_dir / "unmatched_generalization_records.json",
        {
            "unmatched_count": len(exports["unmatched_generalization_records"]),
            "families": exports["unmatched_generalization_records"],
        },
    )

    component_sizes = [item["node_count"] for item in exports["components"]]
    summary = {
        "configuration": {
            "top_k": args.top_k,
            "min_rerank_score": args.min_score,
            "mutual_only": mutual_only,
            "mutual_score_rule": "both directional scores must meet the threshold"
            if mutual_only
            else None,
            "edge_weight": "mean of qualifying directional rerank scores",
        },
        "source_files": {
            "catalog": str(catalog_path),
            "catalog_sha256": source_sha256(catalog_path),
            "rerank": str(rerank_path),
            "rerank_sha256": source_sha256(rerank_path),
        },
        "statistics": {
            "total_family_count": len(catalog.get("rule_families", [])),
            "matched_family_count": len(exports["assignments"])
            - len(exports["unmatched_generalization_records"]),
            "unmatched_family_count": len(exports["unmatched_generalization_records"]),
            "candidate_edge_count": len(exports["edges"]),
            "component_count": len(exports["components"]),
            "smallest_component_size": min(component_sizes, default=0),
            "largest_component_size": max(component_sizes, default=0),
            "components_size_2_to_25": sum(
                2 <= size <= 25 for size in component_sizes
            ),
            "components_over_25": sum(size > 25 for size in component_sizes),
        },
        "output_files": {
            "edges": "graph_edges.jsonl",
            "components": "candidate_components.jsonl",
            "llm_batches": "llm_batches.jsonl",
            "unmatched": "unmatched_generalization_records.json",
            "assignments": "generalization_component_assignments.jsonl",
        },
    }
    write_json(output_dir / "graph_snapshot_summary.json", summary)

    stats = summary["statistics"]
    print(
        "Done. "
        f"families={stats['total_family_count']}, "
        f"matched={stats['matched_family_count']}, "
        f"edges={stats['candidate_edge_count']}, "
        f"components={stats['component_count']}, "
        f"unmatched={stats['unmatched_family_count']}, "
        f"output_dir={output_dir}"
    )


if __name__ == "__main__":
    main()

