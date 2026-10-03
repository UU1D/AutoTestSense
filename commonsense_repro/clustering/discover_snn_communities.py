"""Build an exact shared-nearest-neighbor graph and discover communities."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import numpy as np
from networkx.algorithms.community import louvain_communities, modularity


# Edit these paths and parameters for a different SNN experiment.
DEFAULT_NEIGHBORS_FILE = Path(
    "work/stage2/rerank/"
    "reranked_neighbors.jsonl"
)
DEFAULT_METADATA_FILES = [
    Path("data/checkpoints/instance_level_commonsense/gemini_v1_2_extract.json"),
    Path("data/checkpoints/instance_level_commonsense/gemini_v1_2_github.json"),
    Path("data/checkpoints/instance_level_commonsense/gemini_v1_2_negative.json"),
]
DEFAULT_OUTPUT_DIR = Path(
    "work/stage2/snn"
)

DEFAULT_MIN_SHARED_NEIGHBORS = 10
DEFAULT_MUTUAL_ONLY = True
DEFAULT_RESOLUTION = 0.8
DEFAULT_RANDOM_SEED = 42
DEFAULT_REPRESENTATIVES = 10

OUTPUT_FILENAMES = [
    "snn_edges.jsonl",
    "snn_assignments.jsonl",
    "snn_communities.json",
    "snn_communities.md",
    "snn_summary.json",
]


def natural_sort_key(value: str) -> tuple[tuple[int, Any], ...]:
    parts = re.split(r"(\d+)", value.casefold())
    return tuple(
        (1, int(part)) if part.isdigit() else (0, part)
        for part in parts
        if part
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}.")
            records.append(item)
    return records


def load_json_array(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        raise ValueError(f"Expected a JSON array of objects in {path}.")
    return data


def load_neighbors(path: Path) -> tuple[list[str], np.ndarray, int]:
    records = load_jsonl(path)
    if len(records) < 2:
        raise ValueError("At least two neighbor records are required.")

    ids: list[str] = []
    seen_ids: set[str] = set()
    for index, record in enumerate(records):
        unique_id = record.get("id")
        if not isinstance(unique_id, str) or not unique_id:
            raise ValueError(f"Missing id in neighbor record {index}.")
        if unique_id in seen_ids:
            raise ValueError(f"Duplicate neighbor record id: {unique_id}")
        seen_ids.add(unique_id)
        ids.append(unique_id)

    id_to_index = {unique_id: index for index, unique_id in enumerate(ids)}
    rows: list[list[int]] = []
    neighbor_counts: set[int] = set()
    for record in records:
        unique_id = str(record["id"])
        neighbors = record.get("neighbors")
        if not isinstance(neighbors, list) or not neighbors:
            raise ValueError(f"Record {unique_id} has no neighbors list.")

        row: list[int] = []
        row_ids: set[str] = set()
        for expected_rank, neighbor in enumerate(neighbors, start=1):
            if not isinstance(neighbor, dict):
                raise ValueError(f"Invalid neighbor for {unique_id} at rank {expected_rank}.")
            neighbor_id = neighbor.get("id")
            rank = neighbor.get("rank")
            if rank != expected_rank:
                raise ValueError(
                    f"Non-contiguous rank for {unique_id}: expected {expected_rank}, got {rank}."
                )
            if not isinstance(neighbor_id, str) or neighbor_id not in id_to_index:
                raise ValueError(f"Unknown neighbor id for {unique_id}: {neighbor_id!r}")
            if neighbor_id == unique_id:
                raise ValueError(f"Record {unique_id} contains itself as a neighbor.")
            if neighbor_id in row_ids:
                raise ValueError(f"Duplicate neighbor {neighbor_id} for {unique_id}.")
            row_ids.add(neighbor_id)
            row.append(id_to_index[neighbor_id])

        neighbor_counts.add(len(row))
        rows.append(row)

    if len(neighbor_counts) != 1:
        raise ValueError(f"Neighbor counts are inconsistent: {sorted(neighbor_counts)}")
    k = next(iter(neighbor_counts))
    return ids, np.asarray(rows, dtype=np.int32), k


def build_neighbor_mask(neighbor_indices: np.ndarray) -> np.ndarray:
    sample_count = neighbor_indices.shape[0]
    mask = np.zeros((sample_count, sample_count), dtype=np.uint8)
    mask[np.arange(sample_count)[:, None], neighbor_indices] = 1
    return mask


def build_snn_edges(
    *,
    ids: list[str],
    neighbor_mask: np.ndarray,
    k: int,
    min_shared_neighbors: int,
    mutual_only: bool,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    shared_counts = (
        neighbor_mask.astype(np.uint16) @ neighbor_mask.astype(np.uint16).T
    )
    edges: list[dict[str, Any]] = []
    sample_count = len(ids)

    for left in range(sample_count):
        for right in range(left + 1, sample_count):
            if mutual_only and not (
                neighbor_mask[left, right] and neighbor_mask[right, left]
            ):
                continue
            shared_count = int(shared_counts[left, right])
            if shared_count < min_shared_neighbors:
                continue
            jaccard = shared_count / (2 * k - shared_count)
            edges.append(
                {
                    "source": ids[left],
                    "target": ids[right],
                    "shared_neighbor_count": shared_count,
                    "shared_neighbor_jaccard": jaccard,
                }
            )
    return edges, shared_counts


def build_graph(ids: list[str], edges: list[dict[str, Any]]) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(ids)
    graph.add_weighted_edges_from(
        (
            edge["source"],
            edge["target"],
            edge["shared_neighbor_jaccard"],
        )
        for edge in edges
    )
    return graph


def discover_communities(
    graph: nx.Graph,
    resolution: float,
    random_seed: int,
) -> list[set[str]]:
    if graph.number_of_edges() == 0:
        communities = [{str(node)} for node in graph.nodes]
    else:
        communities = [
            set(community)
            for community in louvain_communities(
                graph,
                weight="weight",
                resolution=resolution,
                seed=random_seed,
            )
        ]

    return sorted(
        communities,
        key=lambda community: (
            -len(community),
            natural_sort_key(min(community, key=natural_sort_key)),
        ),
    )


def build_metadata_index(
    paths: list[Path],
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    metadata: dict[str, dict[str, Any]] = {}
    source_sample_counts: dict[str, int] = {}
    for path in paths:
        records = load_json_array(path)
        source_sample_counts[str(path)] = len(records)
        for index, record in enumerate(records):
            unique_id = record.get("id")
            if not isinstance(unique_id, str) or not unique_id.strip():
                raise ValueError(f"Missing id at {path} index {index}.")
            if unique_id in metadata:
                raise ValueError(
                    f"Duplicate metadata id across input files: {unique_id} "
                    f"(found again in {path})"
                )
            metadata[unique_id] = record
    return metadata, source_sample_counts


def rank_community_members(graph: nx.Graph, members: set[str]) -> list[str]:
    return sorted(
        members,
        key=lambda unique_id: (
            -graph.degree(unique_id, weight="weight"),
            natural_sort_key(unique_id),
        ),
    )


def build_community_records(
    *,
    graph: nx.Graph,
    communities: list[set[str]],
    metadata: dict[str, dict[str, Any]],
    representative_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    assignments: list[dict[str, Any]] = []
    community_records: list[dict[str, Any]] = []

    for community_id, members in enumerate(communities):
        ranked_members = rank_community_members(graph, members)
        representatives = ranked_members[:representative_count]
        community_records.append(
            {
                "community_id": community_id,
                "size": len(members),
                "member_ids": sorted(members, key=natural_sort_key),
                "representatives": [
                    {
                        "id": unique_id,
                        "weighted_degree": float(
                            graph.degree(unique_id, weight="weight")
                        ),
                        "violated_commonsense_rule": metadata.get(unique_id, {}).get(
                            "violated_commonsense_rule", ""
                        ),
                        "situation": metadata.get(unique_id, {}).get("situation", ""),
                    }
                    for unique_id in representatives
                ],
            }
        )
        for unique_id in members:
            assignments.append(
                {
                    "id": unique_id,
                    "community_id": community_id,
                    "community_size": len(members),
                    "weighted_degree": float(
                        graph.degree(unique_id, weight="weight")
                    ),
                }
            )

    assignments.sort(key=lambda item: natural_sort_key(item["id"]))
    return assignments, community_records


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            json.dump(record, file, ensure_ascii=False)
            file.write("\n")


def write_communities_markdown(path: Path, communities: list[dict[str, Any]]) -> None:
    lines = ["# SNN Communities", ""]
    for community in communities:
        lines.append(
            f"## Community {community['community_id']} ({community['size']} cases)"
        )
        lines.append("")
        for representative in community["representatives"]:
            lines.append(
                f"- {representative['id']}: "
                f"{representative['violated_commonsense_rule']}"
            )
            if representative["situation"]:
                lines.append(f"  - Situation: {representative['situation']}")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def ensure_output_is_available(output_dir: Path, overwrite: bool) -> None:
    existing = [output_dir / name for name in OUTPUT_FILENAMES if (output_dir / name).exists()]
    if existing and not overwrite:
        paths = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Output files already exist: {paths}. Use --overwrite.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an SNN graph from neighbor sets and run Louvain communities."
    )
    parser.add_argument("--neighbors-file", type=Path, default=DEFAULT_NEIGHBORS_FILE)
    parser.add_argument(
        "--metadata-files",
        "--metadata-file",
        dest="metadata_files",
        type=Path,
        nargs="+",
        default=DEFAULT_METADATA_FILES,
        help="One or more metadata JSON files whose IDs share one neighbor space.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--min-shared-neighbors", type=int, default=DEFAULT_MIN_SHARED_NEIGHBORS
    )
    parser.add_argument(
        "--mutual-only",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_MUTUAL_ONLY,
    )
    parser.add_argument("--resolution", type=float, default=DEFAULT_RESOLUTION)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--representatives", type=int, default=DEFAULT_REPRESENTATIVES)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.neighbors_file.is_file():
        raise FileNotFoundError(f"Neighbor file does not exist: {args.neighbors_file}")
    if not args.metadata_files:
        raise ValueError("At least one metadata file is required.")
    missing_files = [path for path in args.metadata_files if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(f"Metadata file does not exist: {missing_files[0]}")
    if len(args.metadata_files) != len(set(args.metadata_files)):
        raise ValueError("Metadata file arguments contain duplicate paths.")
    if args.resolution <= 0:
        raise ValueError("resolution must be positive.")
    if args.representatives < 1:
        raise ValueError("representatives must be positive.")

    ensure_output_is_available(args.output_dir, args.overwrite)
    ids, neighbor_indices, k = load_neighbors(args.neighbors_file)
    if args.min_shared_neighbors < 0 or args.min_shared_neighbors > k:
        raise ValueError("min_shared_neighbors must be between 0 and neighbor k.")
    metadata, metadata_source_sample_counts = build_metadata_index(
        args.metadata_files
    )
    missing_metadata = [unique_id for unique_id in ids if unique_id not in metadata]
    if missing_metadata:
        raise ValueError(
            f"Metadata is missing {len(missing_metadata)} neighbor ids; "
            f"first={missing_metadata[0]}"
        )

    neighbor_mask = build_neighbor_mask(neighbor_indices)
    edges, shared_counts = build_snn_edges(
        ids=ids,
        neighbor_mask=neighbor_mask,
        k=k,
        min_shared_neighbors=args.min_shared_neighbors,
        mutual_only=args.mutual_only,
    )
    graph = build_graph(ids, edges)
    communities = discover_communities(graph, args.resolution, args.random_seed)
    assignments, community_records = build_community_records(
        graph=graph,
        communities=communities,
        metadata=metadata,
        representative_count=args.representatives,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "snn_edges.jsonl", edges)
    write_jsonl(args.output_dir / "snn_assignments.jsonl", assignments)
    write_json(args.output_dir / "snn_communities.json", community_records)
    write_communities_markdown(
        args.output_dir / "snn_communities.md", community_records
    )

    component_sizes = sorted(
        (len(component) for component in nx.connected_components(graph)), reverse=True
    )
    community_sizes = [len(community) for community in communities]
    nonzero_shared = shared_counts[np.triu(shared_counts > 0, k=1)]
    summary = {
        "neighbors_file": str(args.neighbors_file),
        "metadata_files": [str(path) for path in args.metadata_files],
        "metadata_source_sample_counts": metadata_source_sample_counts,
        "sample_count": len(ids),
        "k": k,
        "min_shared_neighbors": args.min_shared_neighbors,
        "mutual_only": args.mutual_only,
        "resolution": args.resolution,
        "random_seed": args.random_seed,
        "edge_count": len(edges),
        "connected_component_count": len(component_sizes),
        "largest_connected_component": component_sizes[0] if component_sizes else 0,
        "community_count": len(communities),
        "community_sizes": community_sizes,
        "singleton_community_count": sum(size == 1 for size in community_sizes),
        "modularity": float(
            modularity(
                graph,
                communities,
                weight="weight",
                resolution=args.resolution,
            )
        )
        if graph.number_of_edges()
        else None,
        "shared_neighbor_count_nonzero_min": int(nonzero_shared.min())
        if nonzero_shared.size
        else None,
        "shared_neighbor_count_nonzero_max": int(nonzero_shared.max())
        if nonzero_shared.size
        else None,
    }
    write_json(args.output_dir / "snn_summary.json", summary)

    print(
        "Done. "
        f"samples={len(ids)}, k={k}, edges={len(edges)}, "
        f"communities={len(communities)}, modularity={summary['modularity']}, "
        f"output={args.output_dir}"
    )


if __name__ == "__main__":
    main()

