"""Refine oversized SNN communities into Leiden-based LLM candidate batches."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
from pathlib import Path
from typing import Any, Iterable

import networkx as nx


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DEFAULT_INPUT_DIR = Path(
    "work/stage2/snn/"
    "gemini_v1_2_combined_rerank_k30_non_mutual_s15_r0.8"
)
DEFAULT_METADATA_FILES = [
    Path("data/checkpoints/instance_level_commonsense/gemini_v1_2_extract.json"),
    Path("data/checkpoints/instance_level_commonsense/gemini_v1_2_github.json"),
    Path("data/checkpoints/instance_level_commonsense/gemini_v1_2_negative.json"),
]
DEFAULT_OUTPUT_DIR = Path(
    "work/snn_refined/"
    "gemini_v1_2_non_mutual_s15_leiden_max25"
)

DEFAULT_MAX_COMMUNITY_SIZE = 25
DEFAULT_RESOLUTION = 1.0
DEFAULT_RESOLUTION_MULTIPLIER = 1.5
DEFAULT_MAX_REFINEMENT_DEPTH = 10
DEFAULT_RANDOM_SEED = 42
DEFAULT_N_ITERATIONS = -1
DEFAULT_REPRESENTATIVES = 10

EDGES_FILENAME = "snn_edges.jsonl"
COMMUNITIES_FILENAME = "snn_communities.json"
OUTPUT_FILENAMES = [
    "refined_assignments.jsonl",
    "refined_communities.json",
    "refined_communities.md",
    "llm_batches.jsonl",
    "singletons.jsonl",
    "refinement_summary.json",
]


def natural_sort_key(value: str) -> tuple[tuple[int, Any], ...]:
    parts = re.split(r"(\d+)", value.casefold())
    return tuple(
        (1, int(part)) if part.isdigit() else (0, part)
        for part in parts
        if part
    )


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


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


def load_parent_communities(
    path: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    data = read_json(path)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}.")

    communities: list[dict[str, Any]] = []
    all_ids: list[str] = []
    seen_community_ids: set[int] = set()
    seen_member_ids: set[str] = set()

    for index, record in enumerate(data):
        if not isinstance(record, dict):
            raise ValueError(f"Expected an object at {path} index {index}.")
        community_id = record.get("community_id")
        member_ids = record.get("member_ids")
        if not isinstance(community_id, int):
            raise ValueError(f"Invalid community_id at {path} index {index}.")
        if community_id in seen_community_ids:
            raise ValueError(f"Duplicate parent community_id: {community_id}")
        if not isinstance(member_ids, list) or not member_ids:
            raise ValueError(f"Community {community_id} has no member_ids.")
        if record.get("size") != len(member_ids):
            raise ValueError(f"Size mismatch in parent community {community_id}.")

        validated_ids: list[str] = []
        for unique_id in member_ids:
            if not isinstance(unique_id, str) or not unique_id.strip():
                raise ValueError(f"Invalid member ID in community {community_id}.")
            if unique_id in seen_member_ids:
                raise ValueError(f"Member ID appears in multiple communities: {unique_id}")
            seen_member_ids.add(unique_id)
            validated_ids.append(unique_id)
            all_ids.append(unique_id)

        seen_community_ids.add(community_id)
        communities.append(
            {
                "community_id": community_id,
                "member_ids": validated_ids,
                "size": len(validated_ids),
            }
        )

    communities.sort(key=lambda item: item["community_id"])
    return communities, all_ids


def load_snn_graph(path: Path, all_ids: list[str]) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(all_ids)
    known_ids = set(all_ids)
    seen_edges: set[tuple[str, str]] = set()

    for line_number, edge in enumerate(read_jsonl(path), start=1):
        source = edge.get("source")
        target = edge.get("target")
        weight = edge.get("shared_neighbor_jaccard")
        if not isinstance(source, str) or source not in known_ids:
            raise ValueError(f"Unknown edge source at {path}:{line_number}: {source!r}")
        if not isinstance(target, str) or target not in known_ids:
            raise ValueError(f"Unknown edge target at {path}:{line_number}: {target!r}")
        if source == target:
            raise ValueError(f"Self edge at {path}:{line_number}: {source}")
        if not isinstance(weight, (int, float)) or weight < 0:
            raise ValueError(f"Invalid edge weight at {path}:{line_number}.")

        edge_key = tuple(sorted((source, target), key=natural_sort_key))
        if edge_key in seen_edges:
            raise ValueError(f"Duplicate SNN edge at {path}:{line_number}: {edge_key}")
        seen_edges.add(edge_key)
        graph.add_edge(source, target, weight=float(weight))

    return graph


def load_metadata(
    paths: list[Path],
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    metadata: dict[str, dict[str, Any]] = {}
    source_counts: dict[str, int] = {}

    for path in paths:
        data = read_json(path)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON array in {path}.")
        source_counts[str(path)] = len(data)
        for index, record in enumerate(data):
            if not isinstance(record, dict):
                raise ValueError(f"Expected an object at {path} index {index}.")
            unique_id = record.get("id")
            if not isinstance(unique_id, str) or not unique_id.strip():
                raise ValueError(f"Missing id at {path} index {index}.")
            if unique_id in metadata:
                raise ValueError(
                    f"Duplicate metadata id across input files: {unique_id}"
                )
            metadata[unique_id] = record

    return metadata, source_counts


def load_leiden_modules() -> tuple[Any, Any]:
    try:
        import igraph as ig
        import leidenalg as la
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Leiden refinement requires igraph and leidenalg. Install them with "
            "`python -m pip install igraph leidenalg`."
        ) from error
    return ig, la


def to_igraph(subgraph: nx.Graph, ig: Any) -> Any:
    node_ids = sorted((str(node) for node in subgraph.nodes), key=natural_sort_key)
    node_to_index = {unique_id: index for index, unique_id in enumerate(node_ids)}
    edges: list[tuple[int, int]] = []
    weights: list[float] = []

    for source, target, attributes in subgraph.edges(data=True):
        edges.append((node_to_index[str(source)], node_to_index[str(target)]))
        weights.append(float(attributes["weight"]))

    graph = ig.Graph(n=len(node_ids), edges=edges, directed=False)
    graph.vs["name"] = node_ids
    graph.es["weight"] = weights
    return graph


def sort_member_groups(groups: Iterable[set[str]]) -> list[set[str]]:
    return sorted(
        groups,
        key=lambda members: (
            -len(members),
            natural_sort_key(min(members, key=natural_sort_key)),
        ),
    )


def run_leiden(
    subgraph: nx.Graph,
    *,
    max_community_size: int,
    resolution: float,
    resolution_multiplier: float,
    max_refinement_depth: int,
    random_seed: int,
    n_iterations: int,
    ig: Any,
    la: Any,
) -> tuple[list[set[str]], list[dict[str, Any]]]:
    if subgraph.number_of_edges() == 0:
        groups = [{str(node)} for node in subgraph.nodes]
        return sort_member_groups(groups), []

    runs: list[dict[str, Any]] = []

    def refine(current_graph: nx.Graph, current_resolution: float, depth: int) -> list[set[str]]:
        members = {str(node) for node in current_graph.nodes}
        if len(members) <= max_community_size:
            return [members]
        if current_graph.number_of_edges() == 0:
            return [{unique_id} for unique_id in members]
        if depth > max_refinement_depth:
            raise RuntimeError(
                "Leiden could not reduce a community to the requested maximum: "
                f"size={len(members)}, resolution={current_resolution}."
            )

        igraph_graph = to_igraph(current_graph, ig)
        partition = la.find_partition(
            igraph_graph,
            la.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=current_resolution,
            n_iterations=n_iterations,
            seed=random_seed,
        )
        groups = [
            {
                str(igraph_graph.vs[vertex_index]["name"])
                for vertex_index in cluster
            }
            for cluster in partition
        ]
        groups = sort_member_groups(groups)
        runs.append(
            {
                "depth": depth,
                "input_size": len(members),
                "input_edge_count": current_graph.number_of_edges(),
                "resolution": current_resolution,
                "output_sizes": [len(group) for group in groups],
                "quality": float(partition.quality()),
            }
        )

        next_resolution = current_resolution * resolution_multiplier
        if len(groups) == 1 and groups[0] == members:
            return refine(current_graph, next_resolution, depth + 1)

        refined_groups: list[set[str]] = []
        for group in groups:
            if len(group) <= max_community_size:
                refined_groups.append(group)
            else:
                refined_groups.extend(
                    refine(
                        current_graph.subgraph(group).copy(),
                        next_resolution,
                        depth + 1,
                    )
                )
        return refined_groups

    groups = sort_member_groups(refine(subgraph, resolution, 1))
    return groups, runs


def unit_record(unique_id: str, metadata: dict[str, dict[str, Any]]) -> dict[str, Any]:
    record = metadata[unique_id]
    return {
        "id": unique_id,
        "violated_commonsense_rule": record.get("violated_commonsense_rule", ""),
        "situation": record.get("situation", ""),
    }


def rank_members(subgraph: nx.Graph, members: set[str]) -> list[str]:
    child_graph = subgraph.subgraph(members)
    return sorted(
        members,
        key=lambda unique_id: (
            -child_graph.degree(unique_id, weight="weight"),
            natural_sort_key(unique_id),
        ),
    )


def build_outputs(
    *,
    graph: nx.Graph,
    parent_communities: list[dict[str, Any]],
    metadata: dict[str, dict[str, Any]],
    max_community_size: int,
    resolution: float,
    resolution_multiplier: float,
    max_refinement_depth: int,
    random_seed: int,
    n_iterations: int,
    representative_count: int,
    ig: Any,
    la: Any,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    assignments: list[dict[str, Any]] = []
    refined_communities: list[dict[str, Any]] = []
    llm_batches: list[dict[str, Any]] = []
    singletons: list[dict[str, Any]] = []
    parent_refinements: list[dict[str, Any]] = []

    for parent in parent_communities:
        parent_id = int(parent["community_id"])
        parent_members = set(parent["member_ids"])
        parent_size = len(parent_members)
        parent_graph = graph.subgraph(parent_members).copy()
        was_refined = parent_size > max_community_size

        if was_refined:
            child_groups, leiden_runs = run_leiden(
                parent_graph,
                max_community_size=max_community_size,
                resolution=resolution,
                resolution_multiplier=resolution_multiplier,
                max_refinement_depth=max_refinement_depth,
                random_seed=random_seed,
                n_iterations=n_iterations,
                ig=ig,
                la=la,
            )
        else:
            child_groups = [parent_members]
            leiden_runs = []

        parent_refinements.append(
            {
                "parent_community_id": parent_id,
                "parent_size": parent_size,
                "internal_edge_count": parent_graph.number_of_edges(),
                "was_refined": was_refined,
                "child_count": len(child_groups),
                "child_sizes": [len(group) for group in child_groups],
                "leiden_runs": leiden_runs,
            }
        )

        for child_index, members in enumerate(child_groups):
            refined_id = f"{parent_id}.{child_index}"
            child_size = len(members)
            if child_size > max_community_size:
                raise ValueError(
                    f"Leiden returned oversized community {refined_id}: {child_size}"
                )

            ranked_members = rank_members(parent_graph, members)
            member_ids = sorted(members, key=natural_sort_key)
            representatives = ranked_members[:representative_count]
            community_record = {
                "refined_community_id": refined_id,
                "parent_community_id": parent_id,
                "parent_size": parent_size,
                "size": child_size,
                "was_refined": was_refined,
                "member_ids": member_ids,
                "representatives": [
                    unit_record(unique_id, metadata)
                    for unique_id in representatives
                ],
            }
            refined_communities.append(community_record)

            for unique_id in member_ids:
                assignments.append(
                    {
                        "id": unique_id,
                        "parent_community_id": parent_id,
                        "refined_community_id": refined_id,
                        "parent_size": parent_size,
                        "refined_size": child_size,
                        "was_refined": was_refined,
                    }
                )

            if child_size == 1:
                singleton = unit_record(member_ids[0], metadata)
                singleton.update(
                    {
                        "parent_community_id": parent_id,
                        "refined_community_id": refined_id,
                        "was_refined": was_refined,
                    }
                )
                singletons.append(singleton)
            else:
                llm_batches.append(
                    {
                        "batch_id": refined_id,
                        "parent_community_id": parent_id,
                        "parent_size": parent_size,
                        "size": child_size,
                        "was_refined": was_refined,
                        "units": [
                            unit_record(unique_id, metadata)
                            for unique_id in member_ids
                        ],
                    }
                )

    assignments.sort(key=lambda item: natural_sort_key(item["id"]))
    return (
        assignments,
        refined_communities,
        llm_batches,
        singletons,
        parent_refinements,
    )


def validate_outputs(
    *,
    input_ids: list[str],
    assignments: list[dict[str, Any]],
    refined_communities: list[dict[str, Any]],
    llm_batches: list[dict[str, Any]],
    singletons: list[dict[str, Any]],
    max_community_size: int,
) -> None:
    output_ids = [str(record["id"]) for record in assignments]
    if len(output_ids) != len(set(output_ids)):
        raise ValueError("Refined assignments contain duplicate IDs.")
    if set(output_ids) != set(input_ids):
        missing = sorted(set(input_ids) - set(output_ids), key=natural_sort_key)
        extra = sorted(set(output_ids) - set(input_ids), key=natural_sort_key)
        raise ValueError(
            f"Refinement changed the ID set: missing={missing[:1]}, extra={extra[:1]}"
        )
    if any(record["size"] > max_community_size for record in refined_communities):
        raise ValueError("A refined community exceeds max_community_size.")

    batch_ids = {
        unit["id"]
        for batch in llm_batches
        for unit in batch["units"]
    }
    singleton_ids = {record["id"] for record in singletons}
    if batch_ids & singleton_ids:
        raise ValueError("An ID appears in both LLM batches and singletons.")
    if batch_ids | singleton_ids != set(input_ids):
        raise ValueError("LLM batches and singletons do not cover every input ID.")
    if any(not 2 <= batch["size"] <= max_community_size for batch in llm_batches):
        raise ValueError("An LLM batch is outside the allowed size range.")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")
    temporary_path.replace(path)


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        for record in records:
            json.dump(record, file, ensure_ascii=False)
            file.write("\n")
    temporary_path.replace(path)


def write_markdown(path: Path, communities: list[dict[str, Any]]) -> None:
    lines = ["# Leiden-Refined SNN Communities", ""]
    for community in communities:
        lines.append(
            f"## Batch {community['refined_community_id']} "
            f"({community['size']} cases)"
        )
        lines.append("")
        lines.append(
            f"Parent community: {community['parent_community_id']} "
            f"({community['parent_size']} cases); "
            f"refined: {str(community['was_refined']).lower()}"
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


def ensure_output_available(output_dir: Path, overwrite: bool) -> None:
    existing = [
        output_dir / filename
        for filename in OUTPUT_FILENAMES
        if (output_dir / filename).exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(
            f"Output already exists: {existing[0]}. Use --overwrite to rebuild."
        )


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Refine SNN communities larger than a configured maximum with Leiden."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument(
        "--metadata-files",
        "--metadata-file",
        dest="metadata_files",
        type=Path,
        nargs="+",
        default=DEFAULT_METADATA_FILES,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--max-community-size", type=int, default=DEFAULT_MAX_COMMUNITY_SIZE
    )
    parser.add_argument("--resolution", type=float, default=DEFAULT_RESOLUTION)
    parser.add_argument(
        "--resolution-multiplier",
        type=float,
        default=DEFAULT_RESOLUTION_MULTIPLIER,
    )
    parser.add_argument(
        "--max-refinement-depth",
        type=int,
        default=DEFAULT_MAX_REFINEMENT_DEPTH,
    )
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--n-iterations", type=int, default=DEFAULT_N_ITERATIONS)
    parser.add_argument("--representatives", type=int, default=DEFAULT_REPRESENTATIVES)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    edges_file = args.input_dir / EDGES_FILENAME
    communities_file = args.input_dir / COMMUNITIES_FILENAME

    for path in (edges_file, communities_file):
        if not path.is_file():
            raise FileNotFoundError(f"Required SNN file does not exist: {path}")
    if not args.metadata_files:
        raise ValueError("At least one metadata file is required.")
    missing_metadata_files = [
        path for path in args.metadata_files if not path.is_file()
    ]
    if missing_metadata_files:
        raise FileNotFoundError(
            f"Metadata file does not exist: {missing_metadata_files[0]}"
        )
    if len(args.metadata_files) != len(set(args.metadata_files)):
        raise ValueError("Metadata file arguments contain duplicate paths.")
    if args.max_community_size < 2:
        raise ValueError("max_community_size must be at least 2.")
    if args.resolution <= 0:
        raise ValueError("resolution must be positive.")
    if args.resolution_multiplier <= 1:
        raise ValueError("resolution_multiplier must be greater than 1.")
    if args.max_refinement_depth < 1:
        raise ValueError("max_refinement_depth must be positive.")
    if args.n_iterations == 0:
        raise ValueError("n_iterations cannot be zero.")
    if args.representatives < 1:
        raise ValueError("representatives must be positive.")

    ensure_output_available(args.output_dir, args.overwrite)
    parent_communities, input_ids = load_parent_communities(communities_file)
    graph = load_snn_graph(edges_file, input_ids)
    metadata, metadata_source_counts = load_metadata(args.metadata_files)
    missing_metadata = sorted(set(input_ids) - set(metadata), key=natural_sort_key)
    if missing_metadata:
        raise ValueError(
            f"Metadata is missing {len(missing_metadata)} SNN IDs; "
            f"first={missing_metadata[0]}"
        )

    large_parent_count = sum(
        community["size"] > args.max_community_size
        for community in parent_communities
    )
    ig, la = load_leiden_modules()
    (
        assignments,
        refined_communities,
        llm_batches,
        singletons,
        parent_refinements,
    ) = build_outputs(
        graph=graph,
        parent_communities=parent_communities,
        metadata=metadata,
        max_community_size=args.max_community_size,
        resolution=args.resolution,
        resolution_multiplier=args.resolution_multiplier,
        max_refinement_depth=args.max_refinement_depth,
        random_seed=args.random_seed,
        n_iterations=args.n_iterations,
        representative_count=args.representatives,
        ig=ig,
        la=la,
    )
    validate_outputs(
        input_ids=input_ids,
        assignments=assignments,
        refined_communities=refined_communities,
        llm_batches=llm_batches,
        singletons=singletons,
        max_community_size=args.max_community_size,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "refined_assignments.jsonl", assignments)
    write_json(args.output_dir / "refined_communities.json", refined_communities)
    write_markdown(
        args.output_dir / "refined_communities.md", refined_communities
    )
    write_jsonl(args.output_dir / "llm_batches.jsonl", llm_batches)
    write_jsonl(args.output_dir / "singletons.jsonl", singletons)

    refined_sizes = [community["size"] for community in refined_communities]
    summary = {
        "input_dir": str(args.input_dir),
        "edges_file": str(edges_file),
        "communities_file": str(communities_file),
        "metadata_files": [str(path) for path in args.metadata_files],
        "metadata_source_counts": metadata_source_counts,
        "algorithm": "Leiden",
        "partition_type": "RBConfigurationVertexPartition",
        "igraph_version": package_version("igraph"),
        "leidenalg_version": package_version("leidenalg"),
        "resolution": args.resolution,
        "resolution_multiplier": args.resolution_multiplier,
        "max_refinement_depth": args.max_refinement_depth,
        "random_seed": args.random_seed,
        "n_iterations": args.n_iterations,
        "max_community_size": args.max_community_size,
        "input_sample_count": len(input_ids),
        "parent_community_count": len(parent_communities),
        "large_parent_count": large_parent_count,
        "refined_community_count": len(refined_communities),
        "llm_batch_count": len(llm_batches),
        "singleton_count": len(singletons),
        "largest_refined_community": max(refined_sizes, default=0),
        "refined_community_sizes": refined_sizes,
        "parent_refinements": parent_refinements,
    }
    write_json(args.output_dir / "refinement_summary.json", summary)

    print(
        "Done. "
        f"samples={len(input_ids)}, parents={len(parent_communities)}, "
        f"large_parents={large_parent_count}, "
        f"refined_communities={len(refined_communities)}, "
        f"llm_batches={len(llm_batches)}, singletons={len(singletons)}, "
        f"max_size={max(refined_sizes, default=0)}, output={args.output_dir}"
    )


if __name__ == "__main__":
    main()

