from __future__ import annotations

import numpy as np

from commonsense_repro.clustering.build_instance_level_commonsense_neighbors import compute_top_neighbors
from commonsense_repro.clustering.discover_snn_communities import build_neighbor_mask, build_snn_edges
from commonsense_repro.retrieval.batch_retrieve_commonsense_library import (
    FamilyEmbeddingStore,
    retrieve_top_families,
)


def test_cosine_neighbors_exclude_self() -> None:
    matrix = np.asarray([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    indices, scores = compute_top_neighbors(matrix, 1)
    assert indices[:, 0].tolist() == [1, 0, 1]
    assert np.all(scores[:, 0] < 1.0)


def test_snn_shared_neighbor_threshold() -> None:
    neighbor_indices = np.asarray([[1, 2], [0, 2], [0, 1]], dtype=np.int64)
    edges, _ = build_snn_edges(
        ids=["A", "B", "C"],
        neighbor_mask=build_neighbor_mask(neighbor_indices),
        k=2,
        min_shared_neighbors=1,
        mutual_only=False,
    )
    assert {(edge["source"], edge["target"]) for edge in edges} == {
        ("A", "B"),
        ("A", "C"),
        ("B", "C"),
    }


def test_family_recall_uses_max_representation_score() -> None:
    representations = [
        {"representation_id": "F1:BASE", "representation_type": "BASE", "family_id": "F1", "member_id": None, "variant_group_index": None, "situation": "base one"},
        {"representation_id": "F1:V1", "representation_type": "VARIANT_MEMBER", "family_id": "F1", "member_id": "U1", "variant_group_index": 0, "situation": "variant one"},
        {"representation_id": "F2:BASE", "representation_type": "BASE", "family_id": "F2", "member_id": None, "variant_group_index": None, "situation": "base two"},
    ]
    matrix = np.asarray([[0.0, 1.0], [1.0, 0.0], [0.8, 0.2]], dtype=np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    store = FamilyEmbeddingStore(
        representations=representations,
        matrix=matrix,
        family_ids=["F1", "F2"],
        indices_by_family={"F1": np.asarray([0, 1]), "F2": np.asarray([2])},
        model="fixture",
        dimensions=2,
    )

    results = retrieve_top_families(np.asarray([1.0, 0.0]), store, 2)
    assert results[0]["family_id"] == "F1"
    assert results[0]["matched_representation"]["type"] == "VARIANT_MEMBER"
    assert results[0]["family_score"] == 1.0
