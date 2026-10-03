from __future__ import annotations

import hashlib
import json
from pathlib import Path

from rqs.rq4.code.common.instance_level_context import (
    build_instance_level_context,
    normalise_as_singleton_families,
)


ROOT = Path(__file__).resolve().parents[1]
RQ4 = ROOT / "rqs/rq4"


def _case_ids() -> set[str]:
    path = ROOT / "rqs/rq2/data/case_manifest.jsonl"
    return {
        str(json.loads(line)["case_id"])
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def _file_case_ids(path: Path, pattern: str) -> set[str]:
    return {item.stem.rsplit("_", 1)[-1] for item in path.glob(pattern)}


def test_rq4_bundled_case_coverage() -> None:
    expected = _case_ids()
    assert len(expected) == 151
    for track in ("sequence_context", "image_only"):
        path = RQ4 / f"data/situations/{track}/retrieved"
        assert _file_case_ids(path, "Situations_*.json") == expected
    for baseline in ("visiondroid", "vanilla_mllm", "kuitest"):
        path = RQ4 / f"data/detector_outputs/gemini-3.1-pro-preview/{baseline}"
        assert _file_case_ids(path, "BugReport_*.json") == expected


def test_retrieved_situations_store_complete_top_30() -> None:
    for track in ("sequence_context", "image_only"):
        paths = sorted((RQ4 / f"data/situations/{track}/retrieved").glob("*.json"))
        for path in paths:
            document = json.loads(path.read_text(encoding="utf-8-sig"))
            for situation in document["test_situations"]:
                candidates = situation["related_commonsense"]
                assert len(candidates) == 30
                for candidate in candidates:
                    unit = candidate["commonsense_unit"]
                    assert candidate["id"] == unit["unit_id"]
                    assert unit["situation"].strip()
                    assert unit["commonsense_rule"].strip()


def test_detector_manifest_hashes_and_schema() -> None:
    manifest = json.loads(
        (RQ4 / "data/detector_output_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["file_count"] == 453
    assert manifest["condition_count"] == 3
    assert manifest["metrics_included"] is False
    assert manifest["ground_truth_labels_included"] is False
    for condition in manifest["conditions"]:
        assert len(condition["files"]) == 151
        for entry in condition["files"]:
            path = RQ4 / entry["file"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
            result = json.loads(path.read_text(encoding="utf-8-sig"))
            assert isinstance(result.get("bug_found"), bool)


def test_ablation_and_generalized_conditions_keep_detector_schema() -> None:
    generalized_root = (
        ROOT / "rqs/rq2/data/detector_outputs/gemini-3.1-pro-preview"
    )
    ablation_root = RQ4 / "data/detector_outputs/gemini-3.1-pro-preview"
    for baseline in ("visiondroid", "vanilla_mllm", "kuitest"):
        reference = generalized_root / baseline / "with_commonsense"
        for ablation_path in (ablation_root / baseline).glob("BugReport_*.json"):
            generalized_path = reference / ablation_path.name
            assert generalized_path.is_file()
            ablation = json.loads(ablation_path.read_text(encoding="utf-8-sig"))
            generalized = json.loads(
                generalized_path.read_text(encoding="utf-8-sig")
            )
            assert set(ablation) == set(generalized)


def test_flat_context_and_kuitest_adapter_do_not_create_variants() -> None:
    unit = {
        "id": "G000001",
        "rerank_score": 0.9,
        "commonsense_unit": {
            "unit_id": "G000001",
            "situation": "submitting a completed form",
            "commonsense_rule": "The submitted data should be preserved.",
        },
    }
    document = {
        "test_situations": [
            {
                "start_step": 1,
                "end_step": 2,
                "situation_description": "submit a form",
                "related_commonsense": [unit],
            }
        ]
    }
    context = build_instance_level_context(document, top_k=1)
    assert "submitting a completed form" in context
    assert "The submitted data should be preserved." in context
    converted = normalise_as_singleton_families(document, top_k=1)
    candidate = converted["test_situations"][0]["related_rule_families"][0]
    assert candidate["matched_unit"]["type"] == "BASE"
    assert "variant" not in json.dumps(candidate).lower()
