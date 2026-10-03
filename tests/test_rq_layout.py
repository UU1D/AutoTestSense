from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_rq2_and_rq4_reference_distinct_retrieval_methods() -> None:
    rq2 = json.loads((ROOT / "rqs/rq2/experiment.json").read_text(encoding="utf-8"))
    rq4 = json.loads((ROOT / "rqs/rq4/experiment.json").read_text(encoding="utf-8"))

    assert rq2["retrieval_mode"] == "library"
    assert rq4["retrieval_mode"] == "instance_level"
    assert "generalization_records" in rq2["inputs"]
    assert "instance_level_commonsense" in rq4["inputs"]


def test_rq_experiment_paths_resolve_from_package_root() -> None:
    for rq_id in ("rq2", "rq4"):
        experiment = json.loads(
            (ROOT / f"rqs/{rq_id}/experiment.json").read_text(encoding="utf-8")
        )
        assert (ROOT / experiment["runner"]).is_file()
        for implementation in experiment["implementation"]:
            assert (ROOT / implementation).is_file()
