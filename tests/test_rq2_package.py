from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
RQ2_ROOT = PACKAGE_ROOT / "rqs/rq2"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_rq2_case_and_situation_coverage() -> None:
    cases = read_jsonl(RQ2_ROOT / "data/case_manifest.jsonl")
    case_ids = {row["case_id"] for row in cases}
    assert len(cases) == len(case_ids) == 151
    assert all("label" not in row and "bug_found" not in row for row in cases)

    for track in ("sequence_context", "image_only"):
        for stage in ("extracted", "retrieved"):
            paths = list((RQ2_ROOT / f"data/situations/{track}/{stage}").glob("Situations_*.json"))
            assert len(paths) == 151
            ids = {path.stem.rsplit("_", 1)[-1] for path in paths}
            assert ids == case_ids


def test_rq2_historical_detector_outputs_are_complete_and_valid() -> None:
    manifest = json.loads(
        (RQ2_ROOT / "data/detector_output_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["condition_count"] == 12
    assert manifest["case_count_per_condition"] == 151
    assert manifest["file_count"] == 1812
    assert manifest["ground_truth_labels_included"] is False
    assert manifest["metrics_included"] is False

    shared_ids = None
    seen_files = 0
    for condition in manifest["conditions"]:
        ids = set(condition["case_ids"])
        assert len(ids) == 151
        shared_ids = ids if shared_ids is None else shared_ids
        assert ids == shared_ids
        assert len(condition["files"]) == 151
        for record in condition["files"]:
            path = RQ2_ROOT / record["file"]
            value = json.loads(path.read_text(encoding="utf-8"))
            assert type(value.get("bug_found")) is bool
            assert sha256(path) == record["sha256"]
            seen_files += 1
    assert seen_files == 1812


def test_rq2_matrix_and_kuitest_control_metadata() -> None:
    experiment = json.loads((RQ2_ROOT / "experiment.json").read_text(encoding="utf-8"))
    assert experiment["experiment_count"] == 12
    assert experiment["historical_situation_extraction_model"] == "not_recorded"
    assert experiment["models"]["gemini"]["paper_name"] == "gemini-3.1-pro-preview"
    assert (
        experiment["models"]["gemini"]["api_model_identifier"]
        == "gemini-3.1-pro-preview-medium"
    )
    kuitest = experiment["baselines"]["kuitest"]
    assert kuitest["without_commonsense_variant"] == "bug_with_page_descriptions"
    assert kuitest["with_commonsense_variant"] == "bug_with_commonsense"
    assert experiment["retrieval"] == {
        "aggregation": "max(base_score, variant_scores)",
        "cosine_recall_k": 60,
        "rerank_k": 40,
        "stored_enriched_k": 30,
        "detector_injection_k": 10,
    }


def test_rq2_source_boundary_and_empty_evaluation() -> None:
    assert not any((RQ2_ROOT / "evaluation").iterdir())
    required = [
        "code/common/openai_compatible_client.py",
        "code/situation_extraction/run_situations.py",
        "code/situation_extraction/image_only/run_situations.py",
        "code/baselines/visiondroid/run.py",
        "code/baselines/visiondroid/run_commonsense.py",
        "code/baselines/vanilla_mllm/run.py",
        "code/baselines/vanilla_mllm/run_commonsense.py",
        "code/baselines/kuitest/run.py",
        "code/baselines/kuitest/run_commonsense.py",
    ]
    for relative in required:
        assert (RQ2_ROOT / relative).is_file()
    text = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in RQ2_ROOT.rglob("*.py")
    )
    assert "D:\\Desktop\\myprojects\\lab_experiment\\NovaOracle" not in text
    assert "sk-" not in text


def test_rq2_twelve_condition_dry_run(tmp_path: Path) -> None:
    command = [
        sys.executable,
        str(RQ2_ROOT / "run.py"),
        "detect",
        "--dataset-root",
        str(tmp_path),
        "--baseline",
        "all",
        "--model",
        "all",
        "--injection",
        "both",
        "--limit",
        "1",
        "--dry-run",
    ]
    result = subprocess.run(
        command,
        cwd=PACKAGE_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("COMMAND:") == 12
    assert result.stdout.count("--with-page-descriptions") == 2


def test_commonsense_variants_preserve_detector_output_contract() -> None:
    pairs = [
        ("visiondroid", "bug_detect.py", "bug_detect_commonsense.py"),
        ("vanilla_mllm", "bug_detect.py", "bug_detect_commonsense.py"),
        ("kuitest", "response_verification.py", "response_verification_commonsense.py"),
    ]
    for baseline, base_name, injected_name in pairs:
        root = RQ2_ROOT / "code/baselines" / baseline
        base = (root / base_name).read_text(encoding="utf-8")
        injected = (root / injected_name).read_text(encoding="utf-8")
        assert "commonsense" in injected.lower()
        if baseline != "kuitest":
            assert "BugDetectInterface" in injected
            assert "bug_found" in base
        else:
            assert "ResponseVerification" in injected
            assert "judgement" in base and "judgement" in injected
