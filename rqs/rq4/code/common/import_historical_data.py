"""Import the curated Gemini RQ4 artifacts from the baseline source tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from rqs.rq2.code.common.io_utils import write_json


RQ4_ROOT = Path(__file__).resolve().parents[2]
RQ2_ROOT = RQ4_ROOT.parent / "rq2"
MODEL_NAME = "gemini-3.1-pro-preview"
API_MODEL = "gemini-3.1-pro-preview-medium"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def case_ids() -> set[str]:
    result = set()
    with (RQ2_ROOT / "data/case_manifest.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                result.add(str(json.loads(line)["case_id"]))
    if len(result) != 151:
        raise ValueError(f"Expected 151 RQ2 case IDs, found {len(result)}.")
    return result


def copy_situation_track(source: Path, target: Path, expected_ids: set[str]) -> None:
    target.mkdir(parents=True, exist_ok=True)
    paths = sorted(source.glob("Situations_*.json"))
    ids = {path.stem.rsplit("_", 1)[-1] for path in paths}
    if ids != expected_ids:
        raise ValueError(f"Situation coverage mismatch for {source}.")
    for path in paths:
        shutil.copy2(path, target / path.name)


def verify_same_situations(left: Path, right: Path) -> None:
    left_files = {path.name: sha256_file(path) for path in left.glob("Situations_*.json")}
    right_files = {path.name: sha256_file(path) for path in right.glob("Situations_*.json")}
    if left_files != right_files:
        raise ValueError("VisionDroid and KuiTest sequence-context situations differ.")


def copy_detector_outputs(
    sources: dict[str, Path], expected_ids: set[str]
) -> dict:
    conditions = []
    total = 0
    for baseline, source in sources.items():
        target = RQ4_ROOT / "data/detector_outputs" / MODEL_NAME / baseline
        target.mkdir(parents=True, exist_ok=True)
        entries = []
        ids = set()
        for path in sorted(source.glob("BugReport_*.json")):
            case_id = path.stem.rsplit("_", 1)[-1]
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            if not isinstance(data.get("bug_found"), bool):
                raise ValueError(f"{path} has no Boolean bug_found field.")
            ids.add(case_id)
            destination = target / path.name
            shutil.copy2(path, destination)
            entries.append(
                {
                    "case_id": case_id,
                    "file": destination.relative_to(RQ4_ROOT).as_posix(),
                    "sha256": sha256_file(destination),
                }
            )
        if ids != expected_ids:
            raise ValueError(f"Detector coverage mismatch for {baseline}: {len(ids)}.")
        total += len(entries)
        conditions.append(
            {
                "baseline": baseline,
                "paper_model_name": MODEL_NAME,
                "api_model_identifier": API_MODEL,
                "retrieval_mode": "instance_level",
                "commonsense_injected": True,
                "source_directory": {
                    "visiondroid": "baselines/VisionDroid/gemini/ablation/bug_commonsense",
                    "vanilla_mllm": "baselines/VanillaMLLM/gemini/ablation/bug_commonsense",
                    "kuitest": "baselines/KuiTest/ablation/bug_with_commonsense",
                }[baseline],
                "case_ids": sorted(ids, key=int),
                "files": entries,
            }
        )
    if total != 453:
        raise ValueError(f"Expected 453 detector outputs, found {total}.")
    return {
        "schema_version": "1.0",
        "rq_id": "RQ4",
        "output_type": "historical_parsed_detector_outputs",
        "ground_truth_labels_included": False,
        "metrics_included": False,
        "condition_count": 3,
        "case_count_per_condition": 151,
        "file_count": total,
        "conditions": conditions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Root containing the baselines directory.",
    )
    args = parser.parse_args()
    root = args.source_root.resolve()
    ids = case_ids()

    vision = root / "baselines/VisionDroid/gemini/ablation"
    vanilla = root / "baselines/VanillaMLLM/gemini/ablation"
    kuitest = root / "baselines/KuiTest/ablation"
    verify_same_situations(vision / "situations", kuitest / "situations")
    copy_situation_track(
        vision / "situations",
        RQ4_ROOT / "data/situations/sequence_context/retrieved",
        ids,
    )
    copy_situation_track(
        vanilla / "situations",
        RQ4_ROOT / "data/situations/image_only/retrieved",
        ids,
    )
    manifest = copy_detector_outputs(
        {
            "visiondroid": vision / "bug_commonsense",
            "vanilla_mllm": vanilla / "bug_commonsense",
            "kuitest": kuitest / "bug_with_commonsense",
        },
        ids,
    )
    write_json(RQ4_ROOT / "data/detector_output_manifest.json", manifest)
    print("Imported 302 retrieved-situation files and 453 detector outputs.")


if __name__ == "__main__":
    main()
