"""Import the fixed RQ2 cases, situations, and parsed historical outputs.

This maintenance command intentionally copies no screenshots, pickle files,
labels, metrics, raw model responses, or API credentials.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from rqs.rq2.code.common.io_utils import sha256_file, write_json, write_jsonl
from rqs.rq2.code.common.settings import PACKAGE_ROOT, RQ2_ROOT


OUTPUT_NAME = re.compile(r"^BugReport_(?P<app>.+)_(?P<id>\d+)\.json$")

OUTPUT_SOURCES = {
    ("qwen3.7-plus", "visiondroid", "without_commonsense"): [
        "baselines/VisionDroid/bug"
    ],
    ("qwen3.7-plus", "visiondroid", "with_commonsense"): [
        "baselines/VisionDroid/bug_commonsense/positive",
        "baselines/VisionDroid/bug_commonsense/negative",
    ],
    ("qwen3.7-plus", "vanilla_mllm", "without_commonsense"): [
        "baselines/VanillaMLLM/bug"
    ],
    ("qwen3.7-plus", "vanilla_mllm", "with_commonsense"): [
        "baselines/VanillaMLLM/bug_commonsense/positive/positive",
        "baselines/VanillaMLLM/bug_commonsense/negative/negative",
    ],
    ("qwen3.7-plus", "kuitest", "without_commonsense"): [
        "baselines/KuiTest/qwen/bug_with_page_descriptions/positive",
        "baselines/KuiTest/qwen/bug_with_page_descriptions/negative",
    ],
    ("qwen3.7-plus", "kuitest", "with_commonsense"): [
        "baselines/KuiTest/qwen/bug_with_commonsense/positive",
        "baselines/KuiTest/qwen/bug_with_commonsense/negative",
    ],
    ("gemini-3.1-pro-preview", "visiondroid", "without_commonsense"): [
        "baselines/VisionDroid/gemini/bug"
    ],
    ("gemini-3.1-pro-preview", "visiondroid", "with_commonsense"): [
        "baselines/VisionDroid/gemini/bug_commonsense"
    ],
    ("gemini-3.1-pro-preview", "vanilla_mllm", "without_commonsense"): [
        "baselines/VanillaMLLM/gemini/bug"
    ],
    ("gemini-3.1-pro-preview", "vanilla_mllm", "with_commonsense"): [
        "baselines/VanillaMLLM/gemini/bug_commonsense/positive",
        "baselines/VanillaMLLM/gemini/bug_commonsense/negative",
    ],
    ("gemini-3.1-pro-preview", "kuitest", "without_commonsense"): [
        "baselines/KuiTest/bug_with_page_descriptions/positive"
    ],
    ("gemini-3.1-pro-preview", "kuitest", "with_commonsense"): [
        "baselines/KuiTest/bug_with_commonsense/positive"
    ],
}


def normalized_id(value: str) -> str:
    return str(int(float(value)))


def read_cases(dataset_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in ("Odin", "RegDroid"):
        csv_path = dataset_root / dataset / "test_case" / "test_case.csv"
        with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                case_id = normalized_id(row["ID"])
                app = row["AppName"].strip()
                action_dir = (
                    "action_description_data"
                    if (dataset_root / dataset / "action_description_data").is_dir()
                    else "action_description_path"
                )
                base = f"{dataset}"
                rows.append(
                    {
                        "case_id": case_id,
                        "app_name": app,
                        "source_dataset": dataset,
                        "assets": {
                            "test_case_csv": f"{base}/test_case/test_case.csv",
                            "structured_test_case": (
                                f"{base}/struct_case_data/{app}/{case_id}.pkl"
                            ),
                            "action_descriptions": (
                                f"{base}/{action_dir}/{app}/{case_id}.txt"
                            ),
                            "page_descriptions": (
                                f"{base}/page_description_data/{app}/{case_id}.txt"
                            ),
                            "stitched_sequence_image": (
                                f"{base}/images/stitch_images/{app}/{case_id}.jpg"
                            ),
                            "screenshots": f"{base}/images/screenshots/{app}/{case_id}",
                            "annotated_images": (
                                f"{base}/images/annotated_images/{app}/{case_id}"
                            ),
                        },
                    }
                )
    rows.sort(key=lambda item: int(item["case_id"]))
    ids = [row["case_id"] for row in rows]
    if len(rows) != 151 or len(set(ids)) != 151:
        raise ValueError(f"Expected 151 unique cases, found {len(rows)}/{len(set(ids))}.")
    return rows


def copy_situations(nova_root: Path) -> None:
    extracted_dirs = (
        RQ2_ROOT / "data/situations/sequence_context/extracted",
        RQ2_ROOT / "data/situations/image_only/extracted",
    )
    for directory in extracted_dirs:
        count = len(list(directory.glob("Situations_*.json")))
        if count != 151:
            raise ValueError(
                f"Expected 151 extracted situation files in {directory}, found {count}."
            )

    mappings = [
        (
            nova_root / "baselines/VisionDroid/situations",
            RQ2_ROOT / "data/situations/sequence_context/retrieved",
        ),
        (
            nova_root / "baselines/VanillaMLLM/situations",
            RQ2_ROOT / "data/situations/image_only/retrieved",
        ),
    ]
    for source, target in mappings:
        files = sorted(source.glob("Situations_*.json"))
        if len(files) != 151:
            raise ValueError(f"Expected 151 situation files in {source}, found {len(files)}.")
        target.mkdir(parents=True, exist_ok=True)
        for path in files:
            shutil.copy2(path, target / path.name)


def collect_condition_files(nova_root: Path, sources: list[str]) -> list[Path]:
    files: dict[str, Path] = {}
    for logical_source in sources:
        source = nova_root / logical_source
        # Each mapping points at an exact historical result directory. Do not
        # descend into nested test/backup folders left by unrelated trials.
        for path in source.glob("BugReport_*.json"):
            if not OUTPUT_NAME.match(path.name):
                continue
            if path.name in files:
                raise ValueError(
                    f"Duplicate output {path.name} in {files[path.name]} and {path}."
                )
            files[path.name] = path
    return [files[name] for name in sorted(files)]


def copy_outputs(nova_root: Path, case_ids: set[str]) -> dict[str, Any]:
    conditions: list[dict[str, Any]] = []
    shared_ids: set[str] | None = None
    total = 0
    for (model, baseline, condition), sources in OUTPUT_SOURCES.items():
        files = collect_condition_files(nova_root, sources)
        if len(files) != 151:
            raise ValueError(
                f"{model}/{baseline}/{condition} has {len(files)} outputs, expected 151."
            )
        target = RQ2_ROOT / "data/detector_outputs" / model / baseline / condition
        target.mkdir(parents=True, exist_ok=True)
        entries = []
        ids = set()
        for source in files:
            match = OUTPUT_NAME.match(source.name)
            assert match is not None
            case_id = match.group("id")
            ids.add(case_id)
            value = json.loads(source.read_text(encoding="utf-8"))
            if not isinstance(value.get("bug_found"), bool):
                raise ValueError(f"{source} has no Boolean bug_found field.")
            destination = target / source.name
            shutil.copy2(source, destination)
            entries.append(
                {
                    "case_id": case_id,
                    "file": destination.relative_to(RQ2_ROOT).as_posix(),
                    "sha256": sha256_file(destination),
                }
            )
        if ids != case_ids:
            raise ValueError(f"Case coverage mismatch for {model}/{baseline}/{condition}.")
        if shared_ids is None:
            shared_ids = ids
        elif ids != shared_ids:
            raise ValueError("Detector condition ID sets are inconsistent.")
        total += len(entries)
        conditions.append(
            {
                "baseline": baseline,
                "paper_model_name": model,
                "api_model_identifier": (
                    "gemini-3.1-pro-preview-medium"
                    if model == "gemini-3.1-pro-preview"
                    else "qwen3.7-plus"
                ),
                "condition": condition,
                "commonsense_injected": condition == "with_commonsense",
                "source_directories": sources,
                "case_ids": sorted(ids, key=int),
                "files": entries,
            }
        )
    if total != 1812:
        raise ValueError(f"Expected 1812 detector files, found {total}.")
    return {
        "schema_version": "1.0",
        "output_type": "historical_parsed_detector_outputs",
        "ground_truth_labels_included": False,
        "metrics_included": False,
        "condition_count": len(conditions),
        "case_count_per_condition": 151,
        "file_count": total,
        "conditions": conditions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nova-root", type=Path, required=True)
    args = parser.parse_args()
    nova_root = args.nova_root.resolve()
    cases = read_cases(nova_root / "datasets")
    write_jsonl(RQ2_ROOT / "data/case_manifest.jsonl", cases)
    copy_situations(nova_root)
    manifest = copy_outputs(nova_root, {row["case_id"] for row in cases})
    write_json(RQ2_ROOT / "data/detector_output_manifest.json", manifest)
    print("Imported 151 cases, 604 situation files, and 1812 detector outputs.")


if __name__ == "__main__":
    main()
