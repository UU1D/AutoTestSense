"""Unified RQ2 reproduction entry point."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path


RQ2_ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = RQ2_ROOT.parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from rqs.rq2.code.common.runtime import case_ids, run_detector, run_situation_extraction
from rqs.rq2.code.common.settings import (
    BASELINES,
    CONDITIONS,
    DEFAULT_ENRICH_K,
    DEFAULT_INJECTION_K,
    DEFAULT_RECALL_K,
    DEFAULT_RERANK_K,
    MODEL_ALIASES,
    WORK_ROOT,
    model_spec,
)


def add_common_arguments(parser: argparse.ArgumentParser, *, dataset_root: bool = False) -> None:
    if dataset_root:
        parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--ids", nargs="+", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--top-k", type=int, default=DEFAULT_INJECTION_K)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor", help="validate dependencies, configuration, and RQ2 data")

    situations = subparsers.add_parser("situations", help="extract both situation representations")
    add_common_arguments(situations, dataset_root=True)
    situations.add_argument("--model", choices=MODEL_ALIASES, default="qwen")

    retrieve = subparsers.add_parser("retrieve", help="retrieve generalized commonsense")
    add_common_arguments(retrieve)

    detect = subparsers.add_parser("detect", help="run one or more bug detectors")
    add_common_arguments(detect, dataset_root=True)
    detect.add_argument("--baseline", choices=(*BASELINES, "all"), default="all")
    detect.add_argument("--model", choices=(*MODEL_ALIASES, "all"), default="all")
    detect.add_argument(
        "--injection", choices=("without", "with", "both"), default="both"
    )

    all_command = subparsers.add_parser("all", help="run situations, retrieval, and 12 detectors")
    add_common_arguments(all_command, dataset_root=True)
    all_command.add_argument("--situation-model", choices=MODEL_ALIASES, default="qwen")
    return parser


def count_json(path: Path, pattern: str = "*.json") -> int:
    return sum(1 for item in path.glob(pattern) if item.is_file())


def doctor() -> int:
    errors = []
    for module in ("pandas", "PIL", "requests", "numpy", "dotenv"):
        if importlib.util.find_spec(module) is None:
            errors.append(f"missing Python dependency: {module}")
    expected = {
        "case manifest": (RQ2_ROOT / "data/case_manifest.jsonl", 151),
        "sequence extracted": (RQ2_ROOT / "data/situations/sequence_context/extracted", 151),
        "sequence retrieved": (RQ2_ROOT / "data/situations/sequence_context/retrieved", 151),
        "image extracted": (RQ2_ROOT / "data/situations/image_only/extracted", 151),
        "image retrieved": (RQ2_ROOT / "data/situations/image_only/retrieved", 151),
    }
    for label, (path, expected_count) in expected.items():
        count = (
            sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
            if path.is_file()
            else count_json(path, "Situations_*.json") if path.is_dir() else -1
        )
        if count != expected_count:
            errors.append(f"{label}: expected {expected_count}, found {count}")
    detector_count = count_json(RQ2_ROOT / "data/detector_outputs")
    detector_count = sum(
        1 for path in (RQ2_ROOT / "data/detector_outputs").rglob("BugReport_*.json")
    )
    if detector_count != 1812:
        errors.append(f"detector outputs: expected 1812, found {detector_count}")
    for alias in MODEL_ALIASES:
        spec = model_spec(alias)
        missing = [
            name
            for name, value in (
                (f"RQ2_{alias.upper()}_API_KEY", spec.api_key),
                (f"RQ2_{alias.upper()}_URL", spec.api_url),
                (f"RQ2_{alias.upper()}_MODEL", spec.api_model),
            )
            if not value
        ]
        if missing:
            print(f"WARN {alias}: missing {', '.join(missing)}")
    if errors:
        for error in errors:
            print(f"ERROR {error}")
        return 1
    print("RQ2 package is structurally valid: 151 cases and 1812 detector outputs.")
    return 0


def select_input_dir(track: str, ids: list[int] | None, limit: int | None) -> Path:
    generated = WORK_ROOT / "situations" / track / "extracted"
    bundled = RQ2_ROOT / "data/situations" / track / "extracted"
    if ids is None and limit is None:
        generated_count = sum(1 for _ in generated.glob("Situations_*.json"))
        return generated if generated_count == 151 else bundled
    selected_ids = case_ids(ids, limit)
    selected = set(selected_ids)
    fingerprint = hashlib.sha256(
        ",".join(selected_ids).encode("ascii")
    ).hexdigest()[:12]
    target = WORK_ROOT / "_retrieval_input" / f"{track}_{fingerprint}"
    target.mkdir(parents=True, exist_ok=True)
    by_id = {
        path.stem.rsplit("_", 1)[-1]: path
        for source in (bundled, generated)
        for path in source.glob("Situations_*.json")
    }
    for case_id in selected:
        path = by_id.get(case_id)
        if path is None:
            raise FileNotFoundError(f"No {track} situation file for case {case_id}.")
        shutil.copy2(path, target / path.name)
    return target


def run_retrieval(args: argparse.Namespace) -> int:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PACKAGE_ROOT)
    for track in ("sequence_context", "image_only"):
        input_dir = (
            RQ2_ROOT / "data/situations" / track / "extracted"
            if args.dry_run
            else select_input_dir(track, args.ids, args.limit)
        )
        raw_dir = WORK_ROOT / "situations" / track / "retrieval_raw"
        enriched_dir = WORK_ROOT / "situations" / track / "retrieved"
        retrieve = [
            sys.executable,
            "-m",
            "commonsense_repro.retrieval.batch_retrieve_commonsense_library",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(raw_dir),
            "--embeddings-file",
            "data/reference/commonsense_library/commonsense_library_situation_embeddings.jsonl",
            "--recall-k",
            str(DEFAULT_RECALL_K),
            "--rerank-top-k",
            str(DEFAULT_RERANK_K),
        ]
        if args.overwrite:
            retrieve.append("--overwrite")
        enrich = [
            sys.executable,
            "-m",
            "commonsense_repro.retrieval.enrich_retrieved_commonsense_library",
            "--input-dir",
            str(raw_dir),
            "--output-dir",
            str(enriched_dir),
            "--catalog-file",
            "data/reference/commonsense_library/final_commonsense_generalization_records.json",
            "--source-files",
            "data/checkpoints/instance_level_commonsense/gemini_v1_2_extract.json",
            "data/checkpoints/instance_level_commonsense/gemini_v1_2_github.json",
            "data/checkpoints/instance_level_commonsense/gemini_v1_2_negative.json",
            "--top-k",
            str(DEFAULT_ENRICH_K),
        ]
        for command in (retrieve, enrich):
            print("COMMAND:", subprocess.list2cmdline(command))
            if args.dry_run:
                continue
            status = subprocess.run(
                command, cwd=PACKAGE_ROOT, env=environment, check=False
            ).returncode
            if status:
                return status
    return 0


def run_situations(args: argparse.Namespace) -> int:
    selected = case_ids(args.ids, args.limit)
    spec = model_spec(args.model if hasattr(args, "model") else args.situation_model)
    for track in ("sequence_context", "image_only"):
        status = run_situation_extraction(
            track=track,
            model=spec,
            dataset_root=args.dataset_root,
            selected_ids=selected,
            workers=args.workers,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        if status:
            return status
    return 0


def run_detectors(args: argparse.Namespace) -> int:
    selected = case_ids(args.ids, args.limit)
    baselines = BASELINES if args.baseline == "all" else (args.baseline,)
    models = MODEL_ALIASES if args.model == "all" else (args.model,)
    conditions = {
        "without": ("without_commonsense",),
        "with": ("with_commonsense",),
        "both": CONDITIONS,
    }[args.injection]
    for alias in models:
        spec = model_spec(alias)
        for baseline in baselines:
            for condition in conditions:
                status = run_detector(
                    baseline=baseline,
                    model=spec,
                    condition=condition,
                    dataset_root=args.dataset_root,
                    selected_ids=selected,
                    workers=args.workers,
                    overwrite=args.overwrite,
                    top_k=args.top_k,
                    dry_run=args.dry_run,
                )
                if status:
                    return status
    return 0


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "doctor":
        return doctor()
    if args.command == "situations":
        return run_situations(args)
    if args.command == "retrieve":
        return run_retrieval(args)
    if args.command == "detect":
        return run_detectors(args)
    if args.command == "all":
        status = run_situations(args)
        if status:
            return status
        status = run_retrieval(args)
        if status:
            return status
        args.baseline = "all"
        args.model = "all"
        args.injection = "both"
        return run_detectors(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
