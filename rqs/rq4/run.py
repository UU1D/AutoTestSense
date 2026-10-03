"""Unified RQ4 instance-level commonsense ablation entry point."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


RQ4_ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = RQ4_ROOT.parents[1]
RQ2_ROOT = PACKAGE_ROOT / "rqs/rq2"
WORK_ROOT = PACKAGE_ROOT / "work/rq4"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from rqs.rq2.code.common.runtime import case_ids
from rqs.rq2.code.common.settings import model_spec
from rqs.rq4.code.common.runtime import run_ablation_detector


BASELINES = ("visiondroid", "vanilla_mllm", "kuitest")
TRACKS = ("sequence_context", "image_only")
DEFAULT_RECALL_K = 60
DEFAULT_RERANK_K = 40
DEFAULT_ENRICH_K = 30
DEFAULT_INJECTION_K = 10
UNIT_FILES = (
    "data/checkpoints/instance_level_commonsense/gemini_v1_2_extract.json",
    "data/checkpoints/instance_level_commonsense/gemini_v1_2_github.json",
    "data/checkpoints/instance_level_commonsense/gemini_v1_2_negative.json",
)
EMBEDDING_FILES = (
    "data/reference/instance_level_commonsense_situation_embeddings/gemini_v1_2_extract_embeddings.jsonl",
    "data/reference/instance_level_commonsense_situation_embeddings/gemini_v1_2_github_embeddings.jsonl",
    "data/reference/instance_level_commonsense_situation_embeddings/gemini_v1_2_negative_embeddings.jsonl",
)


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ids", nargs="+", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=DEFAULT_INJECTION_K)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor", help="validate the RQ4 package and bundled data")
    retrieve = subparsers.add_parser(
        "retrieve", help="retrieve from 3,708 instance-level commonsense records"
    )
    add_common_arguments(retrieve)
    detect = subparsers.add_parser(
        "detect", help="run one or more Gemini ablation detectors"
    )
    add_common_arguments(detect)
    detect.add_argument("--dataset-root", type=Path, required=True)
    detect.add_argument("--baseline", choices=(*BASELINES, "all"), default="all")
    all_command = subparsers.add_parser(
        "all", help="run instance-level retrieval and all three detectors"
    )
    add_common_arguments(all_command)
    all_command.add_argument("--dataset-root", type=Path, required=True)
    return parser


def _count(path: Path, pattern: str) -> int:
    return sum(1 for item in path.glob(pattern) if item.is_file())


def doctor() -> int:
    errors: list[str] = []
    for module in ("pandas", "PIL", "requests", "numpy", "dotenv"):
        if importlib.util.find_spec(module) is None:
            errors.append(f"missing Python dependency: {module}")
    expected_ids = set(case_ids(None, None))
    expected_dirs = {
        "sequence-context retrieval": RQ4_ROOT / "data/situations/sequence_context/retrieved",
        "image-only retrieval": RQ4_ROOT / "data/situations/image_only/retrieved",
    }
    for label, path in expected_dirs.items():
        count = _count(path, "Situations_*.json") if path.is_dir() else -1
        if count != 151:
            errors.append(f"{label}: expected 151 files, found {count}")
        ids = (
            {
                item.stem.rsplit("_", 1)[-1]
                for item in path.glob("Situations_*.json")
            }
            if path.is_dir()
            else set()
        )
        if ids != expected_ids:
            errors.append(f"{label}: case ID coverage differs from the RQ2 manifest")
    output_root = RQ4_ROOT / "data/detector_outputs/gemini-3.1-pro-preview"
    output_count = 0
    for baseline in BASELINES:
        paths = list((output_root / baseline).glob("BugReport_*.json"))
        output_count += len(paths)
        ids = {path.stem.rsplit("_", 1)[-1] for path in paths}
        if ids != expected_ids:
            errors.append(f"{baseline}: detector case ID coverage is incomplete")
        for path in paths:
            try:
                result = json.loads(path.read_text(encoding="utf-8-sig"))
            except (OSError, json.JSONDecodeError) as exc:
                errors.append(f"{path}: invalid JSON ({exc})")
                continue
            if not isinstance(result.get("bug_found"), bool):
                errors.append(f"{path}: bug_found is not Boolean")
    if output_count != 453:
        errors.append(f"detector outputs: expected 453 files, found {output_count}")
    if not (RQ4_ROOT / "data/detector_output_manifest.json").is_file():
        errors.append("detector output manifest is missing")
    for relative in (*UNIT_FILES, *EMBEDDING_FILES):
        if not (PACKAGE_ROOT / relative).is_file():
            errors.append(f"missing package input: {relative}")
    spec = model_spec("gemini")
    for name, value in (
        ("RQ2_GEMINI_API_KEY", spec.api_key),
        ("RQ2_GEMINI_URL", spec.api_url),
        ("RQ2_GEMINI_MODEL", spec.api_model),
    ):
        if not value:
            print(f"WARN Gemini execution configuration is missing: {name}")
    if errors:
        for error in errors:
            print(f"ERROR {error}")
        return 1
    print("RQ4 package is structurally valid: 151 cases and 453 Gemini outputs.")
    return 0


def _selected_input_dir(track: str, selected_ids: list[str]) -> Path:
    source = RQ2_ROOT / "data/situations" / track / "extracted"
    fingerprint = hashlib.sha256(
        ",".join(selected_ids).encode("ascii")
    ).hexdigest()[:12]
    target = WORK_ROOT / "_retrieval_input" / f"{track}_{fingerprint}"
    target.mkdir(parents=True, exist_ok=True)
    by_id = {
        path.stem.rsplit("_", 1)[-1]: path
        for path in source.glob("Situations_*.json")
    }
    for case_id in selected_ids:
        if case_id not in by_id:
            raise FileNotFoundError(f"No {track} situation file for case {case_id}.")
        shutil.copy2(by_id[case_id], target / by_id[case_id].name)
    return target


def run_retrieval(args: argparse.Namespace) -> int:
    selected = case_ids(args.ids, args.limit)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PACKAGE_ROOT)
    for track in TRACKS:
        input_dir = (
            RQ2_ROOT / "data/situations" / track / "extracted"
            if args.dry_run
            else _selected_input_dir(track, selected)
        )
        raw_dir = WORK_ROOT / "situations" / track / "retrieval_raw"
        enriched_dir = WORK_ROOT / "situations" / track / "retrieved"
        retrieve = [
            sys.executable,
            "-m",
            "commonsense_repro.retrieval.batch_retrieve_instance_level_commonsense",
            "--input-dir", str(input_dir),
            "--output-dir", str(raw_dir),
            "--embeddings-files", *EMBEDDING_FILES,
            "--metadata-files", *UNIT_FILES,
            "--recall-k", str(DEFAULT_RECALL_K),
            "--rerank-top-k", str(DEFAULT_RERANK_K),
        ]
        if args.overwrite:
            retrieve.append("--overwrite")
        enrich = [
            sys.executable,
            "-m",
            "commonsense_repro.retrieval.enrich_retrieved_instance_level_commonsense",
            "--input-dir", str(raw_dir),
            "--output-dir", str(enriched_dir),
            "--source-files", *UNIT_FILES,
            "--top-k", str(DEFAULT_ENRICH_K),
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


def _detector_situation_dir(track: str, selected_ids: list[str]) -> Path:
    bundled = RQ4_ROOT / "data/situations" / track / "retrieved"
    generated = WORK_ROOT / "situations" / track / "retrieved"
    fingerprint = hashlib.sha256(
        ",".join(selected_ids).encode("ascii")
    ).hexdigest()[:12]
    target = WORK_ROOT / "_detector_input" / f"{track}_{fingerprint}"
    target.mkdir(parents=True, exist_ok=True)
    by_id = {}
    for source in (bundled, generated):
        for path in source.glob("Situations_*.json"):
            by_id[path.stem.rsplit("_", 1)[-1]] = path
    for case_id in selected_ids:
        if case_id not in by_id:
            raise FileNotFoundError(f"No retrieved {track} file for case {case_id}.")
        shutil.copy2(by_id[case_id], target / by_id[case_id].name)
    return target


def run_detectors(args: argparse.Namespace) -> int:
    if args.top_k < 1 or args.top_k > DEFAULT_ENRICH_K:
        raise ValueError(f"--top-k must be between 1 and {DEFAULT_ENRICH_K}.")
    selected = case_ids(args.ids, args.limit)
    baselines = BASELINES if args.baseline == "all" else (args.baseline,)
    model = model_spec("gemini")
    for baseline in baselines:
        track = "image_only" if baseline == "vanilla_mllm" else "sequence_context"
        source = (
            RQ4_ROOT / "data/situations" / track / "retrieved"
            if args.dry_run
            else _detector_situation_dir(track, selected)
        )
        status = run_ablation_detector(
            baseline=baseline,
            model=model,
            dataset_root=args.dataset_root,
            situation_source=source,
            selected_ids=selected,
            workers=args.workers,
            top_k=args.top_k,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        if status:
            return status
    return 0


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "doctor":
        return doctor()
    if args.command == "retrieve":
        return run_retrieval(args)
    if args.command == "detect":
        return run_detectors(args)
    if args.command == "all":
        status = run_retrieval(args)
        if status:
            return status
        args.baseline = "all"
        return run_detectors(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
