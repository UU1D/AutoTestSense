"""Isolated execution helpers for RQ2 situation extraction and detectors."""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

from .io_utils import write_json
from .settings import DEFAULT_INJECTION_K, RQ2_ROOT, WORK_ROOT, ModelSpec


def case_ids(ids: list[int] | None, limit: int | None) -> list[str]:
    rows = []
    with (RQ2_ROOT / "data/case_manifest.jsonl").open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    selected = {str(value) for value in ids} if ids else None
    result = [row["case_id"] for row in rows if selected is None or row["case_id"] in selected]
    if selected is not None and set(result) != selected:
        missing = sorted(selected - set(result), key=int)
        raise ValueError(f"Unknown RQ2 case IDs: {missing}")
    return result[:limit] if limit is not None else result


def write_runtime_config(code_root: Path, dataset_root: Path, model: ModelSpec) -> None:
    config = (
        "[S]\n"
        "api_key=\n"
        f"model_name={model.api_model}\n"
        f"url={model.api_url}\n"
        "enable_thinking=false\n"
        f"data_path={dataset_root.resolve()}\n"
        "api_key_gemini=\n"
        f"model_name_gemini={model.api_model}\n"
        f"url_gemini={model.api_url}\n"
    )
    (code_root / "config.ini").write_text(config, encoding="utf-8")
    sequence = code_root / "situation_extraction"
    sequence.mkdir(parents=True, exist_ok=True)
    (sequence / "config.ini").write_text(config, encoding="utf-8")


def prepare_runtime(name: str, dataset_root: Path, model: ModelSpec) -> Path:
    runtime_root = WORK_ROOT / "_runtime" / name
    code_root = runtime_root / "code"
    shutil.copytree(RQ2_ROOT / "code", code_root, dirs_exist_ok=True)
    write_runtime_config(code_root, dataset_root, model)
    return runtime_root


def child_environment(model: ModelSpec) -> dict[str, str]:
    environment = os.environ.copy()
    environment["OPENAI_API_KEY"] = model.api_key
    environment["PYTHONUTF8"] = "1"
    return environment


def run_command(command: list[str], cwd: Path, model: ModelSpec, dry_run: bool) -> int:
    print("COMMAND:", subprocess.list2cmdline(command))
    if dry_run:
        return 0
    if not model.api_key or not model.api_url or not model.api_model:
        raise ValueError(f"RQ2 API configuration is incomplete for {model.alias}.")
    return subprocess.run(
        command, cwd=cwd, env=child_environment(model), check=False
    ).returncode


def copy_json_outputs(source: Path, target: Path, selected_ids: Iterable[str]) -> int:
    wanted = set(selected_ids)
    target.mkdir(parents=True, exist_ok=True)
    copied = 0
    for path in source.glob("BugReport_*.json"):
        case_id = path.stem.rsplit("_", 1)[-1]
        if case_id in wanted:
            shutil.copy2(path, target / path.name)
            copied += 1
    return copied


def usage_summary(root: Path) -> dict[str, int]:
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for path in root.rglob("*.tsv"):
        try:
            with path.open("r", encoding="utf-8", newline="") as stream:
                for row in csv.DictReader(stream, delimiter="\t"):
                    for key in totals:
                        source_key = "token_usage" if key == "total_tokens" else key
                        value = row.get(key) or row.get(source_key)
                        if value:
                            totals[key] += int(float(value))
        except (OSError, ValueError, csv.Error):
            continue
    return totals


def run_detector(
    *,
    baseline: str,
    model: ModelSpec,
    condition: str,
    dataset_root: Path,
    selected_ids: list[str],
    workers: int,
    overwrite: bool,
    top_k: int = DEFAULT_INJECTION_K,
    dry_run: bool = False,
) -> int:
    runtime_name = f"detectors/{model.paper_name}/{baseline}/{condition}"
    if dry_run:
        baseline_dir = RQ2_ROOT / "code/baselines" / baseline
    else:
        runtime = prepare_runtime(runtime_name, dataset_root, model)
        baseline_dir = runtime / "code/baselines" / baseline
    with_commonsense = condition == "with_commonsense"
    if with_commonsense and not dry_run:
        track = "image_only" if baseline == "vanilla_mllm" else "sequence_context"
        situation_source = RQ2_ROOT / "data/situations" / track / "retrieved"
        situation_target = baseline_dir / "situations"
        situation_target.mkdir(parents=True, exist_ok=True)
        for path in situation_source.glob("Situations_*.json"):
            shutil.copy2(path, situation_target / path.name)

    script = "run_commonsense.py" if with_commonsense else "run.py"
    command = [
        sys.executable,
        script,
        "--ids",
        *selected_ids,
        "--workers",
        str(workers),
    ]
    if with_commonsense and baseline in {"vanilla_mllm", "kuitest"}:
        command.extend(("--top-k", str(top_k)))
    if not with_commonsense and baseline == "kuitest":
        command.append("--with-page-descriptions")
    if overwrite:
        command.append("--overwrite")
    status = run_command(command, baseline_dir, model, dry_run)
    if dry_run or status:
        return status

    output_roots = {
        ("visiondroid", False): baseline_dir / "bug",
        ("visiondroid", True): baseline_dir / "bug_commonsense",
        ("vanilla_mllm", False): baseline_dir / "bug",
        ("vanilla_mllm", True): baseline_dir / "bug_commonsense",
        ("kuitest", False): baseline_dir / "bug_with_page_descriptions",
        ("kuitest", True): baseline_dir / "bug_with_commonsense",
    }
    output_root = output_roots[(baseline, with_commonsense)]
    target = WORK_ROOT / model.paper_name / baseline / condition
    copied = copy_json_outputs(output_root, target, selected_ids)
    write_json(
        target / "run_manifest.json",
        {
            "baseline": baseline,
            "paper_model_name": model.paper_name,
            "api_model_identifier": model.api_model,
            "condition": condition,
            "dataset_root": str(dataset_root.resolve()),
            "requested_case_ids": selected_ids,
            "output_count": copied,
            "top_k_injected": top_k if with_commonsense else 0,
            "usage": usage_summary(baseline_dir),
        },
    )
    return 0


def run_situation_extraction(
    *,
    track: str,
    model: ModelSpec,
    dataset_root: Path,
    selected_ids: list[str],
    workers: int,
    overwrite: bool,
    dry_run: bool,
) -> int:
    runtime = None if dry_run else prepare_runtime(f"situations/{track}", dataset_root, model)
    if track == "sequence_context":
        code_dir = (
            RQ2_ROOT / "code/situation_extraction"
            if runtime is None
            else runtime / "code/situation_extraction"
        )
        output_dir = WORK_ROOT / "situations/sequence_context/extracted"
        environment_variable = ("RQ2_SITUATION_OUTPUT_DIR", str(output_dir))
    else:
        code_dir = (
            RQ2_ROOT / "code/situation_extraction/image_only"
            if runtime is None
            else runtime / "code/situation_extraction/image_only"
        )
        output_dir = code_dir / "situations"
        environment_variable = None
    command = [
        sys.executable,
        "run_situations.py",
        "--ids",
        *selected_ids,
        "--workers",
        str(workers),
    ]
    if overwrite:
        command.append("--overwrite")
    print("COMMAND:", subprocess.list2cmdline(command))
    if dry_run:
        return 0
    if not model.api_key or not model.api_url or not model.api_model:
        raise ValueError(f"RQ2 API configuration is incomplete for {model.alias}.")
    environment = child_environment(model)
    if environment_variable:
        environment[environment_variable[0]] = environment_variable[1]
    status = subprocess.run(command, cwd=code_dir, env=environment, check=False).returncode
    if status:
        return status
    if track == "sequence_context":
        target = output_dir
    else:
        target = WORK_ROOT / "situations/image_only/extracted"
        target.mkdir(parents=True, exist_ok=True)
        for path in output_dir.glob("Situations_*.json"):
            shutil.copy2(path, target / path.name)
    write_json(
        target / "run_manifest.json",
        {
            "situation_track": track,
            "paper_model_name": model.paper_name,
            "api_model_identifier": model.api_model,
            "dataset_root": str(dataset_root.resolve()),
            "requested_case_ids": selected_ids,
            "output_count": sum(1 for path in target.glob("Situations_*.json")),
            "usage": usage_summary(code_dir),
        },
    )
    return 0
