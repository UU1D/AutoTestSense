"""Isolated execution helpers for the RQ4 ablation detectors."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from rqs.rq2.code.common.io_utils import write_json
from rqs.rq2.code.common.runtime import (
    copy_json_outputs,
    usage_summary,
    write_runtime_config,
)
from rqs.rq2.code.common.settings import ModelSpec


RQ4_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = RQ4_ROOT.parents[1]
RQ2_ROOT = PACKAGE_ROOT / "rqs/rq2"
WORK_ROOT = PACKAGE_ROOT / "work/rq4"
ADAPTER_ROOT = RQ4_ROOT / "code/baselines"
CONTEXT_HELPER = RQ4_ROOT / "code/common/instance_level_context.py"


def prepare_runtime(baseline: str, dataset_root: Path, model: ModelSpec) -> Path:
    runtime_root = WORK_ROOT / "_runtime" / model.paper_name / baseline
    code_root = runtime_root / "code"
    shutil.copytree(RQ2_ROOT / "code", code_root, dirs_exist_ok=True)
    write_runtime_config(code_root, dataset_root, model)
    baseline_dir = code_root / "baselines" / baseline
    shutil.copy2(ADAPTER_ROOT / baseline / "run_ablation.py", baseline_dir)
    shutil.copy2(CONTEXT_HELPER, baseline_dir / "rq4_instance_level_context.py")
    return baseline_dir


def run_ablation_detector(
    *,
    baseline: str,
    model: ModelSpec,
    dataset_root: Path,
    situation_source: Path,
    selected_ids: list[str],
    workers: int,
    top_k: int,
    overwrite: bool,
    dry_run: bool,
) -> int:
    baseline_dir = (
        RQ2_ROOT / "code/baselines" / baseline
        if dry_run
        else prepare_runtime(baseline, dataset_root, model)
    )
    command = [
        sys.executable,
        "run_ablation.py",
        "--ids",
        *selected_ids,
        "--workers",
        str(workers),
    ]
    if baseline != "visiondroid":
        command.extend(("--top-k", str(top_k)))
    if overwrite:
        command.append("--overwrite")
    print("COMMAND:", subprocess.list2cmdline(command))
    if dry_run:
        return 0
    if not model.api_key or not model.api_url or not model.api_model:
        raise ValueError("RQ2 Gemini API configuration is incomplete.")

    situation_target = baseline_dir / "situations"
    situation_target.mkdir(parents=True, exist_ok=True)
    for path in situation_source.glob("Situations_*.json"):
        shutil.copy2(path, situation_target / path.name)

    environment = os.environ.copy()
    environment["OPENAI_API_KEY"] = model.api_key
    environment["PYTHONUTF8"] = "1"
    environment["RQ4_INJECTION_TOP_K"] = str(top_k)
    status = subprocess.run(
        command,
        cwd=baseline_dir,
        env=environment,
        check=False,
    ).returncode
    if status:
        return status

    output_roots = {
        "visiondroid": baseline_dir / "bug_commonsense",
        "vanilla_mllm": baseline_dir / "bug_commonsense",
        "kuitest": baseline_dir / "bug_with_commonsense",
    }
    target = WORK_ROOT / model.paper_name / baseline
    copied = copy_json_outputs(output_roots[baseline], target, selected_ids)
    write_json(
        target / "run_manifest.json",
        {
            "rq_id": "RQ4",
            "baseline": baseline,
            "paper_model_name": model.paper_name,
            "api_model_identifier": model.api_model,
            "retrieval_mode": "instance_level",
            "dataset_root": str(dataset_root.resolve()),
            "requested_case_ids": selected_ids,
            "output_count": copied,
            "top_k_injected": top_k,
            "usage": usage_summary(baseline_dir),
        },
    )
    return 0
