"""Extract sequence-context situations from stitched GUI test sequences."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from jsontool import parse_detector_output
from situation_extract import SituationExtractInterface
from tools import load_persisted_testcase


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = Path(os.getenv("RQ2_SITUATION_OUTPUT_DIR", BASE_DIR / "situations"))


def read_config_data_root() -> Path:
    import configparser

    config = configparser.ConfigParser()
    if not config.read(BASE_DIR / "config.ini"):
        raise FileNotFoundError(f"Config file not found: {BASE_DIR / 'config.ini'}")
    return Path(config.get("S", "data_path")).resolve()


def normalize_id(value: object) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def discover_jobs(data_root: Path, selected: set[str] | None) -> list[tuple[str, Path, object]]:
    jobs = []
    for dataset_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        csv_path = dataset_dir / "test_case" / "test_case.csv"
        if not csv_path.is_file():
            continue
        for _, row in pd.read_csv(csv_path).iterrows():
            case_id = normalize_id(row["ID"])
            if selected is None or case_id in selected:
                jobs.append((dataset_dir.name, dataset_dir, row))
    return sorted(jobs, key=lambda item: int(normalize_id(item[2]["ID"])))


def build_context(dataset_dir: Path, app: str, case_id: str) -> str:
    test_case = load_persisted_testcase(
        str(dataset_dir / "struct_case_data" / app / f"{case_id}.pkl")
    )
    if test_case is None:
        raise FileNotFoundError(f"Missing structured test case: {app}/{case_id}")
    action_root = dataset_dir / "action_description_data"
    if not action_root.is_dir():
        action_root = dataset_dir / "action_description_path"
    pages = [
        line.strip().replace('"', "'")
        for line in (dataset_dir / "page_description_data" / app / f"{case_id}.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    actions = [
        re.sub(r"Step\d+:", "", line).strip().replace('"', "'")
        for line in (action_root / app / f"{case_id}.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    step_count = len(test_case)
    if len(pages) < step_count + 1 or len(actions) < step_count:
        raise ValueError(f"Incomplete page/action descriptions for {app}/{case_id}.")
    lines = [
        f"({index + 1}) Page {index + 1}:{pages[index]}{actions[index]}"
        for index in range(step_count)
    ]
    lines.append(f"({step_count + 1}) Page {step_count + 1}:{pages[step_count]}")
    return "\n".join(lines)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def process_case(dataset_name: str, dataset_dir: Path, row: object, overwrite: bool) -> dict:
    case_id = normalize_id(row["ID"])
    app = str(row["AppName"])
    output_path = OUTPUT_DIR / f"Situations_{app}_{case_id}.json"
    if output_path.is_file() and not overwrite:
        return {"status": "skipped", "case_id": case_id}
    image = dataset_dir / "images" / "stitch_images" / app / f"{case_id}.jpg"
    if not image.is_file():
        raise FileNotFoundError(image)
    context = build_context(dataset_dir, app, case_id)
    started = time.perf_counter()
    output, total, prompt, completion = SituationExtractInterface().get_output(
        str(image), context
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / f"Situations_{app}_{case_id}.raw.txt").write_text(
        output, encoding="utf-8"
    )
    parsed = parse_detector_output(output)
    if not isinstance(parsed, dict) or not isinstance(parsed.get("test_situations"), list):
        raise ValueError(f"Invalid situation output for {app}/{case_id}.")
    atomic_json(output_path, parsed)
    return {
        "status": "done",
        "case_id": case_id,
        "dataset": dataset_name,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", nargs="+", type=int)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.all and args.ids:
        parser.error("--all and --ids are mutually exclusive")
    selected = None if args.all else {str(value) for value in (args.ids or [])}
    jobs = discover_jobs(read_config_data_root(), selected)
    if args.limit is not None:
        jobs = jobs[: args.limit]
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_case, dataset, path, row, args.overwrite): normalize_id(row["ID"])
            for dataset, path, row in jobs
        }
        for future in as_completed(futures):
            case_id = futures[future]
            try:
                result = future.result()
                print(f"case={case_id} status={result['status']}", flush=True)
            except Exception as error:
                failures.append((case_id, str(error)))
                print(f"case={case_id} error={error}", flush=True)
    if failures:
        raise SystemExit(f"Situation extraction failed for {len(failures)} cases.")


if __name__ == "__main__":
    main()
