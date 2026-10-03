"""Run VanillaMLLM with retrieved common-sense context injection."""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

import run as baseline
from bug_detect_commonsense import BugDetectCommonsenseInterface
from jsontool import DetectorParseError, extract_bug_record, parse_detector_output


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SITUATIONS_DIR = os.path.join(BASE_DIR, "situations")

OUTPUT_ROOT = os.path.join(BASE_DIR, "bug_commonsense")

USAGE_FILE = os.path.join(OUTPUT_ROOT, "_usage.tsv")
DEFAULT_TOP_K = 10
# Quick local selection used only when neither --ids nor --all is supplied.
# Edit this set directly when repeatedly running a small group of cases.
DEFAULT_IDS = set()
def situation_filename(app_name, testid):
    return f"Situations_{app_name}_{testid}.json"


def load_situation_document(app_name, testid):
    path = os.path.join(SITUATIONS_DIR, situation_filename(app_name, testid))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Retrieved common-sense file not found: {path}")
    with open(path, "r", encoding="utf-8") as file:
        document = json.load(file)
    if not isinstance(document, dict) or not isinstance(document.get("test_situations"), list):
        raise ValueError(f"Invalid situation document: {path}")
    return document


def resolve_situation_split(app_name, testid):
    """Use the existing positive/negative situation folders as labels."""
    filename = situation_filename(app_name, testid)
    matches = [
        split
        for split in ("positive", "negative")
        if os.path.isfile(os.path.join(SITUATIONS_DIR, split, filename))
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one situation split for {filename}; found {matches or 'none'}."
        )
    return matches[0]


def build_commonsense_context(document, top_k=DEFAULT_TOP_K):
    """Render each inferred situation and its top-k retrieved rule families."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    lines = []
    for situation_index, situation in enumerate(document["test_situations"], 1):
        if not isinstance(situation, dict):
            raise ValueError(f"test_situations[{situation_index - 1}] is invalid.")
        start_step = situation.get("start_step")
        end_step = situation.get("end_step")
        description = situation.get("situation_description", "")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"Situation {situation_index} has no description.")
        lines.append(
            f"[Situation {situation_index}] inferred operations {start_step}-{end_step}: {description.strip()}"
        )

        candidates = situation.get("related_rule_families")
        if not isinstance(candidates, list) or len(candidates) < top_k:
            count = len(candidates) if isinstance(candidates, list) else 0
            raise ValueError(
                f"Situation {situation_index} has {count} retrieved families; expected at least {top_k}."
            )
        for rank, candidate in enumerate(candidates[:top_k], 1):
            if not isinstance(candidate, dict):
                raise ValueError(f"Situation {situation_index} candidate {rank} is invalid.")
            base_unit = candidate.get("base_unit")
            matched_unit = candidate.get("matched_unit")
            if not isinstance(base_unit, dict) or not isinstance(matched_unit, dict):
                raise ValueError(
                    f"Situation {situation_index} candidate {rank} lacks base_unit or matched_unit."
                )
            base_situation = base_unit.get("situation")
            base_rule = base_unit.get("commonsense_rule")
            if not isinstance(base_situation, str) or not base_situation.strip():
                raise ValueError(f"Situation {situation_index} candidate {rank} has no base situation.")
            if not isinstance(base_rule, str) or not base_rule.strip():
                raise ValueError(f"Situation {situation_index} candidate {rank} has no base rule.")
            lines.append(f"  {rank}. Family base:")
            lines.append(f"     Situation: {base_situation.strip()}")
            lines.append(f"     Expected behavior: {base_rule.strip()}")

            matched_type = matched_unit.get("type")
            if matched_type == "BASE":
                continue
            if matched_type != "VARIANT_MEMBER":
                raise ValueError(
                    f"Situation {situation_index} candidate {rank} has invalid matched_unit.type={matched_type!r}."
                )
            matched_situation = matched_unit.get("situation")
            matched_rule = matched_unit.get("commonsense_rule")
            if not isinstance(matched_situation, str) or not matched_situation.strip():
                raise ValueError(f"Situation {situation_index} candidate {rank} has no matched situation.")
            if not isinstance(matched_rule, str) or not matched_rule.strip():
                raise ValueError(f"Situation {situation_index} candidate {rank} has no matched rule.")
            lines.append("     Matched variant:")
            lines.append(f"       Situation: {matched_situation.strip()}")
            lines.append(f"       Expected behavior: {matched_rule.strip()}")
        lines.append("")
    return "\n".join(lines).strip()


def process_case(dataset_name, dataset_path, row, detector, top_k, overwrite=False):
    if pd.isna(row["TestCase"]):
        return {"status": "skip", "reason": "empty testcase"}
    testid = baseline.normalize_id(row["ID"])
    app = str(row["AppName"])
    split = resolve_situation_split(app, testid)
    output_dir = os.path.join(OUTPUT_ROOT, split)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"BugReport_{app}_{testid}.json")
    raw_file = output_file + ".raw.txt"

    if not overwrite:
        existing = baseline.read_valid_json(output_file)
        if baseline.valid_bug_output(existing):
            return {"status": "skip", "testid": testid, "app": app, "split": split, "reason": "existing result"}
        raw = baseline.read_nonempty_text(raw_file)
        if raw is not None:
            try:
                parsed = parse_detector_output(raw)
                if baseline.valid_bug_output(parsed):
                    baseline.atomic_write_json(output_file, extract_bug_record(parsed))
                    return {"status": "recovered", "testid": testid, "app": app, "split": split}
            except DetectorParseError:
                pass

    image_path = os.path.join(dataset_path, "images", "stitch_images", app, f"{testid}.jpg")
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Complete sequence image not found: {image_path}")
    document = load_situation_document(app, testid)
    context = build_commonsense_context(document, top_k=top_k)
    print(f"Dataset: {dataset_name}, split: {split}, Appname: {app}, Test ID: {testid}", flush=True)
    start = time.perf_counter()
    output, total, prompt_tokens, completion_tokens = detector.get_output(image_path, context)
    elapsed = time.perf_counter() - start
    baseline.atomic_write_text(raw_file, output)
    parsed = parse_detector_output(output)
    if not baseline.valid_bug_output(parsed):
        raise ValueError(f"Detector output lacks bug_found for case {testid}")
    baseline.atomic_write_json(output_file, extract_bug_record(parsed))
    return {
        "status": "done", "testid": testid, "app": app, "split": split,
        "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
        "total_tokens": total, "elapsed": elapsed,
    }


def append_usage_records(results):
    if not results:
        return
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    needs_header = not os.path.exists(USAGE_FILE)
    with open(USAGE_FILE, "a", encoding="utf-8") as file:
        if needs_header:
            file.write("testid\tapp\tsplit\tprompt_tokens\tcompletion_tokens\ttotal_tokens\telapsed_sec\n")
        for result in results:
            file.write(
                f"{result['testid']}\t{result['app']}\t{result['split']}\t{result['prompt_tokens']}\t"
                f"{result['completion_tokens']}\t{result['total_tokens']}\t{result['elapsed']:.6f}\n"
            )


def parse_args():
    parser = argparse.ArgumentParser(description="Run VanillaMLLM with retrieved common-sense injection.")
    parser.add_argument("--ids", nargs="+", type=int, help="test IDs to run")
    parser.add_argument("--all", action="store_true", help="run all dataset cases")
    parser.add_argument("--workers", type=int, default=baseline.DEFAULT_WORKERS)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="families injected per situation")
    parser.add_argument("--overwrite", action="store_true", help="rerun completed cases")
    args = parser.parse_args()
    if args.all and args.ids:
        parser.error("--all and --ids cannot be used together")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    return args


def main():
    args = parse_args()
    selected_ids = None if args.all else {str(value) for value in (args.ids or DEFAULT_IDS)}
    jobs = baseline.select_jobs(selected_ids)
    detector = BugDetectCommonsenseInterface()
    completed = []
    skipped = recovered = 0
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_case, dataset_name, dataset_path, row, detector, args.top_k, args.overwrite): baseline.normalize_id(row["ID"])
            for dataset_name, dataset_path, row in jobs
        }
        for future in as_completed(futures):
            testid = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                failures.append((testid, str(exc)))
                print(f"Case {testid} failed: {exc}", flush=True)
                continue
            if result["status"] == "done":
                completed.append(result)
                print(f"Case {testid} done: tokens={result['total_tokens']}, elapsed={result['elapsed']:.1f}s", flush=True)
            elif result["status"] == "recovered":
                recovered += 1
                print(f"Case {testid} recovered from raw response", flush=True)
            else:
                skipped += 1
                print(f"Case {testid} skipped: {result['reason']}", flush=True)
    append_usage_records(completed)
    print(f"Finished: done={len(completed)}, recovered={recovered}, skipped={skipped}, failed={len(failures)}", flush=True)
    if failures:
        for testid, error in failures:
            print(f"  failed {testid}: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
