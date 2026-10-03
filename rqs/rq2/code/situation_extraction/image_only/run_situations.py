"""Parallel and resumable image-only situation extraction."""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from jsontool import DetectorParseError, parse_detector_output
from run import (
    BASE_DIR,
    DATA_PATH,
    atomic_write_json,
    atomic_write_text,
    normalize_id,
    read_nonempty_text,
    read_valid_json,
    select_jobs,
)
from situation_extract import SituationExtractInterface


OUT_PATH = os.path.join(BASE_DIR, 'situations')
USAGE_FILE = os.path.join(OUT_PATH, '_usage.tsv')
DEFAULT_WORKERS = 4
DEFAULT_IDS = set()
os.makedirs(OUT_PATH, exist_ok=True)


def valid_situation_output(parsed):
    return (
        isinstance(parsed, dict)
        and isinstance(parsed.get('recognized_operations'), list)
        and isinstance(parsed.get('test_situations'), list)
    )


def validate_situation_output(parsed, testid):
    if not valid_situation_output(parsed):
        raise ValueError(
            f'Situation output for case {testid} must contain list fields '
            f'recognized_operations and test_situations'
        )

    for index, operation in enumerate(parsed['recognized_operations'], 1):
        if not isinstance(operation, dict):
            raise ValueError(f'recognized_operations[{index - 1}] is not an object')
        step_id = operation.get('step_id')
        if not isinstance(step_id, int) or isinstance(step_id, bool):
            raise ValueError(f'recognized_operations[{index - 1}].step_id must be an integer')

    for index, situation in enumerate(parsed['test_situations'], 1):
        if not isinstance(situation, dict):
            raise ValueError(f'test_situations[{index - 1}] is not an object')
        start = situation.get('start_step')
        end = situation.get('end_step')
        description = situation.get('situation_description')
        if not isinstance(start, int) or isinstance(start, bool):
            raise ValueError(f'test_situations[{index - 1}].start_step must be an integer')
        if not isinstance(end, int) or isinstance(end, bool):
            raise ValueError(f'test_situations[{index - 1}].end_step must be an integer')
        if start < 1 or end < start:
            raise ValueError(f'Invalid situation range {start}-{end}')
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f'test_situations[{index - 1}] lacks a description')
    return parsed


def process_case(dataset_name, dataset_path, row, interface, overwrite=False):
    if pd.isna(row.get('TestCase')):
        return {'status': 'skip', 'reason': 'empty testcase'}

    testid = normalize_id(row['ID'])
    app = str(row['AppName'])
    output_file = os.path.join(OUT_PATH, f'Situations_{app}_{testid}.json')
    raw_file = os.path.join(OUT_PATH, f'Situations_{app}_{testid}.raw.txt')

    if not overwrite:
        existing = read_valid_json(output_file)
        if valid_situation_output(existing):
            return {
                'status': 'skip', 'testid': testid, 'app': app,
                'reason': 'existing result',
            }

        raw = read_nonempty_text(raw_file)
        if raw is not None:
            try:
                parsed = validate_situation_output(parse_detector_output(raw), testid)
                atomic_write_json(output_file, parsed)
                return {'status': 'recovered', 'testid': testid, 'app': app}
            except (DetectorParseError, ValueError):
                pass

    image_path = os.path.join(
        dataset_path, 'images', 'stitch_images', app, f'{testid}.jpg'
    )
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f'Complete sequence image not found: {image_path}')

    print(f'Dataset: {dataset_name}, Appname: {app}, Test ID: {testid}', flush=True)
    start = time.perf_counter()
    output, total, prompt_tokens, completion_tokens = interface.get_output(image_path)
    elapsed = time.perf_counter() - start
    atomic_write_text(raw_file, output)
    parsed = validate_situation_output(parse_detector_output(output), testid)
    atomic_write_json(output_file, parsed)
    return {
        'status': 'done',
        'testid': testid,
        'app': app,
        'operations': len(parsed['recognized_operations']),
        'situations': len(parsed['test_situations']),
        'prompt_tokens': prompt_tokens,
        'completion_tokens': completion_tokens,
        'total_tokens': total,
        'elapsed': elapsed,
    }


def append_usage(results):
    if not results:
        return
    needs_header = not os.path.exists(USAGE_FILE)
    with open(USAGE_FILE, 'a', encoding='utf-8') as f:
        if needs_header:
            f.write(
                'testid\tapp\toperations\tsituations\tprompt_tokens\t'
                'completion_tokens\ttotal_tokens\telapsed_sec\n'
            )
        for result in results:
            f.write(
                f"{result['testid']}\t{result['app']}\t{result['operations']}\t"
                f"{result['situations']}\t{result['prompt_tokens']}\t"
                f"{result['completion_tokens']}\t{result['total_tokens']}\t"
                f"{result['elapsed']:.6f}\n"
            )


def parse_args():
    parser = argparse.ArgumentParser(
        description='Extract operations and situations from sequence images only.'
    )
    parser.add_argument('--ids', nargs='+', type=int, help='test IDs to run')
    parser.add_argument('--all', action='store_true', help='run all dataset cases')
    parser.add_argument('--workers', type=int, default=DEFAULT_WORKERS)
    parser.add_argument('--overwrite', action='store_true', help='rerun completed cases')
    args = parser.parse_args()
    if args.all and args.ids:
        parser.error('--all and --ids cannot be used together')
    if args.workers < 1:
        parser.error('--workers must be at least 1')
    return args


def main():
    args = parse_args()
    selected_ids = None if args.all else {str(value) for value in (args.ids or DEFAULT_IDS)}
    jobs = select_jobs(selected_ids)
    interface = SituationExtractInterface()
    completed = []
    skipped = recovered = 0
    failures = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_case, dataset_name, dataset_path, row, interface, args.overwrite
            ): normalize_id(row['ID'])
            for dataset_name, dataset_path, row in jobs
        }
        for future in as_completed(futures):
            testid = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                failures.append((testid, str(exc)))
                print(f'Case {testid} failed: {exc}', flush=True)
                continue
            if result['status'] == 'done':
                completed.append(result)
                print(
                    f"Case {testid} done: operations={result['operations']}, "
                    f"situations={result['situations']}, tokens={result['total_tokens']}, "
                    f"elapsed={result['elapsed']:.1f}s",
                    flush=True,
                )
            elif result['status'] == 'recovered':
                recovered += 1
                print(f'Case {testid} recovered from raw response', flush=True)
            else:
                skipped += 1
                print(f"Case {testid} skipped: {result['reason']}", flush=True)

    append_usage(completed)
    print(
        f'Finished: done={len(completed)}, recovered={recovered}, '
        f'skipped={skipped}, failed={len(failures)}',
        flush=True,
    )
    if failures:
        for testid, error in failures:
            print(f'  failed {testid}: {error}')
        raise SystemExit(1)


if __name__ == '__main__':
    main()
