"""Run the image-only VanillaMLLM bug-detection baseline."""

import argparse
import configparser
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from bug_detect import BugDetectInterface
from jsontool import DetectorParseError, extract_bug_record, parse_detector_output


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..', '..'))
CONFIG_PATH = os.path.join(PROJECT_ROOT, 'config.ini')

config = configparser.ConfigParser()
if not config.read(CONFIG_PATH):
    raise FileNotFoundError(f'Config file not found: {CONFIG_PATH}')

DATA_PATH = config.get('S', 'data_path')
if not os.path.isabs(DATA_PATH):
    DATA_PATH = os.path.join(PROJECT_ROOT, DATA_PATH)


BUG_PATH = os.path.join(BASE_DIR, 'bug')
BUGFREE_PATH = os.path.join(BASE_DIR, 'bugfree')

USAGE_FILE = os.path.join(BUG_PATH, 'usage_records.tsv')
DEFAULT_WORKERS = 4
DEFAULT_IDS = set()
os.makedirs(BUG_PATH, exist_ok=True)
os.makedirs(BUGFREE_PATH, exist_ok=True)


def first_existing_path(*paths):
    for path in paths:
        if os.path.exists(path):
            return path
    return None


def discover_datasets(root_path):
    if not os.path.isdir(root_path):
        raise FileNotFoundError(f'Dataset root not found: {root_path}')
    datasets = []
    for name in sorted(os.listdir(root_path)):
        candidate = os.path.join(root_path, name)
        if os.path.isdir(os.path.join(candidate, 'test_case')):
            datasets.append((name, candidate))
    if not datasets:
        raise FileNotFoundError(f'No dataset containing test_case/ found under: {root_path}')
    return datasets


def normalize_id(value):
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def read_valid_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            value = json.load(f)
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def read_nonempty_text(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            value = f.read().strip()
        return value or None
    except OSError:
        return None


def atomic_write_json(path, data):
    tmp_path = path + '.tmp'
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    os.replace(tmp_path, path)


def atomic_write_text(path, text):
    tmp_path = path + '.tmp'
    with open(tmp_path, 'w', encoding='utf-8') as f:
        f.write(text)
    os.replace(tmp_path, path)


def valid_bug_output(parsed):
    return isinstance(parsed, dict) and 'bug_found' in parsed


def process_case(dataset_name, dataset_path, row, detector, overwrite=False):
    if pd.isna(row['TestCase']):
        return {'status': 'skip', 'reason': 'empty testcase'}

    testid = normalize_id(row['ID'])
    app = str(row['AppName'])
    output_path = BUGFREE_PATH if dataset_name == 'BugFree' else BUG_PATH
    output_file = os.path.join(output_path, f'BugReport_{app}_{testid}.json')
    raw_file = output_file + '.raw.txt'

    if not overwrite:
        existing = read_valid_json(output_file)
        if existing is not None and 'bug_found' in existing:
            return {
                'status': 'skip',
                'testid': testid,
                'app': app,
                'reason': 'existing result',
            }

        # Recover a result from a complete raw response left by an interrupted
        # run, avoiding another paid request.
        raw = read_nonempty_text(raw_file)
        if raw is not None:
            try:
                parsed = parse_detector_output(raw)
                if valid_bug_output(parsed):
                    atomic_write_json(output_file, extract_bug_record(parsed))
                    return {
                        'status': 'recovered',
                        'testid': testid,
                        'app': app,
                        'reason': 'parsed existing raw response',
                    }
            except DetectorParseError:
                pass

    image_path = os.path.join(
        dataset_path,
        'images',
        'stitch_images',
        app,
        f'{testid}.jpg',
    )
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f'Complete sequence image not found: {image_path}')

    print(f'Dataset: {dataset_name}, Appname: {app}, Test ID: {testid}', flush=True)
    start = time.perf_counter()
    output, total, prompt_tokens, completion_tokens = detector.get_output(image_path)
    elapsed = time.perf_counter() - start
    atomic_write_text(raw_file, output)
    parsed = parse_detector_output(output)
    if not valid_bug_output(parsed):
        raise ValueError(f'Detector output lacks bug_found for case {testid}')
    atomic_write_json(output_file, extract_bug_record(parsed))
    return {
        'status': 'done',
        'testid': testid,
        'app': app,
        'prompt_tokens': prompt_tokens,
        'completion_tokens': completion_tokens,
        'total_tokens': total,
        'elapsed': elapsed,
    }


def append_usage_records(results):
    if not results:
        return
    needs_header = not os.path.exists(USAGE_FILE)
    with open(USAGE_FILE, 'a', encoding='utf-8') as f:
        if needs_header:
            f.write(
                'testid\tapp\tprompt_tokens\tcompletion_tokens\t'
                'total_tokens\telapsed_sec\n'
            )
        for result in results:
            f.write(
                f"{result['testid']}\t{result['app']}\t{result['prompt_tokens']}\t"
                f"{result['completion_tokens']}\t{result['total_tokens']}\t"
                f"{result['elapsed']:.6f}\n"
            )


def select_jobs(selected_ids):
    jobs = []
    for dataset_name, dataset_path in discover_datasets(DATA_PATH):
        testcase_path = first_existing_path(
            os.path.join(dataset_path, 'test_case', 'test_case.csv'),
            os.path.join(dataset_path, 'test_case', 'no_bug_case.csv'),
        )
        if testcase_path is None:
            print(f'Loaded dataset: {dataset_name}, selected test cases: 0')
            continue
        dataframe = pd.read_csv(testcase_path, header=0)
        rows = [
            row
            for _, row in dataframe.iterrows()
            if selected_ids is None or normalize_id(row['ID']) in selected_ids
        ]
        print(f'Loaded dataset: {dataset_name}, selected test cases: {len(rows)}')
        jobs.extend((dataset_name, dataset_path, row) for row in rows)

    resolved = {normalize_id(row['ID']) for _, _, row in jobs}
    missing = sorted(selected_ids - resolved) if selected_ids is not None else []
    if missing:
        print(f'Warning: test IDs not found: {missing}')
    return jobs


def parse_args():
    parser = argparse.ArgumentParser(description='Run the image-only VanillaMLLM baseline.')
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
    detector = BugDetectInterface()
    completed = []
    skipped = recovered = 0
    failures = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_case,
                dataset_name,
                dataset_path,
                row,
                detector,
                args.overwrite,
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
                    f"Case {testid} done: tokens={result['total_tokens']}, "
                    f"elapsed={result['elapsed']:.1f}s",
                    flush=True,
                )
            elif result['status'] == 'recovered':
                recovered += 1
                print(f"Case {testid} recovered from raw response", flush=True)
            else:
                skipped += 1
                print(f"Case {testid} skipped: {result['reason']}", flush=True)

    append_usage_records(completed)
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
