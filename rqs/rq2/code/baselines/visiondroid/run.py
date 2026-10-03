import pandas as pd
import os, re, json
import time
import argparse
from tools import load_persisted_testcase
from func_interface import FuncInterface
from bug_detect import BugDetectInterface
from jsontool import DetectorParseError, extract_bug_record, parse_detector_output
from concurrent.futures import ThreadPoolExecutor, as_completed
import configparser

config = configparser.ConfigParser()
base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(base_dir, '..', '..'))
config_path = os.path.join(project_root, 'config.ini')
if not config.read(config_path):
    raise FileNotFoundError(f"Config file not found: {config_path}")

max_workers = 4
data_path = config.get('S', 'data_path')

if not os.path.isabs(data_path):
    data_path = os.path.join(project_root, data_path)

bug_path = os.path.join(base_dir, 'bug')
bugfree_path = os.path.join(base_dir, 'bugfree')
function_cache_path = os.path.join(base_dir, 'function_cache')

# Default to the same sample currently selected by
# run_bugdetect_enhanced_parallel.py. Use --ids or --all to override.
DEFAULT_IDS = set()

os.makedirs(bug_path, exist_ok=True)
os.makedirs(bugfree_path, exist_ok=True)
os.makedirs(function_cache_path, exist_ok=True)


def first_existing_path(*paths):
    for path in paths:
        if os.path.exists(path):
            return path
    return None


def discover_datasets(root_path):
    if not os.path.isdir(root_path):
        raise FileNotFoundError(f"Dataset root not found: {root_path}")
    datasets = []
    for name in sorted(os.listdir(root_path)):
        candidate_path = os.path.join(root_path, name)
        if os.path.isdir(os.path.join(candidate_path, 'test_case')):
            datasets.append((name, candidate_path))

    if not datasets:
        raise FileNotFoundError(f"No dataset containing test_case/ found under: {root_path}")

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


def get_function_field(function_dict, field_name, aliases):
    for key in aliases:
        value = function_dict.get(key)
        if value:
            return value

    available_keys = ', '.join(function_dict.keys())
    raise KeyError(f"Missing {field_name}. Available keys: {available_keys}")


def normalize_function_record(function_dict):
    """Normalize the three fields consumed by BugDetectInterface."""
    return {
        'funcname': get_function_field(
            function_dict,
            'function_name',
            ['funcname', 'function_name', 'Function name', 'Function Name', 'name'],
        ),
        'funcdescription': get_function_field(
            function_dict,
            'function_description',
            [
                'funcdescription',
                'function_description',
                'Function description',
                'Function Description',
                'description',
            ],
        ),
        'funcgoal': get_function_field(
            function_dict,
            'function_goal',
            ['funcgoal', 'function_goal', 'Function goal', 'Function Goal', 'goal'],
        ),
    }


def load_or_generate_function(func_generator, app_name, testid, actual_path, refresh=False):
    cache_file = os.path.join(function_cache_path, f'Function_{app_name}_{testid}.json')
    raw_file = os.path.join(function_cache_path, f'Function_{app_name}_{testid}.raw.txt')

    if not refresh:
        cached = read_valid_json(cache_file)
        if cached is not None:
            try:
                return normalize_function_record(cached), 0, 0, 0, True
            except KeyError:
                pass

        # If a previous run wrote the full response but stopped before writing
        # structured JSON, recover from it without another paid API request.
        try:
            with open(raw_file, 'r', encoding='utf-8') as f:
                record = normalize_function_record(parse_detector_output(f.read()))
            atomic_write_json(cache_file, record)
            return record, 0, 0, 0, True
        except (OSError, DetectorParseError, ValueError, KeyError):
            pass

    output, total_tokens, prompt_tokens, completion_tokens = func_generator.get_output(actual_path)
    atomic_write_text(raw_file, output)
    record = normalize_function_record(parse_detector_output(output))
    atomic_write_json(cache_file, record)
    return record, total_tokens, prompt_tokens, completion_tokens, False


def prepare_case(dataset_name, dataset_path, row, require_image=False):
    """Build the shared, local-only inputs needed by both LLM stages."""
    testcase = row['TestCase']
    if pd.isna(testcase):
        return None
    testid = normalize_id(row['ID'])
    app_name = str(row['AppName'])

    stitch_path = os.path.join(dataset_path, 'images', 'stitch_images')
    structcase_path = os.path.join(dataset_path, 'struct_case_data')
    action_desc_path = first_existing_path(
        os.path.join(dataset_path, 'action_description_data'),
        os.path.join(dataset_path, 'action_description_path'),
    )
    if action_desc_path is None:
        raise FileNotFoundError(f"No action description directory found under: {dataset_path}")
    page_desc_path = os.path.join(dataset_path, 'page_description_data')

    persist_path = os.path.join(structcase_path, app_name, f"{testid}.pkl")
    test_case_list = load_persisted_testcase(persist_path)
    if test_case_list is None:
        raise FileNotFoundError(f"No persisted test case list found: {persist_path}")
    step_count = len(test_case_list)

    page_description_txt_path = os.path.join(page_desc_path, app_name, f"{testid}.txt")
    with open(page_description_txt_path, "r", encoding="utf-8") as f:
        page_description_list = [line.strip().replace('"', "'") for line in f.readlines() if line.strip()]

    action_description_txt_path = os.path.join(action_desc_path, app_name, f"{testid}.txt")
    with (open(action_description_txt_path, "r", encoding="utf-8") as f):
        action_description_list = [re.sub(r"Step\d+:", "", line).strip().replace('"', "'") for line in f.readlines() if line.strip()]

    if len(page_description_list) < step_count + 1:
        raise ValueError(
            f"Case {testid} needs {step_count + 1} page descriptions, got {len(page_description_list)}"
        )
    if len(action_description_list) < step_count:
        raise ValueError(
            f"Case {testid} needs {step_count} action descriptions, got {len(action_description_list)}"
        )

    actual_path = ""
    for step in range(1, step_count + 1):
        actual_path +=  f'({step}) Page {step}:' + page_description_list[step - 1] + action_description_list[step - 1] + "\n"
    actual_path += f'({step_count + 1}) Page {step_count + 1}:' + page_description_list[step_count]

    stitch_image_path = os.path.join(stitch_path, app_name, f"{testid}.jpg")
    if require_image and not os.path.isfile(stitch_image_path):
        raise FileNotFoundError(f"Stitched image not found: {stitch_image_path}")

    return {
        'dataset_name': dataset_name,
        'testid': testid,
        'app': app_name,
        'actual_path': actual_path,
        'stitch_image': stitch_image_path,
    }


def load_cached_function(app_name, testid):
    """Read function variables without ever calling FuncInterface."""
    cache_file = os.path.join(function_cache_path, f'Function_{app_name}_{testid}.json')
    raw_file = os.path.join(function_cache_path, f'Function_{app_name}_{testid}.raw.txt')

    cached = read_valid_json(cache_file)
    if cached is not None:
        try:
            return normalize_function_record(cached)
        except KeyError:
            pass

    try:
        with open(raw_file, 'r', encoding='utf-8') as f:
            record = normalize_function_record(parse_detector_output(f.read()))
        atomic_write_json(cache_file, record)
        return record
    except (OSError, DetectorParseError, ValueError, KeyError) as exc:
        raise FileNotFoundError(
            f"Function cache unavailable for {app_name}/{testid}. "
            f"Run --stage functions --ids {testid} first."
        ) from exc


def process_function_case(dataset_name, dataset_path, row, func_generator, refresh=False):
    case = prepare_case(dataset_name, dataset_path, row)
    if case is None:
        return {'status': 'skip', 'stage': 'functions', 'reason': 'empty testcase'}

    start_time = time.perf_counter()
    _, total, prompt_tokens, completion_tokens, cache_hit = load_or_generate_function(
        func_generator,
        case['app'],
        case['testid'],
        case['actual_path'],
        refresh=refresh,
    )
    return {
        'status': 'done',
        'stage': 'functions',
        'testid': case['testid'],
        'app': case['app'],
        'cache_hit': cache_hit,
        'token_usage': total,
        'prompt_tokens': prompt_tokens,
        'completion_tokens': completion_tokens,
        'elapsed': time.perf_counter() - start_time,
    }


def process_bug_case(dataset_name, dataset_path, row, bug_detector, overwrite=False):
    case = prepare_case(dataset_name, dataset_path, row, require_image=True)
    if case is None:
        return {'status': 'skip', 'stage': 'bugs', 'reason': 'empty testcase'}

    output_path = bugfree_path if dataset_name == 'BugFree' else bug_path
    output_file = os.path.join(
        output_path,
        f"BugReport_{case['app']}_{case['testid']}.json",
    )
    existing_result = read_valid_json(output_file)
    if not overwrite and existing_result is not None and 'bug_found' in existing_result:
        return {
            'status': 'skip',
            'stage': 'bugs',
            'testid': case['testid'],
            'app': case['app'],
            'reason': 'existing result',
        }

    # This stage is intentionally cache-only. It cannot trigger FuncInterface.
    function_record = load_cached_function(case['app'], case['testid'])
    start_time = time.perf_counter()

    bug_output, bug_token_usage, bug_prompt_tokens, bug_completion_tokens = bug_detector.get_output(
        case['stitch_image'],
        function_record['funcname'],
        function_record['funcdescription'],
        function_record['funcgoal'],
        case['actual_path'],
    )
    atomic_write_text(output_file + '.raw.txt', bug_output)
    parsed = parse_detector_output(bug_output)
    bug = extract_bug_record(parsed)
    atomic_write_json(output_file, bug)
    return {
        'status': 'done',
        'stage': 'bugs',
        'testid': case['testid'],
        'app': case['app'],
        'cache_hit': True,
        'token_usage': bug_token_usage,
        'prompt_tokens': bug_prompt_tokens,
        'completion_tokens': bug_completion_tokens,
        'elapsed': time.perf_counter() - start_time,
    }


def append_usage_records(results):
    if not results:
        return
    usage_file = os.path.join(base_dir, 'usage_records.tsv')
    needs_header = not os.path.exists(usage_file)
    with open(usage_file, 'a', encoding='utf-8') as f:
        if needs_header:
            f.write(
                'testid\tapp\tstage\tcache_hit\tprompt_tokens\t'
                'completion_tokens\ttotal_tokens\telapsed_sec\n'
            )
        for result in results:
            f.write(
                f"{result['testid']}\t{result['app']}\t{result['stage']}\t"
                f"{result['cache_hit']}\t{result['prompt_tokens']}\t"
                f"{result['completion_tokens']}\t{result['token_usage']}\t"
                f"{result['elapsed']:.6f}\n"
            )


def parse_args():
    parser = argparse.ArgumentParser(description='Run the VisionDroid baseline.')
    parser.add_argument('--ids', nargs='+', type=int, help='test IDs to run')
    parser.add_argument('--all', action='store_true', help='run all dataset cases')
    parser.add_argument(
        '--stage',
        choices=['functions', 'bugs', 'all'],
        default='all',
        help='run function generation, bug detection, or both as two ordered phases',
    )
    parser.add_argument('--workers', type=int, default=max_workers)
    parser.add_argument('--overwrite', action='store_true', help='rerun completed bug reports')
    parser.add_argument(
        '--refresh-functions',
        action='store_true',
        help='regenerate funcname, funcdescription, and funcgoal instead of using their cache',
    )
    args = parser.parse_args()
    if args.all and args.ids:
        parser.error('--all and --ids cannot be used together')
    if args.workers < 1:
        parser.error('--workers must be at least 1')
    return args


def main():
    args = parse_args()
    selected_ids = None if args.all else {str(value) for value in (args.ids or DEFAULT_IDS)}
    datasets = discover_datasets(data_path)
    jobs = []
    for dataset_name, dataset_path in datasets:
        testcase_path = first_existing_path(
            os.path.join(dataset_path, 'test_case', 'test_case.csv'),
            os.path.join(dataset_path, 'test_case', 'no_bug_case.csv'),
        )
        if testcase_path is None:
            print(f"Loaded dataset: {dataset_name}, test cases: 0")
            continue
        testcase_file = pd.read_csv(testcase_path, header=0)
        selected_rows = [
            row
            for _, row in testcase_file.iterrows()
            if selected_ids is None or normalize_id(row['ID']) in selected_ids
        ]
        print(f"Loaded dataset: {dataset_name}, selected test cases: {len(selected_rows)}")
        jobs.extend((dataset_name, dataset_path, row) for row in selected_rows)

    resolved_ids = {normalize_id(row['ID']) for _, _, row in jobs}
    missing_ids = sorted(selected_ids - resolved_ids) if selected_ids is not None else []
    if missing_ids:
        print(f"Warning: test IDs not found: {missing_ids}")

    stages = ['functions', 'bugs'] if args.stage == 'all' else [args.stage]
    any_failures = []
    for stage in stages:
        print(f"Starting stage: {stage}", flush=True)
        if stage == 'functions':
            interface = FuncInterface()
            worker = process_function_case
            extra_arg = args.refresh_functions
        else:
            interface = BugDetectInterface()
            worker = process_bug_case
            extra_arg = args.overwrite

        completed, skipped, failures = run_stage(
            stage,
            jobs,
            worker,
            interface,
            extra_arg,
            args.workers,
        )
        append_usage_records(completed)
        print(
            f"Stage {stage} finished: done={len(completed)}, "
            f"skipped={skipped}, failed={len(failures)}",
            flush=True,
        )
        if failures:
            any_failures.extend((stage, testid, error) for testid, error in failures)
            # In --stage all, do not begin bug detection with an incomplete
            # function-generation phase.
            break

    if any_failures:
        for stage, testid, error in any_failures:
            print(f"  failed {stage}/{testid}: {error}")
        raise SystemExit(1)


def run_stage(stage, jobs, worker, interface, extra_arg, workers):
    completed = []
    skipped = 0
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                worker,
                dataset_name,
                dataset_path,
                row,
                interface,
                extra_arg,
            ): normalize_id(row['ID'])
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

            if result['status'] == 'done':
                completed.append(result)
                cache_status = 'hit' if result['cache_hit'] else 'miss'
                print(
                    f"Case {testid} {stage} done: cache={cache_status}, "
                    f"tokens={result['token_usage']}, elapsed={result['elapsed']:.1f}s",
                    flush=True,
                )
            else:
                skipped += 1
                print(f"Case {testid} {stage} skipped: {result['reason']}", flush=True)
    return completed, skipped, failures


if __name__ == '__main__':
    main()
