"""Run the KuiTest baseline with isolated, resumable LLM stages.

The original method inputs are preserved:
  * exact comparison of each original before/after screenshot pair;
  * annotated before image + action description for function identification;
  * single after image + component-function text for response verification.

By default, page descriptions and stitched screenshots are not used. The explicit
page-description variant adds the current pre-interaction description to function
identification and the pre/post descriptions to response verification.
"""

import argparse
import configparser
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from compare_images import compare_images
from function_identification import FunctionIdentification
from jsontool import extract_bug_record, parse_detector_output
from response_verification import ResponseVerification
from tools import load_persisted_testcase


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
FUNCTION_CACHE_PATH = os.path.join(BASE_DIR, 'function_cache')
FUNCTION_WITH_PAGES_CACHE_PATH = os.path.join(
    BASE_DIR, 'function_cache_with_page_descriptions'
)
RESPONSE_CACHE_PATH = os.path.join(BASE_DIR, 'response_cache')
BUG_WITH_PAGES_PATH = os.path.join(BASE_DIR, 'bug_with_page_descriptions')
BUGFREE_WITH_PAGES_PATH = os.path.join(BASE_DIR, 'bugfree_with_page_descriptions')
RESPONSE_WITH_PAGES_CACHE_PATH = os.path.join(
    BASE_DIR, 'response_cache_with_page_descriptions'
)
USAGE_FILE = os.path.join(BASE_DIR, 'usage_records.tsv')

DEFAULT_WORKERS = 4
DEFAULT_IDS = set()

for directory in (
    BUG_PATH,
    BUGFREE_PATH,
    FUNCTION_CACHE_PATH,
    FUNCTION_WITH_PAGES_CACHE_PATH,
    RESPONSE_CACHE_PATH,
    BUG_WITH_PAGES_PATH,
    BUGFREE_WITH_PAGES_PATH,
    RESPONSE_WITH_PAGES_CACHE_PATH,
):
    os.makedirs(directory, exist_ok=True)


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


def prepare_case(dataset_name, dataset_path, row, include_page_descriptions=False):
    """Resolve only the per-step inputs required by the original KuiTest method."""
    if pd.isna(row['TestCase']):
        return None

    testid = normalize_id(row['ID'])
    app = str(row['AppName'])
    action_root = first_existing_path(
        os.path.join(dataset_path, 'action_description_data'),
        os.path.join(dataset_path, 'action_description_path'),
    )
    if action_root is None:
        raise FileNotFoundError(f'No action description directory found under: {dataset_path}')

    persist_path = os.path.join(dataset_path, 'struct_case_data', app, f'{testid}.pkl')
    test_case_list = load_persisted_testcase(persist_path)
    if test_case_list is None:
        raise FileNotFoundError(f'No persisted test case list found: {persist_path}')
    step_count = len(test_case_list)

    action_file = os.path.join(action_root, app, f'{testid}.txt')
    with open(action_file, 'r', encoding='utf-8') as f:
        actions = [
            re.sub(r'Step\d+:', '', line).strip().replace('"', "'")
            for line in f
            if line.strip()
        ]
    if len(actions) < step_count:
        raise ValueError(
            f'Case {testid} needs {step_count} action descriptions, got {len(actions)}'
        )

    page_descriptions = None
    if include_page_descriptions:
        page_file = os.path.join(
            dataset_path,
            'page_description_data',
            app,
            f'{testid}.txt',
        )
        with open(page_file, 'r', encoding='utf-8') as f:
            page_descriptions = [
                line.strip().replace('"', "'") for line in f if line.strip()
            ]
        if len(page_descriptions) < step_count + 1:
            raise ValueError(
                f'Case {testid} needs {step_count + 1} page descriptions, '
                f'got {len(page_descriptions)}'
            )

    screenshot_root = os.path.join(dataset_path, 'images', 'screenshots', app, testid)
    annotated_root = os.path.join(dataset_path, 'images', 'annotated_images', app, testid)
    steps = []
    for step_id in range(1, step_count + 1):
        step = {
            'step_id': step_id,
            'action_description': actions[step_id - 1],
            'annotated_before': os.path.join(annotated_root, f'{step_id}-processed.jpg'),
            'before': os.path.join(screenshot_root, f'{step_id}.jpg'),
            'after': os.path.join(screenshot_root, f'{step_id + 1}.jpg'),
        }
        if page_descriptions is not None:
            step['before_page_description'] = page_descriptions[step_id - 1]
            step['after_page_description'] = page_descriptions[step_id]
        for label in ('annotated_before', 'before', 'after'):
            if not os.path.isfile(step[label]):
                raise FileNotFoundError(f"Missing {label} image: {step[label]}")
        steps.append(step)

    return {
        'dataset_name': dataset_name,
        'testid': testid,
        'app': app,
        'steps': steps,
    }


def function_cache_file(app, testid, step_id, with_page_descriptions=False):
    cache_root = (
        FUNCTION_WITH_PAGES_CACHE_PATH
        if with_page_descriptions
        else FUNCTION_CACHE_PATH
    )
    return os.path.join(
        cache_root,
        f'Function_{app}_{testid}_Step_{step_id}.txt',
    )


def response_cache_files(app, testid, step_id, with_page_descriptions=False):
    cache_root = (
        RESPONSE_WITH_PAGES_CACHE_PATH
        if with_page_descriptions
        else RESPONSE_CACHE_PATH
    )
    stem = os.path.join(
        cache_root,
        f'Response_{app}_{testid}_Step_{step_id}',
    )
    return stem + '.json', stem + '.raw.txt'


def read_nonempty_text(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        return text or None
    except OSError:
        return None


def load_or_identify_function(
    interface,
    case,
    step,
    refresh=False,
    with_page_descriptions=False,
):
    cache_file = function_cache_file(
        case['app'],
        case['testid'],
        step['step_id'],
        with_page_descriptions=with_page_descriptions,
    )
    if not refresh:
        cached = read_nonempty_text(cache_file)
        if cached is not None:
            return cached, 0, 0, 0, True

    output, total, prompt_tokens, completion_tokens = interface.func_identification(
        step['annotated_before'],
        step['action_description'],
        step.get('before_page_description'),
    )
    if not output.strip():
        raise ValueError(
            f"Empty function-identification output for {case['testid']} step {step['step_id']}"
        )
    atomic_write_text(cache_file, output)
    return output, total, prompt_tokens, completion_tokens, False


def load_cached_function(case, step, with_page_descriptions=False):
    cache_file = function_cache_file(
        case['app'],
        case['testid'],
        step['step_id'],
        with_page_descriptions=with_page_descriptions,
    )
    cached = read_nonempty_text(cache_file)
    if cached is None:
        raise FileNotFoundError(
            f"Function cache unavailable for {case['app']}/{case['testid']} "
            f"step {step['step_id']}. Run --stage functions --ids {case['testid']}"
            f"{' --with-page-descriptions' if with_page_descriptions else ''} first."
        )
    return cached


def response_is_valid(parsed):
    return isinstance(parsed, dict) and any(
        key in parsed for key in ('judgement', 'judgment', 'meets_expectation', 'bug_found')
    )


def load_or_verify_response(
    interface,
    case,
    step,
    component_function,
    refresh=False,
    with_page_descriptions=False,
):
    json_file, raw_file = response_cache_files(
        case['app'],
        case['testid'],
        step['step_id'],
        with_page_descriptions=with_page_descriptions,
    )
    request_context = {'component_function': component_function}
    if with_page_descriptions:
        request_context.update(
            {
                'before_page_description': step['before_page_description'],
                'after_page_description': step['after_page_description'],
            }
        )
    if not refresh:
        cached = read_valid_json(json_file)
        if (
            cached is not None
            and (
                cached.get('request_context') == request_context
                or (
                    not with_page_descriptions
                    and cached.get('component_function') == component_function
                )
            )
            and response_is_valid(cached.get('response'))
        ):
            return cached['response'], 0, 0, 0, True

    output, total, prompt_tokens, completion_tokens = interface.response_verification(
        step['after'],
        component_function,
        step.get('before_page_description'),
        step.get('after_page_description'),
    )
    atomic_write_text(raw_file, output)
    parsed = parse_detector_output(output)
    if not response_is_valid(parsed):
        raise ValueError(
            f"Response output lacks a judgement field for {case['testid']} step {step['step_id']}"
        )
    atomic_write_json(
        json_file,
        {
            'request_context': request_context,
            'response': parsed,
        },
    )
    return parsed, total, prompt_tokens, completion_tokens, False


def get_bool_field(parsed, *keys, default=False):
    for key in keys:
        value = parsed.get(key)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {'true', 'yes', '1'}:
                return True
            if normalized in {'false', 'no', '0'}:
                return False
    return default


def get_float_field(parsed, *keys, default=None):
    for key in keys:
        value = parsed.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        if isinstance(value, str):
            normalized = value.strip().rstrip('%')
            try:
                confidence = float(normalized)
            except ValueError:
                continue
            return confidence / 100 if value.strip().endswith('%') else confidence
    return default


def build_response_step(step_id, parsed, component_function):
    judgement = get_bool_field(
        parsed,
        'judgement',
        'judgment',
        'meets_expectation',
        default=not bool(parsed.get('bug_found', False)),
    )
    return {
        'judgement': judgement,
        'reason': parsed.get('reason', parsed.get('bug_description', '')),
        'step_id': step_id,
        'func_iden': component_function,
    }


def build_non_response_step(step_id):
    return {
        'judgement': False,
        'reason': 'non-response bug',
        'step_id': step_id,
    }


def build_bug_record(evidence_steps, confidence_by_step):
    failed_steps = [step for step in evidence_steps if not step.get('judgement', True)]
    if failed_steps:
        first_bug = failed_steps[0]
        reason = first_bug.get('reason', '')
        detector_like_output = {
            'bug_found': True,
            'bug_type': 'missing_effect' if reason == 'non-response bug' else 'other',
            'bug_page_step': first_bug.get('step_id'),
            'bug_description': reason,
            'confidence': confidence_by_step.get(first_bug.get('step_id')),
            'step_analysis': evidence_steps,
            'exploration_feedback': [],
        }
    else:
        valid_confidences = [
            confidence_by_step[step['step_id']]
            for step in evidence_steps
            if step.get('step_id') in confidence_by_step
        ]
        detector_like_output = {
            'bug_found': False,
            'bug_type': 'none',
            'bug_page_step': 0,
            'bug_description': 'No bug found.',
            'confidence': min(valid_confidences) if valid_confidences else None,
            'step_analysis': evidence_steps,
            'exploration_feedback': [],
        }
    return extract_bug_record(detector_like_output)


def process_function_case(
    dataset_name,
    dataset_path,
    row,
    interface,
    refresh=False,
    with_page_descriptions=False,
):
    case = prepare_case(
        dataset_name,
        dataset_path,
        row,
        include_page_descriptions=with_page_descriptions,
    )
    if case is None:
        return {
            'status': 'skip',
            'stage': (
                'functions_with_page_descriptions'
                if with_page_descriptions
                else 'functions'
            ),
            'reason': 'empty testcase',
        }

    start = time.perf_counter()
    total = prompt_tokens = completion_tokens = llm_steps = cache_hits = 0
    unchanged_steps = 0
    for step in case['steps']:
        if compare_images(step['before'], step['after']):
            unchanged_steps += 1
            continue
        _, usage, prompt, completion, cache_hit = load_or_identify_function(
            interface,
            case,
            step,
            refresh=refresh,
            with_page_descriptions=with_page_descriptions,
        )
        total += usage
        prompt_tokens += prompt
        completion_tokens += completion
        llm_steps += 1
        cache_hits += int(cache_hit)

    return {
        'status': 'done',
        'stage': (
            'functions_with_page_descriptions'
            if with_page_descriptions
            else 'functions'
        ),
        'testid': case['testid'],
        'app': case['app'],
        'cache_hit': cache_hits == llm_steps,
        'llm_steps': llm_steps,
        'unchanged_steps': unchanged_steps,
        'token_usage': total,
        'prompt_tokens': prompt_tokens,
        'completion_tokens': completion_tokens,
        'elapsed': time.perf_counter() - start,
    }


def process_response_case(
    dataset_name,
    dataset_path,
    row,
    interface,
    overwrite=False,
    refresh=False,
    with_page_descriptions=False,
):
    case = prepare_case(
        dataset_name,
        dataset_path,
        row,
        include_page_descriptions=with_page_descriptions,
    )
    if case is None:
        return {
            'status': 'skip',
            'stage': (
                'responses_with_page_descriptions'
                if with_page_descriptions
                else 'responses'
            ),
            'reason': 'empty testcase',
        }

    if with_page_descriptions:
        output_path = (
            BUGFREE_WITH_PAGES_PATH
            if dataset_name == 'BugFree'
            else BUG_WITH_PAGES_PATH
        )
    else:
        output_path = BUGFREE_PATH if dataset_name == 'BugFree' else BUG_PATH
    stage_name = (
        'responses_with_page_descriptions'
        if with_page_descriptions
        else 'responses'
    )
    output_file = os.path.join(output_path, f"BugReport_{case['app']}_{case['testid']}.json")
    existing = read_valid_json(output_file)
    if not overwrite and not refresh and existing is not None and all(
        key in existing for key in ('bug_found', 'evidence_steps')
    ):
        return {
            'status': 'skip',
            'stage': stage_name,
            'testid': case['testid'],
            'app': case['app'],
            'reason': 'existing result',
        }

    start = time.perf_counter()
    total = prompt_tokens = completion_tokens = llm_steps = cache_hits = 0
    unchanged_steps = 0
    evidence_steps = []
    confidence_by_step = {}

    for step in case['steps']:
        step_id = step['step_id']
        if compare_images(step['before'], step['after']):
            evidence_steps.append(build_non_response_step(step_id))
            confidence_by_step[step_id] = 1.0
            unchanged_steps += 1
            continue

        component_function = load_cached_function(
            case,
            step,
            with_page_descriptions=with_page_descriptions,
        )
        parsed, usage, prompt, completion, cache_hit = load_or_verify_response(
            interface,
            case,
            step,
            component_function,
            refresh=refresh,
            with_page_descriptions=with_page_descriptions,
        )
        evidence_steps.append(build_response_step(step_id, parsed, component_function))
        confidence = get_float_field(parsed, 'confidence')
        if confidence is not None:
            confidence_by_step[step_id] = confidence
        total += usage
        prompt_tokens += prompt
        completion_tokens += completion
        llm_steps += 1
        cache_hits += int(cache_hit)

    atomic_write_json(output_file, build_bug_record(evidence_steps, confidence_by_step))
    return {
        'status': 'done',
        'stage': stage_name,
        'testid': case['testid'],
        'app': case['app'],
        'cache_hit': cache_hits == llm_steps,
        'llm_steps': llm_steps,
        'unchanged_steps': unchanged_steps,
        'token_usage': total,
        'prompt_tokens': prompt_tokens,
        'completion_tokens': completion_tokens,
        'elapsed': time.perf_counter() - start,
    }


def append_usage_records(results):
    if not results:
        return
    needs_header = not os.path.exists(USAGE_FILE)
    with open(USAGE_FILE, 'a', encoding='utf-8') as f:
        if needs_header:
            f.write(
                'testid\tapp\tstage\tcache_hit\tllm_steps\tunchanged_steps\t'
                'prompt_tokens\tcompletion_tokens\ttotal_tokens\telapsed_sec\n'
            )
        for result in results:
            f.write(
                f"{result['testid']}\t{result['app']}\t{result['stage']}\t"
                f"{result['cache_hit']}\t{result['llm_steps']}\t"
                f"{result['unchanged_steps']}\t{result['prompt_tokens']}\t"
                f"{result['completion_tokens']}\t{result['token_usage']}\t"
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


def run_stage(stage, jobs, worker, interface, worker_args, workers):
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
                *worker_args,
            ): normalize_id(row['ID'])
            for dataset_name, dataset_path, row in jobs
        }
        for future in as_completed(futures):
            testid = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                failures.append((testid, str(exc)))
                print(f'Case {testid} {stage} failed: {exc}', flush=True)
                continue

            if result['status'] == 'done':
                completed.append(result)
                cache = 'hit' if result['cache_hit'] else 'miss'
                print(
                    f"Case {testid} {stage} done: cache={cache}, "
                    f"llm_steps={result['llm_steps']}, "
                    f"unchanged={result['unchanged_steps']}, "
                    f"tokens={result['token_usage']}, elapsed={result['elapsed']:.1f}s",
                    flush=True,
                )
            else:
                skipped += 1
                print(f"Case {testid} {stage} skipped: {result['reason']}", flush=True)
    append_usage_records(completed)
    print(
        f'Stage {stage} finished: done={len(completed)}, skipped={skipped}, '
        f'failed={len(failures)}',
        flush=True,
    )
    return failures


def parse_args():
    parser = argparse.ArgumentParser(description='Run the KuiTest baseline.')
    parser.add_argument('--ids', nargs='+', type=int, help='test IDs to run')
    parser.add_argument('--all', action='store_true', help='run all dataset cases')
    parser.add_argument(
        '--stage',
        choices=['functions', 'responses', 'all'],
        default='all',
        help='run function identification, response verification, or both in order',
    )
    parser.add_argument('--workers', type=int, default=DEFAULT_WORKERS)
    parser.add_argument('--overwrite', action='store_true', help='rebuild completed bug reports')
    parser.add_argument(
        '--refresh-functions',
        action='store_true',
        help='regenerate cached component-function descriptions',
    )
    parser.add_argument(
        '--refresh-responses',
        action='store_true',
        help='regenerate cached response-verification outputs',
    )
    parser.add_argument(
        '--with-page-descriptions',
        action='store_true',
        help=(
            'run the page-context variant: add the current pre-interaction page '
            'description to function identification and add the pre/post page '
            'descriptions to response verification'
        ),
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
    jobs = select_jobs(selected_ids)

    stages = ['functions', 'responses'] if args.stage == 'all' else [args.stage]
    failures = []
    for stage in stages:
        print(f'Starting stage: {stage}', flush=True)
        if stage == 'functions':
            stage_failures = run_stage(
                stage,
                jobs,
                process_function_case,
                FunctionIdentification(),
                (args.refresh_functions, args.with_page_descriptions),
                args.workers,
            )
        else:
            stage_failures = run_stage(
                stage,
                jobs,
                process_response_case,
                ResponseVerification(),
                (
                    args.overwrite,
                    args.refresh_responses,
                    args.with_page_descriptions,
                ),
                args.workers,
            )
        if stage_failures:
            failures.extend((stage, testid, error) for testid, error in stage_failures)
            # Do not enter response verification after an incomplete function phase.
            break

    if failures:
        for stage, testid, error in failures:
            print(f'  failed {stage}/{testid}: {error}')
        raise SystemExit(1)


if __name__ == '__main__':
    main()
