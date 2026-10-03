"""Run the KuiTest page-context + retrieved-commonsense variant.

Situation bounds are page-interface IDs. Action Step s is performed on Page s
and its response is observed on Page s+1. This script builds and persists that
mapping before injecting only the mapped situation prefix and Top-K rule
families into each changed-image response-verification request.
"""

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import run as baseline
from compare_images import compare_images
from commonsense_mapping import (
    MAPPING_SCHEMA_VERSION,
    build_case_mapping,
    get_step_mapping,
    load_situation_document,
    render_step_context,
    rule_family_ids,
)
from function_identification import FunctionIdentification
from jsontool import parse_detector_output
from response_verification_commonsense import ResponseVerificationCommonsense


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SITUATIONS_DIR = os.path.join(BASE_DIR, "situations")
MAPPING_DIR = os.path.join(BASE_DIR, "commonsense_mapping_cache")
RESPONSE_CACHE_DIR = os.path.join(BASE_DIR, "response_cache_with_commonsense")
BUG_DIR = os.path.join(BASE_DIR, "bug_with_commonsense")
BUGFREE_DIR = os.path.join(BASE_DIR, "bugfree_with_commonsense")
USAGE_FILE = os.path.join(BASE_DIR, "usage_records_commonsense.tsv")

REPORT_SCHEMA_VERSION = 2
VARIANT_NAME = "kuitest_page_context_commonsense"
DEFAULT_TOP_K = 10
DEFAULT_WORKERS = 2
DEFAULT_IDS = set()


for directory in (
    MAPPING_DIR,
    RESPONSE_CACHE_DIR,
    BUG_DIR,
    BUGFREE_DIR,
):
    os.makedirs(directory, exist_ok=True)


def situation_file(app, testid):
    return os.path.join(SITUATIONS_DIR, f"Situations_{app}_{testid}.json")


def mapping_file(app, testid):
    return os.path.join(MAPPING_DIR, f"Mapping_{app}_{testid}.json")


def response_cache_files(app, testid, step_id):
    stem = os.path.join(
        RESPONSE_CACHE_DIR,
        f"Response_{app}_{testid}_Step_{step_id}",
    )
    return stem + ".json", stem + ".raw.txt"


def build_and_save_mapping(case, top_k):
    source = situation_file(case["app"], case["testid"])
    if not os.path.isfile(source):
        raise FileNotFoundError(f"Retrieved common-sense file not found: {source}")
    document = load_situation_document(source)
    mapping = build_case_mapping(
        case,
        document,
        top_k=top_k,
        source_file=os.path.relpath(source, BASE_DIR),
    )
    target = mapping_file(case["app"], case["testid"])
    existing = baseline.read_valid_json(target)
    if existing != mapping:
        baseline.atomic_write_json(target, mapping)
    return mapping


def mapping_summary(mapping):
    mapped_steps = sum(
        bool(item.get("matched_situations"))
        for item in mapping.get("step_mappings", [])
    )
    return {
        "mapped_steps": mapped_steps,
        "unmapped_steps": mapping["action_count"] - mapped_steps,
    }


def process_mapping_case(dataset_name, dataset_path, row, _interface, top_k):
    case = baseline.prepare_case(
        dataset_name,
        dataset_path,
        row,
        include_page_descriptions=True,
    )
    if case is None:
        return {"status": "skip", "stage": "mappings", "reason": "empty testcase"}
    start = time.perf_counter()
    mapping = build_and_save_mapping(case, top_k)
    summary = mapping_summary(mapping)
    return {
        "status": "done",
        "stage": "mappings",
        "testid": case["testid"],
        "app": case["app"],
        "cache_hit": False,
        "llm_steps": 0,
        "unchanged_steps": 0,
        "mapped_steps": summary["mapped_steps"],
        "unmapped_steps": summary["unmapped_steps"],
        "token_usage": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "elapsed": time.perf_counter() - start,
    }


def load_or_verify_response(
    interface,
    case,
    step,
    component_function,
    step_mapping,
    mapping_fingerprint,
    refresh=False,
):
    json_file, raw_file = response_cache_files(
        case["app"], case["testid"], step["step_id"]
    )
    commonsense_context = render_step_context(step_mapping)
    request_context = {
        "component_function": component_function,
        "action_description": step["action_description"],
        "before_page_description": step["before_page_description"],
        "after_page_description": step["after_page_description"],
        "mapping_fingerprint": mapping_fingerprint,
        "action_step_id": step["step_id"],
        "before_page_id": step["step_id"],
        "observed_page_id": step["step_id"] + 1,
    }
    if not refresh:
        cached = baseline.read_valid_json(json_file)
        if (
            cached is not None
            and cached.get("request_context") == request_context
            and baseline.response_is_valid(cached.get("response"))
        ):
            return cached["response"], 0, 0, 0, True

    output, total, prompt_tokens, completion_tokens = interface.response_verification(
        step["after"],
        component_function,
        step["action_description"],
        step["step_id"],
        step["step_id"] + 1,
        step["before_page_description"],
        step["after_page_description"],
        commonsense_context,
    )
    baseline.atomic_write_text(raw_file, output)
    parsed = parse_detector_output(output)
    if not baseline.response_is_valid(parsed):
        raise ValueError(
            f"Response output lacks a judgement field for "
            f"{case['testid']} step {step['step_id']}"
        )
    baseline.atomic_write_json(
        json_file,
        {"request_context": request_context, "response": parsed},
    )
    return parsed, total, prompt_tokens, completion_tokens, False


def normalise_confidence(parsed, default=None):
    confidence = baseline.get_float_field(parsed, "confidence", default=default)
    if confidence is not None and 1 < confidence <= 100:
        confidence /= 100
    return confidence


def function_cache_fingerprint(case):
    records = []
    for step in case["steps"]:
        if compare_images(step["before"], step["after"]):
            continue
        records.append(
            {
                "step_id": step["step_id"],
                "component_function": baseline.load_cached_function(
                    case, step, with_page_descriptions=True
                ),
            }
        )
    encoded = json.dumps(
        records, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parsed_judgement(parsed):
    if any(key in parsed for key in ("judgement", "judgment", "meets_expectation")):
        return baseline.get_bool_field(
            parsed, "judgement", "judgment", "meets_expectation", default=False
        )
    bug_found = baseline.get_bool_field(parsed, "bug_found", default=False)
    return not bug_found


def build_llm_evidence(step, parsed, component_function, step_mapping):
    judgement = parsed_judgement(parsed)
    return {
        "step_id": step["step_id"],
        "action_step_id": step["step_id"],
        "before_page_id": step["step_id"],
        "observed_page_id": step["step_id"] + 1,
        "action_description": step["action_description"],
        "before_page_description": step["before_page_description"],
        "after_page_description": step["after_page_description"],
        "func_iden": component_function,
        "component_function": component_function,
        "judgement": judgement,
        "bug_type": (
            "none" if judgement else (parsed.get("bug_type") or "response_mismatch")
        ),
        "reason": parsed.get("reason") or parsed.get("bug_description") or "",
        "confidence": normalise_confidence(parsed),
        "verification_source": "llm_with_commonsense",
        "situation_indices": [
            item["situation_index"]
            for item in step_mapping.get("matched_situations", [])
        ],
        "rule_family_ids": rule_family_ids(step_mapping),
        "applicable_rule_family_ids": parsed.get(
            "applicable_rule_family_ids", []
        ),
    }


def build_non_response_evidence(step, step_mapping):
    return {
        "step_id": step["step_id"],
        "action_step_id": step["step_id"],
        "before_page_id": step["step_id"],
        "observed_page_id": step["step_id"] + 1,
        "action_description": step["action_description"],
        "before_page_description": step["before_page_description"],
        "after_page_description": step["after_page_description"],
        "func_iden": None,
        "component_function": None,
        "judgement": False,
        "bug_type": "missing_effect",
        "reason": "The before and after screenshots are pixel-identical.",
        "confidence": 1.0,
        "verification_source": "pixel_comparison",
        "situation_indices": [
            item["situation_index"]
            for item in step_mapping.get("matched_situations", [])
        ],
        "rule_family_ids": rule_family_ids(step_mapping),
        "applicable_rule_family_ids": [],
    }


def build_bug_report(case, evidence_steps, mapping, functions_fingerprint):
    failed = [item for item in evidence_steps if not item["judgement"]]
    if failed:
        primary = failed[0]
        descriptions = [
            (
                f"Action Step {item['action_step_id']} (response observed on "
                f"Page {item['observed_page_id']}): {item['reason']}"
            )
            for item in failed
        ]
        bug_description = (
            f"Detected {len(failed)} failing interaction(s). " + " ".join(descriptions)
        )
        bug_found = True
        bug_type = primary["bug_type"]
        bug_page_step = primary["action_step_id"]
        confidence = primary["confidence"]
    else:
        primary = None
        bug_description = "No bug found."
        bug_found = False
        bug_type = "none"
        bug_page_step = 0
        confidences = [
            item["confidence"]
            for item in evidence_steps
            if item.get("confidence") is not None
        ]
        confidence = min(confidences) if confidences else None

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "variant": VARIANT_NAME,
        "app": case["app"],
        "testid": case["testid"],
        "mapping_schema_version": MAPPING_SCHEMA_VERSION,
        "mapping_fingerprint": mapping["fingerprint"],
        "function_cache_fingerprint": functions_fingerprint,
        "bug_found": bug_found,
        "bug_type": bug_type,
        # Compatibility: this remains the action step, not the response page.
        "bug_page_step": bug_page_step,
        "bug_description": bug_description,
        "confidence": confidence,
        "primary_failure": primary,
        "failed_steps": failed,
        "evidence_steps": evidence_steps,
        "exploration_feedback": [],
    }


def process_response_case(
    dataset_name,
    dataset_path,
    row,
    interface,
    overwrite=False,
    refresh=False,
    top_k=DEFAULT_TOP_K,
):
    case = baseline.prepare_case(
        dataset_name,
        dataset_path,
        row,
        include_page_descriptions=True,
    )
    if case is None:
        return {
            "status": "skip",
            "stage": "responses_with_commonsense",
            "reason": "empty testcase",
        }

    mapping = build_and_save_mapping(case, top_k)
    functions_fingerprint = function_cache_fingerprint(case)
    output_root = BUGFREE_DIR if dataset_name == "BugFree" else BUG_DIR
    output_file = os.path.join(
        output_root, f"BugReport_{case['app']}_{case['testid']}.json"
    )
    existing = baseline.read_valid_json(output_file)
    if (
        not overwrite
        and not refresh
        and existing is not None
        and existing.get("schema_version") == REPORT_SCHEMA_VERSION
        and existing.get("variant") == VARIANT_NAME
        and existing.get("mapping_fingerprint") == mapping["fingerprint"]
        and existing.get("function_cache_fingerprint") == functions_fingerprint
    ):
        return {
            "status": "skip",
            "stage": "responses_with_commonsense",
            "testid": case["testid"],
            "app": case["app"],
            "reason": "existing current-schema result",
        }

    start = time.perf_counter()
    total = prompt_tokens = completion_tokens = llm_steps = cache_hits = 0
    unchanged_steps = 0
    evidence_steps = []

    for step in case["steps"]:
        step_mapping = get_step_mapping(mapping, step["step_id"])
        if compare_images(step["before"], step["after"]):
            evidence_steps.append(build_non_response_evidence(step, step_mapping))
            unchanged_steps += 1
            continue

        component_function = baseline.load_cached_function(
            case, step, with_page_descriptions=True
        )
        parsed, usage, prompt, completion, cache_hit = load_or_verify_response(
            interface,
            case,
            step,
            component_function,
            step_mapping,
            mapping["fingerprint"],
            refresh=refresh,
        )
        evidence_steps.append(
            build_llm_evidence(step, parsed, component_function, step_mapping)
        )
        total += usage
        prompt_tokens += prompt
        completion_tokens += completion
        llm_steps += 1
        cache_hits += int(cache_hit)

    baseline.atomic_write_json(
        output_file,
        build_bug_report(case, evidence_steps, mapping, functions_fingerprint),
    )
    return {
        "status": "done",
        "stage": "responses_with_commonsense",
        "testid": case["testid"],
        "app": case["app"],
        "cache_hit": cache_hits == llm_steps,
        "llm_steps": llm_steps,
        "unchanged_steps": unchanged_steps,
        "token_usage": total,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "elapsed": time.perf_counter() - start,
    }


def append_usage(results):
    if not results:
        return
    needs_header = not os.path.exists(USAGE_FILE)
    with open(USAGE_FILE, "a", encoding="utf-8") as file:
        if needs_header:
            file.write(
                "testid\tapp\tstage\tcache_hit\tllm_steps\tunchanged_steps\t"
                "prompt_tokens\tcompletion_tokens\ttotal_tokens\telapsed_sec\n"
            )
        for result in results:
            file.write(
                f"{result['testid']}\t{result['app']}\t{result['stage']}\t"
                f"{result['cache_hit']}\t{result['llm_steps']}\t"
                f"{result['unchanged_steps']}\t{result['prompt_tokens']}\t"
                f"{result['completion_tokens']}\t{result['token_usage']}\t"
                f"{result['elapsed']:.6f}\n"
            )


def run_jobs(stage, jobs, worker, interface, worker_args, workers):
    completed = []
    skipped = 0
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                worker, dataset_name, dataset_path, row, interface, *worker_args
            ): baseline.normalize_id(row["ID"])
            for dataset_name, dataset_path, row in jobs
        }
        for future in as_completed(futures):
            testid = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                failures.append((testid, str(exc)))
                print(f"Case {testid} {stage} failed: {exc}", flush=True)
                continue
            if result["status"] == "done":
                completed.append(result)
                if stage == "mappings":
                    print(
                        f"Case {testid} mappings done: "
                        f"mapped={result['mapped_steps']}, "
                        f"unmapped={result['unmapped_steps']}, "
                        f"elapsed={result['elapsed']:.1f}s",
                        flush=True,
                    )
                else:
                    cache = "hit" if result["cache_hit"] else "miss"
                    print(
                        f"Case {testid} {stage} done: cache={cache}, "
                        f"llm_steps={result['llm_steps']}, "
                        f"unchanged={result['unchanged_steps']}, "
                        f"tokens={result['token_usage']}, "
                        f"elapsed={result['elapsed']:.1f}s",
                        flush=True,
                    )
            else:
                skipped += 1
                print(f"Case {testid} {stage} skipped: {result['reason']}", flush=True)
    append_usage(completed)
    print(
        f"Stage {stage} finished: done={len(completed)}, skipped={skipped}, "
        f"failed={len(failures)}",
        flush=True,
    )
    return failures


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run KuiTest with page descriptions and retrieved commonsense."
    )
    parser.add_argument("--ids", nargs="+", type=int, help="test IDs to run")
    parser.add_argument("--all", action="store_true", help="run all dataset cases")
    parser.add_argument(
        "--stage",
        choices=["functions", "mappings", "responses", "all"],
        default="all",
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--refresh-functions", action="store_true")
    parser.add_argument("--refresh-responses", action="store_true")
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
    selected_ids = None if args.all else {
        str(value) for value in (args.ids or DEFAULT_IDS)
    }
    jobs = baseline.select_jobs(selected_ids)
    stages = (
        ["functions", "mappings", "responses"]
        if args.stage == "all"
        else [args.stage]
    )
    failures = []
    for stage in stages:
        print(f"Starting stage: {stage}", flush=True)
        if stage == "functions":
            stage_failures = run_jobs(
                stage,
                jobs,
                baseline.process_function_case,
                FunctionIdentification(),
                (args.refresh_functions, True),
                args.workers,
            )
        elif stage == "mappings":
            stage_failures = run_jobs(
                stage,
                jobs,
                process_mapping_case,
                None,
                (args.top_k,),
                args.workers,
            )
        else:
            stage_failures = run_jobs(
                stage,
                jobs,
                process_response_case,
                ResponseVerificationCommonsense(),
                (args.overwrite, args.refresh_responses, args.top_k),
                args.workers,
            )
        if stage_failures:
            failures.extend((stage, testid, error) for testid, error in stage_failures)
            break
    if failures:
        for stage, testid, error in failures:
            print(f"  failed {stage}/{testid}: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
