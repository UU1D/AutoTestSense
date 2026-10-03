"""Run the VisionDroid baseline with retrieved common-sense injection.

This entry point reuses ``run.py`` and changes only the bug-detection stage:
retrieved rules are loaded from ``situations/`` and inserted into the detector
prompt. Function generation, caching, parsing, concurrency, and CLI behavior
remain those of the baseline.
"""

import json
import os
import time

import run as baseline
from bug_detect_commonsense import BugDetectCommonsenseInterface
from jsontool import extract_bug_record, parse_detector_output


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SITUATIONS_DIR = os.path.join(BASE_DIR, "situations")
TOP_K = 10

# Used when neither --ids nor --all is supplied. Command-line arguments still
# take precedence, so this can be edited for quick local experiments.
DEFAULT_WORKERS = 4

DEFAULT_IDS = set()

BUG_OUTPUT_DIR = os.path.join(BASE_DIR, "bug_commonsense")
BUGFREE_OUTPUT_DIR = os.path.join(BASE_DIR, "bugfree_commonsense")


def load_situation_document(app_name, testid):
    path = os.path.join(
        SITUATIONS_DIR,
        f"Situations_{app_name}_{testid}.json",
    )
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Retrieved common-sense file not found: {path}")
    with open(path, "r", encoding="utf-8") as file:
        document = json.load(file)
    if not isinstance(document, dict) or not isinstance(
        document.get("test_situations"), list
    ):
        raise ValueError(f"Invalid situation document: {path}")
    return document


def build_commonsense_context(document, top_k=TOP_K):
    """Render every family base and, when applicable, its matched variant."""
    lines = []
    for situation_index, situation in enumerate(document["test_situations"], 1):
        if not isinstance(situation, dict):
            raise ValueError(f"test_situations[{situation_index - 1}] is invalid.")
        start_step = situation.get("start_step")
        end_step = situation.get("end_step")
        description = situation.get("situation_description", "")
        lines.append(
            f"[Situation {situation_index}] steps {start_step}-{end_step}: "
            f"{description}"
        )

        candidates = situation.get("related_rule_families")
        if not isinstance(candidates, list) or len(candidates) < top_k:
            count = len(candidates) if isinstance(candidates, list) else 0
            raise ValueError(
                f"Situation {situation_index} has {count} retrieved families; "
                f"expected at least {top_k}."
            )
        for rank, candidate in enumerate(candidates[:top_k], 1):
            if not isinstance(candidate, dict):
                raise ValueError(
                    f"Situation {situation_index} candidate {rank} is invalid."
                )
            matched_unit = candidate.get("matched_unit")
            if not isinstance(matched_unit, dict):
                raise ValueError(
                    f"Situation {situation_index} candidate {rank} has no "
                    "matched_unit."
                )
            base_unit = candidate.get("base_unit")
            if not isinstance(base_unit, dict):
                raise ValueError(
                    f"Situation {situation_index} candidate {rank} has no "
                    "base_unit."
                )
            base_situation = base_unit.get("situation")
            base_rule = base_unit.get("commonsense_rule")
            if not isinstance(base_situation, str) or not base_situation.strip():
                raise ValueError(
                    f"Situation {situation_index} candidate {rank} has no "
                    "base situation."
                )
            if not isinstance(base_rule, str) or not base_rule.strip():
                raise ValueError(
                    f"Situation {situation_index} candidate {rank} has no "
                    "base common-sense rule."
                )
            lines.append(f"  {rank}. Family base:")
            lines.append(f"     Situation: {base_situation.strip()}")
            lines.append(f"     Expected behavior: {base_rule.strip()}")

            matched_type = matched_unit.get("type")
            if matched_type == "BASE":
                continue
            if matched_type != "VARIANT_MEMBER":
                raise ValueError(
                    f"Situation {situation_index} candidate {rank} has invalid "
                    f"matched_unit.type={matched_type!r}."
                )
            matched_situation = matched_unit.get("situation")
            matched_rule = matched_unit.get("commonsense_rule")
            if not isinstance(matched_situation, str) or not matched_situation.strip():
                raise ValueError(
                    f"Situation {situation_index} candidate {rank} has no "
                    "matched variant situation."
                )
            if not isinstance(matched_rule, str) or not matched_rule.strip():
                raise ValueError(
                    f"Situation {situation_index} candidate {rank} has no "
                    "matched variant common-sense rule."
                )
            lines.append("     Matched variant:")
            lines.append(f"       Situation: {matched_situation.strip()}")
            lines.append(f"       Expected behavior: {matched_rule.strip()}")
        lines.append("")
    return "\n".join(lines).strip()


def process_bug_case(
    dataset_name,
    dataset_path,
    row,
    bug_detector,
    overwrite=False,
):
    case = baseline.prepare_case(dataset_name, dataset_path, row, require_image=True)
    if case is None:
        return {"status": "skip", "stage": "bugs", "reason": "empty testcase"}

    output_path = (
        BUGFREE_OUTPUT_DIR if dataset_name == "BugFree" else BUG_OUTPUT_DIR
    )
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(
        output_path,
        f"BugReport_{case['app']}_{case['testid']}.json",
    )
    existing_result = baseline.read_valid_json(output_file)
    if not overwrite and existing_result is not None and "bug_found" in existing_result:
        return {
            "status": "skip",
            "stage": "bugs",
            "testid": case["testid"],
            "app": case["app"],
            "reason": "existing result",
        }

    function_record = baseline.load_cached_function(case["app"], case["testid"])
    situation_document = load_situation_document(case["app"], case["testid"])
    commonsense_context = build_commonsense_context(situation_document)
    start_time = time.perf_counter()

    output, total_tokens, prompt_tokens, completion_tokens = bug_detector.get_output(
        case["stitch_image"],
        function_record["funcname"],
        function_record["funcdescription"],
        function_record["funcgoal"],
        case["actual_path"],
        commonsense_context,
    )
    baseline.atomic_write_text(output_file + ".raw.txt", output)
    parsed = parse_detector_output(output)
    bug = extract_bug_record(parsed)
    baseline.atomic_write_json(output_file, bug)
    return {
        "status": "done",
        "stage": "bugs",
        "testid": case["testid"],
        "app": case["app"],
        "cache_hit": True,
        "token_usage": total_tokens,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "elapsed": time.perf_counter() - start_time,
    }


def main():
    baseline.DEFAULT_IDS = DEFAULT_IDS
    baseline.max_workers = DEFAULT_WORKERS
    baseline.BugDetectInterface = BugDetectCommonsenseInterface
    baseline.process_bug_case = process_bug_case
    baseline.bug_path = BUG_OUTPUT_DIR
    baseline.bugfree_path = BUGFREE_OUTPUT_DIR
    baseline.main()


if __name__ == "__main__":
    main()
