"""Judge whether pending instance-level commonsense items belong to recalled commonsense generalization records."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
from commonsense_repro.common.paths import PACKAGE_ROOT as PROJECT_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from commonsense_repro.common import llm_pipeline_common as llm_io
from commonsense_repro.commonsense_generalization.local.generalize_local_commonsense import load_source_units

load_dotenv(PROJECT_ROOT / ".env")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_URL = os.getenv("DEEPSEEK_URL", "").strip()
LLM_MODEL = os.getenv("DEEPSEEK_V4_FLASH", "").strip()

DATASET_ROOT = Path(
    "work/stage4/"
    "commonsense_consolidation"
)
DEFAULT_CANDIDATES_FILE = (
    DATASET_ROOT / "rerank/reranked_consolidation_candidates_k10.jsonl"
)
DEFAULT_CATALOG_FILE = Path(
    "work/stage4/"
    "catalog_assembly/final_commonsense_generalization_records.json"
)
DEFAULT_SOURCE_FILES = (
    Path("data/checkpoints/instance_level_commonsense/gemini_v1_2_extract.json"),
    Path("data/checkpoints/instance_level_commonsense/gemini_v1_2_github.json"),
    Path("data/checkpoints/instance_level_commonsense/gemini_v1_2_negative.json"),
)
DEFAULT_SYSTEM_PROMPT_FILE = Path(
    "prompts/judge_unassigned_commonsense_consolidation_v1.2_system.md"
)
DEFAULT_USER_PROMPT_FILE = Path(
    "prompts/judge_unassigned_commonsense_consolidation_v1.2_user.md"
)
DEFAULT_OUTPUT_DIR = DATASET_ROOT / "consolidation_judgment/deepseek_flash_v1_2"

DEFAULT_CANDIDATE_K = 10
DEFAULT_MIN_RERANK_SCORE = 0.7
DEFAULT_MIN_CANDIDATES = 3
DEFAULT_WORKERS = 12
DEFAULT_MAX_TOKENS: int | None = None
DEFAULT_TIMEOUT = 1200
DEFAULT_MAX_RETRIES = 3
DEFAULT_OUTPUT_RETRIES = 2
DEFAULT_RETRY_BACKOFF = 2.0

INPUT_PLACEHOLDER = "{{formatted_input}}"
RUNTIME_FILENAME = "membership_runtime.jsonl"
RESULTS_FILENAME = "membership_results.jsonl"
SUMMARY_FILENAME = "membership_summary.json"
RAW_RESPONSES_DIRNAME = "raw_responses"

DECISION_ADD = "ADD_TO_FAMILY"
DECISION_NO_MATCH = "NO_MATCH"
MEMBERSHIP_TYPES = {"EQUIVALENT_MEMBER", "VARIANT_MEMBER"}


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _normalize_delta(value: Any, location: str) -> dict[str, list[str]]:
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be an object.")
    llm_io.require_exact_keys(
        value,
        {"situation_differences", "commonsense_rule_differences"},
        location,
    )
    return {
        "situation_differences": llm_io.normalize_string_array(
            value["situation_differences"], f"{location}.situation_differences"
        ),
        "commonsense_rule_differences": llm_io.normalize_string_array(
            value["commonsense_rule_differences"],
            f"{location}.commonsense_rule_differences",
        ),
    }


def load_catalog(path: Path) -> dict[str, dict[str, Any]]:
    root = llm_io.read_json(path)
    families = root.get("rule_families") if isinstance(root, dict) else None
    if not isinstance(families, list):
        raise ValueError(f"{path} must contain a rule_families array.")

    catalog: dict[str, dict[str, Any]] = {}
    for family_index, family in enumerate(families):
        location = f"{path}.rule_families[{family_index}]"
        if not isinstance(family, dict):
            raise ValueError(f"{location} must be an object.")
        family_id = llm_io.require_nonempty_string(
            family.get("family_id"), f"{location}.family_id"
        )
        if family_id in catalog:
            raise ValueError(f"Duplicate family_id: {family_id}")
        base = family.get("base_unit")
        if not isinstance(base, dict):
            raise ValueError(f"{location}.base_unit must be an object.")
        normalized_base = {
            "situation": llm_io.require_nonempty_string(
                base.get("situation"), f"{location}.base_unit.situation"
            ),
            "commonsense_rule": llm_io.require_nonempty_string(
                base.get("commonsense_rule"),
                f"{location}.base_unit.commonsense_rule",
            ),
        }

        groups = family.get("variant_groups")
        if not isinstance(groups, list):
            raise ValueError(f"{location}.variant_groups must be an array.")
        normalized_groups: list[dict[str, Any]] = []
        seen_variant_ids: set[str] = set()
        for group_index, group in enumerate(groups):
            group_location = f"{location}.variant_groups[{group_index}]"
            if not isinstance(group, dict):
                raise ValueError(f"{group_location} must be an object.")
            member_ids = llm_io.normalize_string_array(
                group.get("member_ids"), f"{group_location}.member_ids"
            )
            if not member_ids:
                raise ValueError(f"{group_location}.member_ids cannot be empty.")
            overlap = set(member_ids) & seen_variant_ids
            if overlap:
                raise ValueError(
                    f"Variant members occur in multiple groups for {family_id}: "
                    f"{sorted(overlap, key=llm_io.natural_sort_key)}"
                )
            seen_variant_ids.update(member_ids)
            normalized_groups.append(
                {
                    "member_ids": member_ids,
                    "delta": _normalize_delta(
                        group.get("delta"), f"{group_location}.delta"
                    ),
                }
            )
        catalog[family_id] = {
            "family_id": family_id,
            "base_unit": normalized_base,
            "variant_groups": normalized_groups,
        }
    if not catalog:
        raise ValueError(f"Catalog is empty: {path}")
    return catalog


def _format_list(values: list[str], indent: str = "    ") -> list[str]:
    return [f"{indent}- {value}" for value in values] or [f"{indent}- None"]


def format_judgment_input(
    query: dict[str, Any], candidates: list[dict[str, Any]]
) -> str:
    lines = [
        "NEW COMMON-SENSE UNIT",
        "",
        "Situation:",
        query["situation"],
        "",
        "Common-sense rule:",
        query["violated_commonsense_rule"],
    ]
    for candidate in candidates:
        base = candidate["base_unit"]
        lines.extend(
            [
                "",
                f"CANDIDATE {candidate['candidate_id']}",
                "",
                "Base unit:",
                "  Situation:",
                f"  {base['situation']}",
                "  Common-sense rule:",
                f"  {base['commonsense_rule']}",
                "",
                "Matched representation:",
            ]
        )
        matched = candidate["matched_representation"]
        if matched["type"] == "BASE":
            lines.append("  Base unit")
            continue
        lines.extend(
            [
                "  Existing variant",
                "  Delta from the base:",
                "    Situation differences:",
                *_format_list(matched["delta"]["situation_differences"], "      "),
                "    Common-sense rule differences:",
                *_format_list(
                    matched["delta"]["commonsense_rule_differences"], "      "
                ),
                "  Original variant unit:",
                "    Situation:",
                f"    {matched['instance_level_commonsense']['situation']}",
                "    Common-sense rule:",
                f"    {matched['instance_level_commonsense']['violated_commonsense_rule']}",
            ]
        )
    return "\n".join(lines)


def build_user_prompt(template: str, formatted_input: str) -> str:
    if template.count(INPUT_PLACEHOLDER) != 1:
        raise ValueError(
            f"User prompt must contain exactly one {INPUT_PLACEHOLDER!r}."
        )
    return template.replace(INPUT_PLACEHOLDER, formatted_input)


def select_reranked_candidates(
    candidates: list[dict[str, Any]],
    *,
    min_rerank_score: float,
    min_candidates: int,
) -> list[dict[str, Any]]:
    selected = [
        candidate
        for candidate in candidates
        if float(candidate["rerank_score"]) >= min_rerank_score
    ]
    if len(selected) < min_candidates:
        return candidates[:min_candidates]
    return selected


def load_judgment_batches(
    candidates_path: Path,
    catalog_path: Path,
    source_paths: list[Path],
    candidate_k: int,
    min_rerank_score: float = DEFAULT_MIN_RERANK_SCORE,
    min_candidates: int = DEFAULT_MIN_CANDIDATES,
) -> list[dict[str, Any]]:
    catalog = load_catalog(catalog_path)
    source = load_source_units(source_paths)
    candidate_rows = llm_io.read_jsonl(candidates_path)
    batches: list[dict[str, Any]] = []
    seen_query_ids: set[str] = set()

    for order, row in enumerate(candidate_rows):
        location = f"{candidates_path}:{order + 1}"
        unit_id = llm_io.require_nonempty_string(
            row.get("unit_id"), f"{location}.unit_id"
        )
        if unit_id in seen_query_ids:
            raise ValueError(f"Duplicate candidate query unit_id: {unit_id}")
        query = source.get(unit_id)
        if query is None:
            raise ValueError(f"No source text found for pending unit {unit_id}.")
        raw_neighbors = row.get("neighbors")
        if not isinstance(raw_neighbors, list) or len(raw_neighbors) < candidate_k:
            raise ValueError(
                f"{unit_id} has fewer than {candidate_k} reranked candidates."
            )

        ranked_neighbors = raw_neighbors[:candidate_k]
        for candidate_index, raw_candidate in enumerate(ranked_neighbors):
            candidate_location = f"{location}.neighbors[{candidate_index}]"
            if not isinstance(raw_candidate, dict):
                raise ValueError(f"{candidate_location} must be an object.")
            if raw_candidate.get("rank") != candidate_index + 1:
                raise ValueError(f"Invalid rerank order at {candidate_location}.")
            score = raw_candidate.get("rerank_score")
            if not isinstance(score, (int, float)) or isinstance(score, bool):
                raise ValueError(f"{candidate_location}.rerank_score must be numeric.")

        selected_neighbors = select_reranked_candidates(
            ranked_neighbors,
            min_rerank_score=min_rerank_score,
            min_candidates=min_candidates,
        )

        candidates: list[dict[str, Any]] = []
        model_to_source: dict[str, str] = {}
        seen_family_ids: set[str] = set()
        for candidate_index, raw_candidate in enumerate(selected_neighbors):
            candidate_location = f"{location}.neighbors[{candidate_index}]"
            family_id = llm_io.require_nonempty_string(
                raw_candidate.get("family_id"), f"{candidate_location}.family_id"
            )
            if family_id in seen_family_ids:
                raise ValueError(f"Repeated candidate family {family_id} for {unit_id}.")
            family = catalog.get(family_id)
            if family is None:
                raise ValueError(f"Unknown candidate family {family_id} for {unit_id}.")
            candidate_id = f"C{candidate_index}"
            model_to_source[candidate_id] = family_id

            raw_matched = raw_candidate.get("matched_representation")
            if not isinstance(raw_matched, dict):
                raise ValueError(f"{candidate_location}.matched_representation invalid.")
            representation_type = raw_matched.get("type")
            if representation_type == "BASE":
                matched: dict[str, Any] = {"type": "BASE"}
            elif representation_type == "VARIANT_MEMBER":
                member_id = llm_io.require_nonempty_string(
                    raw_matched.get("member_id"),
                    f"{candidate_location}.matched_representation.member_id",
                )
                group_index = raw_matched.get("variant_group_index")
                if not isinstance(group_index, int) or isinstance(group_index, bool):
                    raise ValueError(
                        f"{candidate_location}.variant_group_index must be an integer."
                    )
                groups = family["variant_groups"]
                if not 0 <= group_index < len(groups):
                    raise ValueError(
                        f"Variant group index {group_index} is invalid for {family_id}."
                    )
                group = groups[group_index]
                if member_id not in group["member_ids"]:
                    raise ValueError(
                        f"Matched member {member_id} is not in {family_id} "
                        f"variant group {group_index}."
                    )
                original = source.get(member_id)
                if original is None:
                    raise ValueError(f"No source text found for variant {member_id}.")
                matched = {
                    "type": "VARIANT_MEMBER",
                    "delta": group["delta"],
                    "instance_level_commonsense": {
                        "situation": original["situation"],
                        "violated_commonsense_rule": original[
                            "violated_commonsense_rule"
                        ],
                    },
                }
            else:
                raise ValueError(
                    f"Unsupported matched representation at {candidate_location}: "
                    f"{representation_type!r}"
                )
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "family_id": family_id,
                    "base_unit": family["base_unit"],
                    "matched_representation": matched,
                }
            )
            seen_family_ids.add(family_id)

        formatted_input = format_judgment_input(query, candidates)
        input_sha256 = llm_io.sha256_text(formatted_input)
        batches.append(
            {
                "order": order,
                "batch_id": unit_id,
                "unit_id": unit_id,
                "units": [{"id": family_id} for family_id in model_to_source.values()],
                "model_to_source_id": model_to_source,
                "identifier_scheme": "batch_local_c_index_v1",
                "candidate_family_ids": list(model_to_source.values()),
                "candidate_selection": {
                    "maximum_k": candidate_k,
                    "min_rerank_score": min_rerank_score,
                    "minimum_candidates": min_candidates,
                },
                "formatted_input": formatted_input,
                "input_sha256": input_sha256,
            }
        )
        seen_query_ids.add(unit_id)
    if not batches:
        raise ValueError(f"Candidate file is empty: {candidates_path}")
    return batches


def parse_judgment_output(
    raw_content: str, *, allowed_candidate_ids: set[str]
) -> dict[str, Any]:
    text = llm_io.strip_json_fence(raw_content)
    if not text:
        raise ValueError("Assistant content is empty.")
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("Output must be a JSON object.")
    llm_io.require_exact_keys(
        parsed,
        {"decision", "selected_candidate_id", "membership_type", "rationale"},
        "output",
    )
    decision = parsed["decision"]
    selected = parsed["selected_candidate_id"]
    membership = parsed["membership_type"]
    rationale = llm_io.require_nonempty_string(parsed["rationale"], "output.rationale")
    if decision == DECISION_ADD:
        selected = llm_io.require_nonempty_string(
            selected, "output.selected_candidate_id"
        )
        if selected not in allowed_candidate_ids:
            raise ValueError(f"Unknown selected_candidate_id: {selected}")
        if membership not in MEMBERSHIP_TYPES:
            raise ValueError(
                "ADD_TO_FAMILY requires EQUIVALENT_MEMBER or VARIANT_MEMBER."
            )
    elif decision == DECISION_NO_MATCH:
        if selected is not None or membership is not None:
            raise ValueError(
                "NO_MATCH requires null selected_candidate_id and membership_type."
            )
    else:
        raise ValueError(f"Unsupported decision: {decision!r}")
    return {
        "decision": decision,
        "selected_candidate_id": selected,
        "membership_type": membership,
        "rationale": rationale,
    }


def read_raw_judgment(
    *, raw_path: Path, batch: dict[str, Any], model: str, prompt_sha256: str
) -> dict[str, Any]:
    raw_record = llm_io.read_json(raw_path)
    if not isinstance(raw_record, dict):
        raise ValueError(f"Raw response must be an object: {raw_path}")
    metadata = raw_record.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"Raw response metadata is invalid: {raw_path}")
    expected_metadata = {
        "batch_id": batch["batch_id"],
        "model": model,
        "prompt_sha256": prompt_sha256,
        "input_sha256": batch["input_sha256"],
    }
    for field, expected in expected_metadata.items():
        if metadata.get(field) != expected:
            raise ValueError(f"Raw response {field} does not match: {raw_path}")

    expected_mapping = batch["model_to_source_id"]
    mapping = llm_io.parse_id_mapping_records(
        raw_record.get("id_mapping"), expected_mapping=expected_mapping
    )
    assistant_output = raw_record.get("assistant_output")
    if assistant_output is not None:
        assistant_content = json.dumps(assistant_output, ensure_ascii=False)
    else:
        assistant_content = raw_record.get("assistant_raw_text")
    if not isinstance(assistant_content, str):
        raise ValueError(f"Raw response contains no assistant output: {raw_path}")
    parsed = parse_judgment_output(
        assistant_content, allowed_candidate_ids=set(mapping)
    )
    selected_alias = parsed["selected_candidate_id"]
    return {
        "unit_id": batch["unit_id"],
        "decision": parsed["decision"],
        "selected_family_id": (
            mapping[selected_alias] if selected_alias is not None else None
        ),
        "membership_type": parsed["membership_type"],
        "rationale": parsed["rationale"],
        "candidate_family_ids": batch["candidate_family_ids"],
    }


def process_batch(
    *,
    batch: dict[str, Any],
    system_prompt: str,
    user_template: str,
    prompt_sha256: str,
    base_url: str,
    api_key: str,
    model: str,
    raw_responses_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    started_at = time.monotonic()
    base_record = {
        "order": batch["order"],
        "batch_id": batch["batch_id"],
        "unit_id": batch["unit_id"],
        "candidate_count": len(batch["candidate_family_ids"]),
        "candidate_selection": batch["candidate_selection"],
        "model": model,
        "stream": args.stream,
        "prompt_sha256": prompt_sha256,
        "input_sha256": batch["input_sha256"],
    }
    user_prompt = build_user_prompt(user_template, batch["formatted_input"])
    usage_total: dict[str, int] = {}
    request_ids: list[str] = []
    last_raw_path = ""
    last_validation_error = ""
    try:
        for output_attempt in range(args.output_retries + 1):
            active_prompt = user_prompt
            if last_validation_error:
                active_prompt += (
                    "\n\nYour previous response was invalid: "
                    f"{last_validation_error}\nReturn a corrected complete JSON object."
                )
            payload: dict[str, Any] = {
                "model": model,
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": active_prompt},
                ],
            }
            if args.max_tokens is not None:
                payload["max_tokens"] = args.max_tokens
            raw_path, response_metadata = llm_io.request_with_retries(
                stream=args.stream,
                base_url=base_url,
                api_key=api_key,
                payload=payload,
                timeout=args.timeout,
                max_retries=args.max_retries,
                retry_backoff=args.retry_backoff,
                raw_responses_dir=raw_responses_dir,
                batch=batch,
                model=model,
                prompt_sha256=prompt_sha256,
                output_attempt=output_attempt,
            )
            last_raw_path = str(raw_path)
            llm_io.add_usage(usage_total, response_metadata.get("usage", {}))
            request_id = response_metadata.get("request_id")
            if isinstance(request_id, str):
                request_ids.append(request_id)
            try:
                read_raw_judgment(
                    raw_path=raw_path,
                    batch=batch,
                    model=model,
                    prompt_sha256=prompt_sha256,
                )
            except (json.JSONDecodeError, ValueError) as error:
                last_validation_error = str(error)
                if output_attempt < args.output_retries:
                    continue
                return {
                    **base_record,
                    "status": "invalid_output",
                    "raw_response_file": last_raw_path,
                    "validation_error": last_validation_error,
                    "output_attempts": output_attempt + 1,
                    "request_ids": request_ids,
                    "usage": usage_total,
                    "elapsed_seconds": round(time.monotonic() - started_at, 3),
                }
            return {
                **base_record,
                "status": "success",
                "raw_response_file": last_raw_path,
                "output_attempts": output_attempt + 1,
                "request_ids": request_ids,
                "usage": usage_total,
                "elapsed_seconds": round(time.monotonic() - started_at, 3),
            }
    except Exception as error:
        return {
            **base_record,
            "status": "error",
            **llm_io.request_error_diagnostics(error),
            "raw_response_file": last_raw_path or None,
            "validation_error": last_validation_error,
            "request_ids": request_ids,
            "usage": usage_total,
            "elapsed_seconds": round(time.monotonic() - started_at, 3),
        }
    raise RuntimeError("Unreachable batch state")


def normalize_runtime_record(
    record: dict[str, Any],
    batch: dict[str, Any],
    model: str,
    prompt_sha256: str,
) -> dict[str, Any]:
    raw_path = record.get("raw_response_file")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("Runtime record has no raw_response_file.")
    normalized = read_raw_judgment(
        raw_path=Path(raw_path),
        batch=batch,
        model=model,
        prompt_sha256=prompt_sha256,
    )
    return {**record, "status": "success", **normalized}


def build_outputs(
    *,
    batches: list[dict[str, Any]],
    runtime_path: Path,
    results_path: Path,
    summary_path: Path,
    model: str,
    prompt_sha256: str,
    candidates_path: Path,
    catalog_path: Path,
    source_paths: list[Path],
    system_prompt_path: Path,
    user_prompt_path: Path,
    raw_responses_dir: Path,
) -> tuple[int, int]:
    latest = llm_io.latest_runtime_records(runtime_path)
    results: list[dict[str, Any]] = []
    for batch in batches:
        key = (batch["batch_id"], model, prompt_sha256, batch["input_sha256"])
        record = latest.get(key)
        if record is None:
            continue
        try:
            results.append(
                normalize_runtime_record(record, batch, model, prompt_sha256)
            )
        except (OSError, json.JSONDecodeError, ValueError) as error:
            print(
                f"unit={batch['unit_id']} cannot parse raw response: {error}",
                file=sys.stderr,
            )
    results.sort(key=lambda item: item["order"])
    llm_io.write_jsonl(results_path, results)

    usage: dict[str, int] = {}
    for result in results:
        llm_io.add_usage(usage, result.get("usage", {}))
    summary = {
        "candidates_file": str(candidates_path),
        "catalog_file": str(catalog_path),
        "source_files": [str(path) for path in source_paths],
        "system_prompt_file": str(system_prompt_path),
        "user_prompt_file": str(user_prompt_path),
        "model": model,
        "prompt_sha256": prompt_sha256,
        "input_unit_count": len(batches),
        "candidate_selection": batches[0]["candidate_selection"] | {
            "total_selected_candidates": sum(
                len(batch["candidate_family_ids"]) for batch in batches
            ),
            "average_selected_candidates": round(
                sum(len(batch["candidate_family_ids"]) for batch in batches)
                / len(batches),
                3,
            ),
            "minimum_selected_candidates": min(
                len(batch["candidate_family_ids"]) for batch in batches
            ),
            "maximum_selected_candidates": max(
                len(batch["candidate_family_ids"]) for batch in batches
            ),
        },
        "success_count": len(results),
        "incomplete_count": len(batches) - len(results),
        "add_to_family_count": sum(
            result["decision"] == DECISION_ADD for result in results
        ),
        "equivalent_member_count": sum(
            result["membership_type"] == "EQUIVALENT_MEMBER" for result in results
        ),
        "variant_member_count": sum(
            result["membership_type"] == "VARIANT_MEMBER" for result in results
        ),
        "no_match_count": sum(
            result["decision"] == DECISION_NO_MATCH for result in results
        ),
        "usage": usage,
        "runtime_output": str(runtime_path),
        "results_output": str(results_path),
        "raw_responses_dir": str(raw_responses_dir),
    }
    llm_io.write_json(summary_path, summary)
    return len(results), len(batches) - len(results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates-file", type=Path, default=DEFAULT_CANDIDATES_FILE)
    parser.add_argument("--catalog-file", type=Path, default=DEFAULT_CATALOG_FILE)
    parser.add_argument(
        "--source-files", type=Path, nargs="+", default=list(DEFAULT_SOURCE_FILES)
    )
    parser.add_argument(
        "--system-prompt-file", type=Path, default=DEFAULT_SYSTEM_PROMPT_FILE
    )
    parser.add_argument(
        "--user-prompt-file", type=Path, default=DEFAULT_USER_PROMPT_FILE
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--candidate-k", type=int, default=DEFAULT_CANDIDATE_K)
    parser.add_argument(
        "--min-rerank-score", type=float, default=DEFAULT_MIN_RERANK_SCORE
    )
    parser.add_argument(
        "--min-candidates", type=int, default=DEFAULT_MIN_CANDIDATES
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--stream", action="store_true", help="Use streaming; non-streaming is default."
    )
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--output-retries", type=int, default=DEFAULT_OUTPUT_RETRIES)
    parser.add_argument("--retry-backoff", type=float, default=DEFAULT_RETRY_BACKOFF)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--parse-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        args.candidate_k < 1
        or not 0 <= args.min_rerank_score <= 1
        or args.min_candidates < 1
        or args.min_candidates > args.candidate_k
        or args.workers < 1
        or args.timeout < 1
        or args.max_retries < 0
        or args.output_retries < 0
    ):
        raise ValueError("Invalid candidate, worker, timeout, or retry configuration.")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive.")
    if args.parse_only and args.overwrite:
        raise ValueError("--parse-only cannot be combined with --overwrite.")

    candidates_path = resolve_path(args.candidates_file).resolve()
    catalog_path = resolve_path(args.catalog_file).resolve()
    source_paths = [resolve_path(path).resolve() for path in args.source_files]
    system_prompt_path = resolve_path(args.system_prompt_file).resolve()
    user_prompt_path = resolve_path(args.user_prompt_file).resolve()
    output_dir = resolve_path(args.output_dir).resolve()
    batches = load_judgment_batches(
        candidates_path,
        catalog_path,
        source_paths,
        args.candidate_k,
        args.min_rerank_score,
        args.min_candidates,
    )
    if args.limit is not None:
        batches = batches[: args.limit]

    system_prompt = system_prompt_path.read_text(encoding="utf-8").strip()
    user_template = user_prompt_path.read_text(encoding="utf-8").strip()
    if user_template.count(INPUT_PLACEHOLDER) != 1:
        raise ValueError(
            f"User prompt must contain exactly one {INPUT_PLACEHOLDER!r}."
        )
    prompt_sha256 = llm_io.sha256_text(system_prompt + "\n\n" + user_template)

    if args.dry_run:
        first = batches[0]
        print(
            json.dumps(
                {
                    "unit_count": len(batches),
                    "model": LLM_MODEL or "<missing>",
                    "stream": args.stream,
                    "workers": args.workers,
                    "candidate_k": args.candidate_k,
                    "min_rerank_score": args.min_rerank_score,
                    "min_candidates": args.min_candidates,
                    "first_selected_candidate_count": len(
                        first["candidate_family_ids"]
                    ),
                    "prompt_sha256": prompt_sha256,
                    "first_unit_id": first["unit_id"],
                    "first_id_mapping": first["model_to_source_id"],
                    "first_user_prompt": build_user_prompt(
                        user_template, first["formatted_input"]
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    required = [("DEEPSEEK_V4_FLASH", LLM_MODEL)]
    if not args.parse_only:
        required.extend(
            [("DEEPSEEK_API_KEY", DEEPSEEK_API_KEY), ("DEEPSEEK_URL", DEEPSEEK_URL)]
        )
    missing = [name for name, value in required if not value]
    if missing:
        raise ValueError(f"Missing .env variables: {', '.join(missing)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_responses_dir = output_dir / RAW_RESPONSES_DIRNAME
    runtime_path = output_dir / RUNTIME_FILENAME
    results_path = output_dir / RESULTS_FILENAME
    summary_path = output_dir / SUMMARY_FILENAME
    raw_responses_dir.mkdir(parents=True, exist_ok=True)

    if args.overwrite:
        for path in (runtime_path, results_path, summary_path):
            if path.exists():
                path.unlink()

    latest = llm_io.latest_runtime_records(runtime_path)
    pending: list[dict[str, Any]] = []
    for batch in batches:
        key = (batch["batch_id"], LLM_MODEL, prompt_sha256, batch["input_sha256"])
        record = latest.get(key)
        if record is not None:
            try:
                normalize_runtime_record(record, batch, LLM_MODEL, prompt_sha256)
                continue
            except (OSError, json.JSONDecodeError, ValueError):
                pass
        pending.append(batch)

    if not args.parse_only and pending:
        with runtime_path.open("a", encoding="utf-8", newline="\n") as runtime_file:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=args.workers
            ) as executor:
                futures = {
                    executor.submit(
                        process_batch,
                        batch=batch,
                        system_prompt=system_prompt,
                        user_template=user_template,
                        prompt_sha256=prompt_sha256,
                        base_url=llm_io.resolve_base_url(DEEPSEEK_URL),
                        api_key=DEEPSEEK_API_KEY,
                        model=LLM_MODEL,
                        raw_responses_dir=raw_responses_dir,
                        args=args,
                    ): batch
                    for batch in pending
                }
                for future in concurrent.futures.as_completed(futures):
                    record = future.result()
                    llm_io.append_jsonl(runtime_file, record)
                    print(
                        f"unit={record['unit_id']} status={record['status']} "
                        f"elapsed={record['elapsed_seconds']}s"
                    )

    success_count, incomplete_count = build_outputs(
        batches=batches,
        runtime_path=runtime_path,
        results_path=results_path,
        summary_path=summary_path,
        model=LLM_MODEL,
        prompt_sha256=prompt_sha256,
        candidates_path=candidates_path,
        catalog_path=catalog_path,
        source_paths=source_paths,
        system_prompt_path=system_prompt_path,
        user_prompt_path=user_prompt_path,
        raw_responses_dir=raw_responses_dir,
    )
    print(
        f"Done. success={success_count}, incomplete={incomplete_count}, "
        f"output_dir={output_dir}"
    )


if __name__ == "__main__":
    main()

