"""Generalize common-sense commonsense generalization records with base rules and member deltas."""

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

from commonsense_repro.common import llm_pipeline_common as llm_io


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
from commonsense_repro.common.paths import PACKAGE_ROOT as PROJECT_ROOT
load_dotenv(PROJECT_ROOT / ".env")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_URL = os.getenv("DEEPSEEK_URL", "").strip()
LLM_MODEL = os.getenv("DEEPSEEK_V4_FLASH", "").strip()

DEFAULT_BATCHES_FILE = Path(
    "work/stage2/refined/"
    "llm_batches.jsonl"
)
DEFAULT_SOURCE_FILES = (
    Path("data/checkpoints/instance_level_commonsense/gemini_v1_2_extract.json"),
    Path("data/checkpoints/instance_level_commonsense/gemini_v1_2_github.json"),
    Path("data/checkpoints/instance_level_commonsense/gemini_v1_2_negative.json"),
)
DEFAULT_SYSTEM_PROMPT_FILE = Path(
    "prompts/generalize_local_commonsense_v1.4_system.md"
)
DEFAULT_USER_PROMPT_FILE = Path(
    "prompts/generalize_local_commonsense_v1.4_user.md"
)
DEFAULT_OUTPUT_DIR = Path(
    "work/stage3/"
    "gemini_v1_2_non_mutual_s15_leiden_max25_deepseek_flash_v1_4"
)

DEFAULT_WORKERS = 5
DEFAULT_MAX_TOKENS: int | None = None
DEFAULT_TIMEOUT = 1200
DEFAULT_MAX_RETRIES = 3
DEFAULT_OUTPUT_RETRIES = 2
DEFAULT_RETRY_BACKOFF = 2.0

UNITS_PLACEHOLDER = "{{units_json}}"
RUNTIME_FILENAME = "commonsense_generalization_runtime.jsonl"
RESULTS_FILENAME = "commonsense_generalization_results.jsonl"
SUMMARY_FILENAME = "commonsense_generalization_summary.json"
RAW_RESPONSES_DIRNAME = "raw_responses"

DELTA_FIELDS = (
    "situation_differences",
    "commonsense_rule_differences",
)


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


resolve_base_url = llm_io.resolve_base_url
sha256_text = llm_io.sha256_text
natural_sort_key = llm_io.natural_sort_key
read_json = llm_io.read_json
read_jsonl = llm_io.read_jsonl
require_nonempty_string = llm_io.require_nonempty_string
normalize_string_array = llm_io.normalize_string_array
require_exact_keys = llm_io.require_exact_keys
build_user_prompt = llm_io.build_user_prompt
build_model_id_mapping = llm_io.build_model_id_mapping
build_model_units = llm_io.build_model_units
strip_json_fence = llm_io.strip_json_fence
add_usage = llm_io.add_usage
stream_response_to_files = llm_io.stream_response_to_files
non_streaming_response_to_files = llm_io.non_streaming_response_to_files
is_retryable_stream_error = llm_io.is_retryable_request_error
write_json = llm_io.write_json
write_jsonl = llm_io.write_jsonl
write_text = llm_io.write_text
append_jsonl = llm_io.append_jsonl
latest_runtime_records = llm_io.latest_runtime_records


def load_source_units(paths: list[Path]) -> dict[str, dict[str, Any]]:
    source_units: dict[str, dict[str, Any]] = {}
    for path in paths:
        records = read_json(path)
        if not isinstance(records, list):
            raise ValueError(f"{path} must contain a JSON array.")
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(f"{path}[{index}] must be an object.")
            unique_id = require_nonempty_string(record.get("id"), f"{path}[{index}].id")
            if unique_id in source_units:
                raise ValueError(f"Duplicate source unit ID: {unique_id}")
            source_units[unique_id] = {
                "situation": require_nonempty_string(
                    record.get("situation"), f"{path}[{index}].situation"
                ),
                "violated_commonsense_rule": require_nonempty_string(
                    record.get("violated_commonsense_rule"),
                    f"{path}[{index}].violated_commonsense_rule",
                ),
                "applicability_conditions": normalize_string_array(
                    record.get("applicability_conditions"),
                    f"{path}[{index}].applicability_conditions",
                ),
            }
    return source_units


def load_enriched_batches(
    batches_path: Path,
    source_units: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    batches = read_jsonl(batches_path)
    seen_batch_ids: set[str] = set()
    seen_unit_ids: set[str] = set()
    for order, batch in enumerate(batches):
        batch_id = require_nonempty_string(
            batch.get("batch_id"), f"{batches_path}:{order + 1}.batch_id"
        )
        units = batch.get("units")
        if batch_id in seen_batch_ids:
            raise ValueError(f"Duplicate batch_id: {batch_id}")
        if not isinstance(units, list) or not 2 <= len(units) <= 25:
            raise ValueError(f"Batch {batch_id} must contain 2 to 25 units.")
        if batch.get("size") != len(units):
            raise ValueError(f"Batch {batch_id} has an inconsistent size field.")

        enriched_units: list[dict[str, Any]] = []
        for unit_index, unit in enumerate(units):
            if not isinstance(unit, dict):
                raise ValueError(f"Batch {batch_id} unit {unit_index} is invalid.")
            unique_id = require_nonempty_string(
                unit.get("id"), f"Batch {batch_id} unit {unit_index}.id"
            )
            if unique_id in seen_unit_ids:
                raise ValueError(f"Unit {unique_id} appears in multiple batches.")
            source = source_units.get(unique_id)
            if source is None:
                raise ValueError(f"No source data found for unit {unique_id}.")

            situation = require_nonempty_string(
                unit.get("situation"), f"Batch {batch_id} unit {unique_id}.situation"
            )
            rule = require_nonempty_string(
                unit.get("violated_commonsense_rule"),
                f"Batch {batch_id} unit {unique_id}.violated_commonsense_rule",
            )
            if situation != source["situation"] or rule != source["violated_commonsense_rule"]:
                raise ValueError(f"Source text mismatch for unit {unique_id}.")

            enriched_units.append(
                {
                    "id": unique_id,
                    "situation": situation,
                    "violated_commonsense_rule": rule,
                    "applicability_conditions": source["applicability_conditions"],
                }
            )
            seen_unit_ids.add(unique_id)

        batch["order"] = order
        batch["batch_id"] = batch_id
        batch["units"] = enriched_units
        batch["input_sha256"] = sha256_text(
            json.dumps(enriched_units, ensure_ascii=False, sort_keys=True)
        )
        seen_batch_ids.add(batch_id)
    return batches


def restore_source_ids(
    normalized: dict[str, Any], model_to_source_id: dict[str, str]
) -> dict[str, Any]:
    for family in normalized["rule_families"]:
        family["equivalent_member_ids"] = sorted(
            (
                model_to_source_id[model_id]
                for model_id in family["equivalent_member_ids"]
            ),
            key=natural_sort_key,
        )
        for group in family["variant_groups"]:
            group["member_ids"] = sorted(
                (model_to_source_id[model_id] for model_id in group["member_ids"]),
                key=natural_sort_key,
            )
        family["variant_groups"].sort(
            key=lambda group: natural_sort_key(group["member_ids"][0])
        )
    normalized["singleton_units"] = [
        {"id": model_to_source_id[item["id"]]}
        for item in normalized["singleton_units"]
    ]
    normalized["singleton_units"].sort(key=lambda item: natural_sort_key(item["id"]))
    return normalized


def parse_and_normalize_output(
    raw_content: str,
    *,
    batch_id: str,
    input_ids: list[str],
) -> dict[str, Any]:
    text = strip_json_fence(raw_content)
    if not text:
        raise ValueError("Assistant content is empty.")
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("Output must be a JSON object.")
    require_exact_keys(parsed, {"rule_families"}, "output")
    families = parsed["rule_families"]
    if not isinstance(families, list):
        raise ValueError("rule_families must be an array.")

    input_id_set = set(input_ids)
    used_ids: set[str] = set()
    normalized_families: list[dict[str, Any]] = []
    for family_index, family in enumerate(families):
        location = f"rule_families[{family_index}]"
        if not isinstance(family, dict):
            raise ValueError(f"{location} must be an object.")
        require_exact_keys(
            family,
            {
                "base_unit",
                "equivalent_member_ids",
                "variant_groups",
                "family_rationale",
            },
            location,
        )

        base = family["base_unit"]
        if not isinstance(base, dict):
            raise ValueError(f"{location}.base_unit must be an object.")
        require_exact_keys(
            base,
            {"situation", "commonsense_rule"},
            f"{location}.base_unit",
        )
        normalized_base = {
            "situation": require_nonempty_string(
                base["situation"], f"{location}.base_unit.situation"
            ),
            "commonsense_rule": require_nonempty_string(
                base["commonsense_rule"], f"{location}.base_unit.commonsense_rule"
            ),
        }

        equivalent_ids = family["equivalent_member_ids"]
        variant_groups = family["variant_groups"]
        if not isinstance(equivalent_ids, list) or not all(
            isinstance(unique_id, str) and unique_id.strip()
            for unique_id in equivalent_ids
        ):
            raise ValueError(f"{location}.equivalent_member_ids is invalid.")
        equivalent_ids = [unique_id.strip() for unique_id in equivalent_ids]
        if len(equivalent_ids) != len(set(equivalent_ids)):
            raise ValueError(f"{location}.equivalent_member_ids has duplicates.")
        if not isinstance(variant_groups, list):
            raise ValueError(f"{location}.variant_groups must be an array.")

        family_ids = set(equivalent_ids)
        normalized_variant_groups: list[dict[str, Any]] = []
        for group_index, group in enumerate(variant_groups):
            group_location = f"{location}.variant_groups[{group_index}]"
            if not isinstance(group, dict):
                raise ValueError(f"{group_location} must be an object.")
            require_exact_keys(group, {"member_ids", "delta"}, group_location)
            member_ids = normalize_string_array(
                group["member_ids"], f"{group_location}.member_ids"
            )
            if not member_ids:
                raise ValueError(f"{group_location}.member_ids cannot be empty.")
            repeated_ids = family_ids & set(member_ids)
            if repeated_ids:
                duplicate = sorted(repeated_ids, key=natural_sort_key)[0]
                raise ValueError(f"ID {duplicate} is repeated within {location}.")
            delta = group["delta"]
            if not isinstance(delta, dict):
                raise ValueError(f"{group_location}.delta must be an object.")
            require_exact_keys(delta, set(DELTA_FIELDS), f"{group_location}.delta")
            normalized_delta = {
                field: normalize_string_array(
                    delta[field], f"{group_location}.delta.{field}"
                )
                for field in DELTA_FIELDS
            }
            if not any(normalized_delta.values()):
                raise ValueError("A variant delta must contain at least one item.")
            member_ids.sort(key=natural_sort_key)
            normalized_variant_groups.append(
                {
                    "member_ids": member_ids,
                    "delta": normalized_delta,
                }
            )
            family_ids.update(member_ids)

        if len(family_ids) < 2:
            raise ValueError(f"{location} must contain at least two source IDs.")
        llm_io.validate_disjoint_known_ids(
            sorted(family_ids, key=natural_sort_key),
            allowed_ids=input_id_set,
            used_ids=used_ids,
            location=location,
            min_size=2,
        )

        equivalent_ids.sort(key=natural_sort_key)
        normalized_variant_groups.sort(
            key=lambda item: natural_sort_key(item["member_ids"][0])
        )
        normalized_families.append(
            {
                "base_unit": normalized_base,
                "base_origin": "ANCHORED" if equivalent_ids else "SYNTHESIZED",
                "equivalent_member_ids": equivalent_ids,
                "variant_groups": normalized_variant_groups,
                "family_rationale": require_nonempty_string(
                    family["family_rationale"], f"{location}.family_rationale"
                ),
                "_sort_id": min(family_ids, key=natural_sort_key),
            }
        )

    normalized_families.sort(key=lambda item: natural_sort_key(item["_sort_id"]))
    for number, family in enumerate(normalized_families, start=1):
        family["family_id"] = f"{batch_id}.F{number}"
        del family["_sort_id"]

    singleton_units = [
        {"id": unique_id}
        for unique_id in llm_io.infer_singleton_ids(input_ids, used_ids)
    ]
    return {
        "rule_families": normalized_families,
        "singleton_units": singleton_units,
    }



def load_and_normalize_raw_response(
    *,
    raw_path: Path,
    batch: dict[str, Any],
    model: str,
    prompt_sha256: str,
) -> dict[str, Any]:
    raw_record = read_json(raw_path)
    if not isinstance(raw_record, dict):
        raise ValueError(f"Raw response must be a JSON object: {raw_path}")

    metadata = raw_record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = raw_record
    expected_metadata = {
        "batch_id": str(batch["batch_id"]),
        "model": model,
        "prompt_sha256": prompt_sha256,
        "input_sha256": batch["input_sha256"],
    }
    for field, expected in expected_metadata.items():
        if metadata.get(field) != expected:
            raise ValueError(
                f"Raw response metadata mismatch for {field}: {raw_path}"
            )

    expected_mapping = build_model_id_mapping(batch["units"])
    saved_mapping = raw_record.get("id_mapping")
    if saved_mapping is None:
        # Compatibility with responses created before batch-local IDs were used.
        model_to_source_id = {
            str(unit["id"]): str(unit["id"])
            for unit in batch["units"]
        }
    else:
        model_to_source_id = llm_io.parse_id_mapping_records(
            saved_mapping,
            expected_mapping=expected_mapping,
        )

    assistant_output = raw_record.get("assistant_output")
    if assistant_output is not None:
        assistant_content = json.dumps(assistant_output, ensure_ascii=False)
    else:
        assistant_content = raw_record.get("assistant_raw_text")
        if not isinstance(assistant_content, str):
            # Compatibility with raw files created before assistant_output existed.
            assistant_content = raw_record.get("assistant_content")
    if not isinstance(assistant_content, str):
        diagnostics = raw_record.get("diagnostics")
        extraction_error = (
            diagnostics.get("assistant_content_extraction_error")
            if isinstance(diagnostics, dict)
            else raw_record.get("assistant_content_extraction_error")
        )
        detail = f" ({extraction_error})" if extraction_error else ""
        raise ValueError(
            f"Raw response contains no assistant text: {raw_path}{detail}"
        )
    normalized = parse_and_normalize_output(
        assistant_content,
        batch_id=str(batch["batch_id"]),
        input_ids=list(model_to_source_id),
    )
    return restore_source_ids(normalized, model_to_source_id)


def normalized_runtime_record(
    *,
    runtime_record: dict[str, Any],
    batch: dict[str, Any],
    model: str,
    prompt_sha256: str,
) -> dict[str, Any]:
    raw_response_file = runtime_record.get("raw_response_file")
    if not isinstance(raw_response_file, str) or not raw_response_file:
        raise ValueError("Runtime record has no raw_response_file.")
    normalized = load_and_normalize_raw_response(
        raw_path=Path(raw_response_file),
        batch=batch,
        model=model,
        prompt_sha256=prompt_sha256,
    )
    rebuilt = {**runtime_record, "status": "success", **normalized}
    if runtime_record.get("status") != "success":
        rebuilt["recovered_from_edited_raw_response"] = True
    return rebuilt


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
    batch_id = str(batch["batch_id"])
    base_result = {
        "order": batch["order"],
        "batch_id": batch_id,
        "parent_community_id": batch.get("parent_community_id"),
        "parent_size": batch.get("parent_size"),
        "input_size": len(batch["units"]),
        "was_refined": batch.get("was_refined"),
        "model": model,
        "stream": args.stream,
        "prompt_sha256": prompt_sha256,
        "input_sha256": batch["input_sha256"],
    }
    model_to_source_id = build_model_id_mapping(batch["units"])
    model_units = build_model_units(batch["units"], model_to_source_id)
    user_prompt = build_user_prompt(user_template, model_units)
    usage_total: dict[str, int] = {}
    request_ids: list[str] = []
    last_raw_response_file = ""
    last_validation_error = ""

    try:
        for output_attempt in range(args.output_retries + 1):
            active_user_prompt = user_prompt
            if last_validation_error:
                active_user_prompt += (
                    "\n\nThe previous response failed validation with this error: "
                    f"{last_validation_error}\nRegenerate the complete JSON result."
                )
            payload: dict[str, Any] = {
                "model": model,
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": active_user_prompt},
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
            add_usage(usage_total, response_metadata["usage"])
            request_id = response_metadata.get("request_id")
            if isinstance(request_id, str):
                request_ids.append(request_id)
            last_raw_response_file = str(raw_path)
            try:
                load_and_normalize_raw_response(
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
                    **base_result,
                    "status": "invalid_output",
                    "raw_response_file": last_raw_response_file,
                    "validation_error": last_validation_error,
                    "output_attempts": output_attempt + 1,
                    "request_ids": request_ids,
                    "usage": usage_total,
                    "elapsed_seconds": round(time.monotonic() - started_at, 3),
                }
            return {
                **base_result,
                "status": "success",
                "raw_response_file": last_raw_response_file,
                "output_attempts": output_attempt + 1,
                "request_ids": request_ids,
                "usage": usage_total,
                "elapsed_seconds": round(time.monotonic() - started_at, 3),
            }
    except Exception as error:
        return {
            **base_result,
            "status": "error",
            "error_type": type(error).__name__,
            "error": str(error),
            "raw_response_file": last_raw_response_file or None,
            "validation_error": last_validation_error,
            "request_ids": request_ids,
            "usage": usage_total,
            "elapsed_seconds": round(time.monotonic() - started_at, 3),
        }
    raise RuntimeError("Unreachable batch processing state.")



def build_final_outputs(
    *,
    batches: list[dict[str, Any]],
    runtime_path: Path,
    results_path: Path,
    summary_path: Path,
    model: str,
    prompt_sha256: str,
    input_file: Path,
    source_files: list[Path],
    system_prompt_file: Path,
    user_prompt_file: Path,
    raw_responses_dir: Path,
) -> tuple[int, int]:
    latest = llm_io.latest_runtime_records(runtime_path)
    successful: list[dict[str, Any]] = []
    for batch in batches:
        key = (batch["batch_id"], model, prompt_sha256, batch["input_sha256"])
        record = latest.get(key)
        if record is not None:
            try:
                successful.append(
                    normalized_runtime_record(
                        runtime_record=record,
                        batch=batch,
                        model=model,
                        prompt_sha256=prompt_sha256,
                    )
                )
            except (OSError, json.JSONDecodeError, ValueError) as error:
                print(
                    f"batch={batch['batch_id']} cannot rebuild from raw response: "
                    f"{error}",
                    file=sys.stderr,
                )
    successful.sort(key=lambda record: record["order"])
    llm_io.write_jsonl(results_path, successful)

    usage_total: dict[str, int] = {}
    for record in successful:
        add_usage(usage_total, record.get("usage", {}))

    families = [
        family for record in successful for family in record["rule_families"]
    ]
    summary = {
        "input_file": str(input_file),
        "source_files": [str(path) for path in source_files],
        "system_prompt_file": str(system_prompt_file),
        "user_prompt_file": str(user_prompt_file),
        "model": model,
        "prompt_sha256": prompt_sha256,
        "input_batch_count": len(batches),
        "success_count": len(successful),
        "incomplete_count": len(batches) - len(successful),
        "rule_family_count": len(families),
        "anchored_family_count": sum(
            family["base_origin"] == "ANCHORED" for family in families
        ),
        "synthesized_family_count": sum(
            family["base_origin"] == "SYNTHESIZED" for family in families
        ),
        "equivalent_member_count": sum(
            len(family["equivalent_member_ids"]) for family in families
        ),
        "variant_group_count": sum(
            len(family["variant_groups"]) for family in families
        ),
        "variant_member_count": sum(
            len(group["member_ids"])
            for family in families
            for group in family["variant_groups"]
        ),
        "singleton_unit_count": sum(
            len(record["singleton_units"]) for record in successful
        ),
        "usage": usage_total,
        "raw_responses_dir": str(raw_responses_dir),
        "runtime_output": str(runtime_path),
        "results_output": str(results_path),
    }
    llm_io.write_json(summary_path, summary)
    return len(successful), len(batches) - len(successful)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generalize common-sense commonsense generalization records with an OpenAI-compatible LLM."
    )
    parser.add_argument("--batches-file", type=Path, default=DEFAULT_BATCHES_FILE)
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
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Use streaming responses. Non-streaming requests are the default.",
    )
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--output-retries", type=int, default=DEFAULT_OUTPUT_RETRIES)
    parser.add_argument("--retry-backoff", type=float, default=DEFAULT_RETRY_BACKOFF)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--parse-only",
        action="store_true",
        help="Rebuild final outputs from saved raw responses without calling the LLM.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.workers < 1 or args.timeout < 1:
        raise ValueError("workers and timeout must be positive.")
    if args.max_tokens is not None and args.max_tokens < 1:
        raise ValueError("max_tokens must be positive when provided.")
    if args.max_retries < 0 or args.output_retries < 0:
        raise ValueError("retry counts cannot be negative.")
    if args.retry_backoff < 0:
        raise ValueError("retry_backoff cannot be negative.")
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive.")
    if args.parse_only and args.overwrite:
        raise ValueError("--parse-only cannot be combined with --overwrite.")


def main() -> None:
    args = parse_args()
    validate_args(args)
    batches_path = resolve_path(args.batches_file)
    source_paths = [resolve_path(path) for path in args.source_files]
    system_prompt_path = resolve_path(args.system_prompt_file)
    user_prompt_path = resolve_path(args.user_prompt_file)
    output_dir = resolve_path(args.output_dir)
    for path in [batches_path, *source_paths, system_prompt_path, user_prompt_path]:
        if not path.is_file():
            raise FileNotFoundError(f"Required file does not exist: {path}")

    system_prompt = system_prompt_path.read_text(encoding="utf-8").strip()
    user_template = user_prompt_path.read_text(encoding="utf-8").strip()
    if UNITS_PLACEHOLDER in system_prompt:
        raise ValueError("The system prompt must not contain the units placeholder.")
    source_units = load_source_units(source_paths)
    batches = load_enriched_batches(batches_path, source_units)
    if not batches:
        raise ValueError(f"Input contains no batches: {batches_path}")
    first_mapping = build_model_id_mapping(batches[0]["units"])
    first_model_units = build_model_units(batches[0]["units"], first_mapping)
    build_user_prompt(user_template, first_model_units)
    prompt_hash = sha256_text(system_prompt + "\0" + user_template)

    if args.dry_run:
        print(
            f"Dry run OK. batches={len(batches)}, source_units={len(source_units)}, "
            f"first_batch={batches[0]['batch_id']}, "
            f"first_batch_size={len(batches[0]['units'])}, "
            f"model={LLM_MODEL or '<missing>'}, stream={args.stream}, "
            f"system_chars={len(system_prompt)}"
        )
        print(build_user_prompt(user_template, first_model_units))
        return

    required_env = (
        (("DEEPSEEK_V4_FLASH", LLM_MODEL),)
        if args.parse_only
        else (
            ("DEEPSEEK_API_KEY", DEEPSEEK_API_KEY),
            ("DEEPSEEK_URL", DEEPSEEK_URL),
            ("DEEPSEEK_V4_FLASH", LLM_MODEL),
        )
    )
    missing_env = [
        name
        for name, value in required_env
        if not value
    ]
    if missing_env:
        raise ValueError(f"Missing .env values: {', '.join(missing_env)}")

    runtime_path = output_dir / RUNTIME_FILENAME
    results_path = output_dir / RESULTS_FILENAME
    summary_path = output_dir / SUMMARY_FILENAME
    raw_responses_dir = output_dir / RAW_RESPONSES_DIRNAME
    if args.overwrite:
        for path in (runtime_path, results_path, summary_path):
            if path.exists():
                path.unlink()

    latest = llm_io.latest_runtime_records(runtime_path)
    successful_keys: set[tuple[str, str, str, str]] = set()
    for batch in batches:
        key = (batch["batch_id"], LLM_MODEL, prompt_hash, batch["input_sha256"])
        record = latest.get(key)
        if record is None:
            continue
        try:
            normalized_runtime_record(
                runtime_record=record,
                batch=batch,
                model=LLM_MODEL,
                prompt_sha256=prompt_hash,
            )
        except (OSError, json.JSONDecodeError, ValueError) as error:
            print(
                f"batch={batch['batch_id']} existing raw response is not usable; "
                f"the batch will be retried: {error}",
                file=sys.stderr,
            )
        else:
            successful_keys.add(key)
    pending = [
        batch
        for batch in batches
        if (batch["batch_id"], LLM_MODEL, prompt_hash, batch["input_sha256"])
        not in successful_keys
    ]
    if args.limit is not None:
        pending = pending[: args.limit]
    if args.parse_only:
        pending = []

    output_dir.mkdir(parents=True, exist_ok=True)
    status_counts: dict[str, int] = {}
    with runtime_path.open("a", encoding="utf-8") as runtime_file:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    process_batch,
                    batch=batch,
                    system_prompt=system_prompt,
                    user_template=user_template,
                    prompt_sha256=prompt_hash,
                    base_url=resolve_base_url(DEEPSEEK_URL),
                    api_key=DEEPSEEK_API_KEY,
                    model=LLM_MODEL,
                    raw_responses_dir=raw_responses_dir,
                    args=args,
                ): batch["batch_id"]
                for batch in pending
            }
            for completed, future in enumerate(
                concurrent.futures.as_completed(futures), start=1
            ):
                result = future.result()
                llm_io.append_jsonl(runtime_file, result)
                status = str(result["status"])
                status_counts[status] = status_counts.get(status, 0) + 1
                print(
                    f"[{completed}/{len(pending)}] batch={result['batch_id']} "
                    f"status={status} elapsed={result['elapsed_seconds']}s"
                )

    success_count, incomplete_count = build_final_outputs(
        batches=batches,
        runtime_path=runtime_path,
        results_path=results_path,
        summary_path=summary_path,
        model=LLM_MODEL,
        prompt_sha256=prompt_hash,
        input_file=args.batches_file,
        source_files=args.source_files,
        system_prompt_file=args.system_prompt_file,
        user_prompt_file=args.user_prompt_file,
        raw_responses_dir=raw_responses_dir,
    )
    print(
        f"Done. submitted={len(pending)}, statuses={status_counts}, "
        f"total_success={success_count}, incomplete={incomplete_count}, "
        f"output={output_dir}"
    )
    failed_this_run = len(pending) - status_counts.get("success", 0)
    if failed_this_run or (
        not args.parse_only and args.limit is None and incomplete_count
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

