"""Incrementally integrate accepted units into established commonsense generalization records."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
from commonsense_repro.common.paths import PACKAGE_ROOT as PROJECT_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from commonsense_repro.common import llm_pipeline_common as llm_io

load_dotenv(PROJECT_ROOT / ".env")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_URL = os.getenv("DEEPSEEK_URL", "").strip()
LLM_MODEL = os.getenv("DEEPSEEK_V4_FLASH", "").strip()

DATASET_ROOT = Path(
    "work/stage4/"
    "commonsense_consolidation"
)
DEFAULT_PLAN_FILE = DATASET_ROOT / "batch_consolidation/generalization_plan_max18.json"
DEFAULT_SYSTEM_PROMPT_FILE = Path(
    "prompts/consolidate_batched_unassigned_commonsense_v1.1_system.md"
)
DEFAULT_USER_PROMPT_FILE = Path(
    "prompts/consolidate_batched_unassigned_commonsense_v1.1_user.md"
)
DEFAULT_OUTPUT_DIR = DATASET_ROOT / "batch_consolidation/deepseek_flash_v1_1"

DEFAULT_WORKERS = 12
DEFAULT_MAX_TOKENS: int | None = None
DEFAULT_TIMEOUT = 1200
DEFAULT_MAX_RETRIES = 3
DEFAULT_OUTPUT_RETRIES = 2
DEFAULT_RETRY_BACKOFF = 2.0

INPUT_PLACEHOLDER = "{{input_json}}"
RUNTIME_FILENAME = "batch_consolidation_runtime.jsonl"
BATCH_RESULTS_FILENAME = "batch_consolidation_batch_results.jsonl"
FAMILY_RESULTS_FILENAME = "batch_consolidation_results.jsonl"
ASSIGNMENTS_FILENAME = "pending_unit_assignments.jsonl"
SUMMARY_FILENAME = "batch_consolidation_summary.json"
RAW_RESPONSES_DIRNAME = "raw_responses"

FAMILY_FIELDS = {
    "base_unit",
    "equivalent_member_ids",
    "variant_groups",
    "family_rationale",
}
DELTA_FIELDS = {"situation_differences", "commonsense_rule_differences"}
BASE_ORIGINS = {"ANCHORED", "SYNTHESIZED"}


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def family_member_ids(family: dict[str, Any]) -> list[str]:
    ids = list(family["equivalent_member_ids"])
    for group in family["variant_groups"]:
        ids.extend(group["member_ids"])
    return sorted(ids, key=llm_io.natural_sort_key)


def normalize_delta(value: Any, location: str) -> dict[str, list[str]]:
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be an object.")
    llm_io.require_exact_keys(value, DELTA_FIELDS, location)
    normalized = {
        field: llm_io.normalize_string_array(
            value[field], f"{location}.{field}"
        )
        for field in sorted(DELTA_FIELDS)
    }
    if not any(normalized.values()):
        raise ValueError(f"{location} must contain at least one difference.")
    return {
        "situation_differences": normalized["situation_differences"],
        "commonsense_rule_differences": normalized[
            "commonsense_rule_differences"
        ],
    }


def normalize_family(
    value: Any,
    *,
    allowed_ids: set[str] | None,
    location: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be an object.")
    llm_io.require_exact_keys(value, FAMILY_FIELDS, location)

    base = value["base_unit"]
    if not isinstance(base, dict):
        raise ValueError(f"{location}.base_unit must be an object.")
    llm_io.require_exact_keys(
        base, {"situation", "commonsense_rule"}, f"{location}.base_unit"
    )
    normalized_base = {
        "situation": llm_io.require_nonempty_string(
            base["situation"], f"{location}.base_unit.situation"
        ),
        "commonsense_rule": llm_io.require_nonempty_string(
            base["commonsense_rule"],
            f"{location}.base_unit.commonsense_rule",
        ),
    }

    equivalent_ids = llm_io.normalize_string_array(
        value["equivalent_member_ids"],
        f"{location}.equivalent_member_ids",
    )
    if len(equivalent_ids) != len(set(equivalent_ids)):
        raise ValueError(f"{location}.equivalent_member_ids has duplicates.")

    variant_groups = value["variant_groups"]
    if not isinstance(variant_groups, list):
        raise ValueError(f"{location}.variant_groups must be an array.")
    used_ids = set(equivalent_ids)
    normalized_groups: list[dict[str, Any]] = []
    for group_index, group in enumerate(variant_groups):
        group_location = f"{location}.variant_groups[{group_index}]"
        if not isinstance(group, dict):
            raise ValueError(f"{group_location} must be an object.")
        llm_io.require_exact_keys(group, {"member_ids", "delta"}, group_location)
        member_ids = llm_io.normalize_string_array(
            group["member_ids"], f"{group_location}.member_ids"
        )
        if not member_ids:
            raise ValueError(f"{group_location}.member_ids cannot be empty.")
        if len(member_ids) != len(set(member_ids)):
            raise ValueError(f"{group_location}.member_ids has duplicates.")
        overlap = used_ids & set(member_ids)
        if overlap:
            duplicate = sorted(overlap, key=llm_io.natural_sort_key)[0]
            raise ValueError(f"ID {duplicate} is repeated within {location}.")
        used_ids.update(member_ids)
        member_ids.sort(key=llm_io.natural_sort_key)
        normalized_groups.append(
            {
                "member_ids": member_ids,
                "delta": normalize_delta(
                    group["delta"], f"{group_location}.delta"
                ),
            }
        )

    if len(used_ids) < 2:
        raise ValueError(f"{location} must contain at least two member IDs.")
    if allowed_ids is not None:
        unknown = used_ids - allowed_ids
        if unknown:
            unknown_id = sorted(unknown, key=llm_io.natural_sort_key)[0]
            raise ValueError(f"{location} contains unknown ID {unknown_id}.")

    equivalent_ids.sort(key=llm_io.natural_sort_key)
    normalized_groups.sort(
        key=lambda group: llm_io.natural_sort_key(group["member_ids"][0])
    )
    member_ids = sorted(used_ids, key=llm_io.natural_sort_key)
    return {
        "base_unit": normalized_base,
        "member_count": len(member_ids),
        "member_ids": member_ids,
        "equivalent_member_ids": equivalent_ids,
        "variant_groups": normalized_groups,
        "family_rationale": llm_io.require_nonempty_string(
            value["family_rationale"], f"{location}.family_rationale"
        ),
    }


def normalize_base_origin(value: Any, location: str) -> str:
    origin = llm_io.require_nonempty_string(value, location)
    if origin not in BASE_ORIGINS:
        allowed = ", ".join(sorted(BASE_ORIGINS))
        raise ValueError(f"{location} must be one of: {allowed}.")
    return origin


def resolve_updated_base_origin(
    *,
    previous_base_unit: dict[str, str],
    previous_base_origin: str,
    updated_base_unit: dict[str, str],
) -> str:
    if updated_base_unit == previous_base_unit:
        return previous_base_origin
    return "SYNTHESIZED"


def family_prompt_value(family: dict[str, Any]) -> dict[str, Any]:
    return {
        "base_unit": family["base_unit"],
        "equivalent_member_ids": family["equivalent_member_ids"],
        "variant_groups": family["variant_groups"],
        "family_rationale": family["family_rationale"],
    }


def normalize_plan_family(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be an object.")
    semantic_value = {field: value.get(field) for field in FAMILY_FIELDS}
    normalized = normalize_family(
        semantic_value, allowed_ids=None, location=location
    )
    normalized["base_origin"] = normalize_base_origin(
        value.get("base_origin"), f"{location}.base_origin"
    )
    family_id = llm_io.require_nonempty_string(
        value.get("family_id"), f"{location}.family_id"
    )
    return {"family_id": family_id, **normalized}


def load_plan(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = llm_io.read_json(path)
    if not isinstance(root, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    plans = root.get("family_plans")
    if not isinstance(plans, list) or not plans:
        raise ValueError(f"{path}.family_plans must be a non-empty array.")

    normalized_plans: list[dict[str, Any]] = []
    seen_family_ids: set[str] = set()
    seen_pending_ids: set[str] = set()
    for order, plan in enumerate(plans):
        location = f"{path}.family_plans[{order}]"
        if not isinstance(plan, dict):
            raise ValueError(f"{location} must be an object.")
        family_id = llm_io.require_nonempty_string(
            plan.get("family_id"), f"{location}.family_id"
        )
        if family_id in seen_family_ids:
            raise ValueError(f"Duplicate family plan: {family_id}")
        initial_family = normalize_plan_family(
            plan.get("initial_family"), f"{location}.initial_family"
        )
        if initial_family["family_id"] != family_id:
            raise ValueError(f"{location} has inconsistent family IDs.")

        raw_batches = plan.get("batches")
        if not isinstance(raw_batches, list) or not raw_batches:
            raise ValueError(f"{location}.batches must be a non-empty array.")
        batches: list[dict[str, Any]] = []
        previous_batch_id: str | None = None
        family_pending_ids: set[str] = set()
        for batch_offset, raw_batch in enumerate(raw_batches):
            batch_location = f"{location}.batches[{batch_offset}]"
            if not isinstance(raw_batch, dict):
                raise ValueError(f"{batch_location} must be an object.")
            batch_id = llm_io.require_nonempty_string(
                raw_batch.get("batch_id"), f"{batch_location}.batch_id"
            )
            if raw_batch.get("batch_index") != batch_offset + 1:
                raise ValueError(f"{batch_location} has an invalid batch_index.")
            if raw_batch.get("batch_count") != len(raw_batches):
                raise ValueError(f"{batch_location} has an invalid batch_count.")
            if raw_batch.get("depends_on_batch_id") != previous_batch_id:
                raise ValueError(f"{batch_location} has an invalid dependency.")
            pending_units = raw_batch.get("pending_units")
            if not isinstance(pending_units, list) or not pending_units:
                raise ValueError(
                    f"{batch_location}.pending_units must be a non-empty array."
                )
            normalized_units: list[dict[str, Any]] = []
            for unit_index, unit in enumerate(pending_units):
                unit_location = f"{batch_location}.pending_units[{unit_index}]"
                if not isinstance(unit, dict):
                    raise ValueError(f"{unit_location} must be an object.")
                llm_io.require_exact_keys(
                    unit,
                    {
                        "id",
                        "situation",
                        "violated_commonsense_rule",
                        "applicability_conditions",
                    },
                    unit_location,
                )
                unit_id = llm_io.require_nonempty_string(
                    unit["id"], f"{unit_location}.id"
                )
                if unit_id in family_pending_ids or unit_id in seen_pending_ids:
                    raise ValueError(
                        f"Pending unit {unit_id} appears in multiple plan batches."
                    )
                family_pending_ids.add(unit_id)
                seen_pending_ids.add(unit_id)
                normalized_units.append(
                    {
                        "id": unit_id,
                        "situation": llm_io.require_nonempty_string(
                            unit["situation"], f"{unit_location}.situation"
                        ),
                        "violated_commonsense_rule": llm_io.require_nonempty_string(
                            unit["violated_commonsense_rule"],
                            f"{unit_location}.violated_commonsense_rule",
                        ),
                        "applicability_conditions": llm_io.normalize_string_array(
                            unit["applicability_conditions"],
                            f"{unit_location}.applicability_conditions",
                        ),
                    }
                )
            if raw_batch.get("pending_unit_count") != len(normalized_units):
                raise ValueError(
                    f"{batch_location}.pending_unit_count is inconsistent."
                )
            batches.append(
                {
                    "batch_id": batch_id,
                    "batch_index": batch_offset + 1,
                    "batch_count": len(raw_batches),
                    "depends_on_batch_id": previous_batch_id,
                    "pending_units": normalized_units,
                }
            )
            previous_batch_id = batch_id

        if plan.get("pending_unit_count") != len(family_pending_ids):
            raise ValueError(f"{location}.pending_unit_count is inconsistent.")
        old_ids = set(initial_family["member_ids"])
        overlap = old_ids & family_pending_ids
        if overlap:
            duplicate = sorted(overlap, key=llm_io.natural_sort_key)[0]
            raise ValueError(
                f"Pending unit {duplicate} is already an existing family member."
            )
        normalized_plans.append(
            {
                "order": order,
                "family_id": family_id,
                "initial_family": initial_family,
                "pending_unit_count": len(family_pending_ids),
                "batches": batches,
            }
        )
        seen_family_ids.add(family_id)
    return normalized_plans, root.get("summary", {})


def map_family_ids(
    family: dict[str, Any], source_to_model_id: dict[str, str]
) -> dict[str, Any]:
    return {
        "base_unit": family["base_unit"],
        "equivalent_member_ids": [
            source_to_model_id[source_id]
            for source_id in family["equivalent_member_ids"]
        ],
        "variant_groups": [
            {
                "member_ids": [
                    source_to_model_id[source_id]
                    for source_id in group["member_ids"]
                ],
                "delta": group["delta"],
            }
            for group in family["variant_groups"]
        ],
        "family_rationale": family["family_rationale"],
    }


def materialize_batch(
    plan: dict[str, Any],
    batch_template: dict[str, Any],
    current_family: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    existing_ids = family_member_ids(current_family)
    pending_units = batch_template["pending_units"]
    pending_ids = [str(unit["id"]) for unit in pending_units]
    overlap = set(existing_ids) & set(pending_ids)
    if overlap:
        duplicate = sorted(overlap, key=llm_io.natural_sort_key)[0]
        raise ValueError(f"Unit {duplicate} is already present before its batch.")

    mapping_units = [{"id": source_id} for source_id in existing_ids]
    mapping_units.extend({"id": source_id} for source_id in pending_ids)
    model_to_source_id = llm_io.build_model_id_mapping(mapping_units)
    source_to_model_id = llm_io.invert_id_mapping(model_to_source_id)
    model_input = {
        "existing_family": map_family_ids(current_family, source_to_model_id),
        "pending_units": [
            {
                "id": source_to_model_id[str(unit["id"])],
                "situation": unit["situation"],
                "violated_commonsense_rule": unit[
                    "violated_commonsense_rule"
                ],
                "applicability_conditions": unit[
                    "applicability_conditions"
                ],
            }
            for unit in pending_units
        ],
    }
    source_input = {
        "existing_family": family_prompt_value(current_family),
        "pending_units": pending_units,
    }
    batch = {
        "order": plan["order"],
        "family_id": plan["family_id"],
        "batch_id": batch_template["batch_id"],
        "batch_index": batch_template["batch_index"],
        "batch_count": batch_template["batch_count"],
        "depends_on_batch_id": batch_template["depends_on_batch_id"],
        "existing_member_ids": existing_ids,
        "existing_base_unit": current_family["base_unit"],
        "existing_base_origin": current_family["base_origin"],
        "pending_units": pending_units,
        "pending_unit_ids": pending_ids,
        "units": mapping_units,
        "model_to_source_id": model_to_source_id,
        "identifier_scheme": "batch_local_u_index_v1",
        "input_sha256": llm_io.sha256_text(
            json.dumps(source_input, ensure_ascii=False, sort_keys=True)
        ),
    }
    return batch, model_input


def parse_generalization_output(
    raw_content: str,
    *,
    existing_ids: set[str],
    pending_ids: set[str],
) -> dict[str, Any]:
    text = llm_io.strip_json_fence(raw_content)
    if not text:
        raise ValueError("Assistant content is empty.")
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("Output must be a JSON object.")
    llm_io.require_exact_keys(
        parsed, {"updated_family", "rejected_pending_units"}, "output"
    )
    allowed_ids = existing_ids | pending_ids
    family = normalize_family(
        parsed["updated_family"],
        allowed_ids=allowed_ids,
        location="output.updated_family",
    )
    used_ids = set(family["member_ids"])
    missing_existing = existing_ids - used_ids
    if missing_existing:
        missing = sorted(missing_existing, key=llm_io.natural_sort_key)[0]
        raise ValueError(f"Existing member {missing} is missing from updated_family.")

    rejected = parsed["rejected_pending_units"]
    if not isinstance(rejected, list):
        raise ValueError("output.rejected_pending_units must be an array.")
    normalized_rejected: list[dict[str, str]] = []
    rejected_ids: set[str] = set()
    for index, item in enumerate(rejected):
        location = f"output.rejected_pending_units[{index}]"
        if not isinstance(item, dict):
            raise ValueError(f"{location} must be an object.")
        llm_io.require_exact_keys(item, {"id", "reason"}, location)
        unit_id = llm_io.require_nonempty_string(item["id"], f"{location}.id")
        if unit_id not in pending_ids:
            raise ValueError(f"{location} references non-pending ID {unit_id}.")
        if unit_id in rejected_ids:
            raise ValueError(f"Pending ID {unit_id} is rejected more than once.")
        if unit_id in used_ids:
            raise ValueError(f"Pending ID {unit_id} is both accepted and rejected.")
        rejected_ids.add(unit_id)
        normalized_rejected.append(
            {
                "id": unit_id,
                "reason": llm_io.require_nonempty_string(
                    item["reason"], f"{location}.reason"
                ),
            }
        )

    accepted_pending = used_ids & pending_ids
    if accepted_pending | rejected_ids != pending_ids:
        missing = sorted(
            pending_ids - accepted_pending - rejected_ids,
            key=llm_io.natural_sort_key,
        )[0]
        raise ValueError(f"Pending ID {missing} is neither accepted nor rejected.")
    normalized_rejected.sort(key=lambda item: llm_io.natural_sort_key(item["id"]))
    return {
        "updated_family": family,
        "accepted_pending_unit_ids": sorted(
            accepted_pending, key=llm_io.natural_sort_key
        ),
        "rejected_pending_units": normalized_rejected,
    }


def restore_output_ids(
    normalized: dict[str, Any], model_to_source_id: dict[str, str]
) -> dict[str, Any]:
    family = normalized["updated_family"]
    family["equivalent_member_ids"] = llm_io.restore_ids(
        family["equivalent_member_ids"], model_to_source_id
    )
    for group in family["variant_groups"]:
        group["member_ids"] = llm_io.restore_ids(
            group["member_ids"], model_to_source_id
        )
    family["variant_groups"].sort(
        key=lambda group: llm_io.natural_sort_key(group["member_ids"][0])
    )
    family["member_ids"] = family_member_ids(family)
    family["member_count"] = len(family["member_ids"])
    normalized["accepted_pending_unit_ids"] = llm_io.restore_ids(
        normalized["accepted_pending_unit_ids"], model_to_source_id
    )
    normalized["rejected_pending_units"] = sorted(
        (
            {**item, "id": model_to_source_id[item["id"]]}
            for item in normalized["rejected_pending_units"]
        ),
        key=lambda item: llm_io.natural_sort_key(item["id"]),
    )
    return normalized


def read_raw_generalization(
    *,
    raw_path: Path,
    batch: dict[str, Any],
    model: str,
    prompt_sha256: str,
) -> dict[str, Any]:
    raw_record = llm_io.read_json(raw_path)
    if not isinstance(raw_record, dict):
        raise ValueError(f"Raw response must be an object: {raw_path}")
    metadata = raw_record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = raw_record
    expected_metadata = {
        "batch_id": batch["batch_id"],
        "model": model,
        "prompt_sha256": prompt_sha256,
        "input_sha256": batch["input_sha256"],
    }
    for field, expected in expected_metadata.items():
        if metadata.get(field) != expected:
            raise ValueError(f"Raw response metadata mismatch for {field}: {raw_path}")

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
        raise ValueError(f"Raw response contains no assistant text: {raw_path}")

    source_to_model_id = llm_io.invert_id_mapping(mapping)
    normalized = parse_generalization_output(
        assistant_content,
        existing_ids={source_to_model_id[item] for item in batch["existing_member_ids"]},
        pending_ids={source_to_model_id[item] for item in batch["pending_unit_ids"]},
    )
    restored = restore_output_ids(normalized, mapping)
    restored_family = restored["updated_family"]
    restored_family["base_origin"] = resolve_updated_base_origin(
        previous_base_unit=batch["existing_base_unit"],
        previous_base_origin=batch["existing_base_origin"],
        updated_base_unit=restored_family["base_unit"],
    )
    return restored


def process_batch(
    *,
    batch: dict[str, Any],
    model_input: dict[str, Any],
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
    user_prompt = llm_io.build_user_prompt(
        user_template, model_input, placeholder=INPUT_PLACEHOLDER
    )
    base_record = {
        "order": batch["order"],
        "family_id": batch["family_id"],
        "batch_id": batch["batch_id"],
        "batch_index": batch["batch_index"],
        "batch_count": batch["batch_count"],
        "depends_on_batch_id": batch["depends_on_batch_id"],
        "existing_member_count": len(batch["existing_member_ids"]),
        "pending_unit_count": len(batch["pending_unit_ids"]),
        "pending_unit_ids": batch["pending_unit_ids"],
        "model": model,
        "stream": args.stream,
        "prompt_sha256": prompt_sha256,
        "input_sha256": batch["input_sha256"],
    }
    usage_total: dict[str, int] = {}
    request_ids: list[str] = []
    last_raw_path = ""
    last_validation_error = ""

    try:
        for output_attempt in range(args.output_retries + 1):
            active_prompt = user_prompt
            if last_validation_error:
                active_prompt += (
                    "\n\nThe previous response failed JSON validation with this "
                    f"error: {last_validation_error}\n"
                    "Return a corrected complete JSON object."
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
                read_raw_generalization(
                    raw_path=raw_path,
                    batch=batch,
                    model=model,
                    prompt_sha256=prompt_sha256,
                )
            except (json.JSONDecodeError, KeyError, ValueError) as error:
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
            "error_type": type(error).__name__,
            "error": str(error),
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
    normalized = read_raw_generalization(
        raw_path=Path(raw_path),
        batch=batch,
        model=model,
        prompt_sha256=prompt_sha256,
    )
    rebuilt = {**record, "status": "success", **normalized}
    if record.get("status") != "success":
        rebuilt["recovered_from_edited_raw_response"] = True
    return rebuilt


def execute_family_plan(
    *,
    plan: dict[str, Any],
    latest: dict[tuple[str, str, str, str], dict[str, Any]],
    persist_runtime: Callable[[dict[str, Any]], None],
    system_prompt: str,
    user_template: str,
    prompt_sha256: str,
    base_url: str,
    api_key: str,
    model: str,
    raw_responses_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    current_family = plan["initial_family"]
    completed_batches = 0
    for batch_template in plan["batches"]:
        batch, model_input = materialize_batch(plan, batch_template, current_family)
        key = (batch["batch_id"], model, prompt_sha256, batch["input_sha256"])
        record = latest.get(key)
        normalized: dict[str, Any] | None = None
        if not args.overwrite and record is not None:
            try:
                normalized = normalize_runtime_record(
                    record, batch, model, prompt_sha256
                )
            except (OSError, json.JSONDecodeError, KeyError, ValueError):
                normalized = None
        if normalized is None:
            if args.parse_only:
                break
            record = process_batch(
                batch=batch,
                model_input=model_input,
                system_prompt=system_prompt,
                user_template=user_template,
                prompt_sha256=prompt_sha256,
                base_url=base_url,
                api_key=api_key,
                model=model,
                raw_responses_dir=raw_responses_dir,
                args=args,
            )
            persist_runtime(record)
            if record["status"] != "success":
                break
            normalized = normalize_runtime_record(
                record, batch, model, prompt_sha256
            )
        current_family = normalized["updated_family"]
        completed_batches += 1
    return {
        "family_id": plan["family_id"],
        "completed_batches": completed_batches,
        "batch_count": len(plan["batches"]),
    }


def reconstruct_outputs(
    *,
    plans: list[dict[str, Any]],
    runtime_path: Path,
    batch_results_path: Path,
    family_results_path: Path,
    assignments_path: Path,
    summary_path: Path,
    model: str,
    prompt_sha256: str,
    plan_path: Path,
    system_prompt_path: Path,
    user_prompt_path: Path,
    raw_responses_dir: Path,
) -> tuple[int, int]:
    latest = llm_io.latest_runtime_records(runtime_path)
    batch_results: list[dict[str, Any]] = []
    family_results: list[dict[str, Any]] = []
    assignments: list[dict[str, Any]] = []
    expected_batch_count = sum(len(plan["batches"]) for plan in plans)

    for plan in plans:
        current_family = plan["initial_family"]
        rejected_by_id: dict[str, dict[str, str]] = {}
        successful_batch_ids: list[str] = []
        complete = True
        for batch_template in plan["batches"]:
            batch, _ = materialize_batch(plan, batch_template, current_family)
            key = (batch["batch_id"], model, prompt_sha256, batch["input_sha256"])
            record = latest.get(key)
            if record is None:
                complete = False
                break
            try:
                result = normalize_runtime_record(
                    record, batch, model, prompt_sha256
                )
            except (OSError, json.JSONDecodeError, KeyError, ValueError) as error:
                print(
                    f"batch={batch['batch_id']} cannot parse raw response: {error}",
                    file=sys.stderr,
                )
                complete = False
                break
            result["updated_family"] = {
                "family_id": plan["family_id"], **result["updated_family"]
            }
            batch_results.append(result)
            successful_batch_ids.append(batch["batch_id"])
            for rejected in result["rejected_pending_units"]:
                rejected_by_id[rejected["id"]] = rejected
            current_family = result["updated_family"]

        if not complete:
            continue

        final_family = {
            "family_id": plan["family_id"],
            "family_origin": "INCREMENTALLY_UPDATED",
            **family_prompt_value(current_family),
            "base_origin": current_family["base_origin"],
            "member_count": current_family["member_count"],
            "member_ids": current_family["member_ids"],
            "provenance": {
                "initial_family_id": plan["family_id"],
                "generalization_batch_ids": successful_batch_ids,
                "model": model,
                "prompt_sha256": prompt_sha256,
            },
        }
        pending_ids = {
            unit["id"]
            for batch in plan["batches"]
            for unit in batch["pending_units"]
        }
        accepted_ids = pending_ids & set(final_family["member_ids"])
        if accepted_ids | set(rejected_by_id) != pending_ids:
            raise ValueError(
                f"Final result for {plan['family_id']} does not cover all pending IDs."
            )
        family_results.append(
            {
                "order": plan["order"],
                "family_id": plan["family_id"],
                "input_pending_unit_count": len(pending_ids),
                "accepted_pending_unit_count": len(accepted_ids),
                "rejected_pending_unit_count": len(rejected_by_id),
                "batch_ids": successful_batch_ids,
                "updated_family": final_family,
                "rejected_pending_units": sorted(
                    rejected_by_id.values(),
                    key=lambda item: llm_io.natural_sort_key(item["id"]),
                ),
            }
        )

        equivalent_set = set(final_family["equivalent_member_ids"])
        variant_by_id: dict[str, tuple[int, dict[str, Any]]] = {}
        for group_index, group in enumerate(final_family["variant_groups"]):
            for source_id in group["member_ids"]:
                variant_by_id[source_id] = (group_index, group["delta"])
        for source_id in sorted(pending_ids, key=llm_io.natural_sort_key):
            if source_id in rejected_by_id:
                assignments.append(
                    {
                        "unit_id": source_id,
                        "family_id": plan["family_id"],
                        "status": "REJECTED",
                        "membership_type": None,
                        "variant_group_index": None,
                        "delta": None,
                        "reason": rejected_by_id[source_id]["reason"],
                    }
                )
            elif source_id in equivalent_set:
                assignments.append(
                    {
                        "unit_id": source_id,
                        "family_id": plan["family_id"],
                        "status": "ACCEPTED",
                        "membership_type": "EQUIVALENT_MEMBER",
                        "variant_group_index": None,
                        "delta": None,
                        "reason": None,
                    }
                )
            else:
                group_index, delta = variant_by_id[source_id]
                assignments.append(
                    {
                        "unit_id": source_id,
                        "family_id": plan["family_id"],
                        "status": "ACCEPTED",
                        "membership_type": "VARIANT_MEMBER",
                        "variant_group_index": group_index,
                        "delta": delta,
                        "reason": None,
                    }
                )

    batch_results.sort(key=lambda item: (item["order"], item["batch_index"]))
    family_results.sort(key=lambda item: item["order"])
    assignments.sort(key=lambda item: llm_io.natural_sort_key(item["unit_id"]))
    llm_io.write_jsonl(batch_results_path, batch_results)
    llm_io.write_jsonl(family_results_path, family_results)
    llm_io.write_jsonl(assignments_path, assignments)

    usage: dict[str, int] = {}
    for result in batch_results:
        llm_io.add_usage(usage, result.get("usage", {}))
    accepted = sum(item["status"] == "ACCEPTED" for item in assignments)
    equivalent = sum(
        item["membership_type"] == "EQUIVALENT_MEMBER" for item in assignments
    )
    rejected = sum(item["status"] == "REJECTED" for item in assignments)
    summary = {
        "plan_file": str(plan_path),
        "system_prompt_file": str(system_prompt_path),
        "user_prompt_file": str(user_prompt_path),
        "model": model,
        "prompt_sha256": prompt_sha256,
        "input_family_count": len(plans),
        "input_batch_count": expected_batch_count,
        "input_pending_unit_count": sum(
            plan["pending_unit_count"] for plan in plans
        ),
        "successful_batch_count": len(batch_results),
        "complete_family_count": len(family_results),
        "incomplete_family_count": len(plans) - len(family_results),
        "resolved_pending_unit_count": len(assignments),
        "accepted_pending_unit_count": accepted,
        "equivalent_pending_unit_count": equivalent,
        "variant_pending_unit_count": accepted - equivalent,
        "rejected_pending_unit_count": rejected,
        "usage": usage,
        "runtime_output": str(runtime_path),
        "batch_results_output": str(batch_results_path),
        "family_results_output": str(family_results_path),
        "assignments_output": str(assignments_path),
        "raw_responses_dir": str(raw_responses_dir),
    }
    llm_io.write_json(summary_path, summary)
    return len(family_results), len(plans) - len(family_results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-file", type=Path, default=DEFAULT_PLAN_FILE)
    parser.add_argument(
        "--system-prompt-file", type=Path, default=DEFAULT_SYSTEM_PROMPT_FILE
    )
    parser.add_argument(
        "--user-prompt-file", type=Path, default=DEFAULT_USER_PROMPT_FILE
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--stream", action="store_true", help="Use streaming responses."
    )
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--output-retries", type=int, default=DEFAULT_OUTPUT_RETRIES)
    parser.add_argument("--retry-backoff", type=float, default=DEFAULT_RETRY_BACKOFF)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N family plans, including their dependencies.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--parse-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.workers < 1:
        raise ValueError("--workers must be positive.")
    if args.timeout < 1:
        raise ValueError("--timeout must be positive.")
    if args.max_retries < 0 or args.output_retries < 0:
        raise ValueError("Retry counts cannot be negative.")
    if args.retry_backoff < 0:
        raise ValueError("--retry-backoff cannot be negative.")
    if args.max_tokens is not None and args.max_tokens < 1:
        raise ValueError("--max-tokens must be positive when supplied.")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive.")
    if args.parse_only and args.overwrite:
        raise ValueError("--parse-only cannot be combined with --overwrite.")


def main() -> None:
    args = parse_args()
    validate_args(args)

    plan_path = resolve_path(args.plan_file).resolve()
    system_prompt_path = resolve_path(args.system_prompt_file).resolve()
    user_prompt_path = resolve_path(args.user_prompt_file).resolve()
    output_dir = resolve_path(args.output_dir).resolve()
    plans, plan_summary = load_plan(plan_path)
    if args.limit is not None:
        plans = plans[: args.limit]
    system_prompt = system_prompt_path.read_text(encoding="utf-8").strip()
    user_template = user_prompt_path.read_text(encoding="utf-8").strip()
    if user_template.count(INPUT_PLACEHOLDER) != 1:
        raise ValueError(
            f"User prompt must contain exactly one {INPUT_PLACEHOLDER!r}."
        )
    prompt_sha256 = llm_io.sha256_text(system_prompt + "\n\n" + user_template)

    first_plan = plans[0]
    first_batch, first_model_input = materialize_batch(
        first_plan, first_plan["batches"][0], first_plan["initial_family"]
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "selected_family_count": len(plans),
                    "planned_batch_count": sum(
                        len(plan["batches"]) for plan in plans
                    ),
                    "model": LLM_MODEL or "<missing>",
                    "stream": args.stream,
                    "workers": args.workers,
                    "prompt_sha256": prompt_sha256,
                    "plan_summary": plan_summary,
                    "first_family_id": first_plan["family_id"],
                    "first_batch_id": first_batch["batch_id"],
                    "first_id_mapping": llm_io.id_mapping_records(
                        first_batch["model_to_source_id"]
                    ),
                    "first_user_prompt": llm_io.build_user_prompt(
                        user_template,
                        first_model_input,
                        placeholder=INPUT_PLACEHOLDER,
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
            [
                ("DEEPSEEK_API_KEY", DEEPSEEK_API_KEY),
                ("DEEPSEEK_URL", DEEPSEEK_URL),
            ]
        )
    missing = [name for name, value in required if not value]
    if missing:
        raise ValueError(f"Missing .env variables: {', '.join(missing)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_responses_dir = output_dir / RAW_RESPONSES_DIRNAME
    runtime_path = output_dir / RUNTIME_FILENAME
    batch_results_path = output_dir / BATCH_RESULTS_FILENAME
    family_results_path = output_dir / FAMILY_RESULTS_FILENAME
    assignments_path = output_dir / ASSIGNMENTS_FILENAME
    summary_path = output_dir / SUMMARY_FILENAME
    raw_responses_dir.mkdir(parents=True, exist_ok=True)

    latest = llm_io.latest_runtime_records(runtime_path)
    runtime_lock = threading.Lock()

    def persist_runtime(record: dict[str, Any]) -> None:
        with runtime_lock:
            with runtime_path.open("a", encoding="utf-8", newline="\n") as file:
                llm_io.append_jsonl(file, record)
            print(
                f"family={record['family_id']} batch={record['batch_id']} "
                f"status={record['status']} elapsed={record['elapsed_seconds']}s"
            )

    if not args.parse_only:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.workers
        ) as executor:
            futures = [
                executor.submit(
                    execute_family_plan,
                    plan=plan,
                    latest=latest,
                    persist_runtime=persist_runtime,
                    system_prompt=system_prompt,
                    user_template=user_template,
                    prompt_sha256=prompt_sha256,
                    base_url=llm_io.resolve_base_url(DEEPSEEK_URL),
                    api_key=DEEPSEEK_API_KEY,
                    model=LLM_MODEL,
                    raw_responses_dir=raw_responses_dir,
                    args=args,
                )
                for plan in plans
            ]
            for future in concurrent.futures.as_completed(futures):
                progress = future.result()
                if progress["completed_batches"] < progress["batch_count"]:
                    print(
                        f"family={progress['family_id']} stopped after "
                        f"{progress['completed_batches']}/{progress['batch_count']} batches",
                        file=sys.stderr,
                    )

    complete_count, incomplete_count = reconstruct_outputs(
        plans=plans,
        runtime_path=runtime_path,
        batch_results_path=batch_results_path,
        family_results_path=family_results_path,
        assignments_path=assignments_path,
        summary_path=summary_path,
        model=LLM_MODEL,
        prompt_sha256=prompt_sha256,
        plan_path=plan_path,
        system_prompt_path=system_prompt_path,
        user_prompt_path=user_prompt_path,
        raw_responses_dir=raw_responses_dir,
    )
    print(
        f"Done. complete_families={complete_count}, "
        f"incomplete_families={incomplete_count}, output_dir={output_dir}"
    )


if __name__ == "__main__":
    main()

