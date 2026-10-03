"""Generalize final commonsense generalization records from stage-1 global candidate groups."""

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
from commonsense_repro.commonsense_generalization.local.generalize_local_commonsense import (
    load_and_normalize_raw_response,
    load_source_units,
)

load_dotenv(PROJECT_ROOT / ".env")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_URL = os.getenv("DEEPSEEK_URL", "").strip()
LLM_MODEL = os.getenv("DEEPSEEK_V4_FLASH", "").strip()

DATASET_ROOT = Path(
    "work/stage4"
)
DEFAULT_STAGE1_RESULTS_FILE = DATASET_ROOT / (
    "candidate_grouping/deepseek_flash_v1_0/grouping_results.jsonl"
)
DEFAULT_CATALOG_FILE = Path(
    "work/stage3/"
    "gemini_v1_2_non_mutual_s15_leiden_max25_deepseek_flash_v1_4/"
    "commonsense_generalization_records.json"
)
DEFAULT_SOURCE_FILES = (
    Path("data/checkpoints/instance_level_commonsense/gemini_v1_2_extract.json"),
    Path("data/checkpoints/instance_level_commonsense/gemini_v1_2_github.json"),
    Path("data/checkpoints/instance_level_commonsense/gemini_v1_2_negative.json"),
)
DEFAULT_SYSTEM_PROMPT_FILE = Path(
    "prompts/generalize_global_commonsense_v1.2_system.md"
)
DEFAULT_USER_PROMPT_FILE = Path(
    "prompts/generalize_global_commonsense_v1.2_user.md"
)
DEFAULT_OUTPUT_DIR = DATASET_ROOT / "generalization/deepseek_flash_v1_2"

DEFAULT_WORKERS = 5
DEFAULT_MAX_TOKENS: int | None = None
DEFAULT_TIMEOUT = 1200
DEFAULT_MAX_RETRIES = 3
DEFAULT_OUTPUT_RETRIES = 2
DEFAULT_RETRY_BACKOFF = 2.0

INPUT_PLACEHOLDER = "{{input_json}}"
RUNTIME_FILENAME = "final_generalization_runtime.jsonl"
RESULTS_FILENAME = "final_generalization_results.jsonl"
SUMMARY_FILENAME = "final_generalization_summary.json"
ASSIGNMENTS_FILENAME = "source_unit_assignments.jsonl"
RAW_RESPONSES_DIRNAME = "raw_responses"


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_catalog(path: Path) -> dict[str, dict[str, Any]]:
    root = llm_io.read_json(path)
    if not isinstance(root, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    families = root.get("rule_families")
    if not isinstance(families, list):
        raise ValueError(f"{path}.rule_families must be an array.")

    catalog: dict[str, dict[str, Any]] = {}
    seen_source_ids: set[str] = set()
    for index, family in enumerate(families):
        location = f"{path}.rule_families[{index}]"
        if not isinstance(family, dict):
            raise ValueError(f"{location} must be an object.")
        family_id = llm_io.require_nonempty_string(
            family.get("family_id"), f"{location}.family_id"
        )
        if family_id in catalog:
            raise ValueError(f"Duplicate catalog family ID: {family_id}")

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

        member_ids = family.get("member_ids")
        if not isinstance(member_ids, list):
            raise ValueError(f"{location}.member_ids must be an array.")
        normalized_members = llm_io.normalize_string_array(
            member_ids, f"{location}.member_ids"
        )
        if not normalized_members:
            raise ValueError(f"{location}.member_ids cannot be empty.")
        overlap = set(normalized_members) & seen_source_ids
        if overlap:
            duplicate = sorted(overlap, key=llm_io.natural_sort_key)[0]
            raise ValueError(f"Source unit {duplicate} appears in multiple families.")
        member_count = family.get("member_count")
        if member_count != len(normalized_members):
            raise ValueError(f"{location} has an inconsistent member_count.")
        normalized_members.sort(key=llm_io.natural_sort_key)
        seen_source_ids.update(normalized_members)
        catalog[family_id] = {
            "family_id": family_id,
            "base_unit": normalized_base,
            "member_ids": normalized_members,
        }
    return catalog


def load_stage2_batches(
    stage1_results_path: Path,
    catalog_path: Path,
    source_paths: list[Path],
) -> list[dict[str, Any]]:
    stage1_results = llm_io.read_jsonl(stage1_results_path)
    catalog = load_catalog(catalog_path)
    source_units = load_source_units(source_paths)
    batches: list[dict[str, Any]] = []
    seen_group_ids: set[str] = set()
    seen_family_ids: set[str] = set()
    seen_source_ids: set[str] = set()

    for stage1_record in stage1_results:
        component_id = llm_io.require_nonempty_string(
            stage1_record.get("batch_id"), "stage1.batch_id"
        )
        groups = stage1_record.get("groups")
        if not isinstance(groups, list):
            raise ValueError(f"{component_id}.groups must be an array.")

        for group in groups:
            if not isinstance(group, dict):
                raise ValueError(f"{component_id}.groups contains a non-object.")
            group_id = llm_io.require_nonempty_string(
                group.get("group_id"), f"{component_id}.group_id"
            )
            if group_id in seen_group_ids:
                raise ValueError(f"Duplicate stage-1 group ID: {group_id}")
            family_ids = group.get("member_family_ids")
            if not isinstance(family_ids, list):
                raise ValueError(f"{group_id}.member_family_ids must be an array.")
            family_ids = llm_io.normalize_string_array(
                family_ids, f"{group_id}.member_family_ids"
            )
            if len(family_ids) < 2:
                raise ValueError(f"{group_id} must contain at least two families.")
            repeated_families = set(family_ids) & seen_family_ids
            if repeated_families:
                duplicate = sorted(
                    repeated_families, key=llm_io.natural_sort_key
                )[0]
                raise ValueError(
                    f"Catalog family {duplicate} appears in multiple stage-1 groups."
                )

            candidate_families: list[dict[str, Any]] = []
            expanded_units: list[dict[str, Any]] = []
            for family_id in family_ids:
                family = catalog.get(family_id)
                if family is None:
                    raise ValueError(
                        f"Stage-1 group {group_id} references unknown family {family_id}."
                    )
                candidate_families.append(
                    {
                        "id": family_id,
                        "base_unit": family["base_unit"],
                    }
                )
                for source_id in family["member_ids"]:
                    if source_id in seen_source_ids:
                        raise ValueError(
                            f"Source unit {source_id} appears in multiple stage-2 batches."
                        )
                    source = source_units.get(source_id)
                    if source is None:
                        raise ValueError(f"No source data found for unit {source_id}.")
                    expanded_units.append(
                        {
                            "id": source_id,
                            "prior_family_id": family_id,
                            "situation": source["situation"],
                            "commonsense_rule": source[
                                "violated_commonsense_rule"
                            ],
                            "applicability_conditions": source[
                                "applicability_conditions"
                            ],
                        }
                    )
                    seen_source_ids.add(source_id)

            input_value = {
                "candidate_families": candidate_families,
                "source_units": expanded_units,
            }
            batches.append(
                {
                    "order": len(batches),
                    "batch_id": group_id,
                    "stage1_component_id": component_id,
                    "stage1_group_signature": llm_io.require_nonempty_string(
                        group.get("group_signature"),
                        f"{group_id}.group_signature",
                    ),
                    "candidate_families": candidate_families,
                    "units": expanded_units,
                    "input_sha256": llm_io.sha256_text(
                        json.dumps(
                            input_value,
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                    ),
                }
            )
            seen_group_ids.add(group_id)
            seen_family_ids.update(family_ids)

    if not batches:
        raise ValueError("No stage-1 groups were found for final generalization.")
    return batches


def build_model_input(
    batch: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str], dict[str, str]]:
    family_mapping = {
        f"F{index}": str(family["id"])
        for index, family in enumerate(batch["candidate_families"])
    }
    source_mapping = llm_io.build_model_id_mapping(batch["units"])
    family_to_model = llm_io.invert_id_mapping(family_mapping)
    source_to_model = llm_io.invert_id_mapping(source_mapping)

    candidate_families = [
        {
            "id": family_to_model[str(family["id"])],
            "base_unit": family["base_unit"],
        }
        for family in batch["candidate_families"]
    ]
    model_units = [
        {
            "id": source_to_model[str(unit["id"])],
            "prior_family_id": family_to_model[str(unit["prior_family_id"])],
            "situation": unit["situation"],
            "commonsense_rule": unit["commonsense_rule"],
            "applicability_conditions": unit["applicability_conditions"],
        }
        for unit in batch["units"]
    ]
    return (
        {
            "candidate_families": candidate_families,
            "source_units": model_units,
        },
        family_mapping,
        source_mapping,
    )


def read_raw_generalization(
    *,
    raw_path: Path,
    batch: dict[str, Any],
    model: str,
    prompt_sha256: str,
) -> dict[str, Any]:
    return load_and_normalize_raw_response(
        raw_path=raw_path,
        batch=batch,
        model=model,
        prompt_sha256=prompt_sha256,
    )


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
    model_input, family_mapping, _ = build_model_input(batch)
    user_prompt = llm_io.build_user_prompt(
        user_template,
        model_input,
        placeholder=INPUT_PLACEHOLDER,
    )
    base_record = {
        "order": batch["order"],
        "batch_id": batch["batch_id"],
        "stage1_component_id": batch["stage1_component_id"],
        "stage1_group_signature": batch["stage1_group_signature"],
        "candidate_family_count": len(batch["candidate_families"]),
        "candidate_family_ids": [
            family["id"] for family in batch["candidate_families"]
        ],
        "candidate_family_id_mapping": llm_io.id_mapping_records(
            family_mapping
        ),
        "input_size": len(batch["units"]),
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
                    "\n\nThe previous response failed JSON validation with this error: "
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
                read_raw_generalization(
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
            "error_type": type(error).__name__,
            "error": str(error),
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
    return {**record, "status": "success", **normalized}


def family_member_ids(family: dict[str, Any]) -> list[str]:
    ids = list(family["equivalent_member_ids"])
    for group in family["variant_groups"]:
        ids.extend(group["member_ids"])
    return sorted(ids, key=llm_io.natural_sort_key)


def build_outputs(
    *,
    batches: list[dict[str, Any]],
    runtime_path: Path,
    results_path: Path,
    summary_path: Path,
    assignments_path: Path,
    model: str,
    prompt_sha256: str,
    stage1_results_path: Path,
    catalog_path: Path,
    source_paths: list[Path],
    system_prompt_path: Path,
    user_prompt_path: Path,
    raw_responses_dir: Path,
) -> tuple[int, int]:
    latest = llm_io.latest_runtime_records(runtime_path)
    results: list[dict[str, Any]] = []
    batch_by_id = {batch["batch_id"]: batch for batch in batches}
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
                f"batch={batch['batch_id']} cannot parse raw response: {error}",
                file=sys.stderr,
            )
    results.sort(key=lambda item: item["order"])
    llm_io.write_jsonl(results_path, results)

    assignments: list[dict[str, Any]] = []
    final_families: list[dict[str, Any]] = []
    for result in results:
        batch = batch_by_id[result["batch_id"]]
        prior_by_source = {
            unit["id"]: unit["prior_family_id"] for unit in batch["units"]
        }
        assigned_ids: set[str] = set()
        for family in result["rule_families"]:
            final_families.append(family)
            for source_id in family_member_ids(family):
                assigned_ids.add(source_id)
                assignments.append(
                    {
                        "source_unit_id": source_id,
                        "prior_family_id": prior_by_source[source_id],
                        "stage1_group_id": result["batch_id"],
                        "status": "final_family",
                        "final_family_id": family["family_id"],
                    }
                )
        for singleton in result["singleton_units"]:
            source_id = singleton["id"]
            assignments.append(
                {
                    "source_unit_id": source_id,
                    "prior_family_id": prior_by_source[source_id],
                    "stage1_group_id": result["batch_id"],
                    "status": "singleton",
                    "final_family_id": None,
                }
            )
        if len(assigned_ids) + len(result["singleton_units"]) != len(
            batch["units"]
        ):
            raise ValueError(
                f"Batch {result['batch_id']} does not partition all source units."
            )

    assignments.sort(
        key=lambda item: llm_io.natural_sort_key(item["source_unit_id"])
    )
    llm_io.write_jsonl(assignments_path, assignments)

    usage: dict[str, int] = {}
    for result in results:
        llm_io.add_usage(usage, result.get("usage", {}))
    grouped_source_count = sum(
        len(family_member_ids(family)) for family in final_families
    )
    singleton_count = sum(
        len(result["singleton_units"]) for result in results
    )
    family_sizes = [len(family_member_ids(family)) for family in final_families]
    summary = {
        "stage1_results_file": str(stage1_results_path),
        "catalog_file": str(catalog_path),
        "source_files": [str(path) for path in source_paths],
        "system_prompt_file": str(system_prompt_path),
        "user_prompt_file": str(user_prompt_path),
        "model": model,
        "prompt_sha256": prompt_sha256,
        "input_batch_count": len(batches),
        "success_count": len(results),
        "incomplete_count": len(batches) - len(results),
        "candidate_family_count": sum(
            len(batch["candidate_families"]) for batch in batches
        ),
        "input_source_unit_count": sum(len(batch["units"]) for batch in batches),
        "final_family_count": len(final_families),
        "grouped_source_unit_count": grouped_source_count,
        "singleton_source_unit_count": singleton_count,
        "final_family_size_distribution": {
            str(size): sum(value == size for value in family_sizes)
            for size in sorted(set(family_sizes))
        },
        "usage": usage,
        "runtime_output": str(runtime_path),
        "results_output": str(results_path),
        "assignments_output": str(assignments_path),
        "raw_responses_dir": str(raw_responses_dir),
    }
    llm_io.write_json(summary_path, summary)
    return len(results), len(batches) - len(results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage1-results-file", type=Path, default=DEFAULT_STAGE1_RESULTS_FILE
    )
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
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Use streaming; non-streaming is the default.",
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

    stage1_results_path = resolve_path(args.stage1_results_file).resolve()
    catalog_path = resolve_path(args.catalog_file).resolve()
    source_paths = [resolve_path(path).resolve() for path in args.source_files]
    system_prompt_path = resolve_path(args.system_prompt_file).resolve()
    user_prompt_path = resolve_path(args.user_prompt_file).resolve()
    output_dir = resolve_path(args.output_dir).resolve()

    batches = load_stage2_batches(
        stage1_results_path,
        catalog_path,
        source_paths,
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

    first_input, first_family_mapping, first_source_mapping = build_model_input(
        batches[0]
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "batch_count": len(batches),
                    "model": LLM_MODEL or "<missing>",
                    "stream": args.stream,
                    "workers": args.workers,
                    "prompt_sha256": prompt_sha256,
                    "first_batch_id": batches[0]["batch_id"],
                    "first_candidate_family_id_mapping": first_family_mapping,
                    "first_source_unit_id_mapping": first_source_mapping,
                    "first_user_prompt": llm_io.build_user_prompt(
                        user_template,
                        first_input,
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
    results_path = output_dir / RESULTS_FILENAME
    summary_path = output_dir / SUMMARY_FILENAME
    assignments_path = output_dir / ASSIGNMENTS_FILENAME
    raw_responses_dir.mkdir(parents=True, exist_ok=True)

    latest = llm_io.latest_runtime_records(runtime_path)
    pending: list[dict[str, Any]] = []
    for batch in batches:
        key = (batch["batch_id"], LLM_MODEL, prompt_sha256, batch["input_sha256"])
        record = latest.get(key)
        if not args.overwrite and record is not None:
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
                        f"batch={record['batch_id']} status={record['status']} "
                        f"elapsed={record['elapsed_seconds']}s"
                    )

    success_count, incomplete_count = build_outputs(
        batches=batches,
        runtime_path=runtime_path,
        results_path=results_path,
        summary_path=summary_path,
        assignments_path=assignments_path,
        model=LLM_MODEL,
        prompt_sha256=prompt_sha256,
        stage1_results_path=stage1_results_path,
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

