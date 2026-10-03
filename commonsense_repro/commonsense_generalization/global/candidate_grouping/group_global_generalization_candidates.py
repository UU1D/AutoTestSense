"""Group globally retrieved commonsense generalization record bases with an OpenAI-compatible LLM."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
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

load_dotenv(PROJECT_ROOT / ".env")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_URL = os.getenv("DEEPSEEK_URL", "").strip()
LLM_MODEL = os.getenv("DEEPSEEK_V4_FLASH", "").strip()

DEFAULT_BATCHES_FILE = Path(
    "work/stage4/"
    "candidate_graph/top3_mutual_s070/llm_batches.jsonl"
)
DEFAULT_SYSTEM_PROMPT_FILE = Path(
    "prompts/group_global_generalization_candidates_v1.0_system.md"
)
DEFAULT_USER_PROMPT_FILE = Path(
    "prompts/group_global_generalization_candidates_v1.0_user.md"
)
DEFAULT_OUTPUT_DIR = Path(
    "work/stage4/"
    "candidate_grouping/deepseek_flash_v1_0"
)

DEFAULT_WORKERS = 5
DEFAULT_MAX_TOKENS: int | None = None
DEFAULT_TIMEOUT = 1200
DEFAULT_MAX_RETRIES = 3
DEFAULT_OUTPUT_RETRIES = 2
DEFAULT_RETRY_BACKOFF = 2.0

UNITS_PLACEHOLDER = "{{units_json}}"
RUNTIME_FILENAME = "grouping_runtime.jsonl"
RESULTS_FILENAME = "grouping_results.jsonl"
SUMMARY_FILENAME = "grouping_summary.json"
RAW_RESPONSES_DIRNAME = "raw_responses"


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_batches(path: Path) -> list[dict[str, Any]]:
    batches = llm_io.read_jsonl(path)
    seen_batch_ids: set[str] = set()
    seen_family_ids: set[str] = set()
    normalized: list[dict[str, Any]] = []

    for order, row in enumerate(batches):
        location = f"{path}:{order + 1}"
        batch_id = llm_io.require_nonempty_string(
            row.get("batch_id"), f"{location}.batch_id"
        )
        if batch_id in seen_batch_ids:
            raise ValueError(f"Duplicate batch_id: {batch_id}")
        units = row.get("units")
        if not isinstance(units, list) or not 2 <= len(units) <= 25:
            raise ValueError(f"{batch_id} must contain 2 to 25 units.")
        if row.get("unit_count") != len(units):
            raise ValueError(f"{batch_id} has an inconsistent unit_count.")

        normalized_units: list[dict[str, str]] = []
        local_ids: set[str] = set()
        for unit_index, unit in enumerate(units):
            if not isinstance(unit, dict):
                raise ValueError(f"{batch_id}.units[{unit_index}] must be an object.")
            llm_io.require_exact_keys(
                unit,
                {"id", "situation", "commonsense_rule"},
                f"{batch_id}.units[{unit_index}]",
            )
            family_id = llm_io.require_nonempty_string(
                unit["id"], f"{batch_id}.units[{unit_index}].id"
            )
            if family_id in local_ids:
                raise ValueError(f"Duplicate family ID inside {batch_id}: {family_id}")
            if family_id in seen_family_ids:
                raise ValueError(f"Family {family_id} appears in multiple batches.")
            local_ids.add(family_id)
            seen_family_ids.add(family_id)
            normalized_units.append(
                {
                    "id": family_id,
                    "situation": llm_io.require_nonempty_string(
                        unit["situation"], f"{family_id}.situation"
                    ),
                    "commonsense_rule": llm_io.require_nonempty_string(
                        unit["commonsense_rule"], f"{family_id}.commonsense_rule"
                    ),
                }
            )

        component_signature = llm_io.require_nonempty_string(
            row.get("component_signature"), f"{location}.component_signature"
        )
        input_sha256 = llm_io.sha256_text(
            json.dumps(normalized_units, ensure_ascii=False, sort_keys=True)
        )
        normalized.append(
            {
                "order": order,
                "batch_id": batch_id,
                "component_signature": component_signature,
                "units": normalized_units,
                "input_sha256": input_sha256,
            }
        )
        seen_batch_ids.add(batch_id)
    return normalized


def group_signature(member_ids: list[str]) -> str:
    return hashlib.sha256("\n".join(member_ids).encode("utf-8")).hexdigest()[:16]


def parse_grouping_output(
    raw_content: str,
    *,
    batch_id: str,
    input_ids: list[str],
) -> dict[str, Any]:
    text = llm_io.strip_json_fence(raw_content)
    if not text:
        raise ValueError("Assistant content is empty.")
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("Output must be a JSON object.")
    llm_io.require_exact_keys(parsed, {"groups"}, "output")
    groups = parsed["groups"]
    if not isinstance(groups, list):
        raise ValueError("groups must be an array.")

    input_id_set = set(input_ids)
    used_ids: set[str] = set()
    normalized_groups: list[dict[str, Any]] = []
    for index, group in enumerate(groups):
        location = f"groups[{index}]"
        if not isinstance(group, dict):
            raise ValueError(f"{location} must be an object.")
        llm_io.require_exact_keys(
            group, {"member_ids", "grouping_rationale"}, location
        )
        member_ids = group["member_ids"]
        if not isinstance(member_ids, list):
            raise ValueError(f"{location}.member_ids must be an array.")
        member_ids = llm_io.validate_disjoint_known_ids(
            member_ids,
            allowed_ids=input_id_set,
            used_ids=used_ids,
            location=f"{location}.member_ids",
            min_size=2,
        )
        normalized_groups.append(
            {
                "member_ids": member_ids,
                "grouping_rationale": llm_io.require_nonempty_string(
                    group["grouping_rationale"], f"{location}.grouping_rationale"
                ),
            }
        )

    normalized_groups.sort(key=lambda item: llm_io.natural_sort_key(item["member_ids"][0]))
    singleton_ids = llm_io.infer_singleton_ids(input_ids, used_ids)
    return {
        "batch_id": batch_id,
        "groups": normalized_groups,
        "singleton_ids": singleton_ids,
    }


def read_raw_grouping(
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
        raise ValueError(f"Raw response metadata is invalid: {raw_path}")
    expected = {
        "batch_id": batch["batch_id"],
        "model": model,
        "prompt_sha256": prompt_sha256,
        "input_sha256": batch["input_sha256"],
    }
    for field, value in expected.items():
        if metadata.get(field) != value:
            raise ValueError(f"Raw response {field} does not match: {raw_path}")

    expected_mapping = llm_io.build_model_id_mapping(batch["units"])
    saved_mapping = raw_record.get("id_mapping")
    model_to_source = llm_io.parse_id_mapping_records(
        saved_mapping,
        expected_mapping=expected_mapping,
    )

    assistant_output = raw_record.get("assistant_output")
    if assistant_output is not None:
        assistant_content = json.dumps(assistant_output, ensure_ascii=False)
    else:
        assistant_content = raw_record.get("assistant_raw_text")
    if not isinstance(assistant_content, str):
        raise ValueError(f"Raw response contains no assistant output: {raw_path}")

    normalized = parse_grouping_output(
        assistant_content,
        batch_id=batch["batch_id"],
        input_ids=list(model_to_source),
    )
    restored_groups: list[dict[str, Any]] = []
    for index, group in enumerate(normalized["groups"], start=1):
        source_ids = sorted(
            (model_to_source[model_id] for model_id in group["member_ids"]),
            key=llm_io.natural_sort_key,
        )
        restored_groups.append(
            {
                "group_id": f"{batch['batch_id']}.G{index:02d}",
                "group_signature": group_signature(source_ids),
                "member_family_ids": source_ids,
                "grouping_rationale": group["grouping_rationale"],
            }
        )
    singleton_ids = sorted(
        (model_to_source[model_id] for model_id in normalized["singleton_ids"]),
        key=llm_io.natural_sort_key,
    )
    return {
        "groups": restored_groups,
        "singleton_family_ids": singleton_ids,
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
        "component_signature": batch["component_signature"],
        "input_size": len(batch["units"]),
        "model": model,
        "stream": args.stream,
        "prompt_sha256": prompt_sha256,
        "input_sha256": batch["input_sha256"],
    }
    mapping = llm_io.build_model_id_mapping(batch["units"])
    model_units = llm_io.build_model_units(batch["units"], mapping)
    user_prompt = llm_io.build_user_prompt(user_template, model_units)
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
                read_raw_grouping(
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
    normalized = read_raw_grouping(
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
    batches_path: Path,
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
            results.append(normalize_runtime_record(record, batch, model, prompt_sha256))
        except (OSError, json.JSONDecodeError, ValueError) as error:
            print(f"batch={batch['batch_id']} cannot parse raw response: {error}", file=sys.stderr)
    results.sort(key=lambda item: item["order"])
    llm_io.write_jsonl(results_path, results)

    usage: dict[str, int] = {}
    for result in results:
        llm_io.add_usage(usage, result.get("usage", {}))
    groups = [group for result in results for group in result["groups"]]
    grouped_ids = {
        family_id for group in groups for family_id in group["member_family_ids"]
    }
    singletons = [
        family_id
        for result in results
        for family_id in result["singleton_family_ids"]
    ]
    summary = {
        "batches_file": str(batches_path),
        "system_prompt_file": str(system_prompt_path),
        "user_prompt_file": str(user_prompt_path),
        "model": model,
        "prompt_sha256": prompt_sha256,
        "input_batch_count": len(batches),
        "success_count": len(results),
        "incomplete_count": len(batches) - len(results),
        "group_count": len(groups),
        "grouped_family_count": len(grouped_ids),
        "singleton_family_count": len(singletons),
        "group_size_distribution": {
            str(size): sum(len(group["member_family_ids"]) == size for group in groups)
            for size in sorted({len(group["member_family_ids"]) for group in groups})
        },
        "usage": usage,
        "runtime_output": str(runtime_path),
        "results_output": str(results_path),
        "raw_responses_dir": str(raw_responses_dir),
    }
    llm_io.write_json(summary_path, summary)
    return len(results), len(batches) - len(results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches-file", type=Path, default=DEFAULT_BATCHES_FILE)
    parser.add_argument("--system-prompt-file", type=Path, default=DEFAULT_SYSTEM_PROMPT_FILE)
    parser.add_argument("--user-prompt-file", type=Path, default=DEFAULT_USER_PROMPT_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--stream", action="store_true", help="Use streaming; non-streaming is the default.")
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
    if args.workers < 1 or args.timeout < 1 or args.max_retries < 0 or args.output_retries < 0:
        raise ValueError("Invalid worker, timeout, or retry configuration.")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive.")
    if args.parse_only and args.overwrite:
        raise ValueError("--parse-only cannot be combined with --overwrite.")

    batches_path = resolve_path(args.batches_file).resolve()
    system_prompt_path = resolve_path(args.system_prompt_file).resolve()
    user_prompt_path = resolve_path(args.user_prompt_file).resolve()
    output_dir = resolve_path(args.output_dir).resolve()
    batches = load_batches(batches_path)
    if args.limit is not None:
        batches = batches[: args.limit]
    system_prompt = system_prompt_path.read_text(encoding="utf-8").strip()
    user_template = user_prompt_path.read_text(encoding="utf-8").strip()
    if user_template.count(UNITS_PLACEHOLDER) != 1:
        raise ValueError(f"User prompt must contain exactly one {UNITS_PLACEHOLDER!r}.")
    prompt_sha256 = llm_io.sha256_text(system_prompt + "\n\n" + user_template)

    first_mapping = llm_io.build_model_id_mapping(batches[0]["units"])
    first_units = llm_io.build_model_units(batches[0]["units"], first_mapping)
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
                    "first_id_mapping": first_mapping,
                    "first_user_prompt": llm_io.build_user_prompt(
                        user_template, first_units
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
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
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
        model=LLM_MODEL,
        prompt_sha256=prompt_sha256,
        batches_path=batches_path,
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

