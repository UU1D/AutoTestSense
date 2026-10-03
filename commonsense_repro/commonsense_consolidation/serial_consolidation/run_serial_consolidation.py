"""Serially consolidate previously unmatched units into an evolving commonsense generalization record catalog."""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv


from commonsense_repro.common.paths import PACKAGE_ROOT as PROJECT_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from commonsense_repro.clustering.build_instance_level_commonsense_embeddings import (  # noqa: E402
    extract_embeddings,
    request_embeddings,
    resolve_endpoint,
)
from commonsense_repro.common import llm_pipeline_common as llm_io  # noqa: E402
_rerank_api = importlib.import_module(
    "commonsense_repro.commonsense_generalization.global.rerank_generalized_rule_neighbors"
)
RERANK_INSTRUCT = _rerank_api.RERANK_INSTRUCT
RERANK_MODEL = _rerank_api.RERANK_MODEL
call_rerank = _rerank_api.call_rerank
parse_rerank_results = _rerank_api.parse_rerank_results
validate_rerank_config = _rerank_api.validate_api_config
from commonsense_repro.commonsense_consolidation.generalization_embedding_index.generalization_embedding_index import (  # noqa: E402
    catalog_bases,
)
from commonsense_repro.commonsense_consolidation.generalization_embedding_index.sync_generalized_rule_embeddings import (  # noqa: E402
    indexed_events,
    manifest_active_records,
)
from commonsense_repro.commonsense_consolidation.consolidation_judgment.judge_consolidation_membership import (  # noqa: E402
    DECISION_ADD,
    DECISION_NO_MATCH,
    build_user_prompt as build_membership_user_prompt,
    format_judgment_input,
    parse_judgment_output,
    select_reranked_candidates,
)
from commonsense_repro.commonsense_consolidation.recall_generalization_records import (  # noqa: E402
    build_recall_records,
    load_source_embeddings,
    validate_embedding_configuration,
)
from commonsense_repro.commonsense_consolidation.rerank_consolidation_candidates import (  # noqa: E402
    build_rerank_record,
    candidate_document,
)
from commonsense_repro.commonsense_consolidation.serial_consolidation import (  # noqa: E402
    core,
)


load_dotenv(PROJECT_ROOT / ".env")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_URL = os.getenv("DEEPSEEK_URL", "").strip()
LLM_MODEL = os.getenv("DEEPSEEK_V4_FLASH", "").strip()
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "").strip()
DASHSCOPE_URL = os.getenv("DASHSCOPE_URL", "").strip()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "").strip()

DATA_ROOT = Path(
    "work/stage4/"
    "commonsense_consolidation"
)
DEFAULT_INITIAL_CATALOG = DATA_ROOT / "input_preparation/initial_commonsense_generalization_records.json"
DEFAULT_QUEUE_FILE = DATA_ROOT / "input_preparation/no_match_units.jsonl"
DEFAULT_INITIAL_EMBEDDING_DIR = Path(
    "work/stage4/"
    "serial_family_embeddings"
)
DEFAULT_SOURCE_FILES = (
    Path("data/checkpoints/instance_level_commonsense/gemini_v1_2_extract.json"),
    Path("data/checkpoints/instance_level_commonsense/gemini_v1_2_github.json"),
    Path("data/checkpoints/instance_level_commonsense/gemini_v1_2_negative.json"),
)
DEFAULT_SOURCE_EMBEDDINGS = (
    Path("work/stage2/embeddings/gemini_v1_2_extract_embeddings.jsonl"),
    Path("work/stage2/embeddings/gemini_v1_2_github_embeddings.jsonl"),
    Path("work/stage2/embeddings/gemini_v1_2_negative_embeddings.jsonl"),
)
DEFAULT_MEMBERSHIP_SYSTEM_PROMPT = Path(
    "prompts/judge_unassigned_commonsense_consolidation_v1.2_system.md"
)
DEFAULT_MEMBERSHIP_USER_PROMPT = Path(
    "prompts/judge_unassigned_commonsense_consolidation_v1.2_user.md"
)
DEFAULT_GENERALIZATION_SYSTEM_PROMPT = Path(
    "prompts/consolidate_unassigned_commonsense_v1.0_system.md"
)
DEFAULT_GENERALIZATION_USER_PROMPT = Path(
    "prompts/consolidate_unassigned_commonsense_v1.0_user.md"
)
DEFAULT_OUTPUT_DIR = DATA_ROOT / "serial_consolidation/deepseek_flash_v1_0"

DEFAULT_RECALL_K = 20
DEFAULT_FINAL_K = 10
DEFAULT_MIN_RERANK_SCORE = 0.7
DEFAULT_MIN_CANDIDATES = 3
DEFAULT_CHUNK_SIZE = 256
DEFAULT_TIMEOUT = 1200
DEFAULT_EMBEDDING_TIMEOUT = 120
DEFAULT_MAX_RETRIES = 3
DEFAULT_OUTPUT_RETRIES = 2
DEFAULT_RETRY_BACKOFF = 2.0
DEFAULT_MAX_TOKENS: int | None = None

STATE_FILENAME = "serial_state.json"
CATALOG_FILENAME = "current_commonsense_generalization_records.json"
ACTIVE_BASES_FILENAME = "active_generalized_rule_embeddings.jsonl"
BASE_EVENTS_FILENAME = "base_embedding_updates.jsonl"
RESULTS_FILENAME = "unit_consolidation_results.jsonl"
RUNTIME_FILENAME = "unit_consolidation_runtime.jsonl"
SUMMARY_FILENAME = "input_preparation_summary.json"
RAW_RESPONSES_DIRNAME = "raw_responses"
UNIT_RESULTS_DIRNAME = "unit_results"
GENERALIZATION_PLACEHOLDER = "{{input_json}}"


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as file:
        json.dump(row, file, ensure_ascii=False)
        file.write("\n")
        file.flush()


def unit_result_path(output_dir: Path, serial_order: int, unit_id: str) -> Path:
    return (
        output_dir
        / UNIT_RESULTS_DIRNAME
        / f"{serial_order:04d}_{llm_io.safe_path_component(unit_id)}.json"
    )


def rebuild_results_index(
    output_dir: Path, queue: list[dict[str, Any]], committed_count: int
) -> None:
    rows: list[dict[str, Any]] = []
    for item in queue[:committed_count]:
        path = unit_result_path(output_dir, item["serial_order"], item["id"])
        if not path.is_file():
            raise ValueError(f"Committed state lacks per-unit result: {path}")
        value = llm_io.read_json(path)
        if not isinstance(value, dict):
            raise ValueError(f"Invalid per-unit result: {path}")
        rows.append(value)
    llm_io.write_jsonl(output_dir / RESULTS_FILENAME, rows)


def load_initial_active_embeddings(directory: Path) -> dict[str, dict[str, Any]]:
    events_path = directory / "family_base_embedding_events.jsonl"
    manifest_path = directory / "family_embedding_manifest.json"
    simple_path = directory / "generalized_rule_embeddings.jsonl"
    if simple_path.is_file() and not (events_path.is_file() and manifest_path.is_file()):
        rows = llm_io.read_jsonl(simple_path)
        return {
            row["family_id"]: row
            for row in rows
            if isinstance(row, dict) and isinstance(row.get("family_id"), str)
        }
    rows = llm_io.read_jsonl(events_path)
    manifest = llm_io.read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError(f"Invalid embedding manifest: {manifest_path}")
    return manifest_active_records(manifest, indexed_events(rows))


def load_update_events(path: Path) -> list[dict[str, Any]]:
    return llm_io.read_jsonl(path) if path.exists() else []


def active_base_embeddings(
    catalog: dict[str, Any],
    initial: dict[str, dict[str, Any]],
    updates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    bases, _ = catalog_bases(catalog)
    updates_by_family: dict[str, list[dict[str, Any]]] = {}
    for event in updates:
        family_id = event.get("family_id")
        if isinstance(family_id, str):
            updates_by_family.setdefault(family_id, []).append(event)
    records: list[dict[str, Any]] = []
    for base in bases:
        family_id = base["family_id"]
        candidates = [
            *reversed(updates_by_family.get(family_id, [])),
            initial.get(family_id),
        ]
        record = next(
            (
                item
                for item in candidates
                if isinstance(item, dict)
                and item.get("base_unit_sha256") == base["base_unit_sha256"]
            ),
            None,
        )
        if record is None:
            raise ValueError(f"No active base embedding matches family {family_id}.")
        records.append(
            {
                "family_id": family_id,
                "model": record["model"],
                "dimensions": record["dimensions"],
                "embedding": record["embedding"],
            }
        )
    return records


def raw_assistant_content(path: Path) -> str:
    record = llm_io.read_json(path)
    if not isinstance(record, dict):
        raise ValueError(f"Raw response is not an object: {path}")
    value = record.get("assistant_output")
    if value is not None:
        return json.dumps(value, ensure_ascii=False)
    value = record.get("assistant_raw_text")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Raw response contains no assistant content: {path}")
    return value


def invoke_json_llm(
    *,
    batch: dict[str, Any],
    system_prompt: str,
    user_prompt: str,
    prompt_sha256: str,
    parser: Callable[[str], dict[str, Any]],
    raw_dir: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    usage: dict[str, int] = {}
    request_ids: list[str] = []
    last_error = ""
    last_raw_path: Path | None = None
    for output_attempt in range(args.output_retries + 1):
        active_prompt = user_prompt
        if last_error:
            active_prompt += (
                "\n\nYour previous response was invalid: "
                f"{last_error}\nReturn one corrected complete JSON object."
            )
        payload: dict[str, Any] = {
            "model": LLM_MODEL,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": active_prompt},
            ],
        }
        if args.max_tokens is not None:
            payload["max_tokens"] = args.max_tokens
        raw_path, metadata = llm_io.request_with_retries(
            stream=args.stream,
            base_url=llm_io.resolve_base_url(DEEPSEEK_URL),
            api_key=DEEPSEEK_API_KEY,
            payload=payload,
            timeout=args.timeout,
            max_retries=args.max_retries,
            retry_backoff=args.retry_backoff,
            raw_responses_dir=raw_dir,
            batch=batch,
            model=LLM_MODEL,
            prompt_sha256=prompt_sha256,
            output_attempt=output_attempt,
        )
        last_raw_path = raw_path
        llm_io.add_usage(usage, metadata.get("usage", {}))
        if isinstance(metadata.get("request_id"), str):
            request_ids.append(metadata["request_id"])
        try:
            parsed = parser(raw_assistant_content(raw_path))
            return parsed, {
                "raw_response_file": str(raw_path),
                "output_attempts": output_attempt + 1,
                "request_ids": request_ids,
                "usage": usage,
            }
        except (json.JSONDecodeError, ValueError) as error:
            last_error = str(error)
    raise ValueError(
        f"LLM output remained invalid after {args.output_retries + 1} attempts: "
        f"{last_error}; raw={last_raw_path}"
    )


def rerank_candidates(
    *,
    unit_id: str,
    recalled: list[dict[str, Any]],
    catalog: dict[str, Any],
    source_embeddings: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    simple_catalog = {
        item["family_id"]: item["base_unit"] for item in catalog["rule_families"]
    }
    documents = [
        candidate_document(item, catalog=simple_catalog, source=source_embeddings)
        for item in recalled
    ]
    response = call_rerank(
        query=source_embeddings[unit_id]["embedding_text"],
        documents=documents,
        timeout=args.rerank_timeout,
        max_retries=args.max_retries,
    )
    record = build_rerank_record(
        unit_id=unit_id,
        recalled=recalled,
        response=response,
        query_sha256=llm_io.sha256_text(source_embeddings[unit_id]["embedding_text"]),
        candidates_sha256=llm_io.sha256_text(
            json.dumps(documents, ensure_ascii=False, sort_keys=True)
        ),
    )
    candidates = record["candidates"][: args.final_k]
    for candidate in candidates:
        candidate["contains_consolidateed_no_match_evidence"] = (
            core.candidate_has_dynamic_evidence(candidate)
        )
    return candidates, {
        "request_id": response.get("id"),
        "usage": response.get("usage", {}),
    }


def membership_call(
    *,
    unit: dict[str, Any],
    candidates: list[dict[str, Any]],
    catalog: dict[str, Any],
    source_units: dict[str, dict[str, Any]],
    system_prompt: str,
    user_template: str,
    prompt_sha256: str,
    raw_dir: Path,
    serial_order: int,
    call_index: int,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    model_candidates, mapping = core.model_membership_candidates(
        candidates, catalog=catalog, source_units=source_units
    )
    formatted = format_judgment_input(unit, model_candidates)
    batch = {
        "batch_id": f"{serial_order:04d}_{unit['id']}_membership_{call_index:02d}",
        "input_sha256": llm_io.sha256_text(formatted),
        "model_to_source_id": mapping,
        "identifier_scheme": "batch_local_c_index_v1",
    }
    parsed, metadata = invoke_json_llm(
        batch=batch,
        system_prompt=system_prompt,
        user_prompt=build_membership_user_prompt(user_template, formatted),
        prompt_sha256=prompt_sha256,
        parser=lambda text: parse_judgment_output(
            text, allowed_candidate_ids=set(mapping)
        ),
        raw_dir=raw_dir,
        args=args,
    )
    alias = parsed["selected_candidate_id"]
    return {
        **parsed,
        "selected_family_id": mapping[alias] if alias is not None else None,
        "candidate_family_ids": list(mapping.values()),
    }, metadata


def generalization_call(
    *,
    unit: dict[str, Any],
    family: dict[str, Any],
    preliminary_rationale: str,
    source_units: dict[str, dict[str, Any]],
    system_prompt: str,
    user_template: str,
    prompt_sha256: str,
    raw_dir: Path,
    serial_order: int,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    value, mapping = core.build_single_generalization_input(
        family=family,
        new_unit=unit,
        preliminary_rationale=preliminary_rationale,
        source_units=source_units,
    )
    input_json = json.dumps(value, ensure_ascii=False, indent=2)
    if user_template.count(GENERALIZATION_PLACEHOLDER) != 1:
        raise ValueError(
            f"Generalization user prompt must contain one {GENERALIZATION_PLACEHOLDER}."
        )
    user_prompt = user_template.replace(GENERALIZATION_PLACEHOLDER, input_json)
    batch = {
        "batch_id": f"{serial_order:04d}_{unit['id']}_generalization",
        "input_sha256": llm_io.sha256_text(input_json),
        "model_to_source_id": mapping,
        "identifier_scheme": "batch_local_u_index_v1",
    }
    return invoke_json_llm(
        batch=batch,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        prompt_sha256=prompt_sha256,
        parser=lambda text: core.parse_single_generalization_output(
            text,
            model_to_source=mapping,
            variant_group_count=len(family["variant_groups"]),
        ),
        raw_dir=raw_dir,
        args=args,
    )


def make_base_update_event(
    *,
    family: dict[str, Any],
    vector: list[float],
    model: str,
    dimensions: int,
    serial_order: int,
    update_type: str,
    vector_origin: str,
    source_unit_id: str | None,
) -> dict[str, Any]:
    return {
        "event_id": f"S{serial_order + 1:08d}",
        "serial_order": serial_order,
        "catalog_version": serial_order + 1,
        "update_type": update_type,
        "vector_origin": vector_origin,
        "source_unit_id": source_unit_id,
        "created_at": utc_now(),
        **core.base_record_for_family(
            family, vector, model=model, dimensions=dimensions
        ),
    }


def update_catalog_counts(catalog: dict[str, Any]) -> None:
    catalog["family_count"] = len(catalog["rule_families"])
    catalog["covered_unit_count"] = sum(
        item["member_count"] for item in catalog["rule_families"]
    )


def persist_committed_state(
    *,
    state: dict[str, Any],
    output_dir: Path,
    initial_embeddings: dict[str, dict[str, Any]],
    update_events: list[dict[str, Any]],
) -> None:
    llm_io.write_json(output_dir / STATE_FILENAME, state)
    llm_io.write_json(output_dir / CATALOG_FILENAME, state["catalog"])
    bases = active_base_embeddings(state["catalog"], initial_embeddings, update_events)
    llm_io.write_jsonl(output_dir / ACTIVE_BASES_FILENAME, bases)
    llm_io.write_json(
        output_dir / SUMMARY_FILENAME,
        {
            "updated_at": utc_now(),
            "queue_size": state["queue_size"],
            "next_serial_order": state["next_serial_order"],
            "remaining": state["queue_size"] - state["next_serial_order"],
            "family_count": state["catalog"]["family_count"],
            "covered_unit_count": state["catalog"]["covered_unit_count"],
            "statistics": state["statistics"],
        },
    )


def process_one(
    *,
    unit: dict[str, Any],
    state: dict[str, Any],
    source_units: dict[str, dict[str, Any]],
    source_embeddings: dict[str, dict[str, Any]],
    initial_embeddings: dict[str, dict[str, Any]],
    update_events: list[dict[str, Any]],
    prompts: dict[str, str],
    prompt_hashes: dict[str, str],
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    serial_order = unit["serial_order"]
    catalog = state["catalog"]
    bases = active_base_embeddings(catalog, initial_embeddings, update_events)
    _, mappings = catalog_bases(catalog)
    recall_rows, _ = build_recall_records(
        bases=bases,
        mappings=mappings,
        source=source_embeddings,
        query_ids=[unit["id"]],
        top_k=min(args.recall_k, len(bases)),
        chunk_size=args.chunk_size,
    )
    recalled = core.annotate_candidate_origins(
        recall_rows[0]["candidates"],
        catalog=catalog,
        processed_no_match_ids=set(state["processed_unit_ids"]),
    )
    reranked, rerank_meta = rerank_candidates(
        unit_id=unit["id"],
        recalled=recalled,
        catalog=catalog,
        source_embeddings=source_embeddings,
        args=args,
    )
    selected = select_reranked_candidates(
        reranked,
        min_rerank_score=args.min_rerank_score,
        min_candidates=min(args.min_candidates, len(reranked)),
    )
    dynamic = core.dynamic_candidates(selected)
    trace: dict[str, Any] = {
        "serial_order": serial_order,
        "unit_id": unit["id"],
        "recall_k": len(recalled),
        "rerank_candidates": reranked,
        "selected_candidates": selected,
        "dynamic_candidate_family_ids": [item["family_id"] for item in dynamic],
        "dynamic_candidate_filter": args.dynamic_candidate_filter,
        "rerank": rerank_meta,
        "membership_attempts": [],
        "generalization_attempts": [],
    }
    base_event: dict[str, Any] | None = None
    target_family: dict[str, Any] | None = None
    base_changed = False
    outcome = ""

    should_judge = bool(dynamic) or not args.reuse_initial_no_match
    dynamic_only = args.dynamic_candidate_filter and args.reuse_initial_no_match
    remaining = (
        core.membership_candidates(selected, dynamic_only=dynamic_only)
        if should_judge
        else []
    )
    membership_call_index = 0
    while remaining:
        membership_call_index += 1
        judgment, metadata = membership_call(
            unit=unit,
            candidates=remaining,
            catalog=catalog,
            source_units=source_units,
            system_prompt=prompts["membership_system"],
            user_template=prompts["membership_user"],
            prompt_sha256=prompt_hashes["membership"],
            raw_dir=output_dir / RAW_RESPONSES_DIRNAME,
            serial_order=serial_order,
            call_index=membership_call_index,
            args=args,
        )
        state["statistics"]["membership_calls"] += 1
        trace["membership_attempts"].append({**judgment, **metadata})
        if judgment["decision"] == DECISION_NO_MATCH:
            remaining = []
            break
        if judgment["decision"] != DECISION_ADD:
            raise ValueError(f"Unexpected membership decision: {judgment['decision']}")
        family_id = judgment["selected_family_id"]
        family = core.family_by_id(catalog)[family_id]
        if judgment["membership_type"] == "EQUIVALENT_MEMBER":
            target_family = core.apply_equivalent(family, unit["id"])
            core.replace_family(catalog, target_family)
            core.update_family_state(
                catalog=catalog,
                family=target_family,
                unit_id=unit["id"],
                new_family=False,
                base_changed=False,
            )
            state["statistics"]["equivalent_additions"] += 1
            outcome = "EQUIVALENT_MEMBER"
            break

        generalization, generalization_meta = generalization_call(
            unit=unit,
            family=family,
            preliminary_rationale=judgment["rationale"],
            source_units=source_units,
            system_prompt=prompts["generalization_system"],
            user_template=prompts["generalization_user"],
            prompt_sha256=prompt_hashes["generalization"],
            raw_dir=output_dir / RAW_RESPONSES_DIRNAME,
            serial_order=serial_order,
            args=args,
        )
        trace["generalization_attempts"].append({**generalization, **generalization_meta})
        updated, base_changed = core.apply_single_generalization(
            family, unit["id"], generalization
        )
        if updated is None:
            state["statistics"]["generalization_rejections"] += 1
            remaining = [item for item in remaining if item["family_id"] != family_id]
            continue
        target_family = updated
        core.replace_family(catalog, updated)
        core.update_family_state(
            catalog=catalog,
            family=updated,
            unit_id=unit["id"],
            new_family=False,
            base_changed=base_changed,
        )
        state["statistics"]["variant_additions"] += 1
        if base_changed:
            state["statistics"]["full_family_revisions"] += 1
        outcome = generalization["result_type"]
        break

    if target_family is None:
        family_id = core.next_new_family_id(catalog, serial_order)
        target_family = core.create_new_family(unit, family_id)
        core.append_family(catalog, target_family)
        core.update_family_state(
            catalog=catalog,
            family=target_family,
            unit_id=unit["id"],
            new_family=True,
            base_changed=True,
        )
        state["statistics"]["new_families"] += 1
        if args.reuse_initial_no_match and not dynamic:
            state["statistics"]["cache_hit_new_family"] += 1
            outcome = "CACHE_HIT_NEW_FAMILY"
        else:
            outcome = "LLM_NO_MATCH_NEW_FAMILY"
        source = source_embeddings[unit["id"]]
        base_event = make_base_update_event(
            family=target_family,
            vector=source["embedding"],
            model=source["model"],
            dimensions=source["dimensions"],
            serial_order=serial_order,
            update_type="NEW_FAMILY_REUSE_SOURCE",
            vector_origin="SOURCE_UNIT_REUSED",
            source_unit_id=unit["id"],
        )
    elif base_changed:
        if not all((DASHSCOPE_API_KEY, DASHSCOPE_URL, EMBEDDING_MODEL)):
            raise ValueError("Missing DashScope embedding configuration in .env.")
        base = core.base_record_for_family(
            target_family,
            [],
            model=EMBEDDING_MODEL,
            dimensions=source_embeddings[unit["id"]]["dimensions"],
        )
        response = request_embeddings(
            endpoint=resolve_endpoint(DASHSCOPE_URL),
            api_key=DASHSCOPE_API_KEY,
            model=EMBEDDING_MODEL,
            texts=[base["embedding_text"]],
            timeout=args.embedding_timeout,
            max_retries=args.max_retries,
        )
        vector = extract_embeddings(response, 1)[0]
        base_event = make_base_update_event(
            family=target_family,
            vector=vector,
            model=EMBEDDING_MODEL,
            dimensions=len(vector),
            serial_order=serial_order,
            update_type="CHANGED_BASE_API_GENERATED",
            vector_origin="API_GENERATED",
            source_unit_id=None,
        )

    trace["outcome"] = outcome
    trace["target_family_id"] = target_family["family_id"]
    trace["base_changed"] = base_changed
    return trace, base_event


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-catalog", type=Path, default=DEFAULT_INITIAL_CATALOG)
    parser.add_argument("--queue-file", type=Path, default=DEFAULT_QUEUE_FILE)
    parser.add_argument(
        "--initial-embedding-dir", type=Path, default=DEFAULT_INITIAL_EMBEDDING_DIR
    )
    parser.add_argument("--source-file", type=Path, action="append", dest="source_files")
    parser.add_argument(
        "--source-embedding-file",
        type=Path,
        action="append",
        dest="source_embedding_files",
    )
    parser.add_argument(
        "--membership-system-prompt",
        type=Path,
        default=DEFAULT_MEMBERSHIP_SYSTEM_PROMPT,
    )
    parser.add_argument(
        "--membership-user-prompt", type=Path, default=DEFAULT_MEMBERSHIP_USER_PROMPT
    )
    parser.add_argument(
        "--generalization-system-prompt",
        type=Path,
        default=DEFAULT_GENERALIZATION_SYSTEM_PROMPT,
    )
    parser.add_argument(
        "--generalization-user-prompt", type=Path, default=DEFAULT_GENERALIZATION_USER_PROMPT
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--recall-k", type=int, default=DEFAULT_RECALL_K)
    parser.add_argument("--final-k", type=int, default=DEFAULT_FINAL_K)
    parser.add_argument(
        "--min-rerank-score", type=float, default=DEFAULT_MIN_RERANK_SCORE
    )
    parser.add_argument("--min-candidates", type=int, default=DEFAULT_MIN_CANDIDATES)
    parser.add_argument(
        "--dynamic-candidate-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Submit only dynamic candidates to the membership LLM. Disable with "
            "--no-dynamic-candidate-filter to submit the complete retained set "
            "after a dynamic candidate triggers the call."
        ),
    )
    parser.add_argument(
        "--reuse-initial-no-match",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Reuse a previous NO_MATCH judgment when every candidate belongs to "
            "the initial catalog. Disabled by default in the reproduction package."
        ),
    )
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--rerank-timeout", type=int, default=180)
    parser.add_argument(
        "--embedding-timeout", type=int, default=DEFAULT_EMBEDDING_TIMEOUT
    )
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--output-retries", type=int, default=DEFAULT_OUTPUT_RETRIES)
    parser.add_argument("--retry-backoff", type=float, default=DEFAULT_RETRY_BACKOFF)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.recall_k, args.final_k, args.min_candidates, args.chunk_size) < 1:
        raise ValueError("Recall, rerank, candidate, and chunk sizes must be positive.")
    if args.final_k > args.recall_k:
        raise ValueError("final-k cannot exceed recall-k.")
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive.")

    paths = {
        "catalog": resolve_path(args.initial_catalog),
        "queue": resolve_path(args.queue_file),
        "initial_embeddings": resolve_path(args.initial_embedding_dir),
        "membership_system": resolve_path(args.membership_system_prompt),
        "membership_user": resolve_path(args.membership_user_prompt),
        "generalization_system": resolve_path(args.generalization_system_prompt),
        "generalization_user": resolve_path(args.generalization_user_prompt),
    }
    source_paths = [
        resolve_path(path) for path in (args.source_files or DEFAULT_SOURCE_FILES)
    ]
    embedding_paths = [
        resolve_path(path)
        for path in (args.source_embedding_files or DEFAULT_SOURCE_EMBEDDINGS)
    ]
    output_dir = resolve_path(args.output_dir)
    state_path = output_dir / STATE_FILENAME
    events_path = output_dir / BASE_EVENTS_FILENAME
    for path in [
        paths["catalog"],
        paths["queue"],
        paths["membership_system"],
        paths["membership_user"],
        paths["generalization_system"],
        paths["generalization_user"],
        *source_paths,
        *embedding_paths,
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)

    queue = core.load_no_match_queue(paths["queue"])
    source_units = core.load_source_units(source_paths)
    source_embeddings = load_source_embeddings(embedding_paths)
    missing = sorted(set(item["id"] for item in queue) - set(source_embeddings))
    if missing:
        raise ValueError(f"Queue units missing embeddings: {missing[:20]}")
    initial_embeddings = load_initial_active_embeddings(paths["initial_embeddings"])

    if args.overwrite:
        state = core.initialize_state(llm_io.read_json(paths["catalog"]), len(queue))
        update_events: list[dict[str, Any]] = []
    elif state_path.exists():
        state = llm_io.read_json(state_path)
        if not isinstance(state, dict):
            raise ValueError(f"Invalid state file: {state_path}")
        update_events = load_update_events(events_path)
    else:
        state = core.initialize_state(llm_io.read_json(paths["catalog"]), len(queue))
        update_events = load_update_events(events_path)
    core.validate_state(state, queue)
    initial_active = active_base_embeddings(
        state["catalog"], initial_embeddings, update_events
    )
    embedding_model, embedding_dimensions = validate_embedding_configuration(
        initial_active, source_embeddings
    )
    if EMBEDDING_MODEL and EMBEDDING_MODEL != embedding_model:
        raise ValueError(
            f"EMBEDDING_MODEL={EMBEDDING_MODEL!r} differs from stored model "
            f"{embedding_model!r}."
        )

    prompts = {
        key: paths[key].read_text(encoding="utf-8").strip()
        for key in (
            "membership_system",
            "membership_user",
            "generalization_system",
            "generalization_user",
        )
    }
    prompt_hashes = {
        "membership": llm_io.sha256_text(
            prompts["membership_system"] + "\n" + prompts["membership_user"]
        ),
        "generalization": llm_io.sha256_text(
            prompts["generalization_system"] + "\n" + prompts["generalization_user"]
        ),
    }
    preview = {
        "queue_size": len(queue),
        "already_processed": state["next_serial_order"],
        "remaining": len(queue) - state["next_serial_order"],
        "initial_family_count": len(core.family_by_id(state["catalog"])),
        "recall_k": args.recall_k,
        "final_k": args.final_k,
        "min_rerank_score": args.min_rerank_score,
        "min_candidates": args.min_candidates,
        "dynamic_candidate_filter": args.dynamic_candidate_filter,
        "llm_model": LLM_MODEL,
        "rerank_model": RERANK_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimensions": embedding_dimensions,
    }
    if args.dry_run:
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        return

    if not all((DEEPSEEK_API_KEY, DEEPSEEK_URL, LLM_MODEL)):
        raise ValueError("Missing DEEPSEEK_API_KEY, DEEPSEEK_URL, or DEEPSEEK_V4_FLASH.")
    validate_rerank_config()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for path in (
            events_path,
            output_dir / RESULTS_FILENAME,
            output_dir / RUNTIME_FILENAME,
        ):
            if path.exists():
                path.unlink()

    rebuild_results_index(output_dir, queue, state["next_serial_order"])

    start = state["next_serial_order"]
    stop = len(queue) if args.limit is None else min(len(queue), start + args.limit)
    for index in range(start, stop):
        unit = queue[index]
        started = time.monotonic()
        try:
            trace, base_event = process_one(
                unit=unit,
                state=state,
                source_units=source_units,
                source_embeddings=source_embeddings,
                initial_embeddings=initial_embeddings,
                update_events=update_events,
                prompts=prompts,
                prompt_hashes=prompt_hashes,
                output_dir=output_dir,
                args=args,
            )
            if base_event is not None:
                append_jsonl(events_path, base_event)
                update_events.append(base_event)
            state["next_serial_order"] = index + 1
            state["processed_unit_ids"].append(unit["id"])
            state["statistics"]["processed"] += 1
            state["catalog"]["catalog_version"] = index + 1
            update_catalog_counts(state["catalog"])
            core.validate_state(state, queue)
            trace["completed_at"] = utc_now()
            trace["elapsed_seconds"] = round(time.monotonic() - started, 3)
            llm_io.write_json(
                unit_result_path(output_dir, index, unit["id"]), trace
            )
            persist_committed_state(
                state=state,
                output_dir=output_dir,
                initial_embeddings=initial_embeddings,
                update_events=update_events,
            )
            append_jsonl(output_dir / RESULTS_FILENAME, trace)
            append_jsonl(
                output_dir / RUNTIME_FILENAME,
                {
                    "serial_order": index,
                    "unit_id": unit["id"],
                    "status": "success",
                    "outcome": trace["outcome"],
                    "elapsed_seconds": trace["elapsed_seconds"],
                },
            )
            print(
                f"[{index + 1}/{len(queue)}] unit={unit['id']} "
                f"outcome={trace['outcome']} family={trace['target_family_id']}"
            )
        except Exception as error:
            append_jsonl(
                output_dir / RUNTIME_FILENAME,
                {
                    "serial_order": index,
                    "unit_id": unit["id"],
                    "status": "error",
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    **llm_io.request_error_diagnostics(error),
                },
            )
            raise

    print(
        f"Done. processed={state['next_serial_order']}/{len(queue)} "
        f"families={state['catalog']['family_count']} output={output_dir}"
    )


if __name__ == "__main__":
    main()

