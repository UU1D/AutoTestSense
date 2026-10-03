"""Reusable validation, ID mapping, persistence, and LLM transport helpers."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterable, Iterator

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI


MAX_ERROR_BODY_CHARS = 20_000


def _bounded_error_value(value: Any) -> Any:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value[:MAX_ERROR_BODY_CHARS]
    try:
        json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return repr(value)[:MAX_ERROR_BODY_CHARS]
    return value


def request_error_diagnostics(error: Exception) -> dict[str, Any]:
    """Return JSON-safe transport and server details without request secrets."""
    details: dict[str, Any] = {
        "error_type": type(error).__name__,
        "error_module": type(error).__module__,
        "error": str(error),
        "error_repr": repr(error),
        "retryable": is_retryable_request_error(error),
    }

    exception_chain: list[dict[str, str]] = []
    current: BaseException | None = error
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        exception_chain.append(
            {
                "type": type(current).__name__,
                "module": type(current).__module__,
                "message": str(current),
                "repr": repr(current),
            }
        )
        current = current.__cause__ or current.__context__
    details["exception_chain"] = exception_chain

    request = getattr(error, "request", None)
    response = getattr(error, "response", None)
    if request is None and response is not None:
        try:
            request = response.request
        except (AttributeError, RuntimeError):
            request = None
    if request is not None:
        details["request"] = {
            "method": getattr(request, "method", None),
            "url": str(getattr(request, "url", "")),
        }

    if isinstance(error, APIStatusError):
        details["status_code"] = error.status_code
        request_id = getattr(error, "request_id", None)
        if request_id:
            details["request_id"] = request_id

    if response is not None:
        response_details: dict[str, Any] = {
            "status_code": getattr(response, "status_code", None),
        }
        headers = getattr(response, "headers", None)
        if headers is not None:
            response_details["headers"] = dict(headers)
        try:
            response_details["body"] = _bounded_error_value(response.json())
        except Exception:
            try:
                response_details["body"] = _bounded_error_value(response.text)
            except Exception as body_error:
                response_details["body_read_error"] = repr(body_error)
        details["response"] = response_details
    elif getattr(error, "body", None) is not None:
        details["response_body"] = _bounded_error_value(error.body)

    error_number = getattr(error, "errno", None)
    if error_number is not None:
        details["errno"] = error_number
    details["traceback"] = "".join(
        traceback.format_exception(type(error), error, error.__traceback__)
    )
    return details


def resolve_base_url(base_url: str) -> str:
    clean_url = base_url.rstrip("/")
    if clean_url.endswith("/chat/completions"):
        return clean_url[: -len("/chat/completions")]
    return clean_url


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def natural_sort_key(value: str) -> tuple[tuple[int, Any], ...]:
    parts = re.split(r"(\d+)", value.casefold())
    return tuple(
        (1, int(part)) if part.isdigit() else (0, part)
        for part in parts
        if part
    )


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}.")
            records.append(record)
    return records


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")
    temporary.replace(path)


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as file:
        file.write(content)
    temporary.replace(path)


def append_jsonl(file: Any, record: dict[str, Any]) -> None:
    file.write(json.dumps(record, ensure_ascii=False) + "\n")
    file.flush()
    os.fsync(file.fileno())


def require_nonempty_string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} must be a non-empty string.")
    return value.strip()


def normalize_string_array(value: Any, location: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{location} must be an array.")
    normalized = [
        require_nonempty_string(item, f"{location}[{index}]")
        for index, item in enumerate(value)
    ]
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{location} contains duplicate strings.")
    return normalized


def require_exact_keys(
    value: dict[str, Any], expected: set[str], location: str
) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{location} must contain exactly {sorted(expected)}; "
            f"found {sorted(actual)}."
        )


def validate_disjoint_known_ids(
    ids: list[str],
    *,
    allowed_ids: set[str],
    used_ids: set[str],
    location: str,
    min_size: int = 1,
) -> list[str]:
    normalized = [
        require_nonempty_string(value, f"{location}[{index}]")
        for index, value in enumerate(ids)
    ]
    if len(normalized) < min_size:
        raise ValueError(f"{location} must contain at least {min_size} IDs.")
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{location} contains duplicate IDs.")
    unknown = set(normalized) - allowed_ids
    if unknown:
        raise ValueError(
            f"{location} contains unknown IDs: "
            f"{sorted(unknown, key=natural_sort_key)}"
        )
    overlap = set(normalized) & used_ids
    if overlap:
        raise ValueError(
            f"IDs appear in multiple groups: "
            f"{sorted(overlap, key=natural_sort_key)}"
        )
    normalized.sort(key=natural_sort_key)
    used_ids.update(normalized)
    return normalized


def infer_singleton_ids(input_ids: Iterable[str], used_ids: set[str]) -> list[str]:
    return sorted(set(input_ids) - used_ids, key=natural_sort_key)


def build_user_prompt(
    template: str,
    value: Any,
    placeholder: str = "{{units_json}}",
) -> str:
    count = template.count(placeholder)
    if count != 1:
        raise ValueError(
            f"User prompt must contain exactly one {placeholder!r}; found {count}."
        )
    return template.replace(
        placeholder,
        json.dumps(value, ensure_ascii=False, indent=2),
    )


def build_model_id_mapping(units: list[dict[str, Any]]) -> dict[str, str]:
    return {f"U{index}": str(unit["id"]) for index, unit in enumerate(units)}


def invert_id_mapping(model_to_source_id: dict[str, str]) -> dict[str, str]:
    source_to_model_id = {
        source_id: model_id for model_id, source_id in model_to_source_id.items()
    }
    if len(source_to_model_id) != len(model_to_source_id):
        raise ValueError("ID mapping contains duplicate source IDs.")
    return source_to_model_id


def id_mapping_records(
    model_to_source_id: dict[str, str],
) -> list[dict[str, str]]:
    return [
        {"model_id": model_id, "source_id": source_id}
        for model_id, source_id in model_to_source_id.items()
    ]


def batch_model_id_mapping(batch: dict[str, Any]) -> dict[str, str]:
    """Return an explicit batch mapping or the standard U-index mapping."""
    explicit = batch.get("model_to_source_id")
    if explicit is None:
        return build_model_id_mapping(batch["units"])
    if not isinstance(explicit, dict):
        raise ValueError("batch.model_to_source_id must be an object.")
    mapping: dict[str, str] = {}
    source_ids: set[str] = set()
    for raw_model_id, raw_source_id in explicit.items():
        model_id = require_nonempty_string(
            raw_model_id, "batch.model_to_source_id key"
        )
        source_id = require_nonempty_string(
            raw_source_id, f"batch.model_to_source_id[{model_id!r}]"
        )
        if source_id in source_ids:
            raise ValueError(
                f"batch.model_to_source_id contains duplicate source ID {source_id}."
            )
        mapping[model_id] = source_id
        source_ids.add(source_id)
    if not mapping:
        raise ValueError("batch.model_to_source_id must not be empty.")
    return mapping


def parse_id_mapping_records(
    value: Any,
    *,
    expected_mapping: dict[str, str] | None = None,
    location: str = "id_mapping",
) -> dict[str, str]:
    if not isinstance(value, list):
        raise ValueError(f"{location} must be an array.")
    mapping: dict[str, str] = {}
    source_ids: set[str] = set()
    for index, item in enumerate(value):
        item_location = f"{location}[{index}]"
        if not isinstance(item, dict):
            raise ValueError(f"{item_location} must be an object.")
        require_exact_keys(item, {"model_id", "source_id"}, item_location)
        model_id = require_nonempty_string(
            item["model_id"], f"{item_location}.model_id"
        )
        source_id = require_nonempty_string(
            item["source_id"], f"{item_location}.source_id"
        )
        if model_id in mapping:
            raise ValueError(f"{location} contains duplicate model ID {model_id}.")
        if source_id in source_ids:
            raise ValueError(f"{location} contains duplicate source ID {source_id}.")
        mapping[model_id] = source_id
        source_ids.add(source_id)
    if expected_mapping is not None and mapping != expected_mapping:
        raise ValueError(f"{location} does not match the expected mapping.")
    return mapping


def build_model_units(
    units: list[dict[str, Any]], model_to_source_id: dict[str, str]
) -> list[dict[str, Any]]:
    source_to_model_id = invert_id_mapping(model_to_source_id)
    return [
        {**unit, "id": source_to_model_id[str(unit["id"])]}
        for unit in units
    ]


def restore_ids(ids: Iterable[str], model_to_source_id: dict[str, str]) -> list[str]:
    return sorted(
        (model_to_source_id[model_id] for model_id in ids),
        key=natural_sort_key,
    )


def strip_json_fence(raw_content: str) -> str:
    text = raw_content.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return text


def response_usage(response: dict[str, Any]) -> dict[str, int]:
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return {}
    return {
        key: value
        for key, value in usage.items()
        if isinstance(key, str) and isinstance(value, int)
    }


def add_usage(total: dict[str, int], usage: dict[str, int]) -> None:
    for key, value in usage.items():
        total[key] = total.get(key, 0) + value


def safe_path_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return cleaned or "unknown"


def call_streaming_chat_completions(
    *, base_url: str, api_key: str, payload: dict[str, Any], timeout: int
) -> Iterator[dict[str, Any]]:
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        max_retries=0,
    )
    stream = client.chat.completions.create(
        **payload,
        stream=True,
        stream_options={"include_usage": True},
    )
    try:
        for chunk in stream:
            yield chunk.model_dump(mode="json")
    finally:
        stream.close()


def call_non_streaming_chat_completions(
    *, base_url: str, api_key: str, payload: dict[str, Any], timeout: int
) -> dict[str, Any]:
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        max_retries=0,
    )
    response = client.chat.completions.create(**payload, stream=False)
    return response.model_dump(mode="json")


def _chunk_text(chunk: dict[str, Any], field: str) -> str:
    choices = chunk.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    delta = first.get("delta") if isinstance(first, dict) else None
    value = delta.get(field) if isinstance(delta, dict) else None
    return value if isinstance(value, str) else ""


def _response_message_text(response: dict[str, Any], field: str) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    message = first.get("message") if isinstance(first, dict) else None
    value = message.get(field) if isinstance(message, dict) else None
    return value if isinstance(value, str) else ""


def _id_mapping_records(batch: dict[str, Any]) -> list[dict[str, str]]:
    return id_mapping_records(batch_model_id_mapping(batch))


def stream_response_to_files(
    *,
    base_url: str,
    api_key: str,
    payload: dict[str, Any],
    timeout: int,
    raw_responses_dir: Path,
    batch: dict[str, Any],
    model: str,
    prompt_sha256: str,
    output_attempt: int,
    network_attempt: int,
) -> tuple[Path, dict[str, Any]]:
    batch_id = str(batch["batch_id"])
    stem = (
        f"output_{output_attempt + 1:02d}_network_{network_attempt + 1:02d}_"
        f"{time.time_ns()}"
    )
    batch_dir = raw_responses_dir / safe_path_component(batch_id)
    batch_dir.mkdir(parents=True, exist_ok=True)
    raw_path = batch_dir / f"{stem}_assistant_output.json"
    stream_path = batch_dir / f"{stem}_stream_events.jsonl"
    content_path = batch_dir / f"{stem}_assistant_content.txt"
    reasoning_path = batch_dir / f"{stem}_reasoning_content.txt"
    partial_stream = Path(str(stream_path) + ".partial")
    partial_content = Path(str(content_path) + ".partial")
    partial_reasoning = Path(str(reasoning_path) + ".partial")
    error_path = batch_dir / f"{stem}_stream_error.json"
    usage: dict[str, int] = {}
    request_id: str | None = None
    chunk_count = 0
    reasoning_char_count = 0
    try:
        with (
            partial_stream.open("w", encoding="utf-8", newline="\n") as events,
            partial_content.open("w", encoding="utf-8", newline="") as content,
            partial_reasoning.open("w", encoding="utf-8", newline="") as reasoning,
        ):
            for chunk in call_streaming_chat_completions(
                base_url=base_url,
                api_key=api_key,
                payload=payload,
                timeout=timeout,
            ):
                if not isinstance(chunk, dict):
                    raise ValueError("A streaming response chunk is not an object.")
                events.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                events.flush()
                chunk_count += 1
                if request_id is None and isinstance(chunk.get("id"), str):
                    request_id = chunk["id"]
                chunk_usage = response_usage(chunk)
                if chunk_usage:
                    usage = chunk_usage
                content_piece = _chunk_text(chunk, "content")
                if content_piece:
                    content.write(content_piece)
                    content.flush()
                reasoning_piece = _chunk_text(chunk, "reasoning_content")
                if reasoning_piece:
                    reasoning.write(reasoning_piece)
                    reasoning.flush()
                    reasoning_char_count += len(reasoning_piece)
    except Exception as error:
        write_json(
            error_path,
            {
                "batch_id": batch_id,
                "output_attempt": output_attempt + 1,
                "network_attempt": network_attempt + 1,
                "model": model,
                "stream": True,
                "prompt_sha256": prompt_sha256,
                "input_sha256": batch["input_sha256"],
                "id_mapping": _id_mapping_records(batch),
                "request_id": request_id,
                "chunk_count": chunk_count,
                **request_error_diagnostics(error),
                "partial_files": {
                    "stream_events": str(partial_stream),
                    "assistant_content": str(partial_content),
                    "reasoning_content": str(partial_reasoning),
                },
            },
        )
        raise
    partial_stream.replace(stream_path)
    partial_content.replace(content_path)
    partial_reasoning.replace(reasoning_path)
    assistant_content = content_path.read_text(encoding="utf-8").strip()
    assistant_output: Any = None
    assistant_json_error: str | None = None
    if assistant_content:
        try:
            assistant_output = json.loads(strip_json_fence(assistant_content))
        except json.JSONDecodeError as error:
            assistant_json_error = str(error)
    write_json(
        raw_path,
        {
            "metadata": {
                "batch_id": batch_id,
                "output_attempt": output_attempt + 1,
                "network_attempt": network_attempt + 1,
                "model": model,
                "stream": True,
                "prompt_sha256": prompt_sha256,
                "input_sha256": batch["input_sha256"],
                "identifier_scheme": batch.get(
                    "identifier_scheme", "batch_local_u_index_v1"
                ),
                "request_id": request_id,
                "chunk_count": chunk_count,
                "usage": usage,
            },
            "id_mapping": _id_mapping_records(batch),
            "assistant_output": assistant_output,
            "assistant_raw_text": assistant_content if assistant_output is None else None,
            "diagnostics": {
                "assistant_json_error": assistant_json_error,
                "assistant_content_extraction_error": (
                    None if assistant_content else "The completed stream contained no content."
                ),
            },
            "source_files": {
                "stream_events": str(stream_path),
                "assistant_content": str(content_path),
                "reasoning_content": str(reasoning_path),
            },
        },
    )
    return raw_path, {
        "request_id": request_id,
        "usage": usage,
        "chunk_count": chunk_count,
        "reasoning_char_count": reasoning_char_count,
    }


def non_streaming_response_to_files(
    *,
    base_url: str,
    api_key: str,
    payload: dict[str, Any],
    timeout: int,
    raw_responses_dir: Path,
    batch: dict[str, Any],
    model: str,
    prompt_sha256: str,
    output_attempt: int,
    network_attempt: int,
) -> tuple[Path, dict[str, Any]]:
    batch_id = str(batch["batch_id"])
    stem = (
        f"output_{output_attempt + 1:02d}_network_{network_attempt + 1:02d}_"
        f"{time.time_ns()}"
    )
    batch_dir = raw_responses_dir / safe_path_component(batch_id)
    batch_dir.mkdir(parents=True, exist_ok=True)
    raw_path = batch_dir / f"{stem}_assistant_output.json"
    api_path = batch_dir / f"{stem}_api_response.json"
    content_path = batch_dir / f"{stem}_assistant_content.txt"
    reasoning_path = batch_dir / f"{stem}_reasoning_content.txt"
    error_path = batch_dir / f"{stem}_request_error.json"
    try:
        response = call_non_streaming_chat_completions(
            base_url=base_url,
            api_key=api_key,
            payload=payload,
            timeout=timeout,
        )
    except Exception as error:
        write_json(
            error_path,
            {
                "batch_id": batch_id,
                "output_attempt": output_attempt + 1,
                "network_attempt": network_attempt + 1,
                "model": model,
                "stream": False,
                "prompt_sha256": prompt_sha256,
                "input_sha256": batch["input_sha256"],
                "id_mapping": _id_mapping_records(batch),
                **request_error_diagnostics(error),
            },
        )
        raise
    if not isinstance(response, dict):
        raise ValueError("A non-streaming response is not an object.")
    write_json(api_path, response)
    request_id = response.get("id") if isinstance(response.get("id"), str) else None
    usage = response_usage(response)
    assistant_content = _response_message_text(response, "content").strip()
    reasoning_content = _response_message_text(response, "reasoning_content")
    write_text(content_path, assistant_content)
    write_text(reasoning_path, reasoning_content)
    assistant_output: Any = None
    assistant_json_error: str | None = None
    if assistant_content:
        try:
            assistant_output = json.loads(strip_json_fence(assistant_content))
        except json.JSONDecodeError as error:
            assistant_json_error = str(error)
    write_json(
        raw_path,
        {
            "metadata": {
                "batch_id": batch_id,
                "output_attempt": output_attempt + 1,
                "network_attempt": network_attempt + 1,
                "model": model,
                "stream": False,
                "prompt_sha256": prompt_sha256,
                "input_sha256": batch["input_sha256"],
                "identifier_scheme": batch.get(
                    "identifier_scheme", "batch_local_u_index_v1"
                ),
                "request_id": request_id,
                "usage": usage,
            },
            "id_mapping": _id_mapping_records(batch),
            "assistant_output": assistant_output,
            "assistant_raw_text": assistant_content if assistant_output is None else None,
            "diagnostics": {
                "assistant_json_error": assistant_json_error,
                "assistant_content_extraction_error": (
                    None if assistant_content else "The response contained no content."
                ),
            },
            "source_files": {
                "api_response": str(api_path),
                "assistant_content": str(content_path),
                "reasoning_content": str(reasoning_path),
            },
        },
    )
    return raw_path, {
        "request_id": request_id,
        "usage": usage,
        "reasoning_char_count": len(reasoning_content),
    }


def is_retryable_request_error(error: Exception) -> bool:
    if isinstance(error, (APIConnectionError, APITimeoutError, httpx.TransportError)):
        return True
    if isinstance(error, APIStatusError):
        return error.status_code in {408, 409, 429} or error.status_code >= 500
    return False


def request_with_retries(
    *,
    stream: bool,
    base_url: str,
    api_key: str,
    payload: dict[str, Any],
    timeout: int,
    max_retries: int,
    retry_backoff: float,
    raw_responses_dir: Path,
    batch: dict[str, Any],
    model: str,
    prompt_sha256: str,
    output_attempt: int,
) -> tuple[Path, dict[str, Any]]:
    batch_id = str(batch["batch_id"])
    operation = stream_response_to_files if stream else non_streaming_response_to_files
    for network_attempt in range(max_retries + 1):
        try:
            return operation(
                base_url=base_url,
                api_key=api_key,
                payload=payload,
                timeout=timeout,
                raw_responses_dir=raw_responses_dir,
                batch=batch,
                model=model,
                prompt_sha256=prompt_sha256,
                output_attempt=output_attempt,
                network_attempt=network_attempt,
            )
        except Exception as error:
            if not is_retryable_request_error(error) or network_attempt == max_retries:
                raise
            delay = retry_backoff * (2**network_attempt)
            mode = "streaming" if stream else "request"
            print(
                f"batch={batch_id} {mode} error={error}; "
                f"retry={network_attempt + 1}/{max_retries} delay={delay:.1f}s",
                file=sys.stderr,
            )
            time.sleep(delay)
    raise RuntimeError("Unreachable request retry state")


def runtime_key(record: dict[str, Any]) -> tuple[str, str, str, str] | None:
    values = (
        record.get("batch_id"),
        record.get("model"),
        record.get("prompt_sha256"),
        record.get("input_sha256"),
    )
    if not all(isinstance(value, str) and value for value in values):
        return None
    return tuple(str(value) for value in values)  # type: ignore[return-value]


def latest_runtime_records(
    path: Path,
) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    latest: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    if not path.exists():
        return latest
    for record in read_jsonl(path):
        key = runtime_key(record)
        if key is not None:
            latest[key] = record
    return latest

