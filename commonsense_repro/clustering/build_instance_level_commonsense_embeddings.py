"""Build configurable-field embeddings for common-sense analysis."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Iterator
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from commonsense_repro.common.paths import PACKAGE_ROOT


# Edit these paths and fields for a different embedding experiment.
DEFAULT_INPUT_FILE = Path("data/checkpoints/instance_level_commonsense/gemini_v1_2_negative.json")
DEFAULT_OUTPUT_FILE = Path("work/stage2/embeddings/gemini_v1_2_negative_embeddings.jsonl")
DEFAULT_SUMMARY_FILE = Path("work/stage2/embeddings/gemini_v1_2_negative_embeddings_summary.json")

DEFAULT_EMBEDDING_FIELDS = [
    "violated_commonsense_rule",
    "situation",
]

FIELD_LABELS = {
    "violated_commonsense_rule": "Violated common-sense rule",
    "situation": "Situation",
}

DEFAULT_BATCH_SIZE = 10
DEFAULT_TIMEOUT = 120
DEFAULT_MAX_RETRIES = 3


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def load_default_env() -> None:
    load_dotenv(PACKAGE_ROOT / ".env")


def load_input_records(path: Path, fields: list[str]) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}.")

    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Expected a JSON object at index {index}.")

        unique_id = item.get("id")
        if not isinstance(unique_id, str) or not unique_id.strip():
            raise ValueError(f"Missing non-empty id at index {index}.")
        if unique_id in seen_ids:
            raise ValueError(f"Duplicate id in input: {unique_id}")
        seen_ids.add(unique_id)

        for field in fields:
            value = item.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Record {unique_id} has an empty {field!r} field.")
        records.append(item)

    return records


def build_embedding_text(record: dict[str, Any], fields: list[str]) -> str:
    return "\n".join(
        f"{FIELD_LABELS.get(field, field)}: {record[field].strip()}"
        for field in fields
    )


def iter_batches(records: list[dict[str, Any]], size: int) -> Iterator[list[dict[str, Any]]]:
    for index in range(0, len(records), size):
        yield records[index : index + size]


def resolve_endpoint(base_url: str) -> str:
    clean_url = base_url.rstrip("/")
    if clean_url.endswith("/embeddings"):
        return clean_url
    return f"{clean_url}/embeddings"


def request_embeddings_once(
    *,
    endpoint: str,
    api_key: str,
    model: str,
    texts: list[str],
    timeout: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "input": texts,
    }
    request = Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def request_embeddings(
    *,
    endpoint: str,
    api_key: str,
    model: str,
    texts: list[str],
    timeout: int,
    max_retries: int,
) -> dict[str, Any]:
    for attempt in range(max_retries + 1):
        try:
            return request_embeddings_once(
                endpoint=endpoint,
                api_key=api_key,
                model=model,
                texts=texts,
                timeout=timeout,
            )
        except HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            retryable = error.code == 429 or error.code >= 500
            if not retryable or attempt >= max_retries:
                raise RuntimeError(
                    f"Embedding HTTP error status={error.code} body={body}"
                ) from error
        except (URLError, TimeoutError) as error:
            if attempt >= max_retries:
                raise RuntimeError(f"Embedding request failed: {error}") from error

        time.sleep(2**attempt)

    raise RuntimeError("Embedding request failed after retries.")


def extract_embeddings(response: dict[str, Any], expected_count: int) -> list[list[float]]:
    data = response.get("data")
    if not isinstance(data, list):
        raise ValueError("Embedding response is missing the data list.")

    items = sorted(data, key=lambda item: item.get("index", 0))
    embeddings: list[list[float]] = []
    for item in items:
        embedding = item.get("embedding") if isinstance(item, dict) else None
        if not isinstance(embedding, list) or not embedding:
            raise ValueError("Embedding response item has no embedding vector.")
        embeddings.append(embedding)

    if len(embeddings) != expected_count:
        raise ValueError(f"Expected {expected_count} embeddings, got {len(embeddings)}.")
    return embeddings


def load_completed_ids(path: Path, model: str, fields: list[str]) -> set[str]:
    if not path.exists():
        return set()

    completed: set[str] = set()
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("model") != model or item.get("embedding_fields") != fields:
                raise ValueError(
                    f"Existing output configuration differs at {path}:{line_number}; "
                    "use --overwrite to rebuild it."
                )
            completed.add(str(item["id"]))
    return completed


def append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        for record in records:
            json.dump(record, file, ensure_ascii=False)
            file.write("\n")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


def accumulate_usage(total: dict[str, int], response: dict[str, Any]) -> None:
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return
    for key, value in usage.items():
        if isinstance(value, int):
            total[key] = total.get(key, 0) + value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate configurable-field embeddings for common-sense analysis."
    )
    parser.add_argument("--input-file", type=Path, default=DEFAULT_INPUT_FILE)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--summary-file", type=Path, default=DEFAULT_SUMMARY_FILE)
    parser.add_argument(
        "--fields",
        nargs="+",
        default=DEFAULT_EMBEDDING_FIELDS,
        help="Record fields to concatenate in the given order.",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.input_file.is_file():
        raise FileNotFoundError(f"Input file does not exist: {args.input_file}")
    if args.batch_size < 1 or args.batch_size > 10:
        raise ValueError("batch_size must be between 1 and 10.")
    fields = [field.strip() for field in args.fields if field.strip()]
    if not fields:
        raise ValueError("fields must contain at least one non-empty field name.")
    if len(fields) != len(set(fields)):
        raise ValueError(f"fields contains duplicates: {fields}")

    records = load_input_records(args.input_file, fields)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("limit must be positive.")
        records = records[: args.limit]

    if args.dry_run:
        preview = [
            {
                "id": record["id"],
                "embedding_text": build_embedding_text(record, fields),
            }
            for record in records[:3]
        ]
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        print(f"Dry run OK. records={len(records)} fields={fields}")
        return

    load_default_env()
    api_key = os.getenv("DASHSCOPE_API_KEY")
    base_url = os.getenv("DASHSCOPE_URL")
    model = os.getenv("EMBEDDING_MODEL")
    missing_env = [
        name
        for name, value in [
            ("DASHSCOPE_API_KEY", api_key),
            ("DASHSCOPE_URL", base_url),
            ("EMBEDDING_MODEL", model),
        ]
        if not value
    ]
    if missing_env:
        raise ValueError(f"Missing .env values: {', '.join(missing_env)}")

    if args.overwrite and args.output_file.exists():
        args.output_file.unlink()

    completed_ids = load_completed_ids(args.output_file, model or "", fields)
    pending_records = [record for record in records if record["id"] not in completed_ids]
    endpoint = resolve_endpoint(base_url or "")
    total_usage: dict[str, int] = {}
    vector_dimensions: int | None = None

    for batch in iter_batches(pending_records, args.batch_size):
        texts = [build_embedding_text(record, fields) for record in batch]
        response = request_embeddings(
            endpoint=endpoint,
            api_key=api_key or "",
            model=model or "",
            texts=texts,
            timeout=args.timeout,
            max_retries=args.max_retries,
        )
        embeddings = extract_embeddings(response, len(batch))
        dimensions = len(embeddings[0])
        if vector_dimensions is not None and dimensions != vector_dimensions:
            raise ValueError(
                f"Embedding dimensions changed from {vector_dimensions} to {dimensions}."
            )
        vector_dimensions = dimensions

        output_records = [
            {
                "id": record["id"],
                "model": model,
                "dimensions": len(embedding),
                "embedding_fields": fields,
                "embedding_text": text,
                "embedding": embedding,
            }
            for record, text, embedding in zip(batch, texts, embeddings)
        ]
        append_jsonl(args.output_file, output_records)
        accumulate_usage(total_usage, response)
        print(f"Wrote batch size={len(batch)} first_id={batch[0]['id']}")

    summary = {
        "input_file": str(args.input_file),
        "output_file": str(args.output_file),
        "model": model,
        "dimensions": vector_dimensions,
        "embedding_fields": fields,
        "input_count": len(records),
        "already_completed_count": len(completed_ids),
        "new_count": len(pending_records),
        "usage": total_usage,
    }
    write_json(args.summary_file, summary)
    print(
        f"Done. new={len(pending_records)}, skipped={len(completed_ids)}, "
        f"output={args.output_file}"
    )


if __name__ == "__main__":
    main()

