"""Build separate situation and rule embeddings for local commonsense generalization record bases."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Iterator
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
from commonsense_repro.common.paths import PACKAGE_ROOT as PROJECT_ROOT
load_dotenv(PROJECT_ROOT / ".env")

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "").strip()
DASHSCOPE_URL = os.getenv("DASHSCOPE_URL", "").strip()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "").strip()

DEFAULT_DATA_DIR = Path(
    "work/stage3/"
    "gemini_v1_2_non_mutual_s15_leiden_max25_deepseek_flash_v1_4"
)
DEFAULT_INPUT_FILE = DEFAULT_DATA_DIR / "commonsense_generalization_records.json"
DEFAULT_OUTPUT_DIR = Path(
    "work/stage4"
)
DEFAULT_OUTPUT_FILE = DEFAULT_OUTPUT_DIR / "generalized_rule_embeddings.jsonl"
DEFAULT_SUMMARY_FILE = DEFAULT_OUTPUT_DIR / "generalized_rule_embeddings_summary.json"

DEFAULT_BATCH_SIZE = 10
DEFAULT_TIMEOUT = 120
DEFAULT_MAX_RETRIES = 3

EMBEDDING_FIELDS = ("situation", "commonsense_rule")


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256_json(value: Any) -> str:
    serialized = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def require_nonempty_string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} must be a non-empty string.")
    return value.strip()


def load_rule_families(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        catalog = json.load(file)
    if not isinstance(catalog, dict) or not isinstance(
        catalog.get("rule_families"), list
    ):
        raise ValueError(f"{path} must contain a rule_families array.")

    families: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, family in enumerate(catalog["rule_families"]):
        location = f"{path}.rule_families[{index}]"
        if not isinstance(family, dict):
            raise ValueError(f"{location} must be an object.")
        family_id = require_nonempty_string(family.get("family_id"), f"{location}.family_id")
        if family_id in seen_ids:
            raise ValueError(f"Duplicate family_id: {family_id}")
        base = family.get("base_unit")
        if not isinstance(base, dict):
            raise ValueError(f"{location}.base_unit must be an object.")
        base_unit = {
            "situation": require_nonempty_string(
                base.get("situation"), f"{location}.base_unit.situation"
            ),
            "commonsense_rule": require_nonempty_string(
                base.get("commonsense_rule"),
                f"{location}.base_unit.commonsense_rule",
            ),
        }
        base_origin = require_nonempty_string(
            family.get("base_origin"), f"{location}.base_origin"
        )
        member_count = family.get("member_count")
        if not isinstance(member_count, int) or member_count < 2:
            raise ValueError(f"{location}.member_count must be an integer >= 2.")
        families.append(
            {
                "family_id": family_id,
                "base_origin": base_origin,
                "member_count": member_count,
                "base_unit": base_unit,
                "base_unit_sha256": sha256_json(base_unit),
            }
        )
        seen_ids.add(family_id)
    return families


def iter_batches(
    records: list[dict[str, Any]], size: int
) -> Iterator[list[dict[str, Any]]]:
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
    payload = {"model": model, "input": texts}
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


def extract_embeddings(
    response: dict[str, Any], expected_count: int
) -> list[list[float]]:
    data = response.get("data")
    if not isinstance(data, list):
        raise ValueError("Embedding response is missing the data list.")
    items = sorted(data, key=lambda item: item.get("index", 0))
    embeddings: list[list[float]] = []
    for index, item in enumerate(items):
        embedding = item.get("embedding") if isinstance(item, dict) else None
        if not isinstance(embedding, list) or not embedding:
            raise ValueError(f"Embedding response item {index} has no vector.")
        if not all(isinstance(value, (int, float)) for value in embedding):
            raise ValueError(f"Embedding response item {index} is not numeric.")
        embeddings.append([float(value) for value in embedding])
    if len(embeddings) != expected_count:
        raise ValueError(f"Expected {expected_count} embeddings, got {len(embeddings)}.")
    dimensions = {len(embedding) for embedding in embeddings}
    if len(dimensions) != 1:
        raise ValueError("Embedding response contains inconsistent vector dimensions.")
    return embeddings


def read_existing_embeddings(
    path: Path,
    *,
    model: str,
    families_by_id: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    completed: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}.")
            family_id = require_nonempty_string(
                record.get("family_id"), f"{path}:{line_number}.family_id"
            )
            if family_id in completed:
                raise ValueError(f"Duplicate family_id in existing output: {family_id}")
            source = families_by_id.get(family_id)
            if source is None:
                raise ValueError(
                    f"Existing output contains unknown family_id {family_id}; "
                    "use --overwrite."
                )
            if record.get("model") != model:
                raise ValueError(
                    f"Existing output uses another model at {path}:{line_number}; "
                    "use --overwrite."
                )
            if record.get("base_unit_sha256") != source["base_unit_sha256"]:
                raise ValueError(
                    f"Base unit changed for {family_id}; use --overwrite."
                )
            completed[family_id] = record
    return completed


def append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as file:
        for record in records:
            json.dump(record, file, ensure_ascii=False)
            file.write("\n")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")
    temporary.replace(path)


def accumulate_usage(total: dict[str, int], response: dict[str, Any]) -> None:
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return
    for key, value in usage.items():
        if isinstance(value, int):
            total[key] = total.get(key, 0) + value


def build_output_records(
    batch: list[dict[str, Any]],
    *,
    model: str,
    situation_embeddings: list[list[float]],
    rule_embeddings: list[list[float]],
) -> list[dict[str, Any]]:
    if not (
        len(batch) == len(situation_embeddings) == len(rule_embeddings)
    ):
        raise ValueError("Batch and embedding counts do not match.")
    records: list[dict[str, Any]] = []
    for family, situation_embedding, rule_embedding in zip(
        batch, situation_embeddings, rule_embeddings
    ):
        if len(situation_embedding) != len(rule_embedding):
            raise ValueError(
                f"Embedding dimensions differ for {family['family_id']}."
            )
        records.append(
            {
                "family_id": family["family_id"],
                "model": model,
                "dimensions": len(situation_embedding),
                "base_origin": family["base_origin"],
                "member_count": family["member_count"],
                "base_unit": family["base_unit"],
                "base_unit_sha256": family["base_unit_sha256"],
                "situation_embedding": situation_embedding,
                "commonsense_rule_embedding": rule_embedding,
            }
        )
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate separate situation and common-sense-rule embeddings for "
            "commonsense generalization record bases."
        )
    )
    parser.add_argument("--input-file", type=Path, default=DEFAULT_INPUT_FILE)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--summary-file", type=Path, default=DEFAULT_SUMMARY_FILE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = resolve_path(args.input_file)
    output_path = resolve_path(args.output_file)
    summary_path = resolve_path(args.summary_file)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")
    if not 1 <= args.batch_size <= 10:
        raise ValueError("batch_size must be between 1 and 10.")
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive.")

    all_families = load_rule_families(input_path)
    families = all_families
    if args.limit is not None:
        families = families[: args.limit]

    if args.dry_run:
        preview = [
            {
                "family_id": family["family_id"],
                "situation_text": family["base_unit"]["situation"],
                "commonsense_rule_text": family["base_unit"]["commonsense_rule"],
            }
            for family in families[:3]
        ]
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        print(f"Dry run OK. families={len(families)} fields={list(EMBEDDING_FIELDS)}")
        return

    missing_env = [
        name
        for name, value in (
            ("DASHSCOPE_API_KEY", DASHSCOPE_API_KEY),
            ("DASHSCOPE_URL", DASHSCOPE_URL),
            ("EMBEDDING_MODEL", EMBEDDING_MODEL),
        )
        if not value
    ]
    if missing_env:
        raise ValueError(f"Missing .env values: {', '.join(missing_env)}")

    if args.overwrite:
        if output_path.exists():
            output_path.unlink()
        if summary_path.exists():
            summary_path.unlink()

    families_by_id = {family["family_id"]: family for family in all_families}
    all_completed = read_existing_embeddings(
        output_path,
        model=EMBEDDING_MODEL,
        families_by_id=families_by_id,
    )
    selected_ids = {family["family_id"] for family in families}
    completed = {
        family_id: record
        for family_id, record in all_completed.items()
        if family_id in selected_ids
    }
    pending = [
        family for family in families if family["family_id"] not in completed
    ]
    endpoint = resolve_endpoint(DASHSCOPE_URL)
    usage: dict[str, int] = {}
    completed_dimensions = {
        record.get("dimensions") for record in completed.values()
    }
    if len(completed_dimensions) > 1:
        raise ValueError("Existing output contains inconsistent vector dimensions.")
    dimensions: int | None = (
        next(iter(completed_dimensions)) if completed_dimensions else None
    )
    request_count = 0

    for batch_number, batch in enumerate(
        iter_batches(pending, args.batch_size), start=1
    ):
        situation_response = request_embeddings(
            endpoint=endpoint,
            api_key=DASHSCOPE_API_KEY,
            model=EMBEDDING_MODEL,
            texts=[family["base_unit"]["situation"] for family in batch],
            timeout=args.timeout,
            max_retries=args.max_retries,
        )
        rule_response = request_embeddings(
            endpoint=endpoint,
            api_key=DASHSCOPE_API_KEY,
            model=EMBEDDING_MODEL,
            texts=[family["base_unit"]["commonsense_rule"] for family in batch],
            timeout=args.timeout,
            max_retries=args.max_retries,
        )
        request_count += 2
        situation_embeddings = extract_embeddings(situation_response, len(batch))
        rule_embeddings = extract_embeddings(rule_response, len(batch))
        output_records = build_output_records(
            batch,
            model=EMBEDDING_MODEL,
            situation_embeddings=situation_embeddings,
            rule_embeddings=rule_embeddings,
        )
        batch_dimensions = output_records[0]["dimensions"]
        if dimensions is not None and batch_dimensions != dimensions:
            raise ValueError(
                f"Embedding dimensions changed from {dimensions} to {batch_dimensions}."
            )
        dimensions = batch_dimensions
        append_jsonl(output_path, output_records)
        accumulate_usage(usage, situation_response)
        accumulate_usage(usage, rule_response)
        print(
            f"Wrote batch={batch_number} size={len(batch)} "
            f"first_family_id={batch[0]['family_id']}"
        )

    summary = {
        "input_file": str(args.input_file),
        "output_file": str(args.output_file),
        "model": EMBEDDING_MODEL,
        "dimensions": dimensions,
        "embedding_fields": list(EMBEDDING_FIELDS),
        "input_family_count": len(families),
        "already_completed_count": len(completed),
        "new_count": len(pending),
        "request_count": request_count,
        "usage": usage,
    }
    write_json(summary_path, summary)
    print(
        f"Done. new={len(pending)}, skipped={len(completed)}, "
        f"output={output_path}"
    )


if __name__ == "__main__":
    main()

