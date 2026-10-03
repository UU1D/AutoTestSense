"""Recursively extract JSON payloads from Markdown files into one JSON file."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


# Edit these two paths for the dataset currently being processed.
DEFAULT_INPUT_DIR = Path("data/llm_outputs/v1.2/gemini/github/md")
DEFAULT_OUTPUT_FILE = Path("work/gemini_v1_2_github.json")

FENCED_BLOCK_PATTERN = re.compile(
    r"```(?:json)?\s*(.*?)\s*```",
    flags=re.IGNORECASE | re.DOTALL,
)


def iter_markdown_files(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.rglob("*.md") if path.is_file())


def natural_sort_key(value: str) -> tuple[tuple[int, Any], ...]:
    parts = re.split(r"(\d+)", value.casefold())
    return tuple(
        (1, int(part)) if part.isdigit() else (0, part)
        for part in parts
        if part
    )


def parse_json_object(json_text: str) -> dict[str, Any]:
    data = json.loads(json_text)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object, got {type(data).__name__}.")
    return data


def extract_json_from_markdown(markdown_text: str) -> dict[str, Any]:
    decode_errors: list[str] = []

    for block in FENCED_BLOCK_PATTERN.findall(markdown_text):
        try:
            return parse_json_object(block.strip())
        except (json.JSONDecodeError, ValueError) as error:
            decode_errors.append(str(error))

    stripped = markdown_text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            return parse_json_object(stripped)
        except (json.JSONDecodeError, ValueError) as error:
            decode_errors.append(str(error))

    start = markdown_text.find("{")
    end = markdown_text.rfind("}")
    if start != -1 and end > start:
        try:
            return parse_json_object(markdown_text[start : end + 1])
        except (json.JSONDecodeError, ValueError) as error:
            decode_errors.append(str(error))

    if decode_errors:
        raise ValueError("; ".join(dict.fromkeys(decode_errors)))
    raise ValueError("No JSON object or fenced JSON block found.")


def build_record(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    record: dict[str, Any] = {"id": path.stem}
    record.update(payload)
    record["id"] = path.stem
    return record


def build_error(path: Path, input_dir: Path, error: Exception) -> dict[str, str]:
    return {
        "id": path.stem,
        "source_file": path.relative_to(input_dir).as_posix(),
        "error_type": type(error).__name__,
        "message": str(error),
    }


def extract_all(
    input_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for path in iter_markdown_files(input_dir):
        try:
            markdown_text = path.read_text(encoding="utf-8")
            payload = extract_json_from_markdown(markdown_text)
            records.append(build_record(path, payload))
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
            errors.append(build_error(path, input_dir, error))

    records.sort(key=lambda record: natural_sort_key(str(record["id"])))
    errors.sort(key=lambda error: natural_sort_key(error["id"]))
    return records, errors


def write_json(path: Path, data: Any, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists. Use --overwrite to replace it.")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively extract JSON objects from all Markdown files under an input "
            "directory into one JSON array."
        )
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        nargs="?",
        default=DEFAULT_INPUT_DIR,
        help="Directory to scan recursively for Markdown files.",
    )
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")

    markdown_files = iter_markdown_files(args.input_dir)
    records, errors = extract_all(args.input_dir)

    if errors:
        error_preview = "\n".join(
            f"- {error['source_file']}: {error['message']}" for error in errors[:20]
        )
        raise RuntimeError(
            f"Failed to extract {len(errors)} Markdown file(s); no output was written.\n"
            f"{error_preview}"
        )

    write_json(args.output_file, records, args.overwrite)

    print(
        "Done. "
        f"markdown={len(markdown_files)}, "
        f"valid={len(records)}, "
        "errors=0, "
        f"output={args.output_file}"
    )


if __name__ == "__main__":
    main()

