"""Fetch a fixed ID-to-GitHub-issue manifest with URL-level caching."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from commonsense_repro.common.paths import DATA_ROOT, WORK_ROOT
from commonsense_repro.extraction.fetch_github_issues_to_md import (
    fetch_issue_markdown,
    load_default_env,
)


DEFAULT_MANIFEST = DATA_ROOT / "input" / "issue_manifest.jsonl"
DEFAULT_OUTPUT_DIR = WORK_ROOT / "stage1" / "reports"
DEFAULT_CACHE_DIR = WORK_ROOT / "stage1" / "issue_cache"


def read_manifest(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object.")
            unit_id = str(row.get("id", "")).strip()
            source = str(row.get("source", "")).strip()
            url = str(row.get("url", "")).strip()
            if not unit_id or not source or not url:
                raise ValueError(f"{path}:{line_number} has an empty field.")
            if unit_id in seen_ids:
                raise ValueError(f"Duplicate manifest id: {unit_id}")
            seen_ids.add(unit_id)
            rows.append({"id": unit_id, "source": source, "url": url})
    return rows


def cache_path_for_url(cache_dir: Path, url: str) -> Path:
    return cache_dir / f"{hashlib.sha256(url.encode('utf-8')).hexdigest()}.md"


def write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def fetch_manifest(
    rows: list[dict[str, str]],
    *,
    output_dir: Path,
    cache_dir: Path,
    timeout: int,
    delay: float,
    overwrite: bool,
    dry_run: bool,
) -> dict[str, Any]:
    fetched_urls = 0
    reused_urls = 0
    written_ids = 0
    token = os.environ.get("GITHUB_TOKEN")
    for index, row in enumerate(rows, start=1):
        output_path = output_dir / row["source"] / f"{row['id']}.md"
        if output_path.exists() and not overwrite:
            continue
        cached = cache_path_for_url(cache_dir, row["url"])
        if dry_run:
            print(f"[{index}/{len(rows)}] {row['id']} <- {row['url']}")
            continue
        if cached.exists() and not overwrite:
            markdown = cached.read_text(encoding="utf-8")
            reused_urls += 1
        else:
            markdown = fetch_issue_markdown(
                row["url"], token=token, timeout=timeout
            )
            write_text_atomic(cached, markdown)
            fetched_urls += 1
            if delay > 0:
                time.sleep(delay)
        write_text_atomic(output_path, markdown)
        written_ids += 1
    return {
        "manifest_count": len(rows),
        "unique_url_count": len({row["url"] for row in rows}),
        "fetched_url_count": fetched_urls,
        "reused_url_count": reused_urls,
        "written_id_count": written_ids,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--delay", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive.")
    load_default_env()
    rows = read_manifest(args.manifest)
    if args.limit is not None:
        selected: list[dict[str, str]] = []
        counts: dict[str, int] = {}
        for row in rows:
            source = row["source"]
            if counts.get(source, 0) >= args.limit:
                continue
            selected.append(row)
            counts[source] = counts.get(source, 0) + 1
        rows = selected
    summary = fetch_manifest(
        rows,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        timeout=args.timeout,
        delay=args.delay,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
