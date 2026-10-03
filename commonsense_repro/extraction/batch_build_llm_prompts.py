"""Batch build LLM prompt JSON files from GitHub issue Markdown files."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from commonsense_repro.extraction.build_llm_prompt import build_prompt_payload, write_json


DEFAULT_INPUT_DIR = Path("dataset/github_android_bug_issues/llm_filter/markdown")
DEFAULT_OUTPUT_DIR = Path("data/llm_prompts/v1.2/github")
DEFAULT_TEMPLATE = Path("prompts/extract_instance_level_commonsense_v1.2.md")
DEFAULT_LOG_FILE = Path("data/logs/batch_build_llm_prompts.log")


def setup_logger(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("batch_build_llm_prompts")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def iter_markdown_files(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.rglob("*.md") if path.is_file())


def output_path_for(markdown_path: Path, input_dir: Path, output_dir: Path) -> Path:
    relative_path = markdown_path.relative_to(input_dir)
    return output_dir / relative_path.with_suffix(".json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch build multimodal prompt JSON files from Markdown bug reports."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--task", default="extract")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Build only the first N prompts.")
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG_FILE)
    args = parser.parse_args()

    logger = setup_logger(args.log_file)
    logger.info(
        "Start batch prompt build input_dir=%s template=%s output_dir=%s task=%s limit=%s overwrite=%s",
        args.input_dir,
        args.template,
        args.output_dir,
        args.task,
        args.limit,
        args.overwrite,
    )

    template = args.template.read_text(encoding="utf-8")
    markdown_files = iter_markdown_files(args.input_dir)

    built = 0
    skipped = 0
    failed = 0

    for markdown_path in markdown_files:
        if args.limit is not None and built >= args.limit:
            logger.info("Limit reached limit=%s", args.limit)
            break

        output_path = output_path_for(markdown_path, args.input_dir, args.output_dir)
        if output_path.exists() and not args.overwrite:
            skipped += 1
            logger.info("Skipped existing prompt input=%s output=%s", markdown_path, output_path)
            continue

        try:
            bug_report = markdown_path.read_text(encoding="utf-8")
            payload = build_prompt_payload(
                task=args.task,
                source_file=markdown_path,
                template_file=args.template,
                bug_report=bug_report,
                template=template,
            )
            write_json(output_path, payload, args.overwrite)
            built += 1
            logger.info(
                "Built prompt input=%s output=%s image_count=%s image_urls=%s",
                markdown_path,
                output_path,
                payload["image_count"],
                payload["image_urls"],
            )
        except Exception as error:
            failed += 1
            logger.exception(
                "Failed prompt build input=%s output=%s error=%s",
                markdown_path,
                output_path,
                error,
            )

    logger.info("Done built=%s skipped=%s failed=%s", built, skipped, failed)
    print(f"Done. built={built}, skipped={skipped}, failed={failed}. Log: {args.log_file}")


if __name__ == "__main__":
    main()

