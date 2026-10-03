"""Build a multimodal LLM prompt JSON from a local bug report Markdown file."""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any


DEFAULT_LOG_FILE = Path("data/logs/build_llm_prompt.log")
IMAGE_EXTENSIONS = ("png", "jpg", "jpeg", "webp", "gif", "bmp")


def setup_logger(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("build_llm_prompt")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def add_unique(urls: list[str], seen: set[str], url: str) -> None:
    clean_url = url.strip().strip("<>").rstrip(").,;]")
    if clean_url and clean_url not in seen:
        seen.add(clean_url)
        urls.append(clean_url)


def extract_image_urls(markdown_text: str) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()

    patterns = [
        r"!\[[^\]]*]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)",
        r"<img\b[^>]*\bsrc=[\"']([^\"']+)[\"'][^>]*>",
        r"https?://github\.com/user-attachments/assets/[^\s)>\"]+",
        rf"https?://user-images\.githubusercontent\.com/[^\s)>\"]+\.(?:{'|'.join(IMAGE_EXTENSIONS)})(?:\?[^\s)>\"]*)?",
        rf"https?://[^\s)>\"]+\.(?:{'|'.join(IMAGE_EXTENSIONS)})(?:\?[^\s)>\"]*)?",
    ]

    matches: list[tuple[int, str]] = []
    for pattern in patterns:
        flags = re.IGNORECASE | re.MULTILINE
        for match in re.finditer(pattern, markdown_text, flags):
            url = match.group(1) if match.groups() else match.group(0)
            matches.append((match.start(), url))

    for _, url in sorted(matches, key=lambda item: item[0]):
        add_unique(urls, seen, url)

    return urls


def build_messages(text_prompt: str, image_urls: list[str]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {"type": "image_url", "image_url": {"url": image_url}}
        for image_url in image_urls
    ]
    content.append({"type": "text", "text": text_prompt})
    return [{"role": "user", "content": content}]


def build_prompt_payload(
    *,
    task: str,
    source_file: Path,
    template_file: Path,
    bug_report: str,
    template: str,
) -> dict[str, Any]:
    text_prompt = template.replace("{BUG_REPORT}", bug_report)
    image_urls = extract_image_urls(bug_report)

    return {
        "task": task,
        "source_file": str(source_file),
        "template_file": str(template_file),
        "has_images": bool(image_urls),
        "image_count": len(image_urls),
        "image_urls": image_urls,
        "text_prompt": text_prompt,
        "messages": build_messages(text_prompt, image_urls),
    }


def write_json(path: Path, data: Any, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists. Use --overwrite to replace it.")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a multimodal prompt JSON from a bug report Markdown file."
    )
    parser.add_argument("--input", type=Path, required=True, help="Bug report Markdown file.")
    parser.add_argument("--template", type=Path, required=True, help="Prompt template file.")
    parser.add_argument("--output", type=Path, required=True, help="Output prompt JSON file.")
    parser.add_argument("--task", default="extract")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG_FILE)
    args = parser.parse_args()

    logger = setup_logger(args.log_file)
    logger.info(
        "Start prompt build input=%s template=%s output=%s task=%s",
        args.input,
        args.template,
        args.output,
        args.task,
    )

    try:
        bug_report = args.input.read_text(encoding="utf-8")
        template = args.template.read_text(encoding="utf-8")
        payload = build_prompt_payload(
            task=args.task,
            source_file=args.input,
            template_file=args.template,
            bug_report=bug_report,
            template=template,
        )
        write_json(args.output, payload, args.overwrite)
        logger.info(
            "Prompt built output=%s image_count=%s image_urls=%s",
            args.output,
            payload["image_count"],
            payload["image_urls"],
        )
        print(f"Wrote {args.output}. Images: {payload['image_count']}. Log: {args.log_file}")
    except Exception as error:
        logger.exception("Failed to build prompt error=%s", error)
        raise


if __name__ == "__main__":
    main()

