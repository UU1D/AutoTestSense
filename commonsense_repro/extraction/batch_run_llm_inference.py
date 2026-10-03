"""Batch run LLM inference for prompt JSON files with a thread pool."""

from __future__ import annotations

import argparse
import http.client
import json
import logging
import os
import socket
import ssl
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError

from commonsense_repro.extraction.fetch_github_issues_to_md import load_default_env
from commonsense_repro.extraction.run_llm_inference import (
    DEFAULT_IMAGE_DOWNLOAD_TIMEOUT,
    DEFAULT_MAX_IMAGE_BYTES,
    DEFAULT_TIMEOUT,
    build_request_payload,
    call_chat_completions,
    extract_assistant_content,
    has_image_content,
    iter_image_urls,
    is_multimodal_download_error,
    localize_image_content,
    should_prelocalize_messages,
    load_json,
    remove_image_content,
    resolve_endpoint,
    write_json,
    write_text,
)


DEFAULT_INPUT_DIR = Path("data/llm_prompts/v1.2/github")
DEFAULT_OUTPUT_JSON_DIR = Path("data/llm_outputs/v1.2/gemini/github/json")
DEFAULT_OUTPUT_MD_DIR = Path("data/llm_outputs/v1.2/gemini/github/md")
DEFAULT_LOG_FILE = Path("data/logs/batch_run_llm_inference.log")
DEFAULT_PROVIDER = "outer"
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BACKOFF = 2.0
PROVIDER_ENV_VARS = {
    "outer": {
        "api_key": "OUTER_API_KEY",
        "base_url": "OUTER_URL",
        "model": "OUTER_MODEL",
    },
    "dashscope": {
        "api_key": "DASHSCOPE_API_KEY",
        "base_url": "DASHSCOPE_URL",
        "model": "MODEL",
    },
}


def setup_logger(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("batch_run_llm_inference")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def iter_prompt_files(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.rglob("*.json") if path.is_file())


def resolve_provider_config(
    *,
    provider: str,
    model_override: str | None,
    base_url_override: str | None,
) -> tuple[str, str, str]:
    env_vars = PROVIDER_ENV_VARS[provider]
    api_key = os.environ.get(env_vars["api_key"])
    base_url = base_url_override or os.environ.get(env_vars["base_url"])
    model = model_override or os.environ.get(env_vars["model"])

    missing = [
        env_name
        for value, env_name in (
            (api_key, env_vars["api_key"]),
            (base_url, env_vars["base_url"]),
            (model, env_vars["model"]),
        )
        if not value
    ]
    if missing:
        raise ValueError(
            f"Provider '{provider}' is missing configuration: {', '.join(missing)}"
        )

    return api_key, base_url, model


def output_paths_for(
    prompt_path: Path,
    input_dir: Path,
    output_json_dir: Path,
    output_md_dir: Path,
) -> tuple[Path, Path]:
    relative_path = prompt_path.relative_to(input_dir)
    return output_json_dir / relative_path, output_md_dir / relative_path.with_suffix(".md")


def is_retryable_http_error(error: HTTPError) -> bool:
    """Return whether an HTTP response is normally safe to retry."""
    return error.code == 429 or 502 <= error.code <= 504


def call_with_retries(
    *,
    endpoint: str,
    api_key: str,
    payload: dict[str, Any],
    timeout: int,
    max_retries: int,
    retry_backoff: float,
    logger: logging.Logger,
    prompt_path: Path,
) -> dict[str, Any]:
    """Retry transient connection and temporary service failures."""
    retryable_errors = (
        URLError,
        TimeoutError,
        socket.timeout,
        ssl.SSLError,
        http.client.RemoteDisconnected,
        ConnectionError,
    )

    for attempt in range(max_retries + 1):
        try:
            return call_chat_completions(
                endpoint=endpoint,
                api_key=api_key,
                payload=payload,
                timeout=timeout,
            )
        except HTTPError as error:
            if not is_retryable_http_error(error) or attempt == max_retries:
                raise
            error_description = f"HTTP {error.code}"
        except retryable_errors as error:
            if attempt == max_retries:
                raise
            error_description = str(error)

        delay = retry_backoff * (2**attempt)
        logger.warning(
            "Transient API failure; retrying prompt=%s attempt=%s/%s delay_seconds=%.1f error=%s",
            prompt_path,
            attempt + 1,
            max_retries,
            delay,
            error_description,
        )
        time.sleep(delay)

    raise RuntimeError("Unreachable retry state")


def build_result_record(
    *,
    prompt_path: Path,
    prompt_payload: dict[str, Any],
    model: str,
    temperature: float,
    max_tokens: int | None,
    messages: list[dict[str, Any]],
    response: dict[str, Any],
    assistant_content: str,
    fallback_used: bool,
    fallback_mode: str | None,
    image_input_mode: str,
    localized_image_urls: list[str],
    failed_localized_image_urls: list[str],
    multimodal_error_body: str | None,
) -> dict[str, Any]:
    return {
        "input_prompt_file": str(prompt_path),
        "model": model,
        "source_file": prompt_payload.get("source_file"),
        "template_file": prompt_payload.get("template_file"),
        "task": prompt_payload.get("task"),
        "image_urls": prompt_payload.get("image_urls", []),
        "request": {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": messages,
        },
        "response": response,
        "assistant_content": assistant_content,
        "usage": response.get("usage"),
        "fallback_used": fallback_used,
        "fallback_mode": fallback_mode,
        "image_input_mode": image_input_mode,
        "localized_image_urls": localized_image_urls,
        "failed_localized_image_urls": failed_localized_image_urls,
        "multimodal_error_body": multimodal_error_body,
    }


def run_one(
    *,
    prompt_path: Path,
    input_dir: Path,
    output_json_dir: Path,
    output_md_dir: Path,
    endpoint: str,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int | None,
    timeout: int,
    localize_images: bool,
    image_download_timeout: int,
    max_image_bytes: int,
    overwrite: bool,
    text_only_fallback: bool,
    max_retries: int,
    retry_backoff: float,
    logger: logging.Logger,
) -> dict[str, Any]:
    output_json, output_md = output_paths_for(
        prompt_path, input_dir, output_json_dir, output_md_dir
    )

    if output_json.exists() and output_md.exists() and not overwrite:
        return {
            "status": "skipped",
            "prompt": str(prompt_path),
            "output_json": str(output_json),
            "output_md": str(output_md),
        }

    prompt_payload = load_json(prompt_path)
    messages = prompt_payload.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"Expected 'messages' list in {prompt_path}")

    fallback_used = False
    fallback_mode = None
    multimodal_error_body = None
    localized_image_urls: list[str] = []
    failed_localized_image_urls: list[str] = []
    original_messages = messages
    image_input_mode = "url"

    if iter_image_urls(messages) and (localize_images or should_prelocalize_messages(messages)):
        try:
            messages, localized_image_urls, failed_localized_image_urls = localize_image_content(
                messages,
                timeout=image_download_timeout,
                max_bytes=max_image_bytes,
            )
            if has_image_content(messages):
                image_input_mode = "localized"
            else:
                fallback_used = True
                fallback_mode = "text_only"
                image_input_mode = "text_only"
        except Exception as error:
            if not text_only_fallback:
                raise RuntimeError(f"{prompt_path}: failed to localize images: {error}") from error
            fallback_used = True
            fallback_mode = "text_only"
            image_input_mode = "text_only"
            messages = remove_image_content(original_messages)

    request_payload = build_request_payload(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    try:
        response = call_with_retries(
            endpoint=endpoint,
            api_key=api_key,
            payload=request_payload,
            timeout=timeout,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
            logger=logger,
            prompt_path=prompt_path,
        )
    except HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        if (
            text_only_fallback
            and prompt_payload.get("image_urls")
            and is_multimodal_download_error(body)
        ):
            multimodal_error_body = body
            if image_input_mode != "localized":
                try:
                    messages, localized_image_urls, failed_localized_image_urls = localize_image_content(
                        original_messages,
                        timeout=image_download_timeout,
                        max_bytes=max_image_bytes,
                    )
                    fallback_used = True
                    if has_image_content(messages):
                        image_input_mode = "localized"
                        fallback_mode = "localized_images"
                    else:
                        image_input_mode = "text_only"
                        fallback_mode = "text_only"
                    request_payload = build_request_payload(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    response = call_with_retries(
                        endpoint=endpoint,
                        api_key=api_key,
                        payload=request_payload,
                        timeout=timeout,
                        max_retries=max_retries,
                        retry_backoff=retry_backoff,
                        logger=logger,
                        prompt_path=prompt_path,
                    )
                except Exception:
                    fallback_used = True
                    fallback_mode = "text_only"
                    image_input_mode = "text_only"
                    messages = remove_image_content(original_messages)
                    request_payload = build_request_payload(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    response = call_with_retries(
                        endpoint=endpoint,
                        api_key=api_key,
                        payload=request_payload,
                        timeout=timeout,
                        max_retries=max_retries,
                        retry_backoff=retry_backoff,
                        logger=logger,
                        prompt_path=prompt_path,
                    )
            else:
                fallback_used = True
                fallback_mode = "text_only"
                image_input_mode = "text_only"
                messages = remove_image_content(original_messages)
                request_payload = build_request_payload(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                response = call_with_retries(
                    endpoint=endpoint,
                    api_key=api_key,
                    payload=request_payload,
                    timeout=timeout,
                    max_retries=max_retries,
                    retry_backoff=retry_backoff,
                    logger=logger,
                    prompt_path=prompt_path,
                )
        else:
            raise RuntimeError(f"{prompt_path}: HTTP {error.code}: {body}") from error

    assistant_content = extract_assistant_content(response)
    result = build_result_record(
        prompt_path=prompt_path,
        prompt_payload=prompt_payload,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=messages,
        response=response,
        assistant_content=assistant_content,
        fallback_used=fallback_used,
        fallback_mode=fallback_mode,
        image_input_mode=image_input_mode,
        localized_image_urls=localized_image_urls,
        failed_localized_image_urls=failed_localized_image_urls,
        multimodal_error_body=multimodal_error_body,
    )

    write_json(output_json, result, overwrite=True)
    write_text(output_md, assistant_content.strip() + "\n", overwrite=True)

    return {
        "status": "succeeded",
        "prompt": str(prompt_path),
        "output_json": str(output_json),
        "output_md": str(output_md),
        "usage": response.get("usage"),
        "image_count": prompt_payload.get("image_count"),
        "fallback_used": fallback_used,
        "fallback_mode": fallback_mode,
        "image_input_mode": image_input_mode,
        "localized_image_count": len(localized_image_urls),
        "failed_localized_image_count": len(failed_localized_image_urls),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch run LLM inference for prompt JSON files."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-json-dir", type=Path, default=DEFAULT_OUTPUT_JSON_DIR)
    parser.add_argument("--output-md-dir", type=Path, default=DEFAULT_OUTPUT_MD_DIR)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N prompts.")
    parser.add_argument(
        "--provider",
        choices=sorted(PROVIDER_ENV_VARS),
        default=DEFAULT_PROVIDER,
        help="Environment configuration provider. Defaults to outer.",
    )
    parser.add_argument("--model", default=None, help="Override the selected provider model.")
    parser.add_argument("--base-url", default=None, help="Override the selected provider URL.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="Retries after transient API failures. Defaults to 3.",
    )
    parser.add_argument(
        "--retry-backoff",
        type=float,
        default=DEFAULT_RETRY_BACKOFF,
        help="Initial retry delay in seconds; each retry doubles it. Defaults to 2.",
    )
    parser.add_argument(
        "--localize-images",
        action="store_true",
        help="Download image URLs locally and send them as base64 data URLs.",
    )
    parser.add_argument(
        "--image-download-timeout",
        type=int,
        default=DEFAULT_IMAGE_DOWNLOAD_TIMEOUT,
        help="Timeout in seconds for downloading each image when localizing images.",
    )
    parser.add_argument(
        "--max-image-bytes",
        type=int,
        default=DEFAULT_MAX_IMAGE_BYTES,
        help="Maximum bytes allowed for each downloaded image.",
    )
    parser.add_argument(
        "--no-text-only-fallback",
        action="store_true",
        help="Disable text-only retry when multimodal files cannot be downloaded.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG_FILE)
    args = parser.parse_args()

    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.max_retries < 0:
        raise ValueError("--max-retries must be >= 0")
    if args.retry_backoff < 0:
        raise ValueError("--retry-backoff must be >= 0")

    load_default_env()
    logger = setup_logger(args.log_file)

    api_key, base_url, model = resolve_provider_config(
        provider=args.provider,
        model_override=args.model,
        base_url_override=args.base_url,
    )

    endpoint = resolve_endpoint(base_url)
    prompt_files = iter_prompt_files(args.input_dir)
    if args.limit is not None:
        prompt_files = prompt_files[: args.limit]

    logger.info(
        "Start batch inference provider=%s input_dir=%s output_json_dir=%s output_md_dir=%s prompts=%s workers=%s model=%s endpoint=%s timeout=%s max_retries=%s retry_backoff=%s overwrite=%s",
        args.provider,
        args.input_dir,
        args.output_json_dir,
        args.output_md_dir,
        len(prompt_files),
        args.workers,
        model,
        endpoint,
        args.timeout,
        args.max_retries,
        args.retry_backoff,
        args.overwrite,
    )

    succeeded = 0
    skipped = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                run_one,
                prompt_path=prompt_path,
                input_dir=args.input_dir,
                output_json_dir=args.output_json_dir,
                output_md_dir=args.output_md_dir,
                endpoint=endpoint,
                api_key=api_key,
                model=model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                localize_images=args.localize_images,
                image_download_timeout=args.image_download_timeout,
                max_image_bytes=args.max_image_bytes,
                overwrite=args.overwrite,
                text_only_fallback=not args.no_text_only_fallback,
                max_retries=args.max_retries,
                retry_backoff=args.retry_backoff,
                logger=logger,
            )
            for prompt_path in prompt_files
        ]

        for future in as_completed(futures):
            try:
                result = future.result()
                if result["status"] == "succeeded":
                    succeeded += 1
                    logger.info(
                        "Inference succeeded prompt=%s output_json=%s output_md=%s image_count=%s fallback_used=%s usage=%s",
                        result["prompt"],
                        result["output_json"],
                        result["output_md"],
                        result.get("image_count"),
                        {
                            "used": result.get("fallback_used"),
                            "mode": result.get("fallback_mode"),
                            "image_input_mode": result.get("image_input_mode"),
                            "localized_image_count": result.get("localized_image_count"),
                            "failed_localized_image_count": result.get("failed_localized_image_count"),
                        },
                        result.get("usage"),
                    )
                elif result["status"] == "skipped":
                    skipped += 1
                    logger.info(
                        "Skipped existing output prompt=%s output_json=%s output_md=%s",
                        result["prompt"],
                        result["output_json"],
                        result["output_md"],
                    )
            except (
                URLError,
                TimeoutError,
                socket.timeout,
                ssl.SSLError,
                http.client.RemoteDisconnected,
                ConnectionError,
                json.JSONDecodeError,
                ValueError,
                RuntimeError,
            ) as error:
                failed += 1
                logger.exception("Inference failed error=%s", error)

    logger.info("Done succeeded=%s skipped=%s failed=%s", succeeded, skipped, failed)
    print(
        f"Done. succeeded={succeeded}, skipped={skipped}, failed={failed}. Log: {args.log_file}"
    )


if __name__ == "__main__":
    main()

