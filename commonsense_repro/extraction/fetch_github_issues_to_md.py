"""Fetch one GitHub issue URL and save its title plus first comment as Markdown.

In GitHub's issue page, the first visible comment under the title is the issue
body. The GitHub REST API exposes that content as the issue's ``body`` field.

Set GITHUB_TOKEN in the environment for authenticated GitHub API requests.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from commonsense_repro.common.paths import PACKAGE_ROOT


GITHUB_API_VERSION = "2022-11-28"


def load_dotenv_files(paths: list[Path]) -> None:
    for path in paths:
        if not path.exists():
            continue

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
    load_dotenv_files(
        [
            Path.cwd() / ".env",
            PACKAGE_ROOT / ".env",
        ]
    )


def parse_issue_url(url: str) -> tuple[str, str, int]:
    parsed = urlparse(url)
    parts = [part for part in parsed.path.split("/") if part]

    if parsed.netloc.lower() != "github.com" or len(parts) < 4 or parts[2] != "issues":
        raise ValueError(f"Unsupported GitHub issue URL: {url}")

    return parts[0], parts[1], int(parts[3])


def github_request(api_url: str, token: str | None, timeout: int) -> dict[str, Any]:
    headers = {
        "Accept": "application/vnd.github.raw+json",
        "User-Agent": "commonsense-issue-fetcher",
        "X-GitHub-Api-Version": GITHUB_API_VERSION,
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    request = Request(api_url, headers=headers)
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_issue_from_url(url: str, token: str | None = None, timeout: int = 30) -> dict[str, Any]:
    owner, repo, issue_number = parse_issue_url(url)
    api_url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}"
    return github_request(api_url, token, timeout)


def markdown_for_issue(url: str, issue: dict[str, Any]) -> str:
    title = issue.get("title") or ""
    first_comment = issue.get("body") or ""

    return (
        f"# {title}\n\n"
        f"Source URL: {url}\n\n"
        # "## First Comment\n\n"
        f"{first_comment.strip() or '_No first comment content._'}\n"
    )


def fetch_issue_markdown(
    url: str, token: str | None = None, timeout: int = 30
) -> str:
    issue = fetch_issue_from_url(url, token=token, timeout=timeout)
    return markdown_for_issue(url, issue)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch one GitHub issue URL and save title plus first comment as Markdown."
    )
    parser.add_argument("url", help="GitHub issue URL, for example https://github.com/owner/repo/issues/1")
    parser.add_argument("--output", type=Path, help="Markdown output path. Prints to stdout if omitted.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()
    load_default_env()

    if args.output and args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} already exists. Use --overwrite to replace it.")

    markdown = fetch_issue_markdown(
        args.url,
        token=os.environ.get("GITHUB_TOKEN"),
        timeout=args.timeout,
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        print(markdown)


if __name__ == "__main__":
    main()

