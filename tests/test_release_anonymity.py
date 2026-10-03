from __future__ import annotations

import re
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]

TEXT_SUFFIXES = {
    ".csv",
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".txt",
}

FORBIDDEN_PATH_PATTERNS = (
    re.compile(r"[A-Za-z]:\\\\"),
    re.compile(r"/Users/[^/\s\"']+/", re.IGNORECASE),
    re.compile(r"/home/[^/\s\"']+/", re.IGNORECASE),
    re.compile(r"lab_experiment", re.IGNORECASE),
    re.compile(r"NovaOracle", re.IGNORECASE),
)


def release_text_files() -> list[Path]:
    roots = (
        PACKAGE_ROOT / "commonsense_repro",
        PACKAGE_ROOT / "config",
        PACKAGE_ROOT / "data",
        PACKAGE_ROOT / "prompts",
        PACKAGE_ROOT / "rqs",
    )
    files: list[Path] = [PACKAGE_ROOT / "README.md", PACKAGE_ROOT / "reproduce.py"]
    for root in roots:
        files.extend(
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix.lower() in TEXT_SUFFIXES
            and "__pycache__" not in path.parts
        )
    return files


def test_release_contains_no_private_credential_files() -> None:
    forbidden_names = {".env", "credentials.json"}
    forbidden_suffixes = {".key", ".p12", ".pem", ".pfx"}
    leaked = [
        path.relative_to(PACKAGE_ROOT)
        for path in PACKAGE_ROOT.rglob("*")
        if path.is_file()
        and ".git" not in path.parts
        and (path.name.lower() in forbidden_names or path.suffix.lower() in forbidden_suffixes)
    ]
    assert not leaked, leaked


def test_release_text_contains_no_local_absolute_paths_or_workspace_identity() -> None:
    violations: list[str] = []
    for path in release_text_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in FORBIDDEN_PATH_PATTERNS:
            if pattern.search(text):
                violations.append(f"{path.relative_to(PACKAGE_ROOT)}: {pattern.pattern}")
    assert not violations, violations[:20]
