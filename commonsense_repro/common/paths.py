"""Canonical paths used by the standalone reproduction package."""

from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PACKAGE_ROOT / "data"
WORK_ROOT = PACKAGE_ROOT / "work"
PROMPT_ROOT = PACKAGE_ROOT / "prompts"
CONFIG_ROOT = PACKAGE_ROOT / "config"


def resolve_path(path: Path | str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PACKAGE_ROOT / candidate


