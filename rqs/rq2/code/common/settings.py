"""Paths, model aliases, and experiment constants for RQ2."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


RQ2_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = RQ2_ROOT.parents[1]
WORK_ROOT = PACKAGE_ROOT / "work" / "rq2"
load_dotenv(PACKAGE_ROOT / ".env")


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    paper_name: str
    api_model: str
    api_key: str
    api_url: str


def model_spec(alias: str) -> ModelSpec:
    if alias == "qwen":
        return ModelSpec(
            alias="qwen",
            paper_name="qwen3.7-plus",
            api_model=os.getenv("RQ2_QWEN_MODEL", "qwen3.7-plus").strip(),
            api_key=os.getenv("RQ2_QWEN_API_KEY", "").strip(),
            api_url=os.getenv("RQ2_QWEN_URL", "").strip(),
        )
    if alias == "gemini":
        return ModelSpec(
            alias="gemini",
            paper_name="gemini-3.1-pro-preview",
            api_model=os.getenv(
                "RQ2_GEMINI_MODEL", "gemini-3.1-pro-preview-medium"
            ).strip(),
            api_key=os.getenv("RQ2_GEMINI_API_KEY", "").strip(),
            api_url=os.getenv("RQ2_GEMINI_URL", "").strip(),
        )
    raise ValueError(f"Unsupported model alias: {alias}")


BASELINES = ("visiondroid", "vanilla_mllm", "kuitest")
MODEL_ALIASES = ("qwen", "gemini")
CONDITIONS = ("without_commonsense", "with_commonsense")
SEQUENCE_CONTEXT_BASELINES = {"visiondroid", "kuitest"}
DEFAULT_RECALL_K = 60
DEFAULT_RERANK_K = 40
DEFAULT_ENRICH_K = 30
DEFAULT_INJECTION_K = 10

