"""Compatibility import for the shared RQ2 multimodal client."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.openai_compatible_client import ClientAPIError, LLMInterface

__all__ = ["ClientAPIError", "LLMInterface"]
