"""Compatibility import for the shared resized-image RQ2 client."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.openai_compatible_client import ClientAPIError, ResizedLLMInterface

LLMInterface = ResizedLLMInterface
__all__ = ["ClientAPIError", "LLMInterface"]
