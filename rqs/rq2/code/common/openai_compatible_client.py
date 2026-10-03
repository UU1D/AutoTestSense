"""Shared OpenAI-compatible multimodal HTTP client for the RQ2 baselines."""

from __future__ import annotations

import base64
import io
import os
import time
from pathlib import Path

import requests
from PIL import Image


class ClientAPIError(RuntimeError):
    """A deterministic invalid-request or authentication failure."""


class LLMInterface:
    def __init__(
        self,
        api_key,
        model,
        url,
        temperature=0,
        timeout=360,
        max_retries=5,
        enable_thinking=False,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.url = url
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.enable_thinking = enable_thinking

    @staticmethod
    def encode_image(image_path: str) -> str:
        return base64.b64encode(Path(image_path).read_bytes()).decode("ascii")

    def get_response_from_lm(self, images, prompt, return_json=False):
        content = [{"type": "text", "text": prompt}]
        for image in images:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{self.encode_image(image)}"
                    },
                }
            )
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": self.temperature,
            "enable_thinking": self.enable_thinking,
        }
        if return_json:
            payload["response_format"] = {"type": "json_object"}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Cache-Control": "no-cache",
        }
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(
                    self.url, headers=headers, json=payload, timeout=self.timeout
                )
                if response.status_code in {400, 401, 403, 404}:
                    raise ClientAPIError(
                        f"API client error {response.status_code}: {response.text}"
                    )
                response.raise_for_status()
                body = response.json()
                choices = body.get("choices")
                if not choices:
                    raise RuntimeError(f"API returned no choices: {body}")
                output = choices[0]["message"]["content"].strip()
                usage = body.get("usage") or {}
                prompt_tokens = int(usage.get("prompt_tokens", 0))
                completion_tokens = int(usage.get("completion_tokens", 0))
                total_tokens = int(
                    usage.get("total_tokens", prompt_tokens + completion_tokens)
                )
                return output, total_tokens, prompt_tokens, completion_tokens
            except ClientAPIError:
                raise
            except (requests.RequestException, ValueError, KeyError, RuntimeError) as error:
                last_error = error
                if attempt >= self.max_retries:
                    break
                status = getattr(getattr(error, "response", None), "status_code", None)
                base_delay = 20 if status == 429 else 5
                delay = min(60, base_delay * (2 ** (attempt - 1)))
                print(
                    f"[RQ2 API retry {attempt}/{self.max_retries}] {error}; "
                    f"waiting {delay}s",
                    flush=True,
                )
                time.sleep(delay)
        raise TimeoutError(
            f"RQ2 API failed after {self.max_retries} attempts: {last_error}"
        )


class ResizedLLMInterface(LLMInterface):
    """KuiTest-compatible client that bounds each step image to 1080 pixels."""

    @staticmethod
    def encode_image(image_path: str) -> str:
        with Image.open(image_path) as image:
            if image.mode != "RGB":
                image = image.convert("RGB")
            image.thumbnail((1080, 1080))
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=80)
        return base64.b64encode(buffer.getvalue()).decode("ascii")
