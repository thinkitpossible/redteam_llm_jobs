from __future__ import annotations

import json
from typing import Any

from .config import LlmConfig


def extract_message_content(response_json: dict[str, Any]) -> str:
    """Parse OpenAI-compatible chat completion JSON; return assistant text or raise."""
    choices = response_json.get("choices")
    if not choices:
        raise ValueError("LLM response missing choices")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if content is None or not str(content).strip():
        raise ValueError("LLM response missing message.content")
    return str(content).strip()


class OpenAICompatibleChatClient:
    """Minimal POST /v1/chat/completions client (OpenAI-compatible servers)."""

    def __init__(self, config: LlmConfig) -> None:
        self._config = config
        try:
            import httpx  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover - exercised when extra not installed
            raise ImportError(
                "LLM mode requires the 'httpx' package. Install with: pip install 'redteam-professions[llm]'"
            ) from exc
        self._httpx = httpx

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.65,
        max_tokens: int = 800,
    ) -> str:
        if not self._config.api_key:
            raise ValueError("REDTEAM_LLM_API_KEY is not set; refusing to call remote API.")
        url = f"{self._config.base_url}/chat/completions"
        # Drop system-role messages when the proxy/model filters them
        effective_messages: list[dict[str, str]] = (
            [m for m in messages if m.get("role") != "system"]
            if self._config.no_system_prompt
            else list(messages)
        )
        if not effective_messages:
            effective_messages = list(messages)
        payload: dict[str, Any] = {
            "model": model or self._config.model,
            "messages": effective_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self._config.api_key}",
            "Content-Type": "application/json",
        }
        with self._httpx.Client(timeout=self._config.timeout_sec) as client:
            response = client.post(url, headers=headers, content=json.dumps(payload, ensure_ascii=False))
            response.raise_for_status()
            data = response.json()
        text = extract_message_content(data)
        return _strip_wrapping_quotes(_strip_code_fence(text))


def _strip_code_fence(text: str) -> str:
    s = text.strip()
    if s.startswith("```"):
        lines = s.split("\n")
        if len(lines) >= 2 and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    return s


def _strip_wrapping_quotes(text: str) -> str:
    s = text.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
        return s[1:-1].strip()
    return s
