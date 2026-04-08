from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LlmConfig:
    base_url: str
    api_key: str
    model: str
    timeout_sec: float
    no_system_prompt: bool = False
    """When True, system-role messages are dropped before sending.
    Useful for model proxies that apply content filters on system prompts.
    Set via env: REDTEAM_LLM_NO_SYSTEM_PROMPT=1"""

    @classmethod
    def from_env(cls) -> LlmConfig:
        base = os.environ.get("REDTEAM_LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        key = os.environ.get("REDTEAM_LLM_API_KEY", "").strip()
        model = os.environ.get("REDTEAM_LLM_MODEL", "gpt-4o-mini").strip()
        timeout_raw = os.environ.get("REDTEAM_LLM_TIMEOUT_SEC", "120")
        try:
            timeout_sec = float(timeout_raw)
        except ValueError:
            timeout_sec = 120.0
        no_sys_raw = os.environ.get("REDTEAM_LLM_NO_SYSTEM_PROMPT", "").strip().lower()
        no_system_prompt = no_sys_raw in ("1", "true", "yes")
        return cls(
            base_url=base,
            api_key=key,
            model=model,
            timeout_sec=timeout_sec,
            no_system_prompt=no_system_prompt,
        )
