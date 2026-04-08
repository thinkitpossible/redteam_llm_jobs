from __future__ import annotations

from .models import LINE_KINDS, LineKind, ProfessionProfile, RedTeamSample


class PromptGenerationAgent:
    def __init__(self, llm_client: object | None = None, use_llm: bool = False) -> None:
        self._llm_client = llm_client
        self._use_llm = use_llm

    def generate_prompts(
        self,
        profile: ProfessionProfile,
        line_kinds: list[LineKind],
        generation_round: int = 1,
        *,
        min_response_len: int | None = None,
        min_hooks_response_len: int | None = None,
    ) -> list[RedTeamSample]:
        if not self._use_llm or self._llm_client is None:
            raise ValueError(
                "仅支持通过 prompts 目录下八份 LLM 提示词（主动/被动各一套）生成样本：请使用 use_llm=True 并提供 llm_client。"
            )
        from .prompt_generation_llm import generate_prompts_with_llm

        kinds = line_kinds or list(LINE_KINDS)
        return generate_prompts_with_llm(
            self._llm_client,
            profile,
            kinds,
            generation_round=generation_round,
            min_response_len=min_response_len,
            min_hooks_response_len=min_hooks_response_len,
        )
