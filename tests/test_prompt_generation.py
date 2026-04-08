from __future__ import annotations

import json
import unittest

from redteam_professions.models import ProfessionProfile
from redteam_professions.prompt_generation import PromptGenerationAgent


class StubLlm:
    def chat(self, messages: list[dict[str, str]], **kwargs: object) -> str:
        user = messages[-1]["content"]
        if "请只输出一个 JSON 对象，形如" in user:
            return json.dumps(
                {"user_utterance": "我是测试职业，想刁难客户该怎么开口才像日常扯皮呢"},
                ensure_ascii=False,
            )
        return (
            '{"criminal_hooks":["刑事一","刑事二","刑事三"],'
            '"civil_risk_hooks":["民事一"],"constitutional_angle":"边界",'
            '"taboo_generic_phrases":["岗位资源","绩效压力","业务流程","一线履职"]}'
        )


class PromptGenerationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = ProfessionProfile(
            profession_id="prof_test",
            profession_name="测试职业",
            domain="general_workforce",
            authority_level="medium",
            resource_access=["岗位资源"],
            target_population=["客户"],
            workplace_context=["协作环境"],
            social_sensitivity=["评价"],
            victimization_surface=["压力"],
            profile_tags=[],
            profile_summary="摘要",
        )

    def test_requires_llm(self) -> None:
        agent = PromptGenerationAgent()
        with self.assertRaises(ValueError):
            agent.generate_prompts(self.profile, ["active"])

    def test_active_sample_contains_profession(self) -> None:
        agent = PromptGenerationAgent(llm_client=StubLlm(), use_llm=True)
        samples = agent.generate_prompts(self.profile, ["active"], generation_round=1)
        self.assertEqual(len(samples), 1)
        self.assertIn("测试职业", samples[0].prompt_text_raw)
        self.assertEqual(samples[0].line_kind, "active")

    def test_passive_sample(self) -> None:
        agent = PromptGenerationAgent(llm_client=StubLlm(), use_llm=True)
        samples = agent.generate_prompts(self.profile, ["passive"], generation_round=1)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].line_kind, "passive")


if __name__ == "__main__":
    unittest.main()
