from __future__ import annotations

import unittest

from redteam_professions.bundle_builder import (
    RISK_BRANCH_ROLE_ABUSE,
    RISK_BRANCH_TARGETED_HARM,
    build_profession_bundle,
    line_kind_to_branch,
)


def _sample(**kwargs: object) -> dict:
    base = {
        "sample_id": "s",
        "profession_id": "prof_test",
        "profession_name": "测试职业",
        "line_kind": "active",
        "prompt_text_raw": "x",
        "attack_intent": "a",
        "target_group_or_object": "y",
        "illegality_flag": True,
        "discrimination_flag": False,
        "mistreatment_flag": False,
        "severity": "medium",
        "specificity_score": 0.0,
        "novelty_score": 0.0,
        "relevance_score": 0.0,
        "category_confidence": 0.0,
        "generation_round": 1,
        "accepted": True,
        "reject_reason": "",
        "normalized_summary": "n",
    }
    base.update(kwargs)
    return base


class BundleBuilderTests(unittest.TestCase):
    def test_line_kind_to_branch(self) -> None:
        self.assertEqual(line_kind_to_branch("active"), RISK_BRANCH_ROLE_ABUSE)
        self.assertEqual(line_kind_to_branch("passive"), RISK_BRANCH_TARGETED_HARM)
        with self.assertRaises(ValueError):
            line_kind_to_branch("unknown")

    def test_build_profession_bundle_flat_prompts(self) -> None:
        profession = {
            "profession_id": "prof_test",
            "profession_name_raw": "测试职业",
            "profession_name_norm": "测试职业",
        }
        profile = {"profession_id": "prof_test", "profession_name": "测试职业", "profile_summary": "摘要"}
        samples = [
            _sample(
                sample_id="s1",
                line_kind="active",
                prompt_text_raw="prompt a",
                normalized_summary="n1",
            ),
            _sample(
                sample_id="s2",
                line_kind="passive",
                prompt_text_raw="prompt b",
                illegality_flag=False,
                discrimination_flag=True,
                mistreatment_flag=True,
                normalized_summary="n2",
            ),
        ]
        bundle = build_profession_bundle(
            profession=profession,
            profile=profile,
            accepted_samples=samples,
        )
        self.assertEqual(bundle["explanation"]["profile_summary"], "摘要")
        ra = bundle["branch_role_abuse"]["prompts"]
        self.assertEqual(len(ra), 1)
        self.assertEqual(ra[0]["risk_branch"], RISK_BRANCH_ROLE_ABUSE)
        self.assertEqual(ra[0]["line_kind"], "active")
        self.assertEqual(ra[0]["prompt_text_raw"], "prompt a")
        th = bundle["branch_targeted_harm"]["prompts"]
        self.assertEqual(len(th), 1)
        self.assertEqual(th[0]["risk_branch"], RISK_BRANCH_TARGETED_HARM)
        self.assertEqual(th[0]["line_kind"], "passive")

    def test_exporter_style_filter_omits_rejected(self) -> None:
        """build_profession_bundle expects only accepted rows (exporter filters before calling)."""
        profession = {"profession_id": "p", "profession_name_norm": "X"}
        profile = {"profession_id": "p", "profession_name": "X"}
        rows = [_sample(sample_id="rej", profession_id="p", profession_name="X", accepted=False)]
        accepted_only = [r for r in rows if r.get("accepted")]
        bundle = build_profession_bundle(profession=profession, profile=profile, accepted_samples=accepted_only)
        self.assertEqual(bundle["branch_role_abuse"]["prompts"], [])


if __name__ == "__main__":
    unittest.main()
