from __future__ import annotations

import unittest

from redteam_professions.bundle_markdown import bundle_to_markdown


class BundleMarkdownTests(unittest.TestCase):
    def test_bundle_to_markdown_contains_sections(self) -> None:
        bundle = {
            "profession": {"profession_id": "p1", "profession_name_norm": "测试职业"},
            "explanation": {"profile_summary": "摘要一句", "domain": "general_workforce"},
            "branch_role_abuse": {
                "title_zh": "分支一标题",
                "prompts": [{"prompt_text_raw": "话1"}],
            },
            "branch_targeted_harm": {
                "title_zh": "分支二标题",
                "prompts": [],
            },
        }
        md = bundle_to_markdown(bundle)
        self.assertIn("# 职业红队双分支报告：测试职业", md)
        self.assertIn("## 职业说明", md)
        self.assertIn("摘要一句", md)
        self.assertIn("## 分支一", md)
        self.assertIn("话1", md)
        self.assertIn("## 分支二", md)
        self.assertIn("被动线", md)


if __name__ == "__main__":
    unittest.main()
