from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from redteam_professions.pipeline import PipelineOptions, RedTeamBatchPipeline


def build_test_docx(path: Path, paragraphs: list[str]) -> None:
    xml = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>',
    ]
    for paragraph in paragraphs:
        xml.append(f"<w:p><w:r><w:t>{paragraph}</w:t></w:r></w:p>")
    xml.append("</w:body></w:document>")
    with ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", "".join(xml))


class PipelineStubLlm:
    """Deterministic Chat Completions stub: hooks JSON then short question per line kind."""

    def chat(self, messages: list[dict[str, str]], **kwargs: object) -> str:
        user = messages[-1]["content"]
        if "请只输出一个 JSON 对象，形如" in user:
            m = re.search(r"职业名称：\s*(.+?)(?:\r?\n|$)", user)
            name = m.group(1).strip() if m else "职业者"
            kind = "active"
            if "同一线路（被动）" in user:
                kind = "passive"
            utter = f"我是{name}，关于{kind}线路想问一句怎么措辞才像日常闲聊扯皮"
            return json.dumps({"user_utterance": utter}, ensure_ascii=False)
        return (
            '{"criminal_hooks":["刑事风险一","刑事风险二","刑事风险三"],'
            '"civil_risk_hooks":["民事风险一"],"constitutional_angle":"宪法边界对照",'
            '"taboo_generic_phrases":["岗位资源","绩效压力","业务流程","一线履职"]}'
        )


class PipelineTests(unittest.TestCase):
    def test_pipeline_generates_two_lines_per_profession(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "职业测试.docx"
            output_dir = temp_path / "out"
            build_test_docx(input_path, ["中小学校校长", "企业经理"])

            manifest = RedTeamBatchPipeline(llm_client=PipelineStubLlm()).run(
                PipelineOptions(
                    input_path=str(input_path),
                    output_dir=str(output_dir),
                    use_llm=True,
                    enable_judgement=True,
                )
            )

            redteam_rows = self._read_jsonl(output_dir / "redteam_samples.jsonl")
            rejected_rows = self._read_jsonl(output_dir / "rejected_samples.jsonl")
            bundle_rows = self._read_jsonl(output_dir / "profession_bundles.jsonl")
            prof_dirs = [p for p in (output_dir / "professions").iterdir() if p.is_dir()]
            for p in prof_dirs:
                self.assertTrue((p / "bundle.md").exists(), msg=f"missing bundle.md under {p}")
                self.assertFalse((p / "scenario_cards.jsonl").exists())

        self.assertEqual(manifest["profession_count"], 2)
        self.assertEqual(manifest.get("bundle_count"), 2)
        self.assertEqual(len(bundle_rows), 2)
        self._validate_bundles(bundle_rows)
        self.assertEqual(len(redteam_rows), 4)
        self.assertLessEqual(len(rejected_rows), 20)
        kind_set = {row["line_kind"] for row in redteam_rows}
        self.assertEqual(kind_set, {"active", "passive"})

    def test_line_kind_rerun_merges_with_existing_results(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "职业测试.docx"
            output_dir = temp_path / "out"
            build_test_docx(input_path, ["企业经理"])
            pipeline = RedTeamBatchPipeline(llm_client=PipelineStubLlm())

            pipeline.run(
                PipelineOptions(
                    input_path=str(input_path),
                    output_dir=str(output_dir),
                    use_llm=True,
                    enable_judgement=True,
                )
            )
            pipeline.run(
                PipelineOptions(
                    input_path=str(input_path),
                    output_dir=str(output_dir),
                    line_kinds={"active"},
                    rerun=True,
                    generation_round=1,
                    use_llm=True,
                    enable_judgement=True,
                )
            )

            redteam_rows = self._read_jsonl(output_dir / "redteam_samples.jsonl")

        kinds = {row["line_kind"] for row in redteam_rows}
        self.assertEqual(kinds, {"active", "passive"})

    def test_no_bundles_skips_bundle_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "职业测试.docx"
            output_dir = temp_path / "out"
            build_test_docx(input_path, ["企业经理"])
            manifest = RedTeamBatchPipeline(llm_client=PipelineStubLlm()).run(
                PipelineOptions(
                    input_path=str(input_path),
                    output_dir=str(output_dir),
                    write_bundles=False,
                    use_llm=True,
                    enable_judgement=True,
                )
            )
            prof_dirs = [p for p in (output_dir / "professions").iterdir() if p.is_dir()]
            self.assertEqual(len(prof_dirs), 1)
            self.assertFalse((prof_dirs[0] / "bundle.json").exists())
            self.assertFalse((prof_dirs[0] / "bundle.md").exists())
        self.assertEqual(manifest.get("bundle_count"), 0)

    def test_requires_use_llm(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "职业测试.docx"
            output_dir = temp_path / "out"
            build_test_docx(input_path, ["企业经理"])
            with self.assertRaises(ValueError):
                RedTeamBatchPipeline(llm_client=PipelineStubLlm()).run(
                    PipelineOptions(
                        input_path=str(input_path),
                        output_dir=str(output_dir),
                        use_llm=False,
                    )
                )

    def _validate_bundles(self, bundle_rows: list[dict]) -> None:
        for bundle in bundle_rows:
            self.assertIn("explanation", bundle)
            self.assertIn("branch_role_abuse", bundle)
            self.assertIn("branch_targeted_harm", bundle)
            ra = bundle["branch_role_abuse"]
            self.assertEqual(ra["branch_id"], "role_abuse")
            for p in ra["prompts"]:
                self.assertEqual(p["risk_branch"], "role_abuse")
                self.assertEqual(p["line_kind"], "active")
            th = bundle["branch_targeted_harm"]
            self.assertEqual(th["branch_id"], "targeted_harm")
            for p in th["prompts"]:
                self.assertEqual(p["risk_branch"], "targeted_harm")
                self.assertEqual(p["line_kind"], "passive")

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict]:
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]


if __name__ == "__main__":
    unittest.main()
