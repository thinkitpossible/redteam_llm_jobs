from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from redteam_professions.job_spec import job_dict_to_arg_defaults, load_job_file


class JobSpecTests(unittest.TestCase):
    def test_load_job_file_requires_object(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "bad.json"
            p.write_text("[1,2,3]", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_job_file(p)

    def test_job_dict_to_arg_defaults_maps_aliases_and_llm(self) -> None:
        job = {
            "input": "in.docx",
            "output_dir": "outdir",
            "limit": 3,
            "profession_ids": ["p1", "p2"],
            "line_kinds": ["active"],
            "rerun": True,
            "write_bundles": False,
            "use_llm": True,
            "llm_min_response_len": 20,
            "llm_min_hooks_response_len": 50,
            "enable_judgement": True,
        }
        d = job_dict_to_arg_defaults(job)
        self.assertEqual(d["input_path"], "in.docx")
        self.assertEqual(d["output_dir"], "outdir")
        self.assertEqual(d["limit"], 3)
        self.assertEqual(d["profession_id"], ["p1", "p2"])
        self.assertEqual(d["line_kind"], ["active"])
        self.assertTrue(d["rerun"])
        self.assertTrue(d["no_bundles"])
        self.assertTrue(d["use_llm"])
        self.assertEqual(d["llm_min_response_len"], 20)
        self.assertEqual(d["llm_min_hooks_response_len"], 50)
        self.assertTrue(d["enable_judgement"])

    def test_example_job_file_loads(self) -> None:
        root = Path(__file__).resolve().parent.parent
        path = root / "examples" / "redteam_job.example.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        job_dict_to_arg_defaults(data)

    def test_legacy_risk_families_maps_to_line_kind(self) -> None:
        d = job_dict_to_arg_defaults({"risk_families": ["misconduct", "discrimination"]})
        self.assertEqual(sorted(d["line_kind"]), ["active", "passive"])


if __name__ == "__main__":
    unittest.main()
