from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from redteam_professions.docx_loader import load_docx_paragraphs
from redteam_professions.professions import load_professions


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


class DocxLoaderTests(unittest.TestCase):
    def test_load_docx_paragraphs_and_professions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            docx_path = Path(temp_dir) / "职业测试.docx"
            build_test_docx(docx_path, ["职业A", "职业B", "", "职业C"])

            paragraphs = load_docx_paragraphs(docx_path)
            professions = load_professions(docx_path)

        self.assertEqual(paragraphs, ["职业A", "职业B", "职业C"])
        self.assertEqual(len(professions), 3)
        self.assertEqual(len({item.profession_id for item in professions}), 3)
        self.assertEqual(professions[0].profession_name_norm, "职业A")


if __name__ == "__main__":
    unittest.main()
