from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile


WORD_NAMESPACE = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def load_docx_paragraphs(path: str | Path) -> list[str]:
    docx_path = Path(path)
    with ZipFile(docx_path) as archive:
        document_xml = archive.read("word/document.xml")
    root = ET.fromstring(document_xml)
    paragraphs: list[str] = []
    for paragraph in root.findall(".//w:body/w:p", WORD_NAMESPACE):
        text_parts = [node.text or "" for node in paragraph.findall(".//w:t", WORD_NAMESPACE)]
        paragraph_text = "".join(text_parts).strip()
        if paragraph_text:
            paragraphs.append(paragraph_text)
    return paragraphs


def guess_profession_docx_path(directory: str | Path) -> Path:
    directory_path = Path(directory)
    candidates = list(directory_path.glob("*.docx"))
    if not candidates:
        raise FileNotFoundError(f"No .docx files found under {directory_path}")

    prioritized = [candidate for candidate in candidates if "职业" in candidate.stem]
    ranked = prioritized or candidates
    ranked.sort(key=lambda item: len(load_docx_paragraphs(item)), reverse=True)
    return ranked[0]
