from __future__ import annotations

from pathlib import Path

from .docx_loader import load_docx_paragraphs
from .models import ProfessionRow
from .text_utils import normalize_profession_name, stable_id


def load_professions(input_path: str | Path) -> list[ProfessionRow]:
    path = Path(input_path)
    professions: list[ProfessionRow] = []
    for index, paragraph in enumerate(load_docx_paragraphs(path), start=1):
        normalized = normalize_profession_name(paragraph)
        if not normalized:
            continue
        professions.append(
            ProfessionRow(
                profession_id=stable_id("prof", f"{index}:{normalized}"),
                profession_name_raw=paragraph,
                profession_name_norm=normalized,
                source_path=str(path),
                source_index=index,
            )
        )
    return professions
