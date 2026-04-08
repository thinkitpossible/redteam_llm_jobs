"""
Test script: generate 5 red-team samples per profession for 2 random professions.

Usage (PowerShell):
    $env:REDTEAM_LLM_BASE_URL="https://api.duckcoding.ai/v1"
    $env:REDTEAM_LLM_API_KEY="<your-key>"
    $env:REDTEAM_LLM_MODEL="gpt-5-codex"
    python test_run.py
"""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

# Fix Windows GBK console encoding — model responses may contain emoji
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from redteam_professions.llm.client import OpenAICompatibleChatClient
from redteam_professions.llm.config import LlmConfig
from redteam_professions.profile_builder import ProfessionProfileAgent
from redteam_professions.professions import load_professions
from redteam_professions.prompt_generation_llm import generate_prompts_with_llm

# ── two random professions for the test ──────────────────────────────────────
TEST_PROFESSIONS = ["医疗卫生技术人员", "律师"]
DOCX_PATH = ROOT / "test_职业_2.docx"
OUTPUT_DIR = ROOT / "output_test"
SAMPLES_PER_PROFESSION = 5


def create_minimal_docx(path: Path, profession_names: list[str]) -> None:
    """Write a minimal valid DOCX containing one paragraph per profession."""
    ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    paragraphs_xml = "\n".join(
        f'<w:p xmlns:w="{ns}"><w:r><w:t>{name}</w:t></w:r></w:p>'
        for name in profession_names
    )
    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<w:document xmlns:w="{ns}"><w:body>{paragraphs_xml}'
        f'<w:sectPr/></w:body></w:document>'
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml"'
        ' ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        "</Types>"
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1"'
        ' Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"'
        ' Target="word/document.xml"/>'
        "</Relationships>"
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("word/document.xml", document_xml)


def check_llm_connectivity(client: OpenAICompatibleChatClient) -> None:
    reply = client.chat([{"role": "user", "content": "Reply with the single word OK"}], max_tokens=50)
    safe = reply.encode("ascii", errors="replace").decode("ascii")
    print(f"  LLM connectivity check -> {safe[:80]!r}")


def main() -> int:
    config = LlmConfig.from_env()
    print("=" * 60)
    print("Red-team pipeline test")
    print(f"  model    : {config.model}")
    print(f"  base_url : {config.base_url}")
    print(f"  professions: {TEST_PROFESSIONS}")
    print(f"  samples per profession: {SAMPLES_PER_PROFESSION}")
    print("=" * 60)

    if not config.api_key:
        print("ERROR: REDTEAM_LLM_API_KEY is not set.", file=sys.stderr)
        return 1

    # ── create test DOCX ─────────────────────────────────────────────────────
    print(f"\n[1/4] Creating test DOCX → {DOCX_PATH.name}")
    create_minimal_docx(DOCX_PATH, TEST_PROFESSIONS)
    print("  Done.")

    # ── load professions ─────────────────────────────────────────────────────
    print("\n[2/4] Loading professions from DOCX")
    professions = load_professions(DOCX_PATH)
    if not professions:
        print("ERROR: no professions found in DOCX.", file=sys.stderr)
        return 1
    print(f"  Loaded: {[p.profession_name_norm for p in professions]}")

    # ── connect LLM ──────────────────────────────────────────────────────────
    print("\n[3/4] Connecting to LLM")
    client = OpenAICompatibleChatClient(config)
    try:
        check_llm_connectivity(client)
    except Exception as exc:
        print(f"  [warn] connectivity check failed ({exc}), continuing anyway...")

    # ── generate samples ─────────────────────────────────────────────────────
    print(f"\n[4/4] Generating {SAMPLES_PER_PROFESSION} samples per profession")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    profile_agent = ProfessionProfileAgent()
    grand_total = 0

    for profession in professions:
        pname = profession.profession_name_norm
        print(f"\n  ── {pname} (id={profession.profession_id}) ──")
        profile = profile_agent.build_profession_profile(profession)
        print(f"     domain={profile.domain}  authority={profile.authority_level}")

        all_samples = []
        for round_num in range(1, SAMPLES_PER_PROFESSION + 1):
            print(f"     round {round_num}/{SAMPLES_PER_PROFESSION} … ", end="", flush=True)
            try:
                samples = generate_prompts_with_llm(
                    client,
                    profile,
                    ["active"],
                    generation_round=round_num,
                )
                all_samples.extend(samples)
                safe_preview = samples[0].prompt_text_raw[:60].encode("utf-8", errors="replace").decode("utf-8")
                print(f"OK  -> got sample")
            except Exception as exc:
                print(f"FAILED ({exc})")

        out_file = OUTPUT_DIR / f"{profession.profession_id}_samples.jsonl"
        with out_file.open("w", encoding="utf-8") as fh:
            for s in all_samples:
                fh.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")

        print(f"\n     Saved {len(all_samples)} samples → {out_file.name}")
        grand_total += len(all_samples)

        print(f"\n     Full sample list for [{pname}]:")
        for i, s in enumerate(all_samples, 1):
            print(f"       {i}. {s.prompt_text_raw}")

    # ── summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Done.  {len(professions)} professions  ·  {grand_total} samples total")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
