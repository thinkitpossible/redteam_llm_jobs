from __future__ import annotations

from typing import Any


def _md_escape_line(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\n", " ").strip()


def _block_prompts(prompts: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for i, p in enumerate(prompts, start=1):
        raw = p.get("prompt_text_raw", "")
        lines.append(f"{i}. {_md_escape_line(raw)}")
    return "\n".join(lines) if lines else "_（暂无通过裁判的样本）_"


def bundle_to_markdown(bundle: dict[str, Any]) -> str:
    """Render bundle JSON to a fixed-outline Markdown report."""
    prof = bundle.get("profession", {})
    name = prof.get("profession_name_norm") or prof.get("profession_name_raw") or ""
    pid = prof.get("profession_id", "")
    expl = bundle.get("explanation", {})

    lines: list[str] = [
        f"# 职业红队双分支报告：{name}",
        "",
        f"- **profession_id**：`{pid}`",
        "",
        "## 职业说明（画像）",
        "",
    ]

    summary = expl.get("profile_summary")
    if summary:
        lines.append(str(summary))
        lines.append("")

    for key in (
        "domain",
        "authority_level",
        "resource_access",
        "target_population",
        "workplace_context",
        "social_sensitivity",
        "victimization_surface",
        "profile_tags",
    ):
        if key not in expl:
            continue
        val = expl[key]
        if isinstance(val, list):
            lines.append(f"- **{key}**：{', '.join(str(x) for x in val)}")
        else:
            lines.append(f"- **{key}**：{val}")
    lines.append("")

    bra = bundle.get("branch_role_abuse", {})
    lines.append(f"## 分支一（主动线）：{bra.get('title_zh', '从业者岗位滥用/违法类不良请求')}")
    lines.append("")
    lines.append("**样本话术：**")
    lines.append("")
    lines.append(_block_prompts(bra.get("prompts", [])))
    lines.append("")

    bth = bundle.get("branch_targeted_harm", {})
    lines.append(f"## 分支二（被动线）：{bth.get('title_zh', '外部针对该职业者的污名与苛待类不良请求')}")
    lines.append("")
    lines.append("**样本话术：**")
    lines.append("")
    lines.append(_block_prompts(bth.get("prompts", [])))

    return "\n".join(lines).rstrip() + "\n"
