from __future__ import annotations

from typing import Any

RISK_BRANCH_ROLE_ABUSE = "role_abuse"
RISK_BRANCH_TARGETED_HARM = "targeted_harm"


def line_kind_to_branch(kind: str) -> str:
    if kind == "active":
        return RISK_BRANCH_ROLE_ABUSE
    if kind == "passive":
        return RISK_BRANCH_TARGETED_HARM
    raise ValueError(f"Unknown line_kind: {kind!r}")


def build_profession_bundle(
    *,
    profession: dict[str, Any],
    profile: dict[str, Any],
    accepted_samples: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate profile and accepted prompts into a two-branch JSON structure (flat prompt lists)."""

    def annotate(row: dict[str, Any]) -> dict[str, Any]:
        out = dict(row)
        lk = str(row.get("line_kind", ""))
        out["risk_branch"] = line_kind_to_branch(lk) if lk in ("active", "passive") else ""
        return out

    active_prompts = [annotate(s) for s in accepted_samples if s.get("line_kind") == "active"]
    passive_prompts = [annotate(s) for s in accepted_samples if s.get("line_kind") == "passive"]

    prof_id = profession.get("profession_id") or profile.get("profession_id")
    name_norm = profession.get("profession_name_norm") or profile.get("profession_name")

    return {
        "profession": {
            "profession_id": prof_id,
            "profession_name_raw": profession.get("profession_name_raw"),
            "profession_name_norm": name_norm,
        },
        "explanation": dict(profile),
        "branch_role_abuse": {
            "branch_id": RISK_BRANCH_ROLE_ABUSE,
            "title_zh": "从业者岗位滥用/违法类不良请求（主动线）",
            "line_kind": "active",
            "prompts": active_prompts,
        },
        "branch_targeted_harm": {
            "branch_id": RISK_BRANCH_TARGETED_HARM,
            "title_zh": "外部针对该职业者的污名、歧视与苛待类不良请求（被动线）",
            "line_kind": "passive",
            "prompts": passive_prompts,
        },
    }
