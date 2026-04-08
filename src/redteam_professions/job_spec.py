from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_job_file(path: str | Path) -> dict[str, Any]:
    raw = Path(path).read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("Job file must be a JSON object at the top level.")
    return data


def _legacy_risk_to_line_kinds(rf: str | list[str]) -> list[str]:
    """Map deprecated risk_family values to line_kind for job/CLI compatibility."""
    items = [rf] if isinstance(rf, str) else [str(x) for x in rf]
    out: list[str] = []
    for x in items:
        if x == "misconduct" and "active" not in out:
            out.append("active")
        elif x in ("discrimination", "mistreatment") and "passive" not in out:
            out.append("passive")
    return out


def job_dict_to_arg_defaults(job: dict[str, Any]) -> dict[str, Any]:
    """Map job JSON keys to argparse destinations for set_defaults (CLI still overrides)."""
    out: dict[str, Any] = {}

    if "input_path" in job:
        out["input_path"] = job["input_path"]
    elif "input" in job:
        out["input_path"] = job["input"]

    if "output_dir" in job:
        out["output_dir"] = job["output_dir"]

    if "limit" in job and job["limit"] is not None:
        out["limit"] = int(job["limit"])

    pid = job.get("profession_id", job.get("profession_ids"))
    if pid is not None:
        if isinstance(pid, str):
            out["profession_id"] = [pid]
        elif isinstance(pid, list):
            out["profession_id"] = [str(x) for x in pid]
        else:
            raise TypeError("profession_id / profession_ids must be a string or list of strings.")

    pc = job.get("profession_contains")
    if pc is not None:
        if isinstance(pc, str):
            out["profession_contains"] = [pc]
        elif isinstance(pc, list):
            out["profession_contains"] = [str(x) for x in pc]
        else:
            raise TypeError("profession_contains must be a string or list of strings.")

    lk = job.get("line_kind", job.get("line_kinds"))
    if lk is not None:
        if isinstance(lk, str):
            out["line_kind"] = [lk]
        elif isinstance(lk, list):
            out["line_kind"] = [str(x) for x in lk]
        else:
            raise TypeError("line_kind / line_kinds must be a string or list of strings.")
    else:
        rf = job.get("risk_family", job.get("risk_families"))
        if rf is not None:
            mapped = _legacy_risk_to_line_kinds(rf)
            if mapped:
                out["line_kind"] = mapped

    if "generation_round" in job:
        out["generation_round"] = int(job["generation_round"])

    if "rerun" in job:
        out["rerun"] = bool(job["rerun"])

    if "no_bundles" in job:
        out["no_bundles"] = bool(job["no_bundles"])
    elif "write_bundles" in job:
        out["no_bundles"] = not bool(job["write_bundles"])

    if "use_llm" in job:
        out["use_llm"] = bool(job["use_llm"])

    if "llm_min_response_len" in job and job["llm_min_response_len"] is not None:
        out["llm_min_response_len"] = int(job["llm_min_response_len"])

    if "llm_min_hooks_response_len" in job and job["llm_min_hooks_response_len"] is not None:
        out["llm_min_hooks_response_len"] = int(job["llm_min_hooks_response_len"])

    if "enable_judgement" in job:
        out["enable_judgement"] = bool(job["enable_judgement"])

    return out
