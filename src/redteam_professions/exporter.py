from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from .bundle_builder import build_profession_bundle
from .bundle_markdown import bundle_to_markdown
from .jsonl import read_json, read_jsonl, write_json, write_jsonl
from .models import ExportManifest, LineKind, ProfessionArtifacts


class ArtifactExporter:
    def export_profession_artifacts(
        self,
        output_dir: str | Path,
        artifacts: ProfessionArtifacts,
        generation_round: int,
        selected_lines: set[LineKind] | None = None,
        write_bundles: bool = True,
    ) -> None:
        base_dir = Path(output_dir)
        profession_dir = base_dir / "professions" / artifacts.profession.profession_id
        selected_lines = selected_lines or set()

        existing_accepted = read_jsonl(profession_dir / "accepted_samples.jsonl")
        existing_rejected = read_jsonl(profession_dir / "rejected_samples.jsonl")

        accepted_rows = [sample.to_dict() for sample in artifacts.accepted_samples]
        rejected_rows = [sample.to_dict() for sample in artifacts.rejected_samples]

        if selected_lines:
            accepted_rows = self._merge_line_records(existing_accepted, accepted_rows, selected_lines)
            rejected_rows = self._merge_line_records(existing_rejected, rejected_rows, selected_lines)

        write_json(profession_dir / "profession.json", artifacts.profession.to_dict())
        write_json(profession_dir / "profile.json", artifacts.profile.to_dict())
        write_jsonl(profession_dir / "accepted_samples.jsonl", accepted_rows)
        write_jsonl(profession_dir / "rejected_samples.jsonl", rejected_rows)
        completed_lines = sorted({row["line_kind"] for row in accepted_rows + rejected_rows})
        write_json(
            profession_dir / "status.json",
            {
                "profession_id": artifacts.profession.profession_id,
                "profession_name": artifacts.profession.profession_name_norm,
                "generation_round": generation_round,
                "completed_lines": completed_lines,
                "accepted_count": len(accepted_rows),
                "rejected_count": len(rejected_rows),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )

        if write_bundles:
            bundle = build_profession_bundle(
                profession=artifacts.profession.to_dict(),
                profile=artifacts.profile.to_dict(),
                accepted_samples=[row for row in accepted_rows if row.get("accepted")],
            )
            write_json(profession_dir / "bundle.json", bundle)
            md_path = profession_dir / "bundle.md"
            md_path.write_text(bundle_to_markdown(bundle), encoding="utf-8")

    def build_manifest(
        self,
        output_dir: str | Path,
        generation_round: int,
        skipped_profession_ids: list[str],
    ) -> ExportManifest:
        base_dir = Path(output_dir)
        profession_rows: list[dict] = []
        profile_rows: list[dict] = []
        accepted_rows: list[dict] = []
        rejected_rows: list[dict] = []

        bundle_rows: list[dict] = []
        for status_path in sorted((base_dir / "professions").glob("*/status.json")):
            profession_dir = status_path.parent
            profession_rows.append(read_json(profession_dir / "profession.json"))
            profile_rows.append(read_json(profession_dir / "profile.json"))
            accepted_rows.extend(read_jsonl(profession_dir / "accepted_samples.jsonl"))
            rejected_rows.extend(read_jsonl(profession_dir / "rejected_samples.jsonl"))
            bundle_path = profession_dir / "bundle.json"
            if bundle_path.exists():
                bundle_rows.append(read_json(bundle_path))

        write_jsonl(base_dir / "professions.jsonl", profession_rows)
        write_jsonl(base_dir / "profession_profiles.jsonl", profile_rows)
        write_jsonl(base_dir / "redteam_samples.jsonl", accepted_rows)
        write_jsonl(base_dir / "rejected_samples.jsonl", rejected_rows)
        write_jsonl(base_dir / "profession_bundles.jsonl", bundle_rows)

        warnings: list[str] = []
        for profession_id in {row["profession_id"] for row in accepted_rows}:
            line_counts = {
                kind: sum(
                    1
                    for row in accepted_rows
                    if row["profession_id"] == profession_id and row["line_kind"] == kind
                )
                for kind in ("active", "passive")
            }
            if any(count == 0 for count in line_counts.values()):
                warnings.append(f"{profession_id} line coverage imbalance: {line_counts}")

        manifest = ExportManifest(
            output_dir=str(base_dir),
            profession_count=len(profession_rows),
            accepted_sample_count=len(accepted_rows),
            rejected_sample_count=len(rejected_rows),
            generation_round=generation_round,
            skipped_profession_ids=skipped_profession_ids,
            warnings=warnings,
            bundle_count=len(bundle_rows),
        )
        write_json(base_dir / "manifest.json", manifest.to_dict())
        return manifest

    @staticmethod
    def _merge_line_records(
        existing_rows: list[dict], new_rows: list[dict], selected_lines: set[LineKind]
    ) -> list[dict]:
        kept_existing = [row for row in existing_rows if row.get("line_kind") not in selected_lines]
        return kept_existing + new_rows
