from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .docx_loader import guess_profession_docx_path
from .exporter import ArtifactExporter
from .jsonl import read_json
from .judge import SampleJudgeAgent
from .models import LINE_KINDS, LineKind, ProfessionArtifacts, ProfessionRow
from .professions import load_professions
from .profile_builder import ProfessionProfileAgent
from .prompt_generation import PromptGenerationAgent


@dataclass(slots=True)
class PipelineOptions:
    input_path: str | None = None
    output_dir: str = "output"
    limit: int | None = None
    profession_ids: set[str] = field(default_factory=set)
    profession_contains: list[str] = field(default_factory=list)
    line_kinds: set[LineKind] = field(default_factory=set)
    rerun: bool = False
    generation_round: int = 1
    write_bundles: bool = True
    use_llm: bool = False
    llm_min_response_len: int | None = None
    llm_min_hooks_response_len: int | None = None
    enable_judgement: bool = False


class RedTeamBatchPipeline:
    def __init__(self, llm_client: object | None = None) -> None:
        self.profile_agent = ProfessionProfileAgent()
        self._llm_client = llm_client
        self.judge_agent = SampleJudgeAgent()
        self.exporter = ArtifactExporter()

    def run(self, options: PipelineOptions) -> dict:
        if not options.use_llm:
            raise ValueError(
                "生成流程仅使用 prompts 目录下主动/被动两套 LLM 提示词，请设置 use_llm=True 并提供 Chat 客户端。"
            )
        if self._llm_client is None:
            raise ValueError("use_llm=True 时必须在 RedTeamBatchPipeline(llm_client=...) 中传入 LLM 客户端。")

        input_path = options.input_path or str(guess_profession_docx_path(Path.cwd()))
        professions = load_professions(input_path)
        professions = self._filter_professions(professions, options)

        skipped_profession_ids: list[str] = []
        global_accepteds = []
        output_dir = Path(options.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        line_kinds_list: list[LineKind] = sorted(options.line_kinds) if options.line_kinds else list(LINE_KINDS)

        self.prompt_agent = PromptGenerationAgent(
            llm_client=self._llm_client,
            use_llm=True,
        )

        for profession in professions:
            if self._should_skip(output_dir, profession, options):
                skipped_profession_ids.append(profession.profession_id)
                continue

            profile = self.profile_agent.build_profession_profile(profession)
            samples = self.prompt_agent.generate_prompts(
                profile,
                line_kinds_list,
                generation_round=options.generation_round,
                min_response_len=options.llm_min_response_len,
                min_hooks_response_len=options.llm_min_hooks_response_len,
            )
            accepted, rejected = self.judge_agent.judge_samples(
                profession=profession,
                profile=profile,
                samples=samples,
                global_accepteds=global_accepteds,
                enable_judgement=options.enable_judgement,
            )
            global_accepteds.extend(accepted)

            self.exporter.export_profession_artifacts(
                output_dir=output_dir,
                artifacts=ProfessionArtifacts(
                    profession=profession,
                    profile=profile,
                    accepted_samples=accepted,
                    rejected_samples=rejected,
                ),
                generation_round=options.generation_round,
                selected_lines=options.line_kinds or None,
                write_bundles=options.write_bundles,
            )

        manifest = self.exporter.build_manifest(
            output_dir=output_dir,
            generation_round=options.generation_round,
            skipped_profession_ids=skipped_profession_ids,
        )
        return manifest.to_dict()

    @staticmethod
    def _filter_professions(
        professions: list[ProfessionRow], options: PipelineOptions
    ) -> list[ProfessionRow]:
        filtered = professions
        if options.profession_ids:
            filtered = [item for item in filtered if item.profession_id in options.profession_ids]
        if options.profession_contains:
            filtered = [
                item
                for item in filtered
                if any(keyword in item.profession_name_norm for keyword in options.profession_contains)
            ]
        if options.limit is not None:
            filtered = filtered[: options.limit]
        return filtered

    @staticmethod
    def _completed_lines_from_status(status: dict) -> set[str]:
        raw = status.get("completed_lines")
        if isinstance(raw, list) and raw:
            return {str(x) for x in raw}
        # 兼容旧版 status：由 risk_family 推断线路完成情况
        cf = status.get("completed_families")
        if not isinstance(cf, list):
            return set()
        fam = {str(x) for x in cf}
        inferred: set[str] = set()
        if "misconduct" in fam:
            inferred.add("active")
        if "discrimination" in fam or "mistreatment" in fam:
            inferred.add("passive")
        return inferred

    @staticmethod
    def _should_skip(output_dir: Path, profession: ProfessionRow, options: PipelineOptions) -> bool:
        if options.rerun:
            return False
        status_path = output_dir / "professions" / profession.profession_id / "status.json"
        if not status_path.exists():
            return False
        status = read_json(status_path)
        if not status:
            return False
        if status.get("generation_round") != options.generation_round:
            return False
        if not options.line_kinds:
            return True
        completed = RedTeamBatchPipeline._completed_lines_from_status(status)
        return options.line_kinds.issubset(completed)


def parse_line_kinds(raw_values: list[str]) -> set[LineKind]:
    allowed = set(LINE_KINDS)
    selected = {value for value in raw_values if value in allowed}
    invalid = [value for value in raw_values if value not in allowed]
    if invalid:
        raise ValueError(f"Unsupported line_kind values: {invalid}")
    return selected
