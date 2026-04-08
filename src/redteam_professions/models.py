from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal


LineKind = Literal["active", "passive"]

LINE_KINDS: tuple[LineKind, ...] = ("active", "passive")


@dataclass(slots=True)
class ProfessionRow:
    profession_id: str
    profession_name_raw: str
    profession_name_norm: str
    source_path: str
    source_index: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class ProfessionProfile:
    profession_id: str
    profession_name: str
    domain: str
    authority_level: str
    resource_access: list[str]
    target_population: list[str]
    workplace_context: list[str]
    social_sensitivity: list[str]
    victimization_surface: list[str]
    profile_tags: list[str] = field(default_factory=list)
    profile_summary: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class RedTeamSample:
    sample_id: str
    profession_id: str
    profession_name: str
    line_kind: LineKind
    prompt_text_raw: str
    attack_intent: str
    target_group_or_object: str
    illegality_flag: bool
    discrimination_flag: bool
    mistreatment_flag: bool
    severity: str
    specificity_score: float
    novelty_score: float
    relevance_score: float
    category_confidence: float
    generation_round: int
    accepted: bool
    reject_reason: str
    normalized_summary: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class ProfessionArtifacts:
    profession: ProfessionRow
    profile: ProfessionProfile
    accepted_samples: list[RedTeamSample]
    rejected_samples: list[RedTeamSample]


@dataclass(slots=True)
class ExportManifest:
    output_dir: str
    profession_count: int
    accepted_sample_count: int
    rejected_sample_count: int
    generation_round: int
    skipped_profession_ids: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    bundle_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)
