"""Immutable domain contracts shared by every search stage."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, Mapping, TypeAlias


Language: TypeAlias = Literal["en", "ko", "mixed"]
EvidenceRole: TypeAlias = Literal["disease", "support", "risk"]
RuleSource: TypeAlias = Literal[
    "user_explicit",
    "default_5km",
    "verified_context",
    "taxonomy",
]


@dataclass(frozen=True)
class GeoPoint:
    latitude: float
    longitude: float

    def __post_init__(self) -> None:
        if not -90.0 <= self.latitude <= 90.0:
            raise ValueError("latitude must be between -90 and 90")
        if not -180.0 <= self.longitude <= 180.0:
            raise ValueError("longitude must be between -180 and 180")


@dataclass(frozen=True)
class RuleProvenance:
    source: RuleSource
    source_span: tuple[int, int] | None
    turn_id: str

    def __post_init__(self) -> None:
        if not self.turn_id:
            raise ValueError("turn_id is required")
        if self.source_span is not None:
            start, end = self.source_span
            if start < 0 or end <= start:
                raise ValueError("source_span must be a non-empty forward range")


@dataclass(frozen=True)
class DistanceRule:
    anchor: GeoPoint
    max_km: float
    provenance: RuleProvenance = field(compare=False)

    def __post_init__(self) -> None:
        if not 0.0 < self.max_km <= 100.0:
            raise ValueError("max_km must be greater than 0 and at most 100")


@dataclass(frozen=True)
class AreaRule:
    area_id: str
    area_kind: Literal["city", "district"]
    display_name: str
    provenance: RuleProvenance = field(compare=False)

    def __post_init__(self) -> None:
        if not self.area_id or not self.display_name:
            raise ValueError("area_id and display_name are required")


GeographyRule: TypeAlias = DistanceRule | AreaRule


@dataclass(frozen=True)
class RequiredAttribute:
    concept_id: str
    terms_en: tuple[str, ...]
    terms_ko: tuple[str, ...]
    provenance: RuleProvenance = field(compare=False)

    def __post_init__(self) -> None:
        if not self.concept_id:
            raise ValueError("concept_id is required")


@dataclass(frozen=True)
class HardEligibility:
    specialty_ids: frozenset[str]
    geography: GeographyRule
    prohibited_facility_ids: frozenset[str]
    prohibited_taxonomy_ids: frozenset[str]
    required_attributes: tuple[RequiredAttribute, ...]


@dataclass(frozen=True)
class SoftPreference:
    concept_id: str
    polarity: Literal["positive", "negative"]
    weight_class: Literal["weak", "normal", "strong"]
    provenance: RuleProvenance = field(compare=False)

    def __post_init__(self) -> None:
        if not self.concept_id:
            raise ValueError("concept_id is required")


@dataclass(frozen=True)
class EvidenceRequirement:
    requirement_id: str
    terms_en: tuple[str, ...]
    terms_ko: tuple[str, ...]
    match_mode: Literal["exact_phrase", "all_terms", "semantic"]
    source_types: frozenset[str]
    support_required: bool
    evidence_role: EvidenceRole = "support"

    def __post_init__(self) -> None:
        if not self.requirement_id:
            raise ValueError("requirement_id is required")
        if not self.source_types:
            raise ValueError("source_types cannot be empty")


@dataclass(frozen=True)
class SearchRules:
    schema_version: str
    original_query: str
    language: Language
    hard: HardEligibility
    soft: tuple[SoftPreference, ...]
    evidence: tuple[EvidenceRequirement, ...]
    provenance: Mapping[str, RuleProvenance]
    rules_hash: str

    def __post_init__(self) -> None:
        if not self.schema_version or not self.original_query.strip():
            raise ValueError("schema_version and original_query are required")
        if len(self.rules_hash) != 64:
            raise ValueError("rules_hash must be a SHA-256 hex digest")
        object.__setattr__(
            self,
            "provenance",
            MappingProxyType(dict(self.provenance)),
        )


@dataclass(frozen=True)
class EligibleScope:
    index_version: str
    rules_hash: str
    facility_bitmap_ref: str
    facility_count: int
    scope_digest: str

    def __post_init__(self) -> None:
        if self.facility_count < 0:
            raise ValueError("facility_count cannot be negative")


@dataclass(frozen=True)
class Candidate:
    facility_id: str
    lexical_rank: int | None = None
    dense_rank: int | None = None
    evidence_rank: int | None = None
    matched_rule_ids: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not self.facility_id:
            raise ValueError("facility_id is required")
        for rank in (self.lexical_rank, self.dense_rank, self.evidence_rank):
            if rank is not None and rank < 1:
                raise ValueError("candidate ranks must be positive")


@dataclass(frozen=True)
class RuleIssue:
    code: Literal[
        "invalid_proposal",
        "invalid_distance",
        "location_required",
        "rejected_control",
        "unknown_hard_requirement",
        "unknown_location",
        "unknown_specialty",
    ]
    field: str
    needs_clarification: bool


@dataclass(frozen=True)
class RuleCompilation:
    rules: SearchRules | None
    issues: tuple[RuleIssue, ...] = ()

    def __post_init__(self) -> None:
        if (self.rules is None) == (not self.issues):
            raise ValueError("compilation must contain rules or issues, not both")
