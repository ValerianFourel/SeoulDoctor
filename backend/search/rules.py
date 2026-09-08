"""Deterministic compiler for untrusted bilingual rule proposals."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import re
import unicodedata
from typing import Any, Iterable, Mapping, Sequence

from query_facets import (
    COMMENT_ALIASES,
    DISEASE_ALIASES,
    DISTANCE_PATTERN,
    GENDER_ALIASES,
    PLACE_ALIASES,
    SPECIALTY_ALIASES,
    TRAVEL_PHRASES,
    expand_multilingual_retrieval_terms,
)

from .contracts import (
    AreaRule,
    DistanceRule,
    EvidenceRequirement,
    GeoPoint,
    HardEligibility,
    RequiredAttribute,
    RuleCompilation,
    RuleIssue,
    RuleProvenance,
    SearchRules,
    SoftPreference,
)


SCHEMA_VERSION = "search-rules-v2"

_ALLOWED_PROPOSAL_FIELDS = frozenset({
    "address_korean",
    "comment_terms",
    "disease_terms",
    "district",
    "dong",
    "extraction_error",
    "extraction_source",
    "gender_terms",
    "hard_keywords",
    "inquiries",
    "is_citywide_search",
    "keywords",
    "language_pref",
    "latitude",
    "location",
    "longitude",
    "negative_hard_keywords",
    "negative_keywords",
    "place_terms",
    "required_hours",
    "soft_keywords",
    "specialty",
    "specialty_confidence",
    "travel_confidence",
    "travel_label",
    "visit_reason",
})

SPECIALTY_IDS = {
    "산부인과": "obstetrics_gynecology",
    "치과": "dentistry",
    "피부과": "dermatology",
    "내과": "internal_medicine",
    "정형외과": "orthopedics",
    "소아청소년과": "pediatrics",
    "정신건강의학과": "psychiatry",
    "안과": "ophthalmology",
    "이비인후과": "otolaryngology",
    "비뇨의학과": "urology",
    "신경과": "neurology",
    "가정의학과": "family_medicine",
    "외과": "surgery",
}

HARD_CONCEPT_ALIASES: Mapping[str, Sequence[str]] = {
    "parking": ("parking", "parking available", "주차", "주차 가능"),
    "english_consultation": (
        "english speaking",
        "english consultation",
        "english-speaking",
        "영어 상담",
        "영어 진료",
    ),
    "weekend_hours": (
        "weekend hours",
        "open weekends",
        "weekend appointments",
        "주말 진료",
        "주말 운영",
    ),
    "wheelchair_accessible": (
        "wheelchair accessible",
        "wheelchair access",
        "휠체어 접근",
        "휠체어 이용",
    ),
    "cosmetic_only": (
        "cosmetic only",
        "cosmetic-only",
        "aesthetic only",
        "미용만",
        "미용 전용",
    ),
}

_SOFT_CONCEPT_ALIASES: Mapping[str, Sequence[str]] = {
    "friendly": ("friendly", "kind", "친절", "친절한", "상냥"),
    "expensive": ("expensive", "pricey", "costly", "비싼", "고가"),
    "thorough": ("thorough", "careful", "detailed", "꼼꼼", "세심"),
    "short_wait": (
        "short wait",
        "quick wait",
        "대기 시간이 짧",
        "대기시간 짧",
    ),
    "clear_explanations": (
        "clear explanation",
        "clear explanations",
        "explains well",
        "설명을 잘",
        "설명 잘",
        "자세한 설명",
    ),
}

_SEOUL_ALIASES = frozenset({
    "seoul",
    "seoul city",
    "all seoul",
    "anywhere in seoul",
    "서울",
    "서울시",
    "서울 전체",
    "서울 전역",
})


def _normalized(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _slug(value: str) -> str:
    normalized = _normalized(value)
    slug = re.sub(r"[^0-9a-z가-힣]+", "_", normalized).strip("_")
    return slug


def _contains_hangul(value: str) -> bool:
    return bool(re.search(r"[가-힣ㄱ-ㅎㅏ-ㅣ]", value))


def _source_span(query: str, terms: Iterable[str]) -> tuple[int, int] | None:
    normalized_query = unicodedata.normalize("NFKC", query).casefold()
    for term in sorted(terms, key=len, reverse=True):
        normalized_term = unicodedata.normalize("NFKC", term).casefold()
        start = normalized_query.find(normalized_term)
        if start >= 0:
            return start, start + len(normalized_term)
    return None


def _language(query: str) -> str:
    has_hangul = _contains_hangul(query)
    has_latin = bool(re.search(r"[a-zA-Z]", query))
    if has_hangul and has_latin:
        return "mixed"
    if has_hangul:
        return "ko"
    return "en"


def _alias_lookup(
    groups: Iterable[tuple[str, Sequence[str]]],
    identifiers: Mapping[str, str] | None = None,
) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for canonical, aliases in groups:
        identifier = identifiers[canonical] if identifiers else _slug(canonical)
        for alias in (canonical, *aliases):
            lookup[_normalized(alias)] = identifier
    return lookup


_SPECIALTY_LOOKUP = _alias_lookup(SPECIALTY_ALIASES, SPECIALTY_IDS)
_DISTRICT_LOOKUP = _alias_lookup(PLACE_ALIASES)


def _concept_lookup(groups: Mapping[str, Sequence[str]]) -> dict[str, str]:
    return {
        _normalized(alias): concept_id
        for concept_id, aliases in groups.items()
        for alias in (concept_id, *aliases)
    }


_HARD_LOOKUP = _concept_lookup(HARD_CONCEPT_ALIASES)
_SOFT_LOOKUP = _concept_lookup(_SOFT_CONCEPT_ALIASES)

for canonical, aliases in COMMENT_ALIASES:
    concept_id = _SOFT_LOOKUP.get(_normalized(canonical), _slug(canonical))
    for alias in (canonical, *aliases):
        _SOFT_LOOKUP.setdefault(_normalized(alias), concept_id)

for canonical, aliases in (*GENDER_ALIASES, *DISEASE_ALIASES):
    concept_id = _slug(canonical)
    for alias in (canonical, *aliases):
        _HARD_LOOKUP.setdefault(_normalized(alias), concept_id)


@dataclass(frozen=True)
class SearchRulesDraft:
    specialty: str | None
    location: str | None
    district: str | None
    latitude: float | None
    longitude: float | None
    is_citywide_search: bool
    hard_keywords: tuple[str, ...]
    soft_keywords: tuple[str, ...]
    negative_hard_keywords: tuple[str, ...]
    negative_keywords: tuple[str, ...]
    gender_terms: tuple[str, ...]
    disease_terms: tuple[str, ...]
    comment_terms: tuple[str, ...]
    required_hours: tuple[str, ...]
    inquiries: tuple[str, ...]
    visit_reason: str | None


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError
    stripped = value.strip()
    return stripped or None


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise TypeError
    output: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            raise TypeError
        text = item.strip()
        key = _normalized(text)
        if text and key not in seen:
            output.append(text)
            seen.add(key)
    return tuple(output)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError
    return float(value)


def _parse_draft(
    proposal: Mapping[str, Any],
) -> tuple[SearchRulesDraft | None, tuple[RuleIssue, ...]]:
    rejected = sorted(set(proposal) - _ALLOWED_PROPOSAL_FIELDS)
    if rejected:
        return None, tuple(
            RuleIssue(
                code="rejected_control",
                field=field,
                needs_clarification=False,
            )
            for field in rejected
        )

    try:
        draft = SearchRulesDraft(
            specialty=_string_or_none(proposal.get("specialty")),
            location=_string_or_none(proposal.get("location")),
            district=_string_or_none(proposal.get("district")),
            latitude=_optional_float(proposal.get("latitude")),
            longitude=_optional_float(proposal.get("longitude")),
            is_citywide_search=proposal.get("is_citywide_search") is True,
            hard_keywords=_string_tuple(proposal.get("hard_keywords")),
            soft_keywords=_string_tuple(
                proposal.get("soft_keywords", proposal.get("keywords"))
            ),
            negative_hard_keywords=_string_tuple(
                proposal.get("negative_hard_keywords")
            ),
            negative_keywords=_string_tuple(proposal.get("negative_keywords")),
            gender_terms=_string_tuple(proposal.get("gender_terms")),
            disease_terms=_string_tuple(proposal.get("disease_terms")),
            comment_terms=_string_tuple(proposal.get("comment_terms")),
            required_hours=_string_tuple(proposal.get("required_hours")),
            inquiries=_string_tuple(proposal.get("inquiries")),
            visit_reason=_string_or_none(proposal.get("visit_reason")),
        )
    except (TypeError, ValueError):
        return None, (
            RuleIssue(
                code="invalid_proposal",
                field="proposal",
                needs_clarification=False,
            ),
        )

    if (draft.latitude is None) != (draft.longitude is None):
        return None, (
            RuleIssue(
                code="invalid_proposal",
                field="location",
                needs_clarification=True,
            ),
        )
    return draft, ()


def _explicit_distance(query: str) -> tuple[float, tuple[int, int]] | None:
    match = DISTANCE_PATTERN.search(query)
    if match is None:
        return None
    value = float(match.group("value").replace(",", "."))
    unit = match.group("unit").casefold()
    distance_km = value if unit.startswith("k") or unit == "킬로미터" else value / 1000.0
    return distance_km, match.span()


def _explicit_travel_phrase(query: str) -> tuple[float, tuple[int, int]] | None:
    normalized_query = unicodedata.normalize("NFKC", query).casefold()
    phrase_distances = {
        "Walking Distance": 0.5,
        "Nearby": 1.0,
        "Flexible": 10.0,
        "Willing to Travel": 15.0,
        "Anywhere in Seoul": 25.0,
    }
    for label, phrases in TRAVEL_PHRASES:
        for phrase in phrases:
            start = normalized_query.find(phrase.casefold())
            if start >= 0 and label != "Anywhere in Seoul":
                return phrase_distances[label], (start, start + len(phrase))
    return None


def _canonical_area(draft: SearchRulesDraft) -> tuple[str, str, str] | None:
    location = draft.location or draft.district
    if draft.is_citywide_search or (
        location is not None and _normalized(location) in _SEOUL_ALIASES
    ):
        return "seoul", "city", "서울"

    district = draft.district
    if district is None and location and (
        location.endswith("구") or _normalized(location).endswith("-gu")
    ):
        district = location
    if district:
        area_id = _DISTRICT_LOOKUP.get(_normalized(district))
        if area_id:
            canonical_name = next(
                canonical
                for canonical, _ in PLACE_ALIASES
                if _slug(canonical) == area_id
            )
            return f"district:{area_id}", "district", canonical_name
    return None


def _split_language_terms(aliases: Sequence[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    english = tuple(alias for alias in aliases if not _contains_hangul(alias))
    korean = tuple(alias for alias in aliases if _contains_hangul(alias))
    return english, korean


def _provenance(
    query: str,
    turn_id: str,
    source: str,
    terms: Iterable[str] = (),
    span: tuple[int, int] | None = None,
) -> RuleProvenance:
    return RuleProvenance(
        source=source,
        source_span=span if span is not None else _source_span(query, terms),
        turn_id=turn_id,
    )


def _canonical_concept(
    value: str,
    lookup: Mapping[str, str],
) -> str | None:
    return lookup.get(_normalized(value))


def _rules_hash(
    hard: HardEligibility,
    soft: tuple[SoftPreference, ...],
    evidence: tuple[EvidenceRequirement, ...],
) -> str:
    if isinstance(hard.geography, DistanceRule):
        geography: dict[str, Any] = {
            "kind": "distance",
            "latitude": hard.geography.anchor.latitude,
            "longitude": hard.geography.anchor.longitude,
            "max_km": hard.geography.max_km,
        }
    else:
        geography = {
            "kind": hard.geography.area_kind,
            "area_id": hard.geography.area_id,
        }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "hard": {
            "specialty_ids": sorted(hard.specialty_ids),
            "geography": geography,
            "prohibited_facility_ids": sorted(hard.prohibited_facility_ids),
            "prohibited_taxonomy_ids": sorted(hard.prohibited_taxonomy_ids),
            "required_attributes": sorted(
                item.concept_id for item in hard.required_attributes
            ),
        },
        "soft": sorted(
            (item.concept_id, item.polarity, item.weight_class)
            for item in soft
        ),
        "evidence": sorted(
            (
                item.requirement_id,
                item.terms_en,
                item.terms_ko,
                item.match_mode,
                tuple(sorted(item.source_types)),
                item.support_required,
                item.evidence_role,
            )
            for item in evidence
        ),
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(serialized.encode("utf-8")).hexdigest()


class RulesCompiler:
    """Validate a proposal and compile it into immutable search policy."""

    def __init__(
        self,
        *,
        allowed_specialties: Iterable[str] = (),
        allow_literal_hard_requirements: bool = False,
    ) -> None:
        self._specialty_lookup = dict(_SPECIALTY_LOOKUP)
        for label in allowed_specialties:
            normalized_label = _normalized(str(label))
            if normalized_label:
                self._specialty_lookup.setdefault(
                    normalized_label,
                    f"category:{normalized_label}",
                )
        self._allow_literal_hard_requirements = (
            allow_literal_hard_requirements
        )

    def compile(
        self,
        *,
        original_query: str,
        turn_id: str,
        proposal: Mapping[str, Any],
        previous_rules: SearchRules | None = None,
        preserved_distance_km: float | None = None,
    ) -> RuleCompilation:
        if not original_query.strip() or not turn_id:
            return RuleCompilation(
                rules=None,
                issues=(
                    RuleIssue(
                        code="invalid_proposal",
                        field="original_query",
                        needs_clarification=False,
                    ),
                ),
            )

        draft, issues = _parse_draft(proposal)
        if draft is None:
            return RuleCompilation(rules=None, issues=issues)

        compiler_issues: list[RuleIssue] = []
        provenance = (
            dict(previous_rules.provenance)
            if previous_rules
            else {}
        )

        specialty_ids = previous_rules.hard.specialty_ids if previous_rules else frozenset()
        if draft.specialty:
            specialty_id = self._specialty_lookup.get(_normalized(draft.specialty))
            if specialty_id is None:
                compiler_issues.append(RuleIssue(
                    code="unknown_specialty",
                    field="specialty",
                    needs_clarification=True,
                ))
            else:
                specialty_ids = frozenset({specialty_id})
                provenance["hard.specialty_ids"] = _provenance(
                    original_query,
                    turn_id,
                    "taxonomy",
                    (draft.specialty,),
                )

        explicit_distance = _explicit_distance(original_query)
        explicit_phrase = _explicit_travel_phrase(original_query)
        requested_distance = explicit_distance or explicit_phrase
        if requested_distance and not 0.0 < requested_distance[0] <= 100.0:
            compiler_issues.append(RuleIssue(
                code="invalid_distance",
                field="distance",
                needs_clarification=True,
            ))
            requested_distance = None

        area = _canonical_area(draft)
        point: GeoPoint | None = None
        if draft.latitude is not None and draft.longitude is not None:
            try:
                point = GeoPoint(draft.latitude, draft.longitude)
            except ValueError:
                compiler_issues.append(RuleIssue(
                    code="unknown_location",
                    field="location",
                    needs_clarification=True,
                ))

        location_was_proposed = bool(
            draft.location or draft.district or draft.is_citywide_search or point
        )
        if area and requested_distance is None:
            area_id, area_kind, display_name = area
            geography = AreaRule(
                area_id=area_id,
                area_kind=area_kind,
                display_name=display_name,
                provenance=_provenance(
                    original_query,
                    turn_id,
                    "user_explicit",
                    filter(None, (draft.location, draft.district)),
                ),
            )
        elif point is not None:
            if requested_distance:
                max_km, distance_span = requested_distance
                distance_provenance = _provenance(
                    original_query,
                    turn_id,
                    "user_explicit",
                    span=distance_span,
                )
            elif previous_rules and isinstance(
                previous_rules.hard.geography, DistanceRule
            ):
                max_km = previous_rules.hard.geography.max_km
                distance_provenance = previous_rules.hard.geography.provenance
            elif preserved_distance_km is not None:
                max_km = float(preserved_distance_km)
                distance_provenance = _provenance(
                    original_query,
                    turn_id,
                    "verified_context",
                )
            else:
                max_km = 5.0
                distance_provenance = _provenance(
                    original_query,
                    turn_id,
                    "default_5km",
                )
            geography = DistanceRule(
                anchor=point,
                max_km=max_km,
                provenance=distance_provenance,
            )
        elif requested_distance and previous_rules and isinstance(
            previous_rules.hard.geography, DistanceRule
        ):
            geography = DistanceRule(
                anchor=previous_rules.hard.geography.anchor,
                max_km=requested_distance[0],
                provenance=_provenance(
                    original_query,
                    turn_id,
                    "user_explicit",
                    span=requested_distance[1],
                ),
            )
        elif not location_was_proposed and previous_rules:
            geography = previous_rules.hard.geography
        else:
            code = "unknown_location" if location_was_proposed else "location_required"
            compiler_issues.append(RuleIssue(
                code=code,
                field="location",
                needs_clarification=True,
            ))
            geography = None

        if compiler_issues:
            return RuleCompilation(
                rules=None,
                issues=tuple(compiler_issues),
            )
        assert geography is not None
        provenance["hard.geography"] = geography.provenance

        required_attributes = {
            item.concept_id: item
            for item in (
                previous_rules.hard.required_attributes
                if previous_rules
                else ()
            )
        }
        prohibited_facility_ids = set(
            previous_rules.hard.prohibited_facility_ids
            if previous_rules
            else ()
        )
        prohibited_taxonomy_ids = set(
            previous_rules.hard.prohibited_taxonomy_ids
            if previous_rules
            else ()
        )
        evidence_requirements = {
            item.requirement_id: item
            for item in (
                previous_rules.evidence
                if previous_rules
                else ()
            )
            if not item.requirement_id.startswith("inquiry:")
        }

        for term in draft.hard_keywords:
            concept_id = _canonical_concept(term, _HARD_LOOKUP)
            if concept_id is None and self._allow_literal_hard_requirements:
                concept_id = f"literal:{_normalized(term)}"
            if concept_id is None:
                compiler_issues.append(RuleIssue(
                    code="unknown_hard_requirement",
                    field="hard_keywords",
                    needs_clarification=True,
                ))
                continue
            aliases = HARD_CONCEPT_ALIASES.get(concept_id, (term,))
            terms_en, terms_ko = _split_language_terms(aliases)
            item_provenance = _provenance(
                original_query,
                turn_id,
                "taxonomy",
                (term, *aliases),
            )
            required_attributes[concept_id] = RequiredAttribute(
                concept_id=concept_id,
                terms_en=terms_en,
                terms_ko=terms_ko,
                provenance=item_provenance,
            )
            provenance[f"hard.required_attributes.{concept_id}"] = item_provenance
            evidence_requirements[f"attribute:{concept_id}"] = EvidenceRequirement(
                requirement_id=f"attribute:{concept_id}",
                terms_en=terms_en,
                terms_ko=terms_ko,
                match_mode="all_terms",
                source_types=frozenset({"facility_fact", "verbatim_review"}),
                support_required=True,
            )

        for term in draft.negative_hard_keywords:
            concept_id = _canonical_concept(term, _HARD_LOOKUP)
            if concept_id is None and self._allow_literal_hard_requirements:
                concept_id = f"literal:{_normalized(term)}"
            if concept_id is None:
                compiler_issues.append(RuleIssue(
                    code="unknown_hard_requirement",
                    field="negative_hard_keywords",
                    needs_clarification=True,
                ))
                continue
            prohibited_taxonomy_ids.add(concept_id)
            provenance[f"hard.prohibited_taxonomy_ids.{concept_id}"] = _provenance(
                original_query,
                turn_id,
                "taxonomy",
                (term,),
            )

        soft_preferences = {
            (item.concept_id, item.polarity): item
            for item in (previous_rules.soft if previous_rules else ())
        }
        for polarity, terms in (
            ("positive", draft.soft_keywords),
            ("negative", draft.negative_keywords),
        ):
            for term in terms:
                concept_id = _canonical_concept(term, _SOFT_LOOKUP) or _slug(term)
                if not concept_id:
                    continue
                opposite = "negative" if polarity == "positive" else "positive"
                soft_preferences.pop((concept_id, opposite), None)
                provenance.pop(f"soft.{opposite}.{concept_id}", None)
                item_provenance = _provenance(
                    original_query,
                    turn_id,
                    "taxonomy" if _canonical_concept(term, _SOFT_LOOKUP) else "user_explicit",
                    (term,),
                )
                soft_preferences[(concept_id, polarity)] = SoftPreference(
                    concept_id=concept_id,
                    polarity=polarity,
                    weight_class="normal",
                    provenance=item_provenance,
                )
                provenance[f"soft.{polarity}.{concept_id}"] = item_provenance
                for prior_polarity in ("positive", "negative"):
                    evidence_requirements.pop(
                        f"preference:{prior_polarity}:{concept_id}",
                        None,
                    )
                aliases = tuple(
                    _SOFT_CONCEPT_ALIASES.get(concept_id)
                    or expand_multilingual_retrieval_terms((term,))
                )
                terms_en, terms_ko = _split_language_terms(aliases)
                evidence_requirements[
                    f"preference:{polarity}:{concept_id}"
                ] = EvidenceRequirement(
                    requirement_id=f"preference:{polarity}:{concept_id}",
                    terms_en=terms_en,
                    terms_ko=terms_ko,
                    match_mode="semantic",
                    source_types=frozenset({"verbatim_review"}),
                    support_required=True,
                    evidence_role=(
                        "risk" if polarity == "negative" else "support"
                    ),
                )

        evidence_groups = (
            (
                draft.gender_terms,
                frozenset({
                    "facility_fact",
                    "medical_info",
                    "review_summary",
                    "verbatim_review",
                }),
                "support",
            ),
            (
                draft.disease_terms,
                frozenset({
                    "facility_fact",
                    "medical_info",
                    "review_summary",
                    "verbatim_review",
                }),
                "disease",
            ),
            (draft.comment_terms, frozenset({"verbatim_review"}), "support"),
        )
        for terms, source_types, evidence_role in evidence_groups:
            for term in terms:
                concept_id = (
                    _canonical_concept(term, _HARD_LOOKUP)
                    or _canonical_concept(term, _SOFT_LOOKUP)
                    or _slug(term)
                )
                requirement_id = f"evidence:{concept_id}"
                if not concept_id or requirement_id in evidence_requirements:
                    continue
                aliases = tuple(expand_multilingual_retrieval_terms((term,)))
                terms_en, terms_ko = _split_language_terms(aliases)
                evidence_requirements[requirement_id] = EvidenceRequirement(
                    requirement_id=requirement_id,
                    terms_en=terms_en,
                    terms_ko=terms_ko,
                    match_mode="semantic",
                    source_types=source_types,
                    support_required=True,
                    evidence_role=evidence_role,
                )

        for inquiry in draft.inquiries:
            inquiry_concepts = {
                concept_id: aliases
                for concept_id, aliases in HARD_CONCEPT_ALIASES.items()
                if _source_span(inquiry, aliases) is not None
            }
            if re.search(r"\benglish\b|영어", inquiry, re.I):
                inquiry_concepts["english_consultation"] = HARD_CONCEPT_ALIASES["english_consultation"]
            if not inquiry_concepts:
                inquiry_concepts[sha256(inquiry.encode("utf-8")).hexdigest()[:16]] = (
                    tuple(expand_multilingual_retrieval_terms((inquiry,)))
                )
            for concept_id, aliases in inquiry_concepts.items():
                if f"attribute:{concept_id}" in evidence_requirements:
                    continue
                terms_en, terms_ko = _split_language_terms(aliases)
                requirement_id = f"inquiry:{concept_id}"
                evidence_requirements[requirement_id] = EvidenceRequirement(
                    requirement_id=requirement_id,
                    terms_en=terms_en,
                    terms_ko=terms_ko,
                    match_mode="semantic",
                    source_types=frozenset({"facility_fact", "verbatim_review"}),
                    support_required=False,
                )

        if draft.visit_reason:
            terms_en, terms_ko = _split_language_terms(tuple(
                expand_multilingual_retrieval_terms((draft.visit_reason,))
            ))
            requirement_id = f"visit_reason:{_slug(draft.visit_reason)}"
            evidence_requirements[requirement_id] = EvidenceRequirement(
                requirement_id=requirement_id,
                terms_en=terms_en,
                terms_ko=terms_ko,
                match_mode="semantic",
                source_types=frozenset({"medical_info", "verbatim_review"}),
                support_required=False,
                evidence_role="disease",
            )

        hour_aliases = {
            "tuesday_evening": (
                ("Tuesday evening", "Tuesday night", "21:00"),
                ("화요일 저녁", "화요일 야간", "화:"),
            ),
        }
        for hour in draft.required_hours:
            canonical = _normalized(hour).replace(" ", "_")
            terms_en, terms_ko = hour_aliases.get(canonical, ((hour,), ()))
            concept_id = f"hours:{canonical}"
            item_provenance = _provenance(
                original_query,
                turn_id,
                "taxonomy",
                (*terms_en, *terms_ko),
            )
            required_attributes[concept_id] = RequiredAttribute(
                concept_id=concept_id,
                terms_en=tuple(terms_en),
                terms_ko=tuple(terms_ko),
                provenance=item_provenance,
            )
            provenance[f"hard.required_attributes.{concept_id}"] = item_provenance
            evidence_requirements[concept_id] = EvidenceRequirement(
                requirement_id=concept_id,
                terms_en=tuple(terms_en),
                terms_ko=tuple(terms_ko),
                match_mode="semantic",
                source_types=frozenset({"facility_fact"}),
                support_required=True,
            )

        if compiler_issues:
            return RuleCompilation(
                rules=None,
                issues=tuple(compiler_issues),
            )

        hard = HardEligibility(
            specialty_ids=frozenset(specialty_ids),
            geography=geography,
            prohibited_facility_ids=frozenset(prohibited_facility_ids),
            prohibited_taxonomy_ids=frozenset(prohibited_taxonomy_ids),
            required_attributes=tuple(sorted(
                required_attributes.values(),
                key=lambda item: item.concept_id,
            )),
        )
        soft = tuple(
            soft_preferences[key]
            for key in sorted(soft_preferences)
        )
        evidence = tuple(
            evidence_requirements[key]
            for key in sorted(evidence_requirements)
        )
        rules = SearchRules(
            schema_version=SCHEMA_VERSION,
            original_query=original_query,
            language=_language(original_query),
            hard=hard,
            soft=soft,
            evidence=evidence,
            provenance=provenance,
            rules_hash=_rules_hash(hard, soft, evidence),
        )
        return RuleCompilation(rules=rules)


def compile_shadow_rules(
    original_query: str,
    proposal: Mapping[str, Any],
) -> RuleCompilation:
    """Compile legacy extraction output without changing live search state."""
    sanitized = {
        key: value
        for key, value in proposal.items()
        if key in _ALLOWED_PROPOSAL_FIELDS
    }
    return RulesCompiler().compile(
        original_query=original_query,
        turn_id="shadow",
        proposal=sanitized,
    )


def compile_legacy_state_rules(
    original_query: str,
    state: Any,
    *,
    turn_id: str,
    allowed_specialties: Iterable[str] = (),
) -> RuleCompilation:
    """Translate server-processed legacy state during the staged cutover."""
    is_citywide = bool(getattr(state, "is_citywide_search", False))
    latitude = getattr(state, "latitude", None)
    longitude = getattr(state, "longitude", None)
    has_point = latitude is not None and longitude is not None
    is_zone = getattr(state, "search_mode", None) == "zone"
    state_location = getattr(state, "location", None)
    state_district = getattr(state, "district", None)
    normalized_location = (
        _normalized(state_location) if isinstance(state_location, str) else ""
    )
    explicit_area_location = bool(
        normalized_location.endswith("-gu")
        or normalized_location.endswith("구")
        or (state_district and normalized_location == _normalized(state_district))
    )

    if is_citywide:
        location = "Seoul"
        district = None
        latitude = None
        longitude = None
    elif has_point and not explicit_area_location:
        location = "verified map position"
        district = None
    elif is_zone and state_district:
        location = state_district
        district = state_district
        latitude = None
        longitude = None
    else:
        location = state_location
        district = state_district

    proposal = {
        "specialty": getattr(state, "specialty", None),
        "location": location,
        "district": district,
        "latitude": latitude,
        "longitude": longitude,
        "is_citywide_search": is_citywide,
        "hard_keywords": list(getattr(state, "hard_keywords", ()) or ()),
        "soft_keywords": list(getattr(state, "keywords", ()) or ()),
        "negative_hard_keywords": list(
            getattr(state, "negative_hard_keywords", ()) or ()
        ),
        "negative_keywords": list(
            getattr(state, "negative_keywords", ()) or ()
        ),
        "gender_terms": list(getattr(state, "gender_terms", ()) or ()),
        "disease_terms": list(getattr(state, "disease_terms", ()) or ()),
        "comment_terms": list(getattr(state, "comment_terms", ()) or ()),
        "required_hours": list(getattr(state, "required_hours", ()) or ()),
        "inquiries": list(getattr(state, "inquiries", ()) or ()),
        "visit_reason": getattr(state, "visit_reason", None),
    }
    preserved_distance = None
    if has_point:
        try:
            inherited_distance = float(getattr(state, "max_distance_km"))
        except (TypeError, ValueError):
            inherited_distance = 0.0
        try:
            travel_confidence = float(getattr(state, "travel_confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            travel_confidence = 0.0
        if 0.0 < inherited_distance <= 100.0 and (
            travel_confidence >= 0.6 or inherited_distance < 5.0
        ):
            preserved_distance = inherited_distance
    return RulesCompiler(
        allowed_specialties=allowed_specialties,
        allow_literal_hard_requirements=True,
    ).compile(
        original_query=original_query,
        turn_id=turn_id,
        proposal=proposal,
        preserved_distance_km=preserved_distance,
    )
