"""Compile one bilingual user turn into a safe, deterministic state update."""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from typing import Any, Literal, Mapping

from config import DISTANCE_MAPPING
from models import State
from query_facets import DISTANCE_PATTERN, augment_extracted_facets


TurnOperation = Literal["refine", "replace_context"]

_SEOUL_ALIASES = {
    "seoul",
    "seoul city",
    "all seoul",
    "anywhere in seoul",
    "서울",
    "서울시",
    "서울 전체",
    "서울 전역",
}

_REPLACE_PATTERNS = (
    re.compile(r"\b(?:change|replace)\s+(?:that|the request|my request)\b", re.I),
    re.compile(r"\bstart\s+over\b", re.I),
    re.compile(r"요청(?:을|를)?\s*(?:바꿀|바꿔|변경)"),
)

_REMOVAL_PATTERNS: Mapping[str, tuple[re.Pattern[str], ...]] = {
    "parking": (
        re.compile(
            r"\bparking\b.{0,30}\b(?:no longer|required no longer|remove|drop|not required)\b",
            re.I,
        ),
        re.compile(r"\b(?:remove|drop)\b.{0,20}\bparking\b", re.I),
        re.compile(r"주차.{0,20}(?:필요\s*없|빼|제외|삭제)"),
    ),
    "long_wait": (
        re.compile(r"\blong\s+wait\b.{0,35}\b(?:acceptable|okay|fine)\b", re.I),
        re.compile(r"\b(?:acceptable|okay|fine)\b.{0,35}\blong\s+wait\b", re.I),
        re.compile(r"대기.{0,16}(?:길|긴).{0,20}(?:괜찮|상관없|제외.{0,8}(?:말|않|두지))"),
    ),
    "short_wait": (
        re.compile(r"\b(?:short|no)\s+wait\b.{0,30}\bno longer important\b", re.I),
        re.compile(r"대기.{0,12}(?:없|짧).{0,20}(?:더 이상.{0,8}중요하지|필요\s*없)"),
    ),
}

_CONCEPT_ALIASES: Mapping[str, tuple[str, ...]] = {
    "parking": ("parking", "parking available", "주차", "주차 가능"),
    "long_wait": (
        "long wait",
        "long waiting time",
        "긴 대기",
        "대기가 길",
        "대기 시간이 길",
    ),
    "short_wait": (
        "short wait",
        "no wait",
        "quick wait",
        "대기 없음",
        "대기가 없",
        "대기 시간이 짧",
    ),
}

_TUESDAY_EVENING_PATTERNS = (
    re.compile(r"\btuesday\b.{0,24}\b(?:evening|night|late)\b", re.I),
    re.compile(r"화요일.{0,16}(?:야간|저녁|늦게)"),
)


_AVOID_LONG_WAIT_PATTERNS = (
    re.compile(r"\bavoid\b.{0,24}\blong\s+waits?\b", re.I),
    re.compile(r"\blong\s+waits?\b.{0,24}\b(?:avoid|exclude)\b", re.I),
    re.compile(r"긴\s*대기.{0,16}(?:피|싫|제외)"),
)
_NEGATIVE_SIGNAL_PATTERNS: Mapping[str, tuple[re.Pattern[str], ...]] = {
    "aggressive upselling": (
        re.compile(r"\b(?:aggressive\s+upselling|pressure\s+sales?)\b", re.I),
        re.compile(r"(?:과도한|과잉).{0,8}(?:시술|치료).{0,8}권유|강매"),
    ),
    "unfriendly nurses": (
        re.compile(r"\b(?:unfriendly|rude|brusque)\s+nurs(?:e|es)\b", re.I),
        re.compile(r"간호사.{0,12}(?:불친절|무뚝뚝)"),
        re.compile(r"(?:불친절|무뚝뚝).{0,8}간호사"),
    ),
}




def _normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _optional_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _optional_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_terms(value: Any) -> tuple[str, ...] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        return None
    terms: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = _optional_text(item)
        if text is None:
            continue
        key = _normalize(text)
        if key not in seen:
            terms.append(text)
            seen.add(key)
    return tuple(terms)


def _canonical_comment_terms(value: Any) -> tuple[str, ...] | None:
    incoming = _optional_terms(value)
    if incoming is None:
        return None
    output: list[str] = []
    for term in incoming:
        normalized = _normalize(term)
        if any(
            token in normalized
            for token in ("careful", "thorough", "meticulous", "꼼꼼", "세심")
        ):
            canonical = "thorough"
        elif (
            "explanation" in normalized
            and any(token in normalized for token in ("clear", "detail"))
        ) or (
            "설명" in normalized
            and any(token in normalized for token in ("자세", "상세", "명확"))
        ):
            canonical = "clear explanations"
        else:
            canonical = term
        if _normalize(canonical) not in {_normalize(item) for item in output}:
            output.append(canonical)
    return tuple(output)


def _negative_terms(message: str, value: Any) -> tuple[str, ...] | None:
    incoming = list(_optional_terms(value) or ())
    detected = {
        canonical
        for canonical, patterns in _NEGATIVE_SIGNAL_PATTERNS.items()
        if any(pattern.search(message) for pattern in patterns)
    }
    if "unfriendly nurses" in detected:
        broad_fragments = {
            "friendly",
            "kind",
            "nurse",
            "nurses",
            "rude",
            "staff",
            "unfriendly",
        }
        incoming = [
            term for term in incoming if _normalize(term) not in broad_fragments
        ]
    output: list[str] = []
    for term in incoming:
        normalized = _normalize(term)
        if "long wait" in normalized or "긴 대기" in normalized:
            canonical = "long wait"
        elif any(
            pattern.search(term)
            for pattern in _NEGATIVE_SIGNAL_PATTERNS["aggressive upselling"]
        ):
            canonical = "aggressive upselling"
        elif any(
            pattern.search(term)
            for pattern in _NEGATIVE_SIGNAL_PATTERNS["unfriendly nurses"]
        ):
            canonical = "unfriendly nurses"
        else:
            canonical = term
        if _normalize(canonical) not in {_normalize(item) for item in output}:
            output.append(canonical)
    if any(pattern.search(message) for pattern in _AVOID_LONG_WAIT_PATTERNS):
        if "long wait" not in output:
            output.append("long wait")
    for canonical in detected:
        if canonical not in output:
            output.append(canonical)
    return tuple(output) if output or value is not None else None


def _explicit_distance_km(message: str) -> float | None:
    match = DISTANCE_PATTERN.search(message)
    if match is None:
        return None
    value = float(match.group("value").replace(",", "."))
    unit = match.group("unit").casefold()
    distance = value if unit.startswith("k") or unit == "킬로미터" else value / 1000.0
    return distance if 0.0 < distance <= 100.0 else None


def _travel_label_for_distance(distance_km: float) -> str:
    for label, supported_distance in DISTANCE_MAPPING.items():
        if abs(float(supported_distance) - distance_km) <= 0.001:
            return label
    return f"Within {distance_km:g} km"


def _detect_operation(message: str, proposal: Mapping[str, Any]) -> TurnOperation:
    proposed = _optional_text(proposal.get("operation"))
    if proposed == "replace_context":
        return "replace_context"
    if any(pattern.search(message) for pattern in _REPLACE_PATTERNS):
        return "replace_context"
    return "refine"


def _detect_removed_concepts(
    message: str,
    proposal: Mapping[str, Any],
) -> tuple[str, ...]:
    found: list[str] = []
    proposed = _optional_terms(proposal.get("remove_terms")) or ()
    for value in proposed:
        normalized = _normalize(value)
        for concept, aliases in _CONCEPT_ALIASES.items():
            if normalized == concept or any(alias in normalized for alias in aliases):
                if concept not in found:
                    found.append(concept)
    for concept, patterns in _REMOVAL_PATTERNS.items():
        if any(pattern.search(message) for pattern in patterns) and concept not in found:
            found.append(concept)
    return tuple(found)


def _detect_required_hours(
    message: str,
    proposal: Mapping[str, Any],
) -> tuple[str, ...] | None:
    proposed = list(_optional_terms(proposal.get("required_hours")) or ())
    hard_terms = _optional_terms(proposal.get("hard_keywords")) or ()
    combined_text = " ".join((message, *hard_terms))
    if any(pattern.search(combined_text) for pattern in _TUESDAY_EVENING_PATTERNS):
        proposed.append("tuesday_evening")
    output: list[str] = []
    for value in proposed:
        normalized = _normalize(value).replace(" ", "_")
        if normalized in {"tuesday_evening", "tuesday_night", "화요일_야간", "화요일_저녁"}:
            canonical = "tuesday_evening"
        else:
            canonical = normalized
        if canonical and canonical not in output:
            output.append(canonical)
    return tuple(output) if output else None


@dataclass(frozen=True)
class SearchDelta:
    """A validated user-intent delta; no ranking or routing controls allowed."""

    operation: TurnOperation
    specialty: str | None
    specialty_confidence: float | None
    location: str | None
    latitude: float | None
    longitude: float | None
    address_korean: str | None
    district: str | None
    dong: str | None
    citywide: bool | None
    distance_km: float | None
    travel_label: str | None
    hard_keywords: tuple[str, ...] | None
    soft_keywords: tuple[str, ...] | None
    negative_hard_keywords: tuple[str, ...] | None
    negative_keywords: tuple[str, ...] | None
    place_terms: tuple[str, ...] | None
    gender_terms: tuple[str, ...] | None
    disease_terms: tuple[str, ...] | None
    comment_terms: tuple[str, ...] | None
    required_hours: tuple[str, ...] | None
    remove_concepts: tuple[str, ...]
    response_language: Literal["English", "Korean"]
    extraction_source: str | None
    extraction_error: str | None


def compile_turn_delta(message: str, proposal: Mapping[str, Any] | None) -> SearchDelta:
    """Combine an untrusted model proposal with deterministic user-text facts."""
    raw = dict(proposal or {})
    augmented = augment_extracted_facets(message, raw)
    distance_km = _explicit_distance_km(message)
    if distance_km is None:
        proposed_distance = _optional_float(raw.get("distance_km"))
        if proposed_distance is not None and 0.0 < proposed_distance <= 100.0:
            distance_km = proposed_distance

    location = _optional_text(augmented.get("location"))
    normalized_location = _normalize(location) if location else None
    proposed_citywide = raw.get("is_citywide_search")
    citywide = proposed_citywide if isinstance(proposed_citywide, bool) else None
    if normalized_location in _SEOUL_ALIASES:
        citywide = True
        location = None
    elif location is not None:
        citywide = False

    latitude = _optional_float(augmented.get("latitude"))
    longitude = _optional_float(augmented.get("longitude"))
    if (latitude is None) != (longitude is None):
        latitude = None
        longitude = None

    response_language: Literal["English", "Korean"] = (
        "Korean" if re.search(r"[가-힣ㄱ-ㅎㅏ-ㅣ]", message) else "English"
    )
    return SearchDelta(
        operation=_detect_operation(message, raw),
        specialty=_optional_text(augmented.get("specialty")),
        specialty_confidence=_optional_float(augmented.get("specialty_confidence")),
        location=location,
        latitude=latitude,
        longitude=longitude,
        address_korean=_optional_text(augmented.get("address_korean")),
        district=_optional_text(augmented.get("district")),
        dong=_optional_text(augmented.get("dong")),
        citywide=citywide,
        distance_km=distance_km,
        travel_label=_optional_text(augmented.get("travel_label")),
        hard_keywords=_optional_terms(augmented.get("hard_keywords")),
        soft_keywords=_optional_terms(augmented.get("soft_keywords")),
        negative_hard_keywords=_optional_terms(augmented.get("negative_hard_keywords")),
        negative_keywords=_negative_terms(message, augmented.get("negative_keywords")),
        place_terms=_optional_terms(augmented.get("place_terms")),
        gender_terms=_optional_terms(augmented.get("gender_terms")),
        disease_terms=_optional_terms(augmented.get("disease_terms")),
        comment_terms=_canonical_comment_terms(augmented.get("comment_terms")),
        required_hours=_detect_required_hours(message, raw),
        remove_concepts=_detect_removed_concepts(message, raw),
        response_language=response_language,
        extraction_source=_optional_text(augmented.get("extraction_source")),
        extraction_error=_optional_text(augmented.get("extraction_error")),
    )


def _merge_terms(
    existing: list[str],
    incoming: tuple[str, ...] | None,
    *,
    replace: bool,
) -> list[str]:
    if incoming is None:
        return list(existing)
    values = list(incoming) if replace else [*existing, *incoming]
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = _normalize(value)
        if key and key not in seen:
            output.append(value)
            seen.add(key)
    return output


def _term_matches_concept(term: str, concept: str) -> bool:
    normalized = _normalize(term)
    return any(_normalize(alias) in normalized for alias in _CONCEPT_ALIASES[concept])


def _clear_search_context(state: State) -> None:
    state.specialty = None
    state.specialty_confidence = 0.0
    state.location = None
    state.latitude = None
    state.longitude = None
    state.address_korean = None
    state.district = None
    state.dong = None
    state.is_citywide_search = False
    state.search_mode = None
    state.max_distance_km = 5.0
    state.travel_label = "Moderate"
    state.travel_confidence = 0.5
    for field in (
        "hard_keywords",
        "keywords",
        "negative_hard_keywords",
        "negative_keywords",
        "place_terms",
        "gender_terms",
        "disease_terms",
        "comment_terms",
        "required_hours",
    ):
        setattr(state, field, [])


def reduce_search_state(
    current: State,
    delta: SearchDelta,
    *,
    replace_keywords: bool = False,
) -> State:
    """Apply a compiled delta without allowing the model to rewrite old state."""
    state = current.model_copy(deep=True)
    if delta.operation == "replace_context":
        _clear_search_context(state)
        replace_keywords = True

    if delta.specialty is not None:
        state.specialty = delta.specialty
        state.specialty_confidence = delta.specialty_confidence or 0.7

    location_changed = delta.location is not None or delta.citywide is True
    if delta.citywide is True:
        state.location = None
        state.latitude = None
        state.longitude = None
        state.address_korean = None
        state.district = None
        state.dong = None
        state.is_citywide_search = True
        state.search_mode = "zone"
        state.max_distance_km = 25.0
        state.travel_label = "Anywhere in Seoul"
        state.travel_confidence = 1.0
    elif delta.location is not None:
        state.location = delta.location
        state.is_citywide_search = False
        state.latitude = delta.latitude
        state.longitude = delta.longitude
        state.address_korean = delta.address_korean
        state.district = delta.district
        state.dong = delta.dong
        if delta.latitude is not None and delta.longitude is not None:
            state.search_mode = "distance"
        elif delta.location.endswith("구") or _normalize(delta.location).endswith("-gu"):
            state.search_mode = "zone"
        else:
            state.search_mode = "distance"

    if not state.is_citywide_search:
        if delta.distance_km is not None:
            state.max_distance_km = delta.distance_km
            state.travel_label = _travel_label_for_distance(delta.distance_km)
            state.travel_confidence = 1.0
            state.search_mode = "distance"
        elif delta.travel_label in DISTANCE_MAPPING:
            state.travel_label = delta.travel_label
            state.max_distance_km = float(DISTANCE_MAPPING[delta.travel_label])
            state.travel_confidence = 0.6
        elif location_changed:
            # A new local anchor must never inherit an old city-wide or unrelated
            # radius. The product default is a five-kilometre distance search.
            state.max_distance_km = 5.0
            state.travel_label = "Moderate"
            state.travel_confidence = 0.5

    field_map = {
        "hard_keywords": delta.hard_keywords,
        "keywords": delta.soft_keywords,
        "negative_hard_keywords": delta.negative_hard_keywords,
        "negative_keywords": delta.negative_keywords,
        "place_terms": delta.place_terms,
        "gender_terms": delta.gender_terms,
        "disease_terms": delta.disease_terms,
        "comment_terms": delta.comment_terms,
        "required_hours": delta.required_hours,
    }
    for field, incoming in field_map.items():
        setattr(
            state,
            field,
            _merge_terms(
                getattr(state, field),
                incoming,
                replace=replace_keywords,
            ),
        )

    for concept in delta.remove_concepts:
        for field in (
            "hard_keywords",
            "keywords",
            "negative_hard_keywords",
            "negative_keywords",
            "comment_terms",
        ):
            setattr(
                state,
                field,
                [
                    term
                    for term in getattr(state, field)
                    if not _term_matches_concept(term, concept)
                ],
            )

    state.language_pref = delta.response_language
    state.extraction_source = delta.extraction_source or state.extraction_source
    state.extraction_error = delta.extraction_error
    return state
