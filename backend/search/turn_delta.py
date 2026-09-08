"""Compile one bilingual user turn into a safe, deterministic state update."""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from typing import Any, Literal, Mapping

from config import DISTANCE_MAPPING
from models import State
from query_facets import (
    DISTANCE_PATTERN, augment_extracted_facets, english_consultation_intent,
    intent_clauses, is_exclusion, is_inquiry, is_withdrawal, requests_citywide_search, term_in_clause,
)
from review_presentation import response_language


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
    re.compile(r"\b(?:change|replace)\s+(?:the request|my request)\b", re.I),
    re.compile(r"\bstart\s+over\b", re.I),
    re.compile(r"^(?:please\s+)?(?:reset|restart|new\s+search)[.!?\s]*$", re.I),
    re.compile(r"\b(?:reset|restart|start)\s+(?:(?:the|my|this|a new)\s+)?(?:search|request|conversation)\b", re.I),
    re.compile(r"요청(?:을|를)?\s*(?:바꿀|바꿔|변경)"),
    re.compile(r"새로\s*시작|처음부터\s*(?:(?:다시\s*)?(?:시작|검색)|(?:요)?[.!?\s]*$)"),
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
    "waiting": ("wait", "waits", "waiting", "대기"),
    "english_consultation": ("English", "영어"),
}

_PREFERENCE_FIELDS = (
    "hard_keywords", "keywords", "negative_hard_keywords", "negative_keywords",
    "comment_terms", "gender_terms", "disease_terms", "required_hours",
)

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
    inquiries = {
        canonical
        for canonical, patterns in _NEGATIVE_SIGNAL_PATTERNS.items()
        if any(pattern.search(clause) for pattern in patterns for clause in intent_clauses(message))
        and all(is_inquiry(clause) and not is_exclusion(clause)
                for clause in intent_clauses(message)
                if any(pattern.search(clause) for pattern in patterns))
    }
    incoming = [term for term in incoming if not any(
        _normalize(term) == canonical or any(pattern.search(term) for pattern in _NEGATIVE_SIGNAL_PATTERNS[canonical])
        for canonical in inquiries
    )]
    if "unfriendly nurses" in inquiries:
        incoming = [term for term in incoming if _normalize(term) not in {"unfriendly", "rude", "nurse", "nurses"}]
    detected = {
        canonical
        for canonical, patterns in _NEGATIVE_SIGNAL_PATTERNS.items()
        if any(pattern.search(clause) and is_exclusion(clause)
               for clause in intent_clauses(message) for pattern in patterns)
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
    for clause in intent_clauses(message):
        if is_inquiry(clause) or re.search(r"\b(?:not|don['’]?t|do not|never|isn['’]?t)\b|바꾸지|변경하지|시작하지|말고", clause, re.I):
            continue
        if any(pattern.search(clause) for pattern in _REPLACE_PATTERNS):
            return "replace_context"
    return "refine"


def _detect_removed_concepts(
    message: str,
    proposal: Mapping[str, Any],
) -> tuple[str, ...]:
    found: list[str] = []
    proposed = _optional_terms(proposal.get("remove_terms")) or ()
    for value in proposed:
        concept = _concept_for_term(value)
        if any(_term_matches_concept(clause, concept) and is_withdrawal(clause)
               for clause in intent_clauses(message)):
            found.append(concept)
    for concept, patterns in _REMOVAL_PATTERNS.items():
        if any(pattern.search(clause) and is_withdrawal(clause)
               for pattern in patterns for clause in intent_clauses(message)) and concept not in found:
            found.append(concept)
    if _accepts_waiting(message):
        found.append("waiting")
    if english_consultation_intent(message) == "withdraw":
        found.append("english_consultation")
    return tuple(found)


def _accepts_waiting(message: str) -> bool:
    for clause in intent_clauses(message):
        if not re.search(r"\bwait(?:ing|s)?\b|대기|기다", clause, re.I):
            continue
        if is_inquiry(clause):
            continue
        if re.search(
            r"\b(?:not|isn['’]?t|aren['’]?t)(?:\s+(?:at all|really))?\s+(?:fine|okay|ok|acceptable)\b"
            r"|don['’]?t\s+think|do not\s+think|괜찮지\s*않|안\s*괜찮", clause, re.I,
        ):
            continue
        if is_withdrawal(clause) or re.search(
            r"\b(?:fine|okay|acceptable|willing to wait|can wait|not a dealbreaker)\b|괜찮|상관\s*없", clause, re.I,
        ):
            return True
    return False


def _concept_for_term(term: str) -> str:
    normalized = _normalize(term)
    for concept, aliases in _CONCEPT_ALIASES.items():
        if normalized == concept or normalized in {_normalize(alias) for alias in aliases}:
            return "waiting" if concept in {"short_wait", "long_wait"} else concept
    return normalized


@dataclass(frozen=True)
class TermEdit:
    action: Literal["add", "remove", "replace"]
    field: str
    term: str
    replacement: str | None = None


def _term_edits(message: str, proposal: Mapping[str, Any]) -> tuple[TermEdit, ...]:
    operations = proposal.get("term_operations")
    if not isinstance(operations, list):
        return ()
    edits = []
    for item in operations[:16]:
        if not isinstance(item, dict):
            continue
        action, field = item.get("action"), item.get("field")
        if action not in {"add", "remove", "replace"} or field not in _PREFERENCE_FIELDS:
            continue
        term, span = _optional_text(item.get("term")), _optional_text(item.get("source_span"))
        if term is None or span is None or span not in message:
            continue
        if not term_in_clause(term, span) or (is_inquiry(span) and not is_exclusion(span)):
            continue
        replacement = _optional_text(item.get("replacement"))
        if action == "replace":
            if replacement is None or not term_in_clause(replacement, span) or not re.search(r"\breplace\b|\binstead\b|대신|바꿔|변경", span, re.I):
                continue
        elif action == "remove" and not is_withdrawal(span):
            continue
        elif action == "add":
            cleaned = augment_extracted_facets(span, {"soft_keywords" if field == "keywords" else field: [term]})
            if term not in cleaned.get("soft_keywords" if field == "keywords" else field, []):
                continue
        edits.append(TermEdit(action, field, term, replacement))
    return tuple(edits)


def _detect_required_hours(
    message: str,
    proposal: Mapping[str, Any],
) -> tuple[str, ...] | None:
    proposed = list(_optional_terms(proposal.get("required_hours")) or ())
    requested = any(pattern.search(clause) and not is_inquiry(clause)
                    for pattern in _TUESDAY_EVENING_PATTERNS
                    for clause in intent_clauses(message))
    if requested:
        proposed.append("tuesday_evening")
    output: list[str] = []
    for value in proposed:
        normalized = _normalize(value).replace(" ", "_")
        if normalized in {"tuesday_evening", "tuesday_night", "화요일_야간", "화요일_저녁"}:
            if not requested:
                continue
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
    term_edits: tuple[TermEdit, ...]
    visit_reason: str | None
    inquiries: tuple[str, ...]
    user_message: str
    extraction_source: str | None
    extraction_error: str | None


def compile_turn_delta(message: str, proposal: Mapping[str, Any] | None) -> SearchDelta:
    """Combine an untrusted model proposal with deterministic user-text facts."""
    raw = dict(proposal or {})
    augmented = augment_extracted_facets(message, raw)
    distance_km = _explicit_distance_km(message)

    location = _optional_text(augmented.get("location"))
    normalized_location = _normalize(location) if location else None
    citywide = True if requests_citywide_search(message) else None
    if normalized_location in _SEOUL_ALIASES:
        location = None
    elif location is not None:
        citywide = False

    latitude = _optional_float(augmented.get("latitude"))
    longitude = _optional_float(augmented.get("longitude"))
    if (latitude is None) != (longitude is None):
        latitude = None
        longitude = None

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
        term_edits=_term_edits(message, raw),
        visit_reason=_optional_text(augmented.get("visit_reason")),
        inquiries=_optional_terms(augmented.get("inquiries")) or (),
        user_message=message,
        extraction_source=_optional_text(augmented.get("extraction_source")),
        extraction_error=_optional_text(augmented.get("extraction_error")),
    )


def _merge_terms(
    existing: list[str],
    incoming: tuple[str, ...] | None,
) -> list[str]:
    if incoming is None:
        return list(existing)
    values = [*existing, *incoming]
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
    aliases = _CONCEPT_ALIASES.get(concept)
    if aliases is None:
        return term_in_clause(concept, term)
    return any(term_in_clause(_normalize(alias), normalized) for alias in aliases)


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
    state.visit_reason = None
    state.inquiries = []
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
) -> State:
    """Apply a compiled delta without allowing the model to rewrite old state."""
    state = current.model_copy(deep=True)
    if delta.operation == "replace_context":
        _clear_search_context(state)

    if delta.specialty is not None:
        state.specialty = delta.specialty
        state.specialty_confidence = delta.specialty_confidence or 0.7

    location_changed = delta.location is not None and (
        state.location is None or _normalize(delta.location) != _normalize(state.location)
    )
    if delta.citywide is True:
        state.place_terms = []
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
        if location_changed:
            state.place_terms = []
        state.location = delta.location
        state.is_citywide_search = False
        for field in ("latitude", "longitude", "address_korean", "district", "dong"):
            value = getattr(delta, field)
            if value is not None or location_changed:
                setattr(state, field, value)
        if state.latitude is not None and state.longitude is not None:
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
        elif location_changed and current.is_citywide_search:
            # A city-wide radius is not a local distance preference.
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
            ),
        )

    for concept in delta.remove_concepts:
        for field in _PREFERENCE_FIELDS:
            setattr(
                state,
                field,
                [
                    term
                    for term in getattr(state, field)
                    if not _term_matches_concept(term, concept)
                ],
            )

    for edit in delta.term_edits:
        if edit.action in {"remove", "replace"}:
            for field in _PREFERENCE_FIELDS:
                setattr(state, field, [term for term in getattr(state, field)
                                      if not _term_matches_concept(term, _concept_for_term(edit.term))])
        addition = edit.replacement if edit.action == "replace" else edit.term if edit.action == "add" else None
        if addition is not None:
            setattr(state, edit.field, _merge_terms(getattr(state, edit.field), (addition,)))

    if delta.visit_reason is not None:
        state.visit_reason = delta.visit_reason
    state.inquiries = list(delta.inquiries)
    state.language_pref, explicit = response_language(
        delta.user_message, current.language_pref,
        established=bool(current.turn_count or current.explicit_response_language),
    )
    if explicit is not None:
        state.explicit_response_language = explicit
    constraint_fields = (*field_map, "specialty", "location", "latitude", "longitude",
                         "max_distance_km", "is_citywide_search", "visit_reason", "inquiries")
    if any(getattr(state, field) != getattr(current, field) for field in constraint_fields):
        state.clear_retrieval_telemetry()
    state.extraction_source = delta.extraction_source or state.extraction_source
    state.extraction_error = delta.extraction_error
    return state
