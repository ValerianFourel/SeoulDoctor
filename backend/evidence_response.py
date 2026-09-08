"""Prepare concise replies and evidence cards while retaining original records."""

from copy import deepcopy
from math import isfinite
from numbers import Real

from models import State
from review_presentation import prepare_review_presentations
from search.rules import SPECIALTY_IDS


_INCOMPLETE_STATUSES = {"incomplete", "error", "fallback", "degraded"}
_GENERIC_LOCATIONS = {
    "current location",
    "current map position",
    "my location",
    "seoul",
    "seoul city",
    "서울",
    "서울시",
    "서울특별시",
}
_INCOMPLETE_WITH_CARDS = {
    "English": (
        "Search is incomplete, so some requirements could not be verified. "
        "These are search candidates, and your constraints are unchanged."
    ),
    "Korean": (
        "검색이 완료되지 않아 일부 조건을 확인하지 못했습니다. "
        "아래 시설은 검색 후보이며, 요청하신 조건은 변경하지 않았습니다."
    ),
}
_INCOMPLETE_WITHOUT_CARDS = {
    "English": "Search is incomplete, so I could not verify all your requirements. Your constraints are unchanged.",
    "Korean": "검색이 완료되지 않아 일부 조건을 확인하지 못했습니다. 요청하신 조건은 변경하지 않았습니다.",
}


def _clean_label(value):
    if not isinstance(value, str):
        return None
    cleaned = " ".join(value.split())
    return cleaned or None


def _location_label(state, korean):
    if state is None or state.is_citywide_search:
        return None
    location = _clean_label(state.location)
    if location and location.casefold() not in _GENERIC_LOCATIONS:
        return location
    if not (
        state.latitude is not None
        and -90 <= state.latitude <= 90
        and isfinite(state.latitude)
        and state.longitude is not None
        and -180 <= state.longitude <= 180
        and isfinite(state.longitude)
    ):
        return None
    return (
        _clean_label(state.address_korean)
        or ("검색 기준 위치" if korean else "your search location")
    )


def _specialty_label(state, korean):
    if (
        state is None
        or not isfinite(state.specialty_confidence)
        or state.specialty_confidence < 0.7
    ):
        return None
    specialty = _clean_label(state.specialty)
    if not specialty:
        return None
    if korean:
        return specialty
    identifier = SPECIALTY_IDS.get(specialty)
    if identifier:
        return identifier.replace("_", " ")
    return specialty if specialty.isascii() else None


def _card_distance(card):
    key = "distance_km" if "distance_km" in card else "distance"
    value = card.get(key)
    if (
        not isinstance(value, Real)
        or isinstance(value, bool)
        or not isfinite(float(value))
        or value < 0
    ):
        return None
    return float(value)


def _summary(cards, state, korean):
    count = len(cards)
    specialty = _specialty_label(state, korean)
    location = _location_label(state, korean)
    citywide = bool(state and state.is_citywide_search)
    if korean:
        search = f"{specialty} 검색 결과로 " if specialty else ""
        if citywide:
            opening = f"서울 전역에서 {search}시설 {count}곳을 찾았습니다."
        elif location:
            opening = f"{location} 주변에서 {search}시설 {count}곳을 찾았습니다."
        else:
            opening = f"{search}시설 {count}곳을 찾았습니다."
    else:
        noun = "option" if count == 1 else "options"
        search = f" for {specialty}" if specialty else ""
        if citywide:
            opening = f"I found {count} {noun}{search} across Seoul."
        elif location:
            opening = f"I found {count} {noun}{search} around {location}."
        else:
            opening = f"I found {count} {noun}{search}."

    if not location:
        return opening
    distances = [_card_distance(card) for card in cards]
    if any(distance is None for distance in distances):
        return opening
    closest_index = min(range(len(cards)), key=distances.__getitem__)
    name = _clean_label(cards[closest_index].get("name"))
    if not name:
        return opening
    distance = distances[closest_index]
    if korean:
        closest = (
            f"표시된 후보 중 {name}의 직선거리가 가장 짧으며, "
            f"{location} 기준 약 {distance:.1f} km입니다."
        )
    else:
        closest = (
            f"Of the options shown, {name} is closest, about {distance:.1f} km "
            f"from {location} by straight-line distance."
        )
    return f"{opening} {closest}"


def _follow_up(state, korean):
    default_distance = bool(
        state
        and not state.is_citywide_search
        and state.search_mode == "distance"
        and _location_label(state, korean)
        and isfinite(state.max_distance_km)
        and abs(state.max_distance_km - 5.0) < 1e-9
        and isfinite(state.travel_confidence)
        and state.travel_confidence <= 0.5
    )
    needs_concern = not (state and state.disease_terms)
    first_concern_prompt = needs_concern and (
        state is None or state.turn_count <= 1
    )
    if first_concern_prompt:
        if korean:
            if default_distance:
                return (
                    "어떤 진료나 증상으로 도움이 필요하신가요? "
                    "원하시면 1 km 이내처럼 이동 거리나 언어 또는 접근성 요구사항도 알려 주세요."
                )
            return (
                "어떤 진료나 증상으로 도움이 필요하신가요? "
                "원하시면 언어 또는 접근성 요구사항도 알려 주세요."
            )
        if default_distance:
            return (
                "What would you like the doctor to help with? You can also narrow "
                "the search to within 1 km, or add language or access needs."
            )
        return (
            "What would you like the doctor to help with? You can also add "
            "language or access needs."
        )

    if korean:
        if default_distance:
            return "아래 카드를 비교하거나 1 km 이내 같은 조건으로 검색을 더 좁힐 수 있습니다."
        return "아래 카드를 비교하거나 원하시는 조건으로 검색을 더 조정할 수 있습니다."
    if default_distance:
        return (
            "You can compare the cards below or refine this search, for example "
            "to options within 1 km."
        )
    return "You can compare the cards below or refine this search further."


def finalize_evidence_response(
    response,
    results,
    metadata,
    language,
    *,
    state: State | None = None,
    translation_api_key="",
):
    """Return visible evidence and matching cards, withholding unsafe endorsements.

    Retrieval roles describe search intent, not verified sentiment or suitability.
    Risk matches therefore require review; they do not prove a negative claim.
    """
    cards = deepcopy(results)
    korean = language == "Korean"
    incomplete = (
        metadata.get("retrieval_status") in _INCOMPLETE_STATUSES
        or metadata.get("coverage_sufficient") is False
    )
    has_risk = False
    has_unverified = False
    has_invalid_ownership = False
    for card in cards:
        facility_id = str(card.get("place_id", ""))
        groups = card.get("retrieval_evidence_groups") or {}
        warnings = groups.get("warnings", [])
        selected = card.get("retrieval_evidence", [])
        records = []
        seen = set()
        invalid = False
        for item in [*warnings, *selected, *groups.get("supporting", [])]:
            if not isinstance(item, dict):
                invalid = True
                continue
            if str(item.get("place_id", "")) != facility_id:
                invalid = True
                continue
            evidence_id = item.get("evidence_id")
            if not evidence_id or evidence_id in seen:
                continue
            seen.add(evidence_id)
            records.append(item)
        # Invalid ownership must not survive in either the reply or its cards.
        card["retrieval_evidence"] = records
        for role in ("supporting", "warnings"):
            if role in groups:
                groups[role] = [item for item in groups[role]
                                if isinstance(item, dict)
                                and str(item.get("place_id", "")) == facility_id]
        risk = bool(groups.get("warnings")) or any(
            item.get("evidence_role") in {"risk", "mixed"} for item in records
        )
        unverified = groups.get("unverified", [])
        has_risk |= risk
        has_unverified |= bool(unverified)
        has_invalid_ownership |= invalid
        card["recommendation_status"] = (
            "not_established" if incomplete or invalid or unverified
            else "requires_review" if risk else "evidence_available"
        )

    if not cards:
        if incomplete and (
            (korean and "검색이 완료되지 않아" not in response)
            or (not korean and "Search is incomplete" not in response)
        ):
            response = f"{response}\n\n{_INCOMPLETE_WITHOUT_CARDS[language]}"
        prepare_review_presentations(
            cards,
            language,
            translation_api_key=translation_api_key,
        )
        return response, cards

    paragraphs = [_summary(cards, state, korean)]
    if incomplete:
        paragraphs.append(_INCOMPLETE_WITH_CARDS[language])
    elif has_unverified or has_invalid_ownership:
        paragraphs.append(
            "일부 조건은 아직 확인되지 않아 모든 요청 조건을 충족한다고 볼 수 없습니다."
            if korean else
            "Some requirements remain unverified, so these options are not established as meeting all your requirements."
        )
    if has_risk:
        paragraphs.append(
            "일부 후기는 요청하신 선호 조건과 비교해 더 자세히 확인해야 합니다. 검색 일치만으로 문제가 확인된 것은 아닙니다."
            if korean else
            "Some reviews need a closer look against your preferences. A search match does not confirm a problem."
        )
    paragraphs.append(_follow_up(state, korean))
    response = "\n\n".join(paragraphs)
    prepare_review_presentations(cards, language, translation_api_key=translation_api_key)
    return response, cards
