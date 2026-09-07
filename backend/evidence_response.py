"""Qualify facility replies while retaining evidence for internal checks."""

from copy import deepcopy


def finalize_evidence_response(response, results, metadata, language):
    """Return facility replies and matching cards, withholding unsafe endorsements.

    Retrieval roles describe search intent, not verified sentiment or suitability.
    Risk matches therefore require review; they do not prove a negative claim.
    """
    cards = deepcopy(results)
    korean = language == "Korean"
    incomplete = metadata.get("retrieval_status") in {
        "incomplete", "error", "fallback", "degraded",
    } or metadata.get("coverage_sufficient") is False
    requires_review = False
    for card in cards:
        facility_id = str(card.get("place_id", ""))
        groups = card.get("retrieval_evidence_groups") or {}
        warnings = groups.get("warnings", [])
        selected = card.get("retrieval_evidence", [])
        records = []
        seen = set()
        invalid = False
        for item in [*warnings, *selected]:
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
        requires_review |= risk or invalid or bool(unverified)
        card["recommendation_status"] = (
            "not_established" if incomplete or invalid or unverified
            else "requires_review" if risk else "evidence_available"
        )
    if incomplete:
        response = (
            "검색이 완료되지 않아 조건을 충족하는 시설을 확인할 수 없습니다. "
            "아래는 검증된 추천이 아닌 검색 후보입니다. 조건은 변경하지 않았습니다."
            if korean else
            "Search is incomplete, so I cannot establish which facilities meet your requirements. "
            "These are search candidates, not verified recommendations. Your constraints are unchanged."
        )
    elif requires_review:
        response = (
            "아래 시설이 요청하신 모든 조건을 충족하는지는 확인되지 않았습니다. 방문 전에 시설에 확인해 주세요."
            if korean else
            "Some requirements remain unconfirmed. Please check with the facility before visiting. "
            "These candidates are not established as meeting all your requirements."
        )
    return response, cards
