"""Render retrieved originals without delegating quotation integrity to a model."""

from copy import deepcopy
from html import escape
import re


def _display(text: object) -> str:
    # Quote source data literally; never let review text create Markdown links.
    return re.sub(r"([\\`*_{}\[\]()#+!|>])", r"\\\1", escape(str(text)))


def finalize_evidence_response(response, results, metadata, language):
    """Return visible evidence and matching cards, withholding unsafe endorsements.

    Retrieval roles describe search intent, not verified sentiment or suitability.
    Risk matches therefore require review; they do not prove a negative claim.
    """
    cards = deepcopy(results)
    korean = language == "Korean"
    incomplete = metadata.get("retrieval_status") in {
        "incomplete", "error", "fallback", "degraded",
    } or metadata.get("coverage_sufficient") is False
    sections = []
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
        lines = [f'### {_display(card.get("name", facility_id))}']
        if risk:
            lines.append(
                "주의: 피하고 싶은 조건과 관련된 후기가 있습니다. 조건을 모두 충족한다고 추천할 수 없습니다. "
                "검색 일치만으로 부정적인 내용이 확인된 것은 아닙니다."
                if korean else
                "Caution: reviews relevant to your avoidance preferences need review. "
                "I cannot recommend this as meeting all your requirements. "
                "A search match alone does not establish a negative claim."
            )
        if invalid or unverified:
            lines.append("일부 조건은 근거로 확인되지 않았습니다." if korean else
                         "Some requirements are not established by the available evidence.")
        quoted = False
        for item in records:
            if not item.get("is_verbatim") or not isinstance(item.get("text"), str):
                continue
            quoted = True
            label = "원문 후기" if korean else "Original review"
            lines.append(f'{label}:\n\n> ' + _display(item["text"]).replace("\n", "\n> "))
            lines.append(
                f'{"출처" if korean else "Source"}: '
                f'{_display(facility_id)} / {_display(item["evidence_id"])}'
            )
            requirements = item.get("matched_constraint_ids", [])
            if requirements:
                lines.append(
                    ("관련 검색 조건: " if korean else "Retrieved for requirement: ")
                    + ", ".join(_display(value) for value in requirements)
                )
        if not quoted:
            lines.append("인용할 수 있는 원문 후기를 찾지 못했습니다." if korean else
                         "No original review is available to quote.")
        sections.append("\n\n".join(lines))
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
            "아래 후보의 후기와 확인되지 않은 조건을 검토해 주세요. 모든 조건을 충족하는 추천은 아닙니다."
            if korean else
            "Please review the evidence and unresolved requirements below. "
            "These candidates are not established as meeting all your requirements."
        )
    if sections:
        disclaimer = ("후기는 환자의 경험이며 사실이나 의료적 보장을 의미하지 않습니다." if korean else
                      "Reviews describe patient experiences, not verified facts or clinical guarantees.")
        response = "\n\n".join([response, disclaimer, *sections])
    return response, cards
