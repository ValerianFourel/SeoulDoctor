"""Bilingual query-facet safeguards for medical-facility retrieval.

GPT-OSS remains the primary extractor.  These helpers preserve obvious search
facets when the model/provider returns incomplete JSON so one failed LLM call
cannot silently turn a specific request into a city-wide generic search.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


SPECIALTY_ALIASES: Sequence[Tuple[str, Sequence[str]]] = (
    ("산부인과", ("gynecologist", "gynaecologist", "ob/gyn", "obgyn", "산부인과")),
    ("치과", ("dentist", "dental", "치과")),
    ("피부과", ("dermatologist", "dermatology", "skin doctor", "피부과")),
    ("내과", ("internal medicine", "internist", "내과")),
    ("정형외과", ("orthopedist", "orthopedic", "orthopaedic", "정형외과")),
    ("소아청소년과", ("pediatrician", "paediatrician", "pediatrics", "소아청소년과", "소아과")),
    ("정신건강의학과", ("psychiatrist", "psychiatry", "정신건강의학과", "정신과")),
    ("안과", ("ophthalmologist", "ophthalmology", "eye doctor", "안과")),
    ("이비인후과", ("ent", "otolaryngologist", "ear nose throat", "이비인후과")),
    ("비뇨의학과", ("urologist", "urology", "비뇨의학과", "비뇨기과")),
    ("신경과", ("neurologist", "neurology", "신경과")),
    ("가정의학과", ("family medicine", "family doctor", "가정의학과")),
    ("외과", ("surgeon", "surgery", "외과")),
)

PLACE_ALIASES: Sequence[Tuple[str, Sequence[str]]] = (
    ("강남구", ("gangnam", "gangnam-gu", "강남", "강남구")),
    ("마포구", ("mapo", "mapo-gu", "마포", "마포구", "hongdae", "홍대")),
    ("종로구", ("jongno", "jongno-gu", "종로", "종로구")),
    ("서초구", ("seocho", "seocho-gu", "서초", "서초구")),
    ("송파구", ("songpa", "songpa-gu", "송파", "송파구")),
    ("용산구", ("yongsan", "yongsan-gu", "용산", "용산구", "itaewon", "이태원")),
    ("영등포구", ("yeongdeungpo", "yeouido", "영등포", "영등포구", "여의도")),
    ("중구", ("jung-gu", "myeongdong", "city hall", "중구", "명동", "시청")),
)

GENDER_ALIASES: Sequence[Tuple[str, Sequence[str]]] = (
    ("female", ("female", "woman doctor", "women doctor", "lady doctor", "여성", "여의사", "여자 의사")),
    ("male", ("male", "man doctor", "men doctor", "남성", "남의사", "남자 의사")),
)

DISEASE_ALIASES: Sequence[Tuple[str, Sequence[str]]] = (
    ("endometriosis", ("endometriosis", "자궁내막증")),
    ("PCOS", ("pcos", "polycystic ovary", "다낭성 난소", "다낭성난소증후군")),
    ("diabetes", ("diabetes", "diabetic", "당뇨", "당뇨병")),
    ("hypertension", ("hypertension", "high blood pressure", "고혈압")),
    ("asthma", ("asthma", "천식")),
    ("eczema", ("eczema", "atopic dermatitis", "습진", "아토피")),
    ("migraine", ("migraine", "편두통")),
    ("thyroid", ("thyroid", "갑상선")),
    ("depression", ("depression", "depressive", "우울증")),
    ("anxiety", ("anxiety", "anxiety disorder", "불안", "불안장애")),
    ("back pain", ("back pain", "lower back pain", "허리 통증", "요통")),
)

COMMENT_ALIASES: Sequence[Tuple[str, Sequence[str]]] = (
    ("clear explanations", ("clear explanation", "clear explanations", "explains well", "explained well", "설명을 잘", "설명 잘", "자세한 설명")),
    ("friendly", ("friendly", "kind", "친절", "상냥")),
    ("thorough", ("thorough", "careful", "detailed", "꼼꼼", "세심")),
    ("short wait", ("short wait", "no long wait", "빠른 대기", "대기 시간이 짧", "대기시간 짧")),
)


def _contains(text: str, alias: str) -> bool:
    if re.fullmatch(r"[a-z0-9][a-z0-9 /-]*", alias):
        return bool(re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", text))
    return alias in text


def _find_aliases(query: str, aliases: Sequence[Tuple[str, Sequence[str]]]) -> List[str]:
    normalized = query.casefold()
    return [canonical for canonical, variants in aliases if any(_contains(normalized, item.casefold()) for item in variants)]


def _clean_list(value: Any) -> List[str]:
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes, Mapping)):
        return []
    output: List[str] = []
    for item in value:
        text = str(item or "").strip()
        if text and text.casefold() not in {existing.casefold() for existing in output}:
            output.append(text)
    return output[:8]


def _merge_terms(*groups: Iterable[str]) -> List[str]:
    output: List[str] = []
    for group in groups:
        for item in group:
            text = str(item or "").strip()
            if text and text.casefold() not in {existing.casefold() for existing in output}:
                output.append(text)
    return output[:8]


def augment_extracted_facets(query: str, payload: Mapping[str, Any] | None) -> Dict[str, Any]:
    """Merge model output with conservative bilingual literal safeguards."""
    result = dict(payload or {})

    specialties = _find_aliases(query, SPECIALTY_ALIASES)
    places = _find_aliases(query, PLACE_ALIASES)
    genders = _find_aliases(query, GENDER_ALIASES)
    diseases = _find_aliases(query, DISEASE_ALIASES)
    comment_terms = _find_aliases(query, COMMENT_ALIASES)

    result["place_terms"] = _merge_terms(_clean_list(result.get("place_terms")), places)
    result["gender_terms"] = _merge_terms(_clean_list(result.get("gender_terms")), genders)
    result["disease_terms"] = _merge_terms(_clean_list(result.get("disease_terms")), diseases)
    result["comment_terms"] = _merge_terms(_clean_list(result.get("comment_terms")), comment_terms)

    if not result.get("specialty") and specialties:
        result["specialty"] = specialties[0]
        result["specialty_confidence"] = 0.95
    if not result.get("location") and places:
        result["location"] = places[0]
    result.setdefault("travel_label", "Moderate")
    result.setdefault("hard_keywords", [])
    result.setdefault("soft_keywords", [])
    result.setdefault("negative_hard_keywords", [])
    result.setdefault("negative_keywords", [])
    result.setdefault(
        "extraction_source",
        "gpt_oss+deterministic_safeguards" if payload else "deterministic_fallback",
    )
    return result


def retrieval_terms_from_state(state: Any) -> List[str]:
    """Build literal BM25 terms while preserving structured debug facets."""
    expansions = {
        "female": ("female", "여성", "여의사"),
        "male": ("male", "남성", "남의사"),
        "endometriosis": ("endometriosis", "자궁내막증"),
        "pcos": ("PCOS", "다낭성난소증후군"),
        "clear explanations": ("clear explanations", "explains well", "설명 잘"),
        "friendly": ("friendly", "친절"),
        "thorough": ("thorough", "꼼꼼"),
    }

    facet_terms: List[str] = []
    for field in ("gender_terms", "disease_terms", "comment_terms"):
        for term in getattr(state, field, []):
            facet_terms.extend(expansions.get(str(term).casefold(), (term,)))
    return _merge_terms(
        getattr(state, "hard_keywords", []),
        facet_terms,
    )
