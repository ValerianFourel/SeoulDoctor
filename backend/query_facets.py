"""Bilingual query-facet safeguards for medical-facility retrieval.

GPT-OSS remains the primary extractor.  These helpers preserve obvious search
facets when the model/provider returns incomplete JSON so one failed LLM call
cannot silently turn a specific request into a city-wide generic search.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from config import DISTANCE_MAPPING


SPECIALTY_ALIASES: Sequence[Tuple[str, Sequence[str]]] = (
    ("산부인과", ("gynecologist", "gynaecologist", "ob/gyn", "obgyn", "산부인과")),
    ("치과", ("dentist", "dental", "치과")),
    ("피부과", ("dermatologist", "dermatology", "skin doctor", "피부과")),
    ("내과", ("internal medicine", "internist", "내과")),
    ("정형외과", ("orthopedist", "orthopedic", "orthopedi", "orthopedics", "orthopaedic", "정형외과")),
    ("소아청소년과", ("pediatrician", "paediatrician", "pediatrics", "소아청소년과", "소아과")),
    ("정신건강의학과", ("psychiatrist", "psychiatry", "정신건강의학과", "정신과")),
    ("안과", ("ophthalmologist", "ophthalmology", "eye doctor", "안과")),
    ("이비인후과", ("ent", "otolaryngologist", "ear nose throat", "이비인후과")),
    ("비뇨의학과", ("urologist", "urology", "비뇨의학과", "비뇨기과")),
    ("신경과", ("neurologist", "neurology", "신경과")),
    ("가정의학과", ("family medicine", "family doctor", "가정의학과")),
    ("외과", ("surgeon", "surgery", "외과")),
)
GENERIC_FACILITY_NOUNS = {
    "clinic",
    "doctor",
    "facility",
    "hospital",
    "의료기관",
    "의원",
    "병원",
}

PLACE_ALIASES: Sequence[Tuple[str, Sequence[str]]] = (
    ("강남구", ("gangnam", "gangnam-gu", "강남", "강남구")),
    ("강동구", ("gangdong", "gangdong-gu", "강동", "강동구")),
    ("강북구", ("gangbuk", "gangbuk-gu", "강북", "강북구")),
    ("강서구", ("gangseo", "gangseo-gu", "강서", "강서구")),
    ("관악구", ("gwanak", "gwanak-gu", "관악", "관악구")),
    ("광진구", ("gwangjin", "gwangjin-gu", "광진", "광진구")),
    ("구로구", ("guro", "guro-gu", "구로", "구로구")),
    ("금천구", ("geumcheon", "geumcheon-gu", "금천", "금천구")),
    ("노원구", ("nowon", "nowon-gu", "노원", "노원구")),
    ("도봉구", ("dobong", "dobong-gu", "도봉", "도봉구")),
    ("동대문구", ("dongdaemun", "dongdaemun-gu", "동대문", "동대문구")),
    ("동작구", ("dongjak", "dongjak-gu", "동작", "동작구")),
    ("마포구", ("mapo", "mapo-gu", "마포", "마포구", "hongdae", "홍대")),
    ("서대문구", ("seodaemun", "seodaemun-gu", "서대문", "서대문구")),
    ("서초구", ("seocho", "seocho-gu", "서초", "서초구")),
    ("성동구", ("seongdong", "seongdong-gu", "성동", "성동구")),
    ("성북구", ("seongbuk", "seongbuk-gu", "성북", "성북구")),
    ("송파구", ("songpa", "songpa-gu", "송파", "송파구")),
    ("양천구", ("yangcheon", "yangcheon-gu", "양천", "양천구")),
    ("용산구", ("yongsan", "yongsan-gu", "용산", "용산구", "itaewon", "이태원")),
    (
        "영등포구",
        (
            "yeongdeungpo",
            "yeongdeungpo-gu",
            "yeouido",
            "영등포",
            "영등포구",
            "여의도",
        ),
    ),
    ("은평구", ("eunpyeong", "eunpyeong-gu", "은평", "은평구")),
    ("종로구", ("jongno", "jongno-gu", "종로", "종로구")),
    ("중구", ("jung-gu", "myeongdong", "city hall", "중구", "명동", "시청")),
    ("중랑구", ("jungnang", "jungnang-gu", "중랑", "중랑구")),
)

GENDER_ALIASES: Sequence[Tuple[str, Sequence[str]]] = (
    ("female", ("female", "woman doctor", "women doctor", "lady doctor", "여성", "여의사", "여자 의사")),
    ("male", ("male", "man doctor", "men doctor", "남성", "남의사", "남자 의사")),
)

DISEASE_ALIASES: Sequence[Tuple[str, Sequence[str]]] = (
    ("foot issue", ("foot issue", "foot problem", "발 문제")),
    ("foot pain", ("foot pain", "pain in my foot", "발 통증", "발이 아파")),
    ("ankle pain", ("ankle pain", "ankle hurts", "발목 통증", "발목이 아파")),
    ("ankle sprain", ("ankle sprain", "sprained ankle", "twisted my ankle", "발목 염좌", "발목 삐", "발목 겹질")),
    ("cheilitis", ("cheilitis", "구순염", "입술염")),
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

DISTANCE_PATTERN = re.compile(
    r"(?P<value>\d+(?:[.,]\d+)?)\s*"
    r"(?P<unit>km|kilometers?|kilometres?|킬로미터|m|meters?|metres?|미터)"
    r"(?![a-z가-힣])",
    re.IGNORECASE,
)
TRAVEL_PHRASES: Sequence[Tuple[str, Sequence[str]]] = (
    ("Anywhere in Seoul", ("anywhere in seoul", "all seoul", "서울 전역", "서울 전체")),
    ("Willing to Travel", ("willing to travel", "travel farther", "travel further", "멀리 이동")),
    ("Flexible", ("flexible distance", "flexible radius", "거리 상관", "반경 상관")),
    ("Walking Distance", ("walking distance", "on foot", "도보")),
    ("Nearby", ("nearby", "very close", "아주 가까", "근거리")),
)
TRAVEL_INTENT_PATTERN = re.compile(
    r"\b(radius|distance|closer|farther|further|travel)\b|반경|거리|이내|가까|멀리",
    re.IGNORECASE,
)
PRESERVE_TRAVEL_SCOPE_PATTERNS = (
    re.compile(
        r"\b(?:same|current|existing|previous)\b.{0,32}"
        r"\b(?:search|radius|distance|range|area|scope)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:rather than|without|do not|don't)\b.{0,32}"
        r"\b(?:broaden|broadening|widen|expand|change|increase)\b",
        re.IGNORECASE,
    ),
    re.compile(r"(?:범위|반경|거리)(?:를|을)?\s*.{0,16}(?:넓히지|바꾸지|변경하지|유지)"),
    re.compile(r"(?:같은|기존|현재).{0,16}(?:검색|범위|반경|거리|주변)"),
)

COMMENT_ALIASES: Sequence[Tuple[str, Sequence[str]]] = (
    ("clear explanations", ("clear explanation", "clear explanations", "explains well", "explained well", "explain things clearly", "설명을 잘", "설명 잘", "자세한 설명")),
    ("friendly", ("friendly", "kind", "친절", "상냥")),
    ("thorough", ("thorough", "careful", "detailed", "꼼꼼", "세심")),
    ("no overprescribing", ("no overprescribing", "avoids overprescribing", "does not overprescribe", "don't over-prescribe", "restrained prescribing", "과잉 처방 안", "과잉처방 안", "과하게 약처방하지", "과잉 처방 없는")),
    (
        "short wait",
        (
            "short wait",
            "no wait",
            "no waiting",
            "no long wait",
            "빠른 대기",
            "대기 없음",
            "대기가 없",
            "대기 시간이 짧",
            "대기시간 짧",
        ),
    ),
)

MULTILINGUAL_RETRIEVAL_GROUPS: Sequence[Sequence[str]] = (
    ("foot issue", "foot problem", "발 문제", "발"),
    ("foot pain", "pain in my foot", "발 통증", "발이 아파", "발"),
    ("ankle pain", "ankle hurts", "발목 통증", "발목이 아파", "발목"),
    ("ankle sprain", "sprained ankle", "twisted my ankle", "발목 염좌", "발목 삐", "발목 겹질", "발목"),
    ("female", "woman doctor", "여성", "여의사"),
    ("male", "man doctor", "남성", "남의사"),
    ("endometriosis", "자궁내막증"),
    ("PCOS", "polycystic ovary", "다낭성난소증후군"),
    ("cheilitis", "lip inflammation", "구순염", "입술염", "입술 염증"),
    (
        "clear explanations",
        "detailed explanations",
        "explains well",
        "자세한 설명",
        "상세한 설명",
        "설명 잘",
        "설명도 잘",
    ),
    ("fast treatment", "quick treatment", "fast care", "빠른 진료", "빠른 치료", "신속한 진료"),
    ("friendly", "kind", "친절"),
    ("thorough", "careful", "꼼꼼", "세심"),
    (
        "short wait",
        "no wait",
        "no waiting",
        "대기 없음",
        "대기가 없",
        "대기 시간이 짧",
        "대기시간 짧",
    ),
    (
        "no overprescribing",
        "avoids overprescribing",
        "does not overprescribe",
        "과잉 처방 안",
        "과잉처방 안",
        "과하게 약처방",
    ),
    (
        "kind pediatric care",
        "kind with children",
        "gentle with children",
        "아이에게 친절",
        "아이한테 친절",
        "아이에게 다정",
        "아이한테 다정",
    ),
    (
        "unfriendly nurses",
        "rude nurses",
        "간호사 불친절",
        "불친절한 간호사",
        "간호사 무뚝뚝",
        "간호사분들은 좀 무뚝뚝",
    ),
    (
        "aggressive upselling",
        "pressure sales",
        "pushed unnecessary treatment",
        "과도한 시술 권유",
        "과잉 치료 권유",
        "강매",
    ),
)


HOUR_RETRIEVAL_TERMS: Mapping[str, Sequence[str]] = {
    "tuesday_evening": (
        "Tuesday evening",
        "Tuesday night",
        "화요일 저녁",
        "화요일 야간",
    ),
}

def _contains(text: str, alias: str) -> bool:
    if alias == "친절":
        return bool(re.search(r"(?<!불)친절", text))
    if re.fullmatch(r"[a-z0-9][a-z0-9 /-]*", alias):
        return bool(re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", text))
    return alias in text


def _find_aliases(query: str, aliases: Sequence[Tuple[str, Sequence[str]]]) -> List[str]:
    normalized = query.casefold()
    return [canonical for canonical, variants in aliases if any(_contains(normalized, item.casefold()) for item in variants)]


_INQUIRY = re.compile(
    r"\?|\b(?:can you confirm|could you confirm|whether|is there|are there|"
    r"do they|does (?:the|this)|what (?:do|does|would)|should i check|is .+ available)\b"
    r"|가능한지|알려\s*주|인가요|있나요|어떤\s*의미", re.I,
)
_REQUIRE = re.compile(
    r"\b(?:must|mandatory|required|require|need|essential|only|non-negotiable)\b"
    r"|필수|반드시|꼭|필요", re.I,
)
_WITHDRAW = re.compile(
    r"\b(?:don['’]?t|do not|no longer)\s+(?:require|need|mind|care)\b"
    r"|\b(?:(?:not|isn['’]?t) (?:required|necessary|mandatory)|no longer (?:required|needed|important|matters)|doesn['’]?t matter|drop|remove)\b"
    r"|필요\s*없|필수가\s*아니|상관없|더\s*이상.{0,12}중요하지", re.I,
)
_EXCLUSION = re.compile(
    r"\b(?:avoid|exclude|skip|without|don['’]?t want|do not want)\b"
    r"|피하|피하고|제외|원하지\s*않|싫", re.I,
)
_RESPONSE_DIRECTIVE = re.compile(
    r"\b(?:reply|respond|answer|write|continue)(?:\s+to\s+me)?\s+in\s+English\b"
    r"|\bEnglish\s+please\b|영어로\s*(?:답변|대답|응답|말|해|부탁)[^,.!?;]*", re.I,
)


def intent_clauses(query: str) -> List[str]:
    """Keep negation and requests attached to their own sentence or contrast."""
    return [part.strip() for part in re.split(
        r"(?<=[.!?;])\s+|\n|\s+(?:but|however|다만|하지만)\s+", query, flags=re.I,
    ) if part.strip()]


def is_inquiry(clause: str) -> bool:
    if re.search(r"\b(?:can|could|would)\s+you\s+(?:please\s+)?(?:find|search|show|look)\b", clause, re.I):
        return False
    return bool(_INQUIRY.search(clause))


def is_exclusion(clause: str) -> bool:
    return bool(_EXCLUSION.search(clause))


def is_withdrawal(clause: str) -> bool:
    if re.search(r"\b(?:don['’]?t|do not)\s+(?:remove|drop)\b|삭제하지|빼지", clause, re.I):
        return False
    return bool(_WITHDRAW.search(clause))


def requests_citywide_search(query: str) -> bool:
    """Require a geographic instruction before widening an existing local search."""
    for clause in intent_clauses(query):
        if is_inquiry(clause) or re.search(
            r"\b(?:not|don['’]?t|do not|never|without)\b|말고|하지\s*마|않|아니", clause, re.I,
        ):
            continue
        if clause.strip(" .!?;").casefold() in {"seoul", "seoul city", "서울", "서울시", "서울특별시"}:
            return True
        if re.search(r"\bcity[ -]?wide\b|\b(?:anywhere in|across|all|entire)\s+seoul\b|서울\s*(?:전역|전체)|서울시\s*전체", clause, re.I):
            return True
    return False


def english_consultation_intent(query: str) -> str | None:
    """Separate an operational question from a consultation-language constraint."""
    intents = []
    for clause in intent_clauses(query):
        clinical = _RESPONSE_DIRECTIVE.sub("", clause)
        clinical = re.sub(
            r"\bEnglish\s+translations?\s+of\s+(?:the\s+)?(?:patient\s+)?(?:reviews|comments)\b"
            r"|\btranslate\s+(?:the\s+)?(?:patient\s+)?(?:reviews|comments)\s+(?:into|to|in)\s+English\b"
            r"|(?:후기|리뷰|댓글)(?:를|을)?\s*영어로\s*번역",
            "", clinical, flags=re.I,
        )
        if not re.search(r"\benglish\b|영어", clinical, re.I):
            continue
        if is_withdrawal(clinical):
            intents.append("withdraw")
        elif is_exclusion(clinical):
            intents.append("exclude")
        elif re.search(r"\b(?:must|mandatory|required|essential|non-negotiable)\b|필수|반드시", clinical, re.I):
            intents.append("require")
        elif is_inquiry(clinical):
            intents.append("inquire")
        elif _REQUIRE.search(clinical) or re.search(r"\b(?:find|looking for|with)\b.{0,60}\benglish\b", clinical, re.I):
            intents.append("require")
        else:
            intents.append("prefer")
    # A question does not cancel an explicit requirement stated elsewhere.
    explicit = [intent for intent in intents if intent in {"require", "withdraw", "exclude"}]
    return explicit[-1] if explicit else (intents[-1] if intents else None)


def is_english_consultation_term(term: str) -> bool:
    return bool(re.search(r"\benglish\b|영어", term, re.I))


def term_in_clause(term: str, clause: str) -> bool:
    aliases = next((variants for canonical, variants in COMMENT_ALIASES if canonical == term), (term,))
    reduced = re.sub(r"\s+(?:available|availability|hours)$", "", term, flags=re.I)
    if reduced != term:
        aliases = (*aliases, reduced)
    return any(_contains(clause.casefold(), alias.casefold()) for alias in aliases)


def _inquiry_only_term(query: str, term: str) -> bool:
    clauses = [clause for clause in intent_clauses(query) if term_in_clause(term, clause)]
    return bool(clauses) and all(is_inquiry(clause) and not is_exclusion(clause) for clause in clauses)


def _excluded_only_term(query: str, term: str) -> bool:
    clauses = [clause for clause in intent_clauses(query) if term_in_clause(term, clause)]
    return bool(clauses) and all(is_exclusion(clause) for clause in clauses)


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


def expand_multilingual_retrieval_terms(
    terms: Iterable[str],
    *,
    limit: int = 24,
) -> List[str]:
    """Expand known search concepts without relying on the planner to translate."""
    normalized_groups = [
        {item.casefold() for item in group}
        for group in MULTILINGUAL_RETRIEVAL_GROUPS
    ]
    output: List[str] = []
    for value in terms:
        term = str(value or "").strip()
        if not term:
            continue
        additions: Iterable[str] = (term,)
        normalized = term.casefold()
        for group, normalized_group in zip(
            MULTILINGUAL_RETRIEVAL_GROUPS,
            normalized_groups,
        ):
            if normalized in normalized_group:
                additions = group
                break
        for addition in additions:
            if addition.casefold() not in {
                existing.casefold() for existing in output
            }:
                output.append(addition)
                if len(output) >= limit:
                    return output
    return output


def infer_explicit_travel_label(query: str) -> str | None:
    """Map an explicit bilingual distance expression without inventing one."""
    normalized = query.casefold()
    numeric = DISTANCE_PATTERN.search(normalized)
    if numeric:
        value = float(numeric.group("value").replace(",", "."))
        unit = numeric.group("unit").casefold()
        distance_km = value if unit.startswith("k") or unit == "킬로미터" else value / 1000
        for label, supported_distance in DISTANCE_MAPPING.items():
            if abs(distance_km - float(supported_distance)) <= 0.001:
                return label

    for label, phrases in TRAVEL_PHRASES:
        if any(phrase in normalized for phrase in phrases):
            return label
    return None


def has_explicit_travel_preference(query: str) -> bool:
    return bool(
        DISTANCE_PATTERN.search(query)
        or TRAVEL_INTENT_PATTERN.search(query)
        or any(
            phrase in query.casefold()
            for _, phrases in TRAVEL_PHRASES
            for phrase in phrases
        )
    )


def requests_existing_travel_scope(query: str) -> bool:
    """Recognize a refinement that explicitly keeps the prior radius."""
    return any(pattern.search(query) for pattern in PRESERVE_TRAVEL_SCOPE_PATTERNS)


def may_relax_distance_constraint(travel_confidence: float | None) -> bool:
    """Only an implicit/default radius may expand to fill a result list."""
    return float(travel_confidence or 0.0) < 0.6


def augment_extracted_facets(query: str, payload: Mapping[str, Any] | None) -> Dict[str, Any]:
    """Merge model output with conservative bilingual literal safeguards."""
    result = dict(payload or {})

    specialties = _find_aliases(query, SPECIALTY_ALIASES)
    places = _find_aliases(query, PLACE_ALIASES)
    genders = _find_aliases(query, GENDER_ALIASES)
    diseases = _find_aliases(query, DISEASE_ALIASES)
    comment_terms = _find_aliases(query, COMMENT_ALIASES)
    comment_terms = [term for term in comment_terms if not _inquiry_only_term(query, term)]

    result["place_terms"] = _merge_terms(_clean_list(result.get("place_terms")), places)
    result["gender_terms"] = _merge_terms(_clean_list(result.get("gender_terms")), genders)
    result["disease_terms"] = _merge_terms(_clean_list(result.get("disease_terms")), diseases)
    proposed_comments = [term for term in _clean_list(result.get("comment_terms"))
                         if not _inquiry_only_term(query, term)]
    result["comment_terms"] = _merge_terms(proposed_comments, comment_terms)

    if specialties:
        result["specialty"] = specialties[0]
        result["specialty_confidence"] = 0.95
    elif (parts := re.split(r"[,/|\s]+", str(result.get("specialty") or "").strip().casefold())) and all(
        part in GENERIC_FACILITY_NOUNS for part in parts
    ):
        result["specialty"] = None
    if not result.get("location") and places:
        result["location"] = places[0]
    inferred_travel_label = infer_explicit_travel_label(query)
    if DISTANCE_PATTERN.search(query) and inferred_travel_label:
        result["travel_label"] = inferred_travel_label
    elif requests_existing_travel_scope(query):
        result["travel_label"] = None
    elif inferred_travel_label:
        result["travel_label"] = inferred_travel_label
    else:
        result["travel_label"] = None
    if result.get("travel_label") == "Anywhere in Seoul" and not requests_citywide_search(query):
        result["travel_label"] = None
    result.setdefault("hard_keywords", [])
    result.setdefault("soft_keywords", [])
    result.setdefault("negative_hard_keywords", [])
    result.setdefault("negative_keywords", [])
    clinical_language_intent = english_consultation_intent(query)
    for field in ("hard_keywords", "soft_keywords", "negative_hard_keywords", "negative_keywords", "comment_terms"):
        terms = _clean_list(result.get(field))
        terms = [term for term in terms if not is_english_consultation_term(term)
                 or (clinical_language_intent == "require" and field == "hard_keywords")
                 or (clinical_language_intent == "exclude" and field == "negative_hard_keywords")
                 or (clinical_language_intent == "prefer" and field in {"soft_keywords", "comment_terms"})]
        terms = [term for term in terms if not _inquiry_only_term(query, term)]
        if field in {"soft_keywords", "comment_terms"}:
            terms = [term for term in terms if not _excluded_only_term(query, term)]
        if field in {"soft_keywords", "negative_keywords"}:
            terms = [term for term in terms if term.casefold() != "direct"
                     or re.search(r"\b(?:direct\s+(?:doctor|physician)|(?:doctor|physician)\b[^.!?]{0,24}\bdirect)\b", query, re.I)]
        if field == "negative_keywords":
            terms = [term for term in terms if term not in {canonical for canonical, _ in COMMENT_ALIASES} or not any(
                term_in_clause(term, clause) and not is_inquiry(clause)
                and not is_exclusion(clause) and not is_withdrawal(clause)
                for clause in intent_clauses(query)
            )]
        result[field] = terms
    if clinical_language_intent == "require" and not any(is_english_consultation_term(term) for term in result["hard_keywords"]):
        result["hard_keywords"].append("English-speaking")
    if clinical_language_intent == "exclude" and not any(is_english_consultation_term(term) for term in result["negative_hard_keywords"]):
        result["negative_hard_keywords"].append("English-speaking")
    result["inquiries"] = [clause for clause in intent_clauses(query) if is_inquiry(clause)][:6]
    visit_reason = result.get("visit_reason")
    if isinstance(visit_reason, str) and visit_reason.strip() and visit_reason.casefold() in query.casefold():
        result["visit_reason"] = visit_reason.strip()
    elif vague_foot := re.search(r"\bfoot\s+(?:issue|problem)\b|발\s*문제", query, re.I):
        result["visit_reason"] = vague_foot.group(0)
    elif re.search(r"\broutine\s+check[- ]?up\b|정기\s*검진|일반\s*검진", query, re.I):
        result["visit_reason"] = "routine checkup"
    else:
        result["visit_reason"] = None
    result.setdefault(
        "extraction_source",
        "gpt_oss+deterministic_safeguards" if payload else "deterministic_fallback",
    )
    return result


def retrieval_terms_from_state(state: Any) -> List[str]:
    """Build literal BM25 terms while preserving structured debug facets."""
    terms: List[str] = list(getattr(state, "hard_keywords", []))
    for field in ("gender_terms", "disease_terms", "comment_terms"):
        for term in getattr(state, field, []):
            terms.append(term)
    for hour in getattr(state, "required_hours", []):
        terms.extend(HOUR_RETRIEVAL_TERMS.get(hour, (hour,)))
    return expand_multilingual_retrieval_terms(terms)
