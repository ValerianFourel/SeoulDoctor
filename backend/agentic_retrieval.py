"""Pure helpers for agentic, multi-level retrieval.

The dense index operates on facility-level documents. This module prepares the
finer evidence chunks used by exact-token BM25 retrieval and validates the
bounded actions produced by the retrieval planner.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


TOKEN_PATTERN = re.compile(r"[0-9A-Za-z가-힣]+(?:['’][0-9A-Za-z가-힣]+)?")
ALLOWED_RETRIEVAL_ACTIONS = {"dense_general", "bm25_specific", "hybrid", "finish"}

STOPWORDS = {
    "a", "about", "an", "and", "at", "clinic", "clinics", "doctor", "doctors",
    "facility", "find", "for", "hospital", "hospitals", "in", "me", "medical",
    "near", "of", "or", "please", "seoul", "show", "the", "to", "with",
}


def tokenize_exact(text: Any) -> List[str]:
    """Tokenize without stemming so BM25 retains literal word semantics."""
    return TOKEN_PATTERN.findall(str(text or "").casefold())


def contains_exact_phrase(document_tokens: Sequence[str], phrase: str) -> bool:
    """Return whether every phrase token occurs contiguously in a document."""
    phrase_tokens = tokenize_exact(phrase)
    if not phrase_tokens or len(phrase_tokens) > len(document_tokens):
        return False

    width = len(phrase_tokens)
    return any(
        list(document_tokens[start:start + width]) == phrase_tokens
        for start in range(len(document_tokens) - width + 1)
    )


def extract_literal_terms(query: str, maximum: int = 8) -> List[str]:
    """Extract conservative fallback terms when the planner omits them."""
    terms: List[str] = []
    for token in tokenize_exact(query):
        if token in STOPWORDS or len(token) < 2 or token in terms:
            continue
        terms.append(token)
        if len(terms) >= maximum:
            break
    return terms


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    return isinstance(value, float) and math.isnan(value)


def _as_items(value: Any) -> List[Any]:
    if _is_missing(value):
        return []
    if isinstance(value, (str, bytes, Mapping)):
        return [value]
    if isinstance(value, Iterable):
        return list(value)
    return [value]


def _row_value(row: Any, field: str, default: Any = None) -> Any:
    if hasattr(row, "get"):
        return row.get(field, default)
    try:
        return row[field]
    except (KeyError, TypeError):
        return default


def _flatten_fact(prefix: str, value: Any) -> List[str]:
    """Flatten positive structured facts into human-readable evidence chunks."""
    if _is_missing(value):
        return []
    if isinstance(value, bool):
        return [prefix.replace("_", " ")] if value else []
    if isinstance(value, Mapping):
        facts: List[str] = []
        for key, nested_value in value.items():
            nested_prefix = f"{prefix} {str(key).replace('_', ' ')}".strip()
            facts.extend(_flatten_fact(nested_prefix, nested_value))
        return facts
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        facts = []
        for item in value:
            facts.extend(_flatten_fact(prefix, item))
        return facts

    text = str(value).strip()
    if not text:
        return []
    return [f"{prefix.replace('_', ' ')}: {text}".strip(": ")]


RAW_COMMENT_FIELDS = {
    "raw_reviews", "raw_comments", "review_comments", "review_texts",
    "patient_comments", "comments", "visitor_reviews", "google_reviews",
    "naver_reviews", "kakao_reviews", "reviews_raw", "reviews_data",
}


def _row_fields(row: Any) -> List[str]:
    if hasattr(row, "index"):
        return [str(field) for field in row.index]
    if isinstance(row, Mapping):
        return [str(field) for field in row.keys()]
    return []


def _raw_comment_fields(row: Any) -> List[str]:
    """Discover likely raw-comment fields without treating aggregate counts as text."""
    fields: List[str] = []
    for field in _row_fields(row):
        normalized = field.casefold()
        is_explicit = normalized in RAW_COMMENT_FIELDS
        is_comment = "comment" in normalized
        is_review_payload = "review" in normalized and any(
            marker in normalized
            for marker in ("raw", "text", "content", "data", "item", "entry", "list")
        )
        is_generated = any(marker in normalized for marker in ("summary", "highlight", "count", "score"))
        if (is_explicit or is_comment or is_review_payload) and not is_generated:
            fields.append(field)
    return fields


def _review_text(review: Any) -> tuple[str, str]:
    if isinstance(review, Mapping):
        text = next(
            (
                review.get(key)
                for key in ("text", "comment", "content", "review", "body", "message")
                if review.get(key)
            ),
            "",
        )
        return str(text or "").strip(), str(review.get("language", "Unknown"))
    return str(review or "").strip(), "Unknown"


def build_specific_evidence_records(df: Any) -> List[Dict[str, Any]]:
    """Create comment/fact-level records for the specific BM25 index."""
    records: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        place_id = str(_row_value(row, "place_id", "")).strip()
        if not place_id:
            continue

        seen_text = set()

        def add_record(
            text: Any,
            source_type: str,
            source_index: int,
            language: str,
            is_verbatim: bool = False,
            source_field: str = "",
        ) -> None:
            normalized_text = str(text or "").strip()
            dedupe_key = normalized_text.casefold()
            if not normalized_text or dedupe_key in seen_text:
                return
            seen_text.add(dedupe_key)
            records.append({
                "evidence_id": f"{place_id}:{source_type}:{source_field}:{source_index}",
                "place_id": place_id,
                "text": normalized_text,
                "source_type": source_type,
                "source_index": source_index,
                "language": language,
                "level": "specific",
                "is_verbatim": is_verbatim,
                "source_field": source_field,
            })

        # Raw comment payloads remain one BM25 document per actual comment.
        # Field discovery is intentionally schema-tolerant for remote datasets.
        for field in _raw_comment_fields(row):
            for index, review in enumerate(_as_items(_row_value(row, field))):
                review_text, language = _review_text(review)
                add_record(
                    review_text,
                    "verbatim_review",
                    index,
                    language,
                    is_verbatim=True,
                    source_field=field,
                )

        for field, language in (("Summaries", "English"), ("Summaries_Korean", "Korean")):
            for index, summary in enumerate(_as_items(_row_value(row, field))):
                add_record(summary, "review_summary", index, language, source_field=field)

        for index, highlight in enumerate(_as_items(_row_value(row, "Key_Highlights"))):
            if isinstance(highlight, Mapping):
                topic = str(highlight.get("topic", "")).strip()
                percentage = highlight.get("percentage")
                text = topic
                if topic and percentage is not None:
                    text = f"{topic} ({percentage}% of review highlights)"
            else:
                text = str(highlight).strip()
            add_record(text, "review_highlight", index, "Mixed")

        for source_type, field in (("amenity", "amenities"), ("medical_info", "medical_info_parsed")):
            facts = _flatten_fact("", _row_value(row, field))
            for index, fact in enumerate(facts):
                add_record(fact, source_type, index, "Mixed")

        for index, field in enumerate(("reviews", "business_hours", "phone", "address")):
            value = _row_value(row, field)
            if not _is_missing(value):
                add_record(f"{field.replace('_', ' ')}: {value}", "facility_fact", index, "Mixed")

    return records


@dataclass(frozen=True)
class RetrievalPlan:
    action: str
    query: str
    exact_terms: List[str]
    quote_evidence: bool
    reasoning: str


def validate_retrieval_plan(
    payload: Optional[Mapping[str, Any]],
    fallback_query: str,
    required_exact_terms: Sequence[str] = (),
    has_observations: bool = False,
) -> RetrievalPlan:
    """Validate an LLM tool decision and provide a deterministic fallback."""
    payload = payload or {}
    action = str(payload.get("action", "")).strip()
    if action not in ALLOWED_RETRIEVAL_ACTIONS:
        action = "hybrid" if required_exact_terms else "dense_general"
    if action == "finish" and not has_observations:
        action = "hybrid" if required_exact_terms else "dense_general"

    query = str(payload.get("query") or fallback_query).strip() or fallback_query
    raw_terms = payload.get("exact_terms")
    raw_terms = raw_terms if isinstance(raw_terms, list) else []

    exact_terms: List[str] = []
    for term in [*required_exact_terms, *raw_terms]:
        normalized = str(term).strip()
        if normalized and normalized.casefold() not in {item.casefold() for item in exact_terms}:
            exact_terms.append(normalized)
        if len(exact_terms) >= 8:
            break

    if action in {"bm25_specific", "hybrid"} and not exact_terms:
        exact_terms = extract_literal_terms(query)

    return RetrievalPlan(
        action=action,
        query=query,
        exact_terms=exact_terms,
        quote_evidence=bool(payload.get("quote_evidence", False)),
        reasoning=str(payload.get("reasoning", "Deterministic retrieval fallback.")),
    )
