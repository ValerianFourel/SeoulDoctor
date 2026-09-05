"""Versioned, deterministic documents for immutable search indexes."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import re
from typing import Any, Iterable, Mapping
import unicodedata

import numpy as np


FACILITY_PROFILE_SCHEMA = "seouldoc.facility-profile/v1"
NORMALIZATION_SCHEMA = "seouldoc.nfkc-casefold/v1"
ANALYZER_SCHEMA = "seouldoc.fts5-unicode61-hangul-ngram-trigram/v1"

TOKEN_PATTERN = re.compile(r"[0-9A-Za-z가-힣]+(?:['’][0-9A-Za-z가-힣]+)?")
HANGUL_SEQUENCE_PATTERN = re.compile(r"[가-힣]+")
LATIN_PATTERN = re.compile(r"[A-Za-z]")
HANGUL_PATTERN = re.compile(r"[가-힣]")
FACILITY_LEXICAL_FIELDS = (
    "address",
    "business_hours",
    "phone",
    "Summaries",
    "Summaries_Korean",
    "Key_Highlights",
    "amenities",
    "medical_info_parsed",
)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (float, np.floating)):
        return math.isnan(float(value))
    return False


def _text(value: Any) -> str:
    if _is_missing(value):
        return ""
    rendered = str(value).strip()
    return "" if rendered.casefold() in {"nan", "none"} else rendered


def _array_text(row: Mapping[str, Any], field: str) -> str:
    value = row.get(field)
    if isinstance(value, (list, tuple, np.ndarray)):
        return " ".join(
            rendered
            for item in value
            if (rendered := _text(item))
        )
    return _text(value) if isinstance(value, str) else ""


def render_facility_profile(row: Mapping[str, Any]) -> str:
    """Render the exact facility document used by the existing Chroma index."""
    parts = [
        f"Name: {row.get('name', '')}",
        f"Category: {row.get('category', '')}",
    ]
    for label, field in (
        ("Address", "address"),
        ("District", "file_district"),
        ("Neighborhood", "file_dong"),
    ):
        value = _text(row.get(field))
        if value:
            parts.append(f"{label}: {value}")

    summary_en = _array_text(row, "Summaries")
    summary_ko = _array_text(row, "Summaries_Korean")
    if summary_en:
        parts.append(f"All generated English review summaries: {summary_en}")
    if summary_ko:
        parts.append(f"All generated Korean review summaries: {summary_ko}")

    highlights = row.get("Key_Highlights")
    if isinstance(highlights, (list, np.ndarray)):
        topics = [
            str(item.get("topic", "")).strip()
            for item in highlights
            if isinstance(item, dict) and item.get("topic")
        ]
        if topics:
            parts.append(f"Review highlights: {', '.join(topics)}")

    for label, field in (
        ("Amenities", "amenities"),
        ("Medical information", "medical_info_parsed"),
    ):
        value = row.get(field)
        if isinstance(value, dict) and value:
            parts.append(
                f"{label}: {json.dumps(value, ensure_ascii=False, default=str)}"
            )
    if bool(row.get("has_english", False)):
        parts.append("English speaking support: yes")
    return "\n".join(parts)


def normalize_search_text(value: Any) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(TOKEN_PATTERN.findall(normalized))


def hangul_ngrams(value: Any, sizes: tuple[int, ...] = (2, 3)) -> str:
    """Return whitespace-separated Hangul n-grams for FTS word tokenization."""
    grams: list[str] = []
    for sequence in HANGUL_SEQUENCE_PATTERN.findall(
        unicodedata.normalize("NFKC", str(value or ""))
    ):
        for size in sizes:
            if len(sequence) < size:
                continue
            grams.extend(
                sequence[start:start + size]
                for start in range(len(sequence) - size + 1)
            )
    return " ".join(grams)


def _flatten_search_values(value: Any) -> Iterable[str]:
    if _is_missing(value):
        return
    if isinstance(value, (bool, np.bool_)):
        return
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if isinstance(nested, (bool, np.bool_)):
                if bool(nested):
                    yield _text(key).replace("_", " ")
                continue
            key_text = _text(key).replace("_", " ")
            if key_text:
                yield key_text
            yield from _flatten_search_values(nested)
        return
    if isinstance(value, (list, tuple, set, frozenset, np.ndarray)):
        for item in value:
            yield from _flatten_search_values(item)
        return
    rendered = _text(value)
    if rendered:
        yield rendered


def render_facility_lexical_text(row: Mapping[str, Any]) -> str:
    """Render a reviewer-free multilingual projection for lexical recall."""
    parts = [render_facility_profile(row)]
    for field in FACILITY_LEXICAL_FIELDS:
        values = tuple(_flatten_search_values(row.get(field)))
        if values:
            parts.append(f"{field.replace('_', ' ')}: {' '.join(values)}")
    if bool(row.get("has_english", False)):
        parts.append("English speaking support 영어 진료")
    return "\n".join(part for part in parts if part)


@dataclass(frozen=True)
class FacilityIndexDocument:
    place_id: str
    name: str
    specialty: str
    district: str
    neighborhood: str
    english_terms: str
    korean_terms: str
    korean_ngrams: str
    trusted_facts: str
    exact_phrase: str
    substring_text: str
    profile_text: str
    profile_sha256: str


def facility_index_document(row: Mapping[str, Any]) -> FacilityIndexDocument:
    raw_place_id = row.get("place_id")
    place_id = "" if _is_missing(raw_place_id) else str(raw_place_id).strip()
    if not place_id:
        raise ValueError("facility place_id cannot be empty")
    profile = render_facility_profile(row)
    lexical_text = render_facility_lexical_text(row)
    tokens = TOKEN_PATTERN.findall(
        unicodedata.normalize("NFKC", lexical_text).casefold()
    )
    english_terms = " ".join(token for token in tokens if LATIN_PATTERN.search(token))
    korean_terms = " ".join(token for token in tokens if HANGUL_PATTERN.search(token))
    normalized = normalize_search_text(lexical_text)
    return FacilityIndexDocument(
        place_id=place_id,
        name=_text(row.get("name")),
        specialty=_text(row.get("category")),
        district=_text(row.get("file_district")),
        neighborhood=_text(row.get("file_dong")),
        english_terms=english_terms,
        korean_terms=korean_terms,
        korean_ngrams=hangul_ngrams(profile),
        trusted_facts=lexical_text,
        exact_phrase=normalized,
        substring_text=normalized,
        profile_text=profile,
        profile_sha256=sha256(profile.encode("utf-8")).hexdigest(),
    )


def raw_review_evidence_id(
    place_id: str,
    review_index: int,
    original_text: str,
) -> str:
    """Preserve the evidence identity used by RawReviewStore."""
    identity = f"{place_id}|{review_index}|{original_text}"
    return f"review:{sha256(identity.encode('utf-8')).hexdigest()[:20]}"


def logical_record_digest(parts: Iterable[Any]) -> str:
    """Hash a record with length framing so field boundaries cannot collide."""
    digest = sha256()
    for part in parts:
        encoded = str(part if part is not None else "").encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def fts_query_terms(value: str) -> tuple[str, ...]:
    """Return bounded syntax-free terms for private FTS query rendering."""
    normalized = unicodedata.normalize("NFKC", value).casefold()
    terms = list(dict.fromkeys(TOKEN_PATTERN.findall(normalized)))
    grams = hangul_ngrams(normalized, sizes=(2,)).split()
    return tuple(dict.fromkeys((*terms, *grams)))
