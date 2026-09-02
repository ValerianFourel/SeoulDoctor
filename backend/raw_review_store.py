"""Bounded access to multilingual verbatim patient reviews stored in Parquet.

The retrieval agent never receives reviewer names or unrestricted SQL access.
It can only request comments for facility IDs already found by facility search.
"""

from __future__ import annotations

from hashlib import sha256
import logging
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence

import duckdb
import requests


logger = logging.getLogger(__name__)
HANGUL_PATTERN = re.compile(r"[가-힣]")
LATIN_PATTERN = re.compile(r"[A-Za-z]")
JAPANESE_PATTERN = re.compile(r"[ぁ-ゟ゠-ヿ]")
HAN_PATTERN = re.compile(r"[一-龯]")
CYRILLIC_PATTERN = re.compile(r"[А-Яа-яЁё]")
ARABIC_PATTERN = re.compile(r"[ء-ي]")
RAW_REVIEWS_URL = (
    "https://huggingface.co/datasets/ValerianFourel/"
    "seoul-medical-facilities/resolve/main/"
    "seoul_medical_reviews_merged.parquet"
)
DEFAULT_RAW_REVIEWS_PATH = str(
    Path(__file__).resolve().parent / "local_reviews_cache.parquet"
)


def ensure_raw_review_parquet() -> Optional[str]:
    """Return a local raw-review snapshot, downloading it once when enabled."""
    enabled = os.getenv("ENABLE_RAW_REVIEWS", "true").strip().casefold()
    if enabled not in {"1", "true", "yes", "on"}:
        logger.warning("Raw review retrieval disabled by ENABLE_RAW_REVIEWS")
        return None

    target = Path(
        os.getenv("RAW_REVIEWS_PATH", DEFAULT_RAW_REVIEWS_PATH)
    ).expanduser().resolve()
    if target.is_file() and target.stat().st_size > 0:
        return str(target)

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    hf_token = os.getenv("HF_TOKEN", "").strip()
    request_options: Dict[str, Any] = {
        "stream": True,
        "timeout": (15, 180),
    }
    if hf_token:
        request_options["headers"] = {"Authorization": f"Bearer {hf_token}"}

    logger.info("Downloading raw review snapshot to %s", target)
    try:
        with requests.get(RAW_REVIEWS_URL, **request_options) as response:
            response.raise_for_status()
            with temporary.open("wb") as output:
                for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                    if chunk:
                        output.write(chunk)
        temporary.replace(target)
        return str(target)
    except Exception:
        logger.exception("Raw review snapshot download failed")
        if temporary.exists():
            temporary.unlink()
        return None


def _bounded_strings(values: Optional[Sequence[Any]], maximum: int) -> List[str]:
    result: List[str] = []
    for value in values or []:
        text = str(value or "").strip()
        if not text or text in result:
            continue
        result.append(text)
        if len(result) >= maximum:
            break
    return result


def detect_language_hint(text: Any) -> str:
    """Return an honest script-based language hint without claiming full detection."""
    value = str(text or "")
    has_hangul = bool(HANGUL_PATTERN.search(value))
    has_latin = bool(LATIN_PATTERN.search(value))
    if has_hangul and has_latin:
        return "Korean + Latin (mixed)"
    if has_hangul:
        return "Korean"
    if JAPANESE_PATTERN.search(value):
        return "Japanese-script"
    if has_latin:
        return "Latin-script"
    if HAN_PATTERN.search(value):
        return "CJK (non-Hangul)"
    if CYRILLIC_PATTERN.search(value):
        return "Cyrillic-script"
    if ARABIC_PATTERN.search(value):
        return "Arabic-script"
    return "Unknown/other"


class RawReviewStore:
    """Read-only, parameterized search over the raw-review Parquet snapshot."""

    def __init__(self, parquet_path: str):
        self.parquet_path = str(Path(parquet_path).resolve())
        if not Path(self.parquet_path).is_file():
            raise FileNotFoundError(self.parquet_path)

        escaped_path = self.parquet_path.replace("'", "''")
        self.connection = duckdb.connect(database=":memory:", read_only=False)
        self.connection.execute(
            f"""
            CREATE VIEW raw_reviews AS
            SELECT
                CAST(place_id AS VARCHAR) AS place_id,
                CAST(facility_name AS VARCHAR) AS facility_name,
                CAST(review_index AS BIGINT) AS review_index,
                CAST(review_text AS VARCHAR) AS review_text,
                CAST(visit_date AS VARCHAR) AS visit_date,
                CAST(scraped_at AS VARCHAR) AS scraped_at
            FROM read_parquet('{escaped_path}')
            WHERE review_text IS NOT NULL AND length(trim(review_text)) > 0
            """
        )
        row_count = self.connection.execute(
            "SELECT count(*) FROM raw_reviews"
        ).fetchone()[0]
        self.row_count = int(row_count)
        logger.info("Raw review store ready: %s verbatim comments", row_count)

    def close(self) -> None:
        self.connection.close()

    def search_comments(
        self,
        facility_ids: Sequence[Any],
        query_terms: Optional[Sequence[Any]] = None,
        limit: int = 30,
        per_facility_limit: int = 3,
    ) -> List[Dict[str, Any]]:
        """Return multilingual comments from an explicit facility scope.

        The retrieval agent supplies original-language, Korean, and English
        concept variants. Terms and facility IDs are always bound parameters;
        review text is never interpreted as SQL or model instructions.
        """
        ids = _bounded_strings(facility_ids, maximum=10000)
        terms = _bounded_strings(query_terms, maximum=24)
        if not ids:
            return []

        limit = max(1, min(int(limit or 30), 60))
        per_facility_limit = max(1, min(int(per_facility_limit or 3), 5))
        id_placeholders = ", ".join("?" for _ in ids)

        if terms:
            score_parts = [
                "CASE WHEN review_text ILIKE ? THEN 1 ELSE 0 END"
                for _ in terms
            ]
            score_expression = " + ".join(score_parts)
            sql = f"""
                WITH candidates AS (
                    SELECT
                        place_id,
                        facility_name,
                        review_index,
                        review_text,
                        visit_date,
                        scraped_at,
                        {score_expression} AS relevance_hits
                    FROM raw_reviews
                    WHERE place_id IN ({id_placeholders})
                ), diversified AS (
                    SELECT *, row_number() OVER (
                        PARTITION BY place_id
                        ORDER BY relevance_hits DESC, review_index ASC
                    ) AS facility_row
                    FROM candidates
                    WHERE relevance_hits > 0
                )
                SELECT * EXCLUDE (facility_row)
                FROM diversified
                WHERE facility_row <= ?
                ORDER BY relevance_hits DESC, place_id, review_index ASC
                LIMIT ?
            """
            parameters = (
                [f"%{term}%" for term in terms]
                + ids
                + [per_facility_limit, limit]
            )
        else:
            sql = f"""
                SELECT
                    place_id,
                    facility_name,
                    review_index,
                    review_text,
                    visit_date,
                    scraped_at,
                    0 AS relevance_hits
                FROM raw_reviews
                WHERE place_id IN ({id_placeholders})
                ORDER BY review_index ASC
                LIMIT ?
            """
            parameters = ids + [limit]

        rows = self.connection.execute(sql, parameters).fetchall()
        columns = [item[0] for item in self.connection.description]
        records = []
        for row in rows:
            record = dict(zip(columns, row))
            text = str(record.get("review_text") or "").strip()
            identity = (
                f"{record.get('place_id')}|{record.get('review_index')}|{text}"
            )
            records.append({
                "evidence_id": f"review:{sha256(identity.encode('utf-8')).hexdigest()[:20]}",
                "place_id": str(record.get("place_id") or ""),
                "facility_name": str(record.get("facility_name") or ""),
                "text": text,
                "language": detect_language_hint(text),
                "source_type": "verbatim_review",
                "source_field": "review_text",
                "source_index": int(record.get("review_index") or 0),
                "visit_date": record.get("visit_date"),
                "scraped_at": record.get("scraped_at"),
                "is_verbatim": True,
                "matched_terms": [
                    term for term in terms if term.casefold() in text.casefold()
                ],
                "score": float(record.get("relevance_hits") or 0),
            })
        return records

    def search_korean_comments(
        self,
        facility_ids: Sequence[Any],
        korean_query_terms: Optional[Sequence[Any]] = None,
        limit: int = 30,
    ) -> List[Dict[str, Any]]:
        """Backward-compatible alias for older callers and deployments."""
        return self.search_comments(
            facility_ids,
            query_terms=korean_query_terms,
            limit=limit,
        )

    def coverage_statistics(self) -> Dict[str, int]:
        """Return reproducible script coverage counts for the local snapshot."""
        row = self.connection.execute(
            """
            SELECT
                count(*) AS total,
                count(*) FILTER (
                    WHERE regexp_matches(review_text, '[가-힣]')
                      AND NOT regexp_matches(review_text, '[A-Za-z]')
                ) AS hangul_only,
                count(*) FILTER (
                    WHERE regexp_matches(review_text, '[가-힣]')
                      AND regexp_matches(review_text, '[A-Za-z]')
                ) AS hangul_latin_mixed,
                count(*) FILTER (
                    WHERE NOT regexp_matches(review_text, '[가-힣]')
                      AND regexp_matches(review_text, '[A-Za-z]')
                ) AS latin_no_hangul,
                count(*) FILTER (
                    WHERE NOT regexp_matches(review_text, '[가-힣]')
                      AND NOT regexp_matches(review_text, '[A-Za-z]')
                ) AS other
            FROM raw_reviews
            """
        ).fetchone()
        keys = (
            "total", "hangul_only", "hangul_latin_mixed",
            "latin_no_hangul", "other",
        )
        return {key: int(value) for key, value in zip(keys, row)}
