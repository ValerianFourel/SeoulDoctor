"""Count script groups in a SeoulDoc raw-review Parquet snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import duckdb


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from raw_review_store import HANGUL_CHAR_CLASS, LATIN_CHAR_CLASS  # noqa: E402


DEFAULT_REVIEWS_PATH = Path(__file__).resolve().parents[1] / "local_reviews_cache.parquet"
NON_LATIN_LETTER_PATTERN = "\\p{L}"


def profile_review_languages(parquet_path: Path) -> dict[str, Any]:
    """Return mutually exclusive script counts for nonempty review text."""
    resolved_path = parquet_path.expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(resolved_path)

    connection = duckdb.connect(database=":memory:")
    try:
        row = connection.execute(
            """
            WITH source AS (
                SELECT CAST(review_text AS VARCHAR) AS review_text
                FROM read_parquet(?)
            ), nonempty AS (
                SELECT
                    review_text,
                    regexp_matches(review_text, ?) AS has_hangul,
                    regexp_matches(review_text, ?) AS has_latin,
                    regexp_matches(
                        regexp_replace(review_text, ?, '', 'g'),
                        ?
                    ) AS has_non_hangul_non_latin_letter
                FROM source
                WHERE review_text IS NOT NULL
                  AND length(trim(review_text)) > 0
            )
            SELECT
                (SELECT count(*) FROM source) AS total_rows,
                count(*) AS nonempty_rows,
                count(*) FILTER (WHERE has_hangul AND NOT has_latin)
                    AS hangul_without_latin,
                count(*) FILTER (
                    WHERE has_latin
                      AND NOT has_hangul
                      AND NOT has_non_hangul_non_latin_letter
                ) AS latin_only_english_like,
                count(*) FILTER (WHERE has_hangul AND has_latin)
                    AS mixed_hangul_latin,
                count(*) FILTER (
                    WHERE NOT has_hangul
                      AND (
                          NOT has_latin
                          OR has_non_hangul_non_latin_letter
                      )
                ) AS other
            FROM nonempty
            """,
            [
                str(resolved_path),
                HANGUL_CHAR_CLASS,
                LATIN_CHAR_CLASS,
                HANGUL_CHAR_CLASS + "|" + LATIN_CHAR_CLASS,
                NON_LATIN_LETTER_PATTERN,
            ],
        ).fetchone()
    finally:
        connection.close()

    keys = (
        "total_rows",
        "nonempty_rows",
        "hangul_without_latin",
        "latin_only_english_like",
        "mixed_hangul_latin",
        "other",
    )
    counts = {key: int(value) for key, value in zip(keys, row)}
    bucket_total = sum(counts[key] for key in keys[2:])
    if bucket_total != counts["nonempty_rows"]:
        raise RuntimeError(
            f"script buckets total {bucket_total}, expected {counts['nonempty_rows']}"
        )
    return {
        "snapshot": str(resolved_path),
        "method": "Unicode script presence, not full language identification",
        **counts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile Hangul and Latin script use in raw SeoulDoc comments."
    )
    parser.add_argument(
        "parquet",
        nargs="?",
        type=Path,
        default=DEFAULT_REVIEWS_PATH,
        help=f"raw-review Parquet path (default: {DEFAULT_REVIEWS_PATH})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(json.dumps(profile_review_languages(args.parquet), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
