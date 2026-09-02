"""Tests for the raw-review script profile command."""

from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import duckdb


TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from profile_review_languages import profile_review_languages  # noqa: E402
from raw_review_store import RawReviewStore  # noqa: E402


class ReviewLanguageProfileTests(unittest.TestCase):
    def test_counts_mutually_exclusive_script_groups(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            parquet_path = Path(temporary_directory) / "reviews.parquet"
            connection = duckdb.connect(database=":memory:")
            try:
                connection.execute(
                    """
                    CREATE TABLE reviews(
                        place_id VARCHAR,
                        facility_name VARCHAR,
                        review_index BIGINT,
                        review_text VARCHAR,
                        visit_date VARCHAR,
                        scraped_at VARCHAR
                    );
                    INSERT INTO reviews VALUES
                        ('1', 'Clinic', 1, '설명이 자세해요', NULL, NULL),
                        ('1', 'Clinic', 2, 'Great doctor', NULL, NULL),
                        ('1', 'Clinic', 3, '친절한 staff', NULL, NULL),
                        ('1', 'Clinic', 4, 'good 医院', NULL, NULL),
                        ('1', 'Clinic', 5, '12345', NULL, NULL),
                        ('1', 'Clinic', 6, '', NULL, NULL),
                        ('1', 'Clinic', 7, NULL, NULL, NULL);
                    COPY reviews TO ? (FORMAT PARQUET);
                    """,
                    [str(parquet_path)],
                )
            finally:
                connection.close()

            profile = profile_review_languages(parquet_path)
            store = RawReviewStore(str(parquet_path))
            try:
                store_profile = store.coverage_statistics()
            finally:
                store.close()

        self.assertEqual(profile["total_rows"], 7)
        self.assertEqual(profile["nonempty_rows"], 5)
        self.assertEqual(profile["hangul_without_latin"], 1)
        self.assertEqual(profile["latin_only_english_like"], 1)
        self.assertEqual(profile["mixed_hangul_latin"], 1)
        self.assertEqual(profile["other"], 2)
        self.assertEqual(store_profile["total"], profile["nonempty_rows"])
        self.assertEqual(
            store_profile["hangul_without_latin"],
            profile["hangul_without_latin"],
        )
        self.assertEqual(
            store_profile["latin_only_english_like"],
            profile["latin_only_english_like"],
        )
        self.assertEqual(
            store_profile["hangul_latin_mixed"],
            profile["mixed_hangul_latin"],
        )
        self.assertEqual(store_profile["other"], profile["other"])


if __name__ == "__main__":
    unittest.main()
