"""Tests for the sealed reverse-target review probe."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import duckdb
import pandas as pd


TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from probe_reverse_target import run_probe  # noqa: E402


class ReverseTargetProbeTests(unittest.TestCase):
    def test_target_and_decisive_evidence_must_reach_the_stack(self) -> None:
        target_text = "구순염 진료를 받고 자세한 설명과 빠른 진료가 좋았어요"
        identity = f"target|1|{target_text}"
        evidence_id = f"review:{sha256(identity.encode('utf-8')).hexdigest()[:20]}"
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            facilities_path = root / "facilities.parquet"
            reviews_path = root / "reviews.parquet"
            casebook_path = root / "casebook.json"
            pd.DataFrame([
                {
                    "place_id": "other",
                    "category": "피부과",
                    "lat": 37.0005,
                    "lon": 127.0,
                },
                {
                    "place_id": "target",
                    "category": "피부과",
                    "lat": 37.001,
                    "lon": 127.0,
                },
            ]).to_parquet(facilities_path)
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
                    )
                    """
                )
                connection.execute(
                    """
                    INSERT INTO reviews VALUES
                        ('other', 'Other', 1, '설명이 좋아요', NULL, NULL),
                        ('target', 'Target', 1, ?, NULL, NULL)
                    """,
                    [target_text],
                )
                connection.execute(
                    "COPY reviews TO ? (FORMAT PARQUET)",
                    [str(reviews_path)],
                )
            finally:
                connection.close()
            casebook_path.write_text(json.dumps({
                "scenarios": [{
                    "id": "sealed-en",
                    "oracle": {
                        "origin": {"latitude": 37.0, "longitude": 127.0},
                        "maximum_distance_km": 2.0,
                        "expected_specialty": "피부과",
                        "positive_terms": [
                            "cheilitis",
                            "detailed explanation",
                            "fast treatment",
                        ],
                        "negative_terms": [],
                        "reverse_target": {
                            "seed": "fixed-seed",
                            "place_id": "target",
                            "name": "Target",
                            "retrieval_rank_lte": 1,
                            "evidence_requirements": {
                                "decisive": [{"evidence_id": evidence_id}]
                            },
                        },
                    },
                }],
            }), encoding="utf-8")

            report = run_probe(
                casebook_path,
                "sealed-en",
                facilities_path,
                reviews_path,
            )

        self.assertTrue(report["passed"])
        self.assertEqual(report["scope"]["facility_count"], 2)
        self.assertEqual(report["target"]["retrieval_rank_1based"], 1)
        self.assertEqual(
            report["target"]["found_evidence"][0]["evidence_id"],
            evidence_id,
        )


if __name__ == "__main__":
    unittest.main()
