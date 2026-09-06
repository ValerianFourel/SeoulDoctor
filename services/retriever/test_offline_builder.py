from __future__ import annotations

import tempfile
import unittest
from hashlib import sha256
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import offline_builder

MODEL_REVISION = "a" * 40


class FakeEncoder:
    def __init__(self) -> None:
        self.calls = 0

    def encode(self, texts):
        self.calls += 1
        dense = np.zeros(
            (len(texts), offline_builder.MODEL_DIMENSION),
            dtype=np.float32,
        )
        sparse = []
        for row, text in enumerate(texts):
            seed = int(sha256(text.encode("utf-8")).hexdigest()[:8], 16)
            dense[row, seed % offline_builder.MODEL_DIMENSION] = 2.0
            sparse.append(
                {
                    seed % 1000: 0.8,
                    (seed + 1) % 1000: 0.2,
                    (seed + 2) % 1000: 0.0,
                }
            )
        return offline_builder.EncodedBatch(
            dense=dense,
            sparse=tuple(sparse),
        )


class OfflineBuilderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.reviews = self.root / "reviews.parquet"
        pq.write_table(
            pa.table(
                {
                    "place_id": [
                        "clinic-b",
                        "clinic-a",
                        "clinic-a",
                        "clinic-a",
                    ],
                    "review_index": [0, 1, 0, 2],
                    "review_text": [
                        "친절한 의사",
                        "clear explanation",
                        "간호사가 불친절",
                        "   ",
                    ],
                }
            ),
            self.reviews,
        )

    def tearDown(self):
        self.temporary.cleanup()

    def test_builds_runtime_compatible_release_and_resumes(self):
        encoder = FakeEncoder()
        release = offline_builder.build_release(
            reviews_path=self.reviews,
            work_dir=self.root / "work",
            encoder=encoder,
            release_id="fixture-v1",
            model_revision=MODEL_REVISION,
            shard_size=2,
            sparse_top_k=2,
            max_length=128,
        )
        manifest = offline_builder.validate_release(release)
        self.assertEqual(manifest["review_count"], 3)
        self.assertEqual(manifest["encoding"]["sparse_top_k"], 2)
        self.assertEqual(encoder.calls, 2)

        evidence_ids = np.load(
            release / "evidence_ids.npy",
            allow_pickle=False,
        )
        facility_ids = np.load(
            release / "facility_ids.npy",
            allow_pickle=False,
        )
        self.assertEqual(
            list(facility_ids),
            ["clinic-a", "clinic-a", "clinic-b"],
        )
        self.assertEqual(len(set(evidence_ids)), 3)

        second_release = offline_builder.build_release(
            reviews_path=self.reviews,
            work_dir=self.root / "work",
            encoder=encoder,
            release_id="fixture-v1",
            model_revision=MODEL_REVISION,
            shard_size=2,
            sparse_top_k=2,
            max_length=128,
        )
        self.assertEqual(second_release, release)
        self.assertEqual(encoder.calls, 2)

    def test_rejects_duplicate_evidence_identity(self):
        pq.write_table(
            pa.table(
                {
                    "place_id": ["clinic-a", "clinic-a"],
                    "review_index": [0, 0],
                    "review_text": ["same", "same"],
                }
            ),
            self.reviews,
        )
        with self.assertRaisesRegex(
            offline_builder.BuildError,
            "duplicate evidence",
        ):
            offline_builder.load_reviews(self.reviews)


if __name__ == "__main__":
    unittest.main()
