from __future__ import annotations

import tempfile
import unittest
from hashlib import sha256
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import offline_builder
import production_core

MODEL_REVISION = "b" * 40


class ContractEncoder:
    revision = MODEL_REVISION

    def encode(self, texts):
        dense = np.zeros(
            (len(texts), offline_builder.MODEL_DIMENSION),
            dtype=np.float32,
        )
        sparse = []
        for row, text in enumerate(texts):
            seed = int(sha256(text.encode("utf-8")).hexdigest()[:8], 16)
            dense[row, seed % offline_builder.MODEL_DIMENSION] = 1.0
            sparse.append({seed % 997: 1.0})
        return offline_builder.EncodedBatch(dense, tuple(sparse))


class OfflineRuntimeContractTests(unittest.TestCase):
    def test_production_runtime_loads_and_searches_builder_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reviews = root / "reviews.parquet"
            pq.write_table(
                pa.table(
                    {
                        "place_id": ["clinic-a", "clinic-a", "clinic-b"],
                        "review_index": [0, 1, 0],
                        "review_text": [
                            "아이에게 친절해요",
                            "간호사가 불친절해요",
                            "clear explanation",
                        ],
                    }
                ),
                reviews,
            )
            encoder = ContractEncoder()
            release_path = offline_builder.build_release(
                reviews_path=reviews,
                work_dir=root / "work",
                encoder=encoder,
                release_id="runtime-contract-v1",
                model_revision=MODEL_REVISION,
                shard_size=2,
            )
            release = production_core.SemanticRelease(release_path, encoder)
            results = release.retrieve(
                ["clinic-a"],
                [
                    production_core.QueryCell(
                        query_id="support-ko",
                        constraint_id="kindness",
                        role="support",
                        language="ko",
                        text="아이에게 친절해요",
                    )
                ],
                1,
            )
            self.assertEqual(
                {item["channel"] for item in results},
                {"bge_m3_dense", "bge_m3_sparse"},
            )
            self.assertTrue(all(item["facility_id"] == "clinic-a" for item in results))
            mixed_results = release.retrieve(
                ["clinic-without-reviews", "clinic-a"],
                [
                    production_core.QueryCell(
                        query_id="support-ko",
                        constraint_id="kindness",
                        role="support",
                        language="ko",
                        text="아이에게 친절해요",
                    )
                ],
                1,
            )
            self.assertEqual(mixed_results, results)


if __name__ == "__main__":
    unittest.main()
