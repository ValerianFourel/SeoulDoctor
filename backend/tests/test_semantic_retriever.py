from __future__ import annotations

import unittest

from search.semantic_retriever import (
    RemoteBgeM3ReviewRetriever,
    SemanticCellQuery,
)


class FakeResponse:
    def __init__(self, payload: object) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        return self._payload


class FakeSession:
    def __init__(self, payload: object) -> None:
        self.payload = payload
        self.calls: list[dict[str, object]] = []

    def post(self, url: str, **kwargs: object) -> FakeResponse:
        self.calls.append({"url": url, **kwargs})
        return FakeResponse(self.payload)


def query() -> SemanticCellQuery:
    return SemanticCellQuery(
        query_id="kind:ko",
        constraint_id="kind",
        role="support",
        language="ko",
        text="아이에게 친절",
    )


class SemanticRetrieverTests(unittest.TestCase):
    def test_accepts_sparse_and_dense_references_from_matching_release(self) -> None:
        digest = "a" * 64
        session = FakeSession({
            "schema_version": "seouldoc.semantic-results/v1",
            "release": {
                "release_id": "reviews-v1",
                "review_source_sha256": digest,
                "model_id": "BAAI/bge-m3",
            },
            "execution": {"gpu_execution_verified": True, "device": "cuda:0"},
            "results": [
                {
                    "query_id": "kind:ko",
                    "evidence_id": "review:one",
                    "facility_id": "clinic-a",
                    "channel": "bge_m3_sparse",
                    "rank": 1,
                    "score": 0.7,
                },
                {
                    "query_id": "kind:ko",
                    "evidence_id": "review:two",
                    "facility_id": "clinic-a",
                    "channel": "bge_m3_dense",
                    "rank": 1,
                    "score": 0.8,
                },
            ],
        })
        retriever = RemoteBgeM3ReviewRetriever(
            base_url="https://private-retriever.hf.space",
            token="secret",
            release_id="reviews-v1",
            session=session,
        )

        outcome = retriever.retrieve(
            review_source_sha256=digest,
            facility_ids=("clinic-a",),
            queries=(query(),),
            limit_per_facility=5,
        )

        self.assertTrue(outcome.used)
        self.assertTrue(outcome.gpu_execution_verified)
        self.assertEqual(
            {item.channel for item in outcome.references},
            {"bge_m3_sparse", "bge_m3_dense"},
        )
        call = session.calls[0]
        self.assertEqual(
            call["url"],
            "https://private-retriever.hf.space/v1/retrieve",
        )
        self.assertEqual(call["headers"], {"Authorization": "Bearer secret"})
        self.assertNotIn("review_text", str(call["json"]))

    def test_release_digest_mismatch_rejects_all_remote_results(self) -> None:
        session = FakeSession({
            "schema_version": "seouldoc.semantic-results/v1",
            "release": {
                "release_id": "reviews-v1",
                "review_source_sha256": "b" * 64,
                "model_id": "BAAI/bge-m3",
            },
            "results": [],
        })
        retriever = RemoteBgeM3ReviewRetriever(
            base_url="https://private-retriever.hf.space",
            release_id="reviews-v1",
            session=session,
        )

        outcome = retriever.retrieve(
            review_source_sha256="a" * 64,
            facility_ids=("clinic-a",),
            queries=(query(),),
            limit_per_facility=5,
        )

        self.assertEqual(outcome.status, "release_mismatch")
        self.assertEqual(outcome.references, ())

    def test_wrong_facility_reference_fails_open_without_partial_results(self) -> None:
        digest = "a" * 64
        session = FakeSession({
            "schema_version": "seouldoc.semantic-results/v1",
            "release": {
                "release_id": "reviews-v1",
                "review_source_sha256": digest,
                "model_id": "BAAI/bge-m3",
            },
            "results": [{
                "query_id": "kind:ko",
                "evidence_id": "review:one",
                "facility_id": "outside",
                "channel": "bge_m3_dense",
                "rank": 1,
                "score": 0.9,
            }],
        })
        retriever = RemoteBgeM3ReviewRetriever(
            base_url="https://private-retriever.hf.space",
            release_id="reviews-v1",
            session=session,
        )

        outcome = retriever.retrieve(
            review_source_sha256=digest,
            facility_ids=("clinic-a",),
            queries=(query(),),
            limit_per_facility=5,
        )

        self.assertEqual(outcome.status, "invalid_response")
        self.assertEqual(outcome.references, ())


if __name__ == "__main__":
    unittest.main()
