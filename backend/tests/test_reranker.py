from __future__ import annotations

import unittest

from search.indexes.repository import EvidenceHit
from search.reranker import RemoteEvidenceReranker


def evidence_hit(evidence_id: str, facility_id: str, text: str) -> EvidenceHit:
    return EvidenceHit(
        evidence_id=evidence_id,
        facility_id=facility_id,
        ordinal=1,
        score=1.0,
        channel="evidence",
        source_type="verbatim_review",
        source_field="review_text",
        source_index=1,
        source_locator="reviews.parquet#1",
        original_text=text,
        language_hint="ko",
        visit_date=None,
        scraped_at=None,
        is_verbatim=True,
    )


class FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return self._payload


class FakeSession:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.calls: list[dict[str, object]] = []

    def post(self, url: str, **kwargs: object) -> FakeResponse:
        self.calls.append({"url": url, **kwargs})
        return FakeResponse(self.payload)


class RemoteEvidenceRerankerTests(unittest.TestCase):
    def test_remote_scores_reorder_hits_without_trusting_remote_metadata(self) -> None:
        first = evidence_hit("e-1", "alpha", "보통이었어요")
        decisive = evidence_hit("e-2", "bravo", "과잉진료 없이 친절했어요")
        session = FakeSession({
            "model": "BAAI/bge-reranker-v2-m3",
            "results": [
                {"evidence_id": "e-2", "place_id": "wrong", "score": 0.95},
                {"evidence_id": "e-1", "place_id": "wrong", "score": 0.10},
            ],
        })
        reranker = RemoteEvidenceReranker(
            base_url="https://private-space.hf.space",
            token="test-token",
            timeout_seconds=3.0,
            max_candidates=32,
            session=session,
        )

        outcome = reranker.rerank(
            "Direct review evidence for 친절 and no overprescribing",
            (first, decisive),
        )

        self.assertTrue(outcome.used)
        self.assertEqual(outcome.hits, (decisive, first))
        self.assertEqual(outcome.hits[0].facility_id, "bravo")
        self.assertEqual(
            dict(outcome.scores),
            {"e-1": 0.10, "e-2": 0.95},
        )
        call = session.calls[0]
        self.assertEqual(call["url"], "https://private-space.hf.space/rerank")
        self.assertEqual(
            call["headers"], {"Authorization": "Bearer test-token"}
        )
        self.assertEqual(call["timeout"], 3.0)

    def test_tei_index_response_maps_back_to_local_evidence(self) -> None:
        first = evidence_hit("e-1", "alpha", "보통")
        decisive = evidence_hit("e-2", "bravo", "친절하고 과잉진료가 없어요")
        session = FakeSession([
            {"index": 1, "score": 4.2},
            {"index": 0, "score": -1.3},
        ])
        reranker = RemoteEvidenceReranker(
            base_url="https://private-space.hf.space",
            token="test-token",
            session=session,
        )

        outcome = reranker.rerank("친절, no overprescribing", (first, decisive))

        self.assertTrue(outcome.used)
        self.assertEqual(outcome.hits, (decisive, first))
        self.assertEqual(
            session.calls[0]["json"],
            {
                "query": "친절, no overprescribing",
                "texts": ["보통", "친절하고 과잉진료가 없어요"],
            },
        )

    def test_duplicate_evidence_ids_are_sent_once_and_removed_from_output(self) -> None:
        first = evidence_hit("e-1", "alpha", "보통")
        duplicate = evidence_hit("e-1", "alpha", "보통")
        decisive = evidence_hit("e-2", "bravo", "친절하고 과잉진료가 없어요")
        session = FakeSession([
            {"index": 1, "score": 4.2},
            {"index": 0, "score": -1.3},
        ])
        reranker = RemoteEvidenceReranker(
            base_url="https://private-space.hf.space",
            token="test-token",
            session=session,
        )

        outcome = reranker.rerank(
            "친절, no overprescribing",
            (first, duplicate, decisive),
        )

        self.assertTrue(outcome.used)
        self.assertEqual(outcome.hits, (decisive, first))
        self.assertEqual(
            session.calls[0]["json"],
            {
                "query": "친절, no overprescribing",
                "texts": ["보통", "친절하고 과잉진료가 없어요"],
            },
        )

    def test_invalid_remote_response_fails_open_in_original_order(self) -> None:
        hits = (evidence_hit("e-1", "alpha", "friendly"),)
        session = FakeSession({"model": "bad", "results": []})
        reranker = RemoteEvidenceReranker(
            base_url="https://private-space.hf.space",
            token="test-token",
            session=session,
        )

        outcome = reranker.rerank("friendly", hits)

        self.assertFalse(outcome.used)
        self.assertEqual(outcome.hits, hits)
        self.assertEqual(outcome.reason, "invalid_response")


if __name__ == "__main__":
    unittest.main()
