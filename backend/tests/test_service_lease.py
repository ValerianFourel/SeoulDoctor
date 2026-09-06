from datetime import datetime, timedelta, timezone
import unittest
from unittest.mock import patch

from search.service_lease import lease_allows_request, parse_service_expiry
from search.semantic_retriever import RemoteBgeM3ReviewRetriever
from search.reranker import RemoteEvidenceReranker
from backend.tests.test_semantic_retriever import FakeSession, query
from backend.tests.test_reranker import evidence_hit


class ServiceLeaseTests(unittest.TestCase):
    def test_expiry_requires_timezone(self):
        for value in ("tomorrow", "2026-09-06T12:00:00"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_service_expiry(value)
        self.assertIsNone(parse_service_expiry(""))
        self.assertEqual(
            parse_service_expiry("2026-09-06T14:00:00+02:00"),
            datetime(2026, 9, 6, 12, tzinfo=timezone.utc),
        )

    def test_request_budget_includes_shutdown_margin(self):
        now = datetime(2026, 9, 6, tzinfo=timezone.utc)
        with patch("search.service_lease.datetime") as clock:
            clock.now.return_value = now
            self.assertFalse(lease_allows_request(now + timedelta(seconds=38), 8))
            self.assertTrue(lease_allows_request(now + timedelta(seconds=39), 8))
            self.assertTrue(lease_allows_request(None, 8))

    def test_expired_semantic_service_makes_no_network_call(self):
        session = FakeSession({})
        client = RemoteBgeM3ReviewRetriever(
            base_url="https://example.invalid", release_id="v1", session=session,
            expires_at="2000-01-01T00:00:00Z",
        )
        outcome = client.retrieve(
            review_source_sha256="a" * 64, facility_ids=["a"],
            queries=[query()], limit_per_facility=1,
        )
        self.assertEqual(outcome.status, "service_expired")
        self.assertEqual(session.calls, [])

    def test_expired_reranker_preserves_candidates_without_network_call(self):
        session = FakeSession({})
        client = RemoteEvidenceReranker(
            base_url="https://example.invalid", session=session,
            expires_at="2000-01-01T00:00:00Z",
        )
        hits = (evidence_hit("one", "a", "친절"),)
        outcome = client.rerank("kind", hits)
        self.assertFalse(outcome.used)
        self.assertEqual(outcome.reason, "service_expired")
        self.assertEqual(outcome.hits, hits)
        self.assertEqual(session.calls, [])

    def test_wrong_model_revision_rejects_results(self):
        session = FakeSession({
            "schema_version": "seouldoc.semantic-results/v1",
            "release": {"release_id": "v1", "review_source_sha256": "a" * 64,
                        "model_id": "BAAI/bge-m3", "model_revision": "wrong"},
            "results": [],
        })
        client = RemoteBgeM3ReviewRetriever(
            base_url="https://example.invalid", release_id="v1", session=session,
            model_revision="expected",
        )
        outcome = client.retrieve(
            review_source_sha256="a" * 64, facility_ids=["a"],
            queries=[query()], limit_per_facility=1,
        )
        self.assertEqual(outcome.status, "release_mismatch")


if __name__ == "__main__":
    unittest.main()
