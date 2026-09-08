from dataclasses import replace
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.tests.test_evidence_retrieval import hit, rules
from search.evidence_retrieval import ConstraintEvidenceRetriever, EvidenceRecallPolicy
from search.reranker import RerankOutcome


class GeneralReviewIndex:
    review_source_sha256 = "d" * 64

    def __init__(self, reviews=()):
        self.reviews = reviews
        self.queries = []

    def search_evidence_for_facilities(self, query, *, facility_ids, limit_per_facility, source_types):
        self.queries.append((query, facility_ids, source_types))
        return [review for review in self.reviews
                if review.facility_id in facility_ids
                and any(term.casefold() in review.original_text.casefold()
                        for term in query.split())][:limit_per_facility]


class GeneralReviewRetrievalTests(unittest.TestCase):
    def setUp(self):
        base = rules()
        self.rules = replace(base, original_query="orthopedic doctor near Jonggak", evidence=(),
                             hard=replace(base.hard, specialty_ids=frozenset({"orthopedics"})))
        self.reviews = (
            hit("ankle-positive", "alpha", "발목 진료를 자세히 설명해 주었습니다.", 1),
            hit("ankle-negative", "alpha", "발목 치료 후에도 아프고 설명이 부족했습니다.", 2),
            hit("unrelated", "alpha", "주차장이 넓어요.", 3),
            hit("other-owner", "beta", "발목 진료를 받았습니다.", 4),
        )

    def collect(self, index, **kwargs):
        return ConstraintEvidenceRetriever(**kwargs).collect(
            scoped_index=index, rules=self.rules,
            shortlisted_facility_ids=("alpha",), displayed_facility_ids=("alpha",),
            review_query="발목 ankle pain 정형외과",
        )

    def test_no_extra_requirements_retrieve_two_owned_relevant_originals(self):
        result = self.collect(GeneralReviewIndex(self.reviews))
        presented = result.by_facility["alpha"].presented
        self.assertEqual({item.hit.evidence_id for item in presented},
                         {"ankle-positive", "ankle-negative"})
        self.assertEqual({item.hit.original_text for item in presented},
                         {review.original_text for review in self.reviews[:2]})
        self.assertTrue(all(not cell.required for cell in result.coverage))
        self.assertEqual(result.execution_status, "complete")
        self.assertEqual(result.finish_status, "complete")

    def test_specialty_only_uses_canonical_korean_specialty(self):
        reviews = (hit("specialty", "alpha", "정형외과 진료 경험입니다.", 1),)
        index = GeneralReviewIndex(reviews)
        result = ConstraintEvidenceRetriever().collect(
            scoped_index=index, rules=self.rules,
            shortlisted_facility_ids=("alpha",), displayed_facility_ids=("alpha",),
        )
        self.assertEqual([item.hit.evidence_id for item in result.evidence], ["specialty"])
        self.assertTrue(any("정형외과" in query for query, _, _ in index.queries))

    def test_empty_source_runs_search_without_inventing_reviews(self):
        index = GeneralReviewIndex()
        result = self.collect(index)
        self.assertTrue(index.queries)
        self.assertEqual(result.by_facility["alpha"].presented, ())
        self.assertEqual(result.execution_status, "complete")
        self.assertEqual(result.finish_status, "complete")
        self.assertEqual(result.reranker_reason, "no_candidates")

    def test_no_matching_reviews_does_not_pad_with_unrelated_reviews(self):
        result = self.collect(GeneralReviewIndex(self.reviews[2:]))
        self.assertEqual(result.evidence, ())
        self.assertEqual(result.by_facility["alpha"].presented, ())

    def test_local_retrieval_failure_is_not_a_completed_empty_search(self):
        class BrokenIndex(GeneralReviewIndex):
            def search_evidence_for_facilities(self, *args, **kwargs):
                raise OSError("index unavailable")
        with self.assertRaisesRegex(OSError, "index unavailable"):
            self.collect(BrokenIndex())

    def test_remote_failure_preserves_owned_positive_and_negative_reviews(self):
        class BrokenReranker:
            def rerank(self, query, hits):
                return RerankOutcome(tuple(hits), False, "request_failed")
        result = self.collect(GeneralReviewIndex(self.reviews), reranker=BrokenReranker(),
                              policy=EvidenceRecallPolicy(require_remote_services=True))
        self.assertEqual(len(result.by_facility["alpha"].presented), 2)
        self.assertEqual(result.execution_status, "partial")
        self.assertIn("reranker_request_failed", result.reason_codes)
        self.assertEqual(result.finish_status, "complete")

    def test_real_immutable_index_returns_bilingual_owned_originals(self):
        from backend.tests import test_search_indexes

        fixture = test_search_indexes.SearchIndexReleaseTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.repository.publish(fixture._request())
        fixture.repository.activate("test-v1")
        with fixture.repository.open_active() as active:
            with active.within(fixture._scope("test-v1", ("alpha",))) as scoped:
                result = ConstraintEvidenceRetriever().collect(
                    scoped_index=scoped,
                    rules=replace(self.rules, original_query="acne 여드름"),
                    shortlisted_facility_ids=("alpha",),
                    displayed_facility_ids=("alpha",),
                )
                presented = result.by_facility["alpha"].presented
                self.assertEqual({item.hit.original_text for item in presented}, {
                    "English friendly acne treatment", "여드름 치료가 정말 친절해요",
                })
                originals = {item.evidence_id: item for item in scoped.resolve_evidence_ids(
                    tuple(item.hit.evidence_id for item in presented)
                )}
                for item in presented:
                    self.assertEqual(item.hit.facility_id, "alpha")
                    self.assertTrue(item.hit.is_verbatim)
                    self.assertEqual(item.hit.original_text,
                                     originals[item.hit.evidence_id].original_text)
                    self.assertEqual(item.hit.source_locator,
                                     originals[item.hit.evidence_id].source_locator)


if __name__ == "__main__":
    unittest.main()
