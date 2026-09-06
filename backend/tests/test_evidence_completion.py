"""Regression checks separating evidence coverage from service availability."""

from dataclasses import replace
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.tests.test_evidence_retrieval import (
    FakeScopedEvidence, FailedSemanticEvidenceSource, hit, rules,
)
from search.evidence_retrieval import (
    ConstraintEvidenceRetriever, EvidenceRecallPolicy, ScoredEvidence, select_evidence_groups,
)
from search.reranker import RerankOutcome


class FailingReranker:
    def rerank(self, query, hits):
        return RerankOutcome(tuple(hits), False, "request_failed")


class EvidenceCompletionTests(unittest.TestCase):
    def collect(self, **kwargs):
        scoped = FakeScopedEvidence()
        scoped.review_source_sha256 = "d" * 64
        return ConstraintEvidenceRetriever(**kwargs).collect(
            scoped_index=scoped, rules=rules(),
            shortlisted_facility_ids=("alpha",),
            displayed_facility_ids=("alpha",),
        )

    def test_remote_reranker_failure_prevents_complete(self):
        result = self.collect(reranker=FailingReranker())
        self.assertTrue(result.evidence)
        self.assertEqual(result.finish_status, "partial_evidence")

    def test_production_policy_requires_both_services(self):
        result = self.collect(policy=EvidenceRecallPolicy(require_remote_services=True))
        self.assertEqual(result.finish_status, "partial_evidence")

    def test_remote_semantic_failure_prevents_complete(self):
        result = self.collect(semantic_source=FailedSemanticEvidenceSource())
        self.assertTrue(result.evidence)
        self.assertEqual(result.finish_status, "partial_evidence")

    def test_warning_survives_more_distinctive_positive_reviews(self):
        base = ScoredEvidence(
            hit=hit("positive", "alpha", "The doctor explains clearly.", 1),
            matched_constraint_ids=frozenset({"explanation"}),
            roles=frozenset({"support"}), lexical_rank=1,
            rerank_rank=1, rerank_score=1.0, distinctiveness=1.0,
            local_cluster_id="positive", local_cluster_size=1,
            corroboration_count=1, selection_score=1.0,
        )
        risk = replace(
            base, hit=hit("warning", "alpha", "The nurses were unfriendly.", 9),
            matched_constraint_ids=frozenset({"avoid_unfriendly_nurses"}),
            roles=frozenset({"risk"}), distinctiveness=0.0, selection_score=0.01,
        )
        positives = tuple(replace(
            base, hit=hit(f"positive-{i}", "alpha", "Clear explanations", i),
            local_cluster_id=f"positive-{i}",
        ) for i in range(5))
        for limit in (1, 2, 3):
            with self.subTest(limit=limit):
                selected = select_evidence_groups(
                    ("alpha",), (*positives, risk), (), limit=limit,
                )["alpha"].presented
                self.assertIn("warning", [item.hit.evidence_id for item in selected])


if __name__ == "__main__":
    unittest.main()
