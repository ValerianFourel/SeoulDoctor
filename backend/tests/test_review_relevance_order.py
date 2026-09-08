from pathlib import Path
import runpy
from unittest import TestCase
from unittest.mock import patch

from backend.tests.test_evidence_retrieval import FakeScopedEvidence, hit, rules
from backend.tests.test_reranker import FakeSession
from search.evidence_retrieval import (
    ConstraintCell, ConstraintEvidenceRetriever, EvidenceCellCandidate,
    EvidenceConstraint, ScoredEvidence, select_evidence_groups,
)
from search.reranker import RemoteEvidenceReranker, RerankOutcome


def scored(identity, text, score, *, owner="alpha", fallback=0.1, distinctiveness=0.1, role="disease"):
    return ScoredEvidence(
        hit(identity, owner, text, len(identity)), frozenset({role}), frozenset({role}),
        1, 1 if score is not None else None, score, distinctiveness, identity, 1, 1, fallback,
    )


class ReviewRelevanceOrderTests(TestCase):
    def test_actual_relevance_wins_over_novelty_and_unscored_fused_rank(self):
        original = (
            scored("generic", "굿", None, fallback=0.99),
            scored("long-unrelated", "A lengthy description of the waiting room and parking.", 0.00001, distinctiveness=1),
            scored("ankle", "발목 겹질림", 0.03),
            scored("fracture", "The clinic examined my ankle fracture.", 0.02),
            scored("negative", "My foot still hurt after treatment.", 0.01),
            scored("other-owner", "Foot treatment", 0.99, owner="beta"),
        )
        groups = select_evidence_groups(("alpha", "beta"), original, (), limit=3)
        self.assertEqual([x.hit.evidence_id for x in groups["alpha"].presented], ["ankle", "fracture", "negative"])
        self.assertEqual(groups["beta"].presented[0].hit.facility_id, "beta")
        self.assertEqual(groups["alpha"].supporting[-1].hit.evidence_id, "generic")
        self.assertEqual({x.hit for x in groups["alpha"].supporting}, {x.hit for x in original if x.hit.facility_id == "alpha"})

    def test_punctuation_cannot_fill_first_page_but_short_reports_remain(self):
        original = (
            scored("punctuation", "ㆍ . !!!", 0.99),
            scored("short-medical", "발목 겹질림", 0.7),
            scored("short-negative", "Nurse was rude", 0.2, role="risk"),
            scored("short-chinese", "医生解释清楚", 0.4),
        )
        groups = select_evidence_groups(("alpha",), original, (), limit=3)
        self.assertEqual({x.hit.evidence_id for x in groups["alpha"].presented},
                         {"short-medical", "short-negative", "short-chinese"})
        self.assertEqual(groups["alpha"].presented[0].hit.evidence_id, "short-negative")
        two_reports = select_evidence_groups(("alpha",), original[:3], (), limit=3)
        self.assertEqual(len(two_reports["alpha"].presented), 2)

    def test_reranker_default_covers_all_224_admissions_and_rejects_257(self):
        inputs = tuple(hit(f"review-{i}", "alpha", "Foot treatment", i) for i in range(224))
        session = FakeSession([{"index": i, "score": i / 224} for i in range(224)])
        outcome = RemoteEvidenceReranker(base_url="https://fixture", session=session).rerank("foot", inputs)
        self.assertEqual(len(session.calls[0]["json"]["texts"]), 224)
        self.assertEqual(len(outcome.scores), 224)
        self.assertEqual(outcome.reason, "ok")
        with self.assertRaises(ValueError):
            RemoteEvidenceReranker(base_url="https://fixture", max_candidates=257)
        config = Path(__file__).resolve().parents[1] / "config.py"
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(runpy.run_path(str(config))["RERANKER_MAX_CANDIDATES"], 256)
        with patch.dict("os.environ", {"RERANKER_MAX_CANDIDATES": "257"}, clear=True):
            with self.assertRaises(ValueError):
                runpy.run_path(str(config))

    def test_truncated_tail_has_no_gpu_rank_and_marks_partial_execution(self):
        session = FakeSession([{"index": 0, "score": 0.7}])
        reranker = RemoteEvidenceReranker(base_url="https://fixture", max_candidates=1, session=session)
        inputs = (hit("first", "alpha", "Foot pain", 0), hit("tail", "alpha", "Foot fracture", 1))
        outcome = reranker.rerank("foot", inputs)
        self.assertEqual(outcome.reason, "candidate_limit")
        self.assertEqual(outcome.hits, inputs)
        self.assertEqual(outcome.scores, (("first", 0.7),))
        admitted = tuple(EvidenceCellCandidate(h, ConstraintCell("alpha", "foot", "disease"), 1, "en") for h in inputs)
        constraints = (EvidenceConstraint("foot", "disease", ("foot",), (), ("verbatim_review",), True, 0),)
        results, _, reason = ConstraintEvidenceRetriever(reranker=reranker)._score(admitted, admitted, constraints)
        self.assertEqual(reason, "candidate_limit")
        tail = next(x for x in results if x.hit.evidence_id == "tail")
        self.assertIsNone(tail.rerank_rank)
        self.assertIsNone(tail.rerank_score)
        scoped = FakeScopedEvidence()
        scoped.review_source_sha256 = "d" * 64
        collected = ConstraintEvidenceRetriever(reranker=reranker).collect(
            scoped_index=scoped, rules=rules(), shortlisted_facility_ids=("alpha",), displayed_facility_ids=("alpha",),
        )
        self.assertEqual(collected.execution_status, "partial")
        self.assertIn("candidate_limit", collected.reranker_reason)
        self.assertIn("reranker_failed", collected.reason_codes)

    def test_success_without_scores_is_not_claimed_as_complete_reranking(self):
        class RankOnly:
            def rerank(self, query, hits):
                return RerankOutcome(tuple(hits), True, "ok")

        h = hit("first", "alpha", "Foot pain", 0)
        admitted = (EvidenceCellCandidate(h, ConstraintCell("alpha", "foot", "disease"), 1, "en"),)
        constraints = (EvidenceConstraint("foot", "disease", ("foot",), (), ("verbatim_review",), True, 0),)
        results, _, reason = ConstraintEvidenceRetriever(reranker=RankOnly())._score(admitted, admitted, constraints)
        self.assertEqual(reason, "incomplete_scores")
        self.assertIsNone(results[0].rerank_rank)
        self.assertIsNone(results[0].rerank_score)
