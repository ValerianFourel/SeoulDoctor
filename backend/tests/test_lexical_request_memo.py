from collections import Counter
from dataclasses import replace
from unittest import TestCase

from backend.tests.test_evidence_retrieval import hit
from search.evidence_retrieval import (
    ConstraintEvidenceRetriever, EvidenceConstraint, allocate_admissions,
    assess_facility_coverage, select_evidence_groups,
)
from search.reranker import RerankOutcome


class ImmutableBoundary:
    def __init__(self, label="original"):
        self.calls = []
        self.originals = {}
        self.label = label
        self.fail_next = False

    def search_evidence_for_facilities(self, query, *, facility_ids, limit_per_facility, source_types):
        self.calls.append((query, facility_ids, limit_per_facility, tuple(source_types)))
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("fixture search failure")
        source = "verbatim_review" if "verbatim_review" in source_types else "medical_info"
        result = []
        for owner in facility_ids:
            identity = f"{self.label}:{owner}:{query}:{source}"
            if identity not in self.originals:
                original = hit(identity, owner, f"{self.label} {query}", len(self.originals))
                self.originals[identity] = replace(
                    original, source_type=source, is_verbatim=source == "verbatim_review",
                )
            result.append(self.originals[identity])
        return result


class ScoringBoundary:
    def rerank(self, query, hits):
        return RerankOutcome(tuple(hits), True, "ok", scores=tuple(
            (item.evidence_id, 1.0 / (item.ordinal + 1)) for item in hits
        ))


def constraint(identity, role="disease", types=("verbatim_review",), korean=()):
    return EvidenceConstraint(identity, role, ("ankle pain ankle hurts",), korean, types, True, 0)


class LexicalRequestMemoTests(TestCase):
    def test_recorded_duplicate_queries_keep_all_cells_and_owned_admissions(self):
        constraints = (
            constraint("evidence:ankle_pain", types=("facility_fact", "medical_info", "review_summary", "verbatim_review"), korean=("발목 통증 발목이 아파 발목",)),
            constraint("visit_reason:ankle_pain", types=("medical_info", "verbatim_review"), korean=("발목 통증 발목이 아파 발목",)),
        )
        scope = ImmutableBoundary()
        retriever = ConstraintEvidenceRetriever(reranker=ScoringBoundary())
        with self.assertLogs("search.evidence_retrieval", level="INFO") as logs:
            batch = retriever._search(scope, constraints, ("alpha", "beta"), limit=5)
        self.assertEqual(len(scope.calls), 6)
        self.assertEqual(Counter(call[3] for call in scope.calls), {
            ("verbatim_review",): 2,
            ("facility_fact", "medical_info", "review_summary"): 2,
            ("medical_info",): 2,
        })
        self.assertEqual(sum("cache_hits=1" in line for line in logs.output), 2)
        self.assertEqual(len(batch.candidates), 16)
        for owner in ("alpha", "beta"):
            for item in constraints:
                for language in ("en", "ko"):
                    cell = [candidate for candidate in batch.candidates
                            if candidate.cell.facility_id == owner
                            and candidate.cell.constraint_id == item.constraint_id
                            and candidate.query_language == language]
                    self.assertEqual([(candidate.hit.source_type, candidate.lexical_rank) for candidate in cell],
                                     [("verbatim_review", 1), ("medical_info", 2)])
                    self.assertTrue(all(candidate.hit == scope.originals[candidate.hit.evidence_id]
                                        for candidate in cell))
        admitted = allocate_admissions(batch.candidates, budget=224)
        self.assertEqual({item.hit for item in admitted}, set(scope.originals.values()))
        scored, used, reason = retriever._score(admitted, batch.candidates, constraints)
        self.assertTrue(used)
        self.assertEqual(reason, "ok")
        self.assertTrue(all(item.matched_constraint_ids == frozenset(c.constraint_id for c in constraints)
                            for item in scored))
        coverage = assess_facility_coverage(("alpha", "beta"), constraints, scored)
        self.assertEqual(len(coverage), 4)
        self.assertTrue(all(cell.status == "matched" for cell in coverage))
        groups = select_evidence_groups(("alpha", "beta"), scored, coverage, limit=3)
        for owner, group in groups.items():
            expected = {item for item in scope.originals.values() if item.facility_id == owner and item.is_verbatim}
            self.assertEqual({item.hit for item in group.presented}, expected)

    def test_one_lexical_result_keeps_separate_disease_support_and_risk_roles(self):
        constraints = tuple(constraint(role, role=role) for role in ("disease", "support", "risk"))
        scope = ImmutableBoundary()
        retriever = ConstraintEvidenceRetriever(reranker=ScoringBoundary())
        batch = retriever._search(scope, constraints, ("alpha",), limit=5)
        self.assertEqual(len(scope.calls), 1)
        self.assertEqual({item.cell.role for item in batch.candidates}, {"disease", "support", "risk"})
        admitted = allocate_admissions(batch.candidates, budget=224)
        scored, _, _ = retriever._score(admitted, batch.candidates, constraints)
        self.assertEqual(scored[0].roles, frozenset({"disease", "support", "risk"}))
        self.assertEqual(scored[0].matched_constraint_ids, frozenset({"disease", "support", "risk"}))

    def test_source_type_and_limit_are_part_of_the_key(self):
        scope = ImmutableBoundary()
        constraints = (
            constraint("mixed", types=("medical_info", "verbatim_review")),
            constraint("reviews"),
            constraint("facts", types=("medical_info",)),
        )
        ConstraintEvidenceRetriever()._search(scope, constraints, ("alpha",), limit=5)
        self.assertEqual([(call[2], call[3]) for call in scope.calls], [
            (3, ("verbatim_review",)), (2, ("medical_info",)),
            (5, ("verbatim_review",)), (5, ("medical_info",)),
        ])

    def test_memo_cannot_cross_requests_or_scope_instances(self):
        retriever = ConstraintEvidenceRetriever()
        constraints = (constraint("first"), constraint("second"))
        original = ImmutableBoundary()
        changed = ImmutableBoundary("different release")
        first = retriever._search(original, constraints, ("alpha",), limit=5)
        second = retriever._search(changed, constraints, ("alpha",), limit=5)
        third = retriever._search(original, constraints, ("beta",), limit=5)
        self.assertNotEqual(first.candidates[0].hit, second.candidates[0].hit)
        self.assertEqual({item.hit.facility_id for item in third.candidates}, {"beta"})
        self.assertEqual(len(original.calls), 2)
        self.assertEqual(len(changed.calls), 1)

    def test_failed_search_is_not_memoized_as_empty_success(self):
        retriever = ConstraintEvidenceRetriever()
        scope = ImmutableBoundary()
        scope.fail_next = True
        constraints = (constraint("first"), constraint("second"))
        with self.assertRaisesRegex(RuntimeError, "fixture search failure"):
            retriever._search(scope, constraints, ("alpha",), limit=5)
        result = retriever._search(scope, constraints, ("alpha",), limit=5)
        self.assertEqual(len(scope.calls), 2)
        self.assertEqual(len(result.candidates), 2)

    def test_successful_empty_result_is_reused(self):
        scope = ImmutableBoundary()
        result = ConstraintEvidenceRetriever()._search(
            scope, (constraint("first"), constraint("second")), (), limit=5,
        )
        self.assertEqual(len(scope.calls), 1)
        self.assertEqual(result.candidates, ())
