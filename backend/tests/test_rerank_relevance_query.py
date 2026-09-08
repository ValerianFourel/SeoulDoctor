from unittest import TestCase

from search.evidence_retrieval import (
    ConstraintCell,
    ConstraintEvidenceRetriever,
    EvidenceCellCandidate,
    EvidenceConstraint,
    rerank_relevance_query,
)
from search.reranker import RerankOutcome
from backend.tests.test_evidence_retrieval import hit


def constraint(identity, role, english, korean=()):
    return EvidenceConstraint(identity, role, tuple(english), tuple(korean),
                              ("verbatim_review",), True, 0)


class RerankRelevanceQueryTests(TestCase):
    def test_bilingual_aliases_do_not_become_reranker_instructions(self):
        constraints = (
            constraint("ankle", "disease", ("ankle pain", "ankle treatment"), ("발목 통증", "발목")),
            constraint("explanation", "support", ("clear explanations", "detailed explanation"), ("자세한 설명", "설명")),
        )
        self.assertEqual(rerank_relevance_query(constraints),
                         "Patient reviews about ankle pain and clear explanations.")
        self.assertEqual(constraints[0].terms_ko, ("발목 통증", "발목"))
        self.assertEqual(constraints[1].terms_en, ("clear explanations", "detailed explanation"))

    def test_korean_only_constraints_keep_their_meaning(self):
        query = rerank_relevance_query((
            constraint("ankle", "disease", (), ("발목 통증", "발목")),
            constraint("nurses", "risk", (), ("불친절한 간호사", "간호사 불친절")),
        ))
        self.assertEqual(query, "발목 통증, 불친절한 간호사에 관한 환자 후기.")

    def test_missing_translation_does_not_remove_a_constraint(self):
        query = rerank_relevance_query((
            constraint("ankle", "disease", ("ankle pain",)),
            constraint("nurses", "risk", (), ("불친절한 간호사",)),
        ))
        self.assertIn("ankle pain", query)
        self.assertIn("불친절한 간호사", query)
        self.assertNotIn("|", query)

    def test_each_role_keeps_all_its_constraints_and_source_owners(self):
        constraints = (
            constraint("ankle", "disease", ("ankle pain", "foot pain"), ("발목 통증",)),
            constraint("explanation", "support", ("clear explanations",), ("자세한 설명",)),
            constraint("children", "support", ("kind with children",), ("아이에게 친절",)),
            constraint("nurses", "risk", ("unfriendly nurses",), ("간호사 불친절",)),
        )
        admitted = tuple(EvidenceCellCandidate(
            hit("review:" + item.constraint_id, "owner-" + str(index), "Original " + item.constraint_id, index),
            ConstraintCell("owner-" + str(index), item.constraint_id, item.role),
            1, "en",
        ) for index, item in enumerate(constraints))
        calls = []

        class CaptureReranker:
            def rerank(self, query, hits):
                calls.append((query, tuple(hits)))
                return RerankOutcome(tuple(hits), True, "ok")

        scored, used, reason = ConstraintEvidenceRetriever(reranker=CaptureReranker())._score(
            admitted, admitted, constraints,
        )
        self.assertTrue(used)
        self.assertEqual(reason, "ok")
        self.assertEqual([query for query, _ in calls], [
            "Patient reviews about ankle pain.",
            "Patient reviews about kind with children and clear explanations.",
            "Patient reviews about unfriendly nurses.",
        ])
        by_id = {item.hit.evidence_id: item for item in scored}
        for original in admitted:
            selected = by_id[original.hit.evidence_id]
            self.assertEqual(selected.hit, original.hit)
            self.assertEqual(selected.roles, {original.cell.role})
            self.assertEqual(selected.matched_constraint_ids, {original.cell.constraint_id})
