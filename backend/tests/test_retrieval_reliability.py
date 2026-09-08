"""Evidence recall, execution and ownership are separate from interpretation."""

from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.tests.test_evidence_retrieval import (
    FakeScopedEvidence,
    FakeSemanticEvidenceSource,
    HybridScopedEvidence,
    hit,
    rules,
)
from backend.tests.test_live_retrieval import (
    FakeIndex,
    FakePipeline,
    ShortlistEvidenceScopedIndex,
    make_rules,
    make_scope,
)
from backend.tests.test_search_scope import facility
from search.contracts import validate_evidence_source_types
from search.evidence_retrieval import (
    ConstraintEvidenceRetriever,
    EvidenceRecallPolicy,
    ScoredEvidence,
    evidence_payload,
)
from search.live_retrieval import CandidateRetrievalAdapter, RetrievalQuery
from search.reranker import RerankOutcome
from search.rules import RulesCompiler, compile_legacy_state_rules
from search.scope import ScopeBuilder
from search.semantic_retriever import SemanticReviewOutcome


class IdentityReranker:
    def rerank(self, query, hits):
        return RerankOutcome(tuple(hits), True, "ok")


class EmptyScopedEvidence:
    review_source_sha256 = "d" * 64

    def search_evidence_for_facilities(self, *args, **kwargs):
        return []

    def resolve_evidence_ids(self, evidence_ids):
        return []


class EmptySemanticSource:
    def __init__(self, status="ok"):
        self.status = status
        self.calls = 0

    def retrieve(self, **kwargs):
        self.calls += 1
        return SemanticReviewOutcome(self.status)


class RetrievalReliabilityTests(unittest.TestCase):
    def english_rules(self):
        compiled = RulesCompiler().compile(
            original_query="I need a dentist in Seoul. English consultation is required.",
            turn_id="turn-1",
            proposal={"specialty": "dentist", "location": "Seoul",
                      "hard_keywords": ["English consultation"]},
        )
        self.assertIsNotNone(compiled.rules)
        return compiled.rules

    def test_hard_language_queries_both_original_lexical_and_semantic_sources(self):
        class RecordingIndex(HybridScopedEvidence):
            def __init__(self):
                super().__init__()
                self.source_types = []

            def search_evidence_for_facilities(self, query, **kwargs):
                self.source_types.append(kwargs["source_types"])
                return super().search_evidence_for_facilities(query, **kwargs)

        scoped = RecordingIndex()
        semantic = FakeSemanticEvidenceSource()
        result = ConstraintEvidenceRetriever(
            semantic_source=semantic, reranker=IdentityReranker(),
            policy=EvidenceRecallPolicy(require_remote_services=True),
        ).collect(
            scoped_index=scoped, rules=self.english_rules(),
            shortlisted_facility_ids=("alpha",), displayed_facility_ids=("alpha",),
        )
        self.assertTrue(scoped.source_types)
        self.assertTrue(all("verbatim_review" in types for types in scoped.source_types))
        self.assertTrue(semantic.calls)
        self.assertTrue(all(
            query.constraint_id == "attribute:english_consultation"
            for call in semantic.calls for query in call["queries"]
        ))
        self.assertTrue(any(event.lexical_rank for event in result.admissions))
        self.assertTrue(any(event.dense_rank for event in result.admissions))
        original = next(item.hit for item in result.evidence if item.hit.evidence_id == "semantic-decisive")
        self.assertEqual(original.facility_id, "alpha")
        self.assertEqual(original.original_text, "아이에게 천천히 설명하고 눈높이에 맞춰 진료했어요")
        # A retrieval association with English is still only a match, even
        # when the review actually discusses explanations instead of language.
        self.assertEqual(result.coverage[0].status, "matched")
        self.assertEqual(result.execution_status, "complete")

    def test_unknown_source_type_is_rejected_by_rules_and_query_contract(self):
        requirement = self.english_rules().evidence[0]
        with self.assertRaisesRegex(ValueError, "unsupported evidence source"):
            replace(requirement, source_types=frozenset({"review"}))
        for value in ("review", "untrusted_table", "", "VERBATIM_REVIEW"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_evidence_source_types((value,))

    def test_factual_inquiry_retrieves_optional_evidence_without_filtering(self):
        state = SimpleNamespace(
            specialty="치과", is_citywide_search=True,
            inquiries=["Can you confirm English consultations, or should I check directly?"],
            visit_reason="routine checkup",
        )
        compiled = compile_legacy_state_rules(
            state.inquiries[0], state, turn_id="inquiry-turn",
        )
        self.assertIsNotNone(compiled.rules)
        self.assertEqual(compiled.rules.hard.required_attributes, ())
        inquiry = next(item for item in compiled.rules.evidence if item.requirement_id == "inquiry:english_consultation")
        self.assertFalse(inquiry.support_required)
        self.assertIn("verbatim_review", inquiry.source_types)
        result = ConstraintEvidenceRetriever().collect(
            scoped_index=EmptyScopedEvidence(), rules=compiled.rules,
            shortlisted_facility_ids=("alpha",), displayed_facility_ids=("alpha",),
        )
        self.assertEqual(result.finish_status, "complete")
        self.assertEqual(result.execution_status, "complete")
        self.assertIn("inquiry:english_consultation", result.by_facility["alpha"].unverified)
        next_rules = RulesCompiler().compile(
            original_query="Can you compare these facilities?", turn_id="inquiry-turn-2",
            proposal={"location": "Seoul"}, previous_rules=compiled.rules,
        ).rules
        self.assertFalse(any(item.requirement_id.startswith("inquiry:") for item in next_rules.evidence))

    def test_multiple_positive_retrieval_roles_do_not_claim_mixed_sentiment(self):
        original = hit("positive", "alpha", "The doctor explained my wrist treatment clearly.", 2)
        item = ScoredEvidence(
            original, frozenset({"wrist", "explanations"}),
            frozenset({"disease", "support"}), 1, 1, 0.8, 0.5, "positive", 1, 1, 0.8,
        )
        payload = evidence_payload(item)
        self.assertEqual(payload["retrieval_roles"], ["disease", "support"])
        self.assertNotIn("evidence_role", payload)
        self.assertNotIn("sentiment", payload)
        self.assertEqual(payload["text"], original.original_text)
        self.assertEqual(payload["source_locator"], original.source_locator)

    def test_complete_search_can_have_missing_evidence_and_no_reranker_work(self):
        semantic = EmptySemanticSource()
        result = ConstraintEvidenceRetriever(
            semantic_source=semantic,
            policy=EvidenceRecallPolicy(require_remote_services=True),
        ).collect(
            scoped_index=EmptyScopedEvidence(), rules=rules(),
            shortlisted_facility_ids=("alpha",), displayed_facility_ids=("alpha",),
        )
        self.assertEqual(result.execution_status, "complete")
        self.assertEqual(result.reason_codes, ())
        self.assertEqual(result.finish_status, "partial_evidence")
        self.assertFalse(result.reranker_applicable)
        self.assertTrue(all(cell.status == "missing" for cell in result.coverage))

    def test_semantic_timeout_is_not_repeated_and_retains_local_originals(self):
        semantic = EmptySemanticSource("request_failed")
        result = ConstraintEvidenceRetriever(
            semantic_source=semantic, reranker=IdentityReranker(),
            policy=EvidenceRecallPolicy(require_remote_services=True),
        ).collect(
            scoped_index=HybridScopedEvidence(), rules=rules(),
            shortlisted_facility_ids=("alpha", "beta"), displayed_facility_ids=("alpha", "beta"),
        )
        self.assertEqual(semantic.calls, 1)
        self.assertEqual(result.execution_status, "partial")
        self.assertEqual(result.reason_codes, ("semantic_request_failed",))
        self.assertEqual(result.by_facility["alpha"].presented[0].hit.original_text, "친절해요")

    def test_fact_only_search_does_not_require_a_review_semantic_channel(self):
        fact_rules = replace(rules(), evidence=(replace(
            rules().evidence[0], source_types=frozenset({"facility_fact"}),
        ),))
        result = ConstraintEvidenceRetriever(
            policy=EvidenceRecallPolicy(require_remote_services=True),
        ).collect(
            scoped_index=EmptyScopedEvidence(), rules=fact_rules,
            shortlisted_facility_ids=("alpha",), displayed_facility_ids=("alpha",),
        )
        self.assertEqual(result.semantic_status, "not_applicable")
        self.assertFalse(result.semantic_applicable)
        self.assertEqual(result.execution_status, "complete")
        self.assertEqual(result.finish_status, "partial_evidence")

    def test_a_failed_targeted_semantic_retry_stops_remaining_remote_retries(self):
        class FailsOnRetry(EmptySemanticSource):
            def retrieve(self, **kwargs):
                self.calls += 1
                return SemanticReviewOutcome("ok" if self.calls == 1 else "request_failed")

        semantic = FailsOnRetry()
        result = ConstraintEvidenceRetriever(semantic_source=semantic).collect(
            scoped_index=EmptyScopedEvidence(), rules=rules(),
            shortlisted_facility_ids=("alpha", "beta"), displayed_facility_ids=("alpha", "beta"),
        )
        self.assertEqual(semantic.calls, 2)
        self.assertTrue(result.retry_ran)
        self.assertEqual(result.reason_codes, ("semantic_request_failed",))

    def test_reranker_failure_retains_all_admitted_originals(self):
        class ThrowingReranker:
            def rerank(self, query, hits):
                raise TimeoutError("synthetic failure")

        result = ConstraintEvidenceRetriever(reranker=ThrowingReranker()).collect(
            scoped_index=FakeScopedEvidence(), rules=rules(),
            shortlisted_facility_ids=("alpha",), displayed_facility_ids=("alpha",),
        )
        self.assertEqual(result.execution_status, "partial")
        self.assertEqual(result.reason_codes, ("reranker_request_failed",))
        self.assertEqual(result.finish_status, "complete")
        self.assertEqual({item.hit.evidence_id for item in result.evidence}, {"generic-1", "decisive"})

    def test_coverage_includes_attachment_candidates_outside_retry_set(self):
        result = ConstraintEvidenceRetriever().collect(
            scoped_index=FakeScopedEvidence(), rules=rules(),
            shortlisted_facility_ids=("alpha", "beta"), displayed_facility_ids=("alpha",),
            attachment_facility_ids=("alpha", "beta"),
        )
        beta_cells = [cell for cell in result.coverage if cell.facility_id == "beta"]
        self.assertEqual(len(beta_cells), 2)
        self.assertTrue(all(cell.status == "missing" for cell in beta_cells))
        self.assertEqual(set(result.by_facility["beta"].unverified), {"kind_children", "unfriendly_nurses"})

    def test_final_five_cards_have_explicit_coverage_even_outside_shortlist(self):
        identifiers = ("alpha", "bravo", "charlie", "delta", "echo")
        result = CandidateRetrievalAdapter(
            active_index=FakeIndex(ShortlistEvidenceScopedIndex(identifiers)),
            legacy_pipeline=FakePipeline(),
            evidence_policy=EvidenceRecallPolicy(shortlist_limit=3),
        ).rank(
            scope=make_scope(identifiers),
            eligible=pd.DataFrame({"place_id": identifiers, "name": identifiers,
                                   "distance_km": [1, 2, 3, 4, 5]}),
            rules=make_rules(evidence=True),
            query=RetrievalQuery("friendly clinic", 5, "distance", 0.9),
        )
        metadata = result.dataframe.attrs["rag_metadata"]
        self.assertEqual(metadata["retrieval_execution_status"], "complete")
        self.assertFalse(metadata["coverage_sufficient"])
        self.assertEqual({cell["facility_id"] for cell in metadata["evidence_coverage"]}, set(identifiers))
        for _, row in result.dataframe.iterrows():
            group = row["retrieval_evidence_groups"]
            self.assertEqual(group["coverage_scope"], make_rules().rules_hash)
            if row["place_id"] in {"delta", "echo"}:
                self.assertEqual(group["coverage_status"], "unassessed")
                self.assertTrue(group["unverified"])

    def test_legacy_english_boolean_does_not_satisfy_a_hard_language_requirement(self):
        flagged = facility("flagged")
        flagged["has_english"] = True
        textual = facility("textual")
        textual["medical_info_parsed"] = {"consultation": "English consultation"}
        catalog = pd.DataFrame([flagged, textual])
        selected = ScopeBuilder().build(catalog, self.english_rules(), index_version="test-v1")
        self.assertEqual(selected.facility_ids, ("textual",))
        response_language_only = replace(self.english_rules(), hard=replace(
            self.english_rules().hard, required_attributes=(),
        ))
        self.assertEqual(
            ScopeBuilder().build(catalog, response_language_only, index_version="test-v1").facility_ids,
            ("flagged", "textual"),
        )


if __name__ == "__main__":
    unittest.main()
