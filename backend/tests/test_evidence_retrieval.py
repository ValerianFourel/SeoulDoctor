from __future__ import annotations

from dataclasses import replace
import unittest

from search.contracts import (
    AreaRule,
    EvidenceRequirement,
    HardEligibility,
    RuleProvenance,
    SearchRules,
)
from search.evidence_retrieval import (
    ConstraintCell,
    ConstraintEvidenceRetriever,
    EvidenceCellCandidate,
    EvidenceRecallPolicy,
    ScoredEvidence,
    allocate_admissions,
    score_local_distinctiveness,
    select_evidence_groups,
)
from search.indexes.repository import EvidenceHit
from search.semantic_retriever import (
    SemanticEvidenceReference,
    SemanticReviewOutcome,
)


def hit(
    evidence_id: str,
    facility_id: str,
    text: str,
    ordinal: int,
) -> EvidenceHit:
    return EvidenceHit(
        evidence_id=evidence_id,
        facility_id=facility_id,
        ordinal=ordinal,
        score=1.0 / (ordinal + 1),
        channel="facility_evidence_lexical",
        source_type="verbatim_review",
        source_field="review_text",
        source_index=ordinal,
        source_locator=f"review_snapshot:{facility_id}:{ordinal}",
        original_text=text,
        language_hint="ko",
        visit_date="",
        scraped_at="",
        is_verbatim=True,
    )


def rules() -> SearchRules:
    provenance = RuleProvenance("user_explicit", (0, 4), "turn-1")
    requirements = (
        EvidenceRequirement(
            requirement_id="kind_children",
            terms_en=("kind with children",),
            terms_ko=("아이에게 친절",),
            match_mode="semantic",
            source_types=frozenset({"verbatim_review"}),
            support_required=True,
            evidence_role="support",
        ),
        EvidenceRequirement(
            requirement_id="unfriendly_nurses",
            terms_en=("unfriendly nurses",),
            terms_ko=("간호사 불친절",),
            match_mode="semantic",
            source_types=frozenset({"verbatim_review"}),
            support_required=True,
            evidence_role="risk",
        ),
    )
    return SearchRules(
        schema_version="1",
        original_query="kind children but avoid unfriendly nurses",
        language="en",
        hard=HardEligibility(
            specialty_ids=frozenset({"pediatrics"}),
            geography=AreaRule("seoul", "city", "Seoul", provenance),
            prohibited_facility_ids=frozenset(),
            prohibited_taxonomy_ids=frozenset(),
            required_attributes=(),
        ),
        soft=(),
        evidence=requirements,
        provenance={"hard.geography": provenance},
        rules_hash="a" * 64,
    )


class FakeScopedEvidence:
    def __init__(self) -> None:
        self.queries: list[tuple[str, tuple[str, ...], int]] = []

    def search_evidence_for_facilities(
        self,
        query: str,
        *,
        facility_ids: tuple[str, ...],
        limit_per_facility: int,
        source_types: tuple[str, ...],
    ) -> list[EvidenceHit]:
        self.queries.append((query, tuple(facility_ids), limit_per_facility))
        if "kind with children" in query or "아이에게 친절" in query:
            return [
                hit("generic-1", "alpha", "친절해요", 1),
                hit(
                    "decisive",
                    "alpha",
                    "의사는 아이에게 친절하지만 간호사는 불친절하고 무뚝뚝해요",
                    2,
                ),
            ]
        if "unfriendly nurses" in query or "간호사 불친절" in query:
            return [
                hit(
                    "decisive",
                    "alpha",
                    "의사는 아이에게 친절하지만 간호사는 불친절하고 무뚝뚝해요",
                    2,
                )
            ]
        return []


class HybridScopedEvidence:
    review_source_sha256 = "d" * 64

    def __init__(self) -> None:
        self.resolved_ids: list[tuple[str, ...]] = []

    def search_evidence_for_facilities(
        self,
        query: str,
        *,
        facility_ids: tuple[str, ...],
        limit_per_facility: int,
        source_types: tuple[str, ...],
    ) -> list[EvidenceHit]:
        return [hit("generic", "alpha", "친절해요", 10)]

    def resolve_evidence_ids(
        self,
        evidence_ids: tuple[str, ...],
    ) -> list[EvidenceHit]:
        self.resolved_ids.append(tuple(evidence_ids))
        if "semantic-decisive" not in evidence_ids:
            return []
        return [hit(
            "semantic-decisive",
            "alpha",
            "아이에게 천천히 설명하고 눈높이에 맞춰 진료했어요",
            11,
        )]


class RejectingHybridScopedEvidence(HybridScopedEvidence):
    def resolve_evidence_ids(
        self,
        evidence_ids: tuple[str, ...],
    ) -> list[EvidenceHit]:
        self.resolved_ids.append(tuple(evidence_ids))
        raise ValueError("every evidence ID must resolve inside scope")


class InvalidResolvedScopedEvidence(HybridScopedEvidence):
    def __init__(self, mode: str) -> None:
        super().__init__()
        self.mode = mode

    def resolve_evidence_ids(
        self,
        evidence_ids: tuple[str, ...],
    ) -> list[EvidenceHit]:
        resolved = super().resolve_evidence_ids(evidence_ids)
        if self.mode == "facility":
            return [replace(item, facility_id="beta") for item in resolved]
        if self.mode == "non_verbatim":
            return [replace(item, is_verbatim=False) for item in resolved]
        if self.mode == "wrong_type":
            return [replace(item, source_type="review_summary") for item in resolved]
        return resolved


class InvalidReferenceSemanticEvidenceSource:
    def __init__(self, mode: str) -> None:
        self.mode = mode

    def retrieve(self, **kwargs: object) -> SemanticReviewOutcome:
        query = kwargs["queries"][0]
        limit = int(kwargs["limit_per_facility"])
        references = [SemanticEvidenceReference(
            query.query_id,
            "semantic-decisive",
            "alpha",
            "bge_m3_sparse",
            1,
            0.8,
        )]
        if self.mode == "missing_context":
            references.append(SemanticEvidenceReference(
                "unknown-query",
                "semantic-decisive",
                "alpha",
                "bge_m3_dense",
                1,
                0.9,
            ))
        elif self.mode == "rank":
            references.append(SemanticEvidenceReference(
                query.query_id,
                "semantic-decisive",
                "alpha",
                "bge_m3_dense",
                limit + 1,
                0.9,
            ))
        return SemanticReviewOutcome(
            "ok", tuple(references), "reviews-v1", "BAAI/bge-m3"
        )


class FakeSemanticEvidenceSource:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def retrieve(self, **kwargs: object) -> SemanticReviewOutcome:
        self.calls.append(kwargs)
        queries = kwargs["queries"]
        references = []
        for query in queries:
            references.extend((
                SemanticEvidenceReference(
                    query.query_id,
                    "semantic-decisive",
                    "alpha",
                    "bge_m3_sparse",
                    1,
                    0.8,
                ),
                SemanticEvidenceReference(
                    query.query_id,
                    "semantic-decisive",
                    "alpha",
                    "bge_m3_dense",
                    1,
                    0.9,
                ),
            ))
        return SemanticReviewOutcome(
            "ok",
            tuple(references),
            "reviews-v1",
            "BAAI/bge-m3",
        )


class FailedSemanticEvidenceSource:
    def retrieve(self, **kwargs: object) -> SemanticReviewOutcome:
        return SemanticReviewOutcome("request_failed")


class EvidenceRetrievalTests(unittest.TestCase):
    def test_local_distinctiveness_suppresses_repetition_without_claiming_truth(self) -> None:
        hits = (
            hit("generic-1", "alpha", "친절해요", 1),
            hit("generic-2", "alpha", "친절해요!", 2),
            hit("generic-3", "beta", "정말 친절해요", 3),
            hit(
                "specific",
                "alpha",
                "아이에게 천천히 설명하고 처방 이유와 복용 순서를 알려줬어요",
                4,
            ),
        )
        profiles = score_local_distinctiveness(hits)

        self.assertGreater(
            profiles["specific"].distinctiveness,
            profiles["generic-1"].distinctiveness,
        )
        self.assertGreaterEqual(profiles["generic-1"].local_cluster_size, 2)
        self.assertTrue(all(
            0.0 <= profile.distinctiveness <= 1.0
            for profile in profiles.values()
        ))
        self.assertFalse(hasattr(profiles["specific"], "truth_score"))

    def test_distinctiveness_cannot_displace_more_relevant_evidence(self) -> None:
        def scored(
            evidence_id: str,
            ordinal: int,
            selection_score: float,
            distinctiveness: float,
        ) -> ScoredEvidence:
            return ScoredEvidence(
                hit=hit(evidence_id, "alpha", evidence_id, ordinal),
                matched_constraint_ids=frozenset({"kind"}),
                roles=frozenset({"support"}),
                lexical_rank=ordinal,
                rerank_rank=None,
                rerank_score=None,
                distinctiveness=distinctiveness,
                local_cluster_id=evidence_id,
                local_cluster_size=1,
                corroboration_count=1,
                selection_score=selection_score,
            )

        evidence = (
            scored("generic-high", 1, 1.0, 0.2),
            scored("generic-second", 2, 0.9, 0.3),
            scored("generic-third", 3, 0.8, 0.4),
            scored("decisive-distinctive", 4, 0.5, 1.0),
        )
        groups = select_evidence_groups(("alpha",), evidence, (), limit=3)

        self.assertNotIn(
            "decisive-distinctive",
            {item.hit.evidence_id for item in groups["alpha"].presented},
        )

    def test_presentation_skips_a_duplicate_cluster_without_looping(self) -> None:
        def scored(
            evidence_id: str,
            ordinal: int,
            selection_score: float,
            distinctiveness: float,
            cluster_id: str,
        ) -> ScoredEvidence:
            return ScoredEvidence(
                hit=hit(evidence_id, "alpha", evidence_id, ordinal),
                matched_constraint_ids=frozenset({"kind"}),
                roles=frozenset({"support"}),
                lexical_rank=ordinal,
                rerank_rank=None,
                rerank_score=None,
                distinctiveness=distinctiveness,
                local_cluster_id=cluster_id,
                local_cluster_size=2 if cluster_id == "generic" else 1,
                corroboration_count=1,
                selection_score=selection_score,
            )

        evidence = (
            scored("generic-distinctive", 1, 1.0, 1.0, "generic"),
            scored("generic-duplicate", 2, 0.9, 0.5, "generic"),
            scored("different-cluster", 3, 0.9, 0.4, "different"),
        )

        groups = select_evidence_groups(("alpha",), evidence, (), limit=3)

        self.assertEqual(
            [item.hit.evidence_id for item in groups["alpha"].presented],
            ["generic-distinctive", "different-cluster", "generic-duplicate"],
        )

    def test_admission_is_fair_across_facility_constraint_cells(self) -> None:
        candidates = (
            EvidenceCellCandidate(
                hit("a-1", "alpha", "친절해요", 1),
                ConstraintCell("alpha", "kind", "support"),
                lexical_rank=1,
                query_language="ko",
            ),
            EvidenceCellCandidate(
                hit("a-2", "alpha", "아주 친절해요", 2),
                ConstraintCell("alpha", "kind", "support"),
                lexical_rank=2,
                query_language="ko",
            ),
            EvidenceCellCandidate(
                hit("b-1", "beta", "아이에게 친절해요", 3),
                ConstraintCell("beta", "kind", "support"),
                lexical_rank=1,
                query_language="ko",
            ),
            EvidenceCellCandidate(
                hit("risk-1", "alpha", "간호사가 불친절해요", 4),
                ConstraintCell("alpha", "nurses", "risk"),
                lexical_rank=1,
                query_language="ko",
            ),
        )
        admitted = allocate_admissions(candidates, budget=3)

        self.assertEqual(
            {item.cell for item in admitted},
            {
                ConstraintCell("alpha", "kind", "support"),
                ConstraintCell("beta", "kind", "support"),
                ConstraintCell("alpha", "nurses", "risk"),
            },
        )
        self.assertEqual(
            len({item.hit.evidence_id for item in admitted}),
            len(admitted),
        )

    def test_admission_uses_fused_rank_before_individual_channel_ranks(self) -> None:
        cell = ConstraintCell("alpha", "kind", "support")
        candidates = (
            EvidenceCellCandidate(
                hit("channel-first", "alpha", "same text", 1),
                cell,
                lexical_rank=1,
                query_language="en",
                fused_rank=2,
            ),
            EvidenceCellCandidate(
                hit("fused-first", "alpha", "same text", 2),
                cell,
                lexical_rank=5,
                query_language="en",
                fused_rank=1,
            ),
        )

        admitted = allocate_admissions(candidates, budget=1)

        self.assertEqual(admitted[0].hit.evidence_id, "fused-first")

    def test_cross_cell_duplicate_does_not_advance_other_cell_queues(self) -> None:
        shared = hit("shared", "alpha", "구순염을 친절하게 설명", 1)
        candidates = (
            EvidenceCellCandidate(
                shared,
                ConstraintCell("alpha", "disease", "disease"),
                lexical_rank=1,
                query_language="ko",
            ),
            EvidenceCellCandidate(
                shared,
                ConstraintCell("alpha", "explanation", "support"),
                lexical_rank=1,
                query_language="ko",
            ),
            EvidenceCellCandidate(
                hit("support-alt", "alpha", "설명을 자세히 해요", 2),
                ConstraintCell("alpha", "explanation", "support"),
                lexical_rank=2,
                query_language="ko",
            ),
            EvidenceCellCandidate(
                hit("beta", "beta", "아이에게 친절해요", 3),
                ConstraintCell("beta", "kind", "support"),
                lexical_rank=1,
                query_language="ko",
            ),
        )

        admitted = allocate_admissions(candidates, budget=3)

        self.assertEqual(
            {item.hit.evidence_id for item in admitted},
            {"shared", "support-alt", "beta"},
        )


    def test_hybrid_rrf_admits_semantic_only_review_with_channel_ranks(self) -> None:
        scoped = HybridScopedEvidence()
        semantic = FakeSemanticEvidenceSource()
        base_rules = rules()
        result = ConstraintEvidenceRetriever(
            semantic_source=semantic,
        ).collect(
            scoped_index=scoped,
            rules=replace(base_rules, evidence=(base_rules.evidence[0],)),
            shortlisted_facility_ids=("alpha",),
            displayed_facility_ids=("alpha",),
        )

        presented = result.by_facility["alpha"].presented
        self.assertEqual(presented[0].hit.evidence_id, "semantic-decisive")
        semantic_events = [
            event for event in result.admissions
            if event.evidence_id == "semantic-decisive"
        ]
        self.assertEqual(len(semantic_events), 2)
        self.assertEqual(
            {event.query_language for event in semantic_events},
            {"en", "ko"},
        )
        self.assertTrue(all(
            event.lexical_rank is None for event in semantic_events
        ))
        self.assertTrue(all(event.sparse_rank == 1 for event in semantic_events))
        self.assertTrue(all(event.dense_rank == 1 for event in semantic_events))
        self.assertTrue(all(event.fused_rank == 1 for event in semantic_events))
        self.assertEqual(result.semantic_status, "ok")
        self.assertEqual(len(semantic.calls), 1)
        self.assertEqual(
            semantic.calls[0]["review_source_sha256"],
            scoped.review_source_sha256,
        )
        self.assertEqual(
            scoped.resolved_ids,
            [("semantic-decisive",)],
        )

    def test_invalid_semantic_reference_rejects_the_whole_outcome(self) -> None:
        cases = (
            (
                "missing query context",
                HybridScopedEvidence(),
                InvalidReferenceSemanticEvidenceSource("missing_context"),
            ),
            (
                "rank beyond requested limit",
                HybridScopedEvidence(),
                InvalidReferenceSemanticEvidenceSource("rank"),
            ),
            (
                "local facility mismatch",
                InvalidResolvedScopedEvidence("facility"),
                InvalidReferenceSemanticEvidenceSource("valid"),
            ),
            (
                "non-verbatim evidence",
                InvalidResolvedScopedEvidence("non_verbatim"),
                InvalidReferenceSemanticEvidenceSource("valid"),
            ),
            (
                "wrong source type",
                InvalidResolvedScopedEvidence("wrong_type"),
                InvalidReferenceSemanticEvidenceSource("valid"),
            ),
        )
        base_rules = rules()
        for label, scoped, semantic in cases:
            with self.subTest(label=label):
                result = ConstraintEvidenceRetriever(
                    semantic_source=semantic,
                ).collect(
                    scoped_index=scoped,
                    rules=replace(
                        base_rules, evidence=(base_rules.evidence[0],)
                    ),
                    shortlisted_facility_ids=("alpha",),
                    displayed_facility_ids=("alpha",),
                )

                self.assertEqual(result.semantic_status, "request_failed")
                self.assertEqual(
                    result.by_facility["alpha"].presented[0].hit.evidence_id,
                    "generic",
                )
                self.assertTrue(all(
                    event.sparse_rank is None and event.dense_rank is None
                    for event in result.admissions
                ))

    def test_resolver_failure_rejects_neural_results_and_keeps_bm25(self) -> None:
        scoped = RejectingHybridScopedEvidence()
        semantic = FakeSemanticEvidenceSource()
        base_rules = rules()
        result = ConstraintEvidenceRetriever(
            semantic_source=semantic,
        ).collect(
            scoped_index=scoped,
            rules=replace(base_rules, evidence=(base_rules.evidence[0],)),
            shortlisted_facility_ids=("alpha",),
            displayed_facility_ids=("alpha",),
        )

        self.assertEqual(result.semantic_status, "request_failed")
        self.assertEqual(
            result.by_facility["alpha"].presented[0].hit.evidence_id,
            "generic",
        )
        self.assertEqual(scoped.resolved_ids, [("semantic-decisive",)])
        self.assertTrue(all(
            event.sparse_rank is None and event.dense_rank is None
            for event in result.admissions
        ))

    def test_semantic_failure_keeps_local_bm25_results(self) -> None:
        scoped = HybridScopedEvidence()
        base_rules = rules()
        result = ConstraintEvidenceRetriever(
            semantic_source=FailedSemanticEvidenceSource(),
        ).collect(
            scoped_index=scoped,
            rules=replace(base_rules, evidence=(base_rules.evidence[0],)),
            shortlisted_facility_ids=("alpha",),
            displayed_facility_ids=("alpha",),
        )

        self.assertEqual(result.semantic_status, "request_failed")
        self.assertEqual(
            result.by_facility["alpha"].presented[0].hit.evidence_id,
            "generic",
        )
        self.assertTrue(all(
            event.sparse_rank is None and event.dense_rank is None
            for event in result.admissions
        ))

    def test_collector_separates_languages_and_warning_evidence(self) -> None:
        scoped = FakeScopedEvidence()
        result = ConstraintEvidenceRetriever(
            policy=EvidenceRecallPolicy(
                shortlist_limit=20,
                lexical_per_cell=5,
                initial_pool_limit=512,
                initial_rerank_budget=224,
                retry_rerank_budget=32,
                presentation_limit=3,
            )
        ).collect(
            scoped_index=scoped,
            rules=rules(),
            shortlisted_facility_ids=("alpha",),
            displayed_facility_ids=("alpha",),
        )

        searched = {query for query, _, _ in scoped.queries}
        self.assertIn("kind with children", searched)
        self.assertIn("아이에게 친절", searched)
        self.assertIn("unfriendly nurses", searched)
        self.assertIn("간호사 불친절", searched)
        self.assertFalse(any("no wait" in query for query in searched))

        groups = result.by_facility["alpha"]
        self.assertIn(
            "decisive",
            {item.hit.evidence_id for item in groups.supporting},
        )
        self.assertIn(
            "decisive",
            {item.hit.evidence_id for item in groups.warnings},
        )
        self.assertIn(
            "decisive",
            {item.hit.evidence_id for item in groups.presented},
        )
        self.assertEqual(result.finish_status, "complete")
        self.assertTrue(all(event.dense_rank is None for event in result.admissions))


if __name__ == "__main__":
    unittest.main()
