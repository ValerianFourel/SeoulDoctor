from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np
import pandas as pd

from search.contracts import (
    AreaRule,
    EligibleScope,
    EvidenceRequirement,
    HardEligibility,
    RuleProvenance,
    SearchRules,
)
from search.indexes.repository import EvidenceHit, FacilityHit
from search.live_retrieval import (
    CandidateRetrievalAdapter,
    RetrievalQuery,
    _Attempt,
    _evidence_by_facility,
    weighted_rrf,
)
from search.reranker import RerankOutcome
from search.scope import ScopeSelection
from search.semantic_retriever import (
    SemanticEvidenceReference,
    SemanticReviewOutcome,
)


RULES_HASH = "a" * 64


def make_rules(*, evidence: bool = False) -> SearchRules:
    provenance = RuleProvenance("user_explicit", (0, 4), "turn-1")
    requirements = ()
    if evidence:
        requirements = (EvidenceRequirement(
            requirement_id="friendly",
            terms_en=("friendly",),
            terms_ko=("친절",),
            match_mode="all_terms",
            source_types=frozenset({"verbatim_review"}),
            support_required=True,
        ),)
    return SearchRules(
        schema_version="1",
        original_query="friendly 친절",
        language="mixed",
        hard=HardEligibility(
            specialty_ids=frozenset(),
            geography=AreaRule("seoul", "city", "Seoul", provenance),
            prohibited_facility_ids=frozenset(),
            prohibited_taxonomy_ids=frozenset(),
            required_attributes=(),
        ),
        soft=(),
        evidence=requirements,
        provenance={"geography": provenance},
        rules_hash=RULES_HASH,
    )


def make_scope(ids: tuple[str, ...]) -> ScopeSelection:
    return ScopeSelection(
        descriptor=EligibleScope(
            index_version="test-v1",
            rules_hash=RULES_HASH,
            facility_bitmap_ref="memory://scope",
            facility_count=len(ids),
            scope_digest="b" * 64,
        ),
        facility_ids=ids,
        distance_km_by_facility={
            facility_id: float(index + 1) for index, facility_id in enumerate(ids)
        },
    )


def facility_hit(facility_id: str, rank: int, channel: str) -> FacilityHit:
    return FacilityHit(facility_id, rank, 1.0 / rank, channel)


def evidence_hit(facility_id: str, rank: int, text: str = "친절한 review") -> EvidenceHit:
    return EvidenceHit(
        evidence_id=f"e-{rank}-{facility_id}",
        facility_id=facility_id,
        ordinal=rank,
        score=1.0 / rank,
        channel="evidence",
        source_type="verbatim_review",
        source_field="review_text",
        source_index=rank,
        source_locator=f"reviews.parquet#{rank}",
        original_text=text,
        language_hint="ko",
        visit_date="2026-01-01",
        scraped_at="2026-01-02",
        is_verbatim=True,
    )


class FakePipeline:
    def __init__(self) -> None:
        self.embedding_calls: list[list[str]] = []
        self.fallback_ids: list[str] | None = None

    def embedding_function(self, texts: list[str]) -> list[list[float]]:
        self.embedding_calls.append(texts)
        vector = np.zeros(1536, dtype=np.float32)
        vector[0] = 1.0
        return [vector.tolist()]

    def apply_combined_ranking(self, *, df: pd.DataFrame, **_: object) -> pd.DataFrame:
        self.fallback_ids = df["place_id"].astype(str).tolist()
        result = df.copy()
        result["relevance_rank"] = range(1, len(result) + 1)
        return result


class FakeScopedIndex:
    review_source_sha256 = "d" * 64

    def __init__(self, *, escape: bool = False) -> None:
        self.escape = escape
        self.facility_queries: list[str] = []
        self.evidence_queries: list[tuple[str, tuple[str, ...]]] = []
        self.dense_calls = 0

    def __enter__(self) -> "FakeScopedIndex":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def list_original_reviews(self, *, facility_ids, limit_per_facility):
        from dataclasses import replace
        from hashlib import sha256
        return [replace(evidence_hit(facility_id, index + 1, text), evidence_id="review:" +
                        sha256(f"{facility_id}|{index + 1}|{text}".encode()).hexdigest()[:20])
                for facility_id in facility_ids if facility_id != "charlie"
                for index, text in enumerate(("123", "😞", "ㅋㅋ", "친절해요", "접수 직원은 불친절했어요 😞"))]

    def search_facilities(self, query: str, limit: int = 200) -> list[FacilityHit]:
        self.facility_queries.append(query)
        if self.escape:
            return [facility_hit("outside", 1, "lexical")]
        if "친절" in query and query != "primary":
            return [facility_hit("charlie", 1, "lexical")]
        return [facility_hit("alpha", 1, "lexical")]

    def search_dense(self, _: object, limit: int = 200) -> list[FacilityHit]:
        self.dense_calls += 1
        return [facility_hit("bravo", 2, "dense")]

    def search_evidence(
        self,
        query: str,
        limit: int = 100,
        *,
        source_types: tuple[str, ...] | None = None,
    ) -> list[EvidenceHit]:
        accepted = tuple(source_types or ())
        self.evidence_queries.append((query, accepted))
        if "verbatim_review" in accepted and "친절" in query:
            return [evidence_hit("charlie", 1)]
        return []
    def search_evidence_for_facilities(
        self,
        query: str,
        *,
        facility_ids: tuple[str, ...],
        limit_per_facility: int,
        source_types: tuple[str, ...],
    ) -> list[EvidenceHit]:
        hits = self.search_evidence(
            query,
            limit=limit_per_facility,
            source_types=source_types,
        )
        output: list[EvidenceHit] = []
        for facility_id in facility_ids:
            output.extend(
                hit for hit in hits
                if hit.facility_id == facility_id
            )
        return output[:limit_per_facility * len(facility_ids)]


    def resolve_evidence_ids(
        self,
        evidence_ids: tuple[str, ...],
    ) -> list[EvidenceHit]:
        return [
            evidence_hit(
                evidence_id.removeprefix("e-99-"),
                99,
                "semantic friendly review",
            )
            for evidence_id in evidence_ids
            if evidence_id.startswith("e-99-")
        ]


class FakeSemanticEvidenceSource:
    def retrieve(self, **kwargs: object) -> SemanticReviewOutcome:
        references = []
        facility_id = kwargs["facility_ids"][0]
        evidence_id = f"e-99-{facility_id}"
        for query in kwargs["queries"]:
            references.extend((
                SemanticEvidenceReference(
                    query.query_id,
                    evidence_id,
                    facility_id,
                    "bge_m3_sparse",
                    1,
                    0.8,
                ),
                SemanticEvidenceReference(
                    query.query_id,
                    evidence_id,
                    facility_id,
                    "bge_m3_dense",
                    1,
                    0.9,
                ),
            ))
        return SemanticReviewOutcome(
            "ok", tuple(references), "reviews-v1", "BAAI/bge-m3"
        )




class FakeIndex:
    version = "test-v1"

    def __init__(self, scoped: FakeScopedIndex) -> None:
        self.scoped = scoped
        self.seen_scope: ScopeSelection | None = None

    def within(self, scope: ScopeSelection) -> FakeScopedIndex:
        self.seen_scope = scope
        return self.scoped


class MultiEvidenceScopedIndex(FakeScopedIndex):
    def search_evidence(
        self,
        query: str,
        limit: int = 100,
        *,
        source_types: tuple[str, ...] | None = None,
    ) -> list[EvidenceHit]:
        if "verbatim_review" not in tuple(source_types or ()):
            return []
        return [
            evidence_hit("alpha", rank, text)
            for rank, text in enumerate(
                ("친절", "ordinary", "ordinary", "decisive friendly review"),
                start=1,
            )
        ]


class ShortlistEvidenceScopedIndex(FakeScopedIndex):
    def __init__(self, facility_ids: tuple[str, ...]) -> None:
        super().__init__()
        self.facility_ids = facility_ids

    def search_facilities(self, query: str, limit: int = 200) -> list[FacilityHit]:
        self.facility_queries.append(query)
        return [
            facility_hit(facility_id, rank, "lexical")
            for rank, facility_id in enumerate(self.facility_ids, start=1)
        ]

    def search_dense(self, _: object, limit: int = 200) -> list[FacilityHit]:
        self.dense_calls += 1
        return [
            facility_hit(facility_id, rank, "dense")
            for rank, facility_id in enumerate(self.facility_ids, start=1)
        ]

    def search_evidence_for_facilities(
        self,
        query: str,
        *,
        facility_ids: tuple[str, ...],
        limit_per_facility: int,
        source_types: tuple[str, ...],
    ) -> list[EvidenceHit]:
        del query, limit_per_facility, source_types
        return [
            evidence_hit(facility_id, rank, f"friendly review {facility_id}")
            for rank, facility_id in enumerate(facility_ids, start=1)
        ]


class ReverseEvidenceReranker:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def rerank(self, query: str, hits: object) -> RerankOutcome:
        self.queries.append(query)
        values = tuple(hits)
        return RerankOutcome(
            tuple(reversed(values)),
            True,
            "ok",
            "test-model",
        )



class LiveRetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ids = ("alpha", "bravo", "charlie")
        self.scope = make_scope(self.ids)
        self.frame = pd.DataFrame({
            "place_id": self.ids,
            "distance_km": (1.0, 2.0, 3.0),
            "name": ("A", "B", "C"),
        })

    def query(self, text: str = "primary") -> RetrievalQuery:
        return RetrievalQuery(
            text=text,
            max_distance_km=5.0,
            search_mode="distance",
            specialty_confidence=1.0,
            exact_terms=("friendly", "친절"),
            target_language="Korean",
        )

    def test_plain_facility_search_attaches_originals_without_ranking_evidence(self):
        from backend.tests.test_evidence_response import fallback_response
        from models import serialize_results_for_chat
        scoped = FakeScopedIndex()
        adapter = CandidateRetrievalAdapter(active_index=FakeIndex(scoped), legacy_pipeline=FakePipeline())
        result = adapter.rank(scope=self.scope, eligible=self.frame, rules=make_rules(), query=self.query())
        self.assertEqual(result.telemetry.status, "complete")
        cards = result.dataframe.to_dict("records")
        alpha = next(card for card in cards if card["place_id"] == "alpha")
        self.assertEqual([item["text"] for item in alpha["retrieval_evidence"]],
                         ["친절해요", "접수 직원은 불친절했어요 😞"])
        self.assertFalse(alpha["retrieval_evidence_groups"]["supporting"])
        self.assertTrue(all(item["place_id"] == "alpha" for item in alpha["retrieval_evidence"]))
        before_order = result.dataframe["place_id"].tolist()
        from unittest.mock import patch
        with patch.object(scoped, "list_original_reviews", return_value=[]):
            empty = adapter.rank(scope=self.scope, eligible=self.frame, rules=make_rules(), query=self.query())
        self.assertEqual(before_order, empty.dataframe["place_id"].tolist())
        _, presented = fallback_response(cards, {"retrieval_status": "complete"}, "English")
        public = serialize_results_for_chat(presented, include_debug=False)
        original = next(card for card in public if card["place_id"] == "alpha")["retrieval_evidence"][0]
        self.assertEqual(original["text"], "친절해요")
        self.assertEqual(original["review_source_sha256"], "d" * 64)
        self.assertEqual(original["presentation"]["status"], "unavailable")
        self.assertEqual(next(card for card in public if card["place_id"] == "charlie")["retrieval_evidence"], [])

    def test_weighted_rrf_is_deterministic_and_keeps_channel_ranks(self) -> None:
        attempt = _Attempt(
            lexical=(facility_hit("alpha", 1, "lexical"),),
            dense=(
                facility_hit("bravo", 1, "dense"),
                facility_hit("alpha", 2, "dense"),
            ),
            evidence=(evidence_hit("alpha", 1), evidence_hit("alpha", 2)),
        )
        first = weighted_rrf(attempt, frozenset(self.ids))
        second = weighted_rrf(attempt, frozenset(self.ids))
        self.assertEqual(first, second)
        self.assertEqual(first[0].facility_id, "alpha")
        self.assertEqual(first[0].lexical_rank, 1)
        self.assertEqual(first[0].dense_rank, 2)
        self.assertEqual(first[0].evidence_rank, 1)

    def test_weighted_rrf_rejects_scope_escape(self) -> None:
        attempt = _Attempt(
            lexical=(facility_hit("outside", 1, "lexical"),),
            dense=(),
            evidence=(),
        )
        with self.assertRaisesRegex(ValueError, "outside scope"):
            weighted_rrf(attempt, frozenset(self.ids))

    def test_attached_evidence_keeps_facility_ownership(self) -> None:
        grouped = _evidence_by_facility((evidence_hit("alpha", 1),))

        self.assertEqual(grouped["alpha"][0]["place_id"], "alpha")

    def test_gpu_reranker_orders_evidence_before_per_facility_cap(self) -> None:
        scoped = MultiEvidenceScopedIndex()
        reranker = ReverseEvidenceReranker()

        result = CandidateRetrievalAdapter(
            active_index=FakeIndex(scoped),
            legacy_pipeline=FakePipeline(),
            evidence_reranker=reranker,
        ).rank(
            scope=self.scope,
            eligible=self.frame,
            rules=make_rules(evidence=True),
            query=self.query(),
        )

        alpha = result.dataframe.set_index("place_id").loc["alpha"]
        self.assertEqual(
            alpha["retrieval_evidence"][0]["text"],
            "decisive friendly review",
        )
        self.assertEqual(len(alpha["retrieval_evidence"]), 3)
        self.assertEqual(len(reranker.queries), 1)
        self.assertIn("friendly", reranker.queries[0])
        self.assertIn("친절", reranker.queries[0])

    def test_evidence_channel_can_change_facility_order(self) -> None:
        class EvidenceForCharlie(FakeScopedIndex):
            def search_facilities(
                self, query: str, limit: int = 200
            ) -> list[FacilityHit]:
                del query, limit
                return [
                    facility_hit(facility_id, rank, "lexical")
                    for rank, facility_id in enumerate(
                        ("alpha", "bravo", "charlie"), start=1
                    )
                ]

            def search_dense(
                self, _: object, limit: int = 200
            ) -> list[FacilityHit]:
                del limit
                return [
                    facility_hit(facility_id, rank, "dense")
                    for rank, facility_id in enumerate(
                        ("alpha", "bravo", "charlie"), start=1
                    )
                ]

            def search_evidence_for_facilities(
                self,
                query: str,
                *,
                facility_ids: tuple[str, ...],
                limit_per_facility: int,
                source_types: tuple[str, ...],
            ) -> list[EvidenceHit]:
                del query, limit_per_facility, source_types
                return (
                    [evidence_hit("charlie", 1)]
                    if "charlie" in facility_ids
                    else []
                )

        result = CandidateRetrievalAdapter(
            active_index=FakeIndex(EvidenceForCharlie()),
            legacy_pipeline=FakePipeline(),
        ).rank(
            scope=self.scope,
            eligible=self.frame,
            rules=make_rules(evidence=True),
            query=replace(self.query(), search_mode="zone"),
        )

        self.assertEqual(result.dataframe.iloc[0]["place_id"], "charlie")
        charlie = next(
            item
            for item in result.dataframe.attrs["rag_candidates"]
            if item["place_id"] == "charlie"
        )
        self.assertEqual(charlie["evidence_rank"], 1)


    def test_evidence_is_attached_across_the_frozen_shortlist(self) -> None:
        facility_ids = ("alpha", "bravo", "charlie", "delta", "echo")
        frame = pd.DataFrame({
            "place_id": facility_ids,
            "distance_km": tuple(float(index) for index in range(1, 6)),
            "name": tuple(facility_id.title() for facility_id in facility_ids),
        })
        result = CandidateRetrievalAdapter(
            active_index=FakeIndex(ShortlistEvidenceScopedIndex(facility_ids)),
            legacy_pipeline=FakePipeline(),
        ).rank(
            scope=make_scope(facility_ids),
            eligible=frame,
            rules=make_rules(evidence=True),
            query=self.query(),
        )

        rows = result.dataframe.set_index("place_id")
        for facility_id in facility_ids:
            with self.subTest(facility_id=facility_id):
                self.assertTrue(rows.loc[facility_id, "retrieval_evidence"])



    def test_weak_result_retries_once_and_reuses_dense_embedding(self) -> None:
        scoped = FakeScopedIndex()
        pipeline = FakePipeline()
        result = CandidateRetrievalAdapter(
            active_index=FakeIndex(scoped),
            legacy_pipeline=pipeline,
        ).rank(
            scope=self.scope,
            eligible=self.frame,
            rules=make_rules(evidence=True),
            query=self.query(),
        )
        self.assertEqual(result.telemetry.status, "complete")
        self.assertEqual(result.telemetry.execution_status, "complete")
        self.assertEqual(result.telemetry.finish_status, "partial_evidence")
        self.assertEqual(result.telemetry.termination_reason, "finish_search")
        self.assertTrue(result.telemetry.retry_ran)
        self.assertEqual(len(scoped.facility_queries), 1)
        self.assertEqual(len(scoped.evidence_queries), 7)
        self.assertEqual(scoped.dense_calls, 1)
        self.assertEqual(len(pipeline.embedding_calls), 1)
        self.assertEqual(
            set(result.dataframe["place_id"].astype(str)), set(self.ids)
        )
        charlie = result.dataframe.set_index("place_id").loc["charlie"]
        self.assertIn("multilingual_comment_search", charlie["retrieval_methods"])
        self.assertEqual(charlie["retrieval_evidence"][0]["text"], "친절한 review")
        actions = {
            item["action"] for item in result.dataframe.attrs["rag_trace"]
        }
        self.assertTrue({
            "search_facilities",
            "search_indexed_evidence",
            "search_multilingual_comments",
            "select_comment_evidence",
            "assess_search_coverage",
            "finish_search",
        }.issubset(actions))

    def test_review_methods_and_counts_follow_actual_admission_channels(self) -> None:
        result = CandidateRetrievalAdapter(
            active_index=FakeIndex(FakeScopedIndex()),
            legacy_pipeline=FakePipeline(),
            semantic_evidence_source=FakeSemanticEvidenceSource(),
        ).rank(
            scope=self.scope,
            eligible=self.frame,
            rules=make_rules(evidence=True),
            query=self.query(),
        )

        rows = result.dataframe.set_index("place_id")
        self.assertIn("review_bm25", rows.loc["charlie", "retrieval_methods"])
        self.assertIn(
            "review_bge_m3_sparse",
            rows.loc["alpha", "retrieval_methods"],
        )
        self.assertIn(
            "review_bge_m3_dense",
            rows.loc["alpha", "retrieval_methods"],
        )
        counts = result.telemetry.channel_hit_counts
        self.assertGreater(counts["review_bm25"], 0)
        self.assertGreater(counts["review_bge_m3_sparse"], 0)
        self.assertGreater(counts["review_bge_m3_dense"], 0)

    def test_english_and_korean_queries_use_all_scoped_channels(self) -> None:
        for text in ("friendly clinic", "친절한 병원"):
            with self.subTest(text=text):
                scoped = FakeScopedIndex()
                pipeline = FakePipeline()
                result = CandidateRetrievalAdapter(
                    active_index=FakeIndex(scoped),
                    legacy_pipeline=pipeline,
                ).rank(
                    scope=self.scope,
                    eligible=self.frame,
                    rules=make_rules(),
                    query=replace(self.query(text), exact_terms=()),
                )
                self.assertEqual(result.telemetry.status, "complete")
                self.assertEqual(scoped.dense_calls, 1)
                self.assertEqual(len(scoped.facility_queries), 1)
                self.assertEqual(len(scoped.evidence_queries), 0)

    def test_scope_escape_falls_back_on_the_same_complete_scope(self) -> None:
        scoped = FakeScopedIndex(escape=True)
        pipeline = FakePipeline()
        result = CandidateRetrievalAdapter(
            active_index=FakeIndex(scoped),
            legacy_pipeline=pipeline,
        ).rank(
            scope=self.scope,
            eligible=self.frame,
            rules=make_rules(),
            query=replace(self.query(), exact_terms=()),
        )
        self.assertEqual(result.telemetry.status, "legacy_fallback")
        self.assertEqual(result.telemetry.fallback_reason, "ValueError")
        self.assertEqual(pipeline.fallback_ids, list(self.ids))
        self.assertEqual(set(result.dataframe["place_id"]), set(self.ids))

    def test_incomplete_dataframe_is_rejected_before_any_retrieval(self) -> None:
        scoped = FakeScopedIndex()
        pipeline = FakePipeline()
        with self.assertRaisesRegex(ValueError, "complete scope"):
            CandidateRetrievalAdapter(
                active_index=FakeIndex(scoped),
                legacy_pipeline=pipeline,
            ).rank(
                scope=self.scope,
                eligible=self.frame.iloc[:2],
                rules=make_rules(),
                query=self.query(),
            )
        self.assertEqual(pipeline.embedding_calls, [])


if __name__ == "__main__":
    unittest.main()
