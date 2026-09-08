"""Scope-bound hybrid retrieval for the Phase 4 serving path."""

from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter
from typing import Literal, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd

from query_facets import expand_multilingual_retrieval_terms
from review_presentation import useful_review
from search.availability import has_tuesday_evening
from search.contracts import SearchRules
from search.evidence_retrieval import (
    ConstraintEvidenceRetriever,
    EvidenceRecallPolicy,
    EvidenceRecallResult,
    assess_facility_coverage,
    compile_evidence_constraints,
    evidence_payload,
)
from search.indexes.manifest import EXPECTED_DENSE_DIMENSION
from search.indexes.repository import EvidenceHit, FacilityHit, ReadonlyIndex
from search.scope import ScopeSelection
from search.reranker import RerankOutcome
from search.semantic_retriever import SemanticEvidenceSource


LEGACY_INDEX_VERSION = "legacy-dataframe-v1"
STRUCTURED_EVIDENCE_TYPES = (
    "amenity",
    "facility_fact",
    "medical_info",
    "review_highlight",
    "review_summary",
)
COMMENT_EVIDENCE_TYPES = ("verbatim_review",)


@dataclass(frozen=True)
class RRFPolicy:
    version: str = "phase4-weighted-rrf-v2"
    k: float = 60.0
    lexical_weight: float = 1.0
    dense_weight: float = 1.0
    evidence_weight: float = 0.75
    facility_limit: int = 200
    evidence_limit: int = 100
    weak_window: int = 20


@dataclass(frozen=True)
class RetrievalQuery:
    text: str
    max_distance_km: float
    search_mode: str
    specialty_confidence: float
    exact_terms: tuple[str, ...] = ()
    target_language: str = "English"
    manual_mode: str | None = None

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("retrieval query cannot be empty")
        if not 0.0 < self.max_distance_km <= 100.0:
            raise ValueError("retrieval distance must be greater than 0 and at most 100")
        if not 0.0 <= self.specialty_confidence <= 1.0:
            raise ValueError("specialty confidence must be between 0 and 1")


@dataclass(frozen=True)
class FusedCandidate:
    facility_id: str
    score: float
    lexical_rank: int | None
    dense_rank: int | None
    evidence_rank: int | None
    evidence_methods: frozenset[str] = frozenset()

    @property
    def methods(self) -> tuple[str, ...]:
        methods: list[str] = []
        if self.lexical_rank is not None:
            methods.append("facility_bm25")
        if self.dense_rank is not None:
            methods.append("facility_semantic")
        methods.extend(sorted(self.evidence_methods))
        return tuple(methods)


@dataclass(frozen=True)
class RetrievalTelemetry:
    status: str
    termination_reason: str
    index_version: str
    policy_version: str
    rules_hash: str
    scope_digest: str
    candidate_scope_count: int
    channel_hit_counts: Mapping[str, int]
    retry_ran: bool
    retry_reason: str | None
    fallback_reason: str | None
    elapsed_ms: float
    coverage_assessed: bool = True
    coverage_sufficient: bool = False
    finish_status: str = "complete"
    execution_status: Literal["complete", "partial", "failed", "not_run"] = "complete"
    reason_codes: tuple[str, ...] = ()

    def as_private_dict(self) -> dict[str, object]:
        return {
            "retrieval_status": self.status,
            "termination_reason": self.termination_reason,
            "index_version": self.index_version,
            "policy_version": self.policy_version,
            "rules_hash": self.rules_hash,
            "scope_digest": self.scope_digest,
            "candidate_scope_count": self.candidate_scope_count,
            "channel_hit_counts": dict(self.channel_hit_counts),
            "retry_ran": self.retry_ran,
            "retry_reason": self.retry_reason,
            "fallback_reason": self.fallback_reason,
            "elapsed_ms": round(self.elapsed_ms, 3),
            "coverage_assessed": self.coverage_assessed,
            "coverage_sufficient": self.coverage_sufficient,
            "finish_status": self.finish_status,
            "retrieval_execution_status": self.execution_status,
            "retrieval_reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True)
class CandidateRetrievalResult:
    dataframe: pd.DataFrame
    telemetry: RetrievalTelemetry


@dataclass(frozen=True)
class _Attempt:
    lexical: tuple[FacilityHit, ...]
    dense: tuple[FacilityHit, ...]
    evidence: tuple[EvidenceHit, ...]


class EvidenceReranker(Protocol):
    def rerank(
        self,
        query: str,
        hits: Sequence[EvidenceHit],
    ) -> RerankOutcome:
        ...



class CandidateRetrievalAdapter:
    """Hide indexed retrieval, fusion, retry, and legacy rollback behind one call."""

    def __init__(
        self,
        *,
        active_index: ReadonlyIndex | None,
        legacy_pipeline: object | None,
        policy: RRFPolicy = RRFPolicy(),
        evidence_reranker: EvidenceReranker | None = None,
        semantic_evidence_source: SemanticEvidenceSource | None = None,
        evidence_policy: EvidenceRecallPolicy = EvidenceRecallPolicy(),
    ) -> None:
        self._active_index = active_index
        self._legacy_pipeline = legacy_pipeline
        self._policy = policy
        self._evidence_reranker = evidence_reranker
        self._evidence_retriever = ConstraintEvidenceRetriever(
            reranker=evidence_reranker,
            semantic_source=semantic_evidence_source,
            policy=evidence_policy,
        )

    def rank(
        self,
        *,
        scope: ScopeSelection,
        eligible: pd.DataFrame,
        rules: SearchRules,
        query: RetrievalQuery,
    ) -> CandidateRetrievalResult:
        started = perf_counter()
        _validate_complete_scope(scope, eligible)
        if eligible.empty:
            return self._empty_result(scope, eligible, rules, started)
        if self._active_index is None:
            return self._legacy_result(
                scope, eligible, rules, query, started, "index_unavailable"
            )

        try:
            if scope.descriptor.index_version != self._active_index.version:
                raise ValueError("scope_index_version_mismatch")
            embedding = _embed_query(self._legacy_pipeline, query.text)
            allowed = frozenset(scope.facility_ids)
            with self._active_index.within(scope) as scoped:
                scope_evidence = {}
                for evidence_query in (_evidence_queries(rules, query) if rules.evidence else ()):
                    for hit in scoped.search_evidence(
                        evidence_query, limit=100,
                        source_types=("verbatim_review",),
                    ):
                        scope_evidence.setdefault(hit.evidence_id, hit)
                facility_attempt = _Attempt(
                    lexical=tuple(scoped.search_facilities(
                        query.text,
                        limit=self._policy.facility_limit,
                    )),
                    dense=tuple(scoped.search_dense(
                        embedding,
                        limit=self._policy.facility_limit,
                    )),
                    evidence=tuple(scope_evidence.values()),
                )
                _validate_attempt(facility_attempt, allowed)
                candidates = weighted_rrf(facility_attempt, allowed, self._policy)
                preliminary = _rank_dataframe(eligible, candidates, {}, query)
                shortlist = tuple(
                    preliminary["place_id"].astype(str).head(
                        self._evidence_retriever.policy.shortlist_limit
                    )
                )
                displayed = tuple(preliminary["place_id"].astype(str).head(3))
                evidence_result = self._evidence_retriever.collect(
                    scoped_index=scoped,
                    rules=rules,
                    shortlisted_facility_ids=shortlist,
                    displayed_facility_ids=displayed,
                    # Evidence can change facility order after preliminary
                    # fusion. Assess attachments across the shortlist, while
                    # limiting retries to the provisional top three.
                    attachment_facility_ids=shortlist,
                )
                review_source_sha256 = scoped.review_source_sha256
                general_reviews = (
                    scoped.list_original_reviews(facility_ids=shortlist, limit_per_facility=100)
                    if evidence_result.reranker_reason == "no_constraints" else ()
                )

            evidence_hits = tuple(item.hit for item in evidence_result.evidence)
            merged = _Attempt(
                lexical=facility_attempt.lexical,
                dense=facility_attempt.dense,
                evidence=evidence_hits,
            )
            candidates = weighted_rrf(merged, allowed, self._policy)

            hit_by_id = {hit.evidence_id: hit for hit in evidence_hits}
            evidence_methods_by_facility: dict[str, set[str]] = {}
            for event in evidence_result.admissions:
                methods = evidence_methods_by_facility.setdefault(
                    event.facility_id, set()
                )
                if event.lexical_rank is not None:
                    methods.add("review_bm25")
                    hit = hit_by_id.get(event.evidence_id)
                    if hit is not None:
                        methods.add(
                            "multilingual_comment_search"
                            if hit.is_verbatim else "bm25_specific"
                        )
                if event.sparse_rank is not None:
                    methods.add("review_bge_m3_sparse")
                if event.dense_rank is not None:
                    methods.add("review_bge_m3_dense")
            candidates = _attach_evidence_methods(
                candidates, evidence_methods_by_facility
            )
            evidence_by_facility = {
                facility_id: [
                    {**evidence_payload(item), "review_source_sha256": review_source_sha256}
                    for item in groups.presented
                ]
                for facility_id, groups in evidence_result.by_facility.items()
            }
            if evidence_result.reranker_reason == "no_constraints":
                evidence_by_facility.update(_evidence_by_facility([
                    hit for hit in general_reviews if hit.is_verbatim and useful_review(hit.original_text)
                ]))
                for records in evidence_by_facility.values():
                    for record in records:
                        record["review_source_sha256"] = review_source_sha256
            evidence_groups_by_facility = {
                facility_id: {
                    "supporting": [
                        {**evidence_payload(item), "review_source_sha256": review_source_sha256}
                        for item in groups.supporting
                    ],
                    "warnings": [
                        {**evidence_payload(item), "review_source_sha256": review_source_sha256}
                        for item in groups.warnings
                    ],
                    "unverified": list(groups.unverified),
                    "coverage_status": "assessed" if evidence_result.coverage else "not_applicable",
                    "coverage_scope": rules.rules_hash,
                }
                for facility_id, groups in evidence_result.by_facility.items()
            }
            ranked = _rank_dataframe(
                eligible,
                candidates,
                evidence_by_facility,
                query,
            )
            ranked["retrieval_methods"] = [
                list(dict.fromkeys((
                    *methods,
                    *sorted(evidence_methods_by_facility.get(facility_id, ())),
                )))
                for facility_id, methods in zip(
                    ranked["place_id"].astype(str),
                    ranked["retrieval_methods"],
                )
            ]
            constraints = compile_evidence_constraints(rules)
            required_ids = [item.constraint_id for item in constraints if item.required]
            ranked["retrieval_evidence_groups"] = (
                ranked["place_id"].astype(str).map(
                    lambda item: evidence_groups_by_facility.get(
                        item,
                        {
                            "supporting": [], "warnings": [],
                            "unverified": required_ids,
                            "coverage_status": "unassessed" if constraints else "not_applicable",
                            "coverage_scope": rules.rules_hash,
                        },
                    )
                )
            )
            final_candidates = tuple(ranked["place_id"].astype(str).head(5))
            unassessed_ids = tuple(item for item in final_candidates if item not in shortlist)
            if unassessed_ids:
                evidence_result = replace(
                    evidence_result,
                    coverage=(*evidence_result.coverage, *(
                        replace(cell, status="unassessed")
                        for cell in assess_facility_coverage(unassessed_ids, constraints, ())
                    )),
                )
            coverage_sufficient = not any(
                cell.required and cell.status in {"missing", "unassessed"}
                for cell in evidence_result.coverage
                if cell.facility_id in final_candidates
            )
            finish_status = "complete" if coverage_sufficient else "partial_evidence"
            telemetry = RetrievalTelemetry(
                status="complete" if evidence_result.execution_status != "partial" else "incomplete",
                termination_reason="finish_search",
                index_version=self._active_index.version,
                policy_version=(
                    f"{self._policy.version}+"
                    f"{self._evidence_retriever.policy.version}"
                ),
                rules_hash=rules.rules_hash,
                scope_digest=scope.descriptor.scope_digest,
                candidate_scope_count=len(scope.facility_ids),
                channel_hit_counts={
                    "facility_bm25": len(merged.lexical),
                    "facility_semantic": len(merged.dense),
                    "bm25_specific": _evidence_count(
                        merged.evidence, verbatim=False
                    ),
                    "multilingual_comment_search": _evidence_count(
                        merged.evidence, verbatim=True
                    ),
                    "review_bm25": sum(
                        event.lexical_rank is not None
                        for event in evidence_result.admissions
                    ),
                    "review_bge_m3_sparse": sum(
                        event.sparse_rank is not None
                        for event in evidence_result.admissions
                    ),
                    "review_bge_m3_dense": sum(
                        event.dense_rank is not None
                        for event in evidence_result.admissions
                    ),
                },
                retry_ran=evidence_result.retry_ran,
                retry_reason=(
                    "targeted_missing_constraint"
                    if evidence_result.retry_ran
                    else None
                ),
                fallback_reason=None,
                elapsed_ms=(perf_counter() - started) * 1000,
                coverage_sufficient=coverage_sufficient,
                finish_status=finish_status,
                execution_status=(
                    "partial" if evidence_result.execution_status == "partial" else "complete"
                ),
                reason_codes=evidence_result.reason_codes,
            )
            rerank_outcome = RerankOutcome(
                evidence_hits,
                evidence_result.reranker_used,
                evidence_result.reranker_reason,
            )
            _attach_private_metadata(
                ranked,
                telemetry,
                candidates,
                attempt=merged,
                query_terms=_trace_query_terms(rules, query),
                rerank_outcome=rerank_outcome,
                evidence_result=evidence_result,
            )
            scope.assert_contains_only(ranked)
            return CandidateRetrievalResult(ranked, telemetry)
        except Exception as exc:
            return self._legacy_result(
                scope,
                eligible,
                rules,
                query,
                started,
                type(exc).__name__,
            )

    def _empty_result(
        self,
        scope: ScopeSelection,
        eligible: pd.DataFrame,
        rules: SearchRules,
        started: float,
    ) -> CandidateRetrievalResult:
        telemetry = RetrievalTelemetry(
            status="empty_scope",
            termination_reason="empty_scope",
            index_version=scope.descriptor.index_version,
            policy_version=self._policy.version,
            rules_hash=rules.rules_hash,
            scope_digest=scope.descriptor.scope_digest,
            candidate_scope_count=0,
            channel_hit_counts={"lexical": 0, "dense": 0, "evidence": 0},
            retry_ran=False,
            retry_reason=None,
            fallback_reason=None,
            elapsed_ms=(perf_counter() - started) * 1000,
            coverage_assessed=False,
            execution_status="not_run",
        )
        result = eligible.copy()
        _attach_private_metadata(result, telemetry, ())
        return CandidateRetrievalResult(result, telemetry)

    def _legacy_result(
        self,
        scope: ScopeSelection,
        eligible: pd.DataFrame,
        rules: SearchRules,
        query: RetrievalQuery,
        started: float,
        reason: str,
    ) -> CandidateRetrievalResult:
        if self._legacy_pipeline is None:
            ranked = eligible.copy()
            ranked["relevance_rank"] = 9999
        else:
            ranked = self._legacy_pipeline.apply_combined_ranking(
                df=eligible.copy(),
                query_text=query.text,
                max_distance=query.max_distance_km,
                search_mode=query.search_mode,
                n_results=self._policy.facility_limit,
                use_hybrid=True,
                manual_mode=query.manual_mode,
                is_general_search=False,
                specialty_confidence=query.specialty_confidence,
                exact_terms=list(query.exact_terms),
                use_agentic=True,
                target_language=query.target_language,
            )
        scope.assert_contains_only(ranked)
        ranked = scope.restrict_dataframe(ranked)
        telemetry = RetrievalTelemetry(
            status="legacy_fallback",
            termination_reason=reason,
            index_version=scope.descriptor.index_version,
            policy_version=self._policy.version,
            rules_hash=rules.rules_hash,
            scope_digest=scope.descriptor.scope_digest,
            candidate_scope_count=len(scope.facility_ids),
            channel_hit_counts={"lexical": 0, "dense": 0, "evidence": 0},
            retry_ran=False,
            retry_reason=None,
            fallback_reason=reason,
            elapsed_ms=(perf_counter() - started) * 1000,
            coverage_assessed=False,
            finish_status="partial_evidence",
            execution_status="partial" if self._legacy_pipeline is not None else "failed",
            reason_codes=("index_unavailable" if reason == "index_unavailable" else "indexed_retrieval_failed",),
        )
        _attach_private_metadata(
            ranked,
            telemetry,
            tuple(ranked.attrs.get("rag_candidates", ())),
        )
        return CandidateRetrievalResult(ranked, telemetry)


def _search_attempt(
    scoped: object,
    *,
    facility_text: str,
    evidence_queries: Sequence[str],
    embedding: tuple[float, ...] | None,
    policy: RRFPolicy,
) -> _Attempt:
    lexical = tuple(scoped.search_facilities(
        facility_text, limit=policy.facility_limit
    ))
    dense = (
        tuple(scoped.search_dense(embedding, limit=policy.facility_limit))
        if embedding is not None
        else ()
    )
    evidence_by_id: dict[str, EvidenceHit] = {}
    for evidence_text in evidence_queries:
        for source_types in (STRUCTURED_EVIDENCE_TYPES, COMMENT_EVIDENCE_TYPES):
            for hit in scoped.search_evidence(
                evidence_text,
                limit=policy.evidence_limit,
                source_types=source_types,
            ):
                evidence_by_id.setdefault(hit.evidence_id, hit)
    return _Attempt(
        lexical=lexical,
        dense=dense,
        evidence=tuple(evidence_by_id.values()),
    )


def _trace_query_terms(
    rules: SearchRules,
    query: RetrievalQuery,
) -> tuple[str, ...]:
    terms: list[str] = [*query.exact_terms]
    for requirement in rules.evidence:
        terms.extend(requirement.terms_en)
        terms.extend(requirement.terms_ko)
    return tuple(dict.fromkeys(
        " ".join(str(term).split()) for term in terms if str(term).strip()
    ))


def _evidence_queries(
    rules: SearchRules,
    query: RetrievalQuery,
) -> tuple[str, ...]:
    queries: list[str] = []
    for requirement in rules.evidence:
        terms = expand_multilingual_retrieval_terms((
            *requirement.terms_en, *requirement.terms_ko
        ))
        text = " ".join(dict.fromkeys(terms)).strip()
        if text and text not in queries:
            queries.append(text)
    return tuple(queries) or (query.text,)


def _rerank_query(
    rules: SearchRules,
    query: RetrievalQuery,
) -> str:
    requirements: list[str] = []
    for preference in rules.soft:
        requirements.append(f"{preference.polarity}: {preference.concept_id}")
    for requirement in rules.evidence:
        aliases = tuple(dict.fromkeys((
            *requirement.terms_en,
            *requirement.terms_ko,
        )))
        if aliases:
            requirements.append(
                f"{requirement.requirement_id}: {' / '.join(aliases)}"
            )
    if not requirements:
        requirements.extend(query.exact_terms)
    if not requirements:
        return query.text
    return "Find direct evidence relevant to each requirement: " + " | ".join(
        requirements
    )



def _coverage_sufficient(
    rules: SearchRules,
    evidence: Sequence[EvidenceHit],
) -> bool:
    required = tuple(item for item in rules.evidence if item.support_required)
    if not required:
        return True
    for requirement in required:
        if requirement.requirement_id == "hours:tuesday_evening":
            if not any(
                not hit.is_verbatim and has_tuesday_evening(hit.original_text)
                for hit in evidence
            ):
                return False
            continue
        matching_evidence = evidence
        if requirement.source_types == frozenset({"verbatim_review"}):
            matching_evidence = tuple(
                hit for hit in evidence if hit.is_verbatim
            )
        aliases = expand_multilingual_retrieval_terms((
            *requirement.terms_en, *requirement.terms_ko
        ))
        if not aliases or not any(
            alias.casefold() in hit.original_text.casefold()
            for alias in aliases
            for hit in matching_evidence
        ):
            return False
    return True


def _evidence_count(
    evidence: Sequence[EvidenceHit],
    *,
    verbatim: bool,
) -> int:
    return len({
        hit.evidence_id for hit in evidence if hit.is_verbatim is verbatim
    })


def weighted_rrf(
    attempt: _Attempt,
    allowed_facility_ids: frozenset[str],
    policy: RRFPolicy = RRFPolicy(),
) -> tuple[FusedCandidate, ...]:
    """Fuse channel ranks without comparing incompatible raw scores."""
    lexical = _unique_facility_ids(attempt.lexical)
    dense = _unique_facility_ids(attempt.dense)
    evidence = _unique_evidence_facilities(attempt.evidence)
    for facility_id in (*lexical, *dense, *evidence):
        if facility_id not in allowed_facility_ids:
            raise ValueError("retrieval channel returned a facility outside scope")
    rank_maps = {
        "lexical": {value: rank for rank, value in enumerate(lexical, start=1)},
        "dense": {value: rank for rank, value in enumerate(dense, start=1)},
        "evidence": {value: rank for rank, value in enumerate(evidence, start=1)},
    }
    weights = {
        "lexical": policy.lexical_weight,
        "dense": policy.dense_weight,
        "evidence": policy.evidence_weight,
    }
    evidence_methods_by_facility: dict[str, set[str]] = {}
    for hit in attempt.evidence:
        method = (
            "multilingual_comment_search"
            if hit.is_verbatim
            else "bm25_specific"
        )
        evidence_methods_by_facility.setdefault(hit.facility_id, set()).add(method)
    facility_ids = set().union(*(set(values) for values in rank_maps.values()))
    fused: list[FusedCandidate] = []
    for facility_id in facility_ids:
        ranks = {name: values.get(facility_id) for name, values in rank_maps.items()}
        score = sum(
            weights[name] / (policy.k + rank)
            for name, rank in ranks.items()
            if rank is not None
        )
        fused.append(FusedCandidate(
            facility_id=facility_id,
            score=score,
            lexical_rank=ranks["lexical"],
            dense_rank=ranks["dense"],
            evidence_rank=ranks["evidence"],
            evidence_methods=frozenset(
                evidence_methods_by_facility.get(facility_id, ())
            ),
        ))
    return tuple(sorted(
        fused,
        key=lambda item: (
            -item.score,
            min(rank for rank in (
                item.lexical_rank, item.dense_rank, item.evidence_rank
            ) if rank is not None),
            item.facility_id.encode("utf-8"),
        ),
    ))


def _attach_evidence_methods(
    candidates: Sequence[FusedCandidate],
    methods_by_facility: Mapping[str, set[str]],
) -> tuple[FusedCandidate, ...]:
    return tuple(
        replace(
            candidate,
            evidence_methods=frozenset(
                methods_by_facility.get(candidate.facility_id, ())
            ),
        )
        for candidate in candidates
    )


def _embed_query(pipeline: object | None, text: str) -> tuple[float, ...]:
    embedding_function = getattr(pipeline, "embedding_function", None)
    if embedding_function is None:
        raise RuntimeError("query embedding function is unavailable")
    values = np.asarray(embedding_function([text]), dtype=np.float32)
    if values.shape != (1, EXPECTED_DENSE_DIMENSION):
        raise ValueError("query embedding response has the wrong shape")
    vector = values[0]
    if not np.isfinite(vector).all() or float(np.linalg.norm(vector)) <= 0.0:
        raise ValueError("query embedding must contain finite nonzero values")
    return tuple(float(value) for value in vector)


def _validate_complete_scope(scope: ScopeSelection, eligible: pd.DataFrame) -> None:
    if "place_id" not in eligible.columns:
        raise ValueError("eligible dataframe requires place_id")
    ids = tuple(eligible["place_id"].astype(str))
    if len(ids) != len(set(ids)):
        raise ValueError("eligible dataframe contains duplicate facility IDs")
    if set(ids) != set(scope.facility_ids):
        raise ValueError("eligible dataframe must contain the complete scope")


def _validate_attempt(attempt: _Attempt, allowed: frozenset[str]) -> None:
    for hit in (*attempt.lexical, *attempt.dense, *attempt.evidence):
        if hit.facility_id not in allowed:
            raise ValueError("retrieval channel returned a facility outside scope")


def _unique_facility_ids(hits: Sequence[FacilityHit]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(hit.facility_id for hit in hits))


def _unique_evidence_facilities(hits: Sequence[EvidenceHit]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(hit.facility_id for hit in hits))


def _weak_reason(
    attempt: _Attempt,
    rules: SearchRules,
    policy: RRFPolicy,
) -> str | None:
    evidence_ids = _unique_evidence_facilities(attempt.evidence)
    if rules.evidence and not evidence_ids:
        return "missing_required_evidence"
    channels = (
        set(_unique_facility_ids(attempt.lexical)[:policy.weak_window]),
        set(_unique_facility_ids(attempt.dense)[:policy.weak_window]),
        set(evidence_ids[:policy.weak_window]),
    )
    union = set().union(*channels)
    if union and not any(sum(item in channel for channel in channels) >= 2 for item in union):
        return "no_cross_channel_support"
    return None


def _retry_text(rules: SearchRules, query: RetrievalQuery) -> str | None:
    terms: list[str] = [*query.exact_terms]
    terms.extend(preference.concept_id for preference in rules.soft)
    for requirement in rules.evidence:
        terms.extend(requirement.terms_en)
        terms.extend(requirement.terms_ko)
    distinct = list(dict.fromkeys(
        " ".join(str(term).split()) for term in terms if str(term).strip()
    ))
    retry = " ".join(distinct).strip()
    if not retry or retry.casefold() == " ".join(query.text.split()).casefold():
        return None
    return retry


def _variant_fuse(
    primary: tuple[str, ...],
    retry: tuple[str, ...],
    k: float,
) -> tuple[str, ...]:
    if not retry:
        return tuple(dict.fromkeys(primary))
    scores: dict[str, float] = {}
    for values in (primary, retry):
        for rank, facility_id in enumerate(dict.fromkeys(values), start=1):
            scores[facility_id] = scores.get(facility_id, 0.0) + 1.0 / (k + rank)
    return tuple(sorted(scores, key=lambda item: (-scores[item], item.encode("utf-8"))))


def _merge_attempts(
    primary: _Attempt,
    retry: _Attempt | None,
    policy: RRFPolicy,
) -> _Attempt:
    if retry is None:
        return primary
    lexical_ids = _variant_fuse(
        _unique_facility_ids(primary.lexical),
        _unique_facility_ids(retry.lexical),
        policy.k,
    )
    evidence_ids = _variant_fuse(
        _unique_evidence_facilities(primary.evidence),
        _unique_evidence_facilities(retry.evidence),
        policy.k,
    )
    lexical_by_id = {
        hit.facility_id: hit for hit in (*primary.lexical, *retry.lexical)
    }
    evidence_by_id: dict[str, EvidenceHit] = {}
    for hit in (*primary.evidence, *retry.evidence):
        evidence_by_id.setdefault(hit.facility_id, hit)
    return _Attempt(
        lexical=tuple(lexical_by_id[item] for item in lexical_ids),
        dense=primary.dense,
        evidence=tuple(evidence_by_id[item] for item in evidence_ids),
    )


def _evidence_by_facility(
    hits: Sequence[EvidenceHit],
) -> dict[str, list[dict[str, object]]]:
    output: dict[str, list[dict[str, object]]] = {}
    seen: set[str] = set()
    for hit in hits:
        if hit.evidence_id in seen or len(output.get(hit.facility_id, ())) >= 3:
            continue
        seen.add(hit.evidence_id)
        output.setdefault(hit.facility_id, []).append({
            "evidence_id": hit.evidence_id,
            "place_id": hit.facility_id,
            "source_type": hit.source_type,
            "source_field": hit.source_field,
            "source_index": hit.source_index,
            "source_locator": hit.source_locator,
            "text": hit.original_text,
            "language": hit.language_hint,
            "visit_date": hit.visit_date,
            "scraped_at": hit.scraped_at,
            "is_verbatim": hit.is_verbatim,
        })
    return output


def _rank_dataframe(
    eligible: pd.DataFrame,
    candidates: Sequence[FusedCandidate],
    evidence_by_facility: Mapping[str, list[dict[str, object]]],
    query: RetrievalQuery,
) -> pd.DataFrame:
    ranked = eligible.copy()
    candidate_by_id = {item.facility_id: item for item in candidates}
    ranked["retrieval_score"] = ranked["place_id"].astype(str).map(
        lambda item: candidate_by_id[item].score if item in candidate_by_id else 0.0
    )
    max_score = max((item.score for item in candidates), default=0.0)
    ranked["relevance_score"] = (
        ranked["retrieval_score"] / max_score if max_score > 0 else 0.0
    )
    rank_by_id = {
        item.facility_id: rank for rank, item in enumerate(candidates, start=1)
    }
    ranked["relevance_rank"] = ranked["place_id"].astype(str).map(
        lambda item: rank_by_id.get(item, 9999)
    )
    ranked["retrieval_methods"] = ranked["place_id"].astype(str).map(
        lambda item: list(candidate_by_id[item].methods) if item in candidate_by_id else []
    )
    ranked["retrieval_evidence"] = ranked["place_id"].astype(str).map(
        lambda item: list(evidence_by_facility.get(item, ()))
    )
    ranked["retrieval_matched_terms"] = [[] for _ in range(len(ranked))]

    if query.search_mode != "zone" and "distance_km" in ranked.columns:
        distances = pd.to_numeric(ranked["distance_km"], errors="coerce")
        ranked["distance_score"] = (
            1.0 - distances.fillna(query.max_distance_km).clip(
                lower=0.0, upper=query.max_distance_km
            ) / query.max_distance_km
        )
        relevance_weight = _relevance_weight(
            query.max_distance_km, query.specialty_confidence
        )
        ranked["combined_score"] = (
            relevance_weight * ranked["relevance_score"]
            + (1.0 - relevance_weight) * ranked["distance_score"]
        )
        return ranked.sort_values(
            ["combined_score", "relevance_rank", "place_id"],
            ascending=[False, True, True],
            kind="stable",
        )
    ranked["combined_score"] = ranked["relevance_score"]
    return ranked.sort_values(
        ["relevance_rank", "place_id"],
        ascending=[True, True],
        kind="stable",
    )


def _relevance_weight(max_distance_km: float, specialty_confidence: float) -> float:
    if max_distance_km <= 2.0:
        base = 0.70
    elif max_distance_km <= 5.0:
        base = 0.70 + (max_distance_km - 2.0) * 0.05
    elif max_distance_km <= 10.0:
        base = 0.85 + (max_distance_km - 5.0) * 0.02
    else:
        base = min(0.98, 0.95 + (max_distance_km - 10.0) * 0.005)
    multiplier = 0.75 if specialty_confidence < 0.4 else 0.90 if specialty_confidence < 0.7 else 1.0
    return min(0.98, max(0.50, base * multiplier))


def _attach_private_metadata(
    dataframe: pd.DataFrame,
    telemetry: RetrievalTelemetry,
    candidates: Sequence[object],
    *,
    attempt: _Attempt | None = None,
    query_terms: Sequence[str] = (),
    rerank_outcome: RerankOutcome | None = None,
    evidence_result: EvidenceRecallResult | None = None,
) -> None:
    dataframe.attrs["rag_metadata"] = telemetry.as_private_dict()
    if rerank_outcome is not None:
        dataframe.attrs["rag_metadata"].update({
            "reranker_used": rerank_outcome.used,
            "reranker_reason": rerank_outcome.reason,
            "reranker_model": rerank_outcome.model,
            "reranker_candidate_count": len(rerank_outcome.hits),
        })
    if evidence_result is not None:
        dataframe.attrs["rag_metadata"]["retrieval_channel_status"] = {
            "facility_bm25": "ok",
            "facility_semantic": "ok",
            "evidence_bm25": "ok" if evidence_result.execution_status != "not_run" else "not_applicable",
            "review_semantic": evidence_result.semantic_status,
            "evidence_reranker": (
                evidence_result.reranker_reason if evidence_result.reranker_applicable else "not_applicable"
            ),
        }
        dataframe.attrs["rag_metadata"]["evidence_coverage_scope"] = "final_candidates_and_attachment_shortlist"
        dataframe.attrs["rag_metadata"]["stage_timings_ms"] = {
            "evidence_search": evidence_result.search_ms,
            "evidence_reranking": evidence_result.reranking_ms,
        }
        dataframe.attrs["rag_metadata"]["evidence_coverage"] = [
            {
                "facility_id": cell.facility_id,
                "constraint_id": cell.constraint_id,
                "role": cell.role,
                "required": cell.required,
                "status": cell.status,
                "evidence_ids": list(cell.evidence_ids),
            }
            for cell in evidence_result.coverage
        ]
        dataframe.attrs["rag_metadata"]["evidence_admissions"] = [
            {
                "evidence_id": event.evidence_id,
                "facility_id": event.facility_id,
                "constraint_id": event.constraint_id,
                "role": event.role,
                "query_language": event.query_language,
                "lexical_rank": event.lexical_rank,
                "sparse_rank": event.sparse_rank,
                "dense_rank": event.dense_rank,
                "fused_rank": event.fused_rank,
                "admitted_to_gpu": event.admitted_to_gpu,
                "gpu_rank": event.gpu_rank,
                "gpu_score": event.gpu_score,
                "attached": event.attached,
                "rejection_reason": event.rejection_reason,
            }
            for event in evidence_result.admissions
        ]
        dataframe.attrs["rag_metadata"]["retry_ran"] = evidence_result.retry_ran
        dataframe.attrs["rag_metadata"]["semantic_status"] = (
            evidence_result.semantic_status
        )

    counts = telemetry.channel_hit_counts
    trace = [
        {
            "action": "search_facilities",
            "result_count": int(counts.get("facility_bm25", 0)),
        },
        {
            "action": "search_semantic_facilities",
            "result_count": int(counts.get("facility_semantic", 0)),
        },
        {
            "action": "search_indexed_evidence",
            "result_count": int(counts.get("bm25_specific", 0))
            + int(counts.get("review_bm25", 0)),
        },
        {
            "action": "search_multilingual_comments",
            "arguments": {"query_terms": list(query_terms)},
            "result_count": int(counts.get("multilingual_comment_search", 0)),
        },
        {
            "action": "select_comment_evidence",
            "result_count": _evidence_count(
                attempt.evidence if attempt else (), verbatim=True
            ),
        },
        {
            "action": "assess_search_coverage",
            "result_count": 1,
            "coverage_sufficient": telemetry.coverage_sufficient,
        },
    ]
    if rerank_outcome is not None:
        trace.insert(-1, {
            "action": "rerank_evidence",
            "result_count": len(rerank_outcome.hits),
            "used": rerank_outcome.used,
            "reason": rerank_outcome.reason,
            "model": rerank_outcome.model,
        })
    if telemetry.termination_reason == "finish_search":
        trace.append({
            "action": "finish_search",
            "result_count": len(candidates),
        })
    dataframe.attrs["rag_trace"] = trace
    dataframe.attrs["rag_observations"] = [{
        "event": telemetry.termination_reason,
        "status": telemetry.status,
        "finish_status": telemetry.finish_status,
    }]
    dataframe.attrs["rag_candidates"] = [
        {
            "place_id": item.facility_id,
            "retrieval_rank_1based": rank,
            "methods": list(item.methods),
            "lexical_rank": item.lexical_rank,
            "dense_rank": item.dense_rank,
            "evidence_rank": item.evidence_rank,
        }
        for rank, item in enumerate(candidates, start=1)
        if isinstance(item, FusedCandidate)
    ]
    dataframe.attrs["rag_quote_evidence"] = False
