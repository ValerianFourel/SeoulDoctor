"""Facility-scoped, constraint-aware review recall and evidence selection."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from hashlib import sha256
import logging
import math
import re
from time import perf_counter
from typing import Literal, Mapping, Protocol, Sequence
import unicodedata

from query_facets import expand_multilingual_retrieval_terms
from search.contracts import EvidenceRole, SearchRules, validate_evidence_source_types
from search.indexes.repository import EvidenceHit
from search.reranker import RerankOutcome
from search.rules import SPECIALTY_IDS
from search.semantic_retriever import (
    SemanticCellQuery,
    SemanticEvidenceSource,
    SemanticReviewOutcome,
)


logger = logging.getLogger(__name__)


FinishStatus = Literal["complete", "partial_evidence"]
CoverageStatus = Literal["matched", "missing", "unassessed"]


@dataclass(frozen=True)
class EvidenceConstraint:
    constraint_id: str
    role: EvidenceRole
    terms_en: tuple[str, ...]
    terms_ko: tuple[str, ...]
    source_types: tuple[str, ...]
    required: bool
    priority: int

    def __post_init__(self) -> None:
        if not self.source_types:
            raise ValueError("evidence constraint requires source types")
        validate_evidence_source_types(self.source_types)


@dataclass(frozen=True)
class ConstraintCell:
    facility_id: str
    constraint_id: str
    role: EvidenceRole


@dataclass(frozen=True)
class NoveltyProfile:
    local_cluster_id: str
    local_cluster_size: int
    distinctiveness: float
    concreteness: float


@dataclass(frozen=True)
class EvidenceCellCandidate:
    hit: EvidenceHit
    cell: ConstraintCell
    lexical_rank: int | None
    query_language: Literal["en", "ko"]
    sparse_rank: int | None = None
    dense_rank: int | None = None
    fused_rank: int | None = None
    fused_score: float = 0.0
    local_cluster_id: str = ""
    local_cluster_size: int = 1
    distinctiveness: float = 0.0
    concreteness: float = 0.0


@dataclass(frozen=True)
class ScoredEvidence:
    hit: EvidenceHit
    matched_constraint_ids: frozenset[str]
    roles: frozenset[EvidenceRole]
    lexical_rank: int | None
    rerank_rank: int | None
    rerank_score: float | None
    distinctiveness: float
    local_cluster_id: str
    local_cluster_size: int
    corroboration_count: int
    selection_score: float


@dataclass(frozen=True)
class CoverageCell:
    facility_id: str
    constraint_id: str
    role: EvidenceRole
    required: bool
    status: CoverageStatus
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class AdmissionEvent:
    evidence_id: str
    facility_id: str
    constraint_id: str
    role: EvidenceRole
    query_language: str
    lexical_rank: int | None
    dense_rank: int | None
    fused_rank: int | None
    admitted_to_gpu: bool
    gpu_rank: int | None
    gpu_score: float | None
    attached: bool
    rejection_reason: str | None
    sparse_rank: int | None = None


@dataclass(frozen=True)
class EvidenceGroups:
    supporting: tuple[ScoredEvidence, ...]
    warnings: tuple[ScoredEvidence, ...]
    unverified: tuple[str, ...]
    presented: tuple[ScoredEvidence, ...]


@dataclass(frozen=True)
class EvidenceRecallResult:
    by_facility: Mapping[str, EvidenceGroups]
    evidence: tuple[ScoredEvidence, ...]
    coverage: tuple[CoverageCell, ...]
    admissions: tuple[AdmissionEvent, ...]
    finish_status: FinishStatus
    reranker_used: bool
    reranker_reason: str
    retry_ran: bool
    semantic_status: str = "disabled"
    search_ms: float = 0.0
    reranking_ms: float = 0.0
    execution_status: Literal["complete", "partial", "not_run"] = "complete"
    reason_codes: tuple[str, ...] = ()
    semantic_applicable: bool = False
    reranker_applicable: bool = False


@dataclass(frozen=True)
class _CandidateSearchBatch:
    candidates: tuple[EvidenceCellCandidate, ...]
    semantic_status: str


@dataclass(frozen=True)
class EvidenceRecallPolicy:
    version: str = "facility-constraint-cells-v6"
    shortlist_limit: int = 20
    lexical_per_cell: int = 5
    retry_lexical_per_cell: int = 20
    initial_pool_limit: int = 512
    initial_rerank_budget: int = 224
    retry_rerank_budget: int = 32
    presentation_limit: int = 3
    distinctiveness_weight: float = 0.10
    redundancy_penalty: float = 0.15
    hybrid_rrf_k: float = 60.0
    require_remote_services: bool = False

    def __post_init__(self) -> None:
        if not 1 <= self.shortlist_limit <= 50:
            raise ValueError("shortlist_limit must be between 1 and 50")
        if not 1 <= self.lexical_per_cell <= 100:
            raise ValueError("lexical_per_cell must be between 1 and 100")
        if self.retry_lexical_per_cell < self.lexical_per_cell:
            raise ValueError("retry quota cannot be smaller than the initial quota")
        if self.initial_pool_limit < self.initial_rerank_budget:
            raise ValueError("CPU pool cannot be smaller than the GPU budget")
        if self.initial_rerank_budget + self.retry_rerank_budget > 256:
            raise ValueError("initial and retry GPU budgets cannot exceed 256")
        if not 1 <= self.presentation_limit <= 10:
            raise ValueError("presentation_limit must be between 1 and 10")
        if self.hybrid_rrf_k <= 0.0:
            raise ValueError("hybrid_rrf_k must be greater than zero")


class ScopedEvidenceSearch(Protocol):
    @property
    def review_source_sha256(self) -> str:
        ...

    def search_evidence_for_facilities(
        self,
        query: str,
        *,
        facility_ids: Sequence[str],
        limit_per_facility: int,
        source_types: Sequence[str],
    ) -> Sequence[EvidenceHit]:
        ...

    def resolve_evidence_ids(
        self,
        evidence_ids: Sequence[str],
    ) -> Sequence[EvidenceHit]:
        ...


class EvidenceReranker(Protocol):
    def rerank(
        self,
        query: str,
        hits: Sequence[EvidenceHit],
    ) -> RerankOutcome:
        ...


def _execution_reasons(
    semantic_statuses: Sequence[str],
    reranker_statuses: Sequence[str],
    *,
    require_remote: bool,
    semantic_applicable: bool,
    reranker_applicable: bool,
) -> tuple[str, ...]:
    reasons: list[str] = []
    for channel, applicable, statuses in (
        ("semantic", semantic_applicable, semantic_statuses),
        ("reranker", reranker_applicable, reranker_statuses),
    ):
        if not applicable:
            continue
        for status in statuses:
            if status in {"ok", "not_applicable", "no_candidates"}:
                continue
            if status == "disabled":
                if require_remote:
                    reasons.append(f"{channel}_unavailable")
            elif status in {"request_failed", "invalid_response", "release_mismatch", "service_expired"}:
                reasons.append(f"{channel}_{status}")
            else:
                reasons.append(f"{channel}_failed")
    return tuple(dict.fromkeys(reasons))


def compile_evidence_constraints(
    rules: SearchRules,
    *,
    review_query: str | None = None,
) -> tuple[EvidenceConstraint, ...]:
    constraints: list[EvidenceConstraint] = []
    seen: set[str] = set()
    for index, requirement in enumerate(rules.evidence):
        if requirement.requirement_id in seen:
            continue
        seen.add(requirement.requirement_id)
        constraints.append(EvidenceConstraint(
            constraint_id=requirement.requirement_id,
            role=requirement.evidence_role,
            terms_en=_clean_terms(requirement.terms_en),
            terms_ko=_clean_terms(requirement.terms_ko),
            source_types=tuple(sorted(requirement.source_types)),
            required=requirement.support_required,
            priority=index,
        ))

    for preference in rules.soft:
        constraint_id = f"preference:{preference.polarity}:{preference.concept_id}"
        if constraint_id in seen:
            continue
        expanded = tuple(expand_multilingual_retrieval_terms((
            preference.concept_id.replace("_", " "),
        )))
        terms_en = tuple(term for term in expanded if not _contains_hangul(term))
        terms_ko = tuple(term for term in expanded if _contains_hangul(term))
        constraints.append(EvidenceConstraint(
            constraint_id=constraint_id,
            role="risk" if preference.polarity == "negative" else "support",
            terms_en=_clean_terms(terms_en),
            terms_ko=_clean_terms(terms_ko),
            source_types=("verbatim_review",),
            required=True,
            priority=len(constraints),
        ))
        seen.add(constraint_id)
    if not constraints:
        terms = expand_multilingual_retrieval_terms((
            review_query or rules.original_query,
            *(identity.replace("_", " ") for identity in sorted(rules.hard.specialty_ids)),
            *(name for name, identity in SPECIALTY_IDS.items()
              if identity in rules.hard.specialty_ids),
        ))
        constraints.append(EvidenceConstraint(
            constraint_id="review:visit_context",
            role="disease",
            terms_en=_clean_terms(tuple(term for term in terms if not _contains_hangul(term))),
            terms_ko=_clean_terms(tuple(term for term in terms if _contains_hangul(term))),
            source_types=("verbatim_review",),
            required=False,
            priority=0,
        ))
    return tuple(constraints)


def rerank_relevance_query(constraints: Sequence[EvidenceConstraint]) -> str:
    english = sum(bool(item.terms_en) for item in constraints)
    korean = sum(bool(item.terms_ko) for item in constraints)
    prefer_english = english >= korean
    phrases = []
    for item in constraints:
        preferred, fallback = (
            (item.terms_en, item.terms_ko) if prefer_english
            else (item.terms_ko, item.terms_en)
        )
        terms = preferred or fallback
        if terms:
            phrases.append(terms[0])
    phrases = list(dict.fromkeys(phrases))
    if not phrases:
        return ""
    if prefer_english:
        return "Patient reviews about " + " and ".join(phrases) + "."
    return ", ".join(phrases) + "에 관한 환자 후기."


def score_local_distinctiveness(
    hits: Sequence[EvidenceHit],
) -> Mapping[str, NoveltyProfile]:
    unique = tuple({hit.evidence_id: hit for hit in hits}.values())
    normalized = {
        hit.evidence_id: _normalize_review(hit.original_text)
        for hit in unique
    }
    parent = {hit.evidence_id: hit.evidence_id for hit in unique}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    exact_groups: dict[str, list[str]] = defaultdict(list)
    for hit in unique:
        exact_groups[normalized[hit.evidence_id]].append(hit.evidence_id)
    for evidence_ids in exact_groups.values():
        for evidence_id in evidence_ids[1:]:
            union(evidence_ids[0], evidence_id)

    # Comparing every review with every other review made common-query pools
    # quadratic and could hold a T4-backed request open for minutes. Near
    # duplicates with Jaccard >= 0.82 almost always share a stable prefix or
    # suffix. Use those as deterministic blocking keys, then compare at most
    # 64 prior representatives per normalized review.
    representatives = [
        evidence_ids[0]
        for text, evidence_ids in exact_groups.items()
        if text
    ]
    grams = {
        evidence_id: _character_ngrams(normalized[evidence_id])
        for evidence_id in representatives
    }
    blocks: dict[tuple[str, int, str], list[str]] = defaultdict(list)
    for evidence_id in representatives:
        text = normalized[evidence_id]
        band = len(text) // 16
        candidates: set[str] = set()
        for neighbor in (band - 1, band, band + 1):
            candidates.update(blocks.get(("prefix", neighbor, text[:8]), ()))
            candidates.update(blocks.get(("suffix", neighbor, text[-8:]), ()))
        for other_id in sorted(candidates)[:64]:
            if _jaccard(grams[evidence_id], grams[other_id]) >= 0.82:
                union(evidence_id, other_id)
        blocks[("prefix", band, text[:8])].append(evidence_id)
        blocks[("suffix", band, text[-8:])].append(evidence_id)

    members: dict[str, list[str]] = defaultdict(list)
    for evidence_id in parent:
        members[find(evidence_id)].append(evidence_id)

    output: dict[str, NoveltyProfile] = {}
    for hit in unique:
        cluster_members = members[find(hit.evidence_id)]
        cluster_key = "|".join(sorted(cluster_members))
        cluster_id = "local:" + sha256(cluster_key.encode("utf-8")).hexdigest()[:16]
        concreteness = _concreteness(hit.original_text)
        rarity = 1.0 / math.sqrt(len(cluster_members))
        distinctiveness = min(1.0, max(0.0, 0.70 * rarity + 0.30 * concreteness))
        output[hit.evidence_id] = NoveltyProfile(
            local_cluster_id=cluster_id,
            local_cluster_size=len(cluster_members),
            distinctiveness=distinctiveness,
            concreteness=concreteness,
        )
    return output


def allocate_admissions(
    candidates: Sequence[EvidenceCellCandidate],
    *,
    budget: int,
) -> tuple[EvidenceCellCandidate, ...]:
    if budget < 0:
        raise ValueError("budget cannot be negative")
    profiles = score_local_distinctiveness(tuple(item.hit for item in candidates))
    by_cell: dict[ConstraintCell, dict[str, EvidenceCellCandidate]] = defaultdict(dict)
    for item in candidates:
        previous = by_cell[item.cell].get(item.hit.evidence_id)
        if previous is None or _candidate_rank(item) < _candidate_rank(previous):
            profile = profiles[item.hit.evidence_id]
            by_cell[item.cell][item.hit.evidence_id] = replace(
                item,
                local_cluster_id=profile.local_cluster_id,
                local_cluster_size=profile.local_cluster_size,
                distinctiveness=profile.distinctiveness,
                concreteness=profile.concreteness,
            )

    role_order = {"disease": 0, "support": 1, "risk": 2}
    cells = sorted(
        by_cell,
        key=lambda cell: (
            role_order[cell.role],
            cell.facility_id.encode("utf-8"),
            cell.constraint_id,
        ),
    )
    queues = {
        cell: sorted(
            values.values(),
            key=lambda item: (
                (_candidate_rank(item) - 1) // 3,
                -item.distinctiveness,
                _candidate_rank(item),
                item.hit.ordinal,
                item.hit.evidence_id,
            ),
        )
        for cell, values in by_cell.items()
    }

    admitted: list[EvidenceCellCandidate] = []
    seen_ids: set[str] = set()
    positions = {cell: 0 for cell in cells}
    while len(admitted) < budget:
        progressed = False
        for cell in cells:
            queue = queues[cell]
            position = positions[cell]
            while position < len(queue) and queue[position].hit.evidence_id in seen_ids:
                position += 1
            positions[cell] = position
            if position < len(queue):
                item = queue[position]
                positions[cell] = position + 1
                admitted.append(item)
                seen_ids.add(item.hit.evidence_id)
                progressed = True
                if len(admitted) >= budget:
                    break
        if not progressed:
            break
    return tuple(admitted)


class ConstraintEvidenceRetriever:
    """Own facility-cell recall, reranking, retry, coverage, and presentation."""

    def __init__(
        self,
        *,
        reranker: EvidenceReranker | None = None,
        semantic_source: SemanticEvidenceSource | None = None,
        policy: EvidenceRecallPolicy = EvidenceRecallPolicy(),
    ) -> None:
        self._reranker = reranker
        self._semantic_source = semantic_source
        self.policy = policy

    def collect(
        self,
        *,
        scoped_index: ScopedEvidenceSearch,
        rules: SearchRules,
        shortlisted_facility_ids: Sequence[str],
        displayed_facility_ids: Sequence[str],
        attachment_facility_ids: Sequence[str] | None = None,
        review_query: str | None = None,
    ) -> EvidenceRecallResult:
        shortlist = tuple(dict.fromkeys(shortlisted_facility_ids))[
            :self.policy.shortlist_limit
        ]
        displayed = tuple(dict.fromkeys(displayed_facility_ids))
        attachment_facilities = (
            displayed
            if attachment_facility_ids is None
            else tuple(dict.fromkeys(attachment_facility_ids))
        )
        if not set(displayed).issubset(shortlist):
            raise ValueError("displayed facilities must be inside the frozen shortlist")
        if not set(attachment_facilities).issubset(shortlist):
            raise ValueError("attachment facilities must be inside the frozen shortlist")

        constraints = compile_evidence_constraints(rules, review_query=review_query)

        search_started = perf_counter()
        first_search = self._search(
            scoped_index,
            constraints,
            shortlist,
            limit=self.policy.lexical_per_cell,
        )
        search_ms = (perf_counter() - search_started) * 1000
        first_candidates = first_search.candidates
        semantic_statuses = [first_search.semantic_status]
        pool = allocate_admissions(
            first_candidates,
            budget=self.policy.initial_pool_limit,
        )
        admitted = allocate_admissions(
            pool,
            budget=self.policy.initial_rerank_budget,
        )
        reranking_started = perf_counter()
        scored, reranker_used, reranker_reason = self._score(
            admitted,
            first_candidates,
            constraints,
        )
        reranking_ms = (perf_counter() - reranking_started) * 1000
        coverage = assess_facility_coverage(displayed, constraints, scored)

        retry_ran = False
        missing = tuple(cell for cell in coverage if cell.required and cell.status == "missing")
        all_candidates = list(first_candidates)
        if missing and self.policy.retry_rerank_budget:
            retry_ran = True
            retry_constraints = {
                item.constraint_id: item for item in constraints
                if item.constraint_id in {cell.constraint_id for cell in missing}
            }
            retry_candidates: list[EvidenceCellCandidate] = []
            for cell in missing:
                constraint = retry_constraints[cell.constraint_id]
                search_started = perf_counter()
                retry_search = self._search(
                    scoped_index,
                    (constraint,),
                    (cell.facility_id,),
                    limit=self.policy.retry_lexical_per_cell,
                    allow_semantic=all(status == "ok" for status in semantic_statuses),
                )
                search_ms += (perf_counter() - search_started) * 1000
                retry_candidates.extend(retry_search.candidates)
                if retry_search.semantic_status != "not_retried":
                    semantic_statuses.append(retry_search.semantic_status)
            already_admitted = {item.hit.evidence_id for item in admitted}
            retry_candidates = [
                item for item in retry_candidates
                if item.hit.evidence_id not in already_admitted
            ]
            retry_admitted = allocate_admissions(
                retry_candidates,
                budget=self.policy.retry_rerank_budget,
            )
            reranking_started = perf_counter()
            retry_scored, retry_used, retry_reason = self._score(
                retry_admitted,
                retry_candidates,
                constraints,
            )
            reranking_ms += (perf_counter() - reranking_started) * 1000
            all_candidates.extend(retry_candidates)
            scored = _merge_scored(scored, retry_scored)
            reranker_used = reranker_used or retry_used
            if retry_reason != "no_candidates":
                reranker_reason = ",".join(dict.fromkeys((
                    *reranker_reason.split(","), *retry_reason.split(","),
                )))
            admitted = (*admitted, *retry_admitted)

        # Final order can differ from the provisional retry set. Assess every
        # attachment candidate without expanding the bounded retry allowance.
        coverage = assess_facility_coverage(attachment_facilities, constraints, scored)
        by_facility = select_evidence_groups(
            attachment_facilities,
            scored,
            coverage,
            limit=self.policy.presentation_limit,
        )
        attached_ids = {
            item.hit.evidence_id
            for groups in by_facility.values()
            for item in groups.presented
        }
        admissions = _admission_events(
            all_candidates,
            scored,
            admitted,
            attached_ids,
        )
        semantic_applicable = any(
            "verbatim_review" in item.source_types and (item.terms_en or item.terms_ko)
            for item in constraints
        )
        reranker_applicable = bool(admitted)
        reason_codes = _execution_reasons(
            semantic_statuses,
            reranker_reason.split(","),
            require_remote=self.policy.require_remote_services,
            semantic_applicable=semantic_applicable,
            reranker_applicable=reranker_applicable,
        )
        finish_status: FinishStatus = (
            "complete"
            if not any(cell.required and cell.status == "missing" for cell in coverage)
            else "partial_evidence"
        )
        return EvidenceRecallResult(
            by_facility=by_facility,
            evidence=scored,
            coverage=coverage,
            admissions=admissions,
            finish_status=finish_status,
            reranker_used=reranker_used,
            reranker_reason=reranker_reason,
            retry_ran=retry_ran,
            semantic_status=",".join(dict.fromkeys(semantic_statuses)),
            search_ms=search_ms,
            reranking_ms=reranking_ms,
            execution_status="partial" if reason_codes else "complete",
            reason_codes=reason_codes,
            semantic_applicable=semantic_applicable,
            reranker_applicable=reranker_applicable,
        )

    def _search(
        self,
        scoped_index: ScopedEvidenceSearch,
        constraints: Sequence[EvidenceConstraint],
        facility_ids: Sequence[str],
        *,
        limit: int,
        allow_semantic: bool = True,
    ) -> _CandidateSearchBatch:
        search_started = perf_counter()
        candidate_by_key: dict[
            tuple[str, ConstraintCell, Literal["en", "ko"]],
            EvidenceCellCandidate,
        ] = {}
        query_context: dict[
            str,
            tuple[EvidenceConstraint, Literal["en", "ko"]],
        ] = {}
        semantic_queries: list[SemanticCellQuery] = []
        requested_facilities = tuple(facility_ids)
        lexical_hits: dict[
            tuple[str, tuple[str, ...], int, tuple[str, ...]], tuple[EvidenceHit, ...]
        ] = {}

        def upsert(
            hit: EvidenceHit,
            cell: ConstraintCell,
            language: Literal["en", "ko"],
            *,
            lexical_rank: int | None = None,
            sparse_rank: int | None = None,
            dense_rank: int | None = None,
        ) -> None:
            key = (hit.evidence_id, cell, language)
            previous = candidate_by_key.get(key)
            if previous is None:
                candidate_by_key[key] = EvidenceCellCandidate(
                    hit=hit,
                    cell=cell,
                    lexical_rank=lexical_rank,
                    query_language=language,
                    sparse_rank=sparse_rank,
                    dense_rank=dense_rank,
                )
                return
            candidate_by_key[key] = replace(
                previous,
                lexical_rank=_minimum_rank(previous.lexical_rank, lexical_rank),
                sparse_rank=_minimum_rank(previous.sparse_rank, sparse_rank),
                dense_rank=_minimum_rank(previous.dense_rank, dense_rank),
            )

        for constraint in constraints:
            for language, terms in (
                ("en", constraint.terms_en),
                ("ko", constraint.terms_ko),
            ):
                query = " ".join(terms).strip()
                if not query:
                    continue
                query_id = _semantic_query_id(constraint, language, query)
                query_context[query_id] = (constraint, language)
                if "verbatim_review" in constraint.source_types:
                    semantic_queries.append(SemanticCellQuery(
                        query_id=query_id,
                        constraint_id=constraint.constraint_id,
                        role=constraint.role,
                        language=language,
                        text=query,
                    ))
                lexical_started = perf_counter()
                source_limits = ((constraint.source_types, limit),)
                if "verbatim_review" in constraint.source_types and len(constraint.source_types) > 1:
                    review_limit = min(limit, max(2, (limit + 1) // 2))
                    other_types = tuple(source for source in constraint.source_types if source != "verbatim_review")
                    source_limits = ((("verbatim_review",), review_limit), (other_types, limit - review_limit))
                hits = []
                cache_hits = 0
                for source_types, source_limit in source_limits:
                    if source_limit:
                        key = (query, requested_facilities, source_limit, tuple(source_types))
                        if key in lexical_hits:
                            cache_hits += 1
                        else:
                            lexical_hits[key] = tuple(scoped_index.search_evidence_for_facilities(
                                query,
                                facility_ids=requested_facilities,
                                limit_per_facility=source_limit,
                                source_types=source_types,
                            ))
                        hits.extend(lexical_hits[key])
                logger.info(
                    "Evidence recall phase=lexical constraint=%s language=%s "
                    "facilities=%d hits=%d cache_hits=%d elapsed_ms=%.1f",
                    constraint.constraint_id,
                    language,
                    len(facility_ids),
                    len(hits),
                    cache_hits,
                    (perf_counter() - lexical_started) * 1000,
                )
                ranks: dict[str, int] = defaultdict(int)
                for found in hits:
                    ranks[found.facility_id] += 1
                    upsert(
                        found,
                        ConstraintCell(
                            found.facility_id,
                            constraint.constraint_id,
                            constraint.role,
                        ),
                        language,
                        lexical_rank=ranks[found.facility_id],
                    )

        semantic_status = (
            "not_applicable" if not semantic_queries else
            "not_retried" if not allow_semantic else "disabled"
        )
        if self._semantic_source is not None and semantic_queries and allow_semantic:
            semantic_started = perf_counter()
            try:
                outcome: SemanticReviewOutcome = self._semantic_source.retrieve(
                    review_source_sha256=scoped_index.review_source_sha256,
                    facility_ids=tuple(facility_ids),
                    queries=tuple(semantic_queries),
                    limit_per_facility=limit,
                )
                semantic_status = outcome.status
                if outcome.used:
                    resolved = scoped_index.resolve_evidence_ids(tuple(
                        dict.fromkeys(
                            reference.evidence_id
                            for reference in outcome.references
                        )
                    ))
                    resolved_by_id = {item.evidence_id: item for item in resolved}
                    validated = []
                    for reference in outcome.references:
                        context = query_context.get(reference.query_id)
                        found = resolved_by_id.get(reference.evidence_id)
                        if (
                            context is None
                            or found is None
                            or not 1 <= reference.rank <= limit
                        ):
                            raise ValueError(
                                "semantic reference is outside the request"
                            )
                        constraint, language = context
                        if (
                            not found.is_verbatim
                            or found.source_type != "verbatim_review"
                            or found.facility_id != reference.facility_id
                            or found.facility_id not in facility_ids
                            or found.source_type not in constraint.source_types
                        ):
                            raise ValueError(
                                "semantic reference failed local validation"
                            )
                        validated.append((reference, found, constraint, language))

                    for reference, found, constraint, language in validated:
                        ranks = {
                            "sparse_rank": reference.rank
                            if reference.channel == "bge_m3_sparse" else None,
                            "dense_rank": reference.rank
                            if reference.channel == "bge_m3_dense" else None,
                        }
                        upsert(
                            found,
                            ConstraintCell(
                                found.facility_id,
                                constraint.constraint_id,
                                constraint.role,
                            ),
                            language,
                            **ranks,
                        )
            except Exception as exc:
                logger.warning(
                    "Semantic review retrieval degraded to BM25: %s",
                    type(exc).__name__,
                )
                semantic_status = "request_failed"
            logger.info(
                "Evidence recall phase=semantic facilities=%d queries=%d "
                "status=%s elapsed_ms=%.1f",
                len(facility_ids),
                len(semantic_queries),
                semantic_status,
                (perf_counter() - semantic_started) * 1000,
            )

        grouped: dict[
            tuple[ConstraintCell, Literal["en", "ko"]],
            list[EvidenceCellCandidate],
        ] = defaultdict(list)
        for item in candidate_by_key.values():
            score = sum(
                1.0 / (self.policy.hybrid_rrf_k + rank)
                for rank in (
                    item.lexical_rank,
                    item.sparse_rank,
                    item.dense_rank,
                )
                if rank is not None
            )
            grouped[(item.cell, item.query_language)].append(
                replace(item, fused_score=score)
            )

        fused: list[EvidenceCellCandidate] = []
        for group in grouped.values():
            ordered = sorted(
                group,
                key=lambda item: (
                    -item.fused_score,
                    item.hit.ordinal,
                    item.hit.evidence_id,
                ),
            )
            fused.extend(
                replace(item, fused_rank=rank)
                for rank, item in enumerate(ordered, start=1)
            )
        logger.info(
            "Evidence recall phase=search_complete facilities=%d candidates=%d "
            "elapsed_ms=%.1f",
            len(facility_ids),
            len(fused),
            (perf_counter() - search_started) * 1000,
        )
        return _CandidateSearchBatch(tuple(fused), semantic_status)

    def _score(
        self,
        admitted: Sequence[EvidenceCellCandidate],
        all_candidates: Sequence[EvidenceCellCandidate],
        constraints: Sequence[EvidenceConstraint],
    ) -> tuple[tuple[ScoredEvidence, ...], bool, str]:
        cells_by_evidence: dict[str, set[ConstraintCell]] = defaultdict(set)
        lexical_rank_by_evidence: dict[str, int] = {}
        fused_rank_by_evidence: dict[str, int] = {}
        for item in all_candidates:
            evidence_id = item.hit.evidence_id
            cells_by_evidence[evidence_id].add(item.cell)
            if item.lexical_rank is not None:
                lexical_rank_by_evidence[evidence_id] = min(
                    item.lexical_rank,
                    lexical_rank_by_evidence.get(evidence_id, item.lexical_rank),
                )
            if item.fused_rank is not None:
                fused_rank_by_evidence[evidence_id] = min(
                    item.fused_rank,
                    fused_rank_by_evidence.get(evidence_id, item.fused_rank),
                )

        constraint_by_id = {item.constraint_id: item for item in constraints}
        rerank_rank: dict[str, int] = {}
        rerank_score: dict[str, float] = {}
        used = False
        reasons: list[str] = []
        for role in ("disease", "support", "risk"):
            role_items = tuple(item for item in admitted if item.cell.role == role)
            if not role_items:
                continue
            role_constraints = {
                cell.constraint_id
                for item in role_items
                for cell in cells_by_evidence[item.hit.evidence_id]
                if cell.role == role
            }
            query = rerank_relevance_query(tuple(
                constraint_by_id[constraint_id]
                for constraint_id in sorted(role_constraints)
            ))
            ordered_hits = tuple(item.hit for item in role_items)
            rerank_started = perf_counter()
            if self._reranker is None:
                outcome = RerankOutcome(ordered_hits, False, "disabled")
            else:
                try:
                    outcome = self._reranker.rerank(query, ordered_hits)
                except Exception:
                    outcome = RerankOutcome(ordered_hits, False, "request_failed")
            logger.info(
                "Evidence recall phase=rerank role=%s candidates=%d used=%s "
                "reason=%s elapsed_ms=%.1f",
                role,
                len(ordered_hits),
                outcome.used,
                outcome.reason,
                (perf_counter() - rerank_started) * 1000,
            )
            used = used or outcome.used
            score_map = dict(outcome.scores) if outcome.used else {}
            reason = outcome.reason
            if outcome.used and reason == "ok" and set(score_map) != {
                hit.evidence_id for hit in ordered_hits
            }:
                reason = "incomplete_scores"
            reasons.append(reason)
            for rank, found in enumerate(outcome.hits, start=1):
                if found.evidence_id in score_map and (
                    found.evidence_id not in rerank_score
                    or score_map[found.evidence_id] > rerank_score[found.evidence_id]
                ):
                    rerank_rank[found.evidence_id] = rank
                    rerank_score[found.evidence_id] = score_map[found.evidence_id]

        profiles = score_local_distinctiveness(tuple(item.hit for item in admitted))
        scored: list[ScoredEvidence] = []
        for item in admitted:
            evidence_id = item.hit.evidence_id
            cells = cells_by_evidence[evidence_id]
            rank = rerank_rank.get(evidence_id)
            lexical_rank = lexical_rank_by_evidence.get(evidence_id)
            fused_rank = fused_rank_by_evidence.get(evidence_id)
            retrieval_rank = fused_rank or lexical_rank or 1
            relevance = 1.0 / max(1, rank or retrieval_rank)
            profile = profiles[evidence_id]
            coverage_gain = min(1.0, len(cells) / 2.0)
            selection_score = (
                0.65 * relevance
                + 0.15 * (1.0 / retrieval_rank)
                + 0.10 * coverage_gain
                + self.policy.distinctiveness_weight * profile.distinctiveness
            )
            scored.append(ScoredEvidence(
                hit=item.hit,
                matched_constraint_ids=frozenset(cell.constraint_id for cell in cells),
                roles=frozenset(cell.role for cell in cells),
                lexical_rank=lexical_rank,
                rerank_rank=rank,
                rerank_score=rerank_score.get(evidence_id),
                distinctiveness=profile.distinctiveness,
                local_cluster_id=profile.local_cluster_id,
                local_cluster_size=profile.local_cluster_size,
                corroboration_count=1,
                selection_score=selection_score,
            ))
        return (
            tuple(sorted(
                scored,
                key=lambda value: (*_relevance_order(value), value.hit.ordinal, value.hit.evidence_id),
            )),
            used,
            ",".join(dict.fromkeys(reasons)) or "no_candidates",
        )


def assess_facility_coverage(
    displayed_facility_ids: Sequence[str],
    constraints: Sequence[EvidenceConstraint],
    evidence: Sequence[ScoredEvidence],
) -> tuple[CoverageCell, ...]:
    output: list[CoverageCell] = []
    for facility_id in displayed_facility_ids:
        for constraint in constraints:
            matched = tuple(
                item.hit.evidence_id
                for item in evidence
                if item.hit.facility_id == facility_id
                and constraint.constraint_id in item.matched_constraint_ids
                and constraint.role in item.roles
            )
            status: CoverageStatus
            if matched:
                status = "matched"
            else:
                status = "missing"
            output.append(CoverageCell(
                facility_id=facility_id,
                constraint_id=constraint.constraint_id,
                role=constraint.role,
                required=constraint.required,
                status=status,
                evidence_ids=matched,
            ))
    return tuple(output)


def _relevance_order(item: ScoredEvidence) -> tuple[bool, float]:
    return (
        item.rerank_score is None,
        -(item.rerank_score if item.rerank_score is not None else item.selection_score),
    )


def select_evidence_groups(
    facility_ids: Sequence[str],
    evidence: Sequence[ScoredEvidence],
    coverage: Sequence[CoverageCell],
    *,
    limit: int,
) -> Mapping[str, EvidenceGroups]:
    output: dict[str, EvidenceGroups] = {}
    for facility_id in facility_ids:
        candidates = sorted((
            item for item in evidence if item.hit.facility_id == facility_id
        ), key=lambda item: (*_relevance_order(item), item.hit.ordinal, item.hit.evidence_id))
        selected: list[ScoredEvidence] = []
        covered: set[str] = set()
        unused = [
            item for item in candidates
            if item.hit.is_verbatim and item.hit.source_type == "verbatim_review"
            and any(char.isalpha() and not (
                '\u3130' <= char <= '\u318f' or '\u1100' <= char <= '\u11ff'
            ) for char in item.hit.original_text)
        ]
        warnings = sorted(
            (item for item in unused if "risk" in item.roles),
            key=lambda item: (*_relevance_order(item), item.hit.ordinal),
        )
        for warning in warnings:
            if len(selected) >= limit:
                break
            if warning.matched_constraint_ids - covered:
                selected.append(warning)
                covered.update(warning.matched_constraint_ids)
                unused.remove(warning)
        while unused and len(selected) < limit:
            selected_cluster_ids = {
                item.local_cluster_id for item in selected
            }
            unused.sort(
                key=lambda item: (
                    *_relevance_order(item),
                    item.local_cluster_id in selected_cluster_ids,
                    -len(item.matched_constraint_ids - covered),
                    -item.distinctiveness,
                    item.local_cluster_size,
                    item.hit.ordinal,
                )
            )
            choice = unused.pop(0)
            selected.append(choice)
            covered.update(choice.matched_constraint_ids)

        supporting = tuple(
            item for item in candidates
            if item.roles.intersection({"support", "disease"})
        )
        warnings = tuple(item for item in candidates if "risk" in item.roles)
        unverified = tuple(
            cell.constraint_id
            for cell in coverage
            if cell.facility_id == facility_id and cell.status == "missing"
        )
        output[facility_id] = EvidenceGroups(
            supporting=supporting,
            warnings=warnings,
            unverified=unverified,
            presented=tuple(selected),
        )
    return output


def evidence_payload(item: ScoredEvidence) -> dict[str, object]:
    return {
        "evidence_id": item.hit.evidence_id,
        "place_id": item.hit.facility_id,
        "source_type": item.hit.source_type,
        "source_field": item.hit.source_field,
        "source_index": item.hit.source_index,
        "source_locator": item.hit.source_locator,
        "text": item.hit.original_text,
        "language": item.hit.language_hint,
        "visit_date": item.hit.visit_date,
        "scraped_at": item.hit.scraped_at,
        "is_verbatim": item.hit.is_verbatim,
        "retrieval_roles": sorted(item.roles),
        "matched_constraint_ids": sorted(item.matched_constraint_ids),
        "distinctiveness_score": round(item.distinctiveness, 6),
        "similar_review_count": item.local_cluster_size,
        "corroboration_count": item.corroboration_count,
    }


def _admission_events(
    candidates: Sequence[EvidenceCellCandidate],
    scored: Sequence[ScoredEvidence],
    admitted: Sequence[EvidenceCellCandidate],
    attached_ids: set[str],
) -> tuple[AdmissionEvent, ...]:
    admitted_ids = {item.hit.evidence_id for item in admitted}
    scored_by_id = {item.hit.evidence_id: item for item in scored}
    output: list[AdmissionEvent] = []
    seen: set[tuple[str, ConstraintCell, str]] = set()
    for item in candidates:
        key = (item.hit.evidence_id, item.cell, item.query_language)
        if key in seen:
            continue
        seen.add(key)
        scored_item = scored_by_id.get(item.hit.evidence_id)
        was_admitted = item.hit.evidence_id in admitted_ids
        if not was_admitted:
            reason = "gpu_budget_or_duplicate"
        elif item.hit.evidence_id not in attached_ids:
            reason = "presentation_limit"
        else:
            reason = None
        output.append(AdmissionEvent(
            evidence_id=item.hit.evidence_id,
            facility_id=item.hit.facility_id,
            constraint_id=item.cell.constraint_id,
            role=item.cell.role,
            query_language=item.query_language,
            lexical_rank=item.lexical_rank,
            sparse_rank=item.sparse_rank,
            dense_rank=item.dense_rank,
            fused_rank=item.fused_rank,
            admitted_to_gpu=was_admitted,
            gpu_rank=scored_item.rerank_rank if scored_item else None,
            gpu_score=scored_item.rerank_score if scored_item else None,
            attached=item.hit.evidence_id in attached_ids,
            rejection_reason=reason,
        ))
    return tuple(output)


def _merge_scored(
    first: Sequence[ScoredEvidence],
    retry: Sequence[ScoredEvidence],
) -> tuple[ScoredEvidence, ...]:
    by_id = {item.hit.evidence_id: item for item in first}
    for item in retry:
        previous = by_id.get(item.hit.evidence_id)
        if previous is None:
            by_id[item.hit.evidence_id] = item
            continue
        by_id[item.hit.evidence_id] = replace(
            previous,
            matched_constraint_ids=(
                previous.matched_constraint_ids | item.matched_constraint_ids
            ),
            roles=previous.roles | item.roles,
            selection_score=max(previous.selection_score, item.selection_score),
        )
    return tuple(sorted(
        by_id.values(),
        key=lambda item: (-item.selection_score, item.hit.ordinal),
    ))


def _minimum_rank(left: int | None, right: int | None) -> int | None:
    ranks = tuple(value for value in (left, right) if value is not None)
    return min(ranks) if ranks else None


def _candidate_rank(item: EvidenceCellCandidate) -> int:
    if item.fused_rank is not None:
        return item.fused_rank
    channel_ranks = tuple(
        value
        for value in (item.lexical_rank, item.sparse_rank, item.dense_rank)
        if value is not None
    )
    return min(channel_ranks) if channel_ranks else 1_000_000


def _semantic_query_id(
    constraint: EvidenceConstraint,
    language: Literal["en", "ko"],
    query: str,
) -> str:
    payload = "\x00".join((
        constraint.constraint_id,
        constraint.role,
        language,
        query,
    ))
    return "cell:" + sha256(payload.encode("utf-8")).hexdigest()[:24]


def _clean_terms(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(
        " ".join(str(value).split())
        for value in values
        if str(value).strip()
    ))


def _contains_hangul(value: str) -> bool:
    return bool(re.search(r"[가-힣ㄱ-ㅎㅏ-ㅣ]", value))


def _normalize_review(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(re.findall(r"[0-9a-z가-힣]+", normalized))


def _character_ngrams(value: str, size: int = 3) -> frozenset[str]:
    if len(value) <= size:
        return frozenset({value}) if value else frozenset()
    return frozenset(
        value[index:index + size]
        for index in range(len(value) - size + 1)
    )


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _concreteness(value: str) -> float:
    normalized = _normalize_review(value)
    length_score = min(0.55, len(normalized) / 120.0)
    detail_patterns = (
        r"\d",
        r"의사|doctor",
        r"간호사|nurse",
        r"처방|prescri",
        r"설명|explain",
        r"검사|test",
        r"치료|treat",
        r"아이|child",
        r"분|시간|minute|hour",
    )
    detail_score = min(
        0.45,
        0.09 * sum(bool(re.search(pattern, value, re.IGNORECASE)) for pattern in detail_patterns),
    )
    return min(1.0, length_score + detail_score)
