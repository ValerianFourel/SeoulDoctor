"""Authenticated, generation-free access to the application's retrieval adapter."""

from copy import copy
from functools import lru_cache
from dataclasses import asdict
import hmac
import os
from threading import BoundedSemaphore
from uuid import uuid4

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from models import State
from review_presentation import response_language
from search.evidence_retrieval import EvidenceRecallPolicy
from search.live_retrieval import CandidateRetrievalAdapter, RetrievalQuery, RRFPolicy
from search.rules import compile_legacy_state_rules
from search.scope import ScopeBuilder

router = APIRouter()
slots = BoundedSemaphore(4)


class RetrievalInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=4000)
    location: str = Field(min_length=1, max_length=100)
    specialty: str = Field(min_length=1, max_length=100)


def authorize(token):
    expected = os.environ.get("SEOULDOC_EVAL_AUTH_TOKEN", "")
    if not expected or not hmac.compare_digest(token, expected):
        raise HTTPException(403, "Evaluation token required")


def limits():
    return {"candidate_policy": asdict(RRFPolicy()),
            "evidence_policy": asdict(EvidenceRecallPolicy(require_remote_services=True)),
            "evaluation_retry_rerank_budget": 0, "reported_facilities": 5}


@lru_cache(maxsize=1)
def evaluation_pipeline():
    import main
    pipeline = copy(main.rag_pipeline)
    pipeline.embedding_function = copy(main.rag_pipeline.embedding_function)
    pipeline.embedding_function.client = main.rag_pipeline.embedding_function.client.with_options(
        max_retries=0, timeout=12.0,
    )
    return pipeline


@router.get("/internal/retrieval/warmup")
def warmup(x_seouldoc_eval_token: str = Header(default="")):
    authorize(x_seouldoc_eval_token)
    import main
    if main.search_index_release is None or main.rag_pipeline is None:
        raise HTTPException(503, "Application retrieval is not initialized")
    evaluation_pipeline().embedding_function(["retrieval warmup"])
    return {"limits": limits(), "index_version": main.search_index_release.version,
            "semantic_configured": main.semantic_evidence_source is not None,
            "reranker_configured": main.evidence_reranker is not None}


@router.post("/internal/retrieval")
def retrieve(body: RetrievalInput, x_seouldoc_eval_token: str = Header(default="")):
    authorize(x_seouldoc_eval_token)
    if not slots.acquire(blocking=False):
        raise HTTPException(429, "Four retrieval requests are already active")
    try:
        import main
        if main.search_index_release is None or main.rag_pipeline is None:
            raise HTTPException(503, "Application retrieval is not initialized")
        language, _ = response_language(body.query)
        state = State(specialty=body.specialty, specialty_confidence=1.0,
                      location=body.location, district=body.location, search_mode="zone",
                      is_citywide_search=body.location.casefold() == "seoul",
                      comment_terms=[body.query], language_pref=language)
        compiled = compile_legacy_state_rules(body.query, state, turn_id=str(uuid4()),
                                             allowed_specialties=main.available_specialties)
        if compiled.rules is None:
            raise HTTPException(422, "Location or specialty could not be compiled")
        scope = ScopeBuilder().build(main.df_filtered, compiled.rules,
                                     index_version=main.search_index_release.version)
        outcome = CandidateRetrievalAdapter(
            active_index=main.search_index_release, legacy_pipeline=evaluation_pipeline(),
            evidence_reranker=main.evidence_reranker,
            semantic_evidence_source=main.semantic_evidence_source,
            evidence_policy=EvidenceRecallPolicy(require_remote_services=True, retry_rerank_budget=0),
        ).rank(scope=scope, eligible=scope.restrict_dataframe(main.df_filtered),
               rules=compiled.rules, query=RetrievalQuery(
                   text=f"{body.query} {body.specialty}", max_distance_km=5.0,
                   search_mode="zone", specialty_confidence=1.0,
                   exact_terms=(body.query,), target_language=language))
        metadata = outcome.dataframe.attrs.get("rag_metadata", {})
        return {"facilities": [
            {"place_id": str(row["place_id"]),
             "selected": [{"evidence_id": item["evidence_id"], "place_id": item["place_id"]}
                          for item in row.get("retrieval_evidence", [])]}
            for row in outcome.dataframe.head(5).to_dict("records")],
            "retrieved": [{"evidence_id": item["evidence_id"], "place_id": item["facility_id"]}
                          for item in metadata.get("evidence_admissions", [])],
            "status": outcome.telemetry.status,
            "semantic_status": metadata.get("semantic_status"),
            "reranker_reason": metadata.get("reranker_reason"),
            "channel_hit_counts": dict(outcome.telemetry.channel_hit_counts)}
    finally:
        slots.release()
