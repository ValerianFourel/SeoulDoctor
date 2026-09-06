"""Fail-open client for facility-scoped BGE-M3 review retrieval."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from typing import Literal, Protocol, Sequence
from uuid import uuid4

import requests

from search.contracts import EvidenceRole
from search.service_lease import lease_allows_request, parse_service_expiry


logger = logging.getLogger(__name__)

SemanticChannel = Literal["bge_m3_sparse", "bge_m3_dense"]
SemanticStatus = Literal[
    "ok",
    "disabled",
    "request_failed",
    "invalid_response",
    "release_mismatch",
    "service_expired",
]


class _PostSession(Protocol):
    def post(self, url: str, **kwargs: object) -> object:
        ...


@dataclass(frozen=True)
class SemanticCellQuery:
    query_id: str
    constraint_id: str
    role: EvidenceRole
    language: Literal["en", "ko"]
    text: str


@dataclass(frozen=True)
class SemanticEvidenceReference:
    query_id: str
    evidence_id: str
    facility_id: str
    channel: SemanticChannel
    rank: int
    score: float


@dataclass(frozen=True)
class SemanticReviewOutcome:
    status: SemanticStatus
    references: tuple[SemanticEvidenceReference, ...] = ()
    release_id: str | None = None
    model_id: str | None = None

    @property
    def used(self) -> bool:
        return self.status == "ok"


class SemanticEvidenceSource(Protocol):
    def retrieve(
        self,
        *,
        review_source_sha256: str,
        facility_ids: Sequence[str],
        queries: Sequence[SemanticCellQuery],
        limit_per_facility: int,
    ) -> SemanticReviewOutcome:
        ...


class RemoteBgeM3ReviewRetriever:
    """Retrieve sparse and dense references without trusting remote metadata."""

    def __init__(
        self,
        *,
        base_url: str,
        token: str = "",
        release_id: str,
        timeout_seconds: float = 8.0,
        session: _PostSession | None = None,
        expires_at: str = "",
        model_revision: str = "",
    ) -> None:
        normalized_url = base_url.strip().rstrip("/")
        if not normalized_url:
            raise ValueError("BGE-M3 retriever base_url cannot be empty")
        if not release_id.strip():
            raise ValueError("BGE-M3 retriever release_id cannot be empty")
        if not 0.1 <= timeout_seconds <= 60.0:
            raise ValueError("BGE-M3 retriever timeout must be between 0.1 and 60")
        self._base_url = normalized_url
        self._token = token.strip()
        self._release_id = release_id.strip()
        self._timeout_seconds = timeout_seconds
        self._session = session or requests.Session()
        self._expires_at = parse_service_expiry(expires_at)
        self._model_revision = model_revision.strip()

    def retrieve(
        self,
        *,
        review_source_sha256: str,
        facility_ids: Sequence[str],
        queries: Sequence[SemanticCellQuery],
        limit_per_facility: int,
    ) -> SemanticReviewOutcome:
        if not lease_allows_request(self._expires_at, self._timeout_seconds):
            return SemanticReviewOutcome("service_expired")
        facilities = tuple(dict.fromkeys(str(value).strip() for value in facility_ids))
        query_batch = tuple(queries)
        if not facilities or not query_batch:
            return SemanticReviewOutcome("disabled")
        if len(facilities) > 50 or len(query_batch) > 64:
            return SemanticReviewOutcome("invalid_response")
        if not 1 <= limit_per_facility <= 20:
            return SemanticReviewOutcome("invalid_response")
        if len(review_source_sha256) != 64:
            return SemanticReviewOutcome("invalid_response")

        payload = {
            "schema_version": "seouldoc.semantic-query/v1",
            "request_id": uuid4().hex,
            "expected_release": {
                "release_id": self._release_id,
                "review_source_sha256": review_source_sha256,
                "evidence_id_schema": "seouldoc.raw-review-evidence-id/v1",
            },
            "facility_ids": list(facilities),
            "queries": [
                {
                    "query_id": item.query_id,
                    "constraint_id": item.constraint_id,
                    "role": item.role,
                    "language": item.language,
                    "text": item.text,
                }
                for item in query_batch
            ],
            "limit_per_facility_per_channel": limit_per_facility,
        }
        headers = (
            {"Authorization": f"Bearer {self._token}"}
            if self._token else {}
        )
        try:
            response = self._session.post(
                f"{self._base_url}/v1/retrieve",
                json=payload,
                headers=headers,
                timeout=self._timeout_seconds,
            )
            response.raise_for_status()
            body = response.json()
        except (requests.RequestException, TimeoutError, ValueError) as exc:
            logger.warning("BGE-M3 review retrieval failed: %s", type(exc).__name__)
            return SemanticReviewOutcome("request_failed")

        return self._parse_response(
            body,
            review_source_sha256=review_source_sha256,
            facility_ids=frozenset(facilities),
            query_ids=frozenset(item.query_id for item in query_batch),
        )

    def _parse_response(
        self,
        body: object,
        *,
        review_source_sha256: str,
        facility_ids: frozenset[str],
        query_ids: frozenset[str],
    ) -> SemanticReviewOutcome:
        if not isinstance(body, dict):
            return SemanticReviewOutcome("invalid_response")
        release = body.get("release")
        results = body.get("results")
        if (
            body.get("schema_version") != "seouldoc.semantic-results/v1"
            or not isinstance(release, dict)
            or not isinstance(results, list)
        ):
            return SemanticReviewOutcome("invalid_response")
        release_id = release.get("release_id")
        source_digest = release.get("review_source_sha256")
        model_id = release.get("model_id")
        if (
            release_id != self._release_id
            or source_digest != review_source_sha256
            or (self._model_revision and release.get("model_revision") != self._model_revision)
        ):
            return SemanticReviewOutcome("release_mismatch")
        if model_id != "BAAI/bge-m3":
            return SemanticReviewOutcome("invalid_response")

        references: list[SemanticEvidenceReference] = []
        seen: set[tuple[str, str, str]] = set()
        for raw in results:
            if not isinstance(raw, dict):
                return SemanticReviewOutcome("invalid_response")
            query_id = raw.get("query_id")
            evidence_id = raw.get("evidence_id")
            facility_id = raw.get("facility_id")
            channel = raw.get("channel")
            rank = raw.get("rank")
            score = raw.get("score")
            key = (str(query_id), str(evidence_id), str(channel))
            if (
                not isinstance(query_id, str) or query_id not in query_ids
                or not isinstance(evidence_id, str) or not evidence_id
                or not isinstance(facility_id, str) or facility_id not in facility_ids
                or channel not in {"bge_m3_sparse", "bge_m3_dense"}
                or isinstance(rank, bool) or not isinstance(rank, int) or rank < 1
                or isinstance(score, bool) or not isinstance(score, (int, float))
                or not math.isfinite(float(score))
                or key in seen
            ):
                return SemanticReviewOutcome("invalid_response")
            seen.add(key)
            references.append(SemanticEvidenceReference(
                query_id=query_id,
                evidence_id=evidence_id,
                facility_id=facility_id,
                channel=channel,
                rank=rank,
                score=float(score),
            ))
        return SemanticReviewOutcome(
            "ok",
            tuple(references),
            str(release_id),
            model_id,
        )
