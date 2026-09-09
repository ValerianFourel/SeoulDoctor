"""Fail-open client for the private GPU evidence-reranking service."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from typing import Protocol, Sequence

import requests

from search.indexes.repository import EvidenceHit
from search.service_lease import lease_allows_request, parse_service_expiry


logger = logging.getLogger(__name__)


class _PostSession(Protocol):
    def post(self, url: str, **kwargs: object) -> object:
        ...


@dataclass(frozen=True)
class RerankOutcome:
    hits: tuple[EvidenceHit, ...]
    used: bool
    reason: str
    model: str | None = None
    scores: tuple[tuple[str, float], ...] = ()
    gpu_execution_verified: bool = False


class RemoteEvidenceReranker:
    """Rerank retrieved evidence while keeping local metadata authoritative."""

    def __init__(
        self,
        *,
        base_url: str,
        token: str = "",
        timeout_seconds: float = 20.0,
        max_candidates: int = 256,
        session: _PostSession | None = None,
        expires_at: str = "",
    ) -> None:
        normalized_url = base_url.strip().rstrip("/")
        if not normalized_url:
            raise ValueError("reranker base_url cannot be empty")
        if not 0.1 <= timeout_seconds <= 120.0:
            raise ValueError("reranker timeout_seconds must be between 0.1 and 120")
        if not 1 <= max_candidates <= 256:
            raise ValueError("reranker max_candidates must be between 1 and 256")
        self._base_url = normalized_url
        self._token = token.strip()
        self._timeout_seconds = timeout_seconds
        self._max_candidates = max_candidates
        self._session = session or requests.Session()
        self._expires_at = parse_service_expiry(expires_at)

    def rerank(
        self,
        query: str,
        hits: Sequence[EvidenceHit],
    ) -> RerankOutcome:
        original = tuple(hits)
        if not original:
            return RerankOutcome(original, False, "no_candidates")
        if not lease_allows_request(self._expires_at, self._timeout_seconds):
            return RerankOutcome(original, False, "service_expired")
        if not query.strip():
            return RerankOutcome(original, False, "empty_query")

        unique: list[EvidenceHit] = []
        seen_ids: set[str] = set()
        for hit in original:
            if hit.evidence_id not in seen_ids:
                seen_ids.add(hit.evidence_id)
                unique.append(hit)
        unique_original = tuple(unique)
        selected = unique_original[: self._max_candidates]
        payload = {
            "query": query.strip(),
            "texts": [hit.original_text for hit in selected],
        }
        headers = (
            {"Authorization": f"Bearer {self._token}"}
            if self._token
            else {}
        )

        try:
            response = self._session.post(
                f"{self._base_url}/rerank",
                json=payload,
                headers=headers,
                timeout=self._timeout_seconds,
            )
            response.raise_for_status()
            body = response.json()
        except (requests.RequestException, TimeoutError, ValueError) as exc:
            logger.warning(
                "GPU evidence reranker request failed: %s",
                type(exc).__name__,
            )
            return RerankOutcome(original, False, "request_failed")

        parsed = self._validated_scores(body, selected)
        if parsed is None:
            logger.warning("GPU evidence reranker returned an invalid response")
            return RerankOutcome(original, False, "invalid_response")

        score_by_id, model, gpu_execution_verified = parsed
        original_position = {
            hit.evidence_id: position for position, hit in enumerate(selected)
        }
        reranked = tuple(sorted(
            selected,
            key=lambda hit: (
                -score_by_id[hit.evidence_id],
                original_position[hit.evidence_id],
            ),
        ))
        return RerankOutcome(
            (*reranked, *unique_original[len(selected):]),
            True,
            "ok" if len(selected) == len(unique_original) else "candidate_limit",
            model,
            tuple(sorted(score_by_id.items())),
            gpu_execution_verified,
        )

    @staticmethod
    def _validated_scores(
        body: object,
        selected: Sequence[EvidenceHit],
    ) -> tuple[dict[str, float], str | None, bool] | None:
        model: str | None = None
        gpu_execution_verified = False
        if isinstance(body, list):
            results = body
        elif isinstance(body, dict) and isinstance(body.get("results"), list):
            results = body["results"]
            raw_model = body.get("model")
            model = raw_model if isinstance(raw_model, str) else None
            execution = body.get("execution")
            if (
                not isinstance(execution, dict)
                or execution.get("gpu_execution_verified") is not True
                or not isinstance(execution.get("device"), str)
                or not execution["device"].startswith("cuda")
            ):
                return None
            gpu_execution_verified = True
        else:
            return None

        expected_ids = {hit.evidence_id for hit in selected}
        scores: dict[str, float] = {}
        for item in results:
            if not isinstance(item, dict):
                return None
            evidence_id = item.get("evidence_id")
            if evidence_id is None:
                index = item.get("index")
                if (
                    isinstance(index, bool)
                    or not isinstance(index, int)
                    or not 0 <= index < len(selected)
                ):
                    return None
                evidence_id = selected[index].evidence_id
            raw_score = item.get("score")
            if (
                not isinstance(evidence_id, str)
                or evidence_id not in expected_ids
                or evidence_id in scores
                or isinstance(raw_score, bool)
                or not isinstance(raw_score, (int, float))
            ):
                return None
            score = float(raw_score)
            if not math.isfinite(score):
                return None
            scores[evidence_id] = score

        if set(scores) != expected_ids:
            return None
        return scores, model, gpu_execution_verified
