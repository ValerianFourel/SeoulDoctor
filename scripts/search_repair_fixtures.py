"""Isolated, deterministic provider boundaries for the frozen /chat fault cases."""

from contextlib import asynccontextmanager, contextmanager, ExitStack
from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import requests
from fastapi.testclient import TestClient


BACKEND = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND))
from search.indexes.repository import EvidenceHit, FacilityHit
from search.reranker import RerankOutcome
from search.semantic_retriever import SemanticReviewOutcome


LATITUDE, LONGITUDE = 37.5702, 126.9831
REVIEW_PAIRS = (
    ("발목 검사를 자세히 설명해 주었습니다.", "The ankle examination was explained in detail."),
    ("정형외과에서 발목 치료 후에도 아프고 설명이 부족했습니다.", "My ankle still hurt after orthopedic treatment, and the explanation was insufficient."),
    ("발목 물리치료를 받고 계단을 오르기가 편해졌습니다.", "After ankle physical therapy, climbing stairs became easier."),
    ("발목 엑스레이 결과를 의사가 설명했습니다.", "The doctor explained the ankle X-ray results."),
    ("발목 진료를 기다리는 시간이 길었습니다.", "The wait for my ankle appointment was long."),
    ("발목 보호대 착용 방법을 배웠습니다.", "I learned how to wear an ankle brace."),
    ("발목 주사를 맞기 전에 비용을 안내받았습니다.", "I was told the cost before the ankle injection."),
    ("정형외과 발목 진료 후 재활 운동을 알려주었습니다.", "I was shown rehabilitation exercises after the orthopedic ankle appointment."),
    ("발목 치료 때 간호사가 질문을 무시했습니다.", "The nurse ignored my question during ankle treatment."),
    ("발목 통증을 설명할 시간이 충분했습니다.", "There was enough time to describe my ankle pain."),
)


class FixtureIndex:
    version = "search-repair-synthetic-v1"

    def __init__(self, owners, reviews):
        self.owners, self.reviews = tuple(owners), tuple(reviews)
        self.review_source_sha256 = sha256(json.dumps([
            (item.facility_id, item.evidence_id, item.original_text)
            for item in reviews], ensure_ascii=False).encode()).hexdigest()
        self.queries = []

    @contextmanager
    def within(self, scope):
        scoped = FixtureIndex(scope.facility_ids, self.reviews)
        scoped.review_source_sha256 = self.review_source_sha256
        scoped.queries = self.queries
        yield scoped

    def search_facilities(self, query, *, limit):
        return [FacilityHit(owner, index, 1 / (index + 1), "fixture_lexical")
                for index, owner in enumerate(self.owners[:limit])]

    def search_dense(self, embedding, *, limit):
        return self.search_facilities("", limit=limit)

    def search_evidence_for_facilities(self, query, *, facility_ids, limit_per_facility, source_types):
        self.queries.append({"query": query, "facility_ids": list(facility_ids),
                             "source_types": list(source_types)})
        terms = re.findall(r"[\w가-힣]+", query.casefold())
        result = []
        for owner in facility_ids:
            if owner not in self.owners:
                raise ValueError("fixture query escaped candidate scope")
            matches = [item for item in self.reviews if item.facility_id == owner
                       and item.source_type in source_types
                       and any(term in item.original_text.casefold() for term in terms)]
            result.extend(matches[:limit_per_facility])
        return result

    def search_evidence(self, query, *, limit, source_types):
        return self.search_evidence_for_facilities(query, facility_ids=self.owners,
            limit_per_facility=limit, source_types=source_types)[:limit]

    def resolve_evidence_ids(self, ids):
        records = {item.evidence_id: item for item in self.reviews if item.facility_id in self.owners}
        return [records[identity] for identity in ids]


@asynccontextmanager
async def fixture_lifespan(app):
    yield


class FixtureApp:
    """Run real chat, scope, evidence selection and presentation with local inputs."""

    def __init__(self, scenario):
        self.scenario = scenario
        self.fixture = scenario["execution"]["fixture"]
        self.variant = scenario.get("variant", {})
        self.events = []
        self.clock = 1000.0
        self.stack = ExitStack()
        count = self.fixture.get("eligible_reviews_per_specialist",
                                 self.fixture.get("eligible_korean_reviews_per_specialist", 3))
        counts = count if isinstance(count, list) else [count, count]
        distances = self.fixture.get("matching_specialist_distances_km",
                                    [0.3 + index * 0.2 for index in range(len(counts))])
        counts = (counts * len(distances))[:len(distances)]
        rows, reviews = [], []
        self.translations = {"정형외과 후기: " + source: "Orthopedics review: " + translated
                             for source, translated in REVIEW_PAIRS}
        self.expected_counts = {}
        self.irrelevant_ids = set()
        for index, distance in enumerate(distances):
            owner = f"fixture-orthopedics-{index}"
            rows.append(self.facility(owner, distance, "정형외과"))
            texts = self.fixture.get("reviews", ["정형외과 후기: " + pair[0] for pair in REVIEW_PAIRS[:counts[index]]])
            self.expected_counts[owner] = min(len(texts), 3)
            for ordinal, text in enumerate(texts):
                identity = sha256(f"{owner}|{ordinal}|{text}".encode()).hexdigest()[:20]
                reviews.append(EvidenceHit(
                    evidence_id="review:" + identity, facility_id=owner, ordinal=ordinal,
                    score=1 / (ordinal + 1), channel="fixture_review_lexical",
                    source_type="verbatim_review", source_field="comment", source_index=ordinal,
                    source_locator=f"synthetic:{owner}:{ordinal}", original_text=text,
                    language_hint="ko" if re.search("[가-힣]", text) else "en",
                    visit_date="2026-01-01", scraped_at="2026-01-02", is_verbatim=True))
            if self.fixture.get("irrelevant_reviews_excluded"):
                text = "The waiting-room coffee smelled pleasant."
                identity = "review:" + sha256(f"{owner}|99|{text}".encode()).hexdigest()[:20]
                self.irrelevant_ids.add(identity)
                reviews.append(EvidenceHit(identity, owner, 99, 0.01, "fixture_review_lexical",
                    "verbatim_review", "comment", 99, f"synthetic:{owner}:99", text, "en", "", "", True))
        for index, distance in enumerate(self.fixture.get("unrelated_specialist_distances_km", [0.1])):
            rows.append(self.facility(f"fixture-dermatology-{index}", distance, "피부과"))
        self.catalog = pd.DataFrame(rows)
        self.index = FixtureIndex(tuple(self.catalog["place_id"]), reviews)
        self.sources = {item.evidence_id: item for item in reviews}

    @staticmethod
    def facility(owner, distance, category):
        return {"place_id": owner, "name": owner, "category": category,
                "address": "서울특별시 종로구", "file_district": "종로구", "file_dong": "종로1가",
                "lat": LATITUDE + math.degrees(distance / 6371.0088), "lon": LONGITUDE,
                "amenities": {}, "business_hours": "", "medical_info_parsed": {},
                "Summaries": [], "Summaries_Korean": [], "Key_Highlights": [], "has_english": False}

    def __enter__(self):
        import main
        import evidence_response
        import review_presentation
        import utils
        from rate_limit import SlidingWindowRateLimiter

        self.main = main
        settings = {
            "df_filtered": self.catalog, "available_specialties": ["정형외과", "피부과"],
            "search_index_release": self.index, "client": object(),
            "rag_pipeline": SimpleNamespace(embedding_function=lambda texts: [[1.0] * 1536 for _ in texts]),
            "ENABLE_RETRIEVAL_DEBUG": True, "GOOGLE_MAPS_API_KEY": "", "KAKAO_REST_API_KEY": "",
            "chat_rate_limiter": SlidingWindowRateLimiter(max_requests=100, window_seconds=60),
            "evidence_reranker": SimpleNamespace(rerank=lambda query, hits: RerankOutcome(
                tuple(hits), True, "ok", "synthetic-reranker",
                tuple((hit.evidence_id, 1.0 / (index + 1)) for index, hit in enumerate(hits)),
            )),
            "semantic_evidence_source": SimpleNamespace(retrieve=lambda **kwargs: SemanticReviewOutcome(
                "request_failed" if self.fixture.get("semantic_retrieval") == "unavailable" else "ok")),
        }
        for name, value in settings.items():
            self.stack.enter_context(patch.object(main, name, value))
        self.stack.enter_context(patch.object(main.app.router, "lifespan_context", fixture_lifespan))
        self.stack.enter_context(patch.object(main, "request_json_completion", return_value=({"intent": "PROVIDE_INFO"}, None)))
        self.stack.enter_context(patch.object(main, "extract_entities", side_effect=self.extract))
        self.stack.enter_context(patch.object(utils, "verify_and_standardize_address", return_value={
            "lat": LATITUDE, "lon": LONGITUDE, "address_korean": "서울특별시 종로구 종로1가",
            "district": "종로구", "dong": "종로1가"}))
        self.stack.enter_context(patch.object(main, "request_answer_completion", side_effect=self.answer))
        self.stack.enter_context(patch.object(evidence_response, "monotonic", side_effect=lambda: self.clock))
        self.stack.enter_context(patch.object(review_presentation.requests, "post", side_effect=self.translate))
        self.stack.enter_context(patch.object(requests.sessions.Session, "request",
                                             side_effect=AssertionError("fixture attempted network I/O")))
        self.stack.enter_context(patch.dict("os.environ", {"GOOGLE_TRANSLATE_API_KEY":
            "" if self.variant.get("translation_fault") == "missing_credentials" else "synthetic-key"}))
        self.client = self.stack.enter_context(TestClient(main.app))
        return self

    def __exit__(self, *args):
        return self.stack.__exit__(*args)

    def extract(self, message, consent=None):
        return {"operation": "refine", "specialty": "정형외과", "specialty_confidence": 0.95,
                "location": "Jonggak", "latitude": LATITUDE, "longitude": LONGITUDE,
                "address_korean": "서울특별시 종로구 종로1가", "district": "종로구", "dong": "종로1가",
                "distance_km": 1.0 if "only within 1 km" in message else None,
                "travel_label": None, "hard_keywords": [], "soft_keywords": [],
                "negative_hard_keywords": [], "negative_keywords": [], "place_terms": [],
                "gender_terms": [], "disease_terms": [],
                "comment_terms": ["doctor", "nurse", "explanations"] if self.fixture.get("reviews") else [],
                "required_hours": [],
                "remove_terms": [], "visit_reason": "ankle pain" if "ankle pain" in message else None}

    def answer(self, *args, **kwargs):
        payload = json.loads(kwargs["messages"][-1]["content"])
        verification = "proposal" in payload
        self.events.append("verification" if verification else "synthesis")
        fault = self.variant.get("answer_fault")
        if fault == "synthesis_exhausts_answer_deadline":
            self.clock += 100.0
            raise TimeoutError("answer_deadline_exhausted")
        if fault == "malformed_synthesis":
            return {"malformed": True}, None
        if verification:
            rejected = fault == "semantic_verification_rejected"
            return {"accepted": not rejected, "issues": ["synthetic unsupported claim"] if rejected else []}, None
        if fault is None and not self.fixture.get("reviews"):
            raise TimeoutError("synthetic_answer_unavailable")
        items = payload["evidence"][:3]
        citations = [{"marker": index + 1, "place_id": item["place_id"],
                      "evidence_id": item["evidence_id"], "original_excerpt": item["original_text"]}
                     for index, item in enumerate(items)]
        answer = " ".join(f'A patient reports: {item["original_text"]} [{index + 1}].'
                          for index, item in enumerate(items))
        answer += " Compare the mixed experiences and ask the clinic about explanations before booking."
        return {"answer": answer, "assessments": [], "citations": citations}, None

    def translate(self, url, *, headers, json, timeout):
        self.events.append("translation")
        fault = self.variant.get("translation_fault")
        if fault == "timeout":
            raise requests.Timeout("synthetic timeout")
        if fault == "http_error":
            raise requests.HTTPError("synthetic HTTP error")
        payload = {} if fault == "invalid_response" else {"data": {"translations": [
            {"translatedText": self.translations[text]} for text in json["q"]]}}
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: payload)

    def post(self, url, *, json, **kwargs):
        if not url.endswith("/chat"):
            raise ValueError("fixture transport accepts only /chat")
        return self.client.post("/chat", json=json)

    def initial_state(self):
        return deepcopy(self.fixture.get("initial_state", {}))

    def provenance(self):
        return {"mode": "local_synthetic_api", "network_inference": False,
                "boundaries": ["geocoded extraction", "query embeddings", "index contents",
                               "semantic retrieval", "reranker", "answer model", "translation provider"],
                "source_revision": self.index.review_source_sha256,
                "sources": [{"evidence_id": item.evidence_id, "place_id": item.facility_id,
                             "text": item.original_text, "source_locator": item.source_locator}
                            for item in self.sources.values()]}

    def assess(self, body):
        oracle, metadata = self.scenario["oracle"], body["state"].get("last_retrieval_metadata", {})
        cards, checks, failures = body.get("results", []), [], []

        def check(name, passed):
            checks.append(name)
            if not passed:
                failures.append(name)

        for card in cards:
            owner = card["place_id"]
            reviews = card.get("retrieval_evidence", [])
            check(f"{owner}_minimum_source_reviews", len(reviews) >= self.expected_counts.get(owner, 2))
            for review in reviews:
                source = self.sources.get(review.get("evidence_id"))
                check(f'{review.get("evidence_id")}_source_exact', source is not None
                      and source.facility_id == owner and source.original_text == review.get("text")
                      and review.get("review_source_sha256") == self.index.review_source_sha256)
                check(f'{review.get("evidence_id")}_relevant', review.get("evidence_id") not in self.irrelevant_ids)
                status = review.get("presentation", {}).get("status")
                if "translation_status" in oracle:
                    check(f'{review.get("evidence_id")}_translation_status', status == oracle["translation_status"])
                if status == "translated":
                    check(f'{review.get("evidence_id")}_translation_exact',
                          review["presentation"]["text"] == self.translations.get(review["text"]))
        if oracle.get("expected_card_count") != 0:
            check("fixture_specialists_found", bool(cards))
            check("review_query_executed", bool(self.index.queries))
        if "attempted_radii_km" in oracle:
            check("exact_radius_attempts", metadata.get("search_attempted_radii_km") == oracle["attempted_radii_km"])
            check("expanded_radius_recorded", metadata.get("search_radius_expanded") is True)
            check("final_radius", body["state"].get("max_distance_km") == oracle["final_radius_km"])
        answer = metadata.get("answer", {})
        if "answer_status" in oracle:
            check("answer_fault_observed", answer.get("status") == oracle["answer_status"])
            check("translation_after_answer_fault", "translation" in self.events
                  and self.events.index("translation") > self.events.index("synthesis"))
        if "internal_reason" in self.variant:
            check("translation_fault_reason", answer.get("translation", {}).get("reason") == self.variant["internal_reason"])
        if "retrieval_execution_status" in oracle:
            check("retrieval_fault_observed", metadata.get("retrieval_execution_status") in oracle["retrieval_execution_status"])
            text = body["response"].lower()
            check("retrieval_retry_guidance", "review" in body["response"].lower()
                  and any(phrase in text for phrase in ("incomplete", "may have missed", "couldn't finish", "couldn’t finish"))
                  and "retry" in text)
        if self.fixture.get("reviews"):
            visible = {review["text"] for card in cards for review in card.get("retrieval_evidence", [])}
            check("mixed_originals_retained", set(self.fixture["reviews"]) <= visible)
        return {"checks": checks, "failures": failures, "events": list(self.events),
                "review_queries": list(self.index.queries)}
