"""Exercise response preservation through the public chat route."""

from contextlib import asynccontextmanager, ExitStack
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pandas as pd
from fastapi.testclient import TestClient


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from backend.tests.test_search_scope import facility
from models import State
from rate_limit import SlidingWindowRateLimiter


ORIGIN_LAT = 37.5702
ORIGIN_LON = 126.9831
GENERIC_RESPONSE = "See the facility cards and reviews below."


@asynccontextmanager
async def _test_lifespan(app):
    yield


def _catalog() -> pd.DataFrame:
    rows = [
        facility(
            "far",
            category="정형외과",
            latitude=ORIGIN_LAT + 0.018,
            longitude=ORIGIN_LON,
            name="멀리정형외과",
        ),
        facility(
            "closest",
            category="정형외과",
            latitude=ORIGIN_LAT + 0.0039,
            longitude=ORIGIN_LON,
            name="광화문정형외과의원",
        ),
        facility(
            "middle",
            category="정형외과",
            latitude=ORIGIN_LAT + 0.006,
            longitude=ORIGIN_LON,
            name="종각튼튼의원",
        ),
        facility(
            "outer",
            category="정형외과",
            latitude=ORIGIN_LAT + 0.008,
            longitude=ORIGIN_LON,
            name="을지정형외과",
        ),
    ]
    for row in rows:
        row["has_english"] = True
        row["address"] = "서울특별시 종로구"
        row["file_district"] = "종로구"
        row["file_dong"] = "종로1가"
    return pd.DataFrame(rows)


def _extraction(*, location: str | None, distance_km: float | None = None):
    return {
        "operation": "refine",
        "specialty": "정형외과" if location else None,
        "specialty_confidence": 0.95 if location else 0.0,
        "location": location,
        "latitude": None,
        "longitude": None,
        "distance_km": distance_km,
        "travel_label": None,
        "language_pref": "English Preferred",
        "hard_keywords": [],
        "soft_keywords": [],
        "negative_hard_keywords": [],
        "negative_keywords": [],
        "place_terms": [location] if location else [],
        "gender_terms": [],
        "disease_terms": [],
        "comment_terms": [],
        "required_hours": [],
        "remove_terms": [],
    }


class JsonModelBoundary:
    def __init__(self, intents, extractions=()):
        self.intents = list(intents)
        self.extractions = list(extractions)
        self.calls = []

    def __call__(self, *args, **kwargs):
        required_keys = tuple(kwargs["required_keys"])
        self.calls.append(required_keys)
        if required_keys == ("intent",):
            return {"intent": self.intents.pop(0), "confidence": 1.0}, None
        if required_keys == ("specialty", "location", "distance"):
            return {
                "specialty": "keep",
                "location": "keep",
                "distance": "change",
                "keywords": "keep",
                "reasoning": "The patient narrowed the radius.",
            }, None
        if required_keys == ("specialty", "location", "travel_label"):
            return self.extractions.pop(0), None
        raise AssertionError(f"unexpected model request: {required_keys}")


class ChatResponsePreservationTests(unittest.TestCase):
    def setUp(self):
        import main
        import review_presentation
        import utils

        self.main = main
        self.utils = utils
        self.catalog = _catalog()
        self.retrieval_calls = []
        self.text_generation = Mock(
            return_value=("MODEL OUTPUT THAT MUST NOT REPLACE RETRIEVAL", None)
        )
        self.context_builder = Mock(return_value="test facility context")

        retrieval_calls = self.retrieval_calls

        class Adapter:
            def __init__(self, **kwargs):
                pass

            def rank(self, *, eligible, query, **kwargs):
                retrieval_calls.append({
                    "eligible": tuple(eligible["place_id"].astype(str)),
                    "max_distance_km": query.max_distance_km,
                    "target_language": query.target_language,
                })
                order = {place_id: index for index, place_id in enumerate(
                    ("far", "closest", "middle", "outer")
                )}
                ranked = eligible.copy()
                ranked["_test_order"] = ranked["place_id"].map(order)
                ranked = ranked.sort_values("_test_order").drop(columns="_test_order")
                ranked = ranked.reset_index(drop=True)
                evidence = []
                groups = []
                for place_id in ranked["place_id"].astype(str):
                    text = (
                        "진료 과정을 자세히 설명해 주었습니다."
                        if query.target_language == "Korean"
                        else "The clinician explained the visit clearly."
                    )
                    item = {
                        "place_id": place_id,
                        "evidence_id": f"review:{place_id}",
                        "text": text,
                        "language": "ko" if query.target_language == "Korean" else "en",
                        "is_verbatim": True,
                        "evidence_role": "support",
                        "source_type": "verbatim_review",
                        "source_field": "comment",
                        "source_locator": {"row": place_id},
                        "review_source_sha256": "a" * 64,
                    }
                    evidence.append([item])
                    groups.append({
                        "supporting": [item],
                        "warnings": [],
                        "unverified": [],
                    })
                ranked["retrieval_evidence"] = evidence
                ranked["retrieval_evidence_groups"] = groups
                ranked["relevance_rank"] = range(1, len(ranked) + 1)
                ranked.attrs["rag_metadata"] = {
                    "retrieval_status": "complete",
                    "coverage_sufficient": True,
                }
                return SimpleNamespace(
                    dataframe=ranked,
                    telemetry=SimpleNamespace(status="complete"),
                )

        self.stack = ExitStack()
        self.stack.enter_context(patch.object(main, "df_filtered", self.catalog))
        self.stack.enter_context(
            patch.object(main, "available_specialties", ["정형외과"])
        )
        self.stack.enter_context(patch.object(
            main,
            "rag_pipeline",
            SimpleNamespace(build_context_for_llm=self.context_builder),
        ))
        self.stack.enter_context(patch.object(
            main, "search_index_release", SimpleNamespace(version="test-v1")
        ))
        self.stack.enter_context(patch.object(main, "CandidateRetrievalAdapter", Adapter))
        self.stack.enter_context(patch.object(main, "client", object()))
        self.stack.enter_context(
            patch.object(main, "request_text_completion", self.text_generation)
        )
        self.stack.enter_context(patch.object(main, "ENABLE_RETRIEVAL_DEBUG", False))
        self.stack.enter_context(patch.object(main, "GOOGLE_MAPS_API_KEY", ""))
        self.stack.enter_context(patch.object(main, "KAKAO_REST_API_KEY", ""))
        self.stack.enter_context(patch.object(
            main,
            "chat_rate_limiter",
            SlidingWindowRateLimiter(max_requests=100, window_seconds=60),
        ))
        self.translation_request = self.stack.enter_context(
            patch.object(review_presentation.requests, "post")
        )
        self.stack.enter_context(
            patch.object(main.app.router, "lifespan_context", _test_lifespan)
        )
        self.client = self.stack.enter_context(TestClient(main.app))

    def tearDown(self):
        self.stack.close()

    def _install_model_boundary(self, boundary):
        self.stack.enter_context(
            patch.object(self.main, "request_json_completion", side_effect=boundary)
        )
        self.stack.enter_context(
            patch.object(self.utils, "request_json_completion", side_effect=boundary)
        )

    @staticmethod
    def _geocoded(location):
        return {
            "lat": ORIGIN_LAT,
            "lon": ORIGIN_LON,
            "address_korean": f"서울특별시 종로구 {location}",
            "district": "종로구",
            "dong": "종로1가",
        }

    def _install_geocoder(self):
        geocoder = Mock(side_effect=lambda location, **kwargs: self._geocoded(location))
        self.stack.enter_context(
            patch.object(self.main, "verify_and_standardize_address", geocoder)
        )
        return geocoder

    def test_retrieval_reply_survives_initial_search_and_radius_refinement(self):
        boundary = JsonModelBoundary(
            ["PROVIDE_INFO", "CHANGE_CRITERIA"],
            [_extraction(location="Jonggak"), _extraction(location=None, distance_km=1.0)],
        )
        self._install_model_boundary(boundary)
        geocoder = self._install_geocoder()

        first = self.client.post("/chat", json={
            "message": "i need a orthopedic doctor next to Jonggak",
            "current_state": State().model_dump(),
        })

        self.assertEqual(first.status_code, 200, first.text)
        first_body = first.json()
        self.assertEqual(
            first_body["response"],
            "I found 4 options for orthopedics around Jonggak. Of the options "
            "shown, 광화문정형외과의원 is closest, about 0.4 km from Jonggak by "
            "straight-line distance.\n\nWhat would you like the doctor to help with? "
            "You can also narrow the search to within 1 km, or add language or "
            "access needs.",
        )
        self.assertNotIn(GENERIC_RESPONSE, first_body["response"])
        self.assertEqual(
            [card["place_id"] for card in first_body["results"]],
            ["far", "closest", "middle", "outer"],
        )
        self.assertEqual(first_body["state"]["language_pref"], "English")
        self._assert_public_cards(first_body, "English")

        refined = self.client.post("/chat", json={
            "message": "Keep it within 1 km of Jonggak, please.",
            "current_state": first_body["state"],
        })

        self.assertEqual(refined.status_code, 200, refined.text)
        refined_body = refined.json()
        self.assertEqual(
            refined_body["response"],
            "I found 3 options for orthopedics around Jonggak. Of the options "
            "shown, 광화문정형외과의원 is closest, about 0.4 km from Jonggak by "
            "straight-line distance.\n\nYou can compare the cards below or refine "
            "this search further.",
        )
        self.assertNotIn(GENERIC_RESPONSE, refined_body["response"])
        self.assertEqual(
            [card["place_id"] for card in refined_body["results"]],
            ["closest", "middle", "outer"],
        )
        self.assertEqual(refined_body["state"]["max_distance_km"], 1.0)
        self.assertEqual(refined_body["state"]["specialty"], "정형외과")
        self.assertEqual(refined_body["state"]["location"], "Jonggak")
        self._assert_public_cards(refined_body, "English")
        self.assertEqual([call["max_distance_km"] for call in self.retrieval_calls], [5.0, 1.0])
        self.assertNotIn("far", self.retrieval_calls[1]["eligible"])
        self.assertEqual(geocoder.call_count, 1)
        self.text_generation.assert_not_called()
        self.context_builder.assert_not_called()
        self.translation_request.assert_not_called()

    def test_korean_retrieval_reply_and_original_reviews_survive_serialization(self):
        boundary = JsonModelBoundary(
            ["PROVIDE_INFO"],
            [_extraction(location="종각")],
        )
        self._install_model_boundary(boundary)
        self._install_geocoder()

        response = self.client.post("/chat", json={
            "message": "종각 근처 정형외과를 찾아 주세요",
            "current_state": State().model_dump(),
        })

        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(
            body["response"],
            "종각 주변에서 정형외과 검색 결과로 시설 4곳을 찾았습니다. 표시된 후보 중 "
            "광화문정형외과의원의 직선거리가 가장 짧으며, 종각 기준 약 0.4 km입니다.\n\n"
            "어떤 진료나 증상으로 도움이 필요하신가요? 원하시면 1 km 이내처럼 이동 거리나 "
            "언어 또는 접근성 요구사항도 알려 주세요.",
        )
        self.assertNotIn("아래 카드에서 시설과 후기를 확인해 주세요", body["response"])
        self.assertEqual(body["state"]["language_pref"], "Korean")
        self._assert_public_cards(body, "Korean")
        self.text_generation.assert_not_called()
        self.context_builder.assert_not_called()
        self.translation_request.assert_not_called()

    def test_nonretrieval_generated_reply_survives_serialization(self):
        generated = "I generated this nearby comparison for the current request."
        self.text_generation.return_value = generated, None
        boundary = JsonModelBoundary(["CONFIRMATION"])
        self._install_model_boundary(boundary)
        current_state = State(
            location="Jonggak",
            latitude=ORIGIN_LAT,
            longitude=ORIGIN_LON,
            address_korean="서울특별시 종로구 종각",
            district="종로구",
            dong="종로1가",
            search_mode="distance",
            max_distance_km=5.0,
            travel_confidence=1.0,
            is_general_search=True,
            language_pref="English",
            turn_count=1,
        )

        response = self.client.post("/chat", json={
            "message": "yes",
            "current_state": current_state.model_dump(),
        })

        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["response"], generated)
        self.assertNotIn(GENERIC_RESPONSE, body["response"])
        self.assertEqual(len(body["results"]), 4)
        self.assertEqual(body["state"]["language_pref"], "English")
        self.text_generation.assert_called_once()
        self.context_builder.assert_called_once()
        self.assertEqual(self.retrieval_calls, [])
        self.translation_request.assert_not_called()

    def _assert_public_cards(self, body, language):
        expected_text = (
            "진료 과정을 자세히 설명해 주었습니다."
            if language == "Korean"
            else "The clinician explained the visit clearly."
        )
        for card in body["results"]:
            self.assertEqual(card["review_language"], language)
            self.assertEqual(card["recommendation_status"], "evidence_available")
            evidence = card["retrieval_evidence"]
            self.assertEqual(len(evidence), 1)
            self.assertEqual(evidence[0]["place_id"], card["place_id"])
            self.assertEqual(evidence[0]["evidence_id"], f"review:{card['place_id']}")
            self.assertEqual(evidence[0]["text"], expected_text)
            self.assertEqual(evidence[0]["presentation"]["status"], "original")
            supporting = card["retrieval_evidence_groups"]["supporting"]
            self.assertEqual(
                [(item["place_id"], item["evidence_id"]) for item in supporting],
                [(card["place_id"], f"review:{card['place_id']}")],
            )
            self.assertNotIn("relevance_rank", card)
        self.assertNotIn("last_retrieval_metadata", body["state"])


if __name__ == "__main__":
    unittest.main()
