"""Sequential chat regressions at the router, state and retrieval boundaries."""

from contextlib import ExitStack
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.tests import test_chat_response_preservation as fixtures
from models import State
from rate_limit import SlidingWindowRateLimiter


class ChatStateReliabilityTests(unittest.TestCase):
    def setUp(self):
        import main
        import review_presentation

        self.main = main
        self.catalog = fixtures._catalog()
        self.catalog.loc[self.catalog["place_id"] == "far", "has_english"] = False
        self.retrieval_calls = []
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(main, "df_filtered", self.catalog))
        self.stack.enter_context(patch.object(main, "available_specialties", ["정형외과", "소아청소년과"]))
        self.stack.enter_context(patch.object(main, "rag_pipeline", SimpleNamespace()))
        self.stack.enter_context(patch.object(main, "search_index_release", SimpleNamespace(version="test-v1")))
        self.stack.enter_context(patch.object(main, "client", object()))
        self.stack.enter_context(patch.object(main, "ENABLE_RETRIEVAL_DEBUG", False))
        self.stack.enter_context(patch.object(main, "GOOGLE_MAPS_API_KEY", ""))
        self.stack.enter_context(patch.object(main, "KAKAO_REST_API_KEY", ""))
        self.stack.enter_context(patch.object(main, "chat_rate_limiter", SlidingWindowRateLimiter(max_requests=100, window_seconds=60)))
        self.stack.enter_context(patch.object(review_presentation.requests, "post"))
        self.stack.enter_context(patch.object(main.app.router, "lifespan_context", fixtures._test_lifespan))
        # Answer failure is deliberate: it must not prevent validating the real
        # search route or discard the source cards available to that request.
        self.stack.enter_context(patch.object(main, "request_answer_completion", side_effect=TimeoutError("synthetic answer timeout")))
        self.geocoder = self.stack.enter_context(patch.object(main, "verify_and_standardize_address", side_effect=lambda location, **kwargs: {
            "lat": fixtures.ORIGIN_LAT, "lon": fixtures.ORIGIN_LON,
            "address_korean": f"서울특별시 종로구 {location}", "district": "종로구", "dong": "종로1가",
        }))
        calls = self.retrieval_calls

        class Adapter:
            def __init__(self, **kwargs):
                pass

            def rank(self, *, eligible, query, rules, **kwargs):
                calls.append({"eligible": tuple(eligible["place_id"].astype(str)), "query": query, "rules": rules})
                ranked = eligible.copy().reset_index(drop=True)
                evidence = []
                for place_id in ranked["place_id"].astype(str):
                    evidence.append([{
                        "place_id": place_id, "evidence_id": f"review:{place_id}",
                        "text": "The doctor explained clearly. The nurse was rude.",
                        "language": "en", "is_verbatim": True, "source_type": "verbatim_review",
                        "source_field": "comment", "source_locator": {"row": place_id},
                        "review_source_sha256": "a" * 64, "retrieval_roles": ["support", "risk"],
                    }])
                ranked["retrieval_evidence"] = evidence
                ranked["retrieval_evidence_groups"] = [{"supporting": items, "warnings": [], "unverified": []} for items in evidence]
                ranked["relevance_rank"] = range(1, len(ranked) + 1)
                ranked.attrs["rag_metadata"] = {"retrieval_status": "complete", "coverage_sufficient": True}
                return SimpleNamespace(dataframe=ranked, telemetry=SimpleNamespace(status="complete"))

        self.stack.enter_context(patch.object(main, "CandidateRetrievalAdapter", Adapter))
        self.client = self.stack.enter_context(TestClient(main.app))

    def _model(self, intents, extractions):
        boundary = fixtures.JsonModelBoundary(intents, extractions)
        self.stack.enter_context(patch.object(self.main, "request_json_completion", side_effect=boundary))
        return boundary

    def _post(self, message, state=None):
        response = self.client.post("/chat", json={
            "message": message, "current_state": state if state is not None else State().model_dump(),
        })
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertTrue(body["results"], body)
        for card in body["results"]:
            self.assertEqual(card["retrieval_evidence"][0]["place_id"], card["place_id"])
            self.assertEqual(card["retrieval_evidence"][0]["text"], "The doctor explained clearly. The nurse was rude.")
        return body

    @staticmethod
    def _proposal(**fields):
        return {**fixtures._extraction(location=None), **fields}

    def test_ichon_care_preferences_survive_four_real_chat_turns(self):
        self.catalog["category"] = "소아청소년과"
        self._model(["PROVIDE_INFO", "CHANGE_CRITERIA", "PROVIDE_INFO", "CHANGE_CRITERIA"], [
            self._proposal(specialty="소아청소년과", specialty_confidence=0.95),
            self._proposal(location="Ichon", distance_km=1),
            self._proposal(soft_keywords=["short wait"], comment_terms=["short wait"]),
            self._proposal(soft_keywords=["short wait", "kind"], comment_terms=["kind pediatric care"]),
        ])
        first = self._post("I need a pediatric clinic in Seoul for a routine checkup, clear explanations and restrained prescribing.")
        second = self._post("Narrow to within 1 km of Ichon.", first["state"])
        third = self._post("I prefer a short wait.", second["state"])
        fourth = self._post("Some waiting is fine — that's not a dealbreaker for me. Kind treatment of children matters more.", third["state"])

        self.assertEqual(first["state"]["visit_reason"], "routine checkup")
        self.assertEqual(fourth["state"]["visit_reason"], "routine checkup")
        self.assertEqual(fourth["state"]["specialty"], "소아청소년과")
        self.assertEqual(fourth["state"]["location"], "Ichon")
        self.assertEqual(fourth["state"]["max_distance_km"], 1)
        for body in (second, third, fourth):
            self.assertIn("clear explanations", body["state"]["comment_terms"])
            self.assertIn("no overprescribing", body["state"]["comment_terms"])
            self.assertNotIn("far", [card["place_id"] for card in body["results"]])
        self.assertIn("short wait", third["state"]["keywords"])
        for field in ("keywords", "hard_keywords", "negative_keywords", "negative_hard_keywords", "comment_terms"):
            self.assertFalse(any("wait" in term or "대기" in term for term in fourth["state"][field]), field)
        self.assertEqual(len(self.retrieval_calls), 4)
        self.assertIn("short wait", self.retrieval_calls[2]["query"].exact_terms)
        self.assertFalse(any("wait" in term or "대기" in term for term in self.retrieval_calls[3]["query"].exact_terms))
        self.assertNotRegex(self.retrieval_calls[3]["query"].text, r"(?i)wait|대기")
        self.assertFalse(any("wait" in preference.concept_id for preference in self.retrieval_calls[3]["rules"].soft))
        self.assertEqual([call["query"].max_distance_km for call in self.retrieval_calls[1:]], [1, 1, 1])

    def test_english_inquiry_then_mandatory_requirement_reaches_distinct_rules(self):
        closest = self.catalog.index[self.catalog["place_id"] == "closest"][0]
        self.catalog.at[closest, "medical_info_parsed"] = {"English consultation": True}
        self._model(["PROVIDE_INFO", "CHANGE_CRITERIA", "CHANGE_CRITERIA"], [
            self._proposal(specialty="정형외과", specialty_confidence=0.95, location="Jonggak", distance_km=1, disease_terms=["wrist pain"]),
            self._proposal(hard_keywords=["English-speaking"], soft_keywords=["direct"],
                           specialty="병원,의원", distance_km=5, travel_label="Moderate"),
            self._proposal(hard_keywords=["English-speaking"]),
        ])
        first = self._post("I need orthopedics within 1 km of Jonggak for wrist pain.")
        second = self._post("Can you confirm English consultations, or should I check directly?", first["state"])
        self.assertEqual(second["state"]["hard_keywords"], [])
        self.assertNotIn("direct", second["state"]["keywords"])
        self.assertTrue(second["state"]["inquiries"])
        self.assertIn("wrist pain", second["state"]["disease_terms"])
        self.assertEqual(second["state"]["specialty"], "정형외과")
        self.assertEqual(second["state"]["max_distance_km"], 1)
        self.assertFalse(self.retrieval_calls[1]["rules"].hard.required_attributes)
        third = self._post("English consultation is mandatory for me.", second["state"])
        self.assertIn("English-speaking", third["state"]["hard_keywords"])
        self.assertEqual(third["state"]["inquiries"], [])
        self.assertIn("english_consultation", [attribute.concept_id for attribute in self.retrieval_calls[2]["rules"].hard.required_attributes])

    def test_mandatory_english_with_no_matching_facility_returns_empty_scope(self):
        self._model(["PROVIDE_INFO", "CHANGE_CRITERIA"], [
            self._proposal(specialty="정형외과", specialty_confidence=0.95, location="Jonggak", distance_km=1),
            self._proposal(hard_keywords=["English-speaking"]),
        ])
        first = self._post("Find orthopedics within 1 km of Jonggak.")
        response = self.client.post("/chat", json={
            "message": "English consultation is mandatory for me.", "current_state": first["state"],
        })
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["results"], [])
        self.assertIn("English-speaking", body["state"]["hard_keywords"])
        self.assertEqual(body["state"]["max_distance_km"], 1)

    def test_answer_language_changes_preserve_candidate_eligibility(self):
        self._model(["PROVIDE_INFO", "CHANGE_CRITERIA", "CHANGE_CRITERIA"], [
            self._proposal(specialty="정형외과", specialty_confidence=0.95, location="Jonggak"),
            self._proposal(), self._proposal(hard_keywords=["English-speaking"]),
        ])
        first = self._post("Find orthopedics near Jonggak.")
        korean = self._post("Please reply in Korean.", first["state"])
        english = self._post("Please reply in English.", korean["state"])
        expected = set(self.catalog["place_id"])
        for body in (first, korean, english):
            self.assertEqual({card["place_id"] for card in body["results"]}, expected)
            self.assertEqual(body["state"]["hard_keywords"], [])
        self.assertEqual(korean["state"]["language_pref"], "Korean")
        self.assertEqual(english["state"]["language_pref"], "English")
        self.assertEqual({frozenset(call["eligible"]) for call in self.retrieval_calls}, {frozenset(expected)})

    def test_compound_refinement_applies_location_and_named_term_replacement(self):
        replacement = "Move to Ichon and replace short wait with thorough care."
        self._model(["PROVIDE_INFO", "CHANGE_CRITERIA"], [
            self._proposal(specialty="정형외과", specialty_confidence=0.95, location="Jonggak", distance_km=1,
                           place_terms=["Jonggak"], soft_keywords=["short wait"], comment_terms=["clear explanations", "short wait"]),
            self._proposal(location="Ichon", place_terms=["Ichon"], term_operations=[{
                "action": "replace", "field": "comment_terms", "term": "short wait",
                "replacement": "thorough", "source_span": replacement,
            }]),
        ])
        first = self._post("Find orthopedics within 1 km of Jonggak with clear explanations and a short wait.")
        second = self._post(replacement, first["state"])
        self.assertEqual(second["state"]["location"], "Ichon")
        self.assertEqual(second["state"]["max_distance_km"], 1)
        self.assertEqual(set(second["state"]["comment_terms"]), {"clear explanations", "thorough"})
        self.assertNotIn("short wait", second["state"]["keywords"])
        self.assertNotIn("short wait", self.retrieval_calls[1]["query"].exact_terms)
        self.assertEqual(second["state"]["place_terms"], ["Ichon"])
        self.assertNotIn("Jonggak", self.retrieval_calls[1]["query"].text)

    def test_negated_correction_still_passes_through_the_reducer(self):
        self._model(["PROVIDE_INFO", "CHANGE_CRITERIA"], [
            self._proposal(specialty="정형외과", specialty_confidence=0.95, location="Jonggak", soft_keywords=["short wait"]),
            self._proposal(),
        ])
        first = self._post("Find orthopedics near Jonggak with a short wait.")
        second = self._post("No thanks, some waiting is fine. Keep the same location.", first["state"])
        self.assertNotIn("short wait", second["state"]["keywords"])
        self.assertNotIn("short wait", second["state"]["comment_terms"])
        self.assertNotIn("short wait", self.retrieval_calls[1]["query"].exact_terms)

    def test_korean_nurse_inquiry_does_not_become_an_exclusion(self):
        self._model(["PROVIDE_INFO", "CHANGE_CRITERIA"], [
            self._proposal(specialty="정형외과", specialty_confidence=0.95, location="Jonggak"),
            self._proposal(negative_keywords=["unfriendly nurses"]),
        ])
        first = self._post("Find orthopedics near Jonggak.")
        second = self._post("한국어로 답해 주세요. 간호사가 불친절하다는 후기는 어떤 의미인가요?", first["state"])
        self.assertEqual(second["state"]["language_pref"], "Korean")
        self.assertEqual(second["state"]["negative_keywords"], [])
        self.assertTrue(second["state"]["inquiries"])
        self.assertFalse(any(preference.polarity == "negative" for preference in self.retrieval_calls[1]["rules"].soft))

    def test_router_cannot_reset_state_without_patient_reset_instruction(self):
        self._model(["PROVIDE_INFO", "NEW_SEARCH"], [
            self._proposal(specialty="정형외과", specialty_confidence=0.95, location="Jonggak", distance_km=1,
                           comment_terms=["clear explanations"]),
            self._proposal(location="Ichon"),
        ])
        first = self._post("Find orthopedics within 1 km of Jonggak with clear explanations.")
        second = self._post("Move the search to Ichon.", first["state"])
        self.assertEqual(second["state"]["specialty"], "정형외과")
        self.assertEqual(second["state"]["location"], "Ichon")
        self.assertEqual(second["state"]["max_distance_km"], 1)
        self.assertIn("clear explanations", second["state"]["comment_terms"])

    def test_wait_indifference_is_not_a_citywide_geographic_instruction(self):
        messages = ("Waiting doesn't matter to me.", "대기는 상관없어요.")
        self._model(["PROVIDE_INFO", "CHANGE_CRITERIA"] * len(messages), [
            proposal
            for _ in messages
            for proposal in (
                self._proposal(specialty="정형외과", specialty_confidence=0.95,
                               location="Ichon", distance_km=1, soft_keywords=["short wait"]),
                self._proposal(location="Seoul", is_citywide_search=True, travel_label="Anywhere in Seoul"),
            )
        ])
        for message in messages:
            with self.subTest(message=message):
                first = self._post("Find orthopedics within 1 km of Ichon with a short wait.")
                second = self._post(message, first["state"])
                self.assertEqual(second["state"]["location"], "Ichon")
                self.assertEqual(second["state"]["max_distance_km"], 1)
                self.assertFalse(second["state"]["is_citywide_search"])
                self.assertNotIn("short wait", second["state"]["keywords"])
                self.assertNotIn("short wait", second["state"]["comment_terms"])
                self.assertNotIn("short wait", self.retrieval_calls[-1]["query"].exact_terms)
                self.assertNotRegex(self.retrieval_calls[-1]["query"].text, r"(?i)wait|대기")
                self.assertEqual(self.retrieval_calls[-1]["query"].max_distance_km, 1)

    def test_geocoding_failure_retains_requested_scope_without_broadening(self):
        self._model(["PROVIDE_INFO", "CHANGE_CRITERIA"], [
            self._proposal(specialty="정형외과", specialty_confidence=0.95, location="Jonggak", distance_km=1),
            self._proposal(location="Ichon"),
        ])
        first = self._post("Find orthopedics within 1 km of Jonggak.")
        self.geocoder.side_effect = None
        self.geocoder.return_value = None
        with patch("utils.verify_and_standardize_address", return_value=None):
            response = self.client.post("/chat", json={
                "message": "Move the search to Ichon.", "current_state": first["state"],
            })
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["results"], [])
        self.assertEqual(body["state"]["location"], "Ichon")
        self.assertEqual(body["state"]["max_distance_km"], 1)
        self.assertFalse(body["state"]["is_citywide_search"])
        self.assertIsNone(body["state"]["latitude"])
        self.assertEqual(len(self.retrieval_calls), 1)

    def test_confirmation_route_cannot_skip_a_compound_preference_refinement(self):
        self._model(["PROVIDE_INFO", "CONFIRMATION"], [
            self._proposal(specialty="정형외과", specialty_confidence=0.95,
                           location="Ichon", distance_km=1, soft_keywords=["short wait"],
                           comment_terms=["clear explanations"]),
            self._proposal(),
        ])
        first = self._post("Find orthopedics within 1 km of Ichon with clear explanations and a short wait.")
        second = self._post("Yes, some waiting is fine; keep Ichon within 1 km.", first["state"])
        self.assertEqual(second["state"]["specialty"], "정형외과")
        self.assertEqual(second["state"]["location"], "Ichon")
        self.assertEqual(second["state"]["max_distance_km"], 1)
        self.assertIn("clear explanations", second["state"]["comment_terms"])
        self.assertNotIn("short wait", second["state"]["keywords"])
        self.assertNotRegex(self.retrieval_calls[-1]["query"].text, r"(?i)wait|대기")

    def test_explicit_new_search_resets_the_patient_context(self):
        self._model(["PROVIDE_INFO", "NEW_SEARCH"], [
            self._proposal(specialty="정형외과", specialty_confidence=0.95, location="Ichon",
                           comment_terms=["clear explanations"]),
        ])
        first = self._post("Find orthopedics near Ichon with clear explanations.")
        response = self.client.post("/chat", json={"message": "new search", "current_state": first["state"]})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["state"]["comment_terms"], [])
        self.assertIsNone(response.json()["state"]["specialty"])


if __name__ == "__main__":
    unittest.main()
