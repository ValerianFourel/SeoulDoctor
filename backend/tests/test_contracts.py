import sys
import unittest
from pathlib import Path

import pandas as pd
from pydantic import ValidationError


BACKEND_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

from config import (  # noqa: E402
    DISTANCE_MAPPING, GROQ_CHAT_MODEL, LLM_PROVIDER,
)
from cookies import CookieConsent, get_consent_from_cookie  # noqa: E402
from models import ChatRequest, ChatResponse, State  # noqa: E402
from agentic_retrieval import (  # noqa: E402
    build_specific_evidence_records,
    contains_exact_phrase,
    tokenize_exact,
    validate_retrieval_plan,
)
from query_facets import (  # noqa: E402
    augment_extracted_facets,
    may_relax_distance_constraint,
    retrieval_terms_from_state,
)
from raw_review_store import detect_language_hint  # noqa: E402


class ConfigurationContractTests(unittest.TestCase):
    def test_configured_model_matches_provider(self):
        if LLM_PROVIDER == "openrouter":
            self.assertEqual(GROQ_CHAT_MODEL, "openai/gpt-oss-120b")
        else:
            self.assertEqual(GROQ_CHAT_MODEL, "openai/gpt-oss-120b")

    def test_default_travel_preference_is_valid(self):
        state = State()
        self.assertIn(state.travel_label, DISTANCE_MAPPING)
        self.assertEqual(state.max_distance_km, DISTANCE_MAPPING[state.travel_label])

    def test_supported_travel_distances_are_ordered(self):
        self.assertEqual(
            list(DISTANCE_MAPPING.values()),
            [0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 25.0],
        )

    def test_frontend_uses_backend_travel_labels(self):
        chat_source = (
            ROOT_DIR / "frontend" / "components" / "ChatInterface.tsx"
        ).read_text()
        for label in DISTANCE_MAPPING:
            self.assertIn(f'label: "{label}"', chat_source)

    def test_legacy_chat_model_is_not_referenced(self):
        for source_path in BACKEND_DIR.glob("*.py"):
            self.assertNotIn("llama-3.1-8b-instant", source_path.read_text())

    def test_chat_response_results_are_not_shared(self):
        first = ChatResponse(response="one", state=State())
        second = ChatResponse(response="two", state=State())
        first.results.append({"place_id": "one"})
        self.assertEqual(second.results, [])


class RequestBoundaryTests(unittest.TestCase):
    def test_chat_endpoint_runs_blocking_work_in_fastapi_threadpool(self):
        source = (BACKEND_DIR / "main.py").read_text()
        self.assertIn("def chat_endpoint(", source)
        self.assertNotIn("async def chat_endpoint(", source)

    def test_chat_message_rejects_empty_or_oversized_input(self):
        for message in ("", "   ", "x" * 4_001):
            with self.subTest(length=len(message)):
                with self.assertRaises(ValidationError):
                    ChatRequest(message=message, current_state=State())

        accepted = ChatRequest(message=" x ", current_state=State())
        self.assertEqual(accepted.message, "x")

    def test_client_retrieval_telemetry_can_be_cleared_at_the_server_boundary(self):
        state = State(
            last_retrieval_trace=[{"action": "spoofed"}],
            last_retrieval_observations=[{"tool": "spoofed"}],
            last_retrieval_metadata={"retrieval_status": "complete"},
            last_retrieval_candidates=[{"place_id": "spoofed"}],
            last_retrieval_run_id="spoofed",
        )

        state.clear_retrieval_telemetry()

        self.assertEqual(state.last_retrieval_trace, [])
        self.assertEqual(state.last_retrieval_observations, [])
        self.assertEqual(state.last_retrieval_metadata, {})
        self.assertEqual(state.last_retrieval_candidates, [])
        self.assertIsNone(state.last_retrieval_run_id)


class ConsentContractTests(unittest.TestCase):
    def test_missing_consent_is_private_by_default(self):
        consent = get_consent_from_cookie(None)
        self.assertEqual(consent, CookieConsent())

    def test_json_consent_round_trip(self):
        consent = get_consent_from_cookie(
            '{"necessary": true, "analytics": true, "advertising": false}'
        )
        self.assertTrue(consent.analytics)
        self.assertFalse(consent.advertising)


class AgenticRetrievalContractTests(unittest.TestCase):
    def test_omitted_distance_cannot_overwrite_an_existing_radius(self):
        extracted = augment_extracted_facets(
            "Require detailed explanations and avoid aggressive upselling.",
            {"travel_label": "Moderate"},
        )
        self.assertIsNone(extracted["travel_label"])

    def test_refinement_can_explicitly_preserve_the_existing_radius(self):
        for query in (
            "Refine the same nearby search rather than broadening it.",
            "범위를 넓히지 말고 같은 주변 검색을 더 정교하게 해 주세요.",
        ):
            with self.subTest(query=query):
                extracted = augment_extracted_facets(
                    query,
                    {"travel_label": "Nearby"},
                )
                self.assertIsNone(extracted["travel_label"])

    def test_generic_facility_noun_cannot_replace_an_existing_specialty(self):
        for query, model_specialty in (
            ("Which clinic best matches and why?", "clinic"),
            ("어느 병원이 가장 잘 맞는지 알려 주세요.", "병원"),
        ):
            with self.subTest(query=query):
                extracted = augment_extracted_facets(
                    query,
                    {"specialty": model_specialty},
                )
                self.assertIsNone(extracted["specialty"])

    def test_numeric_distance_is_recovered_in_english_and_korean(self):
        for query in (
            "Find a dermatologist within 2 km",
            "대흥역에서 2km 이내 피부과",
        ):
            with self.subTest(query=query):
                extracted = augment_extracted_facets(query, {})
                self.assertEqual(extracted["travel_label"], "Close")

        walking = augment_extracted_facets(
            "Find a dentist within 500 metres",
            {},
        )
        self.assertEqual(walking["travel_label"], "Walking Distance")

    def test_explicit_distance_is_never_relaxed_to_fill_results(self):
        self.assertTrue(may_relax_distance_constraint(0.5))
        self.assertFalse(may_relax_distance_constraint(0.6))
        self.assertFalse(may_relax_distance_constraint(1.0))

    def test_bilingual_facets_survive_an_empty_llm_payload(self):
        extracted = augment_extracted_facets(
            "강남에서 자궁내막증을 진료하는 여성 산부인과 의사",
            {},
        )
        self.assertEqual(extracted["specialty"], "산부인과")
        self.assertEqual(extracted["location"], "강남구")
        self.assertIn("female", extracted["gender_terms"])
        self.assertIn("endometriosis", extracted["disease_terms"])

    def test_retrieval_terms_include_korean_and_english_aliases(self):
        state = State(
            hard_keywords=["fast treatment"],
            gender_terms=["female"],
            disease_terms=["endometriosis", "cheilitis"],
            comment_terms=["clear explanations"],
        )
        terms = retrieval_terms_from_state(state)
        self.assertIn("female", terms)
        self.assertIn("여의사", terms)
        self.assertIn("endometriosis", terms)
        self.assertIn("자궁내막증", terms)
        self.assertIn("clear explanations", terms)
        self.assertIn("설명 잘", terms)
        self.assertIn("cheilitis", terms)
        self.assertIn("구순염", terms)
        self.assertIn("자세한 설명", terms)
        self.assertIn("fast treatment", terms)
        self.assertIn("빠른 진료", terms)

    def test_exact_token_matching_does_not_use_substrings(self):
        document_tokens = tokenize_exact("The consultation felt unprofessional.")
        self.assertFalse(contains_exact_phrase(document_tokens, "professional"))
        self.assertTrue(contains_exact_phrase(document_tokens, "unprofessional"))

    def test_specific_index_uses_individual_evidence_chunks(self):
        facilities = pd.DataFrame([{
            "place_id": "clinic-1",
            "Summaries": ["Friendly staff.", "Parking is available."],
            "Summaries_Korean": ["직원이 친절합니다."],
            "Key_Highlights": [{"topic": "Short wait times", "percentage": 42}],
            "review_comments": [{"text": "The nurse explained every step.", "language": "English"}],
            "amenities": {"parking": True, "elevator": False},
            "medical_info_parsed": {"equipment": ["MRI"]},
        }])

        records = build_specific_evidence_records(facilities)
        texts = [record["text"] for record in records]
        self.assertIn("Friendly staff.", texts)
        self.assertIn("Parking is available.", texts)
        self.assertTrue(any("Short wait times" in text for text in texts))
        self.assertTrue(any("parking" in text for text in texts))
        self.assertTrue(any("MRI" in text for text in texts))
        self.assertTrue(all(record["level"] == "specific" for record in records))
        raw_review = next(
            record for record in records if record["source_type"] == "verbatim_review"
        )
        self.assertEqual(raw_review["text"], "The nurse explained every step.")
        self.assertTrue(raw_review["is_verbatim"])
        self.assertTrue(all(
            record["is_verbatim"] is False
            for record in records
            if record["source_type"] != "verbatim_review"
        ))

    def test_review_language_hints_preserve_mixed_scripts(self):
        self.assertEqual(detect_language_hint("친절합니다"), "Korean")
        self.assertEqual(detect_language_hint("친절"), "Korean")
        self.assertEqual(
            detect_language_hint("MRI 검사 친절해요"),
            "Korean + Latin (mixed)",
        )
        self.assertEqual(detect_language_hint("Good dentist"), "Latin-script")
        self.assertEqual(detect_language_hint("😊"), "Unknown/other")

    def test_required_terms_force_a_search_before_finish(self):
        plan = validate_retrieval_plan(
            {"action": "finish", "quote_evidence": True},
            fallback_query="dentist with parking",
            required_exact_terms=["parking"],
            has_observations=False,
        )
        self.assertEqual(plan.action, "hybrid")
        self.assertEqual(plan.exact_terms, ["parking"])
        self.assertTrue(plan.quote_evidence)


if __name__ == "__main__":
    unittest.main()
