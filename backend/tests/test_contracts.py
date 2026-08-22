import sys
import unittest
from pathlib import Path

import pandas as pd


BACKEND_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

from config import DISTANCE_MAPPING, GROQ_CHAT_MODEL  # noqa: E402
from cookies import CookieConsent, get_consent_from_cookie  # noqa: E402
from models import ChatResponse, State  # noqa: E402
from agentic_retrieval import (  # noqa: E402
    build_specific_evidence_records,
    contains_exact_phrase,
    tokenize_exact,
    validate_retrieval_plan,
)
from query_facets import augment_extracted_facets, retrieval_terms_from_state  # noqa: E402


class ConfigurationContractTests(unittest.TestCase):
    def test_groq_model_is_gpt_oss_20b(self):
        self.assertEqual(GROQ_CHAT_MODEL, "openai/gpt-oss-20b")

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
            gender_terms=["female"],
            disease_terms=["endometriosis"],
            comment_terms=["clear explanations"],
        )
        terms = retrieval_terms_from_state(state)
        self.assertIn("female", terms)
        self.assertIn("여의사", terms)
        self.assertIn("endometriosis", terms)
        self.assertIn("자궁내막증", terms)
        self.assertIn("clear explanations", terms)
        self.assertIn("설명 잘", terms)

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
