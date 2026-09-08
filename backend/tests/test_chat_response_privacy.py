"""Focused checks for the public `/chat` response boundary."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from models import (  # noqa: E402
    PRIVATE_CHAT_STATE_FIELDS,
    PRIVATE_RESULT_FIELDS,
    PUBLIC_EVIDENCE_FIELDS,
    PUBLIC_RESULT_FIELDS,
    State,
    serialize_results_for_chat,
    serialize_state_for_chat,
)


class ChatResponsePrivacyTests(unittest.TestCase):
    def test_public_mode_hides_state_telemetry_and_keeps_patient_state(self):
        state = State(
            specialty="치과",
            district="마포구",
            last_search_query="internal-query-sentinel",
            last_results_count=5,
            last_search_timestamp="2026-09-02T10:00:00",
            last_retrieval_trace=[{"tool_arguments": "trace-sentinel"}],
            last_retrieval_observations=[{"candidate_stack": "observation-sentinel"}],
            last_retrieval_metadata={"planner": "metadata-sentinel"},
            last_retrieval_candidates=[{"place_id": "candidate-sentinel"}],
            last_retrieval_run_id="run-sentinel",
        )

        serialized = serialize_state_for_chat(state, include_debug=False)

        self.assertEqual(serialized["specialty"], "치과")
        self.assertEqual(serialized["district"], "마포구")
        self.assertTrue(PRIVATE_CHAT_STATE_FIELDS.isdisjoint(serialized))
        for sentinel in (
            "internal-query-sentinel",
            "trace-sentinel",
            "observation-sentinel",
            "metadata-sentinel",
            "candidate-sentinel",
            "run-sentinel",
        ):
            self.assertNotIn(sentinel, str(serialized))

    def test_public_mode_keeps_patient_facility_data_without_retrieval_telemetry(self):
        results = [{
            "place_id": "patient-ui-id",
            "name": "Mapo Dental",
            "category": "치과",
            "address": "Mapo-gu, Seoul",
            "phone": "02-1234-5678",
            "business_hours": "Mon-Sat 09:00-18:00",
            "district": "마포구",
            "dong": "서교동",
            "lat": 37.556,
            "lon": 126.923,
            "Summaries": ["Patient-facing summary"],
            "Summaries_Korean": ["환자용 요약"],
            "Key_Highlights": ["Saturday hours"],
            "amenities": ["parking"],
            "medical_info_parsed": {"services": ["cleaning"]},
            "has_english": True,
            "distance": 1.2,
            "distance_km": 1.2,
            "entity_type": "facility",
            "final_rank": 1,
            "is_emergency": True,
            "english_confidence_score": 4.5,
            "website": "https://example.test/mapo-dental",
            "retrieval_methods": ["facility_bm25"],
            "retrieval_matched_terms": ["internal-match-sentinel"],
            "retrieval_trace": [{"tool_arguments": "trace-sentinel"}],
            "relevance_rank": 1,
            "combined_score": 0.99,
            "planner_tool_arguments": "planner-sentinel",
            "retrieval_evidence": [{
                "evidence_id": "review-public-id",
                "place_id": "patient-ui-id",
                "text": "The explanation was clear.",
                "translated_text": "The explanation was clear.",
                "relevance_reason": "Mentions explanations.",
                "visit_date": "2026-01-10",
                "is_verbatim": True,
                "source_type": "verbatim_review",
                "source_field": "original_text",
                "language": "English",
                "matched_terms": ["internal-match-sentinel"],
            }],
        }]

        serialized = serialize_results_for_chat(results, include_debug=False)

        self.assertEqual(serialized[0]["place_id"], "patient-ui-id")
        self.assertEqual(serialized[0]["name"], "Mapo Dental")
        self.assertEqual(serialized[0]["phone"], "02-1234-5678")
        self.assertEqual(serialized[0]["business_hours"], "Mon-Sat 09:00-18:00")
        self.assertEqual(serialized[0]["distance_km"], 1.2)
        self.assertEqual(serialized[0]["final_rank"], 1)
        self.assertTrue(serialized[0]["is_emergency"])
        for field in (
            "district",
            "dong",
            "lat",
            "lon",
            "Summaries",
            "Summaries_Korean",
            "Key_Highlights",
            "amenities",
            "medical_info_parsed",
            "has_english",
            "entity_type",
            "english_confidence_score",
            "website",
        ):
            self.assertIn(field, serialized[0])
        self.assertTrue(set(serialized[0]).issubset(PUBLIC_RESULT_FIELDS))
        self.assertTrue(PRIVATE_RESULT_FIELDS.isdisjoint(serialized[0]))
        self.assertEqual(
            set(serialized[0]["retrieval_evidence"][0]),
            {
                "evidence_id",
                "place_id",
                "text",
                "translated_text",
                "relevance_reason",
                "visit_date",
                "is_verbatim",
                "source_type",
                "source_field",
                "language",
            },
        )
        self.assertTrue(
            set(serialized[0]["retrieval_evidence"][0]).issubset(
                PUBLIC_EVIDENCE_FIELDS
            )
        )
        for sentinel in (
            "internal-match-sentinel",
            "trace-sentinel",
            "planner-sentinel",
        ):
            self.assertNotIn(sentinel, str(serialized))

    def test_travel_preference_endpoint_uses_the_same_boundary(self):
        import asyncio
        import main
        from unittest.mock import patch
        state = State(keywords=["clear explanations"],
                      last_retrieval_metadata={"planner": "private-sentinel"})
        with patch.object(main, "ENABLE_RETRIEVAL_DEBUG", False):
            body = asyncio.run(main.set_travel_preference(
                {"travel_label": "Nearby", "current_state": state.model_dump()},
                cookieConsent=None, consent_header=None,
            ))
        self.assertEqual(body["state"]["max_distance_km"], 1.0)
        self.assertEqual(body["state"]["keywords"], ["clear explanations"])
        self.assertNotIn("last_retrieval_metadata", body["state"])
        self.assertNotIn("private-sentinel", str(body))

    def test_debug_mode_preserves_existing_observability(self):
        state = State(
            last_retrieval_trace=[{"tool_arguments": "trace-sentinel"}],
            last_retrieval_candidates=[{"place_id": "candidate-sentinel"}],
        )
        results = [{
            "retrieval_trace": [{"tool_arguments": "trace-sentinel"}],
            "retrieval_methods": ["facility_bm25"],
            "retrieval_evidence": [{"evidence_id": "sealed-evidence-sentinel"}],
        }]

        self.assertEqual(
            serialize_state_for_chat(state, include_debug=True), state.model_dump()
        )
        self.assertIs(serialize_results_for_chat(results, include_debug=True), results)


if __name__ == "__main__":
    unittest.main()
