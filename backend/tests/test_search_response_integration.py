"""Exercise the real search-to-response boundary without external model calls."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.tests.test_search_scope import facility
from models import State


class SearchResponseIntegrationTests(unittest.TestCase):
    def search(self, *, amenities=None, hours="", hard_keywords=()):
        import main

        row = facility("alpha", amenities=amenities)
        row["business_hours"] = hours
        frame = pd.DataFrame([row])
        review = {"place_id": "alpha", "evidence_id": "review:123",
                  "is_verbatim": True, "source_type": "verbatim_review", "retrieval_roles": ["support", "risk"],
                  "text": "The doctor was kind but the nurses were rude."}

        class Adapter:
            def __init__(self, **kwargs):
                pass

            def rank(self, *, eligible, **kwargs):
                result = eligible.copy()
                result["retrieval_evidence"] = [[review] for _ in range(len(result))]
                result["retrieval_evidence_groups"] = [
                    {"warnings": [review]} for _ in range(len(result))]
                result.attrs["rag_metadata"] = {"retrieval_execution_status": "partial"}
                return SimpleNamespace(dataframe=result,
                                       telemetry=SimpleNamespace(status="incomplete"))

        state = State(specialty="치과", specialty_confidence=0.95,
                      location=None, is_citywide_search=True,
                      hard_keywords=list(hard_keywords),
                      comment_terms=["kind"], language_pref="English")
        with (
            patch.object(main, "df_filtered", frame),
            patch.object(main, "available_specialties", ["치과"]),
            patch.object(main, "rag_pipeline", SimpleNamespace(
                build_context_for_llm=lambda *args, **kwargs: "context")),
            patch.object(main, "search_index_release", SimpleNamespace(version="test-v1")),
            patch.object(main, "CandidateRetrievalAdapter", Adapter),
            patch.object(main, "client", object()),
            patch.object(main, "request_answer_completion",
                         side_effect=TimeoutError("answer_timeout")),
        ):
            return main.execute_search(state, "Find a kind dentist in Seoul")

    def test_actual_response_withholds_incomplete_endorsement_and_quotes_conflict(self):
        response, cards = self.search()
        self.assertIn("I may have missed relevant patient reviews", response)
        self.assertNotIn("but the nurses were rude", response)
        self.assertIn("but the nurses were rude", cards[0]["retrieval_evidence"][0]["text"])
        self.assertNotIn("Highly recommended", response)
        self.assertEqual(cards[0]["recommendation_status"], "not_established")

    def test_structured_business_hours_serialize_without_array_truth_check(self):
        _, cards = self.search(hours=np.array(["Monday 09:00", "Saturday 09:00"]))
        self.assertEqual(cards[0]["business_hours"], ["Monday 09:00", "Saturday 09:00"])

    def test_unmet_amenity_requirement_returns_empty_results(self):
        _, cards = self.search(amenities={"parking": False}, hard_keywords=("parking",))
        self.assertEqual(cards, [])
