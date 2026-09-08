"""The source-aware answer path replaces obsolete context and text generation."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.tests.test_search_scope import facility
from models import State


class ResponseGenerationBoundaryTests(unittest.TestCase):
    def test_complete_retrieval_path_uses_verified_answer_once(self):
        import main

        frame = pd.DataFrame([facility("alpha", category="치과")])

        class Adapter:
            def __init__(self, **kwargs):
                pass

            def rank(self, *, eligible, **kwargs):
                ranked = eligible.copy()
                ranked["retrieval_evidence"] = [[] for _ in range(len(ranked))]
                ranked["retrieval_evidence_groups"] = [
                    {"supporting": [], "warnings": [], "unverified": []}
                    for _ in range(len(ranked))
                ]
                ranked.attrs["rag_metadata"] = {
                    "retrieval_status": "complete",
                    "coverage_sufficient": True,
                }
                return SimpleNamespace(
                    dataframe=ranked,
                    telemetry=SimpleNamespace(status="complete"),
                )

        context_builder = Mock(return_value="unused context")
        accepted = "This clinic is a search candidate. Ask it about appointment availability."
        final_generation = Mock(side_effect=[
            ({"answer": accepted, "assessments": [], "citations": []}, None),
            ({"accepted": True, "issues": []}, None),
        ])
        state = State(
            specialty="치과",
            specialty_confidence=0.95,
            location=None,
            is_citywide_search=True,
            language_pref="English",
            turn_count=1,
        )

        with (
            patch.object(main, "df_filtered", frame),
            patch.object(main, "available_specialties", ["치과"]),
            patch.object(
                main,
                "rag_pipeline",
                SimpleNamespace(build_context_for_llm=context_builder),
            ),
            patch.object(
                main,
                "search_index_release",
                SimpleNamespace(version="test-v1"),
            ),
            patch.object(main, "CandidateRetrievalAdapter", Adapter),
            patch.object(main, "client", object()),
            patch.object(main, "request_answer_completion", final_generation),
        ):
            response, cards = main.execute_search(
                state,
                "Find a dentist anywhere in Seoul",
            )

        context_builder.assert_not_called()
        self.assertEqual(final_generation.call_count, 2)
        self.assertEqual(response, accepted)
        self.assertEqual([card["place_id"] for card in cards], ["alpha"])


if __name__ == "__main__":
    unittest.main()
