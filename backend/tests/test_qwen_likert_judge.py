import json
import sys
import types
import unittest
from pathlib import Path


TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

# The judge imports the production client builder, which imports SDKs these
# pure parsing and projection tests do not exercise.
sys.modules.setdefault("groq", types.SimpleNamespace(Groq=object))
sys.modules.setdefault("openai", types.SimpleNamespace(OpenAI=object))
sys.modules.setdefault(
    "dotenv", types.SimpleNamespace(load_dotenv=lambda *args, **kwargs: None)
)

from qwen_likert_judge import (  # noqa: E402
    _evaluation_policy,
    parse_rating,
    project_journey_for_review,
    scenario_passes,
    weighted_mean,
)


POLICY = {
    "weights": {
        "intent_state": 0.25,
        "refinement": 0.20,
        "tool_orchestration": 0.20,
        "evidence_ranking": 0.20,
        "geography_transport": 0.10,
        "language_medical_safety": 0.05,
    },
    "anchors": {str(score): f"Anchor {score}" for score in range(1, 6)},
    "scenario_pass": {
        "all_hard_gates": True,
        "minimum_weighted_mean": 4.0,
        "minimum_dimension": 3,
    },
}


class QwenLikertJudgeTests(unittest.TestCase):
    def test_review_projection_keeps_target_and_bounded_audit_evidence(self):
        candidates = [
            {"place_id": str(index), "retrieval_rank_1based": index}
            for index in range(1, 26)
        ]
        journey = {
            "language": "English",
            "turns": [{
                "request": {
                    "message": "Bearer abcdefghijklmnop",
                    "current_state": {"bulk": True},
                },
                "response": {
                    "status_code": 200,
                    "body": {
                        "response": "Here are facilities",
                        "state": {
                            "specialty": "피부과",
                            "last_retrieval_candidates": candidates,
                            "last_retrieval_trace": [{
                                "action": "search_multilingual_comments",
                                "arguments": {"query_terms": ["gentle"]},
                                "result_count": 2,
                            }],
                            "last_retrieval_observations": [{
                                "tool": "search_multilingual_comments",
                                "result_count": 2,
                                "error": "Raw review store is unavailable.",
                                "comments": ["bulk payload"],
                            }],
                            "last_retrieval_metadata": {
                                "run_id": "run-1",
                                "retrieval_status": "error",
                                "termination_reason": "agent_error",
                                "unrelated": "bulk metadata",
                            },
                        },
                        "results": [{
                            "place_id": "25",
                            "name": "Clinic",
                            "retrieval_evidence": [{
                                "evidence_id": "review:25",
                                "text": "The doctor explains clearly.",
                                "source_type": "verbatim_review",
                            }],
                            "Summaries": ["bulk summary"],
                        }],
                    },
                },
            }],
        }

        projected = project_journey_for_review(journey, "25")
        body = projected["turns"][0]["response"]["body"]
        projected_ids = {
            item["place_id"]
            for item in body["state"]["last_retrieval_candidates"]
        }

        self.assertIn("25", projected_ids)
        self.assertLessEqual(len(projected_ids), 21)
        self.assertNotIn("current_state", projected["turns"][0]["request"])
        self.assertEqual(
            projected["turns"][0]["request"]["message"], "[REDACTED]"
        )
        self.assertEqual(
            body["state"]["last_retrieval_observations"][0]["error"],
            "Raw review store is unavailable.",
        )
        self.assertNotIn(
            "comments",
            body["state"]["last_retrieval_observations"][0],
        )
        self.assertEqual(
            body["state"]["last_retrieval_trace"][0]["action"],
            "search_multilingual_comments",
        )
        self.assertEqual(
            body["state"]["last_retrieval_metadata"]["retrieval_status"],
            "error",
        )
        self.assertNotIn("Summaries", body["results"][0])
        self.assertIn(
            "path:turns[0].response.body.state.last_retrieval_observations[0]",
            projected["citation_catalog"]["paths"],
        )
        self.assertEqual(
            projected["citation_catalog"]["evidence_ids"]["evidence:review:25"],
            "turns[0].response.body.results[0].retrieval_evidence[0]",
        )

    def test_rating_requires_every_bounded_dimension(self):
        rating = {
            "scores": {
                "intent_state": 5,
                "refinement": 4,
                "tool_orchestration": 4,
                "evidence_ranking": 3,
                "geography_transport": 5,
                "language_medical_safety": 4,
            },
            "citations": {
                "intent_state": ["turn 2 state"],
                "refinement": ["turn 2 request"],
                "tool_orchestration": ["turn 2 trace"],
                "evidence_ranking": ["turn 2 result 1"],
                "geography_transport": ["distance gate"],
                "language_medical_safety": ["turn 2 answer"],
            },
            "rationale": "Mostly grounded.",
            "recommended_fixes": ["Improve review evidence."],
        }

        rating["citations"] = {
            dimension: [f"evidence:{dimension}"]
            for dimension in POLICY["weights"]
        }
        catalog = {
            "paths": [],
            "evidence_ids": {
                citation: "test.path"
                for citations in rating["citations"].values()
                for citation in citations
            },
        }

        parsed = parse_rating(json.dumps(rating), POLICY, catalog)

        self.assertEqual(parsed["scores"]["intent_state"], 5)
        self.assertAlmostEqual(weighted_mean(parsed["scores"], POLICY), 4.15)

    def test_rating_rejects_empty_or_unresolvable_dimension_citations(self):
        rating = {
            "scores": {dimension: 4 for dimension in POLICY["weights"]},
            "citations": {
                dimension: ["path:turns[0].response.body.response"]
                for dimension in POLICY["weights"]
            },
            "rationale": "Grounded.",
            "recommended_fixes": [],
        }
        catalog = {
            "paths": ["path:turns[0].response.body.response"],
            "evidence_ids": {},
        }
        rating["citations"]["intent_state"] = []
        with self.assertRaisesRegex(ValueError, "intent_state citations"):
            parse_rating(json.dumps(rating), POLICY, catalog)

        rating["citations"]["intent_state"] = ["turn 1 answer"]
        with self.assertRaisesRegex(ValueError, "does not resolve"):
            parse_rating(json.dumps(rating), POLICY, catalog)

    def test_policy_controls_weighted_mean_and_pass_thresholds(self):
        policy = {
            **POLICY,
            "weights": {
                "intent_state": 0.9,
                "refinement": 0.02,
                "tool_orchestration": 0.02,
                "evidence_ranking": 0.02,
                "geography_transport": 0.02,
                "language_medical_safety": 0.02,
            },
            "scenario_pass": {
                "all_hard_gates": False,
                "minimum_weighted_mean": 4.5,
                "minimum_dimension": 1,
            },
        }
        scores = {dimension: 5 for dimension in policy["weights"]}
        scores["intent_state"] = 4

        self.assertAlmostEqual(weighted_mean(scores, policy), 4.1)
        self.assertFalse(scenario_passes(scores, False, policy))

    def test_selected_scenario_policy_overrides_casebook_policy(self):
        scenario_policy = {
            **POLICY,
            "scenario_pass": {
                "all_hard_gates": False,
                "minimum_weighted_mean": 4.5,
                "minimum_dimension": 4,
            },
        }

        self.assertIs(
            _evaluation_policy(
                {"likert_policy": POLICY}, {"likert_policy": scenario_policy}
            ),
            scenario_policy,
        )

    def test_out_of_range_rating_is_rejected(self):
        payload = {
            "scores": {
                "intent_state": 6,
                "refinement": 4,
                "tool_orchestration": 4,
                "evidence_ranking": 4,
                "geography_transport": 4,
                "language_medical_safety": 4,
            },
            "citations": {},
            "rationale": "Invalid.",
            "recommended_fixes": [],
        }

        with self.assertRaisesRegex(ValueError, "intent_state"):
            parse_rating(
                json.dumps(payload), POLICY, {"paths": [], "evidence_ids": {}}
            )


if __name__ == "__main__":
    unittest.main()
