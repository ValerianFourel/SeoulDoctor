import json
import sys
import unittest
from pathlib import Path


TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from grounded_journey_grader import (  # noqa: E402
    _matches_language,
    _load_transit_observation,
    grade_pair,
    grade_scenario,
    haversine_km,
    parse_kakao_transit,
)


def scenario():
    return {
        "id": "reverse-one-en",
        "pair_id": "reverse-one",
        "counterpart_id": "reverse-one-ko",
        "source_language": "English",
        "oracle": {
            "expected_turn_count": 1,
            "expected_response_languages": ["English"],
            "origin": {"latitude": 37.5476479, "longitude": 126.9424732},
            "maximum_distance_km": 2.0,
            "strict_radius": True,
            "expected_specialty": "피부과",
            "required_trace_actions": [
                "search_facilities",
                "search_indexed_evidence",
                "search_multilingual_comments",
                "select_comment_evidence",
                "assess_search_coverage",
                "finish_search",
            ],
            "required_methods": [
                "facility_semantic",
                "facility_bm25",
                "bm25_specific",
                "multilingual_comment_search",
            ],
            "negative_terms": ["aggressive upselling", "과도한 시술 권유"],
            "excluded_negative_terms": ["long wait", "긴 대기"],
            "reverse_target": {
                "place_id": "target",
                "name": "Target Clinic",
                "latitude": 37.5563579,
                "longitude": 126.9443618,
                "retrieval_rank_lte": 5,
                "presented_rank_lte": 5,
                "evidence_requirements": {
                    "decisive": [{
                        "evidence_id": "review:target",
                        "supports": ["detailed explanation"],
                    }],
                    "contextual": [{
                        "evidence_id": "review:context",
                        "supports": ["wait-time context"],
                    }],
                },
            },
        },
    }


def artifact(status="complete"):
    facility_lat = 37.5563579
    facility_lon = 126.9443618
    distance = haversine_km(
        37.5476479,
        126.9424732,
        facility_lat,
        facility_lon,
    )
    body = {
        "response": "I found a dermatologist with grounded comments.",
        "state": {
            "specialty": "피부과",
            "latitude": 37.5476479,
            "longitude": 126.9424732,
            "max_distance_km": 2.0,
            "negative_keywords": ["과도한 시술 권유"],
            "negative_hard_keywords": [],
            "keywords": ["구순염", "자세한 설명"],
            "hard_keywords": [],
            "comment_terms": ["빠른 진료"],
            "last_retrieval_run_id": "run-1",
            "last_retrieval_metadata": {
                "run_id": "run-1",
                "retrieval_status": status,
                "coverage_assessed": status == "complete",
                "coverage_sufficient": status == "complete",
                "termination_reason": (
                    "finish_search" if status == "complete" else "iteration_limit"
                ),
            },
            "last_retrieval_trace": [
                {
                    "action": action,
                    **(
                        {"result_count": 1}
                        if action in {
                            "search_facilities",
                            "search_indexed_evidence",
                            "search_multilingual_comments",
                            "select_comment_evidence",
                        }
                        else {}
                    ),
                }
                for action in scenario()["oracle"]["required_trace_actions"]
            ],
            "last_retrieval_candidates": [{
                "place_id": "target",
                "retrieval_rank_1based": 1,
                "combined_rank_1based": 1,
            }],
        },
        "results": [{
            "entity_type": "facility",
            "final_rank": 1,
            "place_id": "target",
            "name": "Target Clinic",
            "lat": facility_lat,
            "lon": facility_lon,
            "distance_km": distance,
            "retrieval_methods": scenario()["oracle"]["required_methods"],
            "retrieval_evidence": [{
                "evidence_id": "review:target",
                "place_id": "target",
                "source_type": "verbatim_review",
                "is_verbatim": True,
                "text": "The doctor gave a detailed explanation.",
            }],
        }],
    }
    return {
        "status": "finished",
        "language": "English",
        "finished_at": "2026-09-02T12:00:00+00:00",
        "finish_reason": "The staged request was completed.",
        "health": {
            "status": "ok",
            "model_provider": "openrouter",
            "model": "openai/gpt-oss-120b",
            "agent_model": "openai/gpt-oss-120b",
            "facilities": 8_484,
            "vector_documents": 8_484,
            "raw_reviews": 1_791_749,
        },
        "turns": [{
            "request": {"message": "find a dermatologist"},
            "response": {"status_code": 200, "body": body},
        }],
    }


class GroundedJourneyGraderTests(unittest.TestCase):
    def test_language_gate_requires_a_real_script_majority(self):
        self.assertTrue(_matches_language("근거가 있는 피부과를 찾았습니다.", "Korean"))
        self.assertTrue(_matches_language("I found a grounded clinic.", "English"))
        self.assertFalse(_matches_language("네, this answer stays English.", "Korean"))
        self.assertFalse(_matches_language("서울 피부과 OK", "English"))
        self.assertTrue(_matches_language("근거있음", "Korean"))

    def test_independent_haversine_is_symmetric(self):
        forward = haversine_km(37.5476479, 126.9424732, 37.5563579, 126.9443618)
        reverse = haversine_km(37.5563579, 126.9443618, 37.5476479, 126.9424732)

        self.assertAlmostEqual(forward, reverse, places=12)
        self.assertAlmostEqual(forward, 0.9827, places=3)

    def test_complete_grounded_target_passes_deterministic_gates(self):
        report = grade_scenario(scenario(), artifact())

        self.assertTrue(report["hard_pass"])
        self.assertTrue(all(gate["passed"] for gate in report["gates"]))
        self.assertEqual(report["target"]["retrieval_rank"], 1)
        self.assertEqual(report["target"]["presented_rank"], 1)
        self.assertEqual(
            report["target"]["missing_contextual_evidence_ids"],
            ["review:context"],
        )

    def test_unfinished_or_incomplete_staged_journey_fails(self):
        case = scenario()
        case["oracle"]["expected_turn_count"] = 2
        case["oracle"]["expected_response_languages"] = ["English", "English"]
        journey = artifact()
        journey["status"] = "active"
        journey["finished_at"] = None

        report = grade_scenario(case, journey)

        integrity = next(
            gate for gate in report["gates"] if gate["id"] == "journey_integrity"
        )
        self.assertFalse(integrity["passed"])
        self.assertEqual(integrity["facts"]["expected_turn_count"], 2)
        self.assertEqual(integrity["facts"]["observed_turn_count"], 1)

    def test_failed_last_turn_is_not_skipped(self):
        case = scenario()
        case["oracle"]["expected_turn_count"] = 2
        case["oracle"]["expected_response_languages"] = ["English", "English"]
        journey = artifact()
        journey["turns"].append({
            "request": {"message": "Refine the request."},
            "response": {"status_code": 500, "body": None},
        })

        report = grade_scenario(case, journey)

        responses = next(
            gate for gate in report["gates"] if gate["id"] == "turn_responses"
        )
        self.assertFalse(responses["passed"])
        self.assertFalse(report["hard_pass"])

    def test_every_search_turn_requires_a_fresh_matching_run_id(self):
        case = scenario()
        case["oracle"]["expected_turn_count"] = 2
        case["oracle"]["expected_response_languages"] = ["English", "English"]
        journey = artifact()
        journey["turns"].append({
            "request": {"message": "Refine the request."},
            "response": {
                "status_code": 200,
                "body": journey["turns"][0]["response"]["body"],
            },
        })

        report = grade_scenario(case, journey)

        runs = next(
            gate for gate in report["gates"] if gate["id"] == "retrieval_run_integrity"
        )
        self.assertFalse(runs["passed"])
        self.assertEqual(runs["facts"]["duplicate_run_ids"], ["run-1"])

    def test_incomplete_retrieval_fails_even_when_target_is_first(self):
        report = grade_scenario(scenario(), artifact(status="incomplete"))

        self.assertFalse(report["hard_pass"])
        completion = next(
            gate for gate in report["gates"]
            if gate["id"] == "retrieval_completion"
        )
        self.assertFalse(completion["passed"])

    def test_target_requires_same_facility_verbatim_decisive_evidence(self):
        journey = artifact()
        evidence = journey["turns"][0]["response"]["body"]["results"][0][
            "retrieval_evidence"
        ][0]
        evidence["place_id"] = "another-facility"

        report = grade_scenario(scenario(), journey)

        target = next(
            gate for gate in report["gates"] if gate["id"] == "reverse_target"
        )
        self.assertFalse(target["passed"])
        self.assertEqual(
            target["facts"]["missing_decisive_evidence_ids"],
            ["review:target"],
        )

    def test_missing_reported_distance_fails_geography_gate(self):
        journey = artifact()
        del journey["turns"][0]["response"]["body"]["results"][0]["distance_km"]

        report = grade_scenario(scenario(), journey)

        geography = next(
            gate for gate in report["gates"] if gate["id"] == "geography_and_distance"
        )
        self.assertFalse(geography["passed"])
        self.assertEqual(
            geography["facts"]["failures"][0]["reason"],
            "missing reported distance",
        )

    def test_required_retrieval_stage_must_have_a_successful_result(self):
        journey = artifact()
        trace = journey["turns"][0]["response"]["body"]["state"][
            "last_retrieval_trace"
        ]
        trace[1]["result_count"] = 0

        report = grade_scenario(scenario(), journey)

        orchestration = next(
            gate for gate in report["gates"]
            if gate["id"] == "retrieval_orchestration"
        )
        self.assertFalse(orchestration["passed"])
        self.assertIn(
            "search_indexed_evidence",
            orchestration["facts"]["unsuccessful_required_actions"],
        )

    def test_staged_distance_checks_the_implicit_five_km_turn(self):
        case = scenario()
        case["oracle"]["turn_radius_expectations_km"] = [5.0]
        case["oracle"]["implicit_default_radius_turns"] = [1]
        journey = artifact()
        journey["turns"][0]["response"]["body"]["state"]["max_distance_km"] = 5.0

        report = grade_scenario(case, journey)

        staged = next(gate for gate in report["gates"] if gate["id"] == "staged_distance")
        self.assertTrue(staged["passed"])
        self.assertEqual(staged["facts"]["implicit_default_turns"], [1])

    def test_each_review_turn_must_finish_the_declared_tool_sequence(self):
        case = scenario()
        case["oracle"]["required_trace_actions_each_turn"] = True
        journey = artifact()
        journey["turns"][0]["response"]["body"]["state"]["last_retrieval_trace"].pop()

        report = grade_scenario(case, journey)

        staged = next(
            gate
            for gate in report["gates"]
            if gate["id"] == "retrieval_orchestration_each_turn"
        )
        self.assertFalse(staged["passed"])
        self.assertEqual(staged["facts"]["failures"][0]["turn"], 1)

    def test_pair_requires_overlap_and_both_hard_passes(self):
        left = grade_scenario(scenario(), artifact())
        korean_artifact = artifact()
        korean_artifact["language"] = "Korean"
        korean_artifact["turns"][0]["response"]["body"]["response"] = (
            "근거가 있는 피부과를 찾았습니다."
        )
        right = grade_scenario(
            {
                **scenario(),
                "id": "reverse-one-ko",
                "counterpart_id": "reverse-one-en",
                "source_language": "Korean",
                "oracle": {
                    **scenario()["oracle"],
                    "expected_response_languages": ["Korean"],
                },
            },
            korean_artifact,
        )

        pair = grade_pair(left, right)

        self.assertTrue(pair["passed"])
        self.assertEqual(pair["top_five_jaccard"], 1.0)

    def test_pair_rejects_non_counterparts_and_missing_required_ranks(self):
        left = grade_scenario(scenario(), artifact())
        right = {**left, "scenario_id": "unrelated-en", "source_language": "English"}
        right["target"] = {**left["target"], "presented_rank": None}

        pair = grade_pair(left, right)

        self.assertFalse(pair["passed"])
        self.assertFalse(pair["counterpart_integrity"])
        self.assertFalse(pair["language_integrity"])
        self.assertFalse(pair["target_rank_consistency"])

    def test_casebook_loads_the_saved_transit_observation(self):
        casebook_path = TESTS_DIR / "grounded_bilingual_scenarios.json"
        casebook = json.loads(casebook_path.read_text(encoding="utf-8"))
        case = next(
            item for item in casebook["scenarios"]
            if item["id"] == "reverse-mapoderm-02-en"
        )

        observation = _load_transit_observation(casebook_path, case)

        self.assertEqual(observation["provider"], "kakao")
        self.assertEqual(observation["destination"]["latitude"], 37.5563578579)
        self.assertEqual(observation["route_distance_m"], 1360)

    def test_saved_transit_observation_is_bound_to_the_case(self):
        case = scenario()
        case["oracle"]["transit_observation"] = {
            "provider": "kakao",
            "observed_on": "2026-09-02",
            "request_sha256": "a" * 64,
            "response_sha256": "b" * 64,
        }
        observation = {
            "provider": "kakao",
            "observed_at": "2026-09-02T11:55:38+00:00",
            "origin": case["oracle"]["origin"],
            "destination": {
                "latitude": case["oracle"]["reverse_target"]["latitude"],
                "longitude": case["oracle"]["reverse_target"]["longitude"],
            },
            "mode": "BUS",
            "route_distance_m": 1360,
            "duration_s": 694,
            "transfers": 0,
            "fare_krw": 1200,
            "request_sha256": "a" * 64,
            "response_sha256": "b" * 64,
        }

        report = grade_scenario(case, artifact(), observation)

        transport = next(
            gate for gate in report["gates"] if gate["id"] == "public_transit"
        )
        self.assertTrue(transport["passed"])
        self.assertAlmostEqual(
            transport["facts"]["straight_line_distance_km"], 0.9827, places=3
        )
        self.assertEqual(transport["facts"]["transit_route_distance_m"], 1360)

    def test_kakao_transit_parser_keeps_route_distance_separate(self):
        parsed = parse_kakao_transit({
            "routes": [{
                "type": "SUBWAY",
                "totalDistance": 9_323,
                "totalTime": 1_374,
                "transferCount": 1,
                "fare": 2_550,
            }]
        })

        self.assertEqual(parsed["mode"], "SUBWAY")
        self.assertEqual(parsed["route_distance_m"], 9_323)
        self.assertEqual(parsed["duration_s"], 1_374)
        self.assertEqual(parsed["transfers"], 1)
        self.assertEqual(parsed["fare_krw"], 2_550)


if __name__ == "__main__":
    unittest.main()
