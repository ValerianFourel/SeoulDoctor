import json
from pathlib import Path
import tempfile
import unittest

import requests

from scripts.search_repair_eval import (
    ACTOR_MODEL, ACTOR_PROVIDER, Budget, Runner, actor_payload, check_turn, expand_fixed, parse_actor,
)


def suite(messages=("Find orthopedics", "Keep the same area")):
    return {"fixed": [{"id": "fixture", "patient": {"messages": list(messages)}, "oracle": {}}],
            "adaptive": [], "grading": {"minimum_each_dimension": 4}}


class Response:
    def __init__(self, body, status=200):
        self.body, self.status_code = body, status
        self.text = json.dumps(body)

    def json(self):
        return self.body


def body(state=None):
    return {"response": "Please clarify the visit reason.", "state": state or {}, "results": []}


class SearchRepairRunnerTests(unittest.TestCase):
    def test_state_is_preserved_and_timeout_retained_without_replay(self):
        calls = []
        state = {"specialty": "정형외과", "private_field": {"nested": [1, 2]}, "turn_count": 1}

        def post(url, **kwargs):
            calls.append(kwargs["json"])
            if len(calls) == 1:
                return Response(body(state))
            raise requests.Timeout("synthetic")

        with tempfile.TemporaryDirectory() as directory:
            runner = Runner("http://localhost:8000", Path(directory) / "run", suite(),
                            "fixed", "fixture-commit", post=post)
            self.assertEqual(runner.run(), 1)
            saved = json.loads((runner.directory / "run.json").read_text())
            self.assertEqual(calls[1]["current_state"], state)
            self.assertEqual(len(calls), 2)
            self.assertEqual(saved["budget"]["app_calls"], 2)
            self.assertEqual(saved["cases"][0]["turns"][1]["app"]["error_type"], "Timeout")
            self.assertEqual(saved["cases"][0]["turns"][0]["response"]["body"]["state"], state)
            self.assertFalse(saved["quality_pass"])
            self.assertTrue((runner.directory / "fixture.judge.json").exists())

    def test_fixed_failure_blocks_later_scenarios(self):
        manifest = suite(("Find orthopedics",))
        manifest["fixed"].append({"id": "later", "patient": {"messages": ["Other query"]}, "oracle": {}})
        calls = []
        def post(url, **kwargs):
            calls.append(url)
            return Response({"error": "upstream unavailable"}, 503)
        with tempfile.TemporaryDirectory() as directory:
            runner = Runner("http://localhost:8000", Path(directory) / "run", manifest,
                            "fixed", "fixture-commit", post=post)
            self.assertEqual(runner.run(), 1)
            self.assertEqual(len(calls), 1)
            self.assertEqual(runner.cases[1]["status"], "not_run")
            self.assertEqual(runner.cases[0]["turns"][0]["app"]["raw_text"], '{"error": "upstream unavailable"}')

    def test_actor_receives_only_patient_brief_and_visible_content(self):
        patient = {"persona": "A patient", "language": "English", "stages": ["Find a doctor"],
                   "turns": 1, "instructions": "Be concise", "oracle": "HIDDEN_ORACLE"}
        turn = {"message": "Find a doctor", "oracle": "HIDDEN_TARGET", "response": {"body": {
            "response": "One option", "state": {"secret": "HIDDEN_STATE"}, "debug": "HIDDEN_DEBUG",
            "results": [{"place_id": "HIDDEN_ID", "name": "Public name", "category": "정형외과",
                         "retrieval_evidence": [{"evidence_id": "HIDDEN_EVIDENCE", "text": "Public original",
                                                 "presentation": {"text": "Public translation", "status": "translated"}}]}]}}}
        encoded = json.dumps(actor_payload(patient, [turn], 0))
        self.assertNotIn("HIDDEN_", encoded)
        self.assertIn("Public original", encoded)
        self.assertIn("Public translation", encoded)

    def test_missing_reviews_never_produce_quality_pass(self):
        response = body({"specialty": "정형외과"})
        response["results"] = [{"place_id": "alpha", "category": "정형외과", "retrieval_evidence": []}]
        result = check_turn(response, {"korean_specialty": "정형외과"}, 0)
        self.assertFalse(result["quality_pass"])
        self.assertIn("review_relevance", result["unscored"])

    def test_wrong_specialty_location_and_owner_fail(self):
        response = body({"latitude": 37.59, "longitude": 127.08})
        response["results"] = [{"place_id": "alpha", "category": "피부과", "retrieval_evidence": [
            {"place_id": "beta", "evidence_id": "review:a", "is_verbatim": True, "text": "Actual text"}]}]
        result = check_turn(response, {"korean_specialty": "정형외과",
            "coordinate_bounds": {"latitude": [37.555, 37.57], "longitude": [126.977, 126.995]}}, 0)
        self.assertIn("matching_specialty_cards", result["failures"])
        self.assertIn("latitude_within_bounds", result["failures"])
        self.assertIn("card_0_review_0_owner", result["failures"])

    def test_local_fault_fixtures_are_not_sent_to_public_or_claimed_passed(self):
        manifest = suite(("Find a doctor",))
        manifest["fixed"][0]["execution"] = {"fixture": {"answer_fault": "timeout"}}
        def forbidden(*args, **kwargs):
            self.fail("Fixture was dispatched")
        with tempfile.TemporaryDirectory() as directory:
            runner = Runner("https://example.hf.space", Path(directory) / "run", manifest,
                            "fixed", "fixture-commit", post=forbidden)
            self.assertEqual(runner.run(), 1)
            self.assertEqual(runner.cases[0]["reason"], "fixture_adapter_required")

    def test_run_directory_cannot_be_replayed(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                Runner("http://localhost", directory, suite(), "fixed", "fixture")

    def test_unknown_actor_cost_stays_reserved(self):
        budget = Budget()
        budget.reserve("actor")
        budget.settle_actor({})
        self.assertEqual(budget.actor_reserved_usd, 0.05)
        self.assertEqual(budget.unknown_actor_costs, 1)
        budget.app_calls = 80
        with self.assertRaises(ValueError):
            budget.reserve("app")

    def test_fixed_variants_are_separate_conversations(self):
        scenarios = [{"id": "spellings", "patient": {"messages": ["Near {location}"]},
                      "variants": [{"location": "Myeongdong"}, {"location": "명동"}]}]
        expanded = expand_fixed(scenarios)
        self.assertEqual([case["id"] for case in expanded], ["spellings-v1", "spellings-v2"])
        self.assertEqual(expanded[1]["patient"]["messages"], ["Near 명동"])

    def test_valid_json_with_truncated_actor_completion_is_rejected(self):
        response = {"model": ACTOR_MODEL, "provider": ACTOR_PROVIDER, "choices": [
            {"finish_reason": "length", "message": {"content": '{"message":"Find a doctor"}'}}]}
        with self.assertRaisesRegex(ValueError, "incomplete actor"):
            parse_actor(response)

    def test_evaluator_exception_is_separate_from_application_failure(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            runner = Runner("http://localhost:8000", Path(directory) / "run", suite(("Hello",)),
                            "fixed", "fixture", post=lambda *args, **kwargs: Response(body()))
            with patch("scripts.search_repair_eval.check_turn", side_effect=RuntimeError("fixture")):
                self.assertEqual(runner.run(), 1)
            self.assertEqual(runner.cases[0]["error_category"], "evaluator")
            self.assertEqual(runner.cases[0]["turns"][0]["app"]["status"], "complete")

    def test_adaptive_requires_reviewed_fixed_gate_before_network(self):
        def forbidden(*args, **kwargs):
            self.fail("Unreviewed adaptive run used network")
        with tempfile.TemporaryDirectory() as directory:
            runner = Runner("http://localhost:8000", Path(directory) / "run", suite(),
                            "adaptive", "fixture", actor_key="fixture", post=forbidden, get=forbidden)
            self.assertEqual(runner.run(), 1)
            self.assertEqual(runner.record["status"], "preflight_failed")
            self.assertEqual(runner.budget.actor_calls, 0)

    def test_actor_catalogue_price_mismatch_blocks_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = Path(directory) / "fixed-evidence.json"
            evidence.write_text('{}')
            runner = Runner("http://localhost:8000", Path(directory) / "run", suite(),
                            "adaptive", "fixture", actor_key="fixture", get=lambda *args, **kwargs: Response({
                                "data": {"id": ACTOR_MODEL, "endpoints": [{"provider_name": ACTOR_PROVIDER,
                                          "pricing": {"prompt": "0.1", "completion": "0.1"}}]}}))
            runner.fixed_gate = {"passed": True, "reviewer": "root GPT-6", "application_revision": "fixture",
                                 "manifest_sha256": runner.record["manifest_sha256"], "evidence_paths": [str(evidence)]}
            self.assertEqual(runner.run(), 1)
            self.assertIn("price ceiling", runner.record["error_reason"])
            self.assertEqual(runner.budget.actor_calls, 0)


if __name__ == "__main__":
    unittest.main()
