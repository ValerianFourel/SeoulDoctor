import json
from pathlib import Path
import tempfile
import unittest

from scripts.conversation_eval import (
    DEFAULT_CONFIG,
    DEFAULT_SCENARIOS,
    build_packet,
    computed_result,
    dry_run,
    judge_run,
    load_object,
    validate_judgment,
)


class ConversationEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_object(DEFAULT_CONFIG)
        cls.manifest = load_object(DEFAULT_SCENARIOS)

    def case(self):
        return {
            "id": "adaptive-fixture",
            "status": "complete",
            "oracle": {"expected": "private target"},
            "turns": [{
                "message": "I need orthopedics near Jonggak.",
                "status": "complete",
                "assessment": {"failures": []},
                "response": {"body": {
                    "response": "Clinic A has a relevant ankle review.",
                    "state": {"specialty": "정형외과"},
                    "results": [{
                        "place_id": "clinic-a", "name": "Clinic A", "category": "정형외과",
                        "address": "Seoul", "distance_km": 0.4,
                        "retrieval_evidence": [{
                            "evidence_id": "review-a", "place_id": "clinic-a",
                            "text": "발목 치료 설명을 잘 해주셨어요.",
                            "presentation": {"status": "translated", "text": "They explained the ankle treatment well."},
                        }],
                    }],
                }},
            }],
        }

    def judgment(self, packet, score=4):
        dimensions = {
            name: {"score": score, "rationale": "Supported by the visible turn.",
                   "citations": ["turn:1"], "not_applicable_reason": None}
            for name in self.config["rubric"]["dimensions"]
        }
        return {
            "scenario_id": packet["scenario_id"], "packet_id": packet["packet_id"],
            "packet_sha256": packet["packet_sha256"], "dimensions": dimensions,
            "critical_failures": [],
            "simulator_adherence": {"status": "valid", "explanation": "The patient followed the brief.",
                                      "citations": ["turn:1"]},
            "proposed_verdict": "pass",
        }

    def test_dry_run_freezes_exactly_seven_scenarios_without_network(self):
        result = dry_run(self.config, self.manifest)
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["network_requests_made"], 0)
        self.assertEqual(result["scenario_count"], 7)
        self.assertGreaterEqual(result["estimated_app_calls"], 7)

    def test_packet_separates_visible_evidence_and_private_oracle(self):
        scenario = {"patient": {"persona": "public"}, "oracle": {"expected": "fallback"}}
        packet = build_packet(self.case(), {"started_at": 1, "application_revision": "abc",
                                             "manifest_sha256": "manifest", "status": "complete"},
                              self.config, scenario)
        self.assertEqual(packet["private_oracle"], {"expected": "private target"})
        visible = json.dumps(packet["user_visible_conversation"], ensure_ascii=False)
        self.assertNotIn("private target", visible)
        self.assertIn("발목 치료", visible)
        self.assertIn("evidence:review-a", packet["citation_catalog"])

    def test_wrong_or_unsupported_citation_is_rejected(self):
        packet = build_packet(self.case(), {"started_at": 1}, self.config)
        judgment = self.judgment(packet)
        first = self.config["rubric"]["dimensions"][0]
        judgment["dimensions"][first]["citations"] = ["evidence:made-up"]
        with self.assertRaisesRegex(ValueError, "does not resolve"):
            validate_judgment(judgment, packet, self.config)

    def test_objective_failure_overrides_generous_judge_verdict(self):
        case = self.case()
        case["turns"][0]["assessment"]["failures"] = ["card_0_review_0_owner"]
        packet = build_packet(case, {"started_at": 1}, self.config)
        judgment = validate_judgment(self.judgment(packet, score=5), packet, self.config)
        result = computed_result(packet, judgment, self.config)
        self.assertFalse(result["passed"])
        self.assertFalse(result["gates"]["objective_checks"])
        self.assertTrue(result["judge_verdict_conflict"])

    def test_incomplete_simulator_behavior_cannot_pass(self):
        packet = build_packet(self.case(), {"started_at": 1}, self.config)
        judgment = self.judgment(packet, score=5)
        judgment["simulator_adherence"]["status"] = "invalid"
        judgment["proposed_verdict"] = "invalid"
        validated = validate_judgment(judgment, packet, self.config)
        self.assertFalse(computed_result(packet, validated, self.config)["passed"])

    def test_null_dimension_is_reported_and_never_counted_as_pass(self):
        packet = build_packet(self.case(), {"started_at": 1}, self.config)
        judgment = self.judgment(packet)
        first = self.config["rubric"]["dimensions"][0]
        judgment["dimensions"][first] = {
            "score": None, "rationale": "Not applicable.", "citations": [],
            "not_applicable_reason": "The scenario never disclosed this concern.",
        }
        validated = validate_judgment(judgment, packet, self.config)
        result = computed_result(packet, validated, self.config)
        self.assertFalse(result["passed"])
        self.assertEqual(result["null_score_count"], 1)

    def test_codex_backend_exports_packets_without_claiming_a_pass(self):
        selected = self.config["smoke_scenario_ids"]
        run = {
            "started_at": 1, "status": "complete", "application_revision": "abc",
            "manifest_sha256": "manifest", "cases": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "run"
            source.mkdir()
            (source / "manifest.json").write_text(json.dumps(self.manifest))
            for scenario_id in selected:
                case = self.case()
                case["id"] = scenario_id
                run["cases"].append(case)
            run_path = source / "run.json"
            run_path.write_text(json.dumps(run))
            output = Path(directory) / "judging"
            report = judge_run(run_path, output, self.config, "codex_subagents")
            self.assertEqual(report["status"], "pending")
            self.assertFalse(report["passed"])
            self.assertEqual(report["pending_or_incomplete_count"], 7)
            self.assertEqual(len(list((output / "packets").glob("*.json"))), 7)


if __name__ == "__main__":
    unittest.main()
