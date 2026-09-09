import json
from hashlib import sha256
from pathlib import Path
import tempfile
import unittest

from scripts.conversation_eval import (
    DEFAULT_CONFIG,
    DEFAULT_SCENARIOS,
    build_packet,
    computed_result,
    digest,
    dry_run,
    judge_run,
    import_codex_judgments,
    load_object,
    make_source_proof,
    make_codex_assignment,
    make_ui_proof,
    runner_manifest_digest,
    validate_judgment,
    validate_packet,
    validate_bound_proof,
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
                            "source_type": "verbatim_review", "is_verbatim": True,
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
                   "citations": (["evidence:review-a"]
                                 if name == "original_comment_relevance_and_attribution"
                                 else ["turn:1"]),
                   "not_applicable_reason": None}
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
        scenario = {"patient": {"persona": "public"}, "oracle": {"expected": "private target"}}
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

    def test_tampered_packet_is_rejected_even_when_judgment_echoes_old_hash(self):
        packet = build_packet(self.case(), {"started_at": 1}, self.config)
        packet["private_oracle"] = {"expected": "forged"}
        with self.assertRaisesRegex(ValueError, "packet hash"):
            validate_packet(packet)

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

    def test_judge_proposed_invalid_cannot_be_computed_as_pass(self):
        packet = build_packet(self.case(), {"started_at": 1}, self.config,
                              source_proof={"status": "passed"}, ui_proof={"status": "passed"})
        judgment = self.judgment(packet)
        judgment["proposed_verdict"] = "invalid"
        result = computed_result(packet, validate_judgment(judgment, packet, self.config), self.config)
        self.assertFalse(result["passed"])
        self.assertFalse(result["gates"]["judge_proposed_pass"])

    def test_codex_backend_exports_packets_without_claiming_a_pass(self):
        selected = self.config["smoke_scenario_ids"]
        run = {
            "started_at": 1, "status": "complete", "application_revision": "abc",
            "manifest_sha256": runner_manifest_digest(self.manifest),
            "selected_scenario_ids": list(selected), "cases": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "run"
            source.mkdir()
            (source / "manifest.json").write_text(json.dumps(self.manifest))
            for scenario_id in selected:
                case = self.case()
                case["id"] = scenario_id
                case["oracle"] = next(item["oracle"] for item in self.manifest["adaptive"]
                                      if item["id"] == scenario_id)
                run["cases"].append(case)
            run_path = source / "run.json"
            run_path.write_text(json.dumps(run))
            output = Path(directory) / "judging"
            report = judge_run(run_path, output, self.config, "codex_subagents")
            self.assertEqual(report["status"], "pending")
            self.assertFalse(report["passed"])
            self.assertEqual(report["pending_or_incomplete_count"], 7)
            self.assertEqual(len(list((output / "packets").glob("*.json"))), 7)

    def test_run_oracle_substitution_is_rejected(self):
        selected = self.config["smoke_scenario_ids"]
        run = {
            "started_at": 1, "status": "complete", "application_revision": "abc",
            "manifest_sha256": runner_manifest_digest(self.manifest),
            "selected_scenario_ids": list(selected), "cases": [],
        }
        for scenario_id in selected:
            case = self.case()
            case["id"] = scenario_id
            case["oracle"] = next(item["oracle"] for item in self.manifest["adaptive"]
                                  if item["id"] == scenario_id)
            run["cases"].append(case)
        run["cases"][0]["oracle"] = {"target": "forged"}
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "run"
            source.mkdir()
            (source / "manifest.json").write_text(json.dumps(self.manifest))
            run_path = source / "run.json"
            run_path.write_text(json.dumps(run))
            with self.assertRaisesRegex(ValueError, "oracle"):
                judge_run(run_path, Path(directory) / "judging", self.config, "codex_subagents")

    def test_import_rejects_a_replaced_self_consistent_packet(self):
        selected = self.config["smoke_scenario_ids"]
        run = {"started_at": 1, "status": "complete", "application_revision": "abc",
               "manifest_sha256": runner_manifest_digest(self.manifest),
               "selected_scenario_ids": list(selected), "cases": []}
        for scenario_id in selected:
            case = self.case()
            case["id"] = scenario_id
            case["oracle"] = next(item["oracle"] for item in self.manifest["adaptive"] if item["id"] == scenario_id)
            run["cases"].append(case)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "run"
            source.mkdir()
            (source / "manifest.json").write_text(json.dumps(self.manifest))
            run_path = source / "run.json"
            run_path.write_text(json.dumps(run))
            output = root / "judging"
            report = judge_run(run_path, output, self.config, "codex_subagents")
            assignment_path = root / "assignment.json"
            assignment = make_codex_assignment(output, assignment_path, "agent-test", "gpt-6-astra", self.config)
            packet_path = Path(report["results"][0]["packet"])
            packet = load_object(packet_path)
            packet["private_oracle"] = {"forged": True}
            packet["packet_sha256"] = digest({key: value for key, value in packet.items() if key != "packet_sha256"})
            packet_path.write_text(json.dumps(packet))
            judgments = []
            for item in report["results"]:
                current = load_object(Path(item["packet"]))
                judgments.append(self.judgment(current))
            bundle = root / "bundle.json"
            bundle.write_text(json.dumps({"judge_backend": "codex_subagents", "isolated_context": True,
                                          "assignment_sha256": assignment["assignment_sha256"],
                                          "agent_id": "agent-test", "judge_model": "gpt-6-astra",
                                          "judgments": judgments}))
            with self.assertRaisesRegex(ValueError, "source run|packet"):
                import_codex_judgments(output, bundle, self.config, assignment_path)

    def test_source_proof_is_bound_to_run_and_report_bytes(self):
        run = {"application_revision": "abc", "manifest_sha256": "manifest",
               "selected_scenario_ids": list(self.config["smoke_scenario_ids"])}
        report = {
            "index": {"status": "opened", "review_source_sha256": "source"},
            "runs": [{"application_revision": "abc", "manifest_sha256": "manifest"}],
            "unique_evidence_ids": 2, "resolved_evidence_ids": 2,
            "source_error_count": 0, "errors": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_path, report_path, proof_path = root / "run.json", root / "report.json", root / "proof.json"
            run_path.write_text(json.dumps(run))
            (root / "manifest.json").write_text(json.dumps(self.manifest))
            report_path.write_text(json.dumps(report))
            proof = make_source_proof(run_path, report_path, proof_path)
            self.assertEqual(proof["status"], "failed")
            validate_bound_proof(proof, "source_audit", run, run_path)
            report_path.write_text(json.dumps({**report, "resolved_evidence_ids": 1}))
            with self.assertRaisesRegex(ValueError, "changed"):
                validate_bound_proof(proof, "source_audit", run, run_path)

    def test_ui_proof_rejects_response_hash_substitution(self):
        scenario_ids = list(self.config["smoke_scenario_ids"])
        run = {"application_revision": "abc", "selected_scenario_ids": scenario_ids,
               "cases": [{"id": scenario_id, "turns": [{"message": "hello"}], "oracle": {}}
                         for scenario_id in scenario_ids]}
        response_raw = '{"response":"ok","results":[]}'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_path, observation_path = root / "run.json", root / "ui.json"
            run_path.write_text(json.dumps(run))
            scenario_observations = [{
                "scenario_id": scenario_id, "status": "failed",
                "response_raws": [response_raw],
                "response_sha256": sha256(response_raw.encode()).hexdigest(),
                "checks": [
                    {"name": "frozen conversation is complete", "passed": True},
                    {"name": "all frozen user turns are replayable", "passed": True},
                    {"name": "turn 1 returned HTTP success", "passed": True},
                    {"name": "turn 1 returned an answer", "passed": True},
                    {"name": "final turn rendered clinic cards", "passed": False},
                    {"name": "final turn exposes at least two original comments", "passed": False},
                ],
            } for scenario_id in scenario_ids]
            observation = {
                "schema_version": 1, "mode": "live_deployed_ui", "application_revision": "abc",
                "target_url": "https://www.seouldoc.io", "run_file_sha256": sha256(run_path.read_bytes()).hexdigest(),
                "runner_sha256": sha256((Path(__file__).parents[2] / "frontend/tests/live-review-deployment.cjs").read_bytes()).hexdigest(),
                "scenario_ids": scenario_ids, "scenarios": scenario_observations,
                "checks": [{"name": "all scenarios replayed", "passed": False}],
            }
            observation_path.write_text(json.dumps(observation))
            proof = make_ui_proof(run_path, observation_path, root / "proof.json")
            self.assertEqual(proof["status"], "failed")
            observation["scenarios"][0]["response_raws"] = ['{"response":"forged"}']
            observation_path.write_text(json.dumps(observation))
            with self.assertRaisesRegex(ValueError, "scenario observation"):
                make_ui_proof(run_path, observation_path, root / "proof-2.json")


if __name__ == "__main__":
    unittest.main()
