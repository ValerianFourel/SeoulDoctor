import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

import run_grounded_bilingual_suite as suite_runner  # noqa: E402
from run_grounded_bilingual_suite import (  # noqa: E402
    aggregate_suite,
    build_parser,
    load_casebook,
    scenario_messages,
    validate_casebook,
)


class GroundedBilingualSuiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.casebook = load_casebook(TESTS_DIR / "grounded_bilingual_scenarios.json")

    def test_current_casebook_has_exact_messages_and_valid_language_switches(self):
        validation = validate_casebook(self.casebook)

        self.assertTrue(validation["passed"], validation["failures"])
        self.assertEqual(validation["scenario_count"], 8)
        self.assertEqual(validation["pair_count"], 4)
        switched = next(
            item
            for item in validation["scenario_checks"]
            if item["scenario_id"] == "code-switch-reset-04-en-ko"
        )
        self.assertEqual(switched["language_sequence"], ["English", "Korean"])
        self.assertEqual(switched["language_checks"], [True, True])

    def test_radius_pair_exercises_implicit_five_km_then_refines(self):
        scenario = next(
            item
            for item in self.casebook["scenarios"]
            if item["id"] == "radius-expand-line8-01-en"
        )

        messages = scenario_messages(scenario)

        self.assertEqual(len(messages), 3)
        self.assertNotRegex(messages[0], r"\d\s*(?:km|metres?)")
        self.assertEqual(
            scenario["oracle"]["turn_radius_expectations_km"],
            [5.0, 0.5, 5.0],
        )
        self.assertEqual(scenario["oracle"]["implicit_default_radius_turns"], [1])

    def test_auth_token_can_come_from_environment_or_cli(self):
        required = ["--run-id", "private", "--run-dir", ".audit/private"]
        with patch.dict(
            os.environ,
            {"SEOULDOC_EVAL_AUTH_TOKEN": "environment-token"},
        ):
            from_environment = build_parser().parse_args(required)
        from_cli = build_parser().parse_args([
            *required,
            "--auth-token",
            "cli-token",
        ])

        self.assertEqual(from_environment.auth_token, "environment-token")
        self.assertEqual(from_cli.auth_token, "cli-token")

    def test_suite_forwards_auth_token_without_recording_it(self):
        token = "private-space-token"
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            with (
                patch.object(
                    suite_runner,
                    "run_journey",
                    return_value={"status": "finished", "turns": []},
                ) as run_journey,
                patch.object(
                    suite_runner,
                    "grade_scenario",
                    return_value={"hard_pass": True},
                ),
            ):
                report = suite_runner.run_suite(
                    run_id="private",
                    run_dir=run_dir,
                    casebook_path=TESTS_DIR / "grounded_bilingual_scenarios.json",
                    endpoint="https://private-space.example/chat",
                    actor_label="test",
                    selected_ids=("radius-expand-line8-01-en",),
                    judge=False,
                    auth_token=token,
                )
            saved_report = (run_dir / "suite_report.json").read_text(encoding="utf-8")

        self.assertEqual(run_journey.call_args.kwargs["auth_token"], token)
        self.assertNotIn(token, saved_report)
        self.assertTrue(report["passed"])

    def test_suite_aggregator_enforces_all_declared_thresholds(self):
        deterministic = {}
        reviews = {}
        for scenario in self.casebook["scenarios"]:
            scenario_id = scenario["id"]
            target_required = "reverse_target" in scenario["oracle"]
            deterministic[scenario_id] = {
                "hard_pass": True,
                "target": {
                    "required": target_required,
                    "presented_rank": 1 if target_required else None,
                },
            }
            reviews[scenario_id] = {"passed": True, "weighted_mean": 4.5}

        report = aggregate_suite(self.casebook, deterministic, reviews)

        self.assertTrue(report["passed"])
        self.assertEqual(report["metrics"]["hard_gate_rate"], 1.0)
        self.assertEqual(report["metrics"]["scenario_pass_rate"], 1.0)
        self.assertEqual(report["metrics"]["reverse_target_hit_at_5"], 1.0)
        self.assertEqual(report["metrics"]["mean_reciprocal_rank"], 1.0)

    def test_language_gap_equal_to_threshold_passes_despite_float_noise(self):
        casebook = json.loads(json.dumps(self.casebook))
        suite_policy = casebook["likert_policy"]["suite_pass"]
        suite_policy["minimum_weighted_mean"] = 0.0
        suite_policy["maximum_language_gap"] = 0.3
        deterministic = {}
        reviews = {}
        for scenario in casebook["scenarios"]:
            scenario_id = scenario["id"]
            target_required = "reverse_target" in scenario["oracle"]
            deterministic[scenario_id] = {
                "hard_pass": True,
                "target": {
                    "required": target_required,
                    "presented_rank": 1 if target_required else None,
                },
            }
            reviews[scenario_id] = {
                "passed": True,
                "weighted_mean": (
                    3.3 if scenario["source_language"] == "English" else 3.6
                ),
            }

        report = aggregate_suite(casebook, deterministic, reviews)

        self.assertAlmostEqual(report["metrics"]["language_gap"], 0.3)
        self.assertTrue(report["gates"]["language_gap"])

    def test_suite_aggregator_rejects_one_failed_scenario_at_eight(self):
        deterministic = {}
        reviews = {}
        for scenario in self.casebook["scenarios"]:
            scenario_id = scenario["id"]
            deterministic[scenario_id] = {
                "hard_pass": True,
                "target": {"required": False, "presented_rank": None},
            }
            reviews[scenario_id] = {"passed": True, "weighted_mean": 4.5}
        reviews[self.casebook["scenarios"][0]["id"]]["passed"] = False

        report = aggregate_suite(self.casebook, deterministic, reviews)

        self.assertFalse(report["passed"])
        self.assertFalse(report["gates"]["scenario_pass_rate"])


if __name__ == "__main__":
    unittest.main()
