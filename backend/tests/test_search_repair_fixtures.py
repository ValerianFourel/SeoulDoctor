import json
from pathlib import Path
import tempfile
import unittest

from scripts.search_repair_eval import Runner, expand_fixed
from scripts.search_repair_fixtures import FixtureApp


MANIFEST = json.loads((Path(__file__).resolve().parents[2] / "scripts/search_repair_scenarios.json").read_text())


class SearchRepairFixtureTests(unittest.TestCase):
    def execute(self, prefix):
        scenario = next(case for case in expand_fixed(MANIFEST["fixed"]) if case["id"] == prefix)
        with FixtureApp(scenario) as app:
            state = app.initial_state()
            bodies = []
            for message in scenario["patient"]["messages"]:
                response = app.post("http://fixture/chat", json={"message": message, "current_state": state})
                self.assertEqual(response.status_code, 200, response.text)
                body = response.json()
                self.assertEqual(app.assess(body)["failures"], [], app.assess(body))
                bodies.append(body)
                state = body["state"]
            return app, bodies

    def test_specialty_only_query_uses_real_owned_review_selection(self):
        app, bodies = self.execute("fixed-09-specialty-only-reviews")
        self.assertTrue(app.index.queries)
        self.assertTrue(all(len(card["retrieval_evidence"]) >= 2 for card in bodies[0]["results"]))

    def test_answer_faults_do_not_starve_translations(self):
        for variant in range(1, 4):
            with self.subTest(variant=variant):
                self.execute(f"fixed-11-answer-fault-independent-translation-v{variant}")

    def test_translation_provider_faults_have_separate_reasons(self):
        for variant in range(1, 5):
            with self.subTest(variant=variant):
                self.execute(f"fixed-12-translation-failure-v{variant}")

    def test_unavailable_retrieval_keeps_originals_and_retry_guidance(self):
        self.execute("fixed-13-retrieval-unavailable")

    def test_hard_radius_does_not_include_far_specialists_or_other_specialties(self):
        _, bodies = self.execute("fixed-08-hard-radius")
        self.assertEqual(bodies[0]["results"], [])

    def test_soft_radius_attempts_the_frozen_one_two_five_ladder(self):
        self.execute("fixed-07-soft-radius-expansion")

    def test_first_page_reviews_exclude_unrelated_source_comments(self):
        self.execute("fixed-10-first-page-reviews")

    def test_mixed_reviews_keep_doctor_and_nurse_originals(self):
        app, bodies = self.execute("fixed-14-mixed-negative-evidence")
        self.assertEqual(app.provenance()["answer_provider_revision"], "controlled-review-interpretation-v2")
        self.assertFalse(app.provenance()["network_inference"])
        self.assertNotEqual(bodies[0]["response"], bodies[1]["response"])
        for body in bodies:
            reply = body["response"]
            self.assertEqual(body["state"]["last_retrieval_metadata"]["answer"]["status"], "generated")
            citations = [citation for card in body["results"] for citation in card["answer_citations"]]
            self.assertEqual(len(citations), 3)
            self.assertEqual(len({citation["place_id"] for citation in citations}), 1)
            self.assertEqual({citation["original_excerpt"] for citation in citations}, set(app.fixture["reviews"]))
            for citation in citations:
                source = app.sources[citation["evidence_id"]]
                self.assertEqual(citation["place_id"], source.facility_id)
                self.assertEqual(citation["original_excerpt"], source.original_text)
                self.assertEqual(reply.count(source.original_text), 1)
            self.assertIn("reduce confidence that this clinic fits those priorities", reply)
            self.assertIn("whether you can take time to decide", reply)
            self.assertIn("compare another orthopedic clinic", reply)

    def test_mixed_review_stub_requires_all_cited_reports_in_the_model_context(self):
        scenario = next(case for case in expand_fixed(MANIFEST["fixed"])
                        if case["id"] == "fixed-14-mixed-negative-evidence")
        app = FixtureApp(scenario)
        evidence = [{"place_id": source.facility_id, "evidence_id": source.evidence_id,
                     "original_text": source.original_text} for source in app.sources.values()
                    if "felt pressured" not in source.original_text]
        with self.assertRaises(KeyError):
            app.answer(messages=[{"content": json.dumps({"evidence": evidence,
                "question": scenario["patient"]["messages"][1]})}])

    def test_fixture_phase_never_dispatches_to_supplied_public_url(self):
        scenario = next(case for case in MANIFEST["fixed"] if case["id"] == "fixed-09-specialty-only-reviews")
        manifest = {**MANIFEST, "fixed": [scenario]}
        with tempfile.TemporaryDirectory() as directory:
            runner = Runner("https://example.hf.space", Path(directory) / "run", manifest,
                            "fixtures", "synthetic-source", post=lambda *args, **kwargs: self.fail("Public request"))
            self.assertEqual(runner.run(), 0, runner.record)
            self.assertEqual(runner.record["completed"], 1)
            self.assertFalse(runner.record["quality_pass"])
            self.assertFalse(runner.cases[0]["fixture"]["network_inference"])
            self.assertTrue(runner.cases[0]["turns"][0]["measurements"]["private_trace_available"])


if __name__ == "__main__":
    unittest.main()
