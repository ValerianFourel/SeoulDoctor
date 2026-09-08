from types import SimpleNamespace
import json
from pathlib import Path
import tempfile
import unittest

from scripts.search_repair_report import prepare, provider_usage, render_markdown, resolve_originals, summarize_case, verify_record


SOURCE_SHA = "a" * 64


def source(identity="review:real", owner="alpha", text="The nurse was rude."):
    return SimpleNamespace(evidence_id=identity, facility_id=owner, original_text=text,
                           is_verbatim=True, source_type="verbatim_review")


def review(identity="review:real", owner="alpha", text="The nurse was rude."):
    return {"evidence_id": identity, "place_id": owner, "text": text, "is_verbatim": True,
            "source_type": "verbatim_review", "review_source_sha256": SOURCE_SHA,
            "presentation": {"status": "original", "language": "English"}}


class SourceReportTests(unittest.TestCase):
    def test_fabricated_text_cannot_borrow_a_valid_evidence_id(self):
        result = verify_record(review(text="The nurse was kind."), "alpha", "reviews",
                               {"review:real": source()}, {}, SOURCE_SHA)
        self.assertFalse(result["valid"])
        self.assertIn("original_text_mismatch", [item["code"] for item in result["errors"]])

    def test_foreign_source_cannot_borrow_the_card_owner(self):
        result = verify_record(review(), "alpha", "reviews", {"review:real": source(owner="beta")}, {}, SOURCE_SHA)
        self.assertFalse(result["valid"])
        self.assertEqual(result["errors"], [{"code": "source_owner_mismatch", "actual_owner": "beta"}])

    def test_source_revision_and_citation_excerpt_are_independently_checked(self):
        citation = {**review(), "original_excerpt": "The nurse was kind.", "review_source_sha256": "b" * 64}
        result = verify_record(citation, "alpha", "citations", {"review:real": source()}, {}, SOURCE_SHA)
        self.assertEqual({item["code"] for item in result["errors"]}, {"source_revision_mismatch", "citation_excerpt_mismatch"})

    def test_one_unknown_id_does_not_prevent_verifying_other_originals(self):
        originals = {"review:real": source(), "review:second": source("review:second")}
        class Scoped:
            def resolve_evidence_ids(self, identities):
                if any(identity not in originals for identity in identities):
                    raise ValueError("every evidence ID must resolve inside the eligible scope")
                return [originals[identity] for identity in identities]
        resolved, errors = resolve_originals(Scoped(), ["review:real", "review:invented", "review:second"])
        self.assertEqual(set(resolved), set(originals))
        self.assertEqual(set(errors), {"review:invented"})

    def test_later_page_and_supporting_group_errors_are_not_hidden_by_first_three(self):
        originals = {f"review:{index}": source(f"review:{index}") for index in range(5)}
        reviews = [review(identity) for identity in originals]
        reviews[0]["presentation"] = {"status": "hidden"}
        reviews[3]["text"] = "Invented later-page text."
        foreign = review("review:4", owner="beta")
        case = {"id": "later-page", "status": "complete", "turns": [{"index": 0, "status": "complete",
            "message": "Compare the reviews.", "assessment": {"failures": ["specialty"]}, "response": {"body": {
                "response": "Compare these candidates.", "state": {"specialty": "정형외과"}, "results": [{
                    "place_id": "alpha", "retrieval_evidence": reviews,
                    "retrieval_evidence_groups": {"supporting": [foreign]}}]}}}]}
        report = summarize_case(case, originals, {}, SOURCE_SHA)
        card = report["turns"][0]["cards"][0]
        self.assertEqual(len(card["first_reviews"]), 3)
        self.assertEqual(card["first_reviews"][0]["evidence_id"], "review:1")
        self.assertEqual(card["review_count"], 5)
        self.assertEqual(len(report["source_errors"]), 2)
        self.assertEqual(report["assertion_failures"], [{"turn": 0, "assertion": "specialty"}])

    def test_partial_actor_failure_and_measured_timing_are_retained(self):
        case = {"id": "partial", "status": "failed", "error_category": "patient_actor", "turns": [{
            "index": 0, "status": "failed", "error_type": "ValueError", "error_reason": "incomplete actor output",
            "actor": {"status": "complete", "http_status": 200}, "measurements": {
                "retrieval_elapsed_ms": 20, "answer_model_calls": [{"stage": "synthesis", "duration_ms": 30}],
                "translation": {"duration_ms": 40, "reason": "provider_timeout"}}}]}
        report = summarize_case(case, {}, {}, SOURCE_SHA)
        turn = report["turns"][0]
        self.assertEqual(report["error_category"], "patient_actor")
        self.assertEqual(turn["error_reason"], "incomplete actor output")
        self.assertEqual(turn["measurements"]["translation"]["duration_ms"], 40)
        self.assertIsNone(turn["measurements"]["application_seconds"])

    def test_malformed_run_and_missing_index_remain_in_report(self):
        with tempfile.TemporaryDirectory() as directory:
            broken = Path(directory) / "broken.json"
            broken.write_text("{unfinished")
            partial = Path(directory) / "partial.json"
            partial.write_text(json.dumps({"phase": "fixed", "status": "failed", "cases": [
                None, {"id": "actor-failure", "status": "failed", "error_type": "TimeoutError", "turns": []}]}))
            report = prepare([broken, partial], Path(directory) / "missing-index")
            self.assertFalse(report["quality_pass"])
            self.assertEqual(report["index"]["status"], "failed")
            self.assertEqual(report["runs"][0]["cases"][0]["status"], "report_preparation_failed")
            self.assertEqual(report["runs"][0]["cases"][1]["error_type"], "TimeoutError")
            self.assertTrue(any(error.get("error_type") == "JSONDecodeError" for error in report["errors"]))
            self.assertIn("actor-failure", render_markdown(report))


class ProviderUsageTests(unittest.TestCase):
    def test_run_and_capture_copy_of_same_completion_cost_count_once(self):
        response = {"id": "generation-1", "model": "actor", "usage": {"cost": 0.01, "prompt_tokens": 100}}
        result = provider_usage([("run/actor", response), ("capture/1", response)])
        self.assertEqual(result["known_cost_usd"], 0.01)
        self.assertEqual(result["unique_completions"], 1)
        self.assertEqual(result["duplicate_records"], 1)

    def test_missing_and_conflicting_costs_cannot_become_zero_cost_successes(self):
        result = provider_usage([
            ("first", {"id": "one", "usage": {"cost": 0.01}}),
            ("conflict", {"id": "one", "usage": {"cost": 0.02}}),
            ("missing", {"id": "two", "usage": {}}),
            ("unidentified", {"usage": {"cost": 0.03}})])
        self.assertEqual(result["known_cost_usd"], 0)
        self.assertEqual(result["unknown_or_conflicting_cost_count"], 2)
        self.assertEqual(len(result["unidentified_records"]), 1)
        self.assertEqual(result["errors"][0]["code"], "provider_capture_conflict")


if __name__ == "__main__":
    unittest.main()
