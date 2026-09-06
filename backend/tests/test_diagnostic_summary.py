import unittest
from scripts.summarize_diagnostics import target_checks


class TargetCheckTests(unittest.TestCase):
    def test_attached_comment_is_not_automatically_visible(self):
        target = {"place_id": "a", "source_evidence_id": "a:review:0", "comment": "nurses are rude"}
        body = {"response": "The doctor is kind.", "results": [{"place_id": "a",
            "retrieval_evidence": [{"place_id": "a", "text": "nurses are rude"}]}]}
        result = target_checks(body, target)
        self.assertTrue(result["exact_comment_attached"])
        self.assertTrue(result["correct_ownership"])
        self.assertFalse(result["exact_original_visible"])

    def test_wrong_facility_fails_ownership(self):
        target = {"place_id": "a", "source_evidence_id": "a:review:0", "comment": "nurses are rude"}
        body = {"response": "nurses are rude", "results": [{"place_id": "b",
            "retrieval_evidence": [{"place_id": "b", "text": "nurses are rude"}]}]}
        result = target_checks(body, target)
        self.assertFalse(result["correct_ownership"])
        self.assertTrue(result["exact_original_visible"])


if __name__ == "__main__":
    unittest.main()
