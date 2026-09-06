import unittest
from scripts.run_parallel_diagnostics import DIMENSIONS, validate_judge


class DiagnosticJudgeTests(unittest.TestCase):
    def test_only_exact_visible_citations_validate(self):
        review = {"scores": {key: 3 for key in DIMENSIONS},
                  "findings": [{"turn": 1, "quote": "nurses were rude", "explanation": "Conflict"}]}
        self.assertTrue(validate_judge(review, [{"reply": "The nurses were rude."}]))
        self.assertFalse(validate_judge(review, [{"reply": "The doctor was kind."}]))

    def test_boolean_score_and_missing_citations_are_invalid(self):
        review = {"scores": {key: True for key in DIMENSIONS}, "findings": []}
        self.assertFalse(validate_judge(review, [{"reply": "text"}]))


if __name__ == "__main__":
    unittest.main()
