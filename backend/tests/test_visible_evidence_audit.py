import unittest

from scripts.audit_visible_evidence import audit, normalized


class VisibleEvidenceAuditTests(unittest.TestCase):
    def test_multiline_blockquote_matches_original(self):
        self.assertEqual(normalized("> Clear doctor.\n> But rude nurses."),
                         normalized("Clear doctor.\nBut rude nurses."))

    def test_internal_attachment_is_not_visible_evidence_success(self):
        target = {"place_id": "a", "evidence_id": "review:1", "text": "Nurses were rude."}
        journey = {"status": "finished", "turns": [{"response": {"body": {
            "response": "A friendly doctor.", "results": [{"place_id": "a",
            "retrieval_evidence": [{**target, "is_verbatim": True}]}],
        }}}]}
        report = audit(journey, target)
        self.assertTrue(report["turns"][0]["survived_evidence_selection"])
        self.assertFalse(report["turns"][0]["original_comment_visible"])
        self.assertFalse(report["passed"])

    def test_visible_comment_still_requires_suitability_judgment(self):
        target = {"place_id": "a", "evidence_id": "review:1", "text": "Nurses were rude."}
        journey = {"status": "active", "turns": [{"response": {"body": {
            "response": target["text"], "results": [{"place_id": "a",
            "retrieval_evidence": [{**target, "is_verbatim": True}]}],
        }}}]}
        report = audit(journey, target)
        self.assertTrue(report["turns"][0]["original_text_preserved"])
        self.assertEqual(report["turns"][0]["recommendation_use"], "requires_independent_review")
        self.assertFalse(report["passed"])
