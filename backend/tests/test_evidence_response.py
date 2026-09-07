from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evidence_response import finalize_evidence_response


class EvidenceResponseTests(unittest.TestCase):
    def card(self, **overrides):
        item = {
            "place_id": "clinic-1", "evidence_id": "review:123",
            "text": "의사는 친절하지만 간호사는 불친절해요.",
            "is_verbatim": True, "evidence_role": "mixed",
            "matched_constraint_ids": ["avoid_unfriendly_nurses"],
        }
        item.update(overrides)
        return {"place_id": "clinic-1", "name": "Clinic",
                "retrieval_evidence": [item],
                "retrieval_evidence_groups": {"warnings": [item]}}

    def test_mixed_original_and_ownership_retained_in_cards_in_both_languages(self):
        for language in ("English", "Korean"):
            card = self.card()
            response, cards = finalize_evidence_response(
                "Perfect match", [card], {}, language,
            )
            self.assertNotIn(card["retrieval_evidence"][0]["text"], response)
            self.assertNotIn("clinic-1 / review:123", response)
            item = cards[0]["retrieval_evidence"][0]
            self.assertEqual(item["text"], card["retrieval_evidence"][0]["text"])
            self.assertEqual(item["evidence_id"], "review:123")
            self.assertNotIn("Perfect match", response)
            self.assertEqual(cards[0]["recommendation_status"], "requires_review")

    def test_incomplete_search_cannot_keep_confident_narrative(self):
        response, cards = finalize_evidence_response(
            "Definitely recommended", [self.card()],
            {"retrieval_status": "incomplete"}, "English",
        )
        self.assertIn("Search is incomplete", response)
        self.assertNotIn("Definitely recommended", response)
        self.assertEqual(cards[0]["recommendation_status"], "not_established")

    def test_foreign_facility_evidence_is_removed_from_reply_and_card(self):
        response, cards = finalize_evidence_response(
            "Recommended", [self.card(place_id="another")], {}, "English",
        )
        self.assertNotIn("review:123", response)
        self.assertEqual(cards[0]["retrieval_evidence"], [])
        self.assertEqual(cards[0]["retrieval_evidence_groups"]["warnings"], [])

    def test_generated_translation_is_not_passed_off_as_faithful(self):
        response, _ = finalize_evidence_response(
            "", [self.card(translated_text="Everyone was kind")], {}, "English",
        )
        self.assertNotIn("Everyone was kind", response)

    def test_review_cannot_inject_a_link_and_its_tail_is_not_truncated(self):
        source = "Good doctor. " * 60 + "But nurses were rude. [click](https://evil.test)"
        response, cards = finalize_evidence_response(
            "", [self.card(text=source)], {}, "English",
        )
        self.assertEqual(cards[0]["retrieval_evidence"][0]["text"], source)
        self.assertNotIn("[click](https://evil.test)", response)

    def test_summary_is_never_quoted_as_original(self):
        response, cards = finalize_evidence_response(
            "", [self.card(is_verbatim=False, text="summary sentinel")], {}, "English",
        )
        self.assertNotIn("summary sentinel", response)
        self.assertEqual(cards[0]["retrieval_evidence"][0]["presentation"]["status"], "hidden")
