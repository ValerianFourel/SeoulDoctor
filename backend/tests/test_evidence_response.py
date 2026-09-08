"""Fallback facts remain conservative when answer generation is unavailable."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evidence_response import answer_search
from models import State


def fallback_response(cards, metadata, language, *, state=None):
    outcome = answer_search(question="Find a clinic", state=state or State(),
                            cards=cards, metadata=metadata, language=language, complete=None)
    return outcome.text, outcome.cards


class EvidenceResponseTests(unittest.TestCase):
    def state(self, **overrides):
        return State(**{
            "specialty": "정형외과", "specialty_confidence": 0.95,
            "location": "Jonggak", "latitude": 37.5702, "longitude": 126.9831,
            "search_mode": "distance", "language_pref": "English", **overrides,
        })

    def card(self, owner="clinic-1", distance=0.5):
        return {"place_id": owner, "name": owner, "distance_km": distance,
                "retrieval_evidence": [{"place_id": owner, "evidence_id": f"review:{owner}",
                "source_type": "verbatim_review", "is_verbatim": True,
                "text": "The doctor was kind but the nurse was rude.",
                "review_source_sha256": "a" * 64}]}

    def test_fallback_keeps_owned_originals_without_endorsing(self):
        cards = [self.card()]
        before = deepcopy(cards)
        reply, prepared = fallback_response(cards, {}, "English", state=self.state())
        self.assertIn("original patient reviews", reply)
        self.assertIn("couldn't verify", reply)
        self.assertEqual(cards, before)
        for key, value in before[0]["retrieval_evidence"][0].items():
            self.assertEqual(prepared[0]["retrieval_evidence"][0][key], value)
        self.assertEqual(prepared[0]["answer_status"], "fallback")
        self.assertEqual(prepared[0]["answer_citations"], [])

    def test_closest_fact_requires_all_distances_and_keeps_order(self):
        cards = [self.card("far", 1.4), self.card("near", 0.4)]
        reply, prepared = fallback_response(cards, {}, "English", state=self.state())
        self.assertIn("near is closest", reply)
        self.assertIn("straight-line", reply)
        self.assertEqual([card["place_id"] for card in prepared], ["far", "near"])
        cards[0].pop("distance_km")
        reply, _ = fallback_response(cards, {}, "English", state=self.state())
        self.assertNotIn("closest", reply)

    def test_citywide_state_never_uses_stale_distance_as_nearby_claim(self):
        reply, _ = fallback_response([self.card()], {}, "English",
                                     state=self.state(is_citywide_search=True))
        self.assertIn("across Seoul", reply)
        self.assertNotIn("Jonggak", reply)
        self.assertNotIn("closest", reply)

    def test_missing_evidence_and_failed_execution_have_different_disclosures(self):
        for cards in ([], [self.card()]):
            complete, _ = fallback_response(cards, {"retrieval_execution_status": "complete"}, "English")
            failed, _ = fallback_response(cards, {"retrieval_execution_status": "failed"}, "English")
            self.assertNotIn("search did not finish", complete)
            self.assertIn("search did not finish", failed)

    def test_known_concern_or_visit_reason_is_not_requested_again(self):
        for state in (self.state(disease_terms=["wrist pain"]), self.state(visit_reason="routine checkup")):
            reply, _ = fallback_response([self.card()], {}, "English", state=state)
            self.assertNotIn("What would you like", reply)

    def test_english_question_stays_unknown_with_a_concrete_check(self):
        state = self.state(inquiries=["Can you confirm English consultations?"])
        reply, _ = fallback_response([self.card()], {}, "English", state=state)
        self.assertIn("English consultations are unconfirmed", reply)
        self.assertIn("Ask the clinic", reply)
        self.assertEqual(state.hard_keywords, [])

    def test_korean_fallback_retains_original_identity(self):
        source = self.card()
        reply, prepared = fallback_response([source], {}, "Korean", state=self.state())
        self.assertIn("원문 후기", reply)
        self.assertEqual(prepared[0]["retrieval_evidence"][0]["text"], source["retrieval_evidence"][0]["text"])


if __name__ == "__main__":
    unittest.main()
