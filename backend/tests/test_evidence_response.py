"""Fallback facts remain conservative when answer generation is unavailable."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evidence_response import answer_search
from models import State


def fallback_response(cards, metadata, language, *, state=None, question="Find a clinic"):
    outcome = answer_search(question=question, state=state or State(),
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
                "retrieval_roles": ["disease"],
                "matched_constraint_ids": ["visit_reason:ankle_pain"],
                "review_source_sha256": "a" * 64}]}

    def test_fallback_keeps_owned_originals_without_endorsing(self):
        cards = [self.card()]
        before = deepcopy(cards)
        reply, prepared = fallback_response(cards, {}, "English", state=self.state(visit_reason="foot pain"))
        self.assertIn("stated concern (foot pain)", reply)
        self.assertIn("comments attached to clinic-1", reply)
        self.assertIn("compare it with the original wording", reply)
        self.assertEqual(cards, before)
        for key, value in before[0]["retrieval_evidence"][0].items():
            self.assertEqual(prepared[0]["retrieval_evidence"][0][key], value)
        self.assertEqual(prepared[0]["answer_status"], "fallback")
        self.assertEqual(prepared[0]["answer_citations"], [])

    def test_fallback_compares_review_selection_with_nearest_named_clinic(self):
        reply, _ = fallback_response(
            [self.card("alpha", 1.8), self.card("beta", 0.4), self.card("gamma", 1.3)],
            {}, "English", state=self.state(visit_reason="ankle pain"),
        )
        self.assertIn("comments attached to alpha", reply)
        self.assertIn("beta, the closest displayed option with comments", reply)
        self.assertNotIn("comments attached to gamma", reply)

    def test_specialty_only_fallback_does_not_imply_clinician_qualification(self):
        reply, _ = fallback_response(
            [self.card()], {}, "English", state=self.state(),
            question="I need an orthopedic doctor near Jonggak",
        )
        self.assertIn("orthopedics facility category", reply)
        self.assertIn("does not confirm an individual clinician's qualification", reply)
        self.assertIn("body area or symptom", reply)
        self.assertNotIn("What would you like the doctor", reply)

    def test_fallback_without_specialty_makes_no_category_claim(self):
        reply, _ = fallback_response([self.card()], {}, "English", state=State())
        self.assertIn("cards include patient comments", reply)
        self.assertNotIn("requested facility category", reply)
        self.assertNotIn("facilities match", reply)

    def test_clinic_request_does_not_raise_clinician_qualification(self):
        reply, _ = fallback_response([self.card()], {}, "English", state=self.state())
        self.assertIn("cards include patient comments", reply)
        self.assertNotIn("clinician", reply)

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
        reply, _ = fallback_response(
            [self.card("far", 1.4), self.card("near", 0.4)], {}, "English",
            state=self.state(is_citywide_search=True, visit_reason="ankle pain"),
        )
        self.assertIn("across Seoul", reply)
        self.assertNotIn("Jonggak", reply)
        self.assertNotIn("closest", reply)

    def test_missing_evidence_and_failed_execution_have_different_disclosures(self):
        for cards in ([], [self.card()]):
            complete, _ = fallback_response(cards, {"retrieval_execution_status": "complete"}, "English")
            failed, _ = fallback_response(cards, {"retrieval_execution_status": "failed"}, "English")
            self.assertNotIn("I may have missed relevant patient reviews", complete)
            self.assertIn("I may have missed relevant patient reviews", failed)

    def test_partial_search_gives_retry_without_unrelated_radius_refinement(self):
        reply, _ = fallback_response([self.card()], {"retrieval_execution_status": "partial"},
                                     "English", state=self.state(visit_reason="ankle pain"))
        self.assertIn("retry this search", reply)
        self.assertNotIn("selected during this review search", reply)
        self.assertNotIn("No usable original reviews", reply)
        self.assertNotIn("Part of the search did not finish", reply)
        self.assertNotIn("narrow", reply)

    def test_completed_expansion_is_disclosed_without_changing_specialty(self):
        reply, _ = fallback_response([self.card()], {"search_attempted_radii_km": [1, 2, 5],
                                     "search_radius_expanded": True}, "English", state=self.state())
        self.assertIn("initial 1 km radius", reply)
        self.assertIn("widened it to 5 km", reply)
        self.assertIn("keeping orthopedics", reply)

    def test_expanded_search_does_not_offer_the_empty_smaller_radius(self):
        for language in ("English", "Korean"):
            for concern in ([], ["ankle pain"]):
                reply, _ = fallback_response([self.card(distance=3.5)], {
                    "search_attempted_radii_km": [1, 2, 5], "search_radius_expanded": True,
                    "retrieval_execution_status": "complete"}, language,
                    state=self.state(disease_terms=concern))
                self.assertNotIn("within 1 km", reply)
                self.assertNotIn("1 km 이내처럼", reply)
                self.assertNotIn("1 km 이내 같은", reply)

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
        reply, prepared = fallback_response([source], {}, "Korean", state=self.state(visit_reason="발 통증"))
        self.assertIn("말씀하신 증상(발 통증)과 관련해서는 clinic-1 카드", reply)
        self.assertIn("원문 표현과 대조", reply)
        self.assertEqual(prepared[0]["retrieval_evidence"][0]["text"], source["retrieval_evidence"][0]["text"])

    def test_fallback_handles_optional_evidence_metadata_and_non_korean_original(self):
        source = self.card()
        source["retrieval_evidence"][0].update(
            language="English", retrieval_roles=None, matched_constraint_ids=None,
        )
        reply, _ = fallback_response([source], {}, "English", state=self.state(visit_reason="ankle pain"))
        self.assertIn("comments attached to clinic-1", reply)
        self.assertNotIn("Korean original", reply)

    def test_no_review_does_not_promise_attached_comments(self):
        source = self.card()
        source["retrieval_evidence"] = []
        reply, _ = fallback_response([source], {}, "English", state=self.state())
        self.assertIn("No usable original reviews", reply)
        self.assertNotIn("narrow the attached comments", reply)


if __name__ == "__main__":
    unittest.main()
