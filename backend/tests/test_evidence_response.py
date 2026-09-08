from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evidence_response import finalize_evidence_response
from models import State


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

    def facility(self, place_id, name, *, distance=None, groups=None, **overrides):
        card = {
            "place_id": place_id,
            "name": name,
            "category": "정형외과",
            "retrieval_evidence": [],
            "retrieval_evidence_groups": groups or {
                "supporting": [], "warnings": [], "unverified": [],
            },
        }
        if distance is not None:
            card["distance_km"] = distance
            card["distance"] = distance
        card.update(overrides)
        return card

    def jonggak_state(self, **overrides):
        values = {
            "specialty": "정형외과",
            "specialty_confidence": 0.95,
            "location": "Jonggak",
            "latitude": 37.5702,
            "longitude": 126.9831,
            "search_mode": "distance",
            "turn_count": 1,
            "language_pref": "English",
        }
        values.update(overrides)
        return State(**values)

    def test_nonempty_response_names_the_closest_shown_option_without_reordering_cards(self):
        source_cards = [
            self.facility("clinic-1", "First Clinic", distance=1.4),
            self.facility("clinic-2", "Second Clinic", distance=2.2),
            self.facility("clinic-3", "광화문정형외과의원", distance=0.888),
            self.facility("clinic-4", "Fourth Clinic", distance=3.1),
            self.facility("clinic-5", "Fifth Clinic", distance=4.0),
        ]

        response, cards = finalize_evidence_response(
            "All five are English-speaking and highly recommended.",
            source_cards,
            {"retrieval_status": "complete", "coverage_sufficient": True},
            "English",
            state=self.jonggak_state(),
        )

        self.assertIn(
            "5 options for orthopedics around Jonggak",
            response,
        )
        self.assertIn("Of the options shown, 광화문정형외과의원 is closest", response)
        self.assertIn("0.9 km", response)
        self.assertIn("straight-line", response)
        self.assertIn("What would you like the doctor to help with?", response)
        self.assertIn("within 1 km", response)
        self.assertNotIn("English-speaking", response)
        self.assertNotIn("highly recommended", response)
        self.assertEqual(
            [card["place_id"] for card in cards],
            [card["place_id"] for card in source_cards],
        )

    def test_incomplete_response_still_gives_grounded_options_and_next_step(self):
        response, cards = finalize_evidence_response(
            "Definitely recommended",
            [
                self.facility("clinic-1", "Far Clinic", distance=1.6),
                self.facility("clinic-2", "Near Clinic", distance=0.4),
            ],
            {"retrieval_status": "incomplete", "coverage_sufficient": False},
            "English",
            state=self.jonggak_state(),
        )

        self.assertIn("I found 2 options", response)
        self.assertIn("Near Clinic", response)
        self.assertIn("Search is incomplete", response)
        self.assertIn("What would you like the doctor to help with?", response)
        self.assertNotIn("Definitely recommended", response)
        self.assertTrue(all(
            card["recommendation_status"] == "not_established"
            for card in cards
        ))

    def test_risk_and_unverified_requirements_are_disclosed_without_endorsement(self):
        warning = self.card()
        unverified = self.facility(
            "clinic-2",
            "Unverified Clinic",
            groups={
                "supporting": [],
                "warnings": [],
                "unverified": ["english_consultation"],
            },
        )

        response, cards = finalize_evidence_response(
            "These are the best suitable clinics.",
            [warning, unverified],
            {"retrieval_status": "complete", "coverage_sufficient": True},
            "English",
            state=self.jonggak_state(is_citywide_search=True, location="Jonggak"),
        )

        self.assertIn("reviews need a closer look against your preferences", response)
        self.assertIn("does not confirm a problem", response)
        self.assertIn("requirements remain unverified", response)
        self.assertNotIn("best", response.lower())
        self.assertNotIn("suitable", response.lower())
        self.assertEqual(cards[0]["recommendation_status"], "requires_review")
        self.assertEqual(cards[1]["recommendation_status"], "not_established")
        self.assertEqual(
            cards[0]["retrieval_evidence"][0]["evidence_role"],
            "mixed",
        )

    def test_korean_composer_uses_state_without_repeating_known_symptoms(self):
        response, _ = finalize_evidence_response(
            "원문 응답",
            [self.facility("clinic-1", "종각정형외과", distance=0.6)],
            {"retrieval_status": "complete", "coverage_sufficient": True},
            "Korean",
            state=self.jonggak_state(
                language_pref="Korean",
                disease_terms=["무릎 통증"],
            ),
        )

        self.assertIn("정형외과 검색", response)
        self.assertIn("Jonggak", response)
        self.assertIn("종각정형외과", response)
        self.assertIn("직선거리", response)
        self.assertIn("카드를 비교", response)
        self.assertNotIn("무릎 통증", response)
        self.assertNotIn("어떤 도움이 필요", response)
        self.assertNotIn("원문 응답", response)

    def test_known_symptoms_are_not_asked_for_again_in_english(self):
        response, _ = finalize_evidence_response(
            "unused",
            [self.facility("clinic-1", "Clinic", distance=0.5)],
            {"retrieval_status": "complete", "coverage_sufficient": True},
            "English",
            state=self.jonggak_state(disease_terms=["knee pain"]),
        )

        self.assertIn("compare the cards", response)
        self.assertNotIn("knee pain", response)
        self.assertNotIn("What would you like the doctor to help with?", response)

    def test_citywide_state_wins_over_location_and_suppresses_distance_claim(self):
        response, _ = finalize_evidence_response(
            "unused",
            [
                self.facility("clinic-1", "First Clinic", distance=0.2),
                self.facility("clinic-2", "Second Clinic", distance=0.5),
            ],
            {"retrieval_status": "complete", "coverage_sufficient": True},
            "English",
            state=self.jonggak_state(is_citywide_search=True),
        )

        self.assertIn("across Seoul", response)
        self.assertNotIn("Jonggak", response)
        self.assertNotIn("closest", response)
        self.assertNotIn("0.2 km", response)
        self.assertNotIn("within 1 km", response)
        self.assertNotIn("straight-line", response)

    def test_closest_claim_requires_a_valid_distance_for_every_shown_card(self):
        response, _ = finalize_evidence_response(
            "unused",
            [
                self.facility("clinic-1", "Measured Clinic", distance=0.2),
                self.facility("clinic-2", "Unknown Clinic"),
            ],
            {"retrieval_status": "complete", "coverage_sufficient": True},
            "English",
            state=self.jonggak_state(),
        )

        self.assertNotIn("closest", response)
        self.assertNotIn("0.2 km", response)

    def test_default_seoul_without_coordinates_is_not_a_nearby_anchor(self):
        response, _ = finalize_evidence_response(
            "unused",
            [self.facility("clinic-1", "Clinic", distance=0.2)],
            {"retrieval_status": "complete", "coverage_sufficient": True},
            "English",
            state=self.jonggak_state(
                location="Seoul", latitude=None, longitude=None,
            ),
        )

        self.assertNotIn("around Seoul", response)
        self.assertNotIn("closest", response)
        self.assertNotIn("0.2 km", response)

    def test_empty_results_preserve_no_result_reply_and_only_add_incomplete_disclosure(self):
        original = "I couldn't find any orthopedics facilities in your search area.\n\nTry expanding your search radius."
        complete, cards = finalize_evidence_response(
            original,
            [],
            {"retrieval_status": "complete", "coverage_sufficient": True},
            "English",
            state=self.jonggak_state(),
        )
        incomplete, incomplete_cards = finalize_evidence_response(
            original,
            [],
            {"retrieval_status": "incomplete", "coverage_sufficient": False},
            "English",
            state=self.jonggak_state(),
        )

        self.assertEqual(complete, original)
        self.assertEqual(cards, [])
        self.assertTrue(incomplete.startswith(original))
        self.assertIn("Search is incomplete", incomplete)
        self.assertNotIn("I found", incomplete)
        self.assertEqual(incomplete_cards, [])

    def test_source_cards_and_evidence_identity_are_not_changed(self):
        source = self.card(
            source_locator="review_snapshot:clinic-1:0",
            review_source_sha256="a" * 64,
            private_marker={"rank": 3},
        )
        before = deepcopy(source)

        _, cards = finalize_evidence_response(
            "unused",
            [source],
            {"retrieval_status": "complete", "coverage_sufficient": True},
            "English",
            state=self.jonggak_state(),
        )

        self.assertEqual(source, before)
        item = cards[0]["retrieval_evidence"][0]
        for field in (
            "evidence_id", "place_id", "text", "source_locator",
            "review_source_sha256", "private_marker",
        ):
            self.assertEqual(item[field], before["retrieval_evidence"][0][field])

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
