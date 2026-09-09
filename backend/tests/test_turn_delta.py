import sys
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from models import State  # noqa: E402
from search.turn_delta import compile_turn_delta, reduce_search_state  # noqa: E402


JAMSIL = (37.51331105877401, 127.10023101886318)
ICHON = (37.52241291408466, 126.97354382085399)
DAEHEUNG = (37.5476479056751, 126.942473188734)


class SearchTurnDeltaTests(unittest.TestCase):
    def apply(self, message, state=None, **proposal):
        current = state or State(location=None)
        delta = compile_turn_delta(message, proposal)
        return reduce_search_state(current, delta)

    def test_canonical_station_name_does_not_replace_medical_purpose(self):
        prior = State(visit_reason="foot issue", disease_terms=["foot issue"],
                      specialty="정형외과", specialty_confidence=0.95)
        for message in ("I need an orthopedic doctor near Jonggak", "종각 근처 정형외과 찾아주세요"):
            with self.subTest(message=message):
                current = self.apply(message, prior, location="종각역", visit_reason=message)
                self.assertEqual(current.visit_reason, "foot issue")
                self.assertIn("foot issue", current.disease_terms)

    def test_provider_lookup_with_medical_purpose_preserves_that_purpose(self):
        for message in ("doctor for pain in my foot", "doctor near Jonggak for ankle pain",
                        "발이 아파서 종각 근처 정형외과 찾아주세요"):
            with self.subTest(message=message):
                current = self.apply(message, location="종각역", visit_reason=message)
                self.assertEqual(current.visit_reason, message)

    def test_additive_ankle_symptom_does_not_replace_established_specialty(self):
        prior = State(
            visit_reason="acne",
            disease_terms=["acne"],
            specialty="피부과",
            specialty_confidence=0.95,
        )
        for message, visit_reason in (
            ("My ankle hurts too", "My ankle hurts"),
            ("발목도 아파요", "발목도 아파요"),
        ):
            with self.subTest(message=message):
                current = self.apply(
                    message,
                    prior,
                    specialty=None,
                    specialty_confidence=0,
                    disease_terms=["ankle pain"],
                    visit_reason=visit_reason,
                )
                self.assertEqual(current.specialty, "피부과")
                self.assertEqual(current.specialty_confidence, 0.95)
                self.assertEqual(current.disease_terms, ["acne", "ankle pain"])

        replacement = self.apply(
            "Actually, I need help because my ankle pain is too severe now",
            prior,
            specialty=None,
            specialty_confidence=0,
            disease_terms=["ankle pain"],
            visit_reason="my ankle pain is too severe",
        )
        self.assertEqual(replacement.specialty, "정형외과")
        self.assertEqual(replacement.disease_terms, ["ankle pain"])

        no_prior_reason = self.apply(
            "Actually, I need help with ankle pain now",
            State(
                specialty="피부과",
                specialty_confidence=0.95,
                disease_terms=["acne"],
            ),
            specialty=None,
            specialty_confidence=0,
            disease_terms=["ankle pain"],
            visit_reason="ankle pain",
        )
        self.assertEqual(no_prior_reason.specialty, "정형외과")
        self.assertEqual(no_prior_reason.disease_terms, ["ankle pain"])

    def test_local_station_without_distance_resets_citywide_radius_to_5km(self):
        state = State(
            location=None,
            is_citywide_search=True,
            max_distance_km=25.0,
            travel_label="Anywhere in Seoul",
            travel_confidence=1.0,
        )
        result = self.apply(
            "I am at Jamsil Station and need a dentist.",
            state,
            specialty="치과",
            specialty_confidence=0.95,
            location="Jamsil Station",
            latitude=JAMSIL[0],
            longitude=JAMSIL[1],
            district="송파구",
        )

        self.assertEqual(result.max_distance_km, 5.0)
        self.assertEqual(result.search_mode, "distance")
        self.assertFalse(result.is_citywide_search)

    def test_explicit_english_distance_wins_over_stale_travel_label(self):
        state = State(
            location="Ichon Station",
            latitude=ICHON[0],
            longitude=ICHON[1],
            max_distance_km=0.5,
            travel_label="Walking Distance",
            travel_confidence=0.6,
        )
        result = self.apply(
            "Use Ichon Station as my location and narrow this to 1 km.",
            state,
            location="Ichon Station",
            latitude=ICHON[0],
            longitude=ICHON[1],
            district="용산구",
            travel_label="Walking Distance",
        )

        self.assertEqual(result.max_distance_km, 1.0)
        self.assertEqual(result.travel_label, "Nearby")
        self.assertEqual(result.search_mode, "distance")

    def test_explicit_korean_distance_is_exact(self):
        result = self.apply(
            "이촌역을 제 위치로 사용해 1km 이내로 좁혀 주세요.",
            location="이촌역",
            latitude=ICHON[0],
            longitude=ICHON[1],
            district="용산구",
        )

        self.assertEqual(result.max_distance_km, 1.0)
        self.assertEqual(result.search_mode, "distance")

    def test_attached_korean_distance_suffixes_preserve_exact_limits(self):
        for phrase, expected in (("2km이내로만", 2), ("2km 이내로만", 2),
                                 ("500m이내", 0.5), ("1.5킬로미터이내", 1.5),
                                 ("750미터내에서", 0.75)):
            with self.subTest(phrase=phrase):
                result = self.apply(f"강남역에서 {phrase} 찾아주세요.",
                                    distance_km=5, travel_label="Moderate")
                self.assertEqual(result.max_distance_km, expected)
                self.assertEqual(result.travel_confidence, 1)
                self.assertIs(result.radius_expansion_allowed, False)
        for phrase in ("5mg", "2kmh"):
            self.assertIsNone(compile_turn_delta(phrase, {}).distance_km)

    def test_conditional_consent_preserves_nearby_radius_and_confidence(self):
        for message in (
            "Find orthopedics near Nowon Station for mild knee pain. "
            "Start nearby; widening is okay if no specialists are available.",
            "Start nearby. If no specialists are available, please widen the search.",
            "근거리 정형외과가 없으면 반경을 넓혀도 돼요.",
        ):
            with self.subTest(message=message):
                result = self.apply(message, travel_label="Flexible", distance_km=10)
                self.assertEqual(result.max_distance_km, 1)
                self.assertEqual(result.travel_confidence, 0.6)
                self.assertIs(result.radius_expansion_allowed, True)
                self.assertEqual(result.inquiries, [])

    def test_ordinary_request_and_model_permission_do_not_grant_consent(self):
        result = self.apply("Find orthopedics near Nowon Station.",
                            radius_expansion_allowed=True)
        self.assertIsNone(result.radius_expansion_allowed)
        nearby = self.apply("Find nearby orthopedics.", radius_expansion_allowed=True)
        self.assertIs(nearby.radius_expansion_allowed, False)
        self.assertEqual(nearby.travel_confidence, 0.6)

    def test_consent_survives_unrelated_refinement_but_can_be_withdrawn(self):
        consent = self.apply("Start nearby; widening is okay if no specialists are available.")
        followup = self.apply("Waiting is fine; keep clear explanations.",
                              consent.model_copy(update={
                                  "comment_terms": ["short wait", "clear explanations"],
                              }))
        self.assertIs(followup.radius_expansion_allowed, True)
        self.assertEqual(followup.max_distance_km, 1)
        self.assertEqual(followup.comment_terms, ["clear explanations"])
        for message in ("Do not widen the search even if no specialists are available.",
                        "Widening is no longer allowed.", "반경을 넓히지 마세요."):
            with self.subTest(message=message):
                stopped = self.apply(message, followup)
                self.assertIs(stopped.radius_expansion_allowed, False)
                self.assertEqual(stopped.max_distance_km, 1)
        self.assertIs(self.apply("Can you widen the search if no specialists are available?",
                                 State()).radius_expansion_allowed, None)

    def test_new_numeric_limit_clears_consent_and_replaces_previous_limit(self):
        consent = self.apply("Start nearby; widening is okay if no specialists are available.")
        hard = self.apply("강남역에서 2km이내로만 찾아주세요.", consent)
        self.assertEqual(hard.max_distance_km, 2)
        self.assertIs(hard.radius_expansion_allowed, False)
        replacement = self.apply("Now only within 0.5 km.", hard)
        self.assertEqual(replacement.max_distance_km, 0.5)
        self.assertIs(replacement.radius_expansion_allowed, False)
        unchanged = self.apply("Start nearby; widening is okay if no specialists are available.",
                               replacement)
        self.assertEqual(unchanged.max_distance_km, 0.5)
        self.assertEqual(unchanged.travel_confidence, 1)
        self.assertIs(unchanged.radius_expansion_allowed, False)
        same_turn = self.apply("Only within 2 km. Widening is okay if no specialists are available.")
        self.assertEqual(same_turn.max_distance_km, 2)
        self.assertIs(same_turn.radius_expansion_allowed, False)

    def test_context_reset_discards_consent_and_invalidates_scope_telemetry(self):
        consent = self.apply("Start nearby; widening is okay if no specialists are available.")
        consent.last_retrieval_metadata = {"coverage_sufficient": True}
        withdrawn = self.apply("Do not widen the search.", consent)
        self.assertEqual(withdrawn.last_retrieval_metadata, {})
        reset = self.apply("요청을 바꿀게요. 치과를 찾아 주세요.", consent,
                           operation="replace_context", specialty="치과")
        self.assertIsNone(reset.radius_expansion_allowed)
        self.assertEqual(reset.max_distance_km, 5)

    def test_removed_parking_disappears_from_every_polarity(self):
        state = State(
            location="Jamsil Station",
            latitude=JAMSIL[0],
            longitude=JAMSIL[1],
            hard_keywords=["parking"],
            keywords=["friendly"],
            negative_hard_keywords=["주차"],
            negative_keywords=["parking inconvenience"],
        )
        result = self.apply(
            "Parking is no longer required; Tuesday evening hours are mandatory.",
            state,
            hard_keywords=["Tuesday evening hours"],
            negative_hard_keywords=["parking"],
        )

        active = " ".join(
            result.hard_keywords
            + result.keywords
            + result.negative_hard_keywords
            + result.negative_keywords
        ).casefold()
        self.assertNotIn("parking", active)
        self.assertNotIn("주차", active)
        self.assertEqual(result.required_hours, ["tuesday_evening"])

    def test_acceptable_wait_removes_prior_wait_exclusion(self):
        state = State(
            negative_keywords=["long wait", "과도한 시술 권유"],
            comment_terms=["긴 대기"],
        )
        result = self.apply(
            "A long wait is acceptable, so do not treat it as an exclusion.",
            state,
        )

        active = " ".join(result.negative_keywords + result.comment_terms).casefold()
        self.assertNotIn("long wait", active)
        self.assertNotIn("긴 대기", active)
        self.assertIn("과도한 시술 권유", result.negative_keywords)

    def test_review_phrases_and_wait_avoidance_are_canonicalized(self):
        result = self.apply(
            "Prefer actual patient comments saying the care is careful and the "
            "explanation is detailed. Avoid long waits.",
            comment_terms=[
                "care is careful",
                "explanation is detailed",
                "thorough",
            ],
            negative_keywords=[],
        )

        self.assertEqual(
            result.comment_terms,
            ["thorough", "clear explanations"],
        )
        self.assertIn("long wait", result.negative_keywords)

    def test_no_wait_preference_is_detected_without_model_help(self):
        english = self.apply(
            "Prefer comments saying there was no wait.",
            comment_terms=[],
        )
        korean = self.apply(
            "대기가 없었다는 후기를 선호해요.",
            comment_terms=[],
        )

        self.assertEqual(english.comment_terms, ["short wait"])
        self.assertEqual(korean.comment_terms, ["short wait"])

    def test_decisive_negative_preferences_do_not_depend_on_model_extraction(self):
        english = self.apply(
            "Avoid aggressive upselling and comments about unfriendly nurses.",
            negative_keywords=[],
        )
        korean = self.apply(
            "과도한 시술 권유와 간호사가 불친절하다는 후기는 피하고 싶어요.",
            negative_keywords=[],
        )

        self.assertEqual(
            set(english.negative_keywords),
            {"aggressive upselling", "unfriendly nurses"},
        )
        self.assertEqual(
            set(korean.negative_keywords),
            {"aggressive upselling", "unfriendly nurses"},
        )

    def test_specific_nurse_risk_discards_model_generated_fragments(self):
        result = self.apply(
            "Avoid comments about unfriendly nurses.",
            negative_keywords=["unfriendly", "friendly", "kind", "nurses"],
        )

        self.assertEqual(result.negative_keywords, ["unfriendly nurses"])

    def test_context_replacement_clears_old_search_preferences(self):
        state = State(
            specialty="치과",
            location="City Hall",
            latitude=37.5668,
            longitude=126.9786,
            hard_keywords=["parking"],
            keywords=["friendly"],
            comment_terms=["friendly"],
        )
        result = self.apply(
            "요청을 바꿀게요. 대흥역에서 2km 이내 피부과를 찾아 주세요.",
            state,
            operation="replace_context",
            specialty="피부과",
            specialty_confidence=0.95,
            location="대흥역",
            latitude=DAEHEUNG[0],
            longitude=DAEHEUNG[1],
            district="마포구",
        )

        self.assertEqual(result.specialty, "피부과")
        self.assertEqual(result.location, "대흥역")
        self.assertEqual(result.max_distance_km, 2.0)
        self.assertEqual(result.hard_keywords, [])
        self.assertEqual(result.keywords, [])
        self.assertEqual(result.comment_terms, [])


if __name__ == "__main__":
    unittest.main()
