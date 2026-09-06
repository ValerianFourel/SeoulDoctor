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
