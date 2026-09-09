import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from search.rules import RulesCompiler  # noqa: E402
from search.scope import ScopeBuilder  # noqa: E402


ORIGIN_LAT = 37.5665
ORIGIN_LON = 126.9780
SEOUL_DISTRICTS = (
    ("강남구", "Gangnam-gu"),
    ("강동구", "Gangdong-gu"),
    ("강북구", "Gangbuk-gu"),
    ("강서구", "Gangseo-gu"),
    ("관악구", "Gwanak-gu"),
    ("광진구", "Gwangjin-gu"),
    ("구로구", "Guro-gu"),
    ("금천구", "Geumcheon-gu"),
    ("노원구", "Nowon-gu"),
    ("도봉구", "Dobong-gu"),
    ("동대문구", "Dongdaemun-gu"),
    ("동작구", "Dongjak-gu"),
    ("마포구", "Mapo-gu"),
    ("서대문구", "Seodaemun-gu"),
    ("서초구", "Seocho-gu"),
    ("성동구", "Seongdong-gu"),
    ("성북구", "Seongbuk-gu"),
    ("송파구", "Songpa-gu"),
    ("양천구", "Yangcheon-gu"),
    ("영등포구", "Yeongdeungpo-gu"),
    ("용산구", "Yongsan-gu"),
    ("은평구", "Eunpyeong-gu"),
    ("종로구", "Jongno-gu"),
    ("중구", "Jung-gu"),
    ("중랑구", "Jungnang-gu"),
)


def facility(
    place_id: str,
    *,
    category: str = "치과",
    district: str = "Jung-gu",
    latitude: float | None = ORIGIN_LAT,
    longitude: float | None = ORIGIN_LON,
    name: str = "Clinic",
    amenities=None,
    business_hours: str = "",
):
    return {
        "place_id": place_id,
        "name": name,
        "category": category,
        "address": f"서울 {district}",
        "file_district": district,
        "file_dong": "",
        "lat": latitude,
        "lon": longitude,
        "amenities": amenities or {},
        "business_hours": business_hours,
        "medical_info_parsed": {},
        "Summaries": [],
        "Summaries_Korean": [],
        "Key_Highlights": [],
        "has_english": False,
    }


class ScopeBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.compiler = RulesCompiler()
        self.builder = ScopeBuilder()
        self.facilities = pd.DataFrame([
            facility("near-parking", amenities={"parking": True}),
            facility("near-no-parking"),
            facility(
                "far-parking",
                latitude=ORIGIN_LAT + 0.060,
                amenities={"parking": True},
            ),
            facility("near-dermatology", category="피부과"),
            facility(
                "near-cosmetic",
                name="Cosmetic-only dental clinic",
                amenities={"parking": True},
            ),
            facility("missing-coordinates", latitude=None, longitude=None),
            facility("mapo-dentist", district="Mapo-gu"),
            facility("gangnam-dentist", district="Gangnam-gu"),
        ])

    def compile(self, query: str, proposal: dict):
        result = self.compiler.compile(
            original_query=query,
            turn_id="scope-test",
            proposal=proposal,
        )
        self.assertEqual(result.issues, ())
        self.assertIsNotNone(result.rules)
        return result.rules

    def point_rules(self, query: str, **proposal):
        return self.compile(query, {
            "specialty": "dentist",
            "location": "current map position",
            "latitude": ORIGIN_LAT,
            "longitude": ORIGIN_LON,
            **proposal,
        })

    def test_radius_and_specialty_are_strict_eligibility_rules(self):
        rules = self.point_rules("Find a dentist near me")
        selection = self.builder.build(
            self.facilities,
            rules,
            index_version="test-v1",
        )

        self.assertEqual(
            set(selection.facility_ids),
            {
                "near-parking",
                "near-no-parking",
                "near-cosmetic",
                "mapo-dentist",
                "gangnam-dentist",
            },
        )
        self.assertNotIn("far-parking", selection.facility_ids)
        self.assertNotIn("near-dermatology", selection.facility_ids)
        self.assertNotIn("missing-coordinates", selection.facility_ids)
        self.assertTrue(all(
            distance <= 5.0
            for distance in selection.distance_km_by_facility.values()
        ))

    def test_required_and_prohibited_attributes_never_become_scores(self):
        rules = self.point_rules(
            "Find a dentist with parking, excluding cosmetic-only clinics",
            hard_keywords=["parking"],
            negative_hard_keywords=["cosmetic only"],
        )
        selection = self.builder.build(
            self.facilities,
            rules,
            index_version="test-v1",
        )

        self.assertEqual(selection.facility_ids, ("near-parking",))

    def test_hard_negative_applies_without_a_positive_hard_requirement(self):
        rules = self.point_rules(
            "Find a dentist, excluding cosmetic-only clinics",
            negative_hard_keywords=["cosmetic only"],
        )
        selection = self.builder.build(
            self.facilities,
            rules,
            index_version="test-v1",
        )

        self.assertNotIn("near-cosmetic", selection.facility_ids)
        self.assertIn("near-no-parking", selection.facility_ids)

    def test_tuesday_evening_is_a_structured_hard_scope_rule(self):
        facilities = pd.DataFrame([
            facility(
                "tuesday-evening",
                business_hours="월: 10:00 - 18:00; 화: 10:00 - 21:00",
            ),
            facility(
                "tuesday-daytime",
                business_hours="월: 10:00 - 21:00; 화: 10:00 - 17:30",
            ),
            facility(
                "tuesday-closed",
                business_hours="화: 정기휴무; 수: 10:00 - 21:00",
            ),
        ])
        rules = self.point_rules(
            "Tuesday evening hours are mandatory",
            required_hours=["tuesday_evening"],
        )

        selection = self.builder.build(facilities, rules, index_version="test-v1")

        self.assertEqual(selection.facility_ids, ("tuesday-evening",))

    def test_english_and_korean_district_rules_select_the_same_ids(self):
        english = self.compile(
            "Find a dentist in Gangnam-gu",
            {
                "specialty": "dentist",
                "location": "Gangnam-gu",
                "district": "Gangnam-gu",
            },
        )
        korean = self.compile(
            "강남구 치과를 찾아 주세요",
            {
                "specialty": "치과",
                "location": "강남구",
                "district": "강남구",
            },
        )

        english_scope = self.builder.build(
            self.facilities,
            english,
            index_version="test-v1",
        )
        korean_scope = self.builder.build(
            self.facilities,
            korean,
            index_version="test-v1",
        )
        self.assertEqual(english.rules_hash, korean.rules_hash)
        self.assertEqual(english_scope.facility_ids, ("gangnam-dentist",))
        self.assertEqual(english_scope.facility_ids, korean_scope.facility_ids)
        self.assertEqual(
            english_scope.descriptor.scope_digest,
            korean_scope.descriptor.scope_digest,
        )

    def test_all_25_districts_have_english_and_korean_scope_parity(self):
        facilities = pd.DataFrame([
            facility(
                f"district-{number}",
                district=roman_name,
            )
            for number, (_, roman_name) in enumerate(SEOUL_DISTRICTS)
        ])
        for number, (korean_name, roman_name) in enumerate(SEOUL_DISTRICTS):
            english = self.compile(
                f"Find a dentist in {roman_name}",
                {
                    "specialty": "dentist",
                    "location": roman_name,
                    "district": roman_name,
                },
            )
            korean = self.compile(
                f"{korean_name} 치과를 찾아 주세요",
                {
                    "specialty": "치과",
                    "location": korean_name,
                    "district": korean_name,
                },
            )
            english_scope = self.builder.build(
                facilities,
                english,
                index_version="district-parity-v1",
            )
            korean_scope = self.builder.build(
                facilities,
                korean,
                index_version="district-parity-v1",
            )
            expected = (f"district-{number}",)
            with self.subTest(district=korean_name):
                self.assertEqual(english_scope.facility_ids, expected)
                self.assertEqual(korean_scope.facility_ids, expected)
                self.assertEqual(english.rules_hash, korean.rules_hash)

    def test_complete_scope_is_not_truncated_for_the_legacy_rag_adapter(self):
        facilities = pd.DataFrame([
            facility(f"facility-{number}")
            for number in range(300)
        ])
        rules = self.compile(
            "Find dentists anywhere in Seoul",
            {
                "specialty": "dentist",
                "location": "Seoul",
                "is_citywide_search": True,
            },
        )
        selection = self.builder.build(
            facilities,
            rules,
            index_version="test-v1",
        )
        rag_input = selection.restrict_dataframe(facilities)

        self.assertEqual(selection.descriptor.facility_count, 300)
        self.assertEqual(len(selection.facility_ids), 300)
        self.assertEqual(len(rag_input), 300)
        self.assertEqual(
            rag_input.attrs["eligible_scope"]["facility_count"],
            300,
        )


class LegacyRagScopeIntegrationTests(unittest.TestCase):
    def test_result_selection_uses_review_backed_candidates_when_available(self):
        import main

        ranked = pd.DataFrame([
            {"place_id": "closest-empty", "retrieval_evidence": []},
            {"place_id": "reviewed", "retrieval_evidence": [{"text": "one"}, {"text": "two"}]},
            {"place_id": "single-review", "retrieval_evidence": [{"text": "one"}]},
        ])

        selected = main.prefer_review_backed_candidates(ranked)

        self.assertEqual(selected["place_id"].tolist(), ["reviewed"])

    def test_result_selection_keeps_candidates_when_no_reviewed_set_exists(self):
        import main

        ranked = pd.DataFrame([
            {"place_id": "closest-empty", "retrieval_evidence": []},
            {"place_id": "single-review", "retrieval_evidence": [{"text": "one"}]},
        ])

        selected = main.prefer_review_backed_candidates(ranked)

        self.assertEqual(selected["place_id"].tolist(), ["closest-empty", "single-review"])

    def test_execute_search_uses_active_index_version_for_live_retrieval(self):
        import main
        from models import State

        facilities = pd.DataFrame([
            facility("alpha"),
            facility("bravo"),
        ])

        class ActiveIndex:
            version = "active-v1"

        class RecordingRag:
            def __init__(self):
                self.legacy_calls = 0

            def apply_combined_ranking(self, **kwargs):
                self.legacy_calls += 1
                return kwargs["df"]

            def build_context_for_llm(self, df, **kwargs):
                return "Scoped facilities"

        class RecordingAdapter:
            seen_scope = None

            def __init__(self, **kwargs):
                pass

            def rank(self, *, scope, eligible, **kwargs):
                RecordingAdapter.seen_scope = scope
                ranked = eligible.copy()
                ranked["relevance_rank"] = range(1, len(ranked) + 1)
                ranked.attrs["rag_metadata"] = {
                    "retrieval_status": "indexed",
                    "candidate_scope_count": len(ranked),
                }
                telemetry = type("Telemetry", (), {"status": "indexed"})()
                return type("Result", (), {
                    "dataframe": ranked,
                    "telemetry": telemetry,
                })()

        rag = RecordingRag()
        state = State(
            specialty="치과",
            specialty_confidence=0.95,
            location=None,
            is_citywide_search=True,
            language_pref="Korean",
        )
        with (
            patch.object(main, "df_filtered", facilities),
            patch.object(main, "available_specialties", ["치과"]),
            patch.object(main, "rag_pipeline", rag),
            patch.object(main, "search_index_release", ActiveIndex()),
            patch.object(main, "CandidateRetrievalAdapter", RecordingAdapter),
            patch.object(main, "client", object()),
            patch.object(
                main,
                "request_answer_completion",
                return_value=("검색 결과입니다.", None),
            ),
        ):
            main.execute_search(state, "서울 전체에서 치과를 찾아 주세요")

        self.assertEqual(
            RecordingAdapter.seen_scope.descriptor.index_version,
            "active-v1",
        )
        self.assertEqual(rag.legacy_calls, 0)

    def test_execute_search_passes_the_complete_scope_to_legacy_rag(self):
        import main
        from models import State

        facilities = pd.DataFrame([
            facility(f"facility-{number}")
            for number in range(300)
        ])

        class RecordingRag:
            def __init__(self):
                self.candidate_ids = []

            def apply_combined_ranking(self, *, df, **kwargs):
                self.candidate_ids = df["place_id"].astype(str).tolist()
                ranked = df.copy()
                ranked["relevance_rank"] = range(len(ranked))
                return ranked

            def build_context_for_llm(self, df, **kwargs):
                return "Scoped facilities"

        rag = RecordingRag()
        state = State(
            specialty="치과",
            specialty_confidence=0.95,
            location=None,
            is_citywide_search=True,
            language_pref="Korean",
        )
        with (
            patch.object(main, "df_filtered", facilities),
            patch.object(main, "available_specialties", ["치과"]),
            patch.object(main, "rag_pipeline", rag),
            patch.object(main, "client", object()),
            patch.object(
                main,
                "request_answer_completion",
                return_value=("검색 결과입니다.", None),
            ),
        ):
            _, results = main.execute_search(
                state,
                "서울 전체에서 치과를 찾아 주세요",
            )

        self.assertEqual(len(rag.candidate_ids), 300)
        self.assertEqual(set(rag.candidate_ids), set(facilities["place_id"]))
        self.assertEqual(len(results), 5)
        self.assertEqual(
            state.last_retrieval_metadata["candidate_scope_count"],
            300,
        )

    def test_execute_search_never_sends_ineligible_rows_to_rag(self):
        import main
        from models import State

        facilities = pd.DataFrame([
            facility("eligible", amenities={"parking": True}),
            facility("missing-parking"),
            facility(
                "excluded-cosmetic",
                name="Cosmetic-only dental clinic",
                amenities={"parking": True},
            ),
            facility(
                "outside-radius",
                latitude=ORIGIN_LAT + 0.020,
                amenities={"parking": True},
            ),
            facility(
                "wrong-specialty",
                category="피부과",
                amenities={"parking": True},
            ),
        ])

        class RecordingRag:
            def __init__(self):
                self.candidate_ids = []

            def apply_combined_ranking(self, *, df, **kwargs):
                self.candidate_ids = df["place_id"].astype(str).tolist()
                ranked = df.copy()
                ranked["relevance_rank"] = range(len(ranked))
                ranked["distance_km"] = 999.0
                return ranked

            def build_context_for_llm(self, df, **kwargs):
                return "Scoped facilities"

        rag = RecordingRag()
        state = State(
            specialty="치과",
            specialty_confidence=0.95,
            location="current map position",
            latitude=ORIGIN_LAT,
            longitude=ORIGIN_LON,
            search_mode="distance",
            max_distance_km=1.0,
            travel_confidence=1.0,
            hard_keywords=["parking"],
            negative_hard_keywords=["cosmetic only"],
            language_pref="Korean",
        )
        with (
            patch.object(main, "df_filtered", facilities),
            patch.object(main, "available_specialties", ["치과", "피부과"]),
            patch.object(main, "rag_pipeline", rag),
            patch.object(main, "client", object()),
            patch.object(
                main,
                "request_answer_completion",
                return_value=("검색 결과입니다.", None),
            ),
        ):
            _, results = main.execute_search(
                state,
                "1km 이내 주차 가능한 치과, 미용 전용 제외",
            )

        self.assertEqual(rag.candidate_ids, ["eligible"])
        self.assertEqual([result["place_id"] for result in results], ["eligible"])
        self.assertLessEqual(results[0]["distance_km"], 1.0)
        self.assertEqual(
            state.last_retrieval_metadata["candidate_scope_count"],
            1,
        )


if __name__ == "__main__":
    unittest.main()
