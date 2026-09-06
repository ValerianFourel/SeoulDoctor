import sys
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from search.contracts import DistanceRule  # noqa: E402
from search.rules import RulesCompiler, compile_legacy_state_rules  # noqa: E402


class RulesCompilerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.compiler = RulesCompiler()

    def compile(self, query: str, proposal: dict, *, previous_rules=None):
        return self.compiler.compile(
            original_query=query,
            turn_id="turn-1",
            proposal=proposal,
            previous_rules=previous_rules,
        )

    def require_rules(self, query: str, proposal: dict, *, previous_rules=None):
        result = self.compile(query, proposal, previous_rules=previous_rules)
        self.assertEqual(result.issues, ())
        self.assertIsNotNone(result.rules)
        return result.rules

    def test_missing_distance_gets_immutable_five_km_default(self):
        rules = self.require_rules(
            "Find a dentist near me",
            {
                "specialty": "dentist",
                "location": "current map position",
                "latitude": 37.5665,
                "longitude": 126.9780,
            },
        )

        self.assertIsInstance(rules.hard.geography, DistanceRule)
        self.assertEqual(rules.hard.geography.max_km, 5.0)
        self.assertEqual(
            rules.hard.geography.provenance.source,
            "default_5km",
        )

    def test_explicit_english_and_korean_distances_are_parsed(self):
        cases = (
            ("Find a dentist within 2 km", 2.0),
            ("2킬로미터 이내 치과를 찾아 주세요", 2.0),
            ("Find a dentist within 500 metres", 0.5),
            ("500미터 이내 치과를 찾아 주세요", 0.5),
        )
        for query, expected_km in cases:
            with self.subTest(query=query):
                rules = self.require_rules(
                    query,
                    {
                        "specialty": "치과",
                        "location": "current map position",
                        "latitude": 37.5665,
                        "longitude": 126.9780,
                    },
                )
                self.assertEqual(rules.hard.geography.max_km, expected_km)
                self.assertEqual(
                    rules.hard.geography.provenance.source,
                    "user_explicit",
                )

    def test_invalid_distance_returns_a_clarification_instead_of_raising(self):
        for distance in ("0 km", "101 km"):
            with self.subTest(distance=distance):
                result = self.compile(
                    f"Find a dentist within {distance}",
                    {
                        "specialty": "dentist",
                        "location": "current map position",
                        "latitude": 37.5665,
                        "longitude": 126.9780,
                    },
                )
                self.assertIsNone(result.rules)
                self.assertIn(
                    ("invalid_distance", "distance"),
                    {(issue.code, issue.field) for issue in result.issues},
                )

    def test_refinement_without_distance_preserves_previous_radius(self):
        first = self.require_rules(
            "Find a dentist within 2 km",
            {
                "specialty": "dentist",
                "location": "current map position",
                "latitude": 37.5665,
                "longitude": 126.9780,
                "hard_keywords": ["parking"],
                "negative_hard_keywords": ["cosmetic only"],
            },
        )
        refined = self.require_rules(
            "Also prefer a friendly clinic",
            {"soft_keywords": ["friendly"]},
            previous_rules=first,
        )

        self.assertEqual(refined.hard.geography, first.hard.geography)
        self.assertEqual(refined.hard.specialty_ids, first.hard.specialty_ids)
        self.assertEqual(
            refined.hard.required_attributes,
            first.hard.required_attributes,
        )
        self.assertEqual(
            refined.hard.prohibited_taxonomy_ids,
            first.hard.prohibited_taxonomy_ids,
        )

    def test_hard_and_soft_polarities_have_distinct_contract_fields(self):
        rules = self.require_rules(
            "Find a friendly dentist with parking, but no cosmetic-only clinic and avoid expensive places",
            {
                "specialty": "dentist",
                "location": "current map position",
                "latitude": 37.5665,
                "longitude": 126.9780,
                "hard_keywords": ["parking"],
                "negative_hard_keywords": ["cosmetic only"],
                "soft_keywords": ["friendly"],
                "negative_keywords": ["expensive"],
            },
        )

        self.assertEqual(
            {item.concept_id for item in rules.hard.required_attributes},
            {"parking"},
        )
        self.assertEqual(
            rules.hard.prohibited_taxonomy_ids,
            frozenset({"cosmetic_only"}),
        )
        self.assertEqual(
            {(item.concept_id, item.polarity) for item in rules.soft},
            {("friendly", "positive"), ("expensive", "negative")},
        )
        evidence_roles = {
            item.requirement_id: item.evidence_role
            for item in rules.evidence
        }
        self.assertEqual(
            evidence_roles["preference:positive:friendly"],
            "support",
        )
        self.assertEqual(evidence_roles["preference:negative:expensive"], "risk")

    def test_english_and_korean_inputs_compile_to_the_same_canonical_rules(self):
        english = self.require_rules(
            "Friendly dentist within 3 km with parking, avoid expensive clinics",
            {
                "specialty": "dentist",
                "location": "current map position",
                "latitude": 37.5665,
                "longitude": 126.9780,
                "hard_keywords": ["parking"],
                "soft_keywords": ["friendly"],
                "negative_keywords": ["expensive"],
            },
        )
        korean = self.require_rules(
            "3킬로미터 이내 주차 가능한 친절한 치과, 비싼 곳은 피하고 싶어요",
            {
                "specialty": "치과",
                "location": "현재 지도 위치",
                "latitude": 37.5665,
                "longitude": 126.9780,
                "hard_keywords": ["주차"],
                "soft_keywords": ["친절"],
                "negative_keywords": ["비싼"],
            },
        )

        self.assertEqual(english.hard, korean.hard)
        self.assertEqual(english.soft, korean.soft)
        self.assertEqual(english.rules_hash, korean.rules_hash)
        self.assertEqual(english.language, "en")
        self.assertEqual(korean.language, "ko")
    def test_disease_and_comment_evidence_have_separate_roles(self):
        rules = self.require_rules(
            "Find cheilitis care with comments about clear explanations",
            {
                "specialty": "dermatologist",
                "location": "current map position",
                "latitude": 37.5665,
                "longitude": 126.9780,
                "disease_terms": ["cheilitis"],
                "comment_terms": ["clear explanations"],
            },
        )

        roles = {
            item.requirement_id: item.evidence_role
            for item in rules.evidence
        }
        self.assertEqual(roles["evidence:cheilitis"], "disease")
        self.assertEqual(roles["evidence:clear_explanations"], "support")



    def test_derived_client_controls_are_rejected_at_the_boundary(self):
        result = self.compile(
            "Find a dentist near me",
            {
                "specialty": "dentist",
                "location": "current map position",
                "latitude": 37.5665,
                "longitude": 126.9780,
                "hybrid_alpha": 1.0,
                "candidate_ids": ["injected-facility"],
            },
        )

        self.assertIsNone(result.rules)
        self.assertEqual(
            {(issue.code, issue.field) for issue in result.issues},
            {
                ("rejected_control", "candidate_ids"),
                ("rejected_control", "hybrid_alpha"),
            },
        )

    def test_unknown_specialty_requests_clarification(self):
        result = self.compile(
            "Find a moon medicine specialist near me",
            {
                "specialty": "moon medicine",
                "location": "current map position",
                "latitude": 37.5665,
                "longitude": 126.9780,
            },
        )

        self.assertIsNone(result.rules)
        self.assertEqual(result.issues[0].code, "unknown_specialty")
        self.assertEqual(result.issues[0].field, "specialty")

    def test_unresolved_location_requests_clarification(self):
        result = self.compile(
            "Find a dentist in Made Up Station",
            {
                "specialty": "dentist",
                "location": "Made Up Station",
            },
        )

        self.assertIsNone(result.rules)
        self.assertEqual(result.issues[0].code, "unknown_location")
        self.assertEqual(result.issues[0].field, "location")

    def test_compiled_rules_and_provenance_are_immutable(self):
        rules = self.require_rules(
            "Find a dentist near me",
            {
                "specialty": "dentist",
                "location": "current map position",
                "latitude": 37.5665,
                "longitude": 126.9780,
            },
        )

        with self.assertRaises(FrozenInstanceError):
            rules.language = "ko"
        with self.assertRaises(TypeError):
            rules.provenance["injected"] = rules.hard.geography.provenance

    def test_legacy_adapter_preserves_only_an_explicit_prior_radius(self):
        state = SimpleNamespace(
            specialty="치과",
            location="current map position",
            district=None,
            latitude=37.5665,
            longitude=126.9780,
            is_citywide_search=False,
            max_distance_km=2.0,
            travel_confidence=0.6,
            hard_keywords=[],
            keywords=[],
            negative_hard_keywords=[],
            negative_keywords=[],
            gender_terms=[],
            disease_terms=[],
            comment_terms=[],
        )
        explicit = compile_legacy_state_rules(
            "yes, search now",
            state,
            turn_id="legacy-explicit",
            allowed_specialties=("치과",),
        )
        self.assertEqual(explicit.issues, ())
        self.assertEqual(explicit.rules.hard.geography.max_km, 2.0)

        state.max_distance_km = 50.0
        state.travel_confidence = 0.5
        inferred = compile_legacy_state_rules(
            "yes, search now",
            state,
            turn_id="legacy-default",
            allowed_specialties=("치과",),
        )
        self.assertEqual(inferred.issues, ())
        self.assertEqual(inferred.rules.hard.geography.max_km, 5.0)
        self.assertEqual(
            inferred.rules.hard.geography.provenance.source,
            "default_5km",
        )

    def test_legacy_adapter_rejects_an_invalid_inherited_radius(self):
        state = SimpleNamespace(
            specialty="치과",
            location="current map position",
            district=None,
            latitude=37.5665,
            longitude=126.9780,
            is_citywide_search=False,
            max_distance_km=500.0,
            travel_confidence=1.0,
            hard_keywords=[],
            keywords=[],
            negative_hard_keywords=[],
            negative_keywords=[],
            gender_terms=[],
            disease_terms=[],
            comment_terms=[],
        )

        result = compile_legacy_state_rules(
            "yes, search now",
            state,
            turn_id="legacy-invalid-distance",
            allowed_specialties=("치과",),
        )

        self.assertEqual(result.issues, ())
        self.assertEqual(result.rules.hard.geography.max_km, 5.0)
        self.assertEqual(
            result.rules.hard.geography.provenance.source,
            "default_5km",
        )

    def test_legacy_adapter_keeps_geocoded_station_as_point_scope(self):
        state = SimpleNamespace(
            specialty="치과",
            location="Jamsil Station",
            district="송파구",
            latitude=37.51331105877401,
            longitude=127.10023101886318,
            is_citywide_search=False,
            search_mode="zone",
            max_distance_km=5.0,
            travel_confidence=0.6,
            hard_keywords=[],
            keywords=[],
            negative_hard_keywords=[],
            negative_keywords=[],
            gender_terms=[],
            disease_terms=[],
            comment_terms=[],
        )

        result = compile_legacy_state_rules(
            "Find a dentist near Jamsil Station",
            state,
            turn_id="legacy-station-point",
            allowed_specialties=("치과",),
        )

        self.assertEqual(result.issues, ())
        self.assertIsInstance(result.rules.hard.geography, DistanceRule)
        self.assertEqual(result.rules.hard.geography.max_km, 5.0)


if __name__ == "__main__":
    unittest.main()
