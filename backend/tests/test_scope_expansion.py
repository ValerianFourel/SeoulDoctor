import sys
import unittest
from dataclasses import replace
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from search.rules import RulesCompiler
from search.scope import ScopeBuilder


class ScopeExpansionTests(unittest.TestCase):
    def rules(self, radius=1.0, specialty="정형외과", hard_keywords=()):
        compilation = RulesCompiler().compile(
            original_query=f"Find a doctor within {radius} km",
            turn_id="expansion-test",
            proposal={
                "specialty": specialty,
                "location": "current map position",
                "latitude": 37.56,
                "longitude": 126.98,
                "hard_keywords": list(hard_keywords),
            },
        )
        self.assertEqual(compilation.issues, ())
        return compilation.rules

    def facilities(self, distance_km, category="정형외과"):
        return pd.DataFrame([
            {"place_id": "unrelated", "category": "피부과", "lat": 37.56, "lon": 126.98},
            {"place_id": "specialist", "category": category,
             "lat": 37.56 + distance_km / 111.2, "lon": 126.98},
        ])

    def search(self, facilities, rules, allow=True):
        return ScopeBuilder().build_with_expansion(
            facilities, rules, index_version="test", allow_expansion=allow,
        )

    def test_stops_at_first_radius_with_requested_specialty(self):
        initial = self.rules()
        result = self.search(self.facilities(3), initial)
        self.assertEqual(result.attempted_radii_km, (1.0, 2.0, 5.0))
        self.assertEqual(result.scope.facility_ids, ("specialist",))
        self.assertEqual(initial.hard.geography.max_km, 1.0)
        self.assertEqual(result.rules.hard.geography.max_km, 5.0)
        self.assertEqual(result.rules.hard.specialty_ids, initial.hard.specialty_ids)
        self.assertNotEqual(result.rules.rules_hash, initial.rules_hash)
        self.assertEqual(result.scope.descriptor.rules_hash, result.rules.rules_hash)
        self.assertEqual(result.rules.hard.geography.provenance.source, "verified_context")
        self.assertIsNone(result.rules.hard.geography.provenance.source_span)

    def test_hard_radius_never_expands(self):
        initial = self.rules()
        result = self.search(self.facilities(3), initial, allow=False)
        self.assertEqual(result.attempted_radii_km, (1.0,))
        self.assertEqual(result.scope.facility_ids, ())
        self.assertIs(result.rules, initial)

    def test_one_matching_facility_does_not_trigger_padding(self):
        result = self.search(self.facilities(0.5), self.rules())
        self.assertEqual(result.attempted_radii_km, (1.0,))
        self.assertEqual(result.scope.facility_ids, ("specialist",))

    def test_expansion_starts_above_actual_radius_and_stops_at_25(self):
        for start, attempted in [(3, (3, 5, 10, 25)), (30, (30,))]:
            with self.subTest(start=start):
                result = self.search(self.facilities(40), self.rules(start))
                self.assertEqual(result.attempted_radii_km, attempted)
                self.assertEqual(result.scope.facility_ids, ())

    def test_prohibited_facility_remains_excluded_at_every_radius(self):
        initial = self.rules()
        initial = replace(initial, hard=replace(
            initial.hard, prohibited_facility_ids=frozenset({"specialist"}),
        ))
        result = self.search(self.facilities(3), initial)
        self.assertEqual(result.attempted_radii_km, (1, 2, 5, 10, 25))
        self.assertEqual(result.scope.facility_ids, ())
        self.assertEqual(result.rules.hard.prohibited_facility_ids, frozenset({"specialist"}))
        self.assertEqual(result.rules.soft, initial.soft)
        self.assertEqual(result.rules.evidence, initial.evidence)

    def test_unresolved_specialty_never_expands(self):
        initial = self.rules(specialty=None)
        empty = self.facilities(40).iloc[0:0]
        result = self.search(empty, initial)
        self.assertEqual(result.attempted_radii_km, (1.0,))
        self.assertIs(result.rules, initial)

    def test_required_attributes_are_not_relaxed(self):
        initial = self.rules(hard_keywords=("parking",))
        facilities = self.facilities(3).assign(amenities=[{}, {}])
        result = self.search(facilities, initial)
        self.assertEqual(result.attempted_radii_km, (1, 2, 5, 10, 25))
        self.assertEqual(result.scope.facility_ids, ())
        self.assertEqual(result.rules.hard.required_attributes, initial.hard.required_attributes)
        self.assertEqual(result.rules.evidence, initial.evidence)

    def test_district_scope_never_expands(self):
        initial = RulesCompiler().compile(
            original_query="orthopedics in Jongno-gu", turn_id="district-test",
            proposal={"specialty": "정형외과", "location": "Jongno-gu"},
        ).rules
        facilities = self.facilities(3).assign(address="서울 중구")
        result = self.search(facilities, initial)
        self.assertEqual(result.attempted_radii_km, ())
        self.assertEqual(result.scope.facility_ids, ())
        self.assertIs(result.rules, initial)


if __name__ == "__main__":
    unittest.main()
