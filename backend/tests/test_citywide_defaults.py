import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


class CitywideDefaultsTests(unittest.TestCase):
    def test_effective_function_preserves_existing_state_contract(self):
        source = Path(__file__).resolve().parents[1] / "utils.py"
        definitions = [node for node in ast.parse(source.read_text()).body
                       if isinstance(node, ast.FunctionDef)
                       and node.name == "ensure_city_wide_defaults"]
        namespace = {"State": SimpleNamespace, "CookieConsent": object,
                     "privacy_safe_log": Mock()}
        exec(compile(ast.Module(body=[definitions[-1]], type_ignores=[]),
                     str(source), "exec"), namespace)
        function = namespace["ensure_city_wide_defaults"]
        cases = [(None, None, None, True), (" Seoul ", None, None, True),
                 ("서울", None, None, True), ("서울시", None, None, True),
                 ("seoul city", None, None, True),
                 ("서울특별시", None, None, False),
                 ("Gangnam", None, None, False),
                 ("Seoul", 37.5, None, False),
                 (None, None, "용산구", False), (None, 0.0, None, True)]
        for location, latitude, district, citywide in cases:
            with self.subTest(location=location, latitude=latitude, district=district):
                state = SimpleNamespace(location=location, latitude=latitude,
                    longitude=127.0, district=district, dong="kept",
                    address_korean="kept", max_distance_km=5.0,
                    search_mode="auto", travel_label="Nearby", travel_confidence=0.5,
                    is_citywide_search=False, specialty="치과", keywords=["kind"])
                before = vars(state).copy()
                self.assertIs(function(state, object()), state)
                if citywide:
                    before.update(location=None, latitude=None, longitude=None,
                        district=None, dong=None, address_korean=None,
                        max_distance_km=25.0, search_mode="distance",
                        travel_label="Anywhere in Seoul", travel_confidence=1.0,
                        is_citywide_search=True)
                self.assertEqual(vars(state), before)


if __name__ == "__main__":
    unittest.main()
