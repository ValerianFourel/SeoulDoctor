import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

with patch("dotenv.load_dotenv"):
    import location


class LocationAliasResolutionTests(unittest.TestCase):
    def setUp(self):
        self.correct = {
            "lat": 37.5636, "lon": 126.985,
            "district": "중구", "dong": "명동",
            "formatted_address": "대한민국 서울특별시 중구 명동",
        }
        self.wrong = {
            "lat": 37.588, "lon": 127.087,
            "district": "중랑구", "dong": "면목동",
            "formatted_address": "대한민국 서울특별시 중랑구 면목동",
        }
        self.enterContext(patch.object(location, "GOOGLE_MAPS_API_KEY", "test"))
        self.enterContext(patch.object(location, "KAKAO_REST_API_KEY", None))

    def test_alias_is_canonical_before_provider_request(self):
        def respond(url, *, params, timeout):
            resolved = self.correct if params["address"] == "서울 중구 명동" else self.wrong
            return Mock(json=lambda: {
                "status": "OK", "results": [{
                    "geometry": {"location": {"lat": resolved["lat"], "lng": resolved["lon"]}},
                    "formatted_address": resolved["formatted_address"],
                    "address_components": [],
                }],
            })

        with patch.object(location.requests, "get", side_effect=respond):
            for alias in ["myeondong", "Myeongdong", "Myeong-dong", "명동", "Seoul Myeongdong"]:
                with self.subTest(alias=alias):
                    result = location.verify_and_standardize_address(alias)
                    self.assertEqual(result["lat"], self.correct["lat"])

    def test_wrong_place_continues_to_next_provider(self):
        with (
            patch.object(location, "google_maps_geocode", side_effect=[self.wrong, self.correct]) as geocode,
            patch.object(location, "google_maps_place_search") as places,
        ):
            self.assertEqual(location.verify_and_standardize_address("myeondong"), self.correct)
            self.assertEqual(geocode.call_count, 2)
            places.assert_not_called()

    def test_all_mismatched_providers_leave_location_unresolved(self):
        with (
            patch.object(location, "google_maps_geocode", return_value=self.wrong),
            patch.object(location, "google_maps_place_search", return_value=self.wrong),
        ):
            self.assertIsNone(location.verify_and_standardize_address("명동"))

    def test_actual_myeonmok_and_precise_addresses_are_unchanged(self):
        for query in ["Myeonmok", "면목동", "Jonggak", "Myeongdong hotel 12"]:
            with self.subTest(query=query), patch.object(location, "google_maps_geocode", return_value=self.wrong) as geocode:
                self.assertEqual(location.verify_and_standardize_address(query), self.wrong)
                geocode.assert_called_once_with(query, add_seoul=False, consent=None)

    def test_myeongdong_station_keeps_station_precision(self):
        with patch.object(location, "google_maps_geocode", return_value=self.correct) as geocode:
            location.verify_and_standardize_address("Myeong-dong station")
            geocode.assert_called_once_with("서울 중구 명동역", add_seoul=False, consent=None)


if __name__ == "__main__":
    unittest.main()
