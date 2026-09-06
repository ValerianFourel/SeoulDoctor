"""Regression checks for consent-aware location diagnostics."""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


try:
    import dotenv  # noqa: F401
except ModuleNotFoundError:
    dotenv_stub = types.ModuleType("dotenv")
    dotenv_stub.load_dotenv = lambda *args, **kwargs: False
    sys.modules["dotenv"] = dotenv_stub


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from cookies import CookieConsent  # noqa: E402
import location  # noqa: E402
import main  # noqa: E402
from models import State  # noqa: E402


class _GeocodeResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self):
        return {
            "status": "OK",
            "results": [{
                "geometry": {"location": {"lat": 37.555, "lng": 126.923}},
                "formatted_address": "Seoul Mapo-gu Seogyo-dong",
                "address_components": [{
                    "types": ["sublocality_level_1"],
                    "long_name": "Mapo-gu",
                }],
            }],
        }


class LocationLoggingPrivacyTests(unittest.TestCase):
    raw_location = "Private Landmark 123, Unit 4"

    def test_no_analytics_consent_does_not_log_raw_location(self):
        with (
            patch.object(location, "GOOGLE_MAPS_API_KEY", "test-key"),
            patch.object(location.requests, "get", return_value=_GeocodeResponse()),
            self.assertNoLogs(location.logger),
        ):
            result = location.google_maps_geocode(
                self.raw_location,
                consent=CookieConsent(analytics=False),
            )

        self.assertEqual(result["district"], "Mapo-gu")

    def test_analytics_consent_allows_verbose_location_diagnostics(self):
        with (
            patch.object(location, "GOOGLE_MAPS_API_KEY", "test-key"),
            patch.object(location.requests, "get", return_value=_GeocodeResponse()),
            self.assertLogs(location.logger, level="INFO") as captured,
        ):
            location.google_maps_geocode(
                self.raw_location,
                consent=CookieConsent(analytics=True),
            )

        self.assertIn(self.raw_location[:30], "\n".join(captured.output))

    def test_state_merge_logs_no_medical_preference_values(self):
        private_values = [
            "private-condition",
            "private-preference",
            "private-exclusion",
            "private-avoidance",
        ]
        extracted = {
            "travel_label": "Moderate",
            "hard_keywords": [private_values[0]],
            "soft_keywords": [private_values[1]],
            "negative_hard_keywords": [private_values[2]],
            "negative_keywords": [private_values[3]],
        }

        with self.assertLogs(main.logger, level="INFO") as captured:
            result = main.merge_extraction_into_state(State(), extracted)

        logs = "\n".join(captured.output)
        for private_value in private_values:
            self.assertNotIn(private_value, logs)
        self.assertEqual(result.hard_keywords, [private_values[0]])


if __name__ == "__main__":
    unittest.main()
