"""Regression tests for public deployment boundaries."""

from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock, patch


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import llm_client
import raw_review_store


class ProviderSelectionTests(unittest.TestCase):
    def test_openrouter_configuration_never_falls_back_to_groq(self) -> None:
        with (
            patch.object(llm_client, "LLM_PROVIDER", "openrouter"),
            patch.object(llm_client, "OPENROUTER_API_KEY", ""),
            patch.object(llm_client, "GROQ_API_KEY", "groq-key"),
            patch.object(llm_client, "Groq") as groq_client,
        ):
            self.assertIsNone(llm_client.build_llm_client())

        groq_client.assert_not_called()

    def test_unknown_provider_is_rejected(self) -> None:
        with patch.object(llm_client, "LLM_PROVIDER", "unknown"):
            with self.assertRaisesRegex(ValueError, "Unsupported LLM_PROVIDER"):
                llm_client.build_llm_client()


class SlidingWindowRateLimiterTests(unittest.TestCase):
    def test_blocks_only_after_client_exhausts_window(self) -> None:
        from rate_limit import SlidingWindowRateLimiter

        limiter = SlidingWindowRateLimiter(max_requests=2, window_seconds=60)

        first = limiter.check("client-a", now=0.0)
        second = limiter.check("client-a", now=0.0)
        blocked = limiter.check("client-a", now=0.0)
        other_client = limiter.check("client-b", now=0.0)
        after_window = limiter.check("client-a", now=60.0)

        self.assertTrue(first.allowed)
        self.assertEqual(first.remaining, 1)
        self.assertTrue(second.allowed)
        self.assertEqual(second.remaining, 0)
        self.assertFalse(blocked.allowed)
        self.assertEqual(blocked.retry_after_seconds, 60)
        self.assertTrue(other_client.allowed)
        self.assertTrue(after_window.allowed)


class HuggingFaceDatasetDownloadTests(unittest.TestCase):
    def test_private_review_download_uses_hf_token(self) -> None:
        response = MagicMock()
        response.__enter__.return_value = response
        response.__exit__.return_value = False
        response.iter_content.return_value = [b"parquet"]

        with TemporaryDirectory() as temporary_directory:
            target = Path(temporary_directory) / "reviews.parquet"
            with (
                patch.dict(
                    "os.environ",
                    {
                        "ENABLE_RAW_REVIEWS": "true",
                        "RAW_REVIEWS_PATH": str(target),
                        "HF_TOKEN": "private-token",
                    },
                    clear=False,
                ),
                patch.object(raw_review_store.requests, "get", return_value=response) as get,
            ):
                result = raw_review_store.ensure_raw_review_parquet()

        self.assertEqual(result, str(target.resolve()))
        get.assert_called_once_with(
            raw_review_store.RAW_REVIEWS_URL,
            stream=True,
            timeout=(15, 180),
            headers={"Authorization": "Bearer private-token"},
        )


if __name__ == "__main__":
    unittest.main()
