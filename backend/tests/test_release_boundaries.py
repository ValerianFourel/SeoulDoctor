"""Regression tests for public deployment boundaries."""

from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import llm_client
import location
import raw_review_store


class ProviderSelectionTests(unittest.TestCase):
    def test_openrouter_client_disables_sdk_retries_and_bounds_timeout(self):
        with patch.object(llm_client, "OpenAI") as openai_client:
            llm_client.build_openrouter_client("router-key")

        kwargs = openai_client.call_args.kwargs
        self.assertEqual(kwargs["max_retries"], 0)
        self.assertEqual(
            kwargs["timeout"],
            llm_client.COMPLETION_TIMEOUT_SECONDS,
        )

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

    def test_structured_completion_retries_invalid_content_with_more_room(self):
        calls = []

        class Completions:
            def create(self, **kwargs):
                calls.append(kwargs)
                content = "{" if len(calls) == 1 else '{"intent":"SEARCH"}'
                return SimpleNamespace(
                    choices=[SimpleNamespace(
                        message=SimpleNamespace(content=content)
                    )]
                )

        client = SimpleNamespace(
            chat=SimpleNamespace(completions=Completions())
        )

        payload, _ = llm_client.request_json_completion(
            client,
            model="openai/gpt-oss-120b",
            messages=[{"role": "system", "content": "Return JSON."}],
            max_completion_tokens=256,
        )

        self.assertEqual(payload, {"intent": "SEARCH"})
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["reasoning_effort"], "low")
        self.assertGreater(
            calls[1]["max_completion_tokens"],
            calls[0]["max_completion_tokens"],
        )

    def test_text_completion_retries_empty_content(self):
        calls = []

        class Completions:
            def create(self, **kwargs):
                calls.append(kwargs)
                content = None if len(calls) == 1 else "Grounded answer"
                return SimpleNamespace(
                    choices=[SimpleNamespace(
                        message=SimpleNamespace(content=content)
                    )]
                )

        client = SimpleNamespace(
            chat=SimpleNamespace(completions=Completions())
        )

        content, _ = llm_client.request_text_completion(
            client,
            model="openai/gpt-oss-120b",
            messages=[{"role": "system", "content": "Answer."}],
            max_completion_tokens=512,
            reasoning_effort="medium",
        )

        self.assertEqual(content, "Grounded answer")
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["reasoning_effort"], "medium")
        self.assertEqual(calls[1]["reasoning_effort"], "low")

    def test_structured_completion_retries_missing_required_keys(self):
        calls = []

        class Completions:
            def create(self, **kwargs):
                calls.append(kwargs)
                content = '{}' if len(calls) == 1 else '{"intent":"SEARCH"}'
                return SimpleNamespace(
                    choices=[SimpleNamespace(
                        message=SimpleNamespace(content=content)
                    )]
                )

        client = SimpleNamespace(
            chat=SimpleNamespace(completions=Completions())
        )
        payload, _ = llm_client.request_json_completion(
            client,
            model="openai/gpt-oss-120b",
            messages=[{"role": "system", "content": "Return JSON."}],
            max_completion_tokens=256,
            required_keys=("intent",),
        )

        self.assertEqual(payload["intent"], "SEARCH")
        self.assertEqual(len(calls), 2)

    def test_structured_completion_does_not_retry_auth_failure(self):
        calls = []

        class AuthenticationError(Exception):
            status_code = 401

        class Completions:
            def create(self, **kwargs):
                calls.append(kwargs)
                raise AuthenticationError("invalid key")

        client = SimpleNamespace(
            chat=SimpleNamespace(completions=Completions())
        )
        with self.assertRaises(AuthenticationError):
            llm_client.request_json_completion(
                client,
                model="openai/gpt-oss-120b",
                messages=[{"role": "system", "content": "Return JSON."}],
                max_completion_tokens=256,
            )

        self.assertEqual(len(calls), 1)

    def test_tool_completion_accepts_contentless_function_call(self):
        call = SimpleNamespace(function=SimpleNamespace(name="search"))
        message = SimpleNamespace(content=None, tool_calls=[call])
        completion = SimpleNamespace(
            choices=[SimpleNamespace(message=message)]
        )
        client = SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=MagicMock(return_value=completion))
        ))

        returned, _ = llm_client.request_tool_completion(
            client,
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": "Search"}],
            tools=[],
            tool_choice="auto",
            max_completion_tokens=256,
            reasoning_effort="medium",
        )

        self.assertIs(returned, message)
        client.chat.completions.create.assert_called_once()

    def test_tool_completion_retries_truly_empty_message(self):
        calls = []

        class Completions:
            def create(self, **kwargs):
                calls.append(kwargs)
                message = (
                    SimpleNamespace(content=None, tool_calls=[])
                    if len(calls) == 1
                    else SimpleNamespace(content="Continue", tool_calls=[])
                )
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=message)]
                )

        client = SimpleNamespace(
            chat=SimpleNamespace(completions=Completions())
        )
        message, _ = llm_client.request_tool_completion(
            client,
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": "Search"}],
            tools=[],
            tool_choice="auto",
            max_completion_tokens=256,
            reasoning_effort="medium",
        )

        self.assertEqual(message.content, "Continue")
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1]["reasoning_effort"], "low")


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


class ContainerSmokeBoundaryTests(unittest.TestCase):
    def test_smoke_script_never_loads_the_project_env_file(self) -> None:
        repo_root = BACKEND_DIR.parent
        script = (repo_root / "scripts" / "smoke_hf_image.sh").read_text(
            encoding="utf-8"
        )
        dockerignore = (repo_root / ".dockerignore").read_text(encoding="utf-8")

        self.assertNotIn("--env-file", script)
        self.assertNotIn("backend/.env", script)
        self.assertIn("offline-smoke-not-a-secret", script)
        self.assertIn("trap cleanup EXIT HUP INT TERM", script)
        self.assertIn('docker rm --force "$container_name"', script)
        self.assertIn("--network none", script)
        self.assertNotIn("--publish", script)
        self.assertIn('$vectors:/smoke-input/chroma_db:ro', script)
        self.assertIn("CHROMA_PATH=/tmp/seouldoc-chroma", script)
        self.assertIn("cp -R /smoke-input/chroma_db /tmp/seouldoc-chroma", script)
        self.assertIn("**/.env", dockerignore)
        self.assertIn("backend/tests", dockerignore)
        self.assertIn("backend/patient_journey.py", dockerignore)
        self.assertIn("backend/tmp*.py", dockerignore)


class LocationCredentialSafetyTests(unittest.TestCase):
    def test_kakao_keyword_search_resolves_seoul_landmark(self) -> None:
        response = MagicMock()
        response.json.return_value = {
            "documents": [
                {
                    "place_name": "대흥역 6호선",
                    "address_name": "서울 마포구 대흥동 128-1",
                    "road_address_name": "",
                    "x": "126.942473188734",
                    "y": "37.5476479056751",
                }
            ]
        }
        reverse = {
            "address_korean": "서울 마포구 대흥동 128-1",
            "district": "마포구",
            "dong": "대흥동",
        }
        with (
            patch.object(location, "KAKAO_REST_API_KEY", "kakao-key"),
            patch.object(
                location.requests,
                "get",
                return_value=response,
            ) as get,
            patch.object(location, "kakao_reverse_geocode", return_value=reverse),
        ):
            result = location.kakao_keyword_search("Daeheung Station")

        self.assertEqual(result["place_name"], "대흥역 6호선")
        self.assertAlmostEqual(result["lat"], 37.5476479056751)
        self.assertAlmostEqual(result["lon"], 126.942473188734)
        self.assertEqual(result["district"], "마포구")
        request_url = get.call_args.args[0]
        request_params = get.call_args.kwargs["params"]
        self.assertNotIn("kakao-key", request_url)
        self.assertNotIn("kakao-key", str(request_params))

    def test_address_verification_uses_kakao_keyword_fallback(self) -> None:
        expected = {
            "place_name": "대흥역 6호선",
            "lat": 37.5476479056751,
            "lon": 126.942473188734,
            "address_korean": "서울 마포구 대흥동 128-1",
            "district": "마포구",
            "dong": "대흥동",
        }
        with (
            patch.object(location, "GOOGLE_MAPS_API_KEY", ""),
            patch.object(location, "KAKAO_REST_API_KEY", "kakao-key"),
            patch.object(location, "kakao_geocode", return_value=None),
            patch.object(
                location,
                "kakao_keyword_search",
                return_value=expected,
                create=True,
            ) as keyword_search,
        ):
            result = location.verify_and_standardize_address("대흥역")

        self.assertEqual(result, expected)
        keyword_search.assert_called_once_with("대흥역", consent=None)

    def test_google_request_error_does_not_log_query_string_key(self) -> None:
        test_key = "unit-test-google-key"
        error = location.requests.exceptions.HTTPError(
            "403 Client Error for url: "
            f"https://maps.googleapis.com/maps/api/geocode/json?key={test_key}"
        )

        with (
            patch.object(location, "GOOGLE_MAPS_API_KEY", test_key),
            patch.object(location.requests, "get", side_effect=error),
            self.assertLogs(location.logger, level="ERROR") as captured,
        ):
            self.assertIsNone(location.google_maps_geocode("Gangnam Station"))

        self.assertNotIn(test_key, "\n".join(captured.output))


if __name__ == "__main__":
    unittest.main()
