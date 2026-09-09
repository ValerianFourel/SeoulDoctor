"""The answer boundary makes one request and rejects partial provider output."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_client import request_answer_completion


class AnswerCompletionTests(unittest.TestCase):
    def client(self, *, content='{"answer":"safe"}', finish_reason="stop", choices=True):
        client = Mock()
        message = SimpleNamespace(content=content)
        completion = SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish_reason)] if choices else [])
        client.with_options.return_value.chat.completions.create.return_value = completion
        return client

    def request(self, client, model="synthetic-answer-model"):
        return request_answer_completion(
            client, model=model,
            messages=[{"role": "user", "content": "synthetic request"}],
            max_completion_tokens=3072, timeout_seconds=17.5,
            response_schema={"type": "object", "properties": {"answer": {"type": "string"}},
                             "required": ["answer"], "additionalProperties": False},
        )

    def test_valid_json_uses_explicit_timeout_and_disables_provider_retries(self):
        client = self.client()
        with patch("llm_client.LLM_PROVIDER", "openrouter"):
            value, _ = self.request(client)
        self.assertEqual(value, {"answer": "safe"})
        client.with_options.assert_called_once_with(max_retries=0, timeout=17.5)
        create = client.with_options.return_value.chat.completions.create
        create.assert_called_once()
        self.assertEqual(create.call_args.kwargs["max_tokens"], 3072)
        self.assertNotIn("max_completion_tokens", create.call_args.kwargs)
        self.assertTrue(create.call_args.kwargs["extra_body"]["provider"]["require_parameters"])
        self.assertEqual(create.call_args.kwargs["temperature"], 0.0)
        self.assertEqual(create.call_args.kwargs["response_format"]["type"], "json_schema")
        self.assertTrue(create.call_args.kwargs["response_format"]["json_schema"]["strict"])

    def test_gpt54_routes_to_supported_standard_openai_endpoint(self):
        client = self.client()
        with patch("llm_client.LLM_PROVIDER", "openrouter"):
            value, _ = self.request(client, "openai/gpt-5.4")
        self.assertEqual(value, {"answer": "safe"})
        client.with_options.assert_called_once_with(max_retries=0, timeout=17.5)
        create = client.with_options.return_value.chat.completions.create
        create.assert_called_once()
        params = create.call_args.kwargs
        self.assertNotIn("temperature", params)
        self.assertEqual(params["reasoning_effort"], "none")
        self.assertEqual(params["max_tokens"], 3072)
        self.assertEqual(params["extra_body"]["provider"], {
            "order": ["openai"], "allow_fallbacks": False, "require_parameters": True,
        })
        self.assertTrue(params["response_format"]["json_schema"]["strict"])
        client = self.client(finish_reason="length")
        with patch("llm_client.LLM_PROVIDER", "openrouter"):
            with self.assertRaisesRegex(ValueError, "answer_completion_incomplete"):
                self.request(client, "openai/gpt-5.4")

    def test_complete_json_with_nonstop_finish_reason_is_rejected_without_retry(self):
        for reason in ("length", "content_filter", "tool_calls", None):
            with self.subTest(reason=reason):
                client = self.client(finish_reason=reason)
                with self.assertRaisesRegex(ValueError, "answer_completion_incomplete"):
                    self.request(client)
                client.with_options.return_value.chat.completions.create.assert_called_once()

    def test_empty_and_malformed_completions_never_become_answers(self):
        for content in (None, "", " ", "{", "[]", '{"answer":'):
            with self.subTest(content=content):
                client = self.client(content=content)
                with self.assertRaises(ValueError):
                    self.request(client)
                client.with_options.return_value.chat.completions.create.assert_called_once()

    def test_no_choices_and_transport_timeout_are_not_retried(self):
        client = self.client(choices=False)
        with self.assertRaisesRegex(ValueError, "answer_completion_incomplete"):
            self.request(client)
        client.with_options.return_value.chat.completions.create.assert_called_once()
        client = self.client()
        client.with_options.return_value.chat.completions.create.side_effect = TimeoutError("synthetic timeout")
        with self.assertRaises(TimeoutError):
            self.request(client)
        client.with_options.return_value.chat.completions.create.assert_called_once()


if __name__ == "__main__":
    unittest.main()
