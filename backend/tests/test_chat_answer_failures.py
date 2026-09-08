"""Answer and translation failures must preserve owned originals through /chat."""

from copy import deepcopy
from hashlib import sha256
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import requests

from backend.tests import test_chat_response_preservation as fixtures
from llm_client import request_answer_completion
from models import State


ORIGINALS = (
    "Nurse was rude.",
    "간호사가 불친절했어요.",
    "The doctor explained clearly. The nurse was rude.",
)
SOURCE_FIELDS = (
    "place_id", "evidence_id", "text", "source_type", "is_verbatim",
    "review_source_sha256", "source_locator", "retrieval_roles",
)


def source(owner, index):
    text = ORIGINALS[index]
    identity = sha256(f"{owner}|{index}|{text}".encode()).hexdigest()[:20]
    return {
        "place_id": owner,
        "evidence_id": f"review:{identity}",
        "text": text,
        "source_index": index,
        "source_type": "verbatim_review",
        "source_field": "comment",
        "is_verbatim": True,
        "language": "ko" if index == 1 else "en",
        "review_source_sha256": "a" * 64,
        "source_locator": {"snapshot": "synthetic-reviews", "owner": owner, "row": index},
        "retrieval_roles": ["risk"] if index < 2 else ["support", "risk"],
        "private_probe": "must not reach the public API",
    }


def completion(content, *, finish_reason="stop"):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish_reason)],
        usage=SimpleNamespace(prompt_tokens=20, completion_tokens=10, total_tokens=30),
    )


class ChatAnswerFailureTests(unittest.TestCase):
    def setUp(self):
        self.harness = fixtures.ChatResponsePreservationTests()
        self.harness.setUp()
        self.addCleanup(self.harness.tearDown)
        self.stack = self.harness.stack
        self.main = self.harness.main
        self.boundary = fixtures.JsonModelBoundary([], [])
        self.harness._install_model_boundary(self.boundary)
        self.harness._install_geocoder()
        self.sources = {
            owner: [source(owner, index) for index in range(len(ORIGINALS))]
            for owner in self.harness.catalog["place_id"]
        }
        sources = self.sources
        base_adapter = self.main.CandidateRetrievalAdapter

        class Adapter(base_adapter):
            def rank(self, **kwargs):
                result = super().rank(**kwargs)
                frame = result.dataframe
                evidence = [deepcopy(sources[owner]) for owner in frame["place_id"]]
                frame["retrieval_evidence"] = evidence
                frame["retrieval_evidence_groups"] = [
                    {"supporting": [items[2]], "warnings": items,
                     "unverified": ["attribute:english_consultation"]}
                    for items in evidence
                ]
                return result

        self.stack.enter_context(patch.object(self.main, "CandidateRetrievalAdapter", Adapter))
        self.provider_create = Mock(side_effect=TimeoutError("synthetic answer timeout"))
        options = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=self.provider_create)))
        self.provider = SimpleNamespace(with_options=Mock(return_value=options))
        self.stack.enter_context(patch.object(self.main, "client", self.provider))
        self.stack.enter_context(patch.object(self.main, "request_answer_completion", request_answer_completion))
        self.stack.enter_context(patch.object(
            self.main.os, "getenv",
            side_effect=lambda name, default=None: "fixture-translation-key" if name == "GOOGLE_TRANSLATE_API_KEY" else default,
        ))
        self.harness.translation_request.side_effect = requests.Timeout("synthetic translation timeout")

    def _post(self, message="Find orthopedics near Jonggak.", state=None):
        self.boundary.intents.append("PROVIDE_INFO" if state is None else "CHANGE_CRITERIA")
        self.boundary.extractions.append(fixtures._extraction(location="Jonggak" if state is None else None))
        response = self.harness.client.post("/chat", json={
            "message": message,
            "current_state": State().model_dump() if state is None else state,
        })
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertTrue(body["response"].strip())
        self.assertNotIn("last_retrieval_metadata", body["state"])
        return body

    def _assert_originals(self, body, *, status="fallback"):
        cards = {card["place_id"]: card for card in body["results"]}
        self.assertEqual(set(cards), set(self.sources))
        for owner, card in cards.items():
            self.assertEqual(card["answer_status"], status)
            self.assertEqual(card["recommendation_status"], "not_established")
            if status == "fallback":
                self.assertEqual(card["answer_citations"], [])
            records = {item["evidence_id"]: item for item in card["retrieval_evidence"]}
            self.assertEqual(set(records), {item["evidence_id"] for item in self.sources[owner]})
            for expected in self.sources[owner]:
                actual = records[expected["evidence_id"]]
                self.assertEqual({field: actual[field] for field in SOURCE_FIELDS},
                                 {field: expected[field] for field in SOURCE_FIELDS})
                self.assertNotIn("private_probe", actual)
                self.assertNotIn("source_index", actual)
                self.assertNotEqual(actual["presentation"]["status"], "hidden")
            for role in ("supporting", "warnings"):
                self.assertTrue(card["retrieval_evidence_groups"][role])
                for item in card["retrieval_evidence_groups"][role]:
                    self.assertEqual(item["place_id"], owner)
                    self.assertEqual(item["text"], records[item["evidence_id"]]["text"])
                    self.assertNotIn("private_probe", item)
        self.harness.context_builder.assert_not_called()

    def _proposal(self, answer=None):
        original = self.sources["closest"][0]
        return {
            "answer": answer or "One patient reports a rude nurse [1]. Ask the clinic about nursing support before booking.",
            "assessments": [{
                "place_id": "closest", "requirement": "courteous nurses",
                "status": "contradicts", "basis": "patient_report", "staff_role": "nurse",
                "evidence_ids": [original["evidence_id"]],
                "explanation": "One patient reports a rude nurse.",
            }],
            "citations": [{
                "marker": 1, "place_id": "closest", "evidence_id": original["evidence_id"],
                "original_excerpt": original["text"],
            }],
        }

    def _responses(self, *responses):
        self.provider_create.reset_mock()
        self.provider.with_options.reset_mock()
        self.provider_create.side_effect = [
            response if isinstance(response, BaseException) else completion(json.dumps(response, ensure_ascii=False))
            for response in responses
        ]

    def _assert_bounded_calls(self, count):
        self.assertEqual(self.provider_create.call_count, count)
        self.assertEqual(self.provider.with_options.call_count, count)
        for call in self.provider.with_options.call_args_list:
            self.assertEqual(call.kwargs["max_retries"], 0)
            self.assertGreater(call.kwargs["timeout"], 0)
            self.assertLessEqual(call.kwargs["timeout"], 30)

    def test_invalid_citations_do_not_hide_originals_or_reach_verification(self):
        invalid = (
            {"place_id": "far"},
            {"evidence_id": "review:missing-source"},
            {"original_excerpt": "Nurse was kind."},
        )
        for changes in invalid:
            with self.subTest(changes=changes):
                proposal = self._proposal()
                proposal["citations"][0].update(changes)
                self._responses(proposal)
                body = self._post()
                self._assert_originals(body)
                self.assertNotEqual(body["response"], proposal["answer"])
                self._assert_bounded_calls(1)

    def test_semantic_rejection_blocks_unsupported_guarantee_and_retains_sources(self):
        proposal = self._proposal("This clinic guarantees kind nurses and fluent English consultations [1]. Book there.")
        self._responses(proposal, {"accepted": False, "issues": ["The nurse report and unconfirmed service do not support these guarantees."]})
        body = self._post()
        self._assert_originals(body)
        self.assertNotIn("guarantees", body["response"])
        self._assert_bounded_calls(2)
        verification = json.loads(self.provider_create.call_args_list[1].kwargs["messages"][-1]["content"])
        self.assertEqual(verification["proposal"], proposal)
        self.assertIn(ORIGINALS[0], [item["original_text"] for item in verification["search"]["evidence"]])
        self.assertEqual(verification["search"]["verified_service_facts"], [])

    def test_synthesis_timeout_preserves_originals_without_retry(self):
        self._responses(TimeoutError("synthetic synthesis timeout"))
        self._assert_originals(self._post())
        self._assert_bounded_calls(1)

    def test_verification_timeout_preserves_originals_and_discards_unverified_answer(self):
        proposal = self._proposal()
        self._responses(proposal, TimeoutError("synthetic verification timeout"))
        body = self._post()
        self._assert_originals(body)
        self.assertNotEqual(body["response"], proposal["answer"])
        self._assert_bounded_calls(2)

    def test_malformed_empty_or_wrong_shape_model_json_retains_originals(self):
        for content in ("{broken", "", "{}", "[]"):
            with self.subTest(content=content):
                self.provider_create.reset_mock()
                self.provider.with_options.reset_mock()
                self.provider_create.side_effect = [completion(content)]
                self._assert_originals(self._post())
                self._assert_bounded_calls(1)

    def test_truncated_completion_is_not_accepted_even_with_valid_json(self):
        self.provider_create.side_effect = [completion(json.dumps(self._proposal()), finish_reason="length")]
        self._assert_originals(self._post())
        self._assert_bounded_calls(1)

    def test_failed_translation_keeps_short_originals_with_accepted_answer(self):
        proposal = self._proposal()
        self._responses(proposal, {"accepted": True, "issues": []})
        body = self._post()
        self._assert_originals(body, status="generated")
        self.assertEqual(body["response"], proposal["answer"])
        for card in body["results"]:
            by_text = {item["text"]: item for item in card["retrieval_evidence"]}
            self.assertEqual(by_text[ORIGINALS[0]]["presentation"], {"status": "original", "language": "English"})
            self.assertEqual(by_text[ORIGINALS[1]]["presentation"], {"status": "unavailable", "language": "English"})
        self.harness.translation_request.assert_called_once()
        cited = next(card for card in body["results"] if card["place_id"] == "closest")["answer_citations"][0]
        self.assertEqual(cited["place_id"], "closest")
        self.assertEqual(cited["evidence_id"], self.sources["closest"][0]["evidence_id"])
        self.assertEqual(cited["original_excerpt"], ORIGINALS[0])
        self.assertEqual(cited["review_source_sha256"], "a" * 64)

    def test_translation_and_code_switching_preserve_same_source_identity_after_answer_failure(self):
        translations = {
            ORIGINALS[0]: "간호사가 불친절했어요.",
            ORIGINALS[1]: "Nurse was rude.",
            ORIGINALS[2]: "의사는 설명을 잘했지만 간호사는 불친절했어요.",
        }

        def translate(url, *, headers, json, timeout):
            self.assertEqual(headers, {"X-Goog-Api-Key": "fixture-translation-key"})
            payload = {"data": {"translations": [{"translatedText": translations[text]} for text in json["q"]]}}
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: payload)

        self.harness.translation_request.side_effect = translate
        first = self._post()
        korean = self._post("한국어로 답해 주세요. Keep Jonggak.", first["state"])
        english = self._post("영어로 답해 주세요. Keep Jonggak.", korean["state"])
        for body, language in ((first, "English"), (korean, "Korean"), (english, "English")):
            self._assert_originals(body)
            self.assertEqual(body["state"]["language_pref"], language)
            for card in body["results"]:
                self.assertEqual(card["review_language"], language)
                for item in card["retrieval_evidence"]:
                    translated = (language == "Korean") != (item["language"] == "ko")
                    expected = {"status": "translated" if translated else "original", "language": language}
                    if translated:
                        expected["text"] = translations[item["text"]]
                    self.assertEqual(item["presentation"], expected)
        self.assertEqual(self.harness.translation_request.call_count, 3)
        self.assertEqual([call.kwargs["json"]["target"] for call in self.harness.translation_request.call_args_list], ["en", "ko", "en"])
        self._assert_bounded_calls(3)


if __name__ == "__main__":
    unittest.main()
