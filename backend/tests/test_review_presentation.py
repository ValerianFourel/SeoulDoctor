import unittest
from backend.tests.test_evidence_response import fallback_response
from models import serialize_results_for_chat


def review(text, identity="review:fixture"):
    return {"place_id": "fixture-clinic", "evidence_id": identity, "text": text,
            "is_verbatim": True, "source_type": "verbatim_review",
            "source_locator": "review_snapshot:fixture-clinic:0",
            "review_source_sha256": "a" * 64,
            "matched_constraint_ids": ["evidence:fixture"]}


class ReviewPresentationTests(unittest.TestCase):
    def test_reply_is_concise_and_api_retains_identity(self):
        source = review("The nurse explained the paperwork clearly.")
        reply, cards = fallback_response([{"place_id": "fixture-clinic",
             "retrieval_evidence": [source]}], {"retrieval_execution_status": "partial"}, "English")
        self.assertIn("search did not finish", reply)
        self.assertNotIn(source["text"], reply)
        self.assertNotIn("review:fixture", reply)
        self.assertNotIn("evidence:fixture", reply)
        item = serialize_results_for_chat(cards, include_debug=False)[0]["retrieval_evidence"][0]
        for key in ("text", "evidence_id", "place_id", "source_locator", "review_source_sha256"):
            self.assertEqual(item[key], source[key])


class ReviewEligibilityTests(unittest.TestCase):
    def test_language_aware_filter(self):
        from review_presentation import useful_review
        for text in ("1", "123 456", "🙄", "👍🏻", "ㅣ", "ㅋㅋ", "123 🙄", "1️⃣"):
            with self.subTest(text=text):
                self.assertFalse(useful_review(text))
        for text in ("Great", "Very good", "친절해요", "친절한 병원", "The nurse explained everything clearly 👍🏻.",
                     "의사선생님이검사결과를자세히설명해주셨어요 😊", "간호사가 검사 절차를 설명했어요.",
                     "Friendly doctor but rude reception 😞"):
            with self.subTest(text=text):
                self.assertTrue(useful_review(text))

    def test_language_preference_persists_across_code_switching(self):
        from review_presentation import response_language
        self.assertEqual(response_language("한국어로 답해주세요", "English", established=True), ("Korean", "Korean"))
        self.assertEqual(response_language("영어로 답해주세요", "Korean", established=True), ("English", "English"))
        self.assertEqual(response_language("Gangnam 피부과 please", "Korean", established=True), ("Korean", None))
        self.assertEqual(response_language("한국어로 답해주세요", explicit="English"), ("Korean", "Korean"))
        self.assertEqual(response_language("근처에 있나요", explicit="English"), ("English", "English"))
        self.assertEqual(response_language("Does the staff speak English?", "Korean", established=True), ("Korean", None))
        self.assertEqual(response_language("피부과를 찾고 있어요"), ("Korean", None))


class TranslationTests(unittest.TestCase):
    def prepare(self, texts, translations=None, *, error=None, language="English", key="fixture-key"):
        from unittest.mock import patch
        from review_presentation import prepare_review_presentations
        cards = [{"retrieval_evidence": [review(text, f"review:{index}") for index, text in enumerate(texts)]}]
        with patch("review_presentation.requests.post") as post:
            post.side_effect = error
            post.return_value.json.return_value = {"data": {"translations": [
                {"translatedText": text} for text in (translations or [])]}}
            prepare_review_presentations(cards, language, translation_api_key=key)
        return cards[0]["retrieval_evidence"], post

    def test_simple_api_request_preserves_original_and_emoji(self):
        source = "의사는 친절하지만 접수 직원은 설명을 해주지 않았어요 😞"
        translated = "The doctor was kind, but reception did not explain things 😞"
        items, post = self.prepare(["1", "🙄", "Very good", source], [translated])
        self.assertEqual([item["presentation"]["status"] for item in items], ["hidden", "hidden", "original", "translated"])
        self.assertEqual(items[3]["text"], source)
        self.assertEqual(items[3]["presentation"]["text"], translated)
        post.assert_called_once_with(
            "https://translation.googleapis.com/language/translate/v2",
            headers={"X-Goog-Api-Key": "fixture-key"},
            json={"q": [source], "target": "en", "format": "text", "model": "nmt"},
            timeout=(2, 5))

    def test_invalid_or_failed_translation_retains_original(self):
        import requests
        source = "간호사가 검사 절차를 자세하게 설명했어요 👍🏻"
        for translated, error in (([], None), (["The nurse explained the procedure."], None),
                                  ([None], None), (None, requests.Timeout()),
                                  (["한국어 그대로 👍🏻"], None)):
            with self.subTest(translated=translated, error=error):
                items, _ = self.prepare([source], translated, error=error)
                self.assertEqual(items[0]["presentation"]["status"], "unavailable")
                self.assertEqual(items[0]["text"], source)

    def test_missing_api_key_does_not_call_any_service(self):
        source = "간호사가 절차를 설명했어요 😊"
        items, post = self.prepare([source], key="")
        post.assert_not_called()
        self.assertEqual(items[0]["text"], source)
        self.assertEqual(items[0]["presentation"]["status"], "unavailable")

    def test_malformed_response_does_not_erase_original(self):
        from unittest.mock import patch
        from review_presentation import prepare_review_presentations
        for payload in ({}, {"data": None}, {"data": {"translations": "bad"}}):
            cards = [{"retrieval_evidence": [review("검사 내용을 자세하게 설명해 주었어요")]}]
            with patch("review_presentation.requests.post") as post:
                post.return_value.json.return_value = payload
                prepare_review_presentations(cards, "English", translation_api_key="fixture")
            self.assertEqual(cards[0]["retrieval_evidence"][0]["presentation"]["status"], "unavailable")

    def test_same_language_and_junk_skip_translation(self):
        items, post = self.prepare(["The receptionist explained the process 👍.", "1"])
        post.assert_not_called()
        self.assertEqual(items[0]["presentation"]["status"], "original")
        self.assertEqual(items[1]["presentation"]["status"], "hidden")

    def test_english_to_korean_translation(self):
        items, _ = self.prepare(["The nurse explained the process 😊"],
                                ["간호사가 절차를 설명해 주었어요 😊"], language="Korean")
        self.assertEqual(items[0]["presentation"]["status"], "translated")

    def test_api_retains_hidden_source_and_presentation(self):
        items, _ = self.prepare(["1", "The nurse explained everything clearly."])
        serialized = serialize_results_for_chat([{"place_id": "fixture-clinic", "review_language": "English",
                                                 "retrieval_evidence": items}], include_debug=False)
        self.assertEqual(serialized[0]["review_language"], "English")
        self.assertEqual(serialized[0]["retrieval_evidence"], items)

    def test_duplicate_sources_share_one_translation_without_losing_ids(self):
        source = "간호사가 검사 절차를 자세하게 설명했어요 😊"
        translation = "The nurse explained the examination procedure in detail 😊"
        items, post = self.prepare([source, source], [translation])
        self.assertEqual(post.call_args.kwargs["json"]["q"], [source])
        self.assertEqual([item["evidence_id"] for item in items], ["review:0", "review:1"])
        self.assertEqual([item["presentation"]["text"] for item in items], [translation, translation])

    def test_budget_exclusion_keeps_original_and_does_not_call_api(self):
        from review_presentation import MAX_TRANSLATION_CHARACTERS
        source = "간호사가 검사 절차를 자세하게 설명했어요 " * MAX_TRANSLATION_CHARACTERS
        items, post = self.prepare([source])
        post.assert_not_called()
        self.assertEqual(items[0]["text"], source)
        self.assertEqual(items[0]["presentation"], {"status": "unavailable", "language": "English"})

    def test_supporting_group_reviews_are_not_lost(self):
        source = review("The nurse explained the paperwork clearly.")
        reply, cards = fallback_response([{"place_id": "fixture-clinic",
            "retrieval_evidence_groups": {"supporting": [source]}}], {}, "English")
        self.assertEqual(cards[0]["retrieval_evidence"][0]["text"], source["text"])
        self.assertEqual(cards[0]["retrieval_evidence"][0]["presentation"]["status"], "original")

    def test_spelled_time_count_is_preserved_but_changed_duration_rejected(self):
        source = "설명을 제대로 듣지 못했고 2시간 넘게 기다렸어요."
        items, _ = self.prepare([source], ["I did not hear the explanation properly and waited over two hours."])
        self.assertEqual(items[0]["presentation"]["status"], "translated")
        items, _ = self.prepare([source], ["I waited over three hours."])
        self.assertEqual(items[0]["presentation"]["status"], "unavailable")
        self.assertEqual(items[0]["text"], source)

    def test_comment_heavy_cards_use_one_bounded_deduplicated_batch(self):
        from review_presentation import MAX_TRANSLATION_REVIEWS
        sources = [f"간호사가 설명을 해주었어요 {index}" for index in range(MAX_TRANSLATION_REVIEWS + 1)]
        translations = [f"The nurse explained things {index}" for index in range(MAX_TRANSLATION_REVIEWS)]
        items, post = self.prepare(sources, translations)
        self.assertEqual(len(post.call_args.kwargs["json"]["q"]), MAX_TRANSLATION_REVIEWS)
        self.assertEqual(items[-1]["presentation"]["status"], "unavailable")
        self.assertTrue(all(item["presentation"]["status"] == "translated" for item in items[:-1]))
