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
        self.assertIn("I may have missed relevant patient reviews", reply)
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
            timeout=(3, 12))

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

    def test_korean_and_english_written_durations_keep_the_same_count(self):
        examples = (
            ("두 시간 기다렸어요", "I waited two hours."),
            ("한시간 넘게 걸렸어요", "It took over an hour."),
            ("삼십 분 기다렸어요", "I waited thirty minutes."),
            ("스물한 시간 걸렸어요", "It took twenty-one hours."),
            ("일주일 기다렸어요", "I waited one week."),
        )
        for source, translation in examples:
            with self.subTest(source=source):
                items, _ = self.prepare([source], [translation])
                self.assertEqual(items[0]["presentation"]["status"], "translated")
                self.assertEqual(items[0]["text"], source)
                reverse, _ = self.prepare([translation], [source], language="Korean")
                self.assertEqual(reverse[0]["presentation"]["status"], "translated")

    def test_written_durations_still_reject_changed_added_and_removed_counts(self):
        for source, translation in (
            ("두 시간 기다렸어요", "I waited three hours."),
            ("삼십 분 기다렸어요", "I waited thirteen minutes."),
            ("기다렸어요", "I waited an hour."),
            ("한시간 기다렸어요", "I waited."),
        ):
            with self.subTest(source=source, translation=translation):
                items, _ = self.prepare([source], [translation])
                self.assertEqual(items[0]["presentation"]["status"], "unavailable")
                self.assertEqual(items[0]["text"], source)

    def test_thousands_separators_do_not_change_amount_but_digits_do(self):
        source = "1,000원 냈어요"
        items, _ = self.prepare([source], ["I paid 1000 won."])
        self.assertEqual(items[0]["presentation"]["status"], "translated")
        for translation in ("I paid 100 won.", "I paid 10000 won.", "I paid 1.000 won."):
            with self.subTest(translation=translation):
                items, _ = self.prepare([source], [translation])
                self.assertEqual(items[0]["presentation"]["status"], "unavailable")
                self.assertEqual(items[0]["text"], source)

    def test_people_counts_and_unqualified_english_cardinals_match(self):
        source = "두분이 안내했고 발목은 90도였어요."
        items, _ = self.prepare([source], ["Two of them guided me and my ankle was at 90 degrees."])
        self.assertEqual(items[0]["presentation"]["status"], "translated")
        changed, _ = self.prepare([source], ["Three of them guided me and my ankle was at 90 degrees."])
        self.assertEqual(changed[0]["presentation"]["status"], "unavailable")
        self.assertEqual(changed[0]["text"], source)

    def test_won_units_and_per_session_do_not_change_the_amount(self):
        examples = (
            ("테스트 하나 없이 1회에 22만원 냈어요.", "There wasn't a single test and I paid 220,000 won per session."),
            ("한 회에 2천원 냈어요.", "I paid 2000 won per session."),
            ("1회에 1.5만원 냈어요.", "I paid 15,000 won per session."),
        )
        for source, translation in examples:
            with self.subTest(source=source):
                items, _ = self.prepare([source], [translation])
                self.assertEqual(items[0]["presentation"]["status"], "translated")
                self.assertEqual(items[0]["text"], source)
                reverse, _ = self.prepare([translation], [source], language="Korean")
                self.assertEqual(reverse[0]["presentation"]["status"], "translated")

    def test_currency_and_session_counts_still_reject_changed_values(self):
        source = "1회에 22만원 냈어요."
        for translation in (
            "I paid 22,000 won per session.",
            "I paid 220,000 won for two sessions.",
            "I paid 220,000 dollars per session.",
            "I paid 220,000 won.",
        ):
            with self.subTest(translation=translation):
                items, _ = self.prepare([source], [translation])
                self.assertEqual(items[0]["presentation"]["status"], "unavailable")
                self.assertEqual(items[0]["text"], source)

    def test_spelled_currency_scale_requires_a_currency_unit(self):
        for source, translation in (
            ("이제 건강해질 일만 남았습니다.", "Now only getting healthy remains."),
            ("이제 건강해질 일만 남았습니다.", "There is one thing left: getting healthy."),
            ("일만원 냈어요.", "I paid 10,000 won."),
            ("이천 원 냈어요.", "I paid 2,000 won."),
            ("일주일 기다렸어요.", "I waited one week."),
        ):
            with self.subTest(source=source, translation=translation):
                items, _ = self.prepare([source], [translation])
                self.assertEqual(items[0]["presentation"]["status"], "translated")
                self.assertEqual(items[0]["text"], source)
        for source, translation in (
            ("일만원 냈어요.", "I paid 1,000 won."),
            ("이천 원 냈어요.", "I paid 3,000 won."),
            ("일주일 기다렸어요.", "I waited two weeks."),
        ):
            with self.subTest(source=source):
                items, _ = self.prepare([source], [translation])
                self.assertEqual(items[0]["presentation"]["status"], "unavailable")

    def test_pronouns_and_superlatives_do_not_create_a_count(self):
        for source, translation in (
            ("아무도 안내해주지 않았어요.", "No one told me where to go."),
            ("정말 좋은 병원이에요.", "It is one of the best clinics."),
            ("엑스레이 한번 찍어보자고 했고 세 명이 있었어요.", "They asked for an X-ray, and three people were there. No one helped."),
        ):
            with self.subTest(source=source):
                items, _ = self.prepare([source], [translation])
                self.assertEqual(items[0]["presentation"]["status"], "translated")
        for source, translation in (
            ("한 명이 안내했어요.", "Two people guided me."),
            ("세 명 중 한 명이 안내했어요.", "Two of the three people guided me."),
            ("엑스레이 한번 찍었어요.", "They took two X-rays."),
        ):
            with self.subTest(source=source):
                items, _ = self.prepare([source], [translation])
                self.assertEqual(items[0]["presentation"]["status"], "unavailable")

    def test_xray_article_is_not_a_count_and_explicit_counts_survive(self):
        for source, translation in (
            ("발이 아파서 엑스레이 찍었어요.", "I had an X-ray because my foot hurt."),
            ("엑스레이 한번 찍어보자고 했어요.", "They suggested getting an X-ray."),
            ("엑스레이 한번 찍었어요.", "I had one X-ray."),
            ("엑스레이 두 번 찍었어요.", "I had two X-rays."),
        ):
            with self.subTest(source=source):
                items, _ = self.prepare([source], [translation])
                self.assertEqual(items[0]["presentation"]["status"], "translated")
                self.assertEqual(items[0]["text"], source)
        for source, translation in (
            ("엑스레이 2번 찍었어요.", "I had three X-rays."),
            ("엑스레이 2번 찍어보자고 했어요.", "They suggested getting an X-ray."),
            ("엑스레이 한번 찍었어요.", "I had two X-rays."),
        ):
            with self.subTest(source=source):
                items, _ = self.prepare([source], [translation])
                self.assertEqual(items[0]["presentation"]["status"], "unavailable")

    def test_occasional_and_never_once_preserve_literal_counts(self):
        for source, translation in (
            ("전에도 한번씩 왔었어요.", "I have visited occasionally before."),
            ("가끔 한번씩 왔었어요.", "I came from time to time."),
            ("한번을 불친절한 적이 없어요.", "Not once were they unkind."),
            ("한번도 불친절하지 않았어요.", "They were never even once unkind."),
            ("매주 한번씩 왔어요.", "I came one time every week."),
        ):
            with self.subTest(source=source):
                items, _ = self.prepare([source], [translation])
                self.assertEqual(items[0]["presentation"]["status"], "translated")
                self.assertEqual(items[0]["text"], source)
        changed, _ = self.prepare(["매주 한번씩 왔어요."], ["I came two times every week."])
        self.assertEqual(changed[0]["presentation"]["status"], "unavailable")

    def test_ordinary_korean_words_are_not_misread_as_nine_minutes(self):
        source = "치료를 구분해서 설명했고 구분도 명확했어요."
        items, _ = self.prepare([source], ["They explained the treatments separately and the distinction was clear."])
        self.assertEqual(items[0]["presentation"]["status"], "translated")
        count, _ = self.prepare(["구 분 기다렸어요."], ["I waited nine minutes."])
        self.assertEqual(count[0]["presentation"]["status"], "translated")
        altered, _ = self.prepare(["구 분 기다렸어요."], ["I waited ten minutes."])
        self.assertEqual(altered[0]["presentation"]["status"], "unavailable")

    def test_return_visit_interval_does_not_invent_a_first_visit_count(self):
        for event in ("time", "visit", "appointment"):
            for interval in ("a while", "a long time", "ages"):
                source = "오랜만에 방문했는데 4층 직원이 기억해주셨어요."
                translation = f"It was my first {event} in {interval}, and the fourth floor staff remembered me."
                items, _ = self.prepare([source], [translation])
                self.assertEqual(items[0]["presentation"]["status"], "translated")
                altered, _ = self.prepare([source], [translation.replace("fourth", "third")])
                self.assertEqual(altered[0]["presentation"]["status"], "unavailable")
        source = "2년 만에 4층에 방문했어요."
        valid, _ = self.prepare([source], ["It was my first visit in 2 years, on the fourth floor."])
        self.assertEqual(valid[0]["presentation"]["status"], "translated")
        changed, _ = self.prepare([source], ["It was my first visit in 3 years, on the fourth floor."])
        self.assertEqual(changed[0]["presentation"]["status"], "unavailable")

    def test_ordinal_opinions_and_visits_keep_their_number(self):
        for source, translation in (("2차 의견을 들었어요.", "I got a second opinion."),
                                    ("첫 번째 방문이었어요.", "It was my first visit."),
                                    ("1초 기다렸어요.", "I waited one second.")):
            with self.subTest(source=source):
                items, _ = self.prepare([source], [translation])
                self.assertEqual(items[0]["presentation"]["status"], "translated")
        altered, _ = self.prepare(["2차 의견을 들었어요."], ["I got a third opinion."])
        self.assertEqual(altered[0]["presentation"]["status"], "unavailable")

    def test_korean_this_time_and_this_month_are_not_two(self):
        for source, translation in (
            ("이번에 발가락 통증으로 3진료실에 갔어요.", "This time I visited consultation room 3 for toe pain."),
            ("이달에 2회 치료받았어요.", "I had two treatment sessions this month."),
            ("이 번 방문했어요.", "I visited two times."),
        ):
            with self.subTest(source=source):
                items, _ = self.prepare([source], [translation])
                self.assertEqual(items[0]["presentation"]["status"], "translated")
                self.assertEqual(items[0]["text"], source)
                reverse, _ = self.prepare([translation], [source], language="Korean")
                self.assertEqual(reverse[0]["presentation"]["status"], "translated")
        for source, translation in (
            ("이번에 3진료실에 갔어요.", "This time I visited consultation room 4."),
            ("이달에 2회 치료받았어요.", "I had three treatment sessions this month."),
            ("이 번 방문했어요.", "I visited three times."),
        ):
            with self.subTest(source=source):
                items, _ = self.prepare([source], [translation])
                self.assertEqual(items[0]["presentation"]["status"], "unavailable")
                self.assertEqual(items[0]["text"], source)

    def test_first_time_in_a_while_is_a_return_without_a_visit_count(self):
        source = "오랜만에 방문했는데 4층 직원이 기억해주셨어요."
        for interval in ("a while", "a long time", "ages"):
            translation = f"I visited for the first time in {interval}, and the fourth floor staff remembered me."
            with self.subTest(interval=interval):
                items, _ = self.prepare([source], [translation])
                self.assertEqual(items[0]["presentation"]["status"], "translated")
                self.assertEqual(items[0]["text"], source)
                reverse, _ = self.prepare([translation], [source], language="Korean")
                self.assertEqual(reverse[0]["presentation"]["status"], "translated")
                altered, _ = self.prepare([source], [translation.replace("fourth", "fifth")])
                self.assertEqual(altered[0]["presentation"]["status"], "unavailable")
        for translation in (
            "It was my first time visiting, and the fourth floor staff remembered me.",
            "I visited for the second time in a while, and the fourth floor staff remembered me.",
            "I visited for the first time in my life, and the fourth floor staff remembered me.",
        ):
            with self.subTest(translation=translation):
                altered, _ = self.prepare([source], [translation])
                self.assertEqual(altered[0]["presentation"]["status"], "unavailable")

    def test_hospital_visit_count_can_follow_its_duration_in_translation(self):
        source = "병원 한 번 가는데 2시간 걸렸어요."
        for translation in ("Two hours just to go to the hospital.",
                            "Going to the hospital took two hours.",
                            "A hospital visit took two hours."):
            with self.subTest(translation=translation):
                items, _ = self.prepare([source], [translation])
                self.assertEqual(items[0]["presentation"]["status"], "translated")
                self.assertEqual(items[0]["text"], source)
        for translation in ("Three hours just to go to the hospital.",
                            "Two hours for two hospital visits."):
            with self.subTest(translation=translation):
                items, _ = self.prepare([source], [translation])
                self.assertEqual(items[0]["presentation"]["status"], "unavailable")
        changed, _ = self.prepare(["병원 두 번 가는데 2시간 걸렸어요."], ["Two hours just to go to the hospital."])
        self.assertEqual(changed[0]["presentation"]["status"], "unavailable")

    def test_each_wait_and_treatment_duration_survives_visit_word_order(self):
        source = "30분 기다리라더니 한시간 넘게 걸렸고 10분 치료가 40분이 되어 두시간 뒤 나왔어요. 병원 한 번 가는데 2시간이네요."
        translation = ("They said 30 minutes, but it took over an hour. The 10 minute treatment took 40 minutes "
                       "and I left two hours later. Two hours just to go to the hospital.")
        items, _ = self.prepare([source], [translation])
        self.assertEqual(items[0]["presentation"]["status"], "translated")
        for old, new in (("30 minutes", "20 minutes"), ("an hour", "three hours"),
                         ("10 minute", "20 minute"), ("40 minutes", "50 minutes"),
                         ("two hours later", "three hours later")):
            with self.subTest(old=old):
                changed, _ = self.prepare([source], [translation.replace(old, new)])
                self.assertEqual(changed[0]["presentation"]["status"], "unavailable")

    def test_provider_elapsed_time_is_retained_on_success_and_failure(self):
        from unittest.mock import patch
        import requests
        from review_presentation import prepare_review_presentations
        for error, reason in ((None, None), (requests.Timeout(), "provider_timeout")):
            cards = [{"retrieval_evidence": [review("두 시간 기다렸어요")]}]
            with patch("review_presentation.monotonic", side_effect=[4.0, 4.25]), patch("review_presentation.requests.post") as post:
                post.side_effect = error
                post.return_value.json.return_value = {"data": {"translations": [
                    {"translatedText": "I waited two hours."}]}}
                trace = prepare_review_presentations(cards, "English", translation_api_key="fixture")
            self.assertEqual(trace["duration_ms"], 250.0)
            self.assertEqual(trace["reason"], reason)
        self.assertEqual(prepare_review_presentations([], "English")["duration_ms"], 0.0)

    def test_comment_heavy_cards_use_one_bounded_deduplicated_batch(self):
        from review_presentation import MAX_TRANSLATION_REVIEWS
        sources = [f"간호사가 설명을 해주었어요 {index}" for index in range(MAX_TRANSLATION_REVIEWS + 1)]
        translations = [f"The nurse explained things {index}" for index in range(MAX_TRANSLATION_REVIEWS)]
        items, post = self.prepare(sources, translations)
        self.assertEqual(len(post.call_args.kwargs["json"]["q"]), MAX_TRANSLATION_REVIEWS)
        self.assertEqual(items[-1]["presentation"]["status"], "unavailable")
        self.assertTrue(all(item["presentation"]["status"] == "translated" for item in items[:-1]))

    def test_first_reviews_across_cards_get_translation_capacity_before_later_pages(self):
        from unittest.mock import patch
        from review_presentation import prepare_review_presentations
        cards = [{"retrieval_evidence": [review(f"친절한 설명 {index}") for index in range(5)]},
                 {"retrieval_evidence": [review("발목 치료 경험")]}]
        with patch("review_presentation.MAX_TRANSLATION_REVIEWS", 2), patch("review_presentation.requests.post") as post:
            post.return_value.json.return_value = {"data": {"translations": [
                {"translatedText": "Kind explanation 0"}, {"translatedText": "Ankle treatment experience"}]}}
            trace = prepare_review_presentations(cards, "English", translation_api_key="fixture")
        self.assertEqual(post.call_args.kwargs["json"]["q"], ["친절한 설명 0", "발목 치료 경험"])
        self.assertEqual(trace["capacity_skipped"], 4)
        self.assertEqual(cards[1]["retrieval_evidence"][0]["presentation"]["status"], "translated")

    def test_missing_credentials_are_distinct_from_translation_not_needed(self):
        from review_presentation import prepare_review_presentations
        korean = [{"retrieval_evidence": [review("친절한 설명")]}]
        english = [{"retrieval_evidence": [review("Kind explanation")]}]
        self.assertEqual(prepare_review_presentations(korean, "English")["reason"], "missing_credentials")
        self.assertEqual(prepare_review_presentations(english, "English")["reason"], "not_needed")


class OpenRouterTranslationTests(unittest.TestCase):
    def prepare(self, originals, entries, *, finish_reason="stop", key="fixture-key", error=None, language="English", response_overrides=None):
        import json
        from unittest.mock import AsyncMock, patch
        from review_presentation import prepare_review_presentations
        cards = [{"place_id": f"clinic-{index}", "retrieval_evidence": [
            {**review(text, f"review:{index}"), "place_id": f"clinic-{index}"}]}
            for index, text in enumerate(originals)]
        response = {"id": "gen-fixture", "model": "openai/gpt-4.1", "provider": "OpenAI",
                    "choices": [{"finish_reason": finish_reason, "message": {
            "content": json.dumps({"translations": entries})}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 30, "cost": 0.00044}}
        response.update(response_overrides or {})
        with patch("review_presentation._request_openrouter", new_callable=AsyncMock) as request, patch("review_presentation.requests.post") as google:
            request.return_value = response
            request.side_effect = error
            trace = prepare_review_presentations(cards, language, translation_provider="openrouter", openrouter_api_key=key)
        google.assert_not_called()
        return cards, trace, request

    def test_indexed_outputs_keep_source_ownership_when_reordered(self):
        import json
        originals = ["의사는 친절해요", "접수 직원은 불친절해요"]
        cards, trace, request = self.prepare(originals, [
            {"index": 1, "text": "The receptionist is unkind."},
            {"index": 0, "text": "The doctor is kind."},
        ])
        self.assertEqual(trace["translated"], 2)
        self.assertEqual(trace["usage"]["cost"], 0.00044)
        self.assertEqual(trace["actual_model"], "openai/gpt-4.1")
        self.assertEqual(trace["actual_provider"], "OpenAI")
        self.assertEqual(trace["completion_id"], "gen-fixture")
        for index, translation in enumerate(("The doctor is kind.", "The receptionist is unkind.")):
            item = cards[index]["retrieval_evidence"][0]
            self.assertEqual(item["presentation"]["text"], translation)
            self.assertEqual((item["place_id"], item["evidence_id"], item["text"]),
                             (f"clinic-{index}", f"review:{index}", originals[index]))
        payload, key = request.call_args.args
        self.assertEqual(key, "fixture-key")
        self.assertEqual(payload["model"], "openai/gpt-4.1")
        self.assertFalse(payload["provider"]["allow_fallbacks"])
        self.assertTrue(payload["response_format"]["json_schema"]["strict"])
        self.assertEqual(payload["max_tokens"], 4096)
        self.assertEqual(json.loads(payload["messages"][1]["content"])["reviews"],
                         [{"index": index, "text": text} for index, text in enumerate(originals)])

    def test_substituted_provider_or_missing_identity_cannot_supply_translations(self):
        for overrides in ({"model": "openai/gpt-4.1-mini"}, {"provider": "Azure"},
                          {"model": None}, {"provider": None}, {"id": ""}, {"id": True}):
            with self.subTest(overrides=overrides):
                cards, trace, _ = self.prepare(["친절한 의사"], [{"index": 0, "text": "Kind doctor"}],
                                                response_overrides=overrides)
                self.assertEqual(trace["reason"], "invalid_response")
                self.assertEqual(cards[0]["retrieval_evidence"][0]["presentation"]["status"], "unavailable")
                self.assertEqual(cards[0]["retrieval_evidence"][0]["text"], "친절한 의사")
                self.assertNotIn("completion_id", trace)

    def test_invalid_usage_never_enters_private_json_trace(self):
        import json
        for value in (True, -1, -0.01, float("nan"), float("inf"), float("-inf"), "0.01", None):
            with self.subTest(value=value):
                cards, trace, _ = self.prepare(["친절한 의사"], [{"index": 0, "text": "Kind doctor"}],
                                                response_overrides={"usage": {"cost": value}})
                self.assertEqual(trace["reason"], "invalid_response")
                self.assertEqual(cards[0]["retrieval_evidence"][0]["presentation"]["status"], "unavailable")
                self.assertNotIn("usage", trace)
                json.dumps(trace, allow_nan=False)

    def test_invalid_indices_and_incomplete_generation_preserve_all_originals(self):
        originals = ["친절한 의사", "불친절한 접수 직원"]
        invalid = (
            ([{"index": 0, "text": "Kind doctor"}, {"index": 0, "text": "Rude receptionist"}], "stop"),
            ([{"index": 0, "text": "Kind doctor"}, {"index": 2, "text": "Rude receptionist"}], "stop"),
            ([{"index": False, "text": "Kind doctor"}, {"index": 1, "text": "Rude receptionist"}], "stop"),
            ([{"index": 0, "text": "Kind doctor"}], "stop"),
            ([{"index": 0, "text": "Kind doctor"}, {"index": 1, "text": "Rude receptionist"}], "length"),
        )
        for entries, reason in invalid:
            with self.subTest(entries=entries, finish_reason=reason):
                cards, trace, _ = self.prepare(originals, entries, finish_reason=reason)
                self.assertEqual(trace["reason"], "invalid_response")
                for index, card in enumerate(cards):
                    item = card["retrieval_evidence"][0]
                    self.assertEqual(item["text"], originals[index])
                    self.assertEqual(item["presentation"]["status"], "unavailable")

    def test_provider_timeout_and_missing_credentials_do_not_fall_back_to_google(self):
        for key, error, reason in (("", None, "missing_credentials"), ("fixture-key", TimeoutError(), "provider_timeout")):
            cards, trace, request = self.prepare(["친절한 의사"], [], key=key, error=error)
            self.assertEqual(trace["reason"], reason)
            self.assertEqual(cards[0]["retrieval_evidence"][0]["text"], "친절한 의사")
            if not key:
                request.assert_not_called()

    def test_generated_translations_still_pass_numeric_and_language_validation(self):
        cards, trace, _ = self.prepare(["2시간 기다렸어요"], [{"index": 0, "text": "I waited three hours."}])
        self.assertEqual(trace["reason"], "translation_rejected")
        self.assertEqual(cards[0]["retrieval_evidence"][0]["presentation"]["status"], "unavailable")
        cards, trace, _ = self.prepare(["The doctor was kind."], [{"index": 0, "text": "의사는 친절했어요."}], language="Korean")
        self.assertEqual(trace["translated"], 1)
        self.assertEqual(cards[0]["retrieval_evidence"][0]["presentation"]["language"], "Korean")

    def test_over_capacity_retains_original_without_dispatch(self):
        from review_presentation import MAX_OPENROUTER_TRANSLATION_CHARACTERS
        source = "의사 설명 " * MAX_OPENROUTER_TRANSLATION_CHARACTERS
        cards, trace, request = self.prepare([source], [])
        request.assert_not_called()
        self.assertEqual(trace["reason"], "capacity_exceeded")
        self.assertEqual(cards[0]["retrieval_evidence"][0]["text"], source)

    def test_total_deadline_cancels_and_closes_provider_request(self):
        import asyncio
        from unittest.mock import AsyncMock, patch
        from review_presentation import prepare_review_presentations
        cancelled = []
        async def slow_request(*args, **kwargs):
            try:
                await asyncio.sleep(1)
            finally:
                cancelled.append(True)
        with patch("review_presentation.OPENROUTER_TIMEOUT_SECONDS", 0.005), patch("review_presentation.httpx.AsyncClient") as factory:
            client = AsyncMock()
            client.post.side_effect = slow_request
            factory.return_value.__aenter__.return_value = client
            cards = [{"retrieval_evidence": [review("친절한 의사")]}]
            trace = prepare_review_presentations(cards, "English", translation_provider="openrouter", openrouter_api_key="fixture-key")
            factory.return_value.__aexit__.assert_awaited_once()
        self.assertEqual(trace["reason"], "provider_timeout")
        self.assertEqual(cancelled, [True])
        self.assertEqual(cards[0]["retrieval_evidence"][0]["presentation"]["status"], "unavailable")
