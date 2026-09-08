import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import State, serialize_results_for_chat
from query_facets import augment_extracted_facets, retrieval_terms_from_state
from search.turn_delta import compile_turn_delta, reduce_search_state


class StateReliabilityTests(unittest.TestCase):
    def apply(self, message, state=None, **proposal):
        return reduce_search_state(
            state or State(location=None), compile_turn_delta(message, proposal)
        )

    def test_ichon_refinement_preserves_visit_and_care_preferences(self):
        initial = self.apply(
            "I need a pediatrician for a routine checkup, with clear explanations "
            "and no overprescribing.", specialty="소아청소년과",
        )
        refined = self.apply(
            "Narrow to within 1 km of Ichon. I prefer a short wait.", initial,
            location="Ichon", comment_terms=["short wait"],
        )
        self.assertEqual(refined.visit_reason, "routine checkup")
        self.assertEqual(refined.specialty, "소아청소년과")
        self.assertEqual(refined.location, "Ichon")
        self.assertEqual(refined.max_distance_km, 1)
        self.assertEqual(set(refined.comment_terms), {
            "clear explanations", "no overprescribing", "short wait",
        })

    def test_wait_withdrawal_clears_every_projection_and_keeps_other_care(self):
        for message in (
            "Some waiting is fine — that's not a dealbreaker for me. Kind treatment of children matters more.",
            "I don't mind waiting; I prefer kind treatment of children.",
            "Waiting time no longer matters to me.",
            "I'm willing to wait if the doctor is kind with children.",
            "대기 시간이 좀 길어도 괜찮아요. 아이에게 친절한 곳이면 좋겠어요.",
            "짧은 대기는 더 이상 중요하지 않아요.",
        ):
            with self.subTest(message=message):
                state = State(
                    location="Ichon", specialty="소아청소년과", max_distance_km=1,
                    keywords=["short wait", "kind"], hard_keywords=["no waiting"],
                    negative_keywords=["long wait"], negative_hard_keywords=["긴 대기"],
                    comment_terms=["short wait", "clear explanations", "no overprescribing"],
                    last_retrieval_metadata={"coverage_sufficient": True},
                )
                result = self.apply(message, state, soft_keywords=["short wait"])
                self.assertEqual(result.location, "Ichon")
                self.assertEqual(result.max_distance_km, 1)
                self.assertIn("clear explanations", result.comment_terms)
                self.assertIn("no overprescribing", result.comment_terms)
                for term in result.keywords + result.hard_keywords + result.negative_keywords + result.negative_hard_keywords + retrieval_terms_from_state(result):
                    self.assertNotRegex(term, r"(?i)wait|대기")
                self.assertEqual(result.last_retrieval_metadata, {})

    def test_wait_intolerance_is_not_misread_as_withdrawal(self):
        state = State(keywords=["short wait"])
        for message in ("Waiting is not fine.", "I don't think waiting is acceptable.", "대기가 길면 괜찮지 않아요."):
            with self.subTest(message=message):
                self.assertIn("short wait", self.apply(message, state).keywords)

    def test_unknown_preferences_survive_location_refinement(self):
        state = State(comment_terms=["welcomes questions about fees"], negative_keywords=["pressure to buy packages"])
        result = self.apply("Move to Ichon without changing my care preferences.", state, location="Ichon")
        self.assertEqual(result.comment_terms, state.comment_terms)
        self.assertEqual(result.negative_keywords, state.negative_keywords)

    def test_repeated_or_changed_local_anchor_keeps_explicit_radius(self):
        state = State(location="Ichon", latitude=37.5224, longitude=126.9735,
                      max_distance_km=1, travel_confidence=1, travel_label="Nearby")
        repeated = self.apply("Same location, please.", state, location="Ichon")
        self.assertEqual(repeated.max_distance_km, 1)
        self.assertEqual(repeated.latitude, state.latitude)
        moved = self.apply("Move the search to Jonggak.", state, location="Jonggak")
        self.assertEqual(moved.max_distance_km, 1)
        self.assertIsNone(moved.latitude)

    def test_search_request_with_question_mark_still_carries_preferences(self):
        result = self.apply("Could you search for clinics with clear explanations and restrained prescribing?")
        self.assertEqual(set(result.comment_terms), {"clear explanations", "no overprescribing"})

    def test_factual_english_question_never_creates_mandatory_filter(self):
        messages = (
            "Can you confirm English consultations, or should I check directly?",
            "Is an English-speaking doctor available?",
            "영어 진료가 가능한지 알려 주세요.",
        )
        for message in messages:
            with self.subTest(message=message):
                result = self.apply(message, hard_keywords=["English-speaking"], soft_keywords=["direct"])
                self.assertEqual(result.hard_keywords, [])
                self.assertNotIn("direct", result.keywords)
                self.assertTrue(result.inquiries)

    def test_english_requirement_and_question_can_coexist(self):
        result = self.apply(
            "English consultation is mandatory. Can you confirm which clinics provide it?",
            hard_keywords=["English-speaking"],
        )
        self.assertIn("English-speaking", result.hard_keywords)
        self.assertTrue(result.inquiries)

    def test_english_withdrawal_does_not_negate_explanation_preference(self):
        state = State(hard_keywords=["English-speaking"], comment_terms=["clear explanations"])
        result = self.apply(
            "I don't require English consultations, but I still value clear explanations.",
            state, hard_keywords=["English-speaking"], negative_keywords=["clear explanations"],
        )
        self.assertEqual(result.hard_keywords, [])
        self.assertIn("clear explanations", result.comment_terms)
        self.assertNotIn("clear explanations", result.negative_keywords)

    def test_response_language_has_no_consultation_requirement(self):
        state = State(language_pref="Korean", explicit_response_language="Korean", keywords=["kind"])
        result = self.apply("Please reply in English.", state, hard_keywords=["English-speaking"])
        self.assertEqual(result.language_pref, "English")
        self.assertEqual(result.hard_keywords, [])
        self.assertEqual(result.keywords, ["kind"])
        mixed = self.apply("광화문 clinic please", result)
        self.assertEqual(mixed.language_pref, "English")

    def test_current_questions_expire_on_next_turn(self):
        state = self.apply("Can you confirm English consultations?")
        result = self.apply("Keep the same radius.", state)
        self.assertEqual(result.inquiries, [])

    def test_hours_question_is_not_mandatory_hours(self):
        result = self.apply("Are they open Tuesday evening?", hard_keywords=["Tuesday evening hours"], required_hours=["tuesday_evening"])
        self.assertEqual(result.required_hours, [])
        self.assertEqual(result.hard_keywords, [])
        self.assertTrue(result.inquiries)

    def test_nurse_question_is_not_a_patient_exclusion(self):
        result = self.apply(
            "What do mixed reviews about unfriendly nurses mean for my decision?",
            negative_keywords=["unfriendly", "unfriendly nurses"],
        )
        self.assertEqual(result.negative_keywords, [])
        self.assertEqual(result.comment_terms, [])
        self.assertTrue(result.inquiries)
        excluded = self.apply("Please exclude clinics with unfriendly nurses.")
        self.assertIn("unfriendly nurses", excluded.negative_keywords)

    def test_model_cannot_reset_context_or_remove_without_user_instruction(self):
        state = State(comment_terms=["clear explanations"], hard_keywords=["parking"])
        result = self.apply(
            "Move the search to Ichon.", state, location="Ichon",
            operation="replace_context", remove_terms=["parking"],
        )
        self.assertEqual(result.comment_terms, ["clear explanations"])
        self.assertEqual(result.hard_keywords, ["parking"])

    def test_explicit_term_replacement_only_changes_named_preference(self):
        state = State(comment_terms=["short wait", "clear explanations"], keywords=["short wait"])
        message = "Replace short wait with thorough care."
        result = self.apply(message, state, term_operations=[{
            "action": "replace", "field": "comment_terms", "term": "short wait",
            "replacement": "thorough", "source_span": message,
        }])
        self.assertEqual(set(result.comment_terms), {"clear explanations", "thorough"})
        self.assertEqual(result.keywords, [])

    def test_ungrounded_destructive_term_operation_is_ignored(self):
        state = State(comment_terms=["clear explanations"])
        result = self.apply("Move to Ichon.", state, term_operations=[{
            "action": "remove", "field": "comment_terms", "term": "clear explanations",
            "source_span": "Remove clear explanations.",
        }])
        self.assertEqual(result.comment_terms, ["clear explanations"])

    def test_negated_removal_instruction_preserves_preference(self):
        state = State(hard_keywords=["parking"])
        message = "Do not remove parking."
        result = self.apply(message, state, remove_terms=["parking"], term_operations=[{
            "action": "remove", "field": "hard_keywords", "term": "parking", "source_span": message,
        }])
        self.assertEqual(result.hard_keywords, ["parking"])
        retained = self.apply("Don't start over. Keep my request.", state, operation="replace_context")
        self.assertEqual(retained.hard_keywords, ["parking"])

    def test_comment_augmentation_does_not_treat_review_question_as_preference(self):
        self.assertEqual(augment_extracted_facets("Are the nurses friendly?", {})["comment_terms"], [])

    def test_excluded_quality_is_not_also_added_as_positive_comment_preference(self):
        result = self.apply("Avoid friendly staff.", negative_keywords=["friendly"])
        self.assertIn("friendly", result.negative_keywords)
        self.assertNotIn("friendly", result.comment_terms)

    def test_clinical_korean_negative_does_not_infer_positive_kindness(self):
        result = self.apply("불친절한 간호사는 피하고 싶어요.")
        self.assertEqual(result.comment_terms, [])
        self.assertIn("unfriendly nurses", result.negative_keywords)

    def test_public_serialization_keeps_answer_and_retrieval_roles(self):
        citation = {"marker": 1, "place_id": "facility-a", "evidence_id": "review:a", "original_excerpt": "The nurse was rude."}
        cards = [{"place_id": "facility-a", "answer_status": "accepted", "answer_citations": [citation],
                  "retrieval_evidence": [{"text": "The nurse was rude.", "retrieval_roles": ["support", "risk"]}]}]
        public = serialize_results_for_chat(cards, include_debug=False)
        self.assertEqual(public[0]["answer_citations"], [citation])
        self.assertEqual(public[0]["answer_status"], "accepted")
        self.assertEqual(public[0]["retrieval_evidence"][0]["retrieval_roles"], ["support", "risk"])


if __name__ == "__main__":
    unittest.main()
