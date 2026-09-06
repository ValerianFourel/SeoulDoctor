"""Check evaluator input isolation and scoring without external calls."""

import importlib.util
from pathlib import Path
import unittest
from pydantic import ValidationError
from mini_retrieval import RetrievalInput, limits

path = Path(__file__).resolve().parents[2] / 'scripts/mini_retrieval_eval.py'
spec = importlib.util.spec_from_file_location('mini_eval', path)
mini = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mini)


class MiniRetrievalTests(unittest.TestCase):
    def test_labels_cannot_enter_retrieval_request(self):
        payload = {'query': 'Explain my treatment', 'location': 'Mapo-gu', 'specialty': '치과'}
        for key in ('expected_facility_id', 'expected_evidence_id', 'original_text'):
            with self.assertRaises(ValidationError):
                RetrievalInput(**payload, **{key: 'private target'})
        self.assertEqual(RetrievalInput(**payload).model_dump(), payload)

    def test_facility_review_selection_and_ownership_are_separate(self):
        target = {'expected_facility_id': 'a', 'expected_evidence_id': 'review-a'}
        response = {'facilities': [{'place_id': 'b', 'selected': []},
                                   {'place_id': 'a', 'selected': []}],
                    'retrieved': [{'evidence_id': 'review-a', 'place_id': 'a'}]}
        self.assertEqual(mini.score(target, response), (2, True, False, 0))
        response['facilities'][1]['selected'] = response['retrieved']
        self.assertEqual(mini.score(target, response), (2, True, True, 0))
        response['retrieved'][0]['place_id'] = 'wrong'
        self.assertEqual(mini.score(target, response)[3], 2)

    def test_limits_are_normal_except_disabled_retry(self):
        config = limits()
        self.assertEqual(config['candidate_policy']['facility_limit'], 200)
        self.assertEqual(config['evidence_policy']['shortlist_limit'], 20)
        self.assertEqual(config['evidence_policy']['presentation_limit'], 3)
        self.assertEqual(config['evaluation_retry_rerank_budget'], 0)
