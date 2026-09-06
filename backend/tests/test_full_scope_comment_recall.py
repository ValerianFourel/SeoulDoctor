from pathlib import Path
import sys
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.tests.test_live_retrieval import (
    FakeIndex, FakePipeline, FakeScopedIndex, evidence_hit, make_rules, make_scope,
)
from search.live_retrieval import CandidateRetrievalAdapter, RetrievalQuery


class FullScopeRecallTests(unittest.TestCase):
    def test_comment_can_promote_facility_beyond_initial_twenty(self):
        target_id = "z-target"

        class Scoped(FakeScopedIndex):
            queried_facilities = ()

            def search_evidence(self, query, **kwargs):
                return [evidence_hit(target_id, 1)]

            def search_evidence_for_facilities(self, query, *, facility_ids, **kwargs):
                self.queried_facilities = facility_ids
                return [evidence_hit(target_id, 1)] if target_id in facility_ids else []

        scoped = Scoped()
        ids = ("alpha", "bravo", *(f"clinic-{i}" for i in range(30)), target_id)
        frame = pd.DataFrame({"place_id": ids, "name": ids, "distance_km": [1.0] * len(ids)})
        result = CandidateRetrievalAdapter(
            active_index=FakeIndex(scoped), legacy_pipeline=FakePipeline(),
        ).rank(scope=make_scope(ids), eligible=frame, rules=make_rules(evidence=True),
               query=RetrievalQuery("friendly", 5.0, "distance", 0.95))
        evidence = result.dataframe.set_index("place_id").loc[target_id, "retrieval_evidence"]
        self.assertTrue(evidence)
        self.assertEqual(evidence[0]["place_id"], target_id)
