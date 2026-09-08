from dataclasses import replace
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.tests.test_evidence_retrieval import hit
from search.evidence_retrieval import ScoredEvidence, select_evidence_groups


def scored(identity, ordinal, *, source_type="verbatim_review", verbatim=True, role="disease"):
    return ScoredEvidence(
        hit=replace(hit(identity, "alpha", "발목 진료 경험", ordinal),
                    source_type=source_type, is_verbatim=verbatim),
        matched_constraint_ids=frozenset({"ankle"}), roles=frozenset({role}),
        lexical_rank=ordinal, rerank_rank=ordinal, rerank_score=1 / ordinal,
        distinctiveness=1 / ordinal, local_cluster_id=identity, local_cluster_size=1,
        corroboration_count=1, selection_score=1 / ordinal,
    )


class OriginalReviewSelectionTests(unittest.TestCase):
    def test_high_ranked_medical_facts_do_not_starve_original_reviews(self):
        facts = tuple(scored(f"fact-{index}", index, source_type="medical_info", verbatim=False)
                      for index in range(1, 5))
        reviews = (scored("review-positive", 5), scored("review-negative", 6, role="risk"))
        groups = select_evidence_groups(("alpha",), (*facts, *reviews), (), limit=3)["alpha"]
        self.assertEqual({item.hit.evidence_id for item in groups.presented},
                         {"review-positive", "review-negative"})
        self.assertTrue(all(item.hit.is_verbatim and item.hit.source_type == "verbatim_review"
                            for item in groups.presented))
        self.assertTrue({item.hit.evidence_id for item in facts}.issubset(
            {item.hit.evidence_id for item in groups.supporting}))
        self.assertEqual([item.hit.evidence_id for item in groups.warnings], ["review-negative"])

    def test_nonreview_source_or_nonverbatim_flag_cannot_enter_review_presentation(self):
        values = (scored("fact", 1, source_type="medical_info"),
                  scored("summary", 2, verbatim=False))
        groups = select_evidence_groups(("alpha",), values, (), limit=3)["alpha"]
        self.assertEqual(groups.presented, ())
        self.assertEqual(len(groups.supporting), 2)


if __name__ == "__main__":
    unittest.main()
