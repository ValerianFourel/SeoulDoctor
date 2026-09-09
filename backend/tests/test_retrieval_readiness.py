from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from search.readiness import probe_retrieval
from search.reranker import RerankOutcome
from search.semantic_retriever import SemanticReviewOutcome, SemanticEvidenceReference
from backend.tests.test_evidence_retrieval import hit


class ReadinessTests(unittest.TestCase):
    def setUp(self):
        self.hit = hit("review:probe", "alpha", "진료 설명", 1)
        self.scoped = Mock(review_source_sha256="d" * 64)
        self.scoped.search_evidence.return_value = [self.hit]
        self.scoped.resolve_evidence_ids.return_value = [self.hit]
        self.semantic = Mock()
        self.semantic.retrieve.return_value = SemanticReviewOutcome("ok", tuple(
            SemanticEvidenceReference("readiness", "review:probe", "alpha", channel, 1, 0.9)
            for channel in ("bge_m3_dense", "bge_m3_sparse")
        ), gpu_execution_verified=True)
        self.reranker = Mock()
        self.reranker.rerank.return_value = RerankOutcome(
            (self.hit,), True, "ok", scores=(("review:probe", 0.8),),
            gpu_execution_verified=True,
        )

    def test_real_operation_contract_not_liveness(self):
        report = probe_retrieval(self.scoped, self.semantic, self.reranker)
        self.assertTrue(report["ready"])
        self.assertTrue(report["gpu_execution_verified"])
        self.scoped.resolve_evidence_ids.assert_called_once()
        self.reranker.rerank.assert_called_once()

    def test_unavailable_service_cannot_be_ready(self):
        self.semantic.retrieve.return_value = SemanticReviewOutcome("request_failed")
        self.assertFalse(probe_retrieval(self.scoped, self.semantic, self.reranker)["ready"])
        self.reranker.rerank.assert_not_called()

    def test_bad_reranking_scores_cannot_be_ready(self):
        self.reranker.rerank.return_value = RerankOutcome((self.hit,), True, "ok")
        self.assertFalse(probe_retrieval(self.scoped, self.semantic, self.reranker)["ready"])

    def test_endpoint_uses_authoritative_scope(self):
        import main
        import pandas as pd
        from unittest.mock import patch
        context = Mock()
        context.__enter__ = Mock(return_value=self.scoped)
        context.__exit__ = Mock(return_value=False)
        index = SimpleNamespace(version="test-v1", within=Mock(return_value=context))
        with (
            patch.object(main, "health_check"),
            patch.object(main, "df_filtered", pd.DataFrame([{"place_id": "alpha"}])),
            patch.object(main, "search_index_release", index),
            patch.object(main, "semantic_evidence_source", self.semantic),
            patch.object(main, "evidence_reranker", self.reranker),
        ):
            self.assertTrue(main.retrieval_readiness()["ready"])
