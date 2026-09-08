from __future__ import annotations

import sys
import unittest
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

import production
import production_core


class RerankerTests(unittest.TestCase):
    def setUp(self):
        self.engine = production.PinnedBgeReranker.__new__(production.PinnedBgeReranker)
        self.engine._model = Mock()
        self.engine.gpu_proof = {"device": "cuda:0", "probe_pairs": 2}
        self.client = TestClient(production.app)

    def test_api_returns_all_input_indexes_sorted_by_score(self):
        self.engine._model.compute_score.return_value = [0.2, 0.9, 0.9]
        with patch.object(production, "reranker", self.engine):
            response = self.client.post("/rerank", json={
                "query": "foot pain 발 통증",
                "texts": ["대기가 길어요", "발 치료를 받았어요", "발목 설명이 좋아요"],
            })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "model": production.RERANKER_MODEL_ID,
            "model_revision": production.RERANKER_MODEL_REVISION,
            "results": [
                {"index": 1, "score": 0.9},
                {"index": 2, "score": 0.9},
                {"index": 0, "score": 0.2},
            ],
        })
        kwargs = self.engine._model.compute_score.call_args.kwargs
        self.assertEqual(kwargs, {"batch_size": 16, "max_length": 512, "normalize": True})

    def test_single_pair_scalar_score_and_batches(self):
        self.engine._model.compute_score.side_effect = [[0.1] * 16, 0.7]
        output = self.engine.rerank("query", ["text"] * 17)
        self.assertEqual(len(output), 17)
        self.assertEqual(output[0], {"index": 16, "score": 0.7})
        self.assertEqual(self.engine._model.compute_score.call_count, 2)

    def test_invalid_inference_returns_503_without_partial_results(self):
        for values in ([0.1], [float("nan"), 0.4], [float("inf"), 0.4]):
            with self.subTest(values=values), patch.object(production, "reranker", self.engine):
                self.engine._model.compute_score.return_value = values
                response = self.client.post("/rerank", json={"query": "q", "texts": ["a", "b"]})
                self.assertEqual(response.status_code, 503)
                self.assertNotIn("results", response.json())

    def test_request_limits_reject_work_before_inference(self):
        invalid = [
            {"query": " ", "texts": ["a"]},
            {"query": "q" * 2001, "texts": ["a"]},
            {"query": "q", "texts": []},
            {"query": "q", "texts": ["a"] * 257},
            {"query": "q", "texts": [" "]},
            {"query": "q", "texts": ["a" * 32001]},
            {"query": "q", "texts": ["a"], "unknown": True},
        ]
        with patch.object(production, "reranker", self.engine):
            for payload in invalid:
                with self.subTest(payload=str(payload)[:80]):
                    self.assertEqual(self.client.post("/rerank", json=payload).status_code, 422)
            self.assertEqual(self.client.post("/rerank", json={
                "query": "q", "texts": ["a" * 32000] * 17,
            }).status_code, 413)
        self.engine._model.compute_score.assert_not_called()

    def test_unready_and_busy_are_explicit(self):
        with patch.object(production, "reranker", None):
            self.assertEqual(self.client.get("/info").status_code, 503)
            self.assertEqual(self.client.post("/rerank", json={"query": "q", "texts": ["a"]}).status_code, 503)
        lock = Mock()
        lock.acquire.return_value = False
        with patch.object(production, "reranker", self.engine), patch.object(production, "_gpu_lock", lock):
            self.assertEqual(self.client.post("/rerank", json={"query": "q", "texts": ["a"]}).status_code, 503)
        lock.release.assert_not_called()
        self.engine._model.compute_score.assert_not_called()

    def test_deadline_stops_remaining_batches_and_releases_gpu(self):
        self.engine._model.compute_score.return_value = [0.1] * 16
        with patch.object(production, "monotonic", side_effect=[0, 0, 61]):
            with self.assertRaises(production_core.HTTPException) as failure:
                self.engine.rerank("q", ["a"] * 17)
        self.assertEqual(failure.exception.status_code, 503)
        self.assertEqual(self.engine._model.compute_score.call_count, 1)
        self.assertFalse(production._gpu_lock.locked())

    def test_encoder_uses_the_same_gpu_lock(self):
        encoder = production.PinnedBgeM3Encoder.__new__(production.PinnedBgeM3Encoder)
        encoder._model = Mock()
        lock = Mock()
        lock.acquire.return_value = False
        with patch.object(production, "_gpu_lock", lock):
            with self.assertRaises(production_core.HTTPException):
                encoder.encode(["q"])
        encoder._model.encode.assert_not_called()

    def test_model_load_is_pinned_and_requires_actual_cuda_inference(self):
        device = Mock()
        device.type = "cuda"
        device.__str__ = Mock(return_value="cuda:0")
        model = Mock()
        model.model.parameters.return_value = iter([SimpleNamespace(device=device)])
        model.compute_score.return_value = [0.8, 0.2]
        torch = Mock()
        torch.cuda.is_available.return_value = True
        torch.cuda.get_device_name.return_value = "fixture T4"
        torch.cuda.memory_allocated.return_value = 1234
        factory = Mock(return_value=model)
        with patch.dict(sys.modules, {"torch": torch, "FlagEmbedding": SimpleNamespace(FlagReranker=factory)}):
            with patch.object(production, "snapshot_download", return_value="/pinned/snapshot") as download:
                engine = production.PinnedBgeReranker()
        self.assertEqual(download.call_args.kwargs["revision"], production.RERANKER_MODEL_REVISION)
        factory.assert_called_once_with("/pinned/snapshot", use_fp16=True, devices=["cuda:0"], trust_remote_code=False)
        model.compute_score.assert_called_once()
        torch.cuda.synchronize.assert_called_once()
        with patch.object(production, "reranker", engine):
            info = self.client.get("/info").json()
        self.assertTrue(info["ready"])
        self.assertTrue(info["gpu_execution_verified"])
        self.assertEqual(info["device"], "cuda:0")
        self.assertEqual(info["probe_pairs"], 2)

    def test_available_cuda_cannot_hide_a_cpu_model(self):
        torch = Mock()
        torch.cuda.is_available.return_value = True
        model = Mock()
        model.model.parameters.return_value = iter([SimpleNamespace(device=SimpleNamespace(type="cpu"))])
        model.compute_score.return_value = [0.5, 0.2]
        with patch.dict(sys.modules, {"torch": torch, "FlagEmbedding": SimpleNamespace(FlagReranker=Mock(return_value=model))}):
            with patch.object(production, "snapshot_download", return_value="/pinned/snapshot"):
                with self.assertRaisesRegex(production_core.ReleaseError, "not on CUDA"):
                    production.PinnedBgeReranker()

    def test_no_cuda_refuses_download_and_readiness(self):
        torch = Mock()
        torch.cuda.is_available.return_value = False
        with patch.dict(sys.modules, {"torch": torch}), patch.object(production, "snapshot_download") as download:
            with self.assertRaisesRegex(production_core.ReleaseError, "CUDA is required"):
                production.PinnedBgeReranker()
        download.assert_not_called()

    def test_combined_startup_loads_both_and_clears_reranker(self):
        events = []

        @asynccontextmanager
        async def retrieval_lifespan(app):
            events.append("retrieval ready")
            yield
            events.append("retrieval stopped")

        with patch.object(production_core, "lifespan", retrieval_lifespan):
            with patch.object(production, "PinnedBgeReranker", return_value=self.engine):
                with TestClient(production.app) as client:
                    self.assertEqual(events, ["retrieval ready"])
                    self.assertTrue(client.get("/info").json()["ready"])
        self.assertIsNone(production.reranker)
        self.assertEqual(events, ["retrieval ready", "retrieval stopped"])


if __name__ == "__main__":
    unittest.main()
