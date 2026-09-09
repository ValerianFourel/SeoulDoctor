"""Pinned production entrypoint for the strict v2 retrieval service."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager, contextmanager
from threading import Lock
from time import monotonic
from typing import Annotated

import numpy as np
from huggingface_hub import snapshot_download
from pydantic import BaseModel, ConfigDict, Field

import production_core

RERANKER_MODEL_ID = "BAAI/bge-reranker-v2-m3"
RERANKER_MODEL_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
GPU_WAIT_SECONDS = 10.0
RERANK_TIMEOUT_SECONDS = 60.0
RERANK_BATCH_SIZE = 16
_gpu_lock = Lock()


@contextmanager
def gpu_inference():
    if not _gpu_lock.acquire(timeout=GPU_WAIT_SECONDS):
        raise production_core.HTTPException(503, "GPU inference is busy")
    try:
        yield
    finally:
        _gpu_lock.release()


class RerankRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    query: str = Field(min_length=1, max_length=2_000)
    texts: list[Annotated[str, Field(min_length=1, max_length=32_000)]] = Field(
        min_length=1, max_length=256
    )


class PinnedBgeReranker:
    def __init__(self) -> None:
        import torch

        if not torch.cuda.is_available():
            raise production_core.ReleaseError("CUDA is required for the ncs reranker")
        snapshot = snapshot_download(
            repo_id=RERANKER_MODEL_ID,
            revision=RERANKER_MODEL_REVISION,
            token=os.getenv("HF_TOKEN", "").strip() or None,
            allow_patterns=("*.json", "*.safetensors", "*.model"),
        )
        from FlagEmbedding import FlagReranker

        self._model = FlagReranker(
            snapshot,
            use_fp16=True,
            devices=["cuda:0"],
            trust_remote_code=False,
        )
        probe = self.rerank("진료 설명 consultation", ["설명을 잘 해주세요", "대기가 길어요"])
        torch.cuda.synchronize()
        device = next(self._model.model.parameters()).device
        if device.type != "cuda":
            raise production_core.ReleaseError("BGE reranker model is not on CUDA")
        self.gpu_proof = {
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
            "model_revision": RERANKER_MODEL_REVISION,
            "probe_pairs": len(probe),
            "cuda_allocated_bytes": torch.cuda.memory_allocated(device),
        }

    def rerank(self, query: str, texts: list[str]) -> list[dict[str, int | float]]:
        deadline = monotonic() + RERANK_TIMEOUT_SECONDS
        scores: list[float] = []
        with gpu_inference():
            for start in range(0, len(texts), RERANK_BATCH_SIZE):
                if monotonic() >= deadline:
                    raise production_core.HTTPException(503, "GPU reranking timed out")
                batch = texts[start:start + RERANK_BATCH_SIZE]
                raw = self._model.compute_score(
                    [[query, text] for text in batch],
                    batch_size=RERANK_BATCH_SIZE,
                    max_length=512,
                    normalize=True,
                )
                values = np.asarray(raw, dtype=np.float64).reshape(-1)
                if len(values) != len(batch) or not np.isfinite(values).all():
                    raise production_core.ReleaseError("BGE reranker returned invalid scores")
                scores.extend(values.tolist())
        if monotonic() >= deadline:
            raise production_core.HTTPException(503, "GPU reranking timed out")
        return sorted(
            ({"index": index, "score": score} for index, score in enumerate(scores)),
            key=lambda item: (-item["score"], item["index"]),
        )


class PinnedBgeM3Encoder:
    """Load BGE-M3 only from the exact Hub commit requested by deployment."""

    def __init__(self, revision: str) -> None:
        normalized_revision = revision.strip()
        if not normalized_revision:
            raise production_core.ReleaseError("BGE_M3_MODEL_REVISION is required")
        snapshot = snapshot_download(
            repo_id=production_core.MODEL_ID,
            revision=normalized_revision,
            token=os.getenv("HF_TOKEN", "").strip() or None,
            ignore_patterns=("onnx/**", "*.jpg", "*.webp", ".DS_Store"),
        )
        import torch
        if not torch.cuda.is_available():
            raise production_core.ReleaseError("CUDA is required for the ncs retriever")
        from FlagEmbedding import BGEM3FlagModel

        self.revision = normalized_revision
        self._model = BGEM3FlagModel(
            snapshot,
            use_fp16=True,
            devices=["cuda:0"],
            trust_remote_code=False,
        )

        probe = self.encode(["clear explanation", "친절한 설명"])
        torch.cuda.synchronize()
        device = next(self._model.model.parameters()).device
        if device.type != "cuda":
            raise production_core.ReleaseError("BGE-M3 model is not on CUDA")
        self.gpu_proof = {
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
            "model_revision": self.revision,
            "probe_queries": 2,
            "dimension": int(probe.dense.shape[1]),
            "sparse_nonempty": all(bool(row) for row in probe.sparse),
            "cuda_allocated_bytes": torch.cuda.memory_allocated(device),
        }
        if probe.dense.shape != (2, 1024) or not self.gpu_proof["sparse_nonempty"]:
            raise production_core.ReleaseError("GPU embedding probe failed")

    def encode(self, texts: list[str]) -> production_core.EncodedQueries:
        with gpu_inference():
            result = self._model.encode(
                texts,
                batch_size=min(32, len(texts)),
                max_length=512,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False,
            )
        dense = np.asarray(result["dense_vecs"], dtype=np.float32)
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        if not np.isfinite(dense).all() or np.any(norms <= 0.0):
            raise production_core.ReleaseError("BGE-M3 returned invalid dense vectors")
        dense /= norms
        sparse = tuple(
            {int(token): float(weight) for token, weight in row.items()}
            for row in result["lexical_weights"]
        )
        return production_core.EncodedQueries(dense, sparse)


production_core.BgeM3Encoder = PinnedBgeM3Encoder
app = production_core.app
reranker: PinnedBgeReranker | None = None


@asynccontextmanager
async def combined_lifespan(application):
    global reranker
    async with production_core.lifespan(application):
        reranker = PinnedBgeReranker()
        try:
            yield
        finally:
            reranker = None


app.router.lifespan_context = combined_lifespan


@app.get("/info")
def reranker_info():
    if reranker is None:
        raise production_core.HTTPException(503, "reranker is not ready")
    return {
        "ready": True,
        "model_id": RERANKER_MODEL_ID,
        "model_revision": RERANKER_MODEL_REVISION,
        "gpu_execution_verified": True,
        **reranker.gpu_proof,
    }


@app.post("/rerank")
def rerank(request: RerankRequest):
    if reranker is None:
        raise production_core.HTTPException(503, "reranker is not ready")
    if sum(map(len, request.texts)) > 512_000:
        raise production_core.HTTPException(413, "rerank text budget exceeded")
    try:
        results = reranker.rerank(request.query, request.texts)
    except production_core.ReleaseError as exc:
        raise production_core.HTTPException(503, "reranker inference failed") from exc
    return {
        "model": RERANKER_MODEL_ID,
        "model_revision": RERANKER_MODEL_REVISION,
        "execution": {
            "gpu_execution_verified": True,
            "device": reranker.gpu_proof["device"],
        },
        "results": results,
    }


@app.get("/ready/gpu")
def gpu_readiness():
    if production_core.release is None:
        raise production_core.HTTPException(503, "semantic release is not ready")
    return {"ready": True, **production_core.release.encoder.gpu_proof}
