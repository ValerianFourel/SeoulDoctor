"""Pinned production entrypoint for the strict v2 retrieval service."""

from __future__ import annotations

import os

import numpy as np
from huggingface_hub import snapshot_download

import production_core


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
        from FlagEmbedding import BGEM3FlagModel

        self.revision = normalized_revision
        self._model = BGEM3FlagModel(
            snapshot,
            use_fp16=True,
            trust_remote_code=False,
        )

    def encode(self, texts: list[str]) -> production_core.EncodedQueries:
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


# The core lifespan resolves this name when the Space starts.
production_core.BgeM3Encoder = PinnedBgeM3Encoder
app = production_core.app
