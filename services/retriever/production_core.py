"""Production BGE-M3 sparse+dense retriever for a private HF Space."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from huggingface_hub import snapshot_download
from pydantic import BaseModel, ConfigDict, Field

MANIFEST_SCHEMA = "seouldoc.bge-m3-review-manifest/v2"
RESULTS_SCHEMA = "seouldoc.semantic-results/v1"
MODEL_ID = "BAAI/bge-m3"
MAX_FACILITIES = 50
MAX_QUERIES = 64
MAX_QUERY_CHARS = 2_000
MAX_LIMIT = 20
ARTIFACT_NAMES = (
    "evidence_ids.npy",
    "facility_ids.npy",
    "facility_ranges.json",
    "dense.npy",
    "sparse_indptr.npy",
    "sparse_indices.npy",
    "sparse_values.npy",
)


class ReleaseError(RuntimeError):
    pass


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class EncodedQueries:
    dense: np.ndarray
    sparse: tuple[dict[int, float], ...]


class BgeM3Encoder:
    def __init__(self, revision: str) -> None:
        if not revision.strip():
            raise ReleaseError("BGE_M3_MODEL_REVISION is required")
        from FlagEmbedding import BGEM3FlagModel

        self.revision = revision.strip()
        self._model = BGEM3FlagModel(
            MODEL_ID,
            use_fp16=True,
            trust_remote_code=False,
            revision=self.revision,
        )

    def encode(self, texts: list[str]) -> EncodedQueries:
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
            raise ReleaseError("BGE-M3 returned invalid dense query vectors")
        dense /= norms
        sparse: list[dict[int, float]] = []
        for row in result["lexical_weights"]:
            parsed = {int(token): float(weight) for token, weight in row.items()}
            if any(
                not math.isfinite(value) or value < 0.0 for value in parsed.values()
            ):
                raise ReleaseError("BGE-M3 returned invalid sparse query weights")
            sparse.append(parsed)
        return EncodedQueries(dense, tuple(sparse))


class SemanticRelease:
    """Validated immutable review vectors, searched only inside facility ranges."""

    def __init__(self, root: Path, encoder: BgeM3Encoder) -> None:
        self.root = root.resolve()
        self.encoder = encoder
        self.manifest = self._read_manifest()
        self.evidence_ids = np.load(
            root / "evidence_ids.npy", mmap_mode="r", allow_pickle=False
        )
        self.facility_ids = np.load(
            root / "facility_ids.npy", mmap_mode="r", allow_pickle=False
        )
        self.dense = np.load(root / "dense.npy", mmap_mode="r", allow_pickle=False)
        self.sparse_indptr = np.load(
            root / "sparse_indptr.npy", mmap_mode="r", allow_pickle=False
        )
        self.sparse_indices = np.load(
            root / "sparse_indices.npy", mmap_mode="r", allow_pickle=False
        )
        self.sparse_values = np.load(
            root / "sparse_values.npy", mmap_mode="r", allow_pickle=False
        )
        self.ranges = self._read_ranges()
        self._validate()

    def _read_manifest(self) -> dict[str, Any]:
        try:
            manifest = json.loads((self.root / "manifest.json").read_text("utf-8"))
        except Exception as exc:
            raise ReleaseError("semantic manifest is unreadable") from exc
        if manifest.get("schema") != MANIFEST_SCHEMA:
            raise ReleaseError("unsupported semantic manifest schema")
        if manifest.get("model_id") != MODEL_ID:
            raise ReleaseError("semantic release uses the wrong model")
        if manifest.get("model_revision") != self.encoder.revision:
            raise ReleaseError("semantic release and query encoder revisions differ")
        if (
            not isinstance(manifest.get("release_id"), str)
            or not manifest["release_id"]
        ):
            raise ReleaseError("semantic release_id is required")
        digest = manifest.get("review_source_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ReleaseError("raw review source digest is invalid")
        return manifest

    def _read_ranges(self) -> dict[str, tuple[int, int]]:
        try:
            raw = json.loads((self.root / "facility_ranges.json").read_text("utf-8"))
        except Exception as exc:
            raise ReleaseError("facility ranges are unreadable") from exc
        if not isinstance(raw, dict):
            raise ReleaseError("facility ranges must be an object")
        output: dict[str, tuple[int, int]] = {}
        for facility_id, bounds in raw.items():
            if (
                not isinstance(facility_id, str)
                or not facility_id
                or not isinstance(bounds, list)
                or len(bounds) != 2
                or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in bounds
                )
            ):
                raise ReleaseError("facility range is invalid")
            output[facility_id] = (bounds[0], bounds[1])
        return output

    def _validate(self) -> None:
        count = int(self.manifest.get("review_count", -1))
        dimension = int(self.manifest.get("dimension", -1))
        if count <= 0 or dimension != 1024:
            raise ReleaseError("semantic release count or dimension is invalid")
        if self.evidence_ids.shape != (count,) or self.facility_ids.shape != (count,):
            raise ReleaseError("semantic ID arrays have the wrong shape")
        if self.dense.shape != (count, dimension) or self.dense.dtype not in {
            np.dtype("<f2"),
            np.dtype("<f4"),
        }:
            raise ReleaseError("semantic dense matrix has the wrong shape or dtype")
        if self.sparse_indptr.shape != (
            count + 1,
        ) or self.sparse_indptr.dtype != np.dtype("<u8"):
            raise ReleaseError("semantic sparse offsets are invalid")
        nnz = int(self.sparse_indptr[-1])
        if (
            int(self.sparse_indptr[0]) != 0
            or np.any(np.diff(self.sparse_indptr) < 0)
            or self.sparse_indices.shape != (nnz,)
            or self.sparse_values.shape != (nnz,)
            or self.sparse_indices.dtype != np.dtype("<u4")
            or self.sparse_values.dtype not in {np.dtype("<f2"), np.dtype("<f4")}
        ):
            raise ReleaseError("semantic sparse matrix is invalid")
        if (
            not np.isfinite(self.dense).all()
            or not np.isfinite(self.sparse_values).all()
        ):
            raise ReleaseError("semantic vectors contain non-finite values")
        if np.any(self.sparse_values < 0.0):
            raise ReleaseError("semantic sparse weights cannot be negative")

        cursor = 0
        for facility_id, bounds in sorted(
            self.ranges.items(), key=lambda item: item[1]
        ):
            start, end = bounds
            if start != cursor or not start < end <= count:
                raise ReleaseError("facility ranges must cover every row exactly once")
            if any(str(value) != facility_id for value in self.facility_ids[start:end]):
                raise ReleaseError("facility range ownership mismatch")
            cursor = end
        if cursor != count:
            raise ReleaseError("facility ranges do not cover every review")
        if len({str(value) for value in self.evidence_ids}) != count:
            raise ReleaseError("semantic release contains duplicate evidence IDs")

        artifacts = self.manifest.get("artifacts")
        if not isinstance(artifacts, dict) or set(artifacts) != set(ARTIFACT_NAMES):
            raise ReleaseError("semantic artifact inventory is incomplete")
        for name in ARTIFACT_NAMES:
            record = artifacts[name]
            path = self.root / name
            if (
                not isinstance(record, dict)
                or record.get("bytes") != path.stat().st_size
                or record.get("sha256") != _sha256_file(path)
            ):
                raise ReleaseError(f"semantic artifact failed validation: {name}")

    def _rows(self, facility_id: str) -> range:
        bounds = self.ranges.get(facility_id)
        if bounds is None:
            raise ValueError("facility is absent from the semantic release")
        return range(bounds[0], bounds[1])

    def _sparse_score(self, row: int, query: dict[int, float]) -> float:
        start = int(self.sparse_indptr[row])
        end = int(self.sparse_indptr[row + 1])
        return sum(
            float(weight) * query.get(int(token), 0.0)
            for token, weight in zip(
                self.sparse_indices[start:end],
                self.sparse_values[start:end],
            )
        )

    @staticmethod
    def _top(rows: Iterator[int], scores: dict[int, float], limit: int) -> list[int]:
        return sorted(rows, key=lambda row: (-scores[row], row))[:limit]

    def retrieve(
        self,
        facility_ids: list[str],
        queries: list[QueryCell],
        limit: int,
    ) -> list[dict[str, object]]:
        encoded = self.encoder.encode([item.text for item in queries])
        if encoded.dense.shape != (len(queries), self.dense.shape[1]):
            raise ReleaseError("query vector dimension mismatch")
        output: list[dict[str, object]] = []
        for query_position, query in enumerate(queries):
            for facility_id in facility_ids:
                if facility_id not in self.ranges:
                    continue
                rows = self._rows(facility_id)
                dense_scores = {
                    row: float(np.dot(self.dense[row], encoded.dense[query_position]))
                    for row in rows
                }
                sparse_scores = {
                    row: self._sparse_score(row, encoded.sparse[query_position])
                    for row in rows
                }
                for channel, scores in (
                    ("bge_m3_sparse", sparse_scores),
                    ("bge_m3_dense", dense_scores),
                ):
                    for rank, row in enumerate(self._top(iter(rows), scores, limit), 1):
                        output.append(
                            {
                                "query_id": query.query_id,
                                "evidence_id": str(self.evidence_ids[row]),
                                "facility_id": facility_id,
                                "channel": channel,
                                "rank": rank,
                                "score": scores[row],
                            }
                        )
        return output


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExpectedRelease(StrictModel):
    release_id: str = Field(min_length=1, max_length=128)
    review_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_id_schema: str = Field(pattern=r"^seouldoc\.raw-review-evidence-id/v1$")


class QueryCell(StrictModel):
    query_id: str = Field(min_length=1, max_length=128)
    constraint_id: str = Field(min_length=1, max_length=128)
    role: str = Field(pattern=r"^(disease|support|risk)$")
    language: str = Field(pattern=r"^(en|ko)$")
    text: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)


class RetrievalRequest(StrictModel):
    schema_version: str = Field(pattern=r"^seouldoc\.semantic-query/v1$")
    request_id: str = Field(min_length=1, max_length=128)
    expected_release: ExpectedRelease
    facility_ids: list[str] = Field(min_length=1, max_length=MAX_FACILITIES)
    queries: list[QueryCell] = Field(min_length=1, max_length=MAX_QUERIES)
    limit_per_facility_per_channel: int = Field(ge=1, le=MAX_LIMIT)


def _release_directory() -> Path:
    configured = Path(os.getenv("RETRIEVER_RELEASE_DIR", "/data/release"))
    if configured.is_dir():
        return configured
    repository = os.getenv("RETRIEVER_DATASET_REPO", "").strip()
    revision = os.getenv("RETRIEVER_DATASET_REVISION", "").strip()
    if not repository or not revision:
        raise ReleaseError(
            "semantic release is absent and no pinned Dataset revision is configured"
        )
    downloaded = snapshot_download(
        repo_id=repository,
        repo_type="dataset",
        revision=revision,
        token=os.getenv("HF_TOKEN", "").strip() or None,
        local_dir=configured,
    )
    return Path(downloaded)


release: SemanticRelease | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global release
    encoder = BgeM3Encoder(os.environ["BGE_M3_MODEL_REVISION"])
    release = SemanticRelease(_release_directory(), encoder)
    yield
    release = None


app = FastAPI(title="SeoulDoc BGE-M3 review retriever", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, object]:
    if release is None:
        raise HTTPException(503, "semantic release is not ready")
    return {
        "status": "ready",
        "release_id": release.manifest["release_id"],
        "review_source_sha256": release.manifest["review_source_sha256"],
        "model_id": release.manifest["model_id"],
        "model_revision": release.manifest["model_revision"],
        "review_count": release.manifest["review_count"],
    }


@app.post("/v1/retrieve")
def retrieve(request: RetrievalRequest) -> dict[str, object]:
    if release is None:
        raise HTTPException(503, "semantic release is not ready")
    expected = request.expected_release
    if (
        expected.release_id != release.manifest["release_id"]
        or expected.review_source_sha256 != release.manifest["review_source_sha256"]
    ):
        raise HTTPException(409, "semantic release does not match the caller")
    if len(set(request.facility_ids)) != len(request.facility_ids):
        raise HTTPException(400, "facility IDs are duplicated")
    if len({item.query_id for item in request.queries}) != len(request.queries):
        raise HTTPException(400, "query IDs are duplicated")
    try:
        results = release.retrieve(
            request.facility_ids,
            request.queries,
            request.limit_per_facility_per_channel,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {
        "schema_version": RESULTS_SCHEMA,
        "release": {
            "release_id": release.manifest["release_id"],
            "review_source_sha256": release.manifest["review_source_sha256"],
            "model_id": release.manifest["model_id"],
            "model_revision": release.manifest["model_revision"],
        },
        "execution": {
            "gpu_execution_verified": True,
            "device": release.encoder.gpu_proof["device"],
        },
        "results": results,
    }
