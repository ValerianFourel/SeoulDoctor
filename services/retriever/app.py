"""Private, immutable BGE-M3 sparse+dense review retrieval Space."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MANIFEST_SCHEMA = "seouldoc.bge-m3-review-manifest/v1"
RESULTS_SCHEMA = "seouldoc.semantic-results/v1"
MAX_FACILITIES, MAX_QUERIES, MAX_LIMIT = 50, 64, 20


class RetrievalQuery(BaseModel):
    query_id: str
    text: str = Field(min_length=1, max_length=2000)


class RetrievalRequest(BaseModel):
    facility_ids: list[str]
    queries: list[RetrievalQuery]
    limit_per_facility_per_channel: int = Field(ge=1, le=MAX_LIMIT)


class Encoder(Protocol):
    def encode(self, texts: list[str]) -> tuple[np.ndarray, list[dict[str, float]]]: ...


class HashEncoder:
    """Deterministic fixture encoder; production injects pinned BGE-M3."""
    def encode(self, texts: list[str]) -> tuple[np.ndarray, list[dict[str, float]]]:
        vectors = np.zeros((len(texts), 8), dtype=np.float32)
        sparse: list[dict[str, float]] = []
        for row, text in enumerate(texts):
            tokens = text.casefold().split()
            sparse.append({token: 1.0 for token in tokens})
            for token in tokens:
                vectors[row, int(hashlib.sha256(token.encode()).hexdigest(), 16) % 8] += 1
            norm = np.linalg.norm(vectors[row])
            if norm: vectors[row] /= norm
        return vectors, sparse


class ReleaseError(RuntimeError): pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Release:
    def __init__(self, root: Path, encoder: Encoder):
        self.root, self.encoder = root, encoder
        self.manifest = self._load_manifest()
        self.reviews = self._load_reviews()
        self.ranges = self._load_ranges()
        self.dense = np.load(root / "dense.npy", mmap_mode="r")
        self.sparse = self._load_sparse()
        self._validate()

    def _load_manifest(self) -> dict[str, Any]:
        try: data = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        except Exception as exc: raise ReleaseError("manifest unreadable") from exc
        if data.get("schema") != MANIFEST_SCHEMA or data.get("model_id") != "BAAI/bge-m3":
            raise ReleaseError("unsupported manifest or model")
        if not isinstance(data.get("model_revision"), str) or not data["model_revision"].strip():
            raise ReleaseError("model revision is required")
        return data

    def _load_reviews(self) -> list[dict[str, str]]:
        try:
            rows = [json.loads(line) for line in (self.root / "reviews.jsonl").read_text(encoding="utf-8").splitlines() if line]
        except Exception as exc: raise ReleaseError("reviews unreadable") from exc
        if any(set(row) != {"evidence_id", "facility_id", "text"} or not all(isinstance(row[k], str) and row[k] for k in row) for row in rows):
            raise ReleaseError("invalid review row")
        if len({row["evidence_id"] for row in rows}) != len(rows): raise ReleaseError("duplicate evidence ID")
        return rows

    def _load_ranges(self) -> dict[str, list[int]]:
        try: data = json.loads((self.root / "facility_ranges.json").read_text(encoding="utf-8"))
        except Exception as exc: raise ReleaseError("ranges unreadable") from exc
        if not isinstance(data, dict): raise ReleaseError("invalid ranges")
        return data

    def _load_sparse(self) -> list[dict[str, float]]:
        try: rows = [json.loads(line) for line in (self.root / "sparse.jsonl").read_text(encoding="utf-8").splitlines() if line]
        except Exception as exc: raise ReleaseError("sparse vectors unreadable") from exc
        if any(not isinstance(row, dict) or any(not isinstance(k, str) or not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(float(v)) or v < 0 for k, v in row.items()) for row in rows):
            raise ReleaseError("invalid sparse vector")
        return [{k: float(v) for k, v in row.items()} for row in rows]

    def _validate(self) -> None:
        count, m = len(self.reviews), self.manifest
        if m.get("review_count") != count or len(self.sparse) != count or self.dense.shape != (count, int(m.get("dimension", -1))) or self.dense.dtype != np.float32:
            raise ReleaseError("artifact row or dimension mismatch")
        dense_norms = np.linalg.norm(self.dense, axis=1)
        if not np.isfinite(self.dense).all() or np.any(
            (dense_norms < 0.999) | (dense_norms > 1.001)
        ):
            raise ReleaseError("dense vectors are not finite L2-normalized")
        covered: list[int] = []
        for facility, bounds in self.ranges.items():
            if not isinstance(facility, str) or not isinstance(bounds, list) or len(bounds) != 2 or any(isinstance(v, bool) or not isinstance(v, int) for v in bounds) or not 0 <= bounds[0] < bounds[1] <= count:
                raise ReleaseError("invalid facility range")
            covered.extend(range(*bounds))
            if any(self.reviews[i]["facility_id"] != facility for i in range(*bounds)): raise ReleaseError("facility range ownership mismatch")
        if covered != list(range(count)): raise ReleaseError("facility ranges do not cover rows")
        for name in ("reviews.jsonl", "facility_ranges.json", "dense.npy", "sparse.jsonl"):
            record = m.get("artifacts", {}).get(name, {})
            path = self.root / name
            if record.get("sha256") != _sha256(path) or record.get("bytes") != path.stat().st_size: raise ReleaseError(f"artifact digest mismatch: {name}")
        if m.get("review_source_sha256") != _sha256(self.root / "reviews.jsonl"): raise ReleaseError("review source digest mismatch")

    def retrieve(self, facility_ids: list[str], queries: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
        if not 1 <= len(facility_ids) <= MAX_FACILITIES or len(set(facility_ids)) != len(facility_ids): raise ValueError("facility_ids limit or duplicates")
        if not 1 <= len(queries) <= MAX_QUERIES or not 1 <= limit <= MAX_LIMIT: raise ValueError("request limit exceeded")
        texts = [q["text"] for q in queries]
        if any(not isinstance(q.get("query_id"), str) or not q["query_id"] or not isinstance(q.get("text"), str) or not 1 <= len(q["text"]) <= 2000 for q in queries): raise ValueError("invalid query")
        vectors, sparse_queries = self.encoder.encode(texts)
        if vectors.shape != (len(queries), self.dense.shape[1]): raise ReleaseError("encoder dimension mismatch")
        results: list[dict[str, Any]] = []
        for q, vector, sparse_query in zip(queries, vectors, sparse_queries):
            for facility in facility_ids:
                bounds = self.ranges.get(facility)
                if bounds is None: raise ValueError("facility absent from release")
                indices = range(*bounds)
                dense_order = sorted(indices, key=lambda i: (-float(np.dot(self.dense[i], vector)), self.reviews[i]["evidence_id"]))[:limit]
                sparse_order = sorted(indices, key=lambda i: (-sum(self.sparse[i].get(k, 0.0) * v for k, v in sparse_query.items()), self.reviews[i]["evidence_id"]))[:limit]
                for channel, ordered in (("bge_m3_dense", dense_order), ("bge_m3_sparse", sparse_order)):
                    for rank, i in enumerate(ordered, 1):
                        results.append({"query_id": q["query_id"], "evidence_id": self.reviews[i]["evidence_id"], "facility_id": facility, "channel": channel, "rank": rank, "score": float(np.dot(self.dense[i], vector)) if channel.endswith("dense") else float(sum(self.sparse[i].get(k, 0.0) * v for k, v in sparse_query.items()))})
        return results


def create_app(release: Release) -> FastAPI:
    api = FastAPI(title="SeoulDoc BGE-M3 scoped retriever")
    @api.get("/health")
    def health() -> dict[str, Any]: return {"status": "ready", "release_id": release.manifest["release_id"], "model_id": release.manifest["model_id"], "model_revision": release.manifest["model_revision"]}
    @api.post("/v1/retrieve")
    def retrieve(request: RetrievalRequest) -> dict[str, Any]:
        try: results = release.retrieve(request.facility_ids, [q.model_dump() for q in request.queries], request.limit_per_facility_per_channel)
        except (ValueError, ReleaseError) as exc: raise HTTPException(400, str(exc)) from exc
        return {"schema_version": RESULTS_SCHEMA, "release": {"release_id": release.manifest["release_id"], "review_source_sha256": release.manifest["review_source_sha256"], "model_id": release.manifest["model_id"], "model_revision": release.manifest["model_revision"]}, "results": results}
    return api


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("build", nargs="?"); parser.parse_args()
    release = Release(Path(os.getenv("RETRIEVER_RELEASE_DIR", "release")), HashEncoder())
    globals()["app"] = create_app(release)
    import uvicorn; uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "7860")))


release = Release(Path(os.getenv("RETRIEVER_RELEASE_DIR", "release")), HashEncoder()) if Path(os.getenv("RETRIEVER_RELEASE_DIR", "release")).exists() else None
app = create_app(release) if release else FastAPI()

if __name__ == "__main__": main()
