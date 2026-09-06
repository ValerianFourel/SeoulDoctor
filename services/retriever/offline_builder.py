"""Build a resumable BGE-M3 sparse+dense review release offline."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pyarrow.parquet as pq

MANIFEST_SCHEMA = "seouldoc.bge-m3-review-manifest/v2"
SHARD_SCHEMA = "seouldoc.bge-m3-review-shard/v1"
EVIDENCE_ID_SCHEMA = "seouldoc.raw-review-evidence-id/v1"
MODEL_ID = "BAAI/bge-m3"
MODEL_DIMENSION = 1024
ARTIFACT_NAMES = (
    "evidence_ids.npy",
    "facility_ids.npy",
    "facility_ranges.json",
    "dense.npy",
    "sparse_indptr.npy",
    "sparse_indices.npy",
    "sparse_values.npy",
)


class BuildError(RuntimeError):
    """The source, checkpoint, model output, or assembled release is invalid."""


@dataclass(frozen=True)
class ReviewRecord:
    facility_id: str
    evidence_id: str
    text: str


@dataclass(frozen=True)
class EncodedBatch:
    dense: np.ndarray
    sparse: tuple[dict[int, float], ...]


class CorpusEncoder(Protocol):
    def encode(self, texts: Sequence[str]) -> EncodedBatch: ...


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    serialized = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return f"{serialized}\n".encode()


def _atomic_write(path: Path, content: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _raw_review_evidence_id(place_id: str, review_index: int, text: str) -> str:
    identity = f"{place_id}|{review_index}|{text}"
    return f"review:{sha256(identity.encode('utf-8')).hexdigest()[:20]}"


def _records_digest(records: Sequence[ReviewRecord]) -> str:
    digest = sha256()
    for record in records:
        for value in (record.facility_id, record.evidence_id, record.text):
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    return digest.hexdigest()


def load_reviews(path: Path, max_reviews: int | None = None) -> list[ReviewRecord]:
    """Read nonempty reviews and return a stable facility/evidence ordering."""
    parquet = pq.ParquetFile(path)
    names = set(parquet.schema_arrow.names)
    required = {"place_id", "review_index", "review_text"}
    if not required.issubset(names):
        raise BuildError(f"review source missing columns: {sorted(required - names)}")
    if max_reviews is not None and max_reviews <= 0:
        raise BuildError("max_reviews must be positive")

    records: list[ReviewRecord] = []
    stop = False
    for batch in parquet.iter_batches(columns=sorted(required), batch_size=65_536):
        columns = batch.to_pydict()
        for row in range(batch.num_rows):
            facility_id = str(columns["place_id"][row] or "").strip()
            text = str(columns["review_text"][row] or "").strip()
            raw_index = columns["review_index"][row]
            if not text:
                continue
            if not facility_id:
                raise BuildError("nonempty review has no place_id")
            if raw_index is None:
                raise BuildError("nonempty review has no review_index")
            evidence_id = _raw_review_evidence_id(facility_id, int(raw_index), text)
            records.append(ReviewRecord(facility_id, evidence_id, text))
            if max_reviews is not None and len(records) >= max_reviews:
                stop = True
                break
        if stop:
            break

    records.sort(
        key=lambda item: (
            item.facility_id.encode("utf-8"),
            item.evidence_id.encode("utf-8"),
        )
    )
    if not records:
        raise BuildError("review source contains no usable reviews")
    if len({item.evidence_id for item in records}) != len(records):
        raise BuildError("review source produces duplicate evidence IDs")
    return records


class BgeM3CorpusEncoder:
    """Pinned single-GPU BGE-M3 corpus encoder."""

    def __init__(
        self,
        revision: str,
        batch_size: int,
        max_length: int,
        token: str | None,
    ) -> None:
        if len(revision) != 40 or any(
            character not in "0123456789abcdef" for character in revision
        ):
            raise BuildError(
                "model_revision must be a lowercase 40-character commit SHA"
            )
        if batch_size <= 0 or max_length <= 0:
            raise BuildError("batch_size and max_length must be positive")
        from huggingface_hub import snapshot_download

        snapshot = snapshot_download(
            repo_id=MODEL_ID,
            revision=revision,
            token=token,
            ignore_patterns=("onnx/**", "*.jpg", "*.webp", ".DS_Store"),
        )
        from FlagEmbedding import BGEM3FlagModel

        self.batch_size = batch_size
        self.max_length = max_length
        self._model = BGEM3FlagModel(
            snapshot,
            use_fp16=True,
            pooling_method="cls",
            devices=["cuda:0"],
            trust_remote_code=False,
        )

    def encode(self, texts: Sequence[str]) -> EncodedBatch:
        result = self._model.encode_corpus(
            list(texts),
            batch_size=min(self.batch_size, len(texts)),
            max_length=self.max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense = np.asarray(result["dense_vecs"], dtype=np.float32)
        sparse = tuple(
            {int(token): float(weight) for token, weight in row.items()}
            for row in result["lexical_weights"]
        )
        return EncodedBatch(dense=dense, sparse=sparse)


def _normalize_batch(
    batch: EncodedBatch,
    count: int,
    sparse_top_k: int,
) -> EncodedBatch:
    dense = np.asarray(batch.dense, dtype=np.float32)
    expected_shape = (count, MODEL_DIMENSION)
    if dense.shape != expected_shape:
        raise BuildError(
            f"encoder returned dense shape {dense.shape}, expected {expected_shape}"
        )
    if not np.isfinite(dense).all():
        raise BuildError("encoder returned non-finite dense values")
    norms = np.linalg.norm(dense, axis=1, keepdims=True)
    if np.any(norms <= 0.0):
        raise BuildError("encoder returned a zero dense vector")
    dense /= norms
    if len(batch.sparse) != count:
        raise BuildError("encoder returned the wrong sparse row count")

    sparse_rows: list[dict[int, float]] = []
    for raw in batch.sparse:
        valid: list[tuple[int, float]] = []
        for token, weight in raw.items():
            token_id = int(token)
            score = float(weight)
            if not 0 <= token_id <= np.iinfo(np.uint32).max:
                raise BuildError("encoder returned a sparse token outside uint32")
            if not math.isfinite(score) or score < 0.0:
                raise BuildError("encoder returned an invalid sparse weight")
            if score > 0.0:
                valid.append((token_id, score))
        strongest = sorted(
            valid,
            key=lambda item: (-item[1], item[0]),
        )[:sparse_top_k]
        sparse_rows.append(dict(sorted(strongest)))
    return EncodedBatch(dense=dense, sparse=tuple(sparse_rows))


def _shard_paths(shards_dir: Path, number: int) -> tuple[Path, Path]:
    stem = f"shard-{number:06d}"
    return shards_dir / f"{stem}.npz", shards_dir / f"{stem}.json"


def _write_shard(
    path: Path,
    metadata_path: Path,
    records: Sequence[ReviewRecord],
    encoded: EncodedBatch,
    metadata: dict[str, Any],
) -> None:
    sparse_indptr = np.zeros(len(records) + 1, dtype="<u8")
    sparse_indices: list[int] = []
    sparse_values: list[float] = []
    for row, weights in enumerate(encoded.sparse):
        sparse_indices.extend(weights.keys())
        sparse_values.extend(weights.values())
        sparse_indptr[row + 1] = len(sparse_indices)

    evidence_width = max(len(item.evidence_id) for item in records)
    facility_width = max(len(item.facility_id) for item in records)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez(
            stream,
            evidence_ids=np.asarray(
                [item.evidence_id for item in records],
                dtype=f"<U{evidence_width}",
            ),
            facility_ids=np.asarray(
                [item.facility_id for item in records],
                dtype=f"<U{facility_width}",
            ),
            dense=np.asarray(encoded.dense, dtype="<f2"),
            sparse_indptr=sparse_indptr,
            sparse_indices=np.asarray(sparse_indices, dtype="<u4"),
            sparse_values=np.asarray(sparse_values, dtype="<f2"),
        )
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    complete = dict(metadata)
    complete.update(
        {
            "artifact_bytes": path.stat().st_size,
            "artifact_sha256": _sha256_file(path),
        }
    )
    _atomic_write(metadata_path, _canonical_json(complete))


def _valid_checkpoint(
    path: Path,
    metadata_path: Path,
    expected: dict[str, Any],
    records: Sequence[ReviewRecord],
) -> bool:
    if not path.is_file() or not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text("utf-8"))
        for key, value in expected.items():
            if metadata.get(key) != value:
                return False
        if metadata.get("records_sha256") != _records_digest(records):
            return False
        if metadata.get("artifact_bytes") != path.stat().st_size:
            return False
        if metadata.get("artifact_sha256") != _sha256_file(path):
            return False
        with np.load(path, allow_pickle=False) as shard:
            return (
                shard["evidence_ids"].shape == (len(records),)
                and shard["facility_ids"].shape == (len(records),)
                and shard["dense"].shape == (len(records), MODEL_DIMENSION)
                and shard["sparse_indptr"].shape == (len(records) + 1,)
                and int(shard["sparse_indptr"][-1]) == shard["sparse_indices"].shape[0]
                and shard["sparse_indices"].shape == shard["sparse_values"].shape
                and list(shard["evidence_ids"])
                == [item.evidence_id for item in records]
                and list(shard["facility_ids"])
                == [item.facility_id for item in records]
            )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def encode_shards(
    records: Sequence[ReviewRecord],
    encoder: CorpusEncoder,
    work_dir: Path,
    source_sha256: str,
    model_revision: str,
    shard_size: int,
    sparse_top_k: int,
    max_length: int,
) -> list[Path]:
    if shard_size <= 0 or sparse_top_k <= 0:
        raise BuildError("shard_size and sparse_top_k must be positive")
    shards_dir = work_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    total = len(records)
    for number, start in enumerate(range(0, total, shard_size)):
        end = min(start + shard_size, total)
        shard_records = records[start:end]
        path, metadata_path = _shard_paths(shards_dir, number)
        expected = {
            "schema": SHARD_SCHEMA,
            "shard_number": number,
            "start": start,
            "end": end,
            "review_source_sha256": source_sha256,
            "model_id": MODEL_ID,
            "model_revision": model_revision,
            "dimension": MODEL_DIMENSION,
            "sparse_top_k": sparse_top_k,
            "max_length": max_length,
        }
        if _valid_checkpoint(path, metadata_path, expected, shard_records):
            print(f"resume shard {number + 1}: rows {start}:{end}", flush=True)
            paths.append(path)
            continue
        started = time.monotonic()
        encoded = _normalize_batch(
            encoder.encode([item.text for item in shard_records]),
            len(shard_records),
            sparse_top_k,
        )
        metadata = dict(expected)
        metadata["records_sha256"] = _records_digest(shard_records)
        _write_shard(path, metadata_path, shard_records, encoded, metadata)
        elapsed = time.monotonic() - started
        print(
            f"encoded shard {number + 1}: rows {start}:{end} in {elapsed:.1f}s",
            flush=True,
        )
        paths.append(path)
    return paths


def _artifact_record(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def _flush_array(array: np.ndarray) -> None:
    flush = getattr(array, "flush", None)
    if callable(flush):
        flush()


def assemble_release(
    records: Sequence[ReviewRecord],
    shard_paths: Sequence[Path],
    work_dir: Path,
    release_id: str,
    source_sha256: str,
    model_revision: str,
    sparse_top_k: int,
    max_length: int,
) -> Path:
    release_dir = work_dir / "release"
    if release_dir.exists():
        manifest_path = release_dir / "manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text("utf-8"))
            if (
                manifest.get("release_id") == release_id
                and manifest.get("review_source_sha256") == source_sha256
                and manifest.get("model_revision") == model_revision
                and manifest.get("review_count") == len(records)
            ):
                validate_release(release_dir)
                return release_dir
        raise BuildError(
            f"release directory already exists with different content: {release_dir}"
        )

    nnz = 0
    for path in shard_paths:
        with np.load(path, allow_pickle=False) as shard:
            nnz += int(shard["sparse_indptr"][-1])

    with tempfile.TemporaryDirectory(
        prefix="release-stage-",
        dir=work_dir,
    ) as temporary:
        stage = Path(temporary)
        count = len(records)
        evidence_width = max(len(item.evidence_id) for item in records)
        facility_width = max(len(item.facility_id) for item in records)
        evidence_ids = np.lib.format.open_memmap(
            stage / "evidence_ids.npy",
            mode="w+",
            dtype=f"<U{evidence_width}",
            shape=(count,),
        )
        facility_ids = np.lib.format.open_memmap(
            stage / "facility_ids.npy",
            mode="w+",
            dtype=f"<U{facility_width}",
            shape=(count,),
        )
        dense = np.lib.format.open_memmap(
            stage / "dense.npy",
            mode="w+",
            dtype="<f2",
            shape=(count, MODEL_DIMENSION),
        )
        sparse_indptr = np.lib.format.open_memmap(
            stage / "sparse_indptr.npy",
            mode="w+",
            dtype="<u8",
            shape=(count + 1,),
        )
        sparse_indices = np.lib.format.open_memmap(
            stage / "sparse_indices.npy",
            mode="w+",
            dtype="<u4",
            shape=(nnz,),
        )
        sparse_values = np.lib.format.open_memmap(
            stage / "sparse_values.npy",
            mode="w+",
            dtype="<f2",
            shape=(nnz,),
        )
        sparse_indptr[0] = 0

        row_cursor = 0
        sparse_cursor = 0
        for path in shard_paths:
            with np.load(path, allow_pickle=False) as shard:
                rows = len(shard["evidence_ids"])
                shard_nnz = int(shard["sparse_indptr"][-1])
                row_end = row_cursor + rows
                sparse_end = sparse_cursor + shard_nnz
                evidence_ids[row_cursor:row_end] = shard["evidence_ids"]
                facility_ids[row_cursor:row_end] = shard["facility_ids"]
                dense[row_cursor:row_end] = shard["dense"]
                sparse_indptr[row_cursor + 1 : row_end + 1] = (
                    shard["sparse_indptr"][1:] + sparse_cursor
                )
                sparse_indices[sparse_cursor:sparse_end] = shard["sparse_indices"]
                sparse_values[sparse_cursor:sparse_end] = shard["sparse_values"]
                row_cursor = row_end
                sparse_cursor = sparse_end
        if row_cursor != count or sparse_cursor != nnz:
            raise BuildError("shards do not cover the assembled release")
        arrays = (
            evidence_ids,
            facility_ids,
            dense,
            sparse_indptr,
            sparse_indices,
            sparse_values,
        )
        for array in arrays:
            _flush_array(array)
        del evidence_ids, facility_ids, dense
        del sparse_indptr, sparse_indices, sparse_values

        ranges: dict[str, list[int]] = {}
        start = 0
        while start < count:
            facility_id = records[start].facility_id
            end = start + 1
            while end < count and records[end].facility_id == facility_id:
                end += 1
            ranges[facility_id] = [start, end]
            start = end
        _atomic_write(stage / "facility_ranges.json", _canonical_json(ranges))

        artifacts = {name: _artifact_record(stage / name) for name in ARTIFACT_NAMES}
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "release_id": release_id,
            "review_source_sha256": source_sha256,
            "evidence_id_schema": EVIDENCE_ID_SCHEMA,
            "model_id": MODEL_ID,
            "model_revision": model_revision,
            "review_count": count,
            "dimension": MODEL_DIMENSION,
            "encoding": {
                "dense_dtype": "float16",
                "dense_normalization": "l2",
                "sparse_index_dtype": "uint32",
                "sparse_value_dtype": "float16",
                "sparse_top_k": sparse_top_k,
                "max_length": max_length,
                "colbert_vectors": False,
            },
            "artifacts": artifacts,
        }
        _atomic_write(stage / "manifest.json", _canonical_json(manifest))
        validate_release(stage)
        os.replace(stage, release_dir)
    return release_dir


def validate_release(root: Path) -> dict[str, Any]:
    """Validate the actual files without loading the BGE-M3 model."""
    try:
        manifest = json.loads((root / "manifest.json").read_text("utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BuildError("release manifest is unreadable") from exc
    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("model_id") != MODEL_ID
    ):
        raise BuildError("release manifest identity is invalid")
    count = int(manifest.get("review_count", -1))
    if count <= 0 or manifest.get("dimension") != MODEL_DIMENSION:
        raise BuildError("release shape metadata is invalid")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(ARTIFACT_NAMES):
        raise BuildError("release artifact inventory is invalid")
    for name in ARTIFACT_NAMES:
        path = root / name
        record = artifacts[name]
        if (
            not path.is_file()
            or record.get("bytes") != path.stat().st_size
            or record.get("sha256") != _sha256_file(path)
        ):
            raise BuildError(f"release artifact failed its digest: {name}")

    evidence_ids = np.load(root / "evidence_ids.npy", mmap_mode="r", allow_pickle=False)
    facility_ids = np.load(root / "facility_ids.npy", mmap_mode="r", allow_pickle=False)
    dense = np.load(root / "dense.npy", mmap_mode="r", allow_pickle=False)
    indptr = np.load(root / "sparse_indptr.npy", mmap_mode="r", allow_pickle=False)
    indices = np.load(root / "sparse_indices.npy", mmap_mode="r", allow_pickle=False)
    values = np.load(root / "sparse_values.npy", mmap_mode="r", allow_pickle=False)
    if (
        evidence_ids.shape != (count,)
        or facility_ids.shape != (count,)
        or dense.shape != (count, MODEL_DIMENSION)
    ):
        raise BuildError("release dense or ID arrays have the wrong shape")
    if (
        dense.dtype != np.dtype("<f2")
        or indptr.dtype != np.dtype("<u8")
        or indices.dtype != np.dtype("<u4")
        or values.dtype != np.dtype("<f2")
    ):
        raise BuildError("release arrays have the wrong dtype")
    if (
        indptr.shape != (count + 1,)
        or int(indptr[0]) != 0
        or np.any(np.diff(indptr) < 0)
    ):
        raise BuildError("release sparse offsets are invalid")
    if indices.shape != values.shape or indices.shape != (int(indptr[-1]),):
        raise BuildError("release sparse payload shape is invalid")
    for start in range(0, count, 50_000):
        end = min(start + 50_000, count)
        if not np.isfinite(dense[start:end]).all():
            raise BuildError("release dense vectors contain non-finite values")
    for start in range(0, len(values), 1_000_000):
        chunk = values[start : start + 1_000_000]
        if not np.isfinite(chunk).all() or np.any(chunk < 0.0):
            raise BuildError("release sparse weights are invalid")

    ranges = json.loads((root / "facility_ranges.json").read_text("utf-8"))
    cursor = 0
    for facility_id, bounds in sorted(ranges.items(), key=lambda item: item[1]):
        start, end = bounds
        if start != cursor or not start < end <= count:
            raise BuildError("facility ranges do not cover the release")
        if any(str(value) != facility_id for value in facility_ids[start:end]):
            raise BuildError("facility range ownership is invalid")
        cursor = end
    if cursor != count or len({str(value) for value in evidence_ids}) != count:
        raise BuildError("release coverage or evidence identity is invalid")
    return manifest


def upload_release(release_dir: Path, repo_id: str, token: str) -> str:
    if not token:
        raise BuildError("HF_TOKEN is required for Dataset publication")
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=True,
        exist_ok=True,
    )
    result = api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=release_dir,
        commit_message=f"Publish {validate_release(release_dir)['release_id']}",
    )
    commit = getattr(result, "oid", None)
    if not commit:
        commit = api.dataset_info(repo_id).sha
    return str(commit)


def build_release(
    reviews_path: Path,
    work_dir: Path,
    encoder: CorpusEncoder,
    release_id: str,
    model_revision: str,
    shard_size: int = 25_000,
    sparse_top_k: int = 128,
    max_length: int = 128,
    max_reviews: int | None = None,
) -> Path:
    reviews_path = reviews_path.resolve()
    if not reviews_path.is_file():
        raise BuildError(f"review source is absent: {reviews_path}")
    work_dir.mkdir(parents=True, exist_ok=True)
    source_sha256 = _sha256_file(reviews_path)
    print(f"review source sha256: {source_sha256}", flush=True)
    records = load_reviews(reviews_path, max_reviews=max_reviews)
    print(f"usable reviews: {len(records)}", flush=True)
    shard_paths = encode_shards(
        records,
        encoder,
        work_dir,
        source_sha256,
        model_revision,
        shard_size,
        sparse_top_k,
        max_length,
    )
    return assemble_release(
        records,
        shard_paths,
        work_dir,
        release_id,
        source_sha256,
        model_revision,
        sparse_top_k,
        max_length,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--shard-size", type=int, default=25_000)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--sparse-top-k", type=int, default=128)
    parser.add_argument("--max-reviews", type=int)
    parser.add_argument("--dataset-repo")
    return parser


def main() -> int:
    args = _parser().parse_args()
    token = os.getenv("HF_TOKEN", "").strip() or None
    encoder = BgeM3CorpusEncoder(
        args.model_revision,
        args.batch_size,
        args.max_length,
        token,
    )
    started = time.monotonic()
    release_dir = build_release(
        reviews_path=args.reviews,
        work_dir=args.work_dir,
        encoder=encoder,
        release_id=args.release_id,
        model_revision=args.model_revision,
        shard_size=args.shard_size,
        sparse_top_k=args.sparse_top_k,
        max_length=args.max_length,
        max_reviews=args.max_reviews,
    )
    result: dict[str, Any] = {
        "release_dir": str(release_dir),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "manifest": validate_release(release_dir),
    }
    if args.dataset_repo:
        result["dataset_repo"] = args.dataset_repo
        result["dataset_revision"] = upload_release(
            release_dir,
            args.dataset_repo,
            token or "",
        )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
