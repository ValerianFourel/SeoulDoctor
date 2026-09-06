"""Canonical release metadata and strict read-only package validation."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sqlite3
import stat
from types import MappingProxyType
from typing import Any, Mapping
from urllib.parse import quote

import numpy as np
import pyarrow as pa
import pyarrow.ipc as pa_ipc

from .documents import (
    ANALYZER_SCHEMA,
    FACILITY_PROFILE_SCHEMA,
    NORMALIZATION_SCHEMA,
)


MANIFEST_SCHEMA = "seouldoc.search-index-manifest/v1"
POINTER_SCHEMA = "seouldoc.active-index/v1"
FACILITY_ARROW_SCHEMA = "seouldoc.facility-ordinals-arrow/v1"
FACILITY_SQLITE_SCHEMA = "seouldoc.facility-fts5/v1"
EVIDENCE_SQLITE_SCHEMA = "seouldoc.evidence-fts5/v1"
DENSE_SCHEMA = "seouldoc.dense-npy/v1"
FACILITY_SQLITE_APPLICATION_ID = 0x53444631
EVIDENCE_SQLITE_APPLICATION_ID = 0x53444531
SQLITE_USER_VERSION = 1
EXPECTED_DENSE_MODEL = "text-embedding-3-small"
EXPECTED_DENSE_DIMENSION = 1536
ARTIFACT_SCHEMAS = MappingProxyType({
    "facility_ordinals.arrow": FACILITY_ARROW_SCHEMA,
    "facility_vectors.npy": DENSE_SCHEMA,
    "facility_lexical.sqlite3": FACILITY_SQLITE_SCHEMA,
    "evidence_lexical.sqlite3": EVIDENCE_SQLITE_SCHEMA,
})
VERSION_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
HEX_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


class IndexLoadError(RuntimeError):
    """A release cannot be trusted or opened."""


@dataclass(frozen=True)
class ArtifactRecord:
    name: str
    schema: str
    byte_size: int
    sha256: str

    def __post_init__(self) -> None:
        if self.name not in ARTIFACT_SCHEMAS:
            raise IndexLoadError(f"unexpected artifact: {self.name}")
        if self.schema != ARTIFACT_SCHEMAS[self.name]:
            raise IndexLoadError(f"unsupported artifact schema: {self.name}")
        if self.byte_size <= 0 or not HEX_DIGEST_PATTERN.fullmatch(self.sha256):
            raise IndexLoadError(f"invalid artifact metadata: {self.name}")


@dataclass(frozen=True)
class IndexManifest:
    index_version: str
    content_id: str
    facility_source_sha256: str
    facility_source_bytes: int
    facility_source_rows: int
    review_source_sha256: str
    review_source_bytes: int
    review_source_rows: int
    embedding_model: str
    embedding_revision: str
    embedding_dimension: int
    source_collection: str
    source_vector_digest: str
    facility_count: int
    indexed_review_count: int
    review_facility_count: int
    orphan_review_count: int
    facility_evidence_count: int
    evidence_count: int
    id_map_digest: str
    profile_digest: str
    evidence_digest: str
    artifacts: tuple[ArtifactRecord, ...]

    def __post_init__(self) -> None:
        validate_version(self.index_version)
        digests = (
            self.content_id,
            self.facility_source_sha256,
            self.review_source_sha256,
            self.source_vector_digest,
            self.id_map_digest,
            self.profile_digest,
            self.evidence_digest,
        )
        if any(not HEX_DIGEST_PATTERN.fullmatch(value) for value in digests):
            raise IndexLoadError("manifest contains an invalid SHA-256 digest")
        counts = (
            self.facility_source_rows,
            self.review_source_rows,
            self.facility_count,
            self.indexed_review_count,
            self.review_facility_count,
            self.orphan_review_count,
            self.facility_evidence_count,
            self.evidence_count,
        )
        if any(value < 0 for value in counts):
            raise IndexLoadError("manifest counts cannot be negative")
        if self.facility_source_bytes <= 0 or self.review_source_bytes <= 0:
            raise IndexLoadError("manifest source files cannot be empty")
        if self.facility_source_rows != self.facility_count:
            raise IndexLoadError("facility source and index counts disagree")
        if self.indexed_review_count > self.review_source_rows:
            raise IndexLoadError("indexed reviews exceed source review rows")
        if self.review_facility_count > self.facility_count:
            raise IndexLoadError("review facility count exceeds facility count")
        if self.evidence_count != (
            self.indexed_review_count + self.facility_evidence_count
        ):
            raise IndexLoadError("evidence component counts disagree")
        if self.orphan_review_count != 0:
            raise IndexLoadError("review source contains orphan facility IDs")
        if self.embedding_model != EXPECTED_DENSE_MODEL:
            raise IndexLoadError("unsupported embedding model")
        if not self.embedding_revision.strip():
            raise IndexLoadError("embedding revision is required")
        if not self.source_collection.strip():
            raise IndexLoadError("source collection is required")
        if self.embedding_dimension != EXPECTED_DENSE_DIMENSION:
            raise IndexLoadError("unsupported embedding dimension")
        if {item.name for item in self.artifacts} != set(ARTIFACT_SCHEMAS):
            raise IndexLoadError("manifest artifact inventory is incomplete")

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": MANIFEST_SCHEMA,
            "index_version": self.index_version,
            "content_id": self.content_id,
            "formats": {
                "facility_ordinals": FACILITY_ARROW_SCHEMA,
                "facility_lexical": FACILITY_SQLITE_SCHEMA,
                "evidence_lexical": EVIDENCE_SQLITE_SCHEMA,
                "dense": DENSE_SCHEMA,
                "facility_profile": FACILITY_PROFILE_SCHEMA,
                "normalization": NORMALIZATION_SCHEMA,
                "analyzer": ANALYZER_SCHEMA,
            },
            "sources": {
                "facilities": {
                    "sha256": self.facility_source_sha256,
                    "bytes": self.facility_source_bytes,
                    "rows": self.facility_source_rows,
                },
                "reviews": {
                    "sha256": self.review_source_sha256,
                    "bytes": self.review_source_bytes,
                    "rows": self.review_source_rows,
                },
            },
            "dense": {
                "model": self.embedding_model,
                "model_revision": self.embedding_revision,
                "dimension": self.embedding_dimension,
                "dtype": "<f4",
                "normalization": "l2",
                "source_kind": "chroma-export",
                "source_collection": self.source_collection,
                "source_vector_digest": self.source_vector_digest,
            },
            "counts": {
                "facilities": self.facility_count,
                "indexed_reviews": self.indexed_review_count,
                "review_facilities": self.review_facility_count,
                "orphan_reviews": self.orphan_review_count,
                "facility_evidence": self.facility_evidence_count,
                "evidence": self.evidence_count,
            },
            "digests": {
                "id_map": self.id_map_digest,
                "profiles": self.profile_digest,
                "evidence_records": self.evidence_digest,
            },
            "artifacts": {
                item.name: {
                    "schema": item.schema,
                    "bytes": item.byte_size,
                    "sha256": item.sha256,
                }
                for item in sorted(self.artifacts, key=lambda item: item.name)
            },
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "IndexManifest":
        expected_keys = {
            "schema", "index_version", "content_id", "formats", "sources",
            "dense", "counts", "digests", "artifacts",
        }
        if set(payload) != expected_keys or payload.get("schema") != MANIFEST_SCHEMA:
            raise IndexLoadError("unsupported manifest shape")
        formats = _mapping(payload, "formats")
        expected_formats = {
            "facility_ordinals": FACILITY_ARROW_SCHEMA,
            "facility_lexical": FACILITY_SQLITE_SCHEMA,
            "evidence_lexical": EVIDENCE_SQLITE_SCHEMA,
            "dense": DENSE_SCHEMA,
            "facility_profile": FACILITY_PROFILE_SCHEMA,
            "normalization": NORMALIZATION_SCHEMA,
            "analyzer": ANALYZER_SCHEMA,
        }
        if formats != expected_formats:
            raise IndexLoadError("manifest format compatibility mismatch")
        sources = _mapping(payload, "sources")
        if set(sources) != {"facilities", "reviews"}:
            raise IndexLoadError("manifest source inventory mismatch")
        facilities = _mapping(sources, "facilities")
        reviews = _mapping(sources, "reviews")
        if set(facilities) != {"sha256", "bytes", "rows"} or set(reviews) != {
            "sha256", "bytes", "rows"
        }:
            raise IndexLoadError("manifest source metadata mismatch")
        dense = _mapping(payload, "dense")
        if set(dense) != {
            "model", "model_revision", "dimension", "dtype", "normalization",
            "source_kind", "source_collection", "source_vector_digest",
        }:
            raise IndexLoadError("manifest dense metadata mismatch")
        if (
            dense.get("dtype") != "<f4"
            or dense.get("normalization") != "l2"
            or dense.get("source_kind") != "chroma-export"
        ):
            raise IndexLoadError("manifest dense compatibility mismatch")
        counts = _mapping(payload, "counts")
        if set(counts) != {
            "facilities", "indexed_reviews", "review_facilities",
            "orphan_reviews", "facility_evidence", "evidence",
        }:
            raise IndexLoadError("manifest count metadata mismatch")
        digests = _mapping(payload, "digests")
        if set(digests) != {"id_map", "profiles", "evidence_records"}:
            raise IndexLoadError("manifest digest metadata mismatch")
        artifact_payload = _mapping(payload, "artifacts")
        if set(artifact_payload) != set(ARTIFACT_SCHEMAS):
            raise IndexLoadError("manifest artifact inventory mismatch")
        for name in artifact_payload:
            if set(_mapping(artifact_payload, name)) != {"schema", "bytes", "sha256"}:
                raise IndexLoadError(f"manifest artifact metadata mismatch: {name}")
        artifacts = tuple(
            ArtifactRecord(
                name=name,
                schema=str(_mapping(artifact_payload, name)["schema"]),
                byte_size=int(_mapping(artifact_payload, name)["bytes"]),
                sha256=str(_mapping(artifact_payload, name)["sha256"]),
            )
            for name in sorted(artifact_payload)
        )
        try:
            return cls(
                index_version=str(payload["index_version"]),
                content_id=str(payload["content_id"]),
                facility_source_sha256=str(facilities["sha256"]),
                facility_source_bytes=int(facilities["bytes"]),
                facility_source_rows=int(facilities["rows"]),
                review_source_sha256=str(reviews["sha256"]),
                review_source_bytes=int(reviews["bytes"]),
                review_source_rows=int(reviews["rows"]),
                embedding_model=str(dense["model"]),
                embedding_revision=str(dense["model_revision"]),
                embedding_dimension=int(dense["dimension"]),
                source_collection=str(dense["source_collection"]),
                source_vector_digest=str(dense["source_vector_digest"]),
                facility_count=int(counts["facilities"]),
                indexed_review_count=int(counts["indexed_reviews"]),
                review_facility_count=int(counts["review_facilities"]),
                orphan_review_count=int(counts["orphan_reviews"]),
                facility_evidence_count=int(counts["facility_evidence"]),
                evidence_count=int(counts["evidence"]),
                id_map_digest=str(digests["id_map"]),
                profile_digest=str(digests["profiles"]),
                evidence_digest=str(digests["evidence_records"]),
                artifacts=artifacts,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise IndexLoadError("invalid manifest values") from exc


@dataclass(frozen=True)
class ActivePointer:
    version: str
    manifest_sha256: str
    content_id: str

    def __post_init__(self) -> None:
        validate_version(self.version)
        if not HEX_DIGEST_PATTERN.fullmatch(self.manifest_sha256):
            raise IndexLoadError("active pointer has an invalid manifest digest")
        if not HEX_DIGEST_PATTERN.fullmatch(self.content_id):
            raise IndexLoadError("active pointer has an invalid content ID")

    def to_payload(self) -> dict[str, str]:
        return {
            "schema": POINTER_SCHEMA,
            "version": self.version,
            "manifest_sha256": self.manifest_sha256,
            "content_id": self.content_id,
        }


@dataclass
class ValidatedRelease:
    directory: Path
    manifest: IndexManifest
    manifest_sha256: str
    facility_ids: tuple[str, ...]
    profile_hashes: tuple[bytes, ...]
    vectors: np.ndarray

    def close(self) -> None:
        mmap = getattr(self.vectors, "_mmap", None)
        if mmap is not None:
            mmap.close()


def validate_version(version: str) -> str:
    if not VERSION_PATTERN.fullmatch(str(version)) or version in {".", ".."}:
        raise IndexLoadError("unsafe index version")
    return str(version)


def _mapping(mapping: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise IndexLoadError(f"manifest field must be an object: {key}")
    return value


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise IndexLoadError(f"cannot safely open artifact: {path.name}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise IndexLoadError(f"artifact is not a private regular file: {path.name}")
        digest = sha256()
        while chunk := os.read(descriptor, 4 * 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
        ):
            raise IndexLoadError(f"artifact changed during validation: {path.name}")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def write_manifest(path: Path, manifest: IndexManifest) -> str:
    data = canonical_json_bytes(manifest.to_payload())
    path.write_bytes(data)
    return sha256(data).hexdigest()


def read_manifest(path: Path) -> tuple[IndexManifest, str]:
    _validate_regular_file(path, require_readonly=False)
    raw = _read_small_regular_file(path, maximum_bytes=2 * 1024 * 1024)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IndexLoadError("manifest is not valid UTF-8 JSON") from exc
    if not isinstance(payload, Mapping) or raw != canonical_json_bytes(payload):
        raise IndexLoadError("manifest JSON is not canonical")
    manifest = IndexManifest.from_payload(payload)
    return manifest, sha256(raw).hexdigest()


def read_active_pointer(root: Path) -> ActivePointer:
    path = root / "active.json"
    _validate_regular_file(path, require_readonly=False)
    raw = _read_small_regular_file(path, maximum_bytes=16 * 1024)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IndexLoadError("active pointer is not valid UTF-8 JSON") from exc
    if (
        not isinstance(payload, Mapping)
        or raw != canonical_json_bytes(payload)
        or set(payload) != {"schema", "version", "manifest_sha256", "content_id"}
        or payload.get("schema") != POINTER_SCHEMA
    ):
        raise IndexLoadError("unsupported active pointer")
    return ActivePointer(
        version=str(payload["version"]),
        manifest_sha256=str(payload["manifest_sha256"]),
        content_id=str(payload["content_id"]),
    )


def write_active_pointer(root: Path, pointer: ActivePointer) -> None:
    root.mkdir(parents=True, exist_ok=True)
    _reject_symlink(root)
    temporary = root / f".active-{os.getpid()}-{os.urandom(6).hex()}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(temporary, flags, 0o600)
        try:
            data = canonical_json_bytes(pointer.to_payload())
            os.write(descriptor, data)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.chmod(temporary, 0o444)
        os.replace(temporary, root / "active.json")
        fsync_directory(root)
    finally:
        if os.path.lexists(temporary):
            temporary.unlink()


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def validate_release(
    directory: Path,
    *,
    expected_version: str,
    require_readonly: bool = True,
) -> ValidatedRelease:
    expected_version = validate_version(expected_version)
    _validate_directory(directory, require_readonly=require_readonly)
    manifest_path = directory / "manifest.json"
    manifest, manifest_digest = read_manifest(manifest_path)
    if manifest.index_version != expected_version:
        raise IndexLoadError("manifest index version does not match its directory")

    artifact_by_name = {item.name: item for item in manifest.artifacts}
    actual_names = {entry.name for entry in directory.iterdir()}
    expected_names = {"manifest.json", *artifact_by_name}
    if actual_names != expected_names:
        raise IndexLoadError("release contains missing or unlisted files")
    if require_readonly:
        _validate_regular_file(manifest_path, require_readonly=True)
    for name, record in artifact_by_name.items():
        path = directory / name
        file_stat = _validate_regular_file(path, require_readonly=require_readonly)
        if file_stat.st_size != record.byte_size:
            raise IndexLoadError(f"artifact byte size mismatch: {name}")
        if sha256_file(path) != record.sha256:
            raise IndexLoadError(f"artifact checksum mismatch: {name}")

    facility_ids, profile_hashes = _validate_arrow(
        directory / "facility_ordinals.arrow",
        manifest,
    )
    vectors = _validate_vectors(directory / "facility_vectors.npy", manifest)
    try:
        _validate_facility_sqlite(
            directory / "facility_lexical.sqlite3",
            manifest,
            facility_ids,
        )
        _validate_evidence_sqlite(
            directory / "evidence_lexical.sqlite3",
            manifest,
            facility_ids,
        )
    except BaseException:
        mmap = getattr(vectors, "_mmap", None)
        if mmap is not None:
            mmap.close()
        raise
    return ValidatedRelease(
        directory=directory,
        manifest=manifest,
        manifest_sha256=manifest_digest,
        facility_ids=facility_ids,
        profile_hashes=profile_hashes,
        vectors=vectors,
    )


def _reject_symlink(path: Path) -> os.stat_result:
    try:
        file_stat = path.lstat()
    except FileNotFoundError as exc:
        raise IndexLoadError(f"missing index path: {path}") from exc
    if stat.S_ISLNK(file_stat.st_mode):
        raise IndexLoadError(f"index path cannot be a symlink: {path}")
    return file_stat


def _validate_directory(path: Path, *, require_readonly: bool) -> os.stat_result:
    file_stat = _reject_symlink(path)
    if not stat.S_ISDIR(file_stat.st_mode):
        raise IndexLoadError(f"index path is not a directory: {path}")
    if require_readonly and file_stat.st_mode & 0o222:
        raise IndexLoadError(f"published index directory is writable: {path}")
    return file_stat


def _validate_regular_file(path: Path, *, require_readonly: bool) -> os.stat_result:
    file_stat = _reject_symlink(path)
    if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_nlink != 1:
        raise IndexLoadError(f"index artifact is not a private regular file: {path}")
    if require_readonly and file_stat.st_mode & 0o222:
        raise IndexLoadError(f"published index artifact is writable: {path}")
    return file_stat


def _read_small_regular_file(path: Path, *, maximum_bytes: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise IndexLoadError(f"cannot safely read index metadata: {path.name}") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size > maximum_bytes
        ):
            raise IndexLoadError(f"unsafe index metadata file: {path.name}")
        chunks: list[bytes] = []
        total = 0
        while chunk := os.read(descriptor, 64 * 1024):
            total += len(chunk)
            if total > maximum_bytes:
                raise IndexLoadError(f"index metadata is too large: {path.name}")
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
        ):
            raise IndexLoadError(f"index metadata changed while reading: {path.name}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _digest_text_rows(rows: tuple[str, ...]) -> str:
    digest = sha256()
    for value in rows:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _validate_arrow(
    path: Path,
    manifest: IndexManifest,
) -> tuple[tuple[str, ...], tuple[bytes, ...]]:
    with pa.memory_map(str(path), "r") as source:
        table = pa_ipc.open_file(source).read_all()
    expected_schema = pa.schema([
        pa.field("ordinal", pa.uint32(), nullable=False),
        pa.field("place_id", pa.string(), nullable=False),
        pa.field("profile_sha256", pa.binary(32), nullable=False),
    ])
    if table.schema != expected_schema or table.num_rows != manifest.facility_count:
        raise IndexLoadError("facility ordinal artifact schema or count mismatch")
    ordinals = table.column("ordinal").to_pylist()
    if ordinals != list(range(manifest.facility_count)):
        raise IndexLoadError("facility ordinals are not contiguous")
    facility_ids = tuple(table.column("place_id").to_pylist())
    if (
        len(set(facility_ids)) != len(facility_ids)
        or any(not value.strip() for value in facility_ids)
        or list(facility_ids) != sorted(facility_ids, key=lambda item: item.encode("utf-8"))
    ):
        raise IndexLoadError("facility IDs are empty, duplicated, or unstable")
    if _digest_text_rows(facility_ids) != manifest.id_map_digest:
        raise IndexLoadError("facility ID map digest mismatch")
    profile_hashes = tuple(table.column("profile_sha256").to_pylist())
    digest = sha256()
    for value in profile_hashes:
        digest.update(value)
    if digest.hexdigest() != manifest.profile_digest:
        raise IndexLoadError("facility profile digest mismatch")
    return facility_ids, profile_hashes


def _validate_vectors(path: Path, manifest: IndexManifest) -> np.ndarray:
    try:
        vectors = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise IndexLoadError("dense vector artifact is not a valid NPY file") from exc
    expected_shape = (manifest.facility_count, manifest.embedding_dimension)
    if vectors.shape != expected_shape or vectors.dtype.str != "<f4":
        raise IndexLoadError("dense vector shape or dtype mismatch")
    for start in range(0, len(vectors), 1024):
        batch = np.asarray(vectors[start:start + 1024], dtype=np.float32)
        if not np.isfinite(batch).all():
            raise IndexLoadError("dense vectors contain non-finite values")
        norms = np.linalg.norm(batch, axis=1)
        if not np.allclose(norms, 1.0, rtol=1e-4, atol=1e-5):
            raise IndexLoadError("dense vectors are not L2-normalized")
    return vectors


def _readonly_sqlite(path: Path) -> sqlite3.Connection:
    uri = f"file:{quote(str(path))}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    connection.execute("PRAGMA query_only = ON")
    return connection


def _sqlite_integrity(connection: sqlite3.Connection) -> None:
    rows = connection.execute("PRAGMA integrity_check").fetchall()
    if rows != [("ok",)]:
        raise IndexLoadError("SQLite integrity check failed")


def _validate_facility_sqlite(
    path: Path,
    manifest: IndexManifest,
    facility_ids: tuple[str, ...],
) -> None:
    connection = _readonly_sqlite(path)
    try:
        if connection.execute("PRAGMA application_id").fetchone()[0] != FACILITY_SQLITE_APPLICATION_ID:
            raise IndexLoadError("facility SQLite application ID mismatch")
        if connection.execute("PRAGMA user_version").fetchone()[0] != SQLITE_USER_VERSION:
            raise IndexLoadError("facility SQLite user version mismatch")
        _sqlite_integrity(connection)
        rows = connection.execute(
            "SELECT ordinal, place_id FROM facility_reference ORDER BY ordinal"
        ).fetchall()
        if rows != list(enumerate(facility_ids)):
            raise IndexLoadError("facility SQLite ordinal map mismatch")
        terms = connection.execute("SELECT count(*) FROM facility_terms").fetchone()[0]
        substrings = connection.execute(
            "SELECT count(*) FROM facility_substrings"
        ).fetchone()[0]
        if terms != manifest.facility_count or substrings != manifest.facility_count:
            raise IndexLoadError("facility FTS count mismatch")
    except sqlite3.DatabaseError as exc:
        raise IndexLoadError("facility SQLite validation failed") from exc
    finally:
        connection.close()


def _validate_evidence_sqlite(
    path: Path,
    manifest: IndexManifest,
    facility_ids: tuple[str, ...],
) -> None:
    connection = _readonly_sqlite(path)
    try:
        if connection.execute("PRAGMA application_id").fetchone()[0] != EVIDENCE_SQLITE_APPLICATION_ID:
            raise IndexLoadError("evidence SQLite application ID mismatch")
        if connection.execute("PRAGMA user_version").fetchone()[0] != SQLITE_USER_VERSION:
            raise IndexLoadError("evidence SQLite user version mismatch")
        _sqlite_integrity(connection)
        references = connection.execute(
            "SELECT ordinal, place_id FROM facility_reference ORDER BY ordinal"
        ).fetchall()
        if references != list(enumerate(facility_ids)):
            raise IndexLoadError("evidence SQLite ordinal map mismatch")
        counts = connection.execute(
            """
            SELECT count(*),
                   count(*) FILTER (WHERE is_verbatim = 1),
                   count(DISTINCT facility_ordinal) FILTER (WHERE is_verbatim = 1)
            FROM evidence_document
            """
        ).fetchone()
        if counts != (
            manifest.evidence_count,
            manifest.indexed_review_count,
            manifest.review_facility_count,
        ):
            raise IndexLoadError("evidence SQLite counts disagree with manifest")
        if connection.execute("PRAGMA foreign_key_check").fetchall():
            raise IndexLoadError("evidence SQLite contains orphan records")
        terms = connection.execute("SELECT count(*) FROM evidence_terms").fetchone()[0]
        substrings = connection.execute(
            "SELECT count(*) FROM evidence_substrings"
        ).fetchone()[0]
        if terms != manifest.evidence_count or substrings != manifest.evidence_count:
            raise IndexLoadError("evidence FTS count mismatch")
    except sqlite3.DatabaseError as exc:
        raise IndexLoadError("evidence SQLite validation failed") from exc
    finally:
        connection.close()
