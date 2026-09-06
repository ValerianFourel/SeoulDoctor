"""Offline construction of immutable facility and evidence index releases."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import logging
import math
import os
from pathlib import Path
import sqlite3
from typing import Any, Mapping, Sequence

import chromadb
from chromadb.config import Settings
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as pa_ipc
import pyarrow.parquet as pq

from agentic_retrieval import build_specific_evidence_records
from raw_review_store import detect_language_hint

from .documents import (
    ANALYZER_SCHEMA,
    FACILITY_PROFILE_SCHEMA,
    NORMALIZATION_SCHEMA,
    FacilityIndexDocument,
    facility_index_document,
    hangul_ngrams,
    logical_record_digest,
    normalize_search_text,
    raw_review_evidence_id,
)
from .manifest import (
    ARTIFACT_SCHEMAS,
    DENSE_SCHEMA,
    EVIDENCE_SQLITE_APPLICATION_ID,
    EVIDENCE_SQLITE_SCHEMA,
    EXPECTED_DENSE_DIMENSION,
    EXPECTED_DENSE_MODEL,
    FACILITY_ARROW_SCHEMA,
    FACILITY_SQLITE_APPLICATION_ID,
    FACILITY_SQLITE_SCHEMA,
    SQLITE_USER_VERSION,
    ArtifactRecord,
    IndexLoadError,
    IndexManifest,
    fsync_directory,
    sha256_file,
    write_manifest,
)


SQLITE_BATCH_SIZE = 2_000
PARQUET_BATCH_SIZE = 50_000
logger = logging.getLogger(__name__)


class IndexBuildError(RuntimeError):
    """A source snapshot cannot produce a trustworthy release."""


@dataclass(frozen=True)
class BuildRequest:
    version: str
    facilities_path: Path
    reviews_path: Path
    chroma_path: Path
    chroma_collection: str = "seoul_med_agentic_v2"
    embedding_model: str = EXPECTED_DENSE_MODEL
    embedding_revision: str = "provider-managed"

    def __post_init__(self) -> None:
        object.__setattr__(self, "facilities_path", Path(self.facilities_path))
        object.__setattr__(self, "reviews_path", Path(self.reviews_path))
        object.__setattr__(self, "chroma_path", Path(self.chroma_path))
        if self.embedding_model != EXPECTED_DENSE_MODEL:
            raise IndexBuildError(
                f"Phase 3 can only export {EXPECTED_DENSE_MODEL} vectors"
            )
        if not self.embedding_revision.strip() or not self.chroma_collection.strip():
            raise IndexBuildError("embedding revision and Chroma collection are required")


@dataclass(frozen=True)
class BuildResult:
    manifest: IndexManifest
    manifest_sha256: str


@dataclass(frozen=True)
class _EvidenceDocument:
    evidence_id: str
    facility_ordinal: int
    source_type: str
    source_field: str
    source_index: int
    source_locator: str
    original_text: str
    language_hint: str
    visit_date: str
    scraped_at: str
    is_verbatim: bool


class _EvidenceDigest:
    def __init__(self) -> None:
        self._digest = sha256()

    def add(self, record: _EvidenceDocument) -> None:
        value = logical_record_digest((
            record.evidence_id,
            record.facility_ordinal,
            record.source_type,
            record.source_field,
            record.source_index,
            record.source_locator,
            record.original_text,
            record.language_hint,
            record.visit_date,
            record.scraped_at,
            int(record.is_verbatim),
        ))
        self._digest.update(bytes.fromhex(value))

    def hexdigest(self) -> str:
        return self._digest.hexdigest()


def release_matches_request(
    request: BuildRequest,
    manifest: IndexManifest,
) -> bool:
    """Check logical inputs without rebuilding the 1.8M-row evidence store."""
    facility_source = _source_metadata(request.facilities_path)
    review_source = _source_metadata(request.reviews_path)
    expected_facility_source = (
        manifest.facility_source_sha256,
        manifest.facility_source_bytes,
        manifest.facility_source_rows,
    )
    expected_review_source = (
        manifest.review_source_sha256,
        manifest.review_source_bytes,
        manifest.review_source_rows,
    )
    if (
        facility_source != expected_facility_source
        or review_source != expected_review_source
        or request.embedding_model != manifest.embedding_model
        or request.embedding_revision != manifest.embedding_revision
        or request.chroma_collection != manifest.source_collection
    ):
        return False
    facilities = _load_facilities(request.facilities_path)
    documents = tuple(
        facility_index_document(row)
        for row in facilities.to_dict(orient="records")
    )
    if (
        _digest_text_rows(tuple(item.place_id for item in documents))
        != manifest.id_map_digest
        or _digest_profile_hashes(documents) != manifest.profile_digest
    ):
        return False
    return _export_dense_vectors(request, documents, None) == manifest.source_vector_digest


def build_release(request: BuildRequest, directory: Path) -> BuildResult:
    """Build all payloads in an empty staging directory."""
    directory = Path(directory)
    if directory.exists():
        raise IndexBuildError(f"staging directory already exists: {directory}")
    directory.mkdir(parents=False, mode=0o700)

    facility_source = _source_metadata(request.facilities_path)
    review_source = _source_metadata(request.reviews_path)
    facilities = _load_facilities(request.facilities_path)
    logger.info("Building index release %s from %s facilities", request.version, len(facilities))
    facility_docs = tuple(
        facility_index_document(row)
        for row in facilities.to_dict(orient="records")
    )
    facility_ids = tuple(document.place_id for document in facility_docs)
    if tuple(sorted(facility_ids, key=lambda item: item.encode("utf-8"))) != facility_ids:
        raise IndexBuildError("internal facility ordering failure")

    id_map_digest = _digest_text_rows(facility_ids)
    profile_digest = _digest_profile_hashes(facility_docs)
    _write_facility_ordinals(
        directory / "facility_ordinals.arrow",
        facility_docs,
    )

    source_vector_digest = _export_dense_vectors(
        request,
        facility_docs,
        directory / "facility_vectors.npy",
    )
    logger.info("Exported and validated %s dense vectors", len(facility_docs))
    _build_facility_sqlite(
        directory / "facility_lexical.sqlite3",
        facility_docs,
    )
    logger.info("Built facility lexical indexes")
    evidence_stats = _build_evidence_sqlite(
        directory / "evidence_lexical.sqlite3",
        facilities,
        facility_ids,
        request.reviews_path,
    )
    logger.info(
        "Built evidence indexes: reviews=%s indexed_review_facilities=%s "
        "source_review_facilities=%s facility_evidence=%s",
        evidence_stats["indexed_reviews"],
        evidence_stats["review_facilities"],
        evidence_stats["source_review_facilities"],
        evidence_stats["facility_evidence"],
    )

    if facility_source[2] != len(facility_docs):
        raise IndexBuildError("facility Parquet row count changed while building")
    if review_source[2] != evidence_stats["review_source_rows"]:
        raise IndexBuildError("review Parquet row count changed while building")
    if _source_metadata(request.facilities_path) != facility_source:
        raise IndexBuildError("facility source changed during the index build")
    if _source_metadata(request.reviews_path) != review_source:
        raise IndexBuildError("review source changed during the index build")

    content_payload = {
        "formats": {
            "arrow": FACILITY_ARROW_SCHEMA,
            "facility_sqlite": FACILITY_SQLITE_SCHEMA,
            "evidence_sqlite": EVIDENCE_SQLITE_SCHEMA,
            "dense": DENSE_SCHEMA,
            "profile": FACILITY_PROFILE_SCHEMA,
            "normalization": NORMALIZATION_SCHEMA,
            "analyzer": ANALYZER_SCHEMA,
        },
        "sources": {
            "facilities": facility_source,
            "reviews": review_source,
        },
        "dense": {
            "model": request.embedding_model,
            "revision": request.embedding_revision,
            "dimension": EXPECTED_DENSE_DIMENSION,
            "collection": request.chroma_collection,
            "source_vector_digest": source_vector_digest,
        },
        "digests": {
            "id_map": id_map_digest,
            "profiles": profile_digest,
            "evidence": evidence_stats["evidence_digest"],
        },
        "counts": {
            "facilities": len(facility_docs),
            "indexed_reviews": evidence_stats["indexed_reviews"],
            "review_facilities": evidence_stats["review_facilities"],
            "orphan_reviews": evidence_stats["orphan_reviews"],
            "facility_evidence": evidence_stats["facility_evidence"],
            "evidence": evidence_stats["evidence_count"],
        },
    }
    content_id = sha256(_canonical_bytes(content_payload)).hexdigest()

    artifacts = tuple(
        ArtifactRecord(
            name=name,
            schema=ARTIFACT_SCHEMAS[name],
            byte_size=(directory / name).stat().st_size,
            sha256=sha256_file(directory / name),
        )
        for name in sorted(ARTIFACT_SCHEMAS)
    )
    manifest = IndexManifest(
        index_version=request.version,
        content_id=content_id,
        facility_source_sha256=facility_source[0],
        facility_source_bytes=facility_source[1],
        facility_source_rows=facility_source[2],
        review_source_sha256=review_source[0],
        review_source_bytes=review_source[1],
        review_source_rows=review_source[2],
        embedding_model=request.embedding_model,
        embedding_revision=request.embedding_revision,
        embedding_dimension=EXPECTED_DENSE_DIMENSION,
        source_collection=request.chroma_collection,
        source_vector_digest=source_vector_digest,
        facility_count=len(facility_docs),
        indexed_review_count=evidence_stats["indexed_reviews"],
        review_facility_count=evidence_stats["review_facilities"],
        orphan_review_count=evidence_stats["orphan_reviews"],
        facility_evidence_count=evidence_stats["facility_evidence"],
        evidence_count=evidence_stats["evidence_count"],
        id_map_digest=id_map_digest,
        profile_digest=profile_digest,
        evidence_digest=evidence_stats["evidence_digest"],
        artifacts=artifacts,
    )
    manifest_sha256 = write_manifest(directory / "manifest.json", manifest)
    _sync_and_seal(directory)
    return BuildResult(manifest=manifest, manifest_sha256=manifest_sha256)


def _source_metadata(path: Path) -> tuple[str, int, int]:
    path = Path(path)
    if not path.is_file():
        raise IndexBuildError(f"missing source Parquet: {path}")
    before = path.stat()
    try:
        rows = int(pq.ParquetFile(path).metadata.num_rows)
        digest = sha256_file(path)
    except (OSError, pa.ArrowException, IndexLoadError) as exc:
        raise IndexBuildError(f"cannot inspect source Parquet: {path}") from exc
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise IndexBuildError(f"source changed during inspection: {path}")
    if before.st_size <= 0:
        raise IndexBuildError(f"source Parquet is empty: {path}")
    return digest, int(before.st_size), rows


def _load_facilities(path: Path) -> pd.DataFrame:
    facilities = pd.read_parquet(path)
    if "place_id" not in facilities.columns:
        raise IndexBuildError("facility source requires place_id")
    facilities = facilities.copy()
    facilities["place_id"] = facilities["place_id"].fillna("").astype(str).str.strip()
    if facilities["place_id"].eq("").any():
        raise IndexBuildError("facility IDs cannot be empty")
    if facilities["place_id"].duplicated().any():
        raise IndexBuildError("facility IDs must be unique")
    return facilities.sort_values(
        "place_id",
        key=lambda values: values.map(lambda item: item.encode("utf-8")),
        kind="stable",
    ).reset_index(drop=True)


def _digest_text_rows(rows: Sequence[str]) -> str:
    digest = sha256()
    for value in rows:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _digest_profile_hashes(documents: Sequence[FacilityIndexDocument]) -> str:
    digest = sha256()
    for document in documents:
        digest.update(bytes.fromhex(document.profile_sha256))
    return digest.hexdigest()


def _write_facility_ordinals(
    path: Path,
    documents: Sequence[FacilityIndexDocument],
) -> None:
    schema = pa.schema([
        pa.field("ordinal", pa.uint32(), nullable=False),
        pa.field("place_id", pa.string(), nullable=False),
        pa.field("profile_sha256", pa.binary(32), nullable=False),
    ])
    table = pa.Table.from_arrays(
        [
            pa.array(range(len(documents)), type=pa.uint32()),
            pa.array((item.place_id for item in documents), type=pa.string()),
            pa.array(
                (bytes.fromhex(item.profile_sha256) for item in documents),
                type=pa.binary(32),
            ),
        ],
        schema=schema,
    )
    with pa.OSFile(str(path), "wb") as sink:
        with pa_ipc.new_file(sink, schema) as writer:
            writer.write_table(table)


def _export_dense_vectors(
    request: BuildRequest,
    documents: Sequence[FacilityIndexDocument],
    path: Path | None,
) -> str:
    try:
        client = chromadb.PersistentClient(
            path=str(request.chroma_path),
            settings=Settings(anonymized_telemetry=False),
        )
        collection = client.get_collection(request.chroma_collection)
        actual_ids = collection.get(include=[])["ids"]
    except Exception as exc:
        raise IndexBuildError("cannot open the source Chroma collection") from exc
    expected_ids = [document.place_id for document in documents]
    if len(actual_ids) != len(expected_ids) or set(actual_ids) != set(expected_ids):
        missing = len(set(expected_ids) - set(actual_ids))
        extra = len(set(actual_ids) - set(expected_ids))
        raise IndexBuildError(
            f"Chroma facility coverage mismatch: missing={missing}, extra={extra}"
        )

    vectors = (
        np.lib.format.open_memmap(
            path,
            mode="w+",
            dtype="<f4",
            shape=(len(documents), EXPECTED_DENSE_DIMENSION),
        )
        if path is not None else None
    )
    source_digest = sha256()
    try:
        for start in range(0, len(documents), 256):
            chunk = documents[start:start + 256]
            chunk_ids = [document.place_id for document in chunk]
            result = collection.get(
                ids=chunk_ids,
                include=["documents", "embeddings"],
            )
            returned_ids = result["ids"]
            returned_documents = result["documents"] or []
            returned_embeddings = result["embeddings"]
            if returned_embeddings is None:
                raise IndexBuildError("Chroma did not return embeddings")
            by_id = {
                identifier: (document, embedding)
                for identifier, document, embedding in zip(
                    returned_ids,
                    returned_documents,
                    returned_embeddings,
                    strict=True,
                )
            }
            if set(by_id) != set(chunk_ids):
                raise IndexBuildError("Chroma batch coverage mismatch")
            for offset, expected in enumerate(chunk):
                stored_document, embedding = by_id[expected.place_id]
                if stored_document != expected.profile_text:
                    raise IndexBuildError(
                        f"Chroma profile mismatch for {expected.place_id}"
                    )
                vector = np.asarray(embedding, dtype="<f4")
                if vector.shape != (EXPECTED_DENSE_DIMENSION,) or not np.isfinite(vector).all():
                    raise IndexBuildError(
                        f"invalid Chroma vector for {expected.place_id}"
                    )
                source_digest.update(vector.tobytes(order="C"))
                norm = float(np.linalg.norm(vector))
                if not math.isfinite(norm) or norm <= 0.0:
                    raise IndexBuildError(
                        f"zero or invalid Chroma vector for {expected.place_id}"
                    )
                if vectors is not None:
                    vectors[start + offset] = vector / norm
        if vectors is not None:
            vectors.flush()
    finally:
        mmap = getattr(vectors, "_mmap", None) if vectors is not None else None
        if mmap is not None:
            mmap.close()
    return source_digest.hexdigest()


def _configure_sqlite(connection: sqlite3.Connection, application_id: int) -> None:
    connection.executescript(f"""
        PRAGMA page_size = 4096;
        PRAGMA journal_mode = OFF;
        PRAGMA synchronous = OFF;
        PRAGMA temp_store = MEMORY;
        PRAGMA locking_mode = EXCLUSIVE;
        PRAGMA foreign_keys = ON;
        PRAGMA application_id = {application_id};
        PRAGMA user_version = {SQLITE_USER_VERSION};
    """)


def _build_facility_sqlite(
    path: Path,
    documents: Sequence[FacilityIndexDocument],
) -> None:
    connection = sqlite3.connect(path)
    try:
        _configure_sqlite(connection, FACILITY_SQLITE_APPLICATION_ID)
        connection.executescript("""
            CREATE TABLE facility_reference (
                ordinal INTEGER PRIMARY KEY,
                place_id TEXT NOT NULL UNIQUE,
                profile_sha256 BLOB NOT NULL CHECK(length(profile_sha256) = 32)
            ) STRICT;
            CREATE VIRTUAL TABLE facility_terms USING fts5(
                specialty,
                name,
                district,
                neighborhood,
                exact_phrase,
                english_terms,
                korean_terms,
                korean_ngrams,
                trusted_facts,
                content='',
                tokenize='unicode61 remove_diacritics 2'
            );
            CREATE VIRTUAL TABLE facility_substrings USING fts5(
                text,
                content='',
                tokenize='trigram case_sensitive 0 remove_diacritics 1'
            );
        """)
        for start in range(0, len(documents), SQLITE_BATCH_SIZE):
            chunk = documents[start:start + SQLITE_BATCH_SIZE]
            connection.executemany(
                "INSERT INTO facility_reference VALUES (?, ?, ?)",
                (
                    (start + offset, item.place_id, bytes.fromhex(item.profile_sha256))
                    for offset, item in enumerate(chunk)
                ),
            )
            connection.executemany(
                "INSERT INTO facility_terms(rowid, specialty, name, district, "
                "neighborhood, exact_phrase, english_terms, korean_terms, "
                "korean_ngrams, trusted_facts) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    (
                        start + offset,
                        item.specialty,
                        item.name,
                        item.district,
                        item.neighborhood,
                        item.exact_phrase,
                        item.english_terms,
                        item.korean_terms,
                        item.korean_ngrams,
                        item.trusted_facts,
                    )
                    for offset, item in enumerate(chunk)
                ),
            )
            connection.executemany(
                "INSERT INTO facility_substrings(rowid, text) VALUES (?, ?)",
                (
                    (start + offset, item.substring_text)
                    for offset, item in enumerate(chunk)
                ),
            )
        connection.execute(
            "INSERT INTO facility_terms(facility_terms) VALUES('optimize')"
        )
        connection.execute(
            "INSERT INTO facility_substrings(facility_substrings) VALUES('optimize')"
        )
        connection.commit()
        _finish_sqlite(connection)
    except sqlite3.DatabaseError as exc:
        raise IndexBuildError("facility SQLite build failed") from exc
    finally:
        connection.close()


def _build_evidence_sqlite(
    path: Path,
    facilities: pd.DataFrame,
    facility_ids: Sequence[str],
    reviews_path: Path,
) -> dict[str, Any]:
    id_to_ordinal = {place_id: ordinal for ordinal, place_id in enumerate(facility_ids)}
    connection = sqlite3.connect(path)
    digest = _EvidenceDigest()
    indexed_reviews = 0
    review_facilities: set[int] = set()
    source_review_facilities: set[int] = set()
    orphan_reviews = 0
    facility_evidence = 0
    evidence_ordinal = 0
    try:
        _configure_sqlite(connection, EVIDENCE_SQLITE_APPLICATION_ID)
        connection.executescript("""
            CREATE TABLE facility_reference (
                ordinal INTEGER PRIMARY KEY,
                place_id TEXT NOT NULL UNIQUE
            ) STRICT;
            CREATE TABLE evidence_document (
                ordinal INTEGER PRIMARY KEY,
                evidence_id TEXT NOT NULL UNIQUE,
                facility_ordinal INTEGER NOT NULL REFERENCES facility_reference(ordinal),
                source_type TEXT NOT NULL,
                source_field TEXT NOT NULL,
                source_index INTEGER NOT NULL,
                source_locator TEXT NOT NULL,
                original_text TEXT NOT NULL
                    CHECK(length(CAST(original_text AS BLOB)) > 0),
                language_hint TEXT NOT NULL,
                visit_date TEXT NOT NULL,
                scraped_at TEXT NOT NULL,
                is_verbatim INTEGER NOT NULL CHECK(is_verbatim IN (0, 1))
            ) STRICT;
            CREATE INDEX evidence_by_facility
                ON evidence_document(facility_ordinal, ordinal);
            CREATE VIRTUAL TABLE evidence_terms USING fts5(
                original_text,
                korean_ngrams,
                content='',
                tokenize='unicode61 remove_diacritics 2'
            );
            CREATE VIRTUAL TABLE evidence_substrings USING fts5(
                text,
                content='',
                tokenize='trigram case_sensitive 0 remove_diacritics 1'
            );
        """)
        connection.executemany(
            "INSERT INTO facility_reference VALUES (?, ?)",
            enumerate(facility_ids),
        )

        pending: list[_EvidenceDocument] = []

        def flush() -> None:
            nonlocal evidence_ordinal
            if not pending:
                return
            first = evidence_ordinal
            connection.executemany(
                "INSERT INTO evidence_document VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    (
                        first + offset,
                        item.evidence_id,
                        item.facility_ordinal,
                        item.source_type,
                        item.source_field,
                        item.source_index,
                        item.source_locator,
                        item.original_text,
                        item.language_hint,
                        item.visit_date,
                        item.scraped_at,
                        int(item.is_verbatim),
                    )
                    for offset, item in enumerate(pending)
                ),
            )
            connection.executemany(
                "INSERT INTO evidence_terms(rowid, original_text, korean_ngrams) VALUES (?, ?, ?)",
                (
                    (
                        first + offset,
                        item.original_text.replace("\x00", " "),
                        hangul_ngrams(item.original_text.replace("\x00", " ")),
                    )
                    for offset, item in enumerate(pending)
                ),
            )
            connection.executemany(
                "INSERT INTO evidence_substrings(rowid, text) VALUES (?, ?)",
                (
                    (
                        first + offset,
                        normalize_search_text(item.original_text),
                    )
                    for offset, item in enumerate(pending)
                ),
            )
            for item in pending:
                digest.add(item)
            evidence_ordinal += len(pending)
            pending.clear()
            if evidence_ordinal % 50_000 == 0:
                connection.commit()

        parquet = pq.ParquetFile(reviews_path)
        names = set(parquet.schema_arrow.names)
        required = {"place_id", "review_index", "review_text"}
        if not required.issubset(names):
            raise IndexBuildError(
                f"review source missing columns: {sorted(required - names)}"
            )
        optional = [name for name in ("visit_date", "scraped_at") if name in names]
        review_source_rows = 0
        next_progress = 250_000
        for batch in parquet.iter_batches(
            columns=["place_id", "review_index", "review_text", *optional],
            batch_size=PARQUET_BATCH_SIZE,
        ):
            columns = batch.to_pydict()
            for row_number in range(batch.num_rows):
                review_source_rows += 1
                place_id = str(columns["place_id"][row_number] or "").strip()
                facility_ordinal = id_to_ordinal.get(place_id)
                if facility_ordinal is None:
                    orphan_reviews += 1
                    continue
                source_review_facilities.add(facility_ordinal)
                text = str(columns["review_text"][row_number] or "").strip()
                if not text:
                    continue
                raw_index = columns["review_index"][row_number]
                if raw_index is None:
                    raise IndexBuildError("nonempty review has no review_index")
                source_index = int(raw_index)
                record = _EvidenceDocument(
                    evidence_id=raw_review_evidence_id(place_id, source_index, text),
                    facility_ordinal=facility_ordinal,
                    source_type="verbatim_review",
                    source_field="review_text",
                    source_index=source_index,
                    source_locator=f"review_snapshot:{place_id}:{source_index}",
                    original_text=text,
                    language_hint=detect_language_hint(text),
                    visit_date=_stable_text(
                        columns["visit_date"][row_number]
                        if "visit_date" in columns else ""
                    ),
                    scraped_at=_stable_text(
                        columns["scraped_at"][row_number]
                        if "scraped_at" in columns else ""
                    ),
                    is_verbatim=True,
                )
                pending.append(record)
                indexed_reviews += 1
                review_facilities.add(facility_ordinal)
                if len(pending) >= SQLITE_BATCH_SIZE:
                    flush()
            if review_source_rows >= next_progress:
                logger.info(
                    "Indexed review source rows: %s/%s",
                    review_source_rows,
                    parquet.metadata.num_rows,
                )
                while next_progress <= review_source_rows:
                    next_progress += 250_000

        derived = build_specific_evidence_records(facilities)
        derived.sort(
            key=lambda item: (
                str(item.get("place_id", "")).encode("utf-8"),
                str(item.get("evidence_id", "")).encode("utf-8"),
            )
        )
        for item in derived:
            place_id = str(item.get("place_id") or "").strip()
            facility_ordinal = id_to_ordinal.get(place_id)
            if facility_ordinal is None:
                raise IndexBuildError("facility-derived evidence has no facility")
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            pending.append(_EvidenceDocument(
                evidence_id=str(item.get("evidence_id") or ""),
                facility_ordinal=facility_ordinal,
                source_type=str(item.get("source_type") or "facility_fact"),
                source_field=str(item.get("source_field") or ""),
                source_index=int(item.get("source_index") or 0),
                source_locator=(
                    f"facility_snapshot:{place_id}:"
                    f"{str(item.get('source_field') or '')}:"
                    f"{int(item.get('source_index') or 0)}"
                ),
                original_text=text,
                language_hint=str(item.get("language") or detect_language_hint(text)),
                visit_date="",
                scraped_at="",
                is_verbatim=False,
            ))
            facility_evidence += 1
            if len(pending) >= SQLITE_BATCH_SIZE:
                flush()
        flush()

        if orphan_reviews:
            raise IndexBuildError(
                f"review source contains {orphan_reviews} orphan facility IDs"
            )
        connection.execute(
            "INSERT INTO evidence_terms(evidence_terms) VALUES('optimize')"
        )
        connection.execute(
            "INSERT INTO evidence_substrings(evidence_substrings) VALUES('optimize')"
        )
        connection.commit()
        _finish_sqlite(connection)
        return {
            "review_source_rows": review_source_rows,
            "indexed_reviews": indexed_reviews,
            "review_facilities": len(review_facilities),
            "source_review_facilities": len(source_review_facilities),
            "orphan_reviews": orphan_reviews,
            "facility_evidence": facility_evidence,
            "evidence_count": evidence_ordinal,
            "evidence_digest": digest.hexdigest(),
        }
    except sqlite3.IntegrityError as exc:
        raise IndexBuildError(
            "duplicate or invalid evidence identity in source snapshots"
        ) from exc
    except sqlite3.DatabaseError as exc:
        raise IndexBuildError("evidence SQLite build failed") from exc
    finally:
        connection.close()


def _finish_sqlite(connection: sqlite3.Connection) -> None:
    if connection.execute("PRAGMA foreign_key_check").fetchall():
        raise IndexBuildError("SQLite artifact contains orphan records")
    if connection.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
        raise IndexBuildError("SQLite artifact failed integrity check")
    connection.execute("VACUUM")
    connection.commit()


def _stable_text(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "isoformat"):
        return str(value.isoformat())
    return str(value)


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sync_and_seal(directory: Path) -> None:
    for path in sorted(directory.iterdir()):
        descriptor = path.open("rb")
        try:
            descriptor.flush()
            os.fsync(descriptor.fileno())
        finally:
            descriptor.close()
        path.chmod(0o444)
    fsync_directory(directory)
    directory.chmod(0o555)
    fsync_directory(directory.parent)
