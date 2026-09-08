"""Publication lifecycle and scope-bound access to immutable index releases."""

from __future__ import annotations

from dataclasses import dataclass
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import sqlite3
import stat
from typing import Iterable, Sequence
from urllib.parse import quote
from uuid import uuid4

import numpy as np

from search.contracts import validate_evidence_source_types
from search.scope import ScopeSelection

from .build import (
    BuildRequest,
    IndexBuildError,
    build_release,
    release_matches_request,
)
from .documents import fts_query_terms, normalize_search_text
from .manifest import (
    ActivePointer,
    EXPECTED_DENSE_DIMENSION,
    IndexLoadError,
    IndexManifest,
    ValidatedRelease,
    fsync_directory,
    read_active_pointer,
    validate_release,
    validate_version,
    write_active_pointer,
)


class IndexCollisionError(IndexBuildError):
    """A release name is already bound to different logical content."""


@dataclass(frozen=True)
class PublishedIndex:
    version: str
    content_id: str
    manifest_sha256: str
    directory: Path
    reused: bool = False


@dataclass(frozen=True)
class FacilityHit:
    facility_id: str
    ordinal: int
    score: float
    channel: str


@dataclass(frozen=True)
class EvidenceHit:
    evidence_id: str
    facility_id: str
    ordinal: int
    score: float
    channel: str
    source_type: str
    source_field: str
    source_index: int
    source_locator: str
    original_text: str
    language_hint: str
    visit_date: str
    scraped_at: str
    is_verbatim: bool


class IndexRepository:
    """Own staging, publication, activation, rollback, and opening."""

    def __init__(self, root: Path):
        self.root = Path(root).absolute()

    @property
    def versions_directory(self) -> Path:
        return self.root / "versions"

    def publish(self, request: BuildRequest) -> PublishedIndex:
        version = validate_version(request.version)
        self._prepare_root()
        final = self.versions_directory / version
        with self._publication_lock():
            if os.path.lexists(final):
                existing = validate_release(final, expected_version=version)
                try:
                    if release_matches_request(request, existing.manifest):
                        return _published(existing, reused=True)
                    raise IndexCollisionError(
                        f"index version already contains different content: {version}"
                    )
                finally:
                    existing.close()
        stage = self.versions_directory / f".staging-{version}-{uuid4().hex}"
        try:
            result = build_release(request, stage)
            validated = validate_release(stage, expected_version=version)
            validated.close()
            with self._publication_lock():
                if os.path.lexists(final):
                    existing = validate_release(final, expected_version=version)
                    try:
                        if existing.manifest.content_id != result.manifest.content_id:
                            raise IndexCollisionError(
                                f"index version already contains different content: {version}"
                            )
                        published = _published(existing, reused=True)
                    finally:
                        existing.close()
                    _remove_stage(stage)
                    return published
                os.rename(stage, final)
                fsync_directory(self.versions_directory)
                fsync_directory(self.root)
                return PublishedIndex(
                    version=version,
                    content_id=result.manifest.content_id,
                    manifest_sha256=result.manifest_sha256,
                    directory=final,
                    reused=False,
                )
        except BaseException:
            if os.path.lexists(stage):
                _remove_stage(stage)
            raise

    def activate(self, version: str) -> PublishedIndex:
        version = validate_version(version)
        self._prepare_root()
        with self._publication_lock():
            release = validate_release(
                self.versions_directory / version,
                expected_version=version,
            )
            try:
                pointer = ActivePointer(
                    version=version,
                    manifest_sha256=release.manifest_sha256,
                    content_id=release.manifest.content_id,
                )
                write_active_pointer(self.root, pointer)
                return _published(release)
            finally:
                release.close()

    def open_active(self) -> "ReadonlyIndex":
        self._validate_existing_root()
        pointer = read_active_pointer(self.root)
        release = validate_release(
            self.versions_directory / pointer.version,
            expected_version=pointer.version,
        )
        if (
            release.manifest_sha256 != pointer.manifest_sha256
            or release.manifest.content_id != pointer.content_id
        ):
            release.close()
            raise IndexLoadError("active pointer does not match its release")
        return ReadonlyIndex(release)

    def open_version(self, version: str) -> "ReadonlyIndex":
        version = validate_version(version)
        self._validate_existing_root()
        return ReadonlyIndex(validate_release(
            self.versions_directory / version,
            expected_version=version,
        ))

    def _prepare_root(self) -> None:
        _reject_existing_symlink_components(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        _require_real_directory(self.root)
        self.versions_directory.mkdir(exist_ok=True)
        _require_real_directory(self.versions_directory)

    def _validate_existing_root(self) -> None:
        _require_real_directory(self.root)
        _require_real_directory(self.versions_directory)

    def _publication_lock(self):
        return _FileLock(self.root / ".publish.lock")


class ReadonlyIndex:
    """A fully validated release with no unscoped search methods."""

    def __init__(self, release: ValidatedRelease):
        self._release = release
        self.manifest: IndexManifest = release.manifest
        self.version = release.manifest.index_version
        self.facility_ids = release.facility_ids
        self._id_to_ordinal = {
            facility_id: ordinal
            for ordinal, facility_id in enumerate(release.facility_ids)
        }
        self._closed = False

    def close(self) -> None:
        if not self._closed:
            self._release.close()
            self._closed = True

    def __enter__(self) -> "ReadonlyIndex":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def within(self, scope: ScopeSelection) -> "ScopedIndex":
        self._assert_open()
        descriptor = scope.descriptor
        if descriptor.index_version != self.version:
            raise IndexLoadError("scope and index versions differ")
        if descriptor.facility_count != len(scope.facility_ids):
            raise IndexLoadError("scope facility count mismatch")
        expected_digest = _scope_digest(
            self.version,
            descriptor.rules_hash,
            scope.facility_ids,
        )
        if descriptor.scope_digest != expected_digest:
            raise IndexLoadError("scope digest mismatch")
        if descriptor.facility_bitmap_ref != f"memory://{expected_digest}":
            raise IndexLoadError("scope bitmap reference mismatch")
        if len(set(scope.facility_ids)) != len(scope.facility_ids):
            raise IndexLoadError("scope facility IDs are duplicated")
        try:
            ordinals = np.fromiter(
                (self._id_to_ordinal[item] for item in scope.facility_ids),
                dtype=np.int64,
                count=len(scope.facility_ids),
            )
        except KeyError as exc:
            raise IndexLoadError(
                f"scope contains a facility absent from the index: {exc.args[0]}"
            ) from exc
        return ScopedIndex(self, ordinals)

    def _assert_open(self) -> None:
        if self._closed:
            raise IndexLoadError("index release is closed")


class ScopedIndex:
    """Search channels that cannot rank outside one complete eligible scope."""

    def __init__(self, index: ReadonlyIndex, ordinals: np.ndarray):
        index._assert_open()
        self._index = index
        self._ordinals = np.asarray(ordinals, dtype=np.int64)
        self._facility_connection = _scoped_connection(
            index._release.directory / "facility_lexical.sqlite3",
            self._ordinals,
        )
        self._evidence_connection = _scoped_connection(
            index._release.directory / "evidence_lexical.sqlite3",
            self._ordinals,
        )
        self._closed = False

    def close(self) -> None:
        if not self._closed:
            self._facility_connection.close()
            self._evidence_connection.close()
            self._closed = True

    def __enter__(self) -> "ScopedIndex":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def search_facilities(self, query: str, limit: int = 200) -> list[FacilityHit]:
        self._assert_open()
        limit = _bounded_limit(limit, maximum=2_000)
        if not query.strip() or not len(self._ordinals):
            return []
        word_query = _word_match(query)
        word_rows = _facility_word_rows(self._facility_connection, word_query, limit)
        substring_rows = _facility_substring_rows(
            self._facility_connection,
            _substring_match(query),
            limit,
        )
        ranked = _reciprocal_rank((word_rows, substring_rows), limit)
        return [
            FacilityHit(
                facility_id=self._index.facility_ids[ordinal],
                ordinal=ordinal,
                score=score,
                channel="lexical",
            )
            for ordinal, score in ranked
        ]

    def search_dense(
        self,
        query_embedding: Sequence[float],
        limit: int = 200,
    ) -> list[FacilityHit]:
        self._assert_open()
        limit = min(_bounded_limit(limit, maximum=2_000), len(self._ordinals))
        if not len(self._ordinals):
            return []
        query = np.asarray(query_embedding, dtype=np.float32).copy()
        if query.shape != (EXPECTED_DENSE_DIMENSION,) or not np.isfinite(query).all():
            raise ValueError(
                f"query embedding must contain {EXPECTED_DENSE_DIMENSION} finite values"
            )
        norm = float(np.linalg.norm(query))
        if norm <= 0.0:
            raise ValueError("query embedding cannot be zero")
        query /= norm
        matrix = np.asarray(self._index._release.vectors[self._ordinals])
        scores = matrix @ query
        if limit == len(scores):
            selected = np.arange(len(scores))
        else:
            selected = np.argpartition(scores, -limit)[-limit:]
        selected = sorted(
            selected.tolist(),
            key=lambda position: (-float(scores[position]), int(self._ordinals[position])),
        )
        return [
            FacilityHit(
                facility_id=self._index.facility_ids[int(self._ordinals[position])],
                ordinal=int(self._ordinals[position]),
                score=float(scores[position]),
                channel="dense",
            )
            for position in selected
        ]

    def search_evidence(
        self,
        query: str,
        limit: int = 100,
        *,
        source_types: Iterable[str] | None = None,
    ) -> list[EvidenceHit]:
        self._assert_open()
        limit = _bounded_limit(limit, maximum=1_000)
        accepted_types = validate_evidence_source_types(source_types or ())
        if not query.strip() or not len(self._ordinals):
            return []
        word_rows = _evidence_rows(
            self._evidence_connection,
            table="evidence_terms",
            match=_word_match(query),
            limit=limit,
            source_types=accepted_types,
        )
        substring_rows = _evidence_rows(
            self._evidence_connection,
            table="evidence_substrings",
            match=_substring_match(query),
            limit=limit,
            source_types=accepted_types,
        )
        ranked = _reciprocal_rank((word_rows, substring_rows), limit)
        row_by_ordinal = {
            int(row[0]): row
            for row in (*word_rows, *substring_rows)
        }
        hits: list[EvidenceHit] = []
        for ordinal, score in ranked:
            row = row_by_ordinal[ordinal]
            hits.append(EvidenceHit(
                evidence_id=str(row[1]),
                facility_id=self._index.facility_ids[int(row[2])],
                ordinal=ordinal,
                score=score,
                channel="evidence_lexical",
                source_type=str(row[3]),
                source_field=str(row[4]),
                source_index=int(row[5]),
                source_locator=str(row[6]),
                original_text=str(row[7]),
                language_hint=str(row[8]),
                visit_date=str(row[9]),
                scraped_at=str(row[10]),
                is_verbatim=bool(row[11]),
            ))
        return hits

    def search_evidence_for_facilities(
        self,
        query: str,
        *,
        facility_ids: Sequence[str],
        limit_per_facility: int,
        source_types: Sequence[str] = (),
    ) -> list[EvidenceHit]:
        """Return a deterministic lexical quota for each requested facility."""
        self._assert_open()
        limit = _bounded_limit(limit_per_facility, maximum=100)
        accepted_types = validate_evidence_source_types(source_types)
        requested = tuple(str(value).strip() for value in facility_ids)
        if not requested:
            return []
        if len(requested) > 50:
            raise ValueError("facility shortlist cannot contain more than 50 IDs")
        if any(not value for value in requested):
            raise ValueError("facility shortlist contains an empty ID")
        if len(set(requested)) != len(requested):
            raise ValueError("facility shortlist IDs are duplicated")

        scoped_ordinals = frozenset(int(value) for value in self._ordinals)
        ordinals: list[int] = []
        for facility_id in requested:
            ordinal = self._index._id_to_ordinal.get(facility_id)
            if ordinal is None or ordinal not in scoped_ordinals:
                raise ValueError("facility shortlist contains an ID outside scope")
            ordinals.append(ordinal)

        if not query.strip():
            return []

        hits: list[EvidenceHit] = []
        for facility_ordinal in ordinals:
            word_rows = _evidence_rows_for_facility(
                self._evidence_connection,
                table="evidence_terms",
                match=_word_match(query),
                facility_ordinal=facility_ordinal,
                limit=limit,
                source_types=accepted_types,
            )
            substring_rows = _evidence_rows_for_facility(
                self._evidence_connection,
                table="evidence_substrings",
                match=_substring_match(query),
                facility_ordinal=facility_ordinal,
                limit=limit,
                source_types=accepted_types,
            )
            ranked = _reciprocal_rank((word_rows, substring_rows), limit)
            row_by_ordinal = {
                int(row[0]): row for row in (*word_rows, *substring_rows)
            }
            for evidence_ordinal, score in ranked:
                row = row_by_ordinal[evidence_ordinal]
                hits.append(EvidenceHit(
                    evidence_id=str(row[1]),
                    facility_id=self._index.facility_ids[int(row[2])],
                    ordinal=evidence_ordinal,
                    score=score,
                    channel="facility_evidence_lexical",
                    source_type=str(row[3]),
                    source_field=str(row[4]),
                    source_index=int(row[5]),
                    source_locator=str(row[6]),
                    original_text=str(row[7]),
                    language_hint=str(row[8]),
                    visit_date=str(row[9]),
                    scraped_at=str(row[10]),
                    is_verbatim=bool(row[11]),
                ))
        return hits

    @property
    def review_source_sha256(self) -> str:
        """Return the review snapshot identity bound to this scoped release."""
        self._assert_open()
        return self._index.manifest.review_source_sha256

    def list_original_reviews(
        self, *, facility_ids: Sequence[str], limit_per_facility: int = 100,
    ) -> list[EvidenceHit]:
        """Read a bounded, source-ordered sample without a relevance claim."""
        self._assert_open()
        limit = _bounded_limit(limit_per_facility, maximum=100)
        requested = tuple(dict.fromkeys(facility_ids))
        if len(requested) > 50:
            raise ValueError("facility shortlist cannot contain more than 50 IDs")
        allowed = frozenset(int(value) for value in self._ordinals)
        ordinals = [self._index._id_to_ordinal.get(identity) for identity in requested]
        if any(ordinal not in allowed for ordinal in ordinals):
            raise ValueError("facility shortlist contains an ID outside scope")
        identities = []
        for ordinal in ordinals:
            rows = self._evidence_connection.execute(
                "SELECT evidence_id FROM evidence_document "
                "WHERE facility_ordinal = ? AND source_type = 'verbatim_review' "
                "AND is_verbatim = 1 ORDER BY ordinal LIMIT ?",
                (ordinal, limit),
            ).fetchall()
            identities.extend(row[0] for row in rows)
        return self.resolve_evidence_ids(identities)

    def resolve_evidence_ids(
        self,
        evidence_ids: Sequence[str],
    ) -> list[EvidenceHit]:
        """Resolve untrusted evidence references inside the eligible scope."""
        self._assert_open()
        normalized = tuple(str(value).strip() for value in evidence_ids)
        if any(not evidence_id for evidence_id in normalized):
            raise ValueError("evidence IDs cannot be empty")
        requested = tuple(dict.fromkeys(normalized))
        if not requested:
            return []
        if len(requested) > 10_000:
            raise ValueError("cannot resolve more than 10000 evidence IDs")

        rows_by_id: dict[str, tuple[object, ...]] = {}
        for start in range(0, len(requested), 400):
            chunk = requested[start:start + 400]
            placeholders = ",".join("?" for _ in chunk)
            rows = self._evidence_connection.execute(
                f"""
                SELECT document.ordinal,
                       document.evidence_id,
                       document.facility_ordinal,
                       document.source_type,
                       document.source_field,
                       document.source_index,
                       document.source_locator,
                       document.original_text,
                       document.language_hint,
                       document.visit_date,
                       document.scraped_at,
                       document.is_verbatim
                FROM evidence_document AS document
                JOIN eligible ON eligible.ordinal = document.facility_ordinal
                WHERE document.evidence_id IN ({placeholders})
                """,
                chunk,
            ).fetchall()
            rows_by_id.update((str(row[1]), row) for row in rows)

        if len(rows_by_id) != len(requested):
            raise ValueError(
                "every evidence ID must resolve inside the eligible scope"
            )
        if any(
            str(rows_by_id[evidence_id][3]) != "verbatim_review"
            or not bool(rows_by_id[evidence_id][11])
            for evidence_id in requested
        ):
            raise ValueError(
                "every evidence ID must resolve to a verbatim review"
            )

        hits: list[EvidenceHit] = []
        for evidence_id in requested:
            row = rows_by_id[evidence_id]
            hits.append(EvidenceHit(
                evidence_id=str(row[1]),
                facility_id=self._index.facility_ids[int(row[2])],
                ordinal=int(row[0]),
                score=0.0,
                channel="resolved_evidence",
                source_type=str(row[3]),
                source_field=str(row[4]),
                source_index=int(row[5]),
                source_locator=str(row[6]),
                original_text=str(row[7]),
                language_hint=str(row[8]),
                visit_date=str(row[9]),
                scraped_at=str(row[10]),
                is_verbatim=bool(row[11]),
            ))
        return hits

    def _assert_open(self) -> None:
        self._index._assert_open()
        if self._closed:
            raise IndexLoadError("scoped index is closed")


class _FileLock:
    def __init__(self, path: Path):
        self.path = path
        self.descriptor: int | None = None

    def __enter__(self) -> "_FileLock":
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        self.descriptor = os.open(self.path, flags, 0o600)
        fcntl.flock(self.descriptor, fcntl.LOCK_EX)
        return self

    def __exit__(self, *_: object) -> None:
        assert self.descriptor is not None
        fcntl.flock(self.descriptor, fcntl.LOCK_UN)
        os.close(self.descriptor)
        self.descriptor = None


def _published(release: ValidatedRelease, *, reused: bool = False) -> PublishedIndex:
    return PublishedIndex(
        version=release.manifest.index_version,
        content_id=release.manifest.content_id,
        manifest_sha256=release.manifest_sha256,
        directory=release.directory,
        reused=reused,
    )


def _require_real_directory(path: Path) -> None:
    _reject_symlink_components(path)
    file_stat = path.lstat()
    if not stat.S_ISDIR(file_stat.st_mode) or stat.S_ISLNK(file_stat.st_mode):
        raise IndexLoadError(f"index repository path is not a real directory: {path}")


def _reject_symlink_components(path: Path) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        try:
            file_stat = current.lstat()
        except FileNotFoundError as exc:
            raise IndexLoadError(f"missing index repository path: {current}") from exc
        if stat.S_ISLNK(file_stat.st_mode):
            raise IndexLoadError(
                f"index repository path cannot contain a symlink: {current}"
            )


def _reject_existing_symlink_components(path: Path) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        try:
            file_stat = current.lstat()
        except FileNotFoundError:
            return
        if stat.S_ISLNK(file_stat.st_mode):
            raise IndexLoadError(
                f"index repository path cannot contain a symlink: {current}"
            )


def _remove_stage(path: Path) -> None:
    if not path.name.startswith(".staging-"):
        raise IndexBuildError("refusing to remove a non-staging directory")
    for root, directories, files in os.walk(path, topdown=False, followlinks=False):
        current = Path(root)
        current.chmod(0o700)
        for name in files:
            (current / name).unlink()
        for name in directories:
            child = current / name
            if child.is_symlink():
                child.unlink()
            else:
                child.chmod(0o700)
                child.rmdir()
    path.rmdir()


def _scope_digest(version: str, rules_hash: str, facility_ids: Sequence[str]) -> str:
    payload = json.dumps(
        {
            "index_version": version,
            "rules_hash": rules_hash,
            "facility_ids": sorted(facility_ids),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def _scoped_connection(path: Path, ordinals: np.ndarray) -> sqlite3.Connection:
    uri = f"file:{quote(str(path))}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    try:
        connection.execute("CREATE TEMP TABLE eligible(ordinal INTEGER PRIMARY KEY) WITHOUT ROWID")
        connection.executemany(
            "INSERT INTO eligible VALUES (?)",
            ((int(value),) for value in ordinals),
        )
        connection.execute("PRAGMA query_only = ON")
        return connection
    except Exception:
        connection.close()
        raise


def _bounded_limit(value: int, *, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("search limit must be an integer") from exc
    if parsed < 1:
        raise ValueError("search limit must be positive")
    return min(parsed, maximum)


def _word_match(query: str) -> str:
    terms = fts_query_terms(query)[:32]
    return " OR ".join(f'"{term}"' for term in terms)


def _substring_match(query: str) -> str:
    normalized = normalize_search_text(query)
    if len(normalized) < 3:
        return ""
    return f'"{normalized}"'


def _facility_word_rows(
    connection: sqlite3.Connection,
    match: str,
    limit: int,
) -> list[tuple[int, float]]:
    if not match:
        return []
    rows = connection.execute(
        """
        SELECT facility_terms.rowid,
               bm25(facility_terms, 7.0, 8.0, 5.0, 5.0, 4.0, 2.0, 2.0, 1.5, 1.0)
        FROM facility_terms
        JOIN eligible ON eligible.ordinal = facility_terms.rowid
        WHERE facility_terms MATCH ?
        ORDER BY 2 ASC, facility_terms.rowid ASC
        LIMIT ?
        """,
        (match, limit),
    ).fetchall()
    return [(int(row[0]), float(row[1])) for row in rows]


def _facility_substring_rows(
    connection: sqlite3.Connection,
    match: str,
    limit: int,
) -> list[tuple[int, float]]:
    if not match:
        return []
    rows = connection.execute(
        """
        SELECT facility_substrings.rowid, bm25(facility_substrings)
        FROM facility_substrings
        JOIN eligible ON eligible.ordinal = facility_substrings.rowid
        WHERE facility_substrings MATCH ?
        ORDER BY 2 ASC, facility_substrings.rowid ASC
        LIMIT ?
        """,
        (match, limit),
    ).fetchall()
    return [(int(row[0]), float(row[1])) for row in rows]


def _evidence_rows(
    connection: sqlite3.Connection,
    *,
    table: str,
    match: str,
    limit: int,
    source_types: Sequence[str],
) -> list[tuple[object, ...]]:
    if not match or table not in {"evidence_terms", "evidence_substrings"}:
        return []
    filters = ""
    parameters: list[object] = [match]
    if source_types:
        filters = " AND document.source_type IN (" + ",".join("?" for _ in source_types) + ")"
        parameters.extend(source_types)
    parameters.append(limit)
    sql = f"""
        SELECT document.ordinal,
               document.evidence_id,
               document.facility_ordinal,
               document.source_type,
               document.source_field,
               document.source_index,
               document.source_locator,
               document.original_text,
               document.language_hint,
               document.visit_date,
               document.scraped_at,
               document.is_verbatim,
               bm25({table})
        FROM {table}
        JOIN evidence_document AS document ON document.ordinal = {table}.rowid
        JOIN eligible ON eligible.ordinal = document.facility_ordinal
        WHERE {table} MATCH ?{filters}
        ORDER BY 13 ASC, document.ordinal ASC
        LIMIT ?
    """
    return connection.execute(sql, parameters).fetchall()


def _evidence_rows_for_facility(
    connection: sqlite3.Connection,
    *,
    table: str,
    match: str,
    facility_ordinal: int,
    limit: int,
    source_types: Sequence[str],
) -> list[tuple[object, ...]]:
    if not match or table not in {"evidence_terms", "evidence_substrings"}:
        return []
    filters = ""
    parameters: list[object] = [match, facility_ordinal]
    if source_types:
        filters = " AND document.source_type IN (" + ",".join(
            "?" for _ in source_types
        ) + ")"
        parameters.extend(source_types)
    parameters.append(limit)
    sql = f"""
        SELECT document.ordinal,
               document.evidence_id,
               document.facility_ordinal,
               document.source_type,
               document.source_field,
               document.source_index,
               document.source_locator,
               document.original_text,
               document.language_hint,
               document.visit_date,
               document.scraped_at,
               document.is_verbatim,
               bm25({table})
        FROM {table}
        JOIN evidence_document AS document ON document.ordinal = {table}.rowid
        JOIN eligible ON eligible.ordinal = document.facility_ordinal
        WHERE {table} MATCH ?
          AND document.facility_ordinal = ?{filters}
        ORDER BY 13 ASC, document.ordinal ASC
        LIMIT ?
    """
    return connection.execute(sql, parameters).fetchall()


def _reciprocal_rank(
    channels: Sequence[Sequence[tuple[object, ...]]],
    limit: int,
) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for rows in channels:
        for rank, row in enumerate(rows, start=1):
            ordinal = int(row[0])
            scores[ordinal] = scores.get(ordinal, 0.0) + 1.0 / (60.0 + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]
