"""End-to-end contracts for immutable Phase 3 search index releases."""

from __future__ import annotations

from hashlib import sha256
import asyncio
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
import unittest

import chromadb
from chromadb.config import Settings
import numpy as np
import pandas as pd

from search.contracts import EligibleScope
from search.indexes import (
    BuildRequest,
    IndexBuildError,
    IndexCollisionError,
    IndexLoadError,
    IndexRepository,
)
from search.indexes.documents import (
    facility_index_document,
    raw_review_evidence_id,
    render_facility_profile,
)
from search.scope import ScopeSelection
from search.rules import RulesCompiler


class SearchIndexReleaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = Path(tempfile.mkdtemp(prefix="seouldoc-index-test-"))
        self.addCleanup(self._cleanup)
        self.facilities_path = self.temporary / "facilities.parquet"
        self.reviews_path = self.temporary / "reviews.parquet"
        self.chroma_path = self.temporary / "chroma"
        self.index_root = self.temporary / "indexes"

        facilities = pd.DataFrame([
            {
                "place_id": "alpha",
                "name": "Gangnam International Skin Clinic",
                "category": "피부과",
                "address": "강남구 테헤란로",
                "file_district": "강남구",
                "file_dong": "역삼동",
                "Summaries": ["English acne and eczema consultations"],
                "Summaries_Korean": ["여드름과 피부염을 친절하게 진료"],
                "Key_Highlights": [
                    {"topic": "Friendly care", "topic_en": "Friendly care", "topic_ko": "친절한 진료"}
                ],
                "amenities": {"wheelchair": True},
                "medical_info_parsed": {"appointment": "available"},
                "has_english": True,
            },
            {
                "place_id": "beta",
                "name": "Seoul Dental Center",
                "category": "치과",
                "address": "마포구 월드컵로",
                "file_district": "마포구",
                "file_dong": "성산동",
                "Summaries": ["Dental implants and checkups"],
                "Summaries_Korean": ["임플란트와 치과 검진"],
                "Key_Highlights": [],
                "amenities": {},
                "medical_info_parsed": {},
                "has_english": False,
            },
        ])
        facilities.to_parquet(self.facilities_path, index=False)
        pd.DataFrame([
            {
                "place_id": "alpha",
                "facility_name": "Gangnam International Skin Clinic",
                "review_index": 0,
                "review_text": "English friendly acne treatment",
                "visit_date": "2025-01",
                "scraped_at": "2025-02-01",
                "reviewer_name": "must-not-be-indexed",
            },
            {
                "place_id": "alpha",
                "facility_name": "Gangnam International Skin Clinic",
                "review_index": 1,
                "review_text": "여드름 치료가 정말 친절해요",
                "visit_date": "2025-02",
                "scraped_at": "2025-03-01",
                "reviewer_name": "비공개 이름",
            },
            {
                "place_id": "beta",
                "facility_name": "Seoul Dental Center",
                "review_index": 0,
                "review_text": "Great dental implant care",
                "visit_date": "2025-03",
                "scraped_at": "2025-04-01",
                "reviewer_name": "also-private",
            },
            {
                "place_id": "beta",
                "facility_name": "Seoul Dental Center",
                "review_index": 1,
                "review_text": "\x00임플란트 상담이 자세해요",
                "visit_date": "2025-04",
                "scraped_at": "2025-05-01",
                "reviewer_name": "nul-private",
            },
            {
                "place_id": "alpha",
                "facility_name": "Gangnam International Skin Clinic",
                "review_index": 2,
                "review_text": "  ",
                "visit_date": None,
                "scraped_at": None,
                "reviewer_name": "empty-review-private",
            },
        ]).to_parquet(self.reviews_path, index=False)
        self._create_chroma()
        self.repository = IndexRepository(self.index_root)

    def _cleanup(self) -> None:
        for root, directories, files in os.walk(self.temporary):
            os.chmod(root, 0o700)
            for name in directories:
                os.chmod(Path(root) / name, 0o700)
            for name in files:
                os.chmod(Path(root) / name, 0o600)
        shutil.rmtree(self.temporary, ignore_errors=True)

    def _create_chroma(self) -> None:
        facilities = pd.read_parquet(self.facilities_path)
        by_id = {
            str(row["place_id"]): render_facility_profile(row)
            for _, row in facilities.iterrows()
        }
        client = chromadb.PersistentClient(
            path=str(self.chroma_path),
            settings=Settings(anonymized_telemetry=False),
        )
        collection = client.create_collection("seoul_med_agentic_v2")
        alpha = np.zeros(1536, dtype=np.float32)
        alpha[0] = 1.0
        beta = np.zeros(1536, dtype=np.float32)
        beta[1] = 1.0
        collection.add(
            ids=["alpha", "beta"],
            documents=[by_id["alpha"], by_id["beta"]],
            embeddings=[alpha.tolist(), beta.tolist()],
        )

    def _request(self, version: str = "test-v1") -> BuildRequest:
        return BuildRequest(
            version=version,
            facilities_path=self.facilities_path,
            reviews_path=self.reviews_path,
            chroma_path=self.chroma_path,
        )

    @staticmethod
    def _scope(version: str, facility_ids: tuple[str, ...]) -> ScopeSelection:
        rules_hash = "a" * 64
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
        digest = sha256(payload.encode("utf-8")).hexdigest()
        return ScopeSelection(
            descriptor=EligibleScope(
                index_version=version,
                rules_hash=rules_hash,
                facility_bitmap_ref=f"memory://{digest}",
                facility_count=len(facility_ids),
                scope_digest=digest,
            ),
            facility_ids=facility_ids,
            distance_km_by_facility={},
        )

    def test_publish_activate_open_and_search_within_scope(self) -> None:
        published = self.repository.publish(self._request())
        self.assertFalse(published.reused)
        self.assertFalse((self.index_root / "active.json").exists())
        self.assertEqual(
            {path.name for path in published.directory.iterdir()},
            {
                "manifest.json",
                "facility_ordinals.arrow",
                "facility_vectors.npy",
                "facility_lexical.sqlite3",
                "evidence_lexical.sqlite3",
            },
        )
        self.assertFalse(published.directory.stat().st_mode & 0o222)
        self.assertTrue(all(
            not path.stat().st_mode & 0o222
            for path in published.directory.iterdir()
        ))

        self.repository.activate("test-v1")
        with self.repository.open_active() as active:
            self.assertEqual(active.manifest.facility_count, 2)
            self.assertEqual(active.manifest.indexed_review_count, 4)
            with active.within(self._scope("test-v1", ("alpha",))) as scoped:
                self.assertEqual(
                    [hit.facility_id for hit in scoped.search_facilities("피부과 여드름")],
                    ["alpha"],
                )
                self.assertEqual(
                    scoped.search_dense(np.eye(1, 1536, 0)[0])[0].facility_id,
                    "alpha",
                )
                korean = scoped.search_evidence("여드름 친절")
                self.assertTrue(korean)
                self.assertTrue(all(hit.facility_id == "alpha" for hit in korean))
                english = scoped.search_evidence(
                    "friendly acne",
                    source_types=("verbatim_review",),
                )
                self.assertEqual(english[0].original_text, "English friendly acne treatment")
                self.assertEqual(english[0].source_locator, "review_snapshot:alpha:0")
                required_english = RulesCompiler().compile(
                    original_query="English consultation is required in Seoul",
                    turn_id="source-type-regression",
                    proposal={"location": "Seoul", "hard_keywords": ["English consultation"]},
                ).rules.evidence[0]
                hard_language_hits = scoped.search_evidence_for_facilities(
                    "English", facility_ids=("alpha",), limit_per_facility=5,
                    source_types=tuple(required_english.source_types),
                )
                self.assertIn(english[0].evidence_id, {hit.evidence_id for hit in hard_language_hits})
                for invalid in (("review",), ("unknown_source",)):
                    with self.subTest(source_types=invalid), self.assertRaises(ValueError):
                        scoped.search_evidence("English", source_types=invalid)
                    with self.subTest(source_types=invalid), self.assertRaises(ValueError):
                        scoped.search_evidence_for_facilities(
                            "English", facility_ids=("alpha",), limit_per_facility=5,
                            source_types=invalid,
                        )

        database = sqlite3.connect(
            f"file:{published.directory / 'evidence_lexical.sqlite3'}?mode=ro&immutable=1",
            uri=True,
        )
        try:
            columns = {
                row[1]
                for row in database.execute("PRAGMA table_info(evidence_document)")
            }
            self.assertNotIn("reviewer_name", columns)
            joined = " ".join(
                row[0]
                for row in database.execute("SELECT original_text FROM evidence_document")
            )
            self.assertNotIn("must-not-be-indexed", joined)
            self.assertNotIn("비공개 이름", joined)
            self.assertIn("\x00임플란트 상담이 자세해요", joined)
        finally:
            database.close()

    def test_repeat_publish_is_noop_and_activation_can_roll_back(self) -> None:
        first = self.repository.publish(self._request("test-v1"))
        repeated = self.repository.publish(self._request("test-v1"))
        self.assertTrue(repeated.reused)
        self.assertEqual(first.content_id, repeated.content_id)

        second = self.repository.publish(self._request("test-v2"))
        self.assertEqual(first.content_id, second.content_id)
        self.repository.activate("test-v2")
        with self.repository.open_active() as active:
            self.assertEqual(active.version, "test-v2")
        self.repository.activate("test-v1")
        with self.repository.open_active() as active:
            self.assertEqual(active.version, "test-v1")

    def test_same_version_rejects_different_source_content(self) -> None:
        self.repository.publish(self._request())
        reviews = pd.read_parquet(self.reviews_path)
        reviews.loc[0, "review_text"] = "changed review text"
        reviews.to_parquet(self.reviews_path, index=False)
        with self.assertRaises(IndexCollisionError):
            self.repository.publish(self._request())
        self.repository.activate("test-v1")
        with self.repository.open_active() as active:
            self.assertEqual(active.manifest.indexed_review_count, 4)

    def test_scope_digest_and_index_membership_are_mandatory(self) -> None:
        self.repository.publish(self._request())
        self.repository.activate("test-v1")
        with self.repository.open_active() as active:
            valid = self._scope("test-v1", ("alpha",))
            invalid_digest = ScopeSelection(
                descriptor=EligibleScope(
                    index_version="test-v1",
                    rules_hash="a" * 64,
                    facility_bitmap_ref="memory://" + "b" * 64,
                    facility_count=1,
                    scope_digest="b" * 64,
                ),
                facility_ids=("alpha",),
                distance_km_by_facility={},
            )
            with self.assertRaises(IndexLoadError):
                active.within(invalid_digest)
            unknown = self._scope("test-v1", ("missing",))
            with self.assertRaises(IndexLoadError):
                active.within(unknown)
            with active.within(valid) as scoped:
                self.assertEqual(scoped.search_facilities("dental"), [])

    def test_orphan_reviews_and_mismatched_chroma_documents_are_rejected(self) -> None:
        reviews = pd.read_parquet(self.reviews_path)
        empty_row = reviews.index[-1]
        self.assertFalse(str(reviews.loc[empty_row, "review_text"]).strip())
        reviews.loc[empty_row, "place_id"] = "unknown"
        reviews.to_parquet(self.reviews_path, index=False)
        with self.assertRaisesRegex(IndexBuildError, "orphan"):
            self.repository.publish(self._request("orphan"))

        reviews.loc[empty_row, "place_id"] = "alpha"
        reviews.to_parquet(self.reviews_path, index=False)
        collection = chromadb.PersistentClient(
            path=str(self.chroma_path),
            settings=Settings(anonymized_telemetry=False),
        ).get_collection("seoul_med_agentic_v2")
        alpha = np.zeros(1536, dtype=np.float32)
        alpha[0] = 1.0
        collection.update(
            ids=["alpha"],
            documents=["tampered profile"],
            embeddings=[alpha.tolist()],
        )
        with self.assertRaisesRegex(IndexBuildError, "profile mismatch"):
            self.repository.publish(self._request("profile-mismatch"))

    def test_tampered_artifact_and_unsafe_pointer_are_rejected(self) -> None:
        published = self.repository.publish(self._request())
        self.repository.activate("test-v1")
        vector_path = published.directory / "facility_vectors.npy"
        vector_path.chmod(0o644)
        with vector_path.open("r+b") as output:
            output.seek(-1, os.SEEK_END)
            previous = output.read(1)
            output.seek(-1, os.SEEK_END)
            output.write(bytes((previous[0] ^ 0xFF,)))
        vector_path.chmod(0o444)
        with self.assertRaises(IndexLoadError):
            self.repository.open_active()

        active_path = self.index_root / "active.json"
        active_path.unlink()
        active_path.symlink_to(Path("versions/test-v1/manifest.json"))
        with self.assertRaises(IndexLoadError):
            self.repository.open_active()

    def test_literal_none_identifier_is_preserved_as_an_existing_identity(self) -> None:
        document = facility_index_document({
            "place_id": "None",
            "name": "연세채움치과의원",
            "category": "치과",
        })
        self.assertEqual(document.place_id, "None")

    def test_repository_rejects_a_symlink_in_any_root_component(self) -> None:
        real_parent = self.temporary / "real-parent"
        real_parent.mkdir()
        (real_parent / "indexes" / "versions").mkdir(parents=True)
        linked_parent = self.temporary / "linked-parent"
        linked_parent.symlink_to(real_parent, target_is_directory=True)
        repository = IndexRepository(linked_parent / "indexes")
        with self.assertRaisesRegex(IndexLoadError, "cannot contain a symlink"):
            repository.open_active()

    def test_required_index_validation_failure_stops_startup(self) -> None:
        import main

        original_download = main.download_and_cache_parquet
        original_reviews = main.ensure_raw_review_parquet
        original_root = main.SEARCH_INDEX_ROOT
        original_required = main.SEARCH_INDEX_REQUIRED
        try:
            main.download_and_cache_parquet = lambda: pd.DataFrame([
                {"place_id": "alpha", "category": "피부과"}
            ])
            main.ensure_raw_review_parquet = lambda: str(self.reviews_path)
            main.SEARCH_INDEX_ROOT = str(self.temporary / "missing-indexes")
            main.SEARCH_INDEX_REQUIRED = True

            async def enter_lifespan() -> None:
                async with main.lifespan(main.app):
                    self.fail("required invalid index must not reach application yield")

            with self.assertRaises(main.RequiredSearchIndexError):
                asyncio.run(enter_lifespan())
        finally:
            main.download_and_cache_parquet = original_download
            main.ensure_raw_review_parquet = original_reviews
            main.SEARCH_INDEX_ROOT = original_root
            main.SEARCH_INDEX_REQUIRED = original_required

    def test_lexical_projection_includes_production_field_shapes(self) -> None:
        document = facility_index_document({
            "place_id": "facility",
            "name": "병원",
            "category": "내과",
            "Key_Highlights": np.array([{
                "topic_en": "Short waiting time",
                "topic_ko": "대기 시간이 짧음",
            }], dtype=object),
            "amenities": "주차, 무선 인터넷",
            "business_hours": "토요일 진료",
            "phone": "02-000-0000",
        })
        self.assertIn("short waiting time", document.english_terms)
        self.assertIn("대기 시간이 짧음", document.korean_terms)
        self.assertIn("주차", document.trusted_facts)
        self.assertIn("토요일 진료", document.trusted_facts)


    def test_general_review_sample_is_original_bounded_and_scoped(self):
        self.repository.publish(self._request())
        self.repository.activate("test-v1")
        with self.repository.open_active() as active:
            with active.within(self._scope("test-v1", ("alpha",))) as scoped:
                hits = scoped.list_original_reviews(facility_ids=("alpha",), limit_per_facility=1)
                self.assertEqual(len(hits), 1)
                self.assertEqual(hits[0].facility_id, "alpha")
                self.assertTrue(hits[0].is_verbatim)
                self.assertEqual(hits, scoped.resolve_evidence_ids([hits[0].evidence_id]))
                self.assertTrue(hits[0].original_text)
                self.assertTrue(hits[0].source_locator)
                with self.assertRaisesRegex(ValueError, "outside scope"):
                    scoped.list_original_reviews(facility_ids=("beta",))
                self.assertEqual(
                    scoped.list_original_reviews(facility_ids=("alpha",), limit_per_facility=101),
                    scoped.list_original_reviews(facility_ids=("alpha",), limit_per_facility=100),
                )
                self.assertEqual(scoped.list_original_reviews(facility_ids=()), [])

    def test_facility_scoped_evidence_enforces_scope_and_per_facility_quota(self) -> None:
        self.repository.publish(self._request())
        self.repository.activate("test-v1")
        with self.repository.open_active() as active:
            scope = self._scope("test-v1", ("alpha", "beta"))
            with active.within(scope) as scoped:
                hits = scoped.search_evidence_for_facilities(
                    "treatment care",
                    facility_ids=("alpha", "beta"),
                    limit_per_facility=1,
                    source_types=("verbatim_review",),
                )
                self.assertEqual(
                    {hit.facility_id for hit in hits},
                    {"alpha", "beta"},
                )
                self.assertEqual(
                    max(
                        sum(hit.facility_id == facility_id for hit in hits)
                        for facility_id in ("alpha", "beta")
                    ),
                    1,
                )
                with self.assertRaisesRegex(ValueError, "duplicated"):
                    scoped.search_evidence_for_facilities(
                        "care",
                        facility_ids=("alpha", "alpha"),
                        limit_per_facility=1,
                    )
                with self.assertRaisesRegex(ValueError, "outside"):
                    scoped.search_evidence_for_facilities(
                        "care",
                        facility_ids=("outside",),
                        limit_per_facility=1,
                    )

    def test_evidence_id_resolution_is_local_ordered_and_scope_bound(self) -> None:
        self.repository.publish(self._request())
        self.repository.activate("test-v1")
        alpha_id = raw_review_evidence_id(
            "alpha", 0, "English friendly acne treatment"
        )
        with self.repository.open_active() as active:
            with active.within(self._scope("test-v1", ("alpha",))) as scoped:
                resolved = scoped.resolve_evidence_ids((alpha_id, alpha_id))
                self.assertEqual(
                    [item.evidence_id for item in resolved],
                    [alpha_id],
                )
                self.assertEqual(resolved[0].facility_id, "alpha")
                self.assertTrue(resolved[0].is_verbatim)
                self.assertEqual(
                    scoped.review_source_sha256,
                    active.manifest.review_source_sha256,
                )


    def test_evidence_id_resolver_rejects_empty_unknown_and_outside_scope(self) -> None:
        self.repository.publish(self._request())
        self.repository.activate("test-v1")
        beta_id = raw_review_evidence_id(
            "beta", 0, "Great dental implant care"
        )
        with self.repository.open_active() as active:
            with active.within(self._scope("test-v1", ("alpha",))) as scoped:
                with self.assertRaisesRegex(ValueError, "cannot be empty"):
                    scoped.resolve_evidence_ids(("",))
                with self.assertRaisesRegex(ValueError, "inside the eligible scope"):
                    scoped.resolve_evidence_ids(("review:unknown",))
                with self.assertRaisesRegex(ValueError, "inside the eligible scope"):
                    scoped.resolve_evidence_ids((beta_id,))

    def test_evidence_id_resolver_rejects_derived_evidence(self) -> None:
        published = self.repository.publish(self._request())
        self.repository.activate("test-v1")
        connection = sqlite3.connect(
            published.directory / "evidence_lexical.sqlite3"
        )
        self.addCleanup(connection.close)
        row = connection.execute(
            """
            SELECT evidence_id
            FROM evidence_document
            WHERE facility_ordinal = 0
              AND (source_type != 'verbatim_review' OR is_verbatim != 1)
            ORDER BY ordinal
            LIMIT 1
            """
        ).fetchone()
        self.assertIsNotNone(row)
        derived_id = str(row[0])

        with self.repository.open_active() as active:
            with active.within(self._scope("test-v1", ("alpha",))) as scoped:
                with self.assertRaisesRegex(ValueError, "verbatim review"):
                    scoped.resolve_evidence_ids((derived_id,))


if __name__ == "__main__":
    unittest.main()
