"""Deployment boundaries for the isolated assessment Spaces."""

import unittest

from scripts.sync_ncs_spaces import destination, TARGETS


class AssessmentDeploymentTests(unittest.TestCase):
    def test_private_artifacts_and_environment_files_are_excluded(self):
        paths = [
            "backend/.env", "frontend/.env.production", ".audit/run.json",
            "backend/local_reviews_cache.parquet", "backend/tests/patient.py",
            "backend/search_indexes/manifest.json", "backend/chroma_db/index.bin",
            "planning/session-logs/raw.json", "frontend/node_modules/pkg/index.js",
            "frontend/out/index.html", "backend/__pycache__/main.pyc",
        ]
        for component in TARGETS:
            for path in paths:
                with self.subTest(component=component, path=path):
                    self.assertIsNone(destination(path, component))

    def test_application_includes_runtime_files(self):
        for path in ["Dockerfile", "backend/main.py", "backend/requirements.txt",
                     "backend/search/indexes/reader.py", "frontend/package-lock.json",
                     "frontend/app/page.tsx", "frontend/public/img/logo.png"]:
            self.assertEqual(destination(path, "app"), path)

    def test_retriever_uses_its_own_root_without_fixture_or_drafts(self):
        self.assertEqual(destination("services/retriever/Dockerfile", "retriever"), "Dockerfile")
        self.assertEqual(destination("services/retriever/production.py", "retriever"), "production.py")
        for path in ["Dockerfile", "services/retriever/app.py",
                     "services/retriever/production_draft.py", "services/reranker/Dockerfile"]:
            self.assertIsNone(destination(path, "retriever"))

    def test_targets_cannot_replace_original_application(self):
        self.assertNotIn("ValerianFourel/SeoulDoctor", TARGETS.values())
