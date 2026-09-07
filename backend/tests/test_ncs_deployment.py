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

    def test_combined_bundle_keeps_gpu_sources_separate_from_backend(self):
        for name in ["production.py", "production_core.py", "requirements.txt", "requirements-prod.txt"]:
            path = "services/retriever/" + name
            self.assertEqual(destination(path, "app"), path)
        for path in ["services/retriever/Dockerfile", "services/retriever/app.py",
                     "services/retriever/production_draft.py", "services/reranker/Dockerfile"]:
            self.assertIsNone(destination(path, "app"))

    def test_targets_cannot_replace_original_application(self):
        self.assertNotIn("ValerianFourel/SeoulDoctor", TARGETS.values())
