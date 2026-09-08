"""Deployment boundaries for the isolated assessment Spaces."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from scripts.sync_ncs_spaces import destination, main, TARGETS


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

    def test_public_deployment_requires_explicit_acknowledgment(self):
        api = Mock()
        api.space_info.return_value = SimpleNamespace(private=False)
        with (
            patch("sys.argv", ["sync_ncs_spaces.py", "app", "--apply"]),
            patch("scripts.sync_ncs_spaces.bundle", return_value=("source-sha", {})),
            patch.dict("os.environ", {"HF_TOKEN": "test-token"}),
            patch("huggingface_hub.HfApi", return_value=api),
            patch("builtins.print"),
        ):
            with self.assertRaisesRegex(RuntimeError, "requires --allow-public"):
                main()
        api.create_commit.assert_not_called()

    def test_acknowledged_public_deployment_preserves_target_and_revision_guard(self):
        api = Mock()
        api.space_info.return_value = SimpleNamespace(private=False, sha="previous-sha", siblings=[])
        api.get_space_variables.return_value = {"NCS_SOURCE_BRANCH": SimpleNamespace(value="ncs")}
        api.create_commit.return_value = SimpleNamespace(oid="new-sha")
        with (
            patch("sys.argv", ["sync_ncs_spaces.py", "app", "--apply", "--allow-public"]),
            patch("scripts.sync_ncs_spaces.bundle", return_value=("source-sha", {"backend/main.py": b"app"})),
            patch.dict("os.environ", {"HF_TOKEN": "test-token"}),
            patch("huggingface_hub.HfApi", return_value=api),
            patch("builtins.print"),
        ):
            main()
        api.create_commit.assert_called_once()
        args, kwargs = api.create_commit.call_args
        self.assertEqual(args, ("ValerianFourel/SeoulDoctor-ncs-retriever",))
        self.assertEqual(kwargs["parent_commit"], "previous-sha")
        self.assertEqual(kwargs["repo_type"], "space")
        self.assertEqual([operation.path_in_repo for operation in kwargs["operations"]], ["backend/main.py"])
        api.update_repo_settings.assert_not_called()
        api.request_space_hardware.assert_not_called()

    def test_public_acknowledgment_does_not_bypass_branch_marker(self):
        api = Mock()
        api.space_info.return_value = SimpleNamespace(private=False)
        api.get_space_variables.return_value = {"NCS_SOURCE_BRANCH": SimpleNamespace(value="main")}
        with (
            patch("sys.argv", ["sync_ncs_spaces.py", "app", "--apply", "--allow-public"]),
            patch("scripts.sync_ncs_spaces.bundle", return_value=("source-sha", {})),
            patch.dict("os.environ", {"HF_TOKEN": "test-token"}),
            patch("huggingface_hub.HfApi", return_value=api),
            patch("builtins.print"),
        ):
            with self.assertRaisesRegex(RuntimeError, "NCS_SOURCE_BRANCH=ncs"):
                main()
        api.create_commit.assert_not_called()
