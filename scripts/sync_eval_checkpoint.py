"""Upload one redacted evaluation directory to a private Hugging Face Dataset."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re

from huggingface_hub import HfApi


DEFAULT_DATASET = "ValerianFourel/seouldoc-eval-handoff-20260905"
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]+")
TOKEN_PATTERN = re.compile(
    r"(?i)(?:bearer\s+|sk-or-v1-|hf_)[A-Za-z0-9._~+/=-]{12,}"
)
ALLOWED_SUFFIXES = {".json", ".jsonl", ".md", ".tsv"}


def files_to_upload(run_dir: Path) -> list[Path]:
    if run_dir.is_symlink():
        raise ValueError(f"run directory may not be a symlink: {run_dir}")
    entries = list(run_dir.rglob("*"))
    symlinks = [path for path in entries if path.is_symlink()]
    if symlinks:
        names = ", ".join(str(path.relative_to(run_dir)) for path in symlinks[:8])
        raise ValueError(f"checkpoint contains symlinks: {names}")
    files = [path for path in entries if path.is_file()]
    if not files:
        raise ValueError(f"run directory contains no files: {run_dir}")
    blocked = [
        path
        for path in files
        if path.name.startswith(".env") or path.suffix.casefold() not in ALLOWED_SUFFIXES
    ]
    if blocked:
        names = ", ".join(str(path.relative_to(run_dir)) for path in blocked[:8])
        raise ValueError(f"checkpoint contains disallowed files: {names}")
    return files


def assert_redacted(files: list[Path]) -> None:
    secret_values = {
        value
        for name in (
            "HF_TOKEN",
            "OPENROUTER_API_KEY",
            "SEOULDOC_EVAL_AUTH_TOKEN",
        )
        if (value := os.getenv(name, "").strip())
    }
    for path in files:
        text = path.read_text(encoding="utf-8")
        if TOKEN_PATTERN.search(text):
            raise ValueError(f"credential-like text found in {path}")
        if any(secret in text for secret in secret_values):
            raise ValueError(f"configured credential value found in {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sync a sanitized SeoulDoc run to its private checkpoint Dataset."
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--dataset",
        default=os.getenv("SEOULDOC_RESULTS_DATASET", DEFAULT_DATASET),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    raw_run_dir = args.run_dir.expanduser()
    if raw_run_dir.is_symlink():
        raise SystemExit("run directory may not be a symlink")
    run_dir = raw_run_dir.resolve()
    run_id = run_dir.name
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise SystemExit("run directory name may contain only letters, numbers, dots, dashes, and underscores")
    token = os.getenv("HF_TOKEN", "").strip()
    if not token:
        raise SystemExit("HF_TOKEN is required")

    files = files_to_upload(run_dir)
    assert_redacted(files)

    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.dataset,
        repo_type="dataset",
        private=True,
        exist_ok=True,
    )
    commit = api.upload_folder(
        repo_id=args.dataset,
        repo_type="dataset",
        folder_path=run_dir,
        path_in_repo=f"runs/{run_id}",
        commit_message=f"Checkpoint SeoulDoc evaluation {run_id}",
    )
    print(f"Uploaded {len(files)} files to {args.dataset}@{commit.oid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
