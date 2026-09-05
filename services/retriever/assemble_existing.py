"""Assemble and publish a release from verified offline encoder checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import offline_builder


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--shard-size", type=int, default=25_000)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--sparse-top-k", type=int, default=128)
    parser.add_argument("--dataset-repo", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    started = time.monotonic()
    source_sha256 = offline_builder._sha256_file(args.reviews)
    records = offline_builder.load_reviews(args.reviews)
    shard_paths: list[Path] = []

    for number, start in enumerate(range(0, len(records), args.shard_size)):
        end = min(start + args.shard_size, len(records))
        path, metadata_path = offline_builder._shard_paths(
            args.checkpoint_dir,
            number,
        )
        expected = {
            "schema": offline_builder.SHARD_SCHEMA,
            "shard_number": number,
            "start": start,
            "end": end,
            "review_source_sha256": source_sha256,
            "model_id": offline_builder.MODEL_ID,
            "model_revision": args.model_revision,
            "dimension": offline_builder.MODEL_DIMENSION,
            "sparse_top_k": args.sparse_top_k,
            "max_length": args.max_length,
        }
        if not offline_builder._valid_checkpoint(
            path,
            metadata_path,
            expected,
            records[start:end],
        ):
            raise offline_builder.BuildError(f"checkpoint {number} failed validation")
        shard_paths.append(path)
        print(
            f"validated checkpoint {number + 1}: rows {start}:{end}",
            flush=True,
        )

    args.work_dir.mkdir(parents=True, exist_ok=True)
    release_dir = offline_builder.assemble_release(
        records=records,
        shard_paths=shard_paths,
        work_dir=args.work_dir,
        release_id=args.release_id,
        source_sha256=source_sha256,
        model_revision=args.model_revision,
        sparse_top_k=args.sparse_top_k,
        max_length=args.max_length,
    )
    token = os.getenv("HF_TOKEN", "").strip()
    dataset_revision = offline_builder.upload_release(
        release_dir,
        args.dataset_repo,
        token,
    )
    print(
        json.dumps(
            {
                "release_dir": str(release_dir),
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "manifest": offline_builder.validate_release(release_dir),
                "dataset_repo": args.dataset_repo,
                "dataset_revision": dataset_revision,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
