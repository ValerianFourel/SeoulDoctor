"""Administrative CLI for immutable search index releases."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import logging
from pathlib import Path
from typing import Sequence

from .build import BuildRequest
from .repository import IndexRepository


BACKEND_DIRECTORY = Path(__file__).resolve().parents[2]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and operate SeoulDoc search index releases.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    publish = subparsers.add_parser(
        "publish",
        help="build and publish an inactive immutable release",
    )
    publish.add_argument("--version", required=True)
    publish.add_argument(
        "--root",
        type=Path,
        default=BACKEND_DIRECTORY / "search_indexes",
    )
    publish.add_argument(
        "--facilities",
        type=Path,
        default=BACKEND_DIRECTORY / "local_facilities_cache.parquet",
    )
    publish.add_argument(
        "--reviews",
        type=Path,
        default=BACKEND_DIRECTORY / "local_reviews_cache.parquet",
    )
    publish.add_argument(
        "--chroma",
        type=Path,
        default=BACKEND_DIRECTORY / "chroma_db",
    )
    publish.add_argument("--collection", default="seoul_med_agentic_v2")
    publish.add_argument("--embedding-model", default="text-embedding-3-small")
    publish.add_argument("--embedding-revision", default="provider-managed")

    for command in ("activate", "validate"):
        operation = subparsers.add_parser(command)
        operation.add_argument("--root", type=Path, required=True)
        operation.add_argument("--version", required=command == "activate")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    arguments = _parser().parse_args(argv)
    repository = IndexRepository(arguments.root)
    if arguments.command == "publish":
        published = repository.publish(BuildRequest(
            version=arguments.version,
            facilities_path=arguments.facilities,
            reviews_path=arguments.reviews,
            chroma_path=arguments.chroma,
            chroma_collection=arguments.collection,
            embedding_model=arguments.embedding_model,
            embedding_revision=arguments.embedding_revision,
        ))
        payload = asdict(published)
        payload["directory"] = str(payload["directory"])
    elif arguments.command == "activate":
        published = repository.activate(arguments.version)
        payload = asdict(published)
        payload["directory"] = str(payload["directory"])
    else:
        if arguments.version:
            release = repository.open_version(arguments.version)
        else:
            release = repository.open_active()
        try:
            payload = release.manifest.to_payload()
            payload["directory"] = str(release._release.directory)
        finally:
            release.close()
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
