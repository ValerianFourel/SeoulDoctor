"""Restore the pinned application release into ephemeral Space storage."""

from pathlib import Path
import os
import shutil

from huggingface_hub import snapshot_download

REPOSITORY = "ValerianFourel/seouldoc-app-release-20260905"
REVISION = "3911d79dc31e6a6ccfa3f64a7e401b88893bf66a"


def restore(source: Path, destination: Path) -> None:
    required = ("sources/facilities.parquet", "sources/reviews.parquet",
                "chroma_db", "search_indexes/active.json")
    for name in required:
        if not (source / name).exists():
            raise RuntimeError(f"Incomplete application release: {name}")
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source / "sources/facilities.parquet", destination / "facilities.parquet")
    shutil.copyfile(source / "sources/reviews.parquet", destination / "reviews.parquet")
    shutil.copytree(source / "chroma_db", destination / "chroma_db", dirs_exist_ok=True)
    indexes = destination / "search_indexes"
    if indexes.exists():
        for path in indexes.rglob("*"):
            path.chmod(0o700 if path.is_dir() else 0o600)
        indexes.chmod(0o700)
    shutil.copytree(source / "search_indexes", indexes, dirs_exist_ok=True)
    for path in indexes.rglob("*"):
        path.chmod(0o555 if path.is_dir() else 0o444)
    indexes.chmod(0o555)
    (destination / f".release-{REVISION}-ready").touch()


def main():
    destination = Path("/data/seouldoc")
    if (destination / f".release-{REVISION}-ready").exists():
        return
    source = Path(snapshot_download(
        REPOSITORY, repo_type="dataset", revision=REVISION,
        token=os.environ["HF_TOKEN"],
        allow_patterns=["sources/facilities.parquet", "sources/reviews.parquet",
                        "chroma_db/**", "search_indexes/**"],
    ))
    restore(source, destination)


if __name__ == "__main__":
    main()
