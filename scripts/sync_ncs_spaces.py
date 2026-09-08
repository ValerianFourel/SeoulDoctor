"""Sync committed ncs application sources to preconfigured assessment Spaces."""

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import subprocess


ROOT = Path(__file__).resolve().parents[1]
TARGETS = {"app": "ValerianFourel/SeoulDoctor-ncs-retriever"}
RETRIEVER_FILES = {"requirements.txt", "requirements-prod.txt", "production.py", "production_core.py"}


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def destination(path, component):
    parts = PurePosixPath(path).parts
    if any(part.startswith(".") or part in {"tests", "node_modules", "out", "__pycache__"}
           for part in parts):
        return None
    if path.startswith("services/retriever/"):
        return path if path.removeprefix("services/retriever/") in RETRIEVER_FILES else None
    if path == "Dockerfile" or path == "backend/requirements.txt":
        return path
    if path.startswith("backend/") and path.endswith(".py"):
        if len(parts) == 2 or parts[1] == "search":
            return path
    if path.startswith("frontend/") and PurePosixPath(path).suffix in {
        ".json", ".js", ".mjs", ".ts", ".tsx", ".css", ".svg", ".png",
        ".jpg", ".jpeg", ".ico", ".webp", ".txt", ".xml",
    }:
        return path
    return None


def bundle(component):
    revision = git("rev-parse", "HEAD").decode().strip()
    if git("branch", "--show-current").decode().strip() != "ncs":
        raise RuntimeError("Sync requires a checked-out ncs branch")
    result = {}
    for raw in git("ls-tree", "-rz", "HEAD").split(b"\0"):
        if not raw:
            continue
        metadata, raw_path = raw.split(b"\t", 1)
        path = raw_path.decode()
        target = destination(path, component)
        if target:
            mode, kind, oid = metadata.decode().split()
            if kind != "blob" or mode not in {"100644", "100755"}:
                raise RuntimeError(f"Refusing nonregular deployment file: {path}")
            result[target] = git("cat-file", "blob", oid)
    result["ncs-source.json"] = (json.dumps({"branch": "ncs", "commit": revision}, indent=2)+"\n").encode()
    return revision, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("component", choices=TARGETS)
    parser.add_argument("--apply", action="store_true", help="Upload to the existing NCS Space")
    parser.add_argument("--allow-public", action="store_true", help="Allow updating the existing NCS Space when it is public")
    args = parser.parse_args()
    revision, files = bundle(args.component)
    target = TARGETS[args.component]
    print(json.dumps({"target": target, "commit": revision, "files": sorted(files), "apply": args.apply}))
    if not args.apply:
        return
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN must be injected through the process environment")
    from huggingface_hub import HfApi, CommitOperationAdd, CommitOperationDelete
    api = HfApi(token=token)
    info = api.space_info(target)
    if not info.private and not args.allow_public:
        raise RuntimeError("Public NCS deployment requires --allow-public")
    variables = api.get_space_variables(target)
    if variables.get("NCS_SOURCE_BRANCH") is None or variables["NCS_SOURCE_BRANCH"].value != "ncs":
        raise RuntimeError("Space must be explicitly provisioned with NCS_SOURCE_BRANCH=ncs")
    operations = [CommitOperationAdd(path_in_repo=name, path_or_fileobj=data)
                  for name, data in files.items()]
    # README carries provisioned Space configuration and is intentionally retained.
    for entry in info.siblings:
        name = entry.rfilename
        managed = destination(name, "app") or name in {
            "production.py", "production_core.py", "requirements.txt", "requirements-prod.txt",
        }
        if managed and name not in files:
            operations.append(CommitOperationDelete(path_in_repo=name))
    commit = api.create_commit(target, repo_type="space", operations=operations,
                              parent_commit=info.sha,
                              commit_message=f"Sync ncs {revision}")
    print(json.dumps({"space_commit": commit.oid, "source_commit": revision}))


if __name__ == "__main__":
    main()
