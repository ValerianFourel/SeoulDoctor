"""Upload immutable completed scenario checkpoints while other cases run."""

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from huggingface_hub import CommitOperationAdd, HfApi
from scripts.sync_eval_checkpoint import TOKEN_PATTERN


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    operations = []
    secrets = [os.environ[key] for key in ("HF_TOKEN", "OPENROUTER_API_KEY", "SEOULDOC_EVAL_AUTH_TOKEN")]
    for result in args.run_dir.glob("*/result.json"):
        record = json.loads(result.read_text())
        if record["status"] == "running":
            continue
        if record["status"] == "conversation_complete" and "judge_citations_valid" not in record:
            continue
        for path in result.parent.glob("*.json"):
            if path.is_symlink():
                raise ValueError("No symlinks in checkpoints")
            content = path.read_bytes()
            text = content.decode("utf-8")
            if TOKEN_PATTERN.search(text) or any(secret in text for secret in secrets):
                raise ValueError("Checkpoint failed credential scan")
            operations.append(CommitOperationAdd(
                path_in_repo=f"runs/{args.run_dir.name}/{path.relative_to(args.run_dir)}",
                path_or_fileobj=content))
    if not operations:
        print("No completed checkpoints ready")
        return
    api = HfApi(token=os.environ["HF_TOKEN"])
    repo = "ValerianFourel/seouldoc-eval-handoff"
    if not api.dataset_info(repo).private:
        raise ValueError("Results Dataset must remain private")
    commit = api.create_commit(repo_id=repo, repo_type="dataset", operations=operations,
                               commit_message="Completed diagnostic scenarios " + args.run_dir.name)
    print(json.dumps({"files": len(operations), "checkpoint_revision": commit.oid}))


if __name__ == "__main__":
    main()
