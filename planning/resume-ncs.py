"""Resume Codex in this worktree with locally authorized credentials."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from dotenv import dotenv_values


def main():
    root = Path(__file__).resolve().parents[1]
    branch = subprocess.check_output(
        ["git", "-C", str(root), "branch", "--show-current"], text=True
    ).strip()
    if branch != "ncs":
        raise SystemExit("Expected branch ncs; found " + branch)
    codex = shutil.which("codex")
    if not codex:
        raise SystemExit("codex is not on PATH")
    common = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "--path-format=absolute", "--git-common-dir"],
        text=True,
    ).strip()
    credential_file = Path(common).parent / "backend" / ".env"
    values = dotenv_values(credential_file)
    required = ("HF_TOKEN", "OPENROUTER_API_KEY", "SEOULDOC_EVAL_AUTH_TOKEN")
    for name in required:
        value = values.get(name) or os.environ.get(name)
        if not value:
            raise SystemExit("Missing credential: " + name)
        os.environ[name] = value
    allowed = [
        "PATH", "HOME", "USER", "SHELL", "TERM", "LANG", "LC_*", "TMPDIR",
        "XDG_*", "SSH_AUTH_SOCK", *required,
    ]
    print("Resuming in " + str(root), flush=True)
    print("Select the NCS session, then ask it to read planning/04-session-reentry.md.", flush=True)
    os.chdir(root)
    os.execv(codex, [
        codex,
        "-c", 'shell_environment_policy.inherit="all"',
        "-c", "shell_environment_policy.ignore_default_excludes=true",
        "-c", "shell_environment_policy.include_only=" + json.dumps(allowed),
        "resume", "--all", "--cd", str(root), *sys.argv[1:],
    ])


if __name__ == "__main__":
    main()
