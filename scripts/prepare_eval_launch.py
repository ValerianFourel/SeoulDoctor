#!/usr/bin/env python3
"""Validate the complete SeoulDoc evaluation launch without making paid calls."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PINNED_ENDPOINT = "https://valerianfourel-seouldoctor.hf.space/chat"
PINNED_RELEASE = "ValerianFourel/seouldoc-app-release-20260905"
PINNED_REVISION = "3911d79dc31e6a6ccfa3f64a7e401b88893bf66a"
PINNED_RESULTS = "ValerianFourel/seouldoc-eval-handoff"
TARGETED_SCENARIOS = (
    "reverse-yongsan-peds-03-en",
    "reverse-yongsan-peds-03-ko",
)


@dataclass(frozen=True)
class SuiteInventory:
    name: str
    path: str
    scenario_count: int
    languages: tuple[str, ...]


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def scenario_list(payload: Any, path: Path) -> list[Mapping[str, Any]]:
    scenarios = payload.get("scenarios") if isinstance(payload, dict) else payload
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError(f"{path} must contain a non-empty scenario list")
    if not all(isinstance(item, dict) for item in scenarios):
        raise ValueError(f"{path} contains a non-object scenario")
    return scenarios


def inventory(root: Path = REPOSITORY_ROOT) -> list[SuiteInventory]:
    definitions = (
        ("sealed_grounded", "backend/tests/grounded_bilingual_scenarios.json", 8),
        ("bilingual_regression", "backend/tests/openrouter_bilingual_scenarios.json", 12),
        ("agentic_stress", "backend/tests/agentic_search_stress_cases.json", 10),
    )
    result = []
    for name, relative_path, expected_count in definitions:
        path = root / relative_path
        scenarios = scenario_list(load_json(path), path)
        if len(scenarios) != expected_count:
            raise ValueError(
                f"{relative_path} has {len(scenarios)} scenarios; expected {expected_count}"
            )
        identifiers = [str(item.get("id", item.get("level", ""))) for item in scenarios]
        if any(not identifier for identifier in identifiers) or len(set(identifiers)) != len(identifiers):
            raise ValueError(f"{relative_path} has missing or duplicate scenario identifiers")
        languages = sorted(
            {
                str(item.get("source_language", item.get("language", "unknown")))
                for item in scenarios
            }
        )
        result.append(SuiteInventory(name, relative_path, len(scenarios), tuple(languages)))

    grounded = scenario_list(
        load_json(root / definitions[0][1]), root / definitions[0][1]
    )
    grounded_ids = {str(item.get("id")) for item in grounded}
    missing_targets = sorted(set(TARGETED_SCENARIOS) - grounded_ids)
    if missing_targets:
        raise ValueError("sealed casebook lacks targeted scenarios: " + ", ".join(missing_targets))
    return result


def git_value(root: Path, *arguments: str) -> str:
    process = subprocess.run(
        ("git", *arguments),
        cwd=root,
        capture_output=True,
        check=False,
        text=True,
        timeout=10,
    )
    return process.stdout.strip() if process.returncode == 0 else "unknown"


def environment_status(environment: Mapping[str, str]) -> dict[str, Any]:
    hf_token = bool(environment.get("HF_TOKEN", "").strip())
    auth_token = bool(environment.get("SEOULDOC_EVAL_AUTH_TOKEN", "").strip())
    values = {
        "HF_TOKEN": hf_token,
        "OPENROUTER_API_KEY": bool(environment.get("OPENROUTER_API_KEY", "").strip()),
        "SEOULDOC_EVAL_AUTH_TOKEN": auth_token,
    }
    return {
        "required": values,
        "ready": all(values.values()),
    }


def build_report(
    root: Path = REPOSITORY_ROOT,
    environment: Mapping[str, str] = os.environ,
) -> dict[str, Any]:
    suites = inventory(root)
    env_status = environment_status(environment)
    branch = git_value(root, "branch", "--show-current")
    commit = git_value(root, "rev-parse", "HEAD")
    tracked_changes = git_value(root, "status", "--short", "--untracked-files=no")
    checks = {
        "expected_branch": branch == "codex/modify-code-and-launch-evaluation",
        "tracked_tree_clean": not tracked_changes,
        "scenario_inventory": True,
        "credentials_available": env_status["ready"],
    }
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "preflight_only_no_network_or_paid_calls",
        "git": {"branch": branch, "commit": commit},
        "resources": {
            "endpoint": environment.get("SEOULDOC_APP_ENDPOINT", PINNED_ENDPOINT),
            "release_dataset": environment.get("SEOULDOC_RELEASE_DATASET", PINNED_RELEASE),
            "release_revision": environment.get("SEOULDOC_RELEASE_REVISION", PINNED_REVISION),
            "results_dataset": environment.get("SEOULDOC_RESULTS_DATASET", PINNED_RESULTS),
        },
        "credentials": env_status,
        "suites": [asdict(item) for item in suites],
        "random_holdout": {
            "frozen_seed": 6688037892107667854,
            "facility_count": 6,
            "scenario_count": 12,
            "languages": ["en", "ko"],
            "status": "regenerate_from_pinned_release_before_patient_runs",
        },
        "launch_order": [
            "local_tests",
            "remote_health",
            "targeted_yongsan_pair",
            "sealed_grounded_suite",
            "bilingual_regression_suite",
            "agentic_stress_ladder",
            "frozen_random_holdout",
        ],
        "targeted_scenarios": list(TARGETED_SCENARIOS),
        "checks": checks,
        "ready": all(checks.values()),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        report = build_report()
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
