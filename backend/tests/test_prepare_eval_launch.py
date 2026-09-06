from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.prepare_eval_launch import build_report, environment_status, inventory


def test_inventory_covers_every_committed_scenario_family() -> None:
    suites = inventory(ROOT)

    assert [(suite.name, suite.scenario_count) for suite in suites] == [
        ("sealed_grounded", 8),
        ("bilingual_regression", 12),
        ("agentic_stress", 10),
    ]


def test_all_live_credentials_are_required() -> None:
    status = environment_status(
        {"HF_TOKEN": "configured", "OPENROUTER_API_KEY": "configured"}
    )

    assert status == {
        "required": {
            "HF_TOKEN": True,
            "OPENROUTER_API_KEY": True,
            "SEOULDOC_EVAL_AUTH_TOKEN": False,
        },
        "ready": False,
    }


def test_preflight_never_serializes_credentials(tmp_path: Path) -> None:
    secret = "private-value-that-must-not-appear"
    report = build_report(
        ROOT,
        {
            "HF_TOKEN": secret,
            "OPENROUTER_API_KEY": secret,
            "SEOULDOC_EVAL_AUTH_TOKEN": secret,
        },
    )

    assert secret not in json.dumps(report)


def test_cli_writes_the_same_report_that_it_prints(tmp_path: Path) -> None:
    output = tmp_path / "launch.json"
    environment = {
        "PATH": __import__("os").environ["PATH"],
        "HF_TOKEN": "configured",
        "OPENROUTER_API_KEY": "configured",
    }
    process = subprocess.run(
        [sys.executable, "scripts/prepare_eval_launch.py", "--output", str(output)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert process.returncode in {0, 1}
    assert json.loads(process.stdout) == json.loads(output.read_text())
    assert json.loads(process.stdout)["mode"] == "preflight_only_no_network_or_paid_calls"
