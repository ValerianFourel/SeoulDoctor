"""Run the grounded bilingual casebook and enforce every pass policy.

The patient messages come directly from each public scenario card. This keeps
staged refinements and code switches deterministic while a Luna agent owns the
run. Every app response, retrieval trace, and selected review remains in the
saved journey artifact. Qwen 3.8-27B judges the saved record only after the
deterministic report exists.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Optional, Sequence


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from grounded_journey_grader import (  # noqa: E402
    _load_transit_observation,
    grade_pair,
    grade_scenario,
)
from llm_client import build_openrouter_client  # noqa: E402
from patient_journey import AUTH_TOKEN_ENV, atomic_write, run_journey  # noqa: E402
from qwen_likert_judge import (  # noqa: E402
    MODEL as QWEN_JUDGE_MODEL,
    _evaluation_policy,
    build_messages,
    project_journey_for_review,
    request_rating,
    scenario_passes,
    weighted_mean,
)


DEFAULT_CASEBOOK = Path(__file__).with_name("grounded_bilingual_scenarios.json")
DEFAULT_ENDPOINT = "http://127.0.0.1:7860/chat"
META_INSTRUCTION_PREFIXES = (
    "after the first reply",
    "then say",
    "첫 답변 뒤",
    "다음에는 이렇게 말하세요",
    "그다음 이렇게 말하세요",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_casebook(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("scenarios"), list):
        raise ValueError("casebook must contain a scenarios list")
    if not isinstance(payload.get("likert_policy"), dict):
        raise ValueError("casebook must contain a likert_policy object")
    return payload


def scenario_messages(scenario: Mapping[str, Any]) -> list[str]:
    card = scenario.get("public_patient_card")
    if not isinstance(card, Mapping):
        raise ValueError(f"scenario {scenario.get('id')} has no public patient card")
    opening = card.get("opening_message")
    staged = card.get("staged_requests")
    if not isinstance(opening, str) or not opening.strip():
        raise ValueError(f"scenario {scenario.get('id')} has no opening message")
    if not isinstance(staged, list) or not all(
        isinstance(message, str) and message.strip() for message in staged
    ):
        raise ValueError(f"scenario {scenario.get('id')} has invalid staged requests")
    messages = [opening.strip(), *(message.strip() for message in staged)]
    for message in messages:
        normalized = message.casefold()
        if normalized.startswith(META_INSTRUCTION_PREFIXES):
            raise ValueError(
                f"scenario {scenario.get('id')} sends evaluator narration to the app"
            )
    oracle = scenario.get("oracle")
    expected = oracle.get("expected_turn_count") if isinstance(oracle, Mapping) else None
    if expected != len(messages):
        raise ValueError(
            f"scenario {scenario.get('id')} declares {expected} turns but has "
            f"{len(messages)} patient messages"
        )
    return messages


def message_matches_language(message: str, language: str) -> bool:
    hangul = sum("가" <= character <= "힣" for character in message)
    latin = sum(character.isascii() and character.isalpha() for character in message)
    if language == "Korean":
        return hangul >= 4 and hangul > latin
    if language == "English":
        return latin >= 4 and latin > hangul
    return False


def validate_casebook(casebook: Mapping[str, Any]) -> dict[str, Any]:
    scenarios = casebook.get("scenarios", [])
    ids = [str(scenario.get("id")) for scenario in scenarios]
    failures: list[str] = []
    pairs: dict[str, list[str]] = defaultdict(list)
    scenario_checks = []
    for scenario in scenarios:
        scenario_id = str(scenario.get("id"))
        try:
            messages = scenario_messages(scenario)
        except ValueError as error:
            failures.append(str(error))
            continue
        oracle = scenario.get("oracle")
        oracle = oracle if isinstance(oracle, Mapping) else {}
        expected_languages = oracle.get("expected_response_languages")
        if (
            not isinstance(expected_languages, list)
            or len(expected_languages) != len(messages)
        ):
            failures.append(
                f"scenario {scenario_id} lacks one expected language per turn"
            )
            continue
        language_checks = [
            message_matches_language(message, str(language))
            for message, language in zip(messages, expected_languages)
        ]
        if not all(language_checks):
            failures.append(f"scenario {scenario_id} patient language sequence fails")
        implicit_turns = oracle.get("implicit_default_radius_turns", [])
        radii = oracle.get("turn_radius_expectations_km")
        if implicit_turns and (
            not isinstance(radii, list)
            or any(float(radii[index - 1]) != 5.0 for index in implicit_turns)
        ):
            failures.append(
                f"scenario {scenario_id} implicit local radius is not 5 km"
            )
        pair_id = scenario.get("pair_id")
        if isinstance(pair_id, str):
            pairs[pair_id].append(scenario_id)
        scenario_checks.append({
            "scenario_id": scenario_id,
            "message_count": len(messages),
            "language_sequence": expected_languages,
            "language_checks": language_checks,
            "implicit_default_radius_turns": implicit_turns,
        })
    if len(ids) != len(set(ids)):
        failures.append("scenario IDs are not unique")
    for pair_id, pair_ids in pairs.items():
        if len(pair_ids) != 2:
            failures.append(f"pair {pair_id} does not contain exactly two scenarios")
    return {
        "passed": not failures,
        "scenario_count": len(scenarios),
        "pair_count": len(pairs),
        "failures": failures,
        "scenario_checks": scenario_checks,
    }


def _scenario_by_id(casebook: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(item["id"]): item for item in casebook["scenarios"]}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"saved artifact is not an object: {path}")
    return payload


def _judge_scenario(
    client: Any,
    scenario: Mapping[str, Any],
    journey: Mapping[str, Any],
    deterministic: Mapping[str, Any],
    casebook: Mapping[str, Any],
    casebook_path: Path,
) -> dict[str, Any]:
    policy = _evaluation_policy(casebook, scenario)
    map_observation = _load_transit_observation(casebook_path, scenario)
    messages = build_messages(
        scenario,
        journey,
        deterministic,
        policy,
        map_observation,
    )
    target = scenario.get("oracle", {}).get("reverse_target")
    target_id = str(target.get("place_id")) if isinstance(target, Mapping) else None
    projected = project_journey_for_review(journey, target_id)
    review = request_rating(
        client,
        messages,
        policy,
        projected["citation_catalog"],
    )
    scores = review["rating"]["scores"]
    hard_pass = bool(deterministic.get("hard_pass"))
    review["weighted_mean"] = weighted_mean(scores, policy)
    review["minimum_dimension"] = min(scores.values())
    review["deterministic_hard_pass"] = hard_pass
    review["passed"] = scenario_passes(scores, hard_pass, policy)
    return review


def aggregate_pair(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    left_review: Mapping[str, Any],
    right_review: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    deterministic = grade_pair(left, right)
    rank_difference = deterministic.get("target_rank_difference")
    targets_required = (
        left.get("target", {}).get("required") is True
        and right.get("target", {}).get("required") is True
    )
    if targets_required:
        target_hit = all(
            isinstance(report.get("target", {}).get("presented_rank"), int)
            and report["target"]["presented_rank"] <= 5
            for report in (left, right)
        )
        rank_difference_ok = (
            isinstance(rank_difference, int)
            and rank_difference <= int(policy["maximum_target_rank_difference"])
        )
    else:
        target_hit = True
        rank_difference_ok = True
    gates = {
        "deterministic_pair": bool(deterministic.get("passed")),
        "both_scenarios_pass": (
            bool(left_review.get("passed")) and bool(right_review.get("passed"))
        ),
        "target_hit_at_5": (
            target_hit if policy.get("target_hit_at_5") else True
        ),
        "target_rank_difference": rank_difference_ok,
        "top_five_jaccard": (
            float(deterministic.get("top_five_jaccard", 0.0))
            >= float(policy["minimum_top_five_jaccard"])
        ),
    }
    return {
        "pair_id": left.get("pair_id"),
        "left_scenario_id": left.get("scenario_id"),
        "right_scenario_id": right.get("scenario_id"),
        "passed": all(gates.values()),
        "gates": gates,
        "deterministic": deterministic,
    }


def aggregate_suite(
    casebook: Mapping[str, Any],
    deterministic_reports: Mapping[str, Mapping[str, Any]],
    reviews: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    scenarios = _scenario_by_id(casebook)
    count = len(scenarios)
    hard_passes = sum(
        bool(deterministic_reports.get(item, {}).get("hard_pass"))
        for item in scenarios
    )
    scenario_passes_count = sum(
        bool(reviews.get(item, {}).get("passed")) for item in scenarios
    )
    means = [
        float(reviews[item]["weighted_mean"])
        for item in scenarios
        if isinstance(reviews.get(item, {}).get("weighted_mean"), (int, float))
    ]
    language_means: dict[str, float] = {}
    for language in ("English", "Korean"):
        values = [
            float(reviews[item]["weighted_mean"])
            for item, scenario in scenarios.items()
            if scenario.get("source_language") == language
            and isinstance(reviews.get(item, {}).get("weighted_mean"), (int, float))
        ]
        language_means[language] = sum(values) / len(values) if values else 0.0
    target_ranks = [
        report.get("target", {}).get("presented_rank")
        for report in deterministic_reports.values()
        if report.get("target", {}).get("required") is True
    ]
    target_hits = sum(
        isinstance(rank, int) and not isinstance(rank, bool) and rank <= 5
        for rank in target_ranks
    )
    reciprocal_ranks = [
        1.0 / rank
        for rank in target_ranks
        if isinstance(rank, int) and not isinstance(rank, bool) and rank > 0
    ]
    metrics = {
        "scenario_count": count,
        "hard_gate_rate": hard_passes / count if count else 0.0,
        "scenario_pass_rate": scenario_passes_count / count if count else 0.0,
        "weighted_mean": sum(means) / len(means) if len(means) == count else 0.0,
        "language_means": language_means,
        "language_gap": abs(language_means["English"] - language_means["Korean"]),
        "reverse_target_hit_at_5": (
            target_hits / len(target_ranks) if target_ranks else 0.0
        ),
        "mean_reciprocal_rank": (
            sum(reciprocal_ranks) / len(target_ranks) if target_ranks else 0.0
        ),
    }
    policy = casebook["likert_policy"]["suite_pass"]
    maximum_language_gap = float(policy["maximum_language_gap"])
    gates = {
        "hard_gate_rate": metrics["hard_gate_rate"] >= float(policy["hard_gate_rate"]),
        "scenario_pass_rate": (
            metrics["scenario_pass_rate"]
            >= float(policy["minimum_scenario_pass_rate"])
        ),
        "weighted_mean": (
            metrics["weighted_mean"] >= float(policy["minimum_weighted_mean"])
        ),
        "language_gap": (
            metrics["language_gap"] < maximum_language_gap
            or math.isclose(
                metrics["language_gap"], maximum_language_gap, abs_tol=1e-12
            )
        ),
        "reverse_target_hit_at_5": (
            metrics["reverse_target_hit_at_5"]
            >= float(policy["reverse_target_hit_at_5"])
        ),
        "mean_reciprocal_rank": (
            metrics["mean_reciprocal_rank"] >= float(policy["minimum_mrr"])
        ),
    }
    return {"passed": all(gates.values()), "metrics": metrics, "gates": gates}


def run_suite(
    *,
    run_id: str,
    run_dir: Path,
    casebook_path: Path,
    endpoint: str,
    actor_label: str,
    selected_ids: Sequence[str] = (),
    timeout: float = 180.0,
    judge: bool = True,
    auth_token: Optional[str] = None,
) -> dict[str, Any]:
    casebook = load_casebook(casebook_path)
    validation = validate_casebook(casebook)
    report_path = run_dir / "suite_report.json"
    report: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "trigger": "user_requested_bilingual_grounded_evaluation",
        "status": "preflight",
        "passed": False,
        "started_at": utc_now(),
        "finished_at": None,
        "configuration": {
            "endpoint": endpoint,
            "casebook": str(casebook_path.resolve()),
            "patient_actor": actor_label,
            "patient_delivery": "exact_public_card_messages",
            "app_model": "openai/gpt-oss-120b",
            "judge_model": QWEN_JUDGE_MODEL if judge else None,
        },
        "preflight": validation,
        "scenario_results": [],
        "pair_results": [],
        "suite": None,
        "errors": [],
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_write(report_path, report)
    if not validation["passed"]:
        report["status"] = "preflight_failed"
        report["finished_at"] = utc_now()
        atomic_write(report_path, report)
        return report

    scenarios = list(casebook["scenarios"])
    if selected_ids:
        wanted = set(selected_ids)
        unknown = sorted(wanted - {str(item["id"]) for item in scenarios})
        if unknown:
            raise ValueError("unknown scenario IDs: " + ", ".join(unknown))
        scenarios = [item for item in scenarios if item["id"] in wanted]
    if selected_ids and len(scenarios) != len(casebook["scenarios"]):
        judge = False

    qwen_client = build_openrouter_client() if judge else None
    if judge and qwen_client is None:
        report["status"] = "preflight_failed"
        report["errors"].append({
            "stage": "qwen_client",
            "message": "OPENROUTER_API_KEY is not configured",
        })
        report["finished_at"] = utc_now()
        atomic_write(report_path, report)
        return report

    report["status"] = "running"
    atomic_write(report_path, report)
    deterministic_reports: dict[str, Mapping[str, Any]] = {}
    reviews: dict[str, Mapping[str, Any]] = {}
    for scenario in scenarios:
        scenario_id = str(scenario["id"])
        journey_path = run_dir / f"journey-{scenario_id}.json"
        deterministic_path = run_dir / f"deterministic-{scenario_id}.json"
        review_path = run_dir / f"qwen-{scenario_id}.json"
        item: dict[str, Any] = {
            "scenario_id": scenario_id,
            "journey": str(journey_path),
            "deterministic_report": str(deterministic_path),
            "qwen_review": str(review_path) if judge else None,
            "status": "running",
            "passed": False,
        }
        report["scenario_results"].append(item)
        atomic_write(report_path, report)
        try:
            if journey_path.is_file():
                journey = _read_json(journey_path)
                if (
                    journey.get("status") != "finished"
                    or len(journey.get("turns", [])) != len(scenario_messages(scenario))
                ):
                    raise ValueError(
                        f"existing journey is incomplete and will not be replayed: {journey_path}"
                    )
                item["journey_reused"] = True
            else:
                journey = run_journey(
                    journey_path,
                    endpoint,
                    language=str(scenario["source_language"]),
                    messages=scenario_messages(scenario),
                    timeout=timeout,
                    auth_token=auth_token,
                )
                item["journey_reused"] = False
            deterministic = grade_scenario(
                scenario,
                journey,
                _load_transit_observation(casebook_path, scenario),
            )
            atomic_write(deterministic_path, deterministic)
            deterministic_reports[scenario_id] = deterministic
            item["hard_pass"] = bool(deterministic["hard_pass"])
            if judge:
                if review_path.is_file():
                    review = _read_json(review_path)
                    if review.get("model") != QWEN_JUDGE_MODEL:
                        raise ValueError(
                            f"saved review uses the wrong model: {review_path}"
                        )
                    item["qwen_review_reused"] = True
                else:
                    review = _judge_scenario(
                        qwen_client,
                        scenario,
                        journey,
                        deterministic,
                        casebook,
                        casebook_path,
                    )
                    atomic_write(review_path, review)
                    item["qwen_review_reused"] = False
                reviews[scenario_id] = review
                item["weighted_mean"] = review["weighted_mean"]
                item["minimum_dimension"] = review["minimum_dimension"]
                item["passed"] = bool(review["passed"])
            else:
                item["passed"] = bool(deterministic["hard_pass"])
            item["status"] = "finished"
        except Exception as error:
            item["status"] = "error"
            item["error"] = {"type": type(error).__name__, "message": str(error)[:800]}
            report["errors"].append({"scenario_id": scenario_id, **item["error"]})
        atomic_write(report_path, report)

    if judge and len(deterministic_reports) == len(casebook["scenarios"]) and len(reviews) == len(casebook["scenarios"]):
        by_pair: dict[str, list[str]] = defaultdict(list)
        for scenario in casebook["scenarios"]:
            by_pair[str(scenario["pair_id"])].append(str(scenario["id"]))
        pair_policy = casebook["likert_policy"]["pair_pass"]
        for pair_id, pair_ids in by_pair.items():
            left_id, right_id = pair_ids
            pair = aggregate_pair(
                deterministic_reports[left_id],
                deterministic_reports[right_id],
                reviews[left_id],
                reviews[right_id],
                pair_policy,
            )
            pair["pair_id"] = pair_id
            report["pair_results"].append(pair)
        report["suite"] = aggregate_suite(casebook, deterministic_reports, reviews)
        report["passed"] = (
            report["suite"]["passed"]
            and all(pair["passed"] for pair in report["pair_results"])
        )
    else:
        report["passed"] = (
            not report["errors"]
            and bool(report["scenario_results"])
            and all(item["passed"] for item in report["scenario_results"])
        )
    report["status"] = "finished" if not report["errors"] else "finished_with_errors"
    report["finished_at"] = utc_now()
    atomic_write(report_path, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run and grade all grounded English/Korean SeoulDoc scenarios."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--casebook", type=Path, default=DEFAULT_CASEBOOK)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--actor-label", default="gpt-5.6-luna-medium")
    parser.add_argument("--scenario", action="append", default=[])
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--skip-qwen", action="store_true")
    parser.add_argument(
        "--auth-token",
        default=os.getenv(AUTH_TOKEN_ENV),
        help=f"Bearer token for a private endpoint; defaults to {AUTH_TOKEN_ENV}.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.run_id):
        raise SystemExit("--run-id may contain only letters, numbers, dots, dashes, and underscores")
    if args.timeout <= 0 or not math.isfinite(args.timeout):
        raise SystemExit("--timeout must be a positive finite number")
    report = run_suite(
        run_id=args.run_id,
        run_dir=args.run_dir,
        casebook_path=args.casebook,
        endpoint=args.endpoint,
        actor_label=args.actor_label,
        selected_ids=args.scenario,
        timeout=args.timeout,
        judge=not args.skip_qwen,
        auth_token=args.auth_token,
    )
    print(json.dumps({
        "run_id": report["run_id"],
        "status": report["status"],
        "passed": report["passed"],
        "scenarios_finished": sum(
            item["status"] == "finished" for item in report["scenario_results"]
        ),
        "scenario_count": len(report["scenario_results"]),
        "errors": report["errors"],
        "suite": report["suite"],
    }, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
