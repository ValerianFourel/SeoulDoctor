"""Run the progressive bilingual search ladder against a SeoulDoc API."""

from __future__ import annotations

import argparse
import os
import json
from pathlib import Path
import sys
import time

import requests


DEFAULT_ENDPOINT = os.getenv(
    "SEOULDOC_EVAL_ENDPOINT", "http://127.0.0.1:7860/chat"
)
CASES_PATH = Path(__file__).with_name("agentic_search_stress_cases.json")


def initial_state() -> dict:
    return {
        "specialty": None,
        "specialty_confidence": 0,
        "location": None,
        "latitude": None,
        "longitude": None,
        "address_korean": None,
        "district": None,
        "dong": None,
        "search_mode": None,
        "max_distance_km": 5,
        "travel_label": "Moderate",
        "travel_confidence": 0.5,
        "keywords": [],
        "hard_keywords": [],
        "negative_keywords": [],
        "negative_hard_keywords": [],
        "place_terms": [],
        "gender_terms": [],
        "disease_terms": [],
        "comment_terms": [],
        "is_general_search": False,
        "is_citywide_search": False,
        "hybrid_alpha": None,
        "query_intent": None,
        "suggested_alpha": None,
        "manual_search_mode": None,
        "language_pref": "English",
        "turn_count": 0,
        "ready_to_search": False,
        "search_executed": False,
        "conversation_phase": "greeting",
        "last_search_query": None,
        "last_results_count": None,
        "last_search_timestamp": None,
    }


def check_case(case: dict, body: dict) -> list[str]:
    failures: list[str] = []
    state = body.get("state", {})
    results = body.get("results", [])

    for field, expected in case.get("expected", {}).items():
        actual = state.get(field)
        if isinstance(expected, list):
            missing = [item for item in expected if item not in (actual or [])]
            if missing:
                failures.append(f"{field} missing {missing}; got {actual}")
        elif actual != expected:
            failures.append(f"{field} expected {expected!r}; got {actual!r}")

    required_methods = set(case.get("require_retrieval", []))
    observed_methods = {
        method
        for result in results
        for method in result.get("retrieval_methods", [])
    }
    if not required_methods.issubset(observed_methods):
        failures.append(
            f"retrieval methods missing {sorted(required_methods - observed_methods)}; "
            f"got {sorted(observed_methods)}"
        )

    if case.get("require_verbatim_comment"):
        has_verbatim = any(
            evidence.get("is_verbatim") is True
            for result in results
            for evidence in result.get("retrieval_evidence", [])
        )
        if not has_verbatim:
            failures.append("no actual/verbatim comment returned by comment-level BM25")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--minimum-level", type=int, default=1)
    parser.add_argument("--maximum-level", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument(
        "--auth-token",
        default=os.getenv("SEOULDOC_EVAL_AUTH_TOKEN"),
        help="Bearer token for the private application; defaults to SEOULDOC_EVAL_AUTH_TOKEN.",
    )
    args = parser.parse_args()

    cases = json.loads(CASES_PATH.read_text())
    selected = [case for case in cases if args.minimum_level <= case["level"] <= args.maximum_level]
    failed = 0

    for case in selected:
        started = time.perf_counter()
        headers = {"Origin": "https://www.seouldoc.io"}
        if args.auth_token:
            if "\r" in args.auth_token or "\n" in args.auth_token:
                raise SystemExit("--auth-token must not contain line breaks")
            headers["Authorization"] = f"Bearer {args.auth_token}"
        response = requests.post(
            args.endpoint,
            json={"message": case["query"], "current_state": initial_state()},
            headers=headers,
            timeout=args.timeout,
        )
        elapsed = time.perf_counter() - started
        response.raise_for_status()
        body = response.json()
        failures = check_case(case, body)
        status = "PASS" if not failures else "FAIL"
        print(f"[{status}] L{case['level']} {case['language']} ({elapsed:.2f}s): {case['query']}")
        print(
            "  facets=",
            json.dumps(
                {
                    key: body.get("state", {}).get(key)
                    for key in ("specialty", "place_terms", "gender_terms", "disease_terms", "comment_terms", "extraction_source", "extraction_error")
                },
                ensure_ascii=False,
            ),
        )
        for failure in failures:
            print(f"  - {failure}")
        failed += bool(failures)

    print(f"\n{len(selected) - failed}/{len(selected)} cases passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
