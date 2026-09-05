"""Deterministic gates for map-grounded bilingual SeoulDoc journeys."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from math import atan2, cos, radians, sin, sqrt
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Optional, Sequence

import requests


EXPECTED_MODEL = "openai/gpt-oss-120b"
FACET_FIELDS = (
    "keywords",
    "hard_keywords",
    "negative_keywords",
    "negative_hard_keywords",
    "place_terms",
    "gender_terms",
    "disease_terms",
    "comment_terms",
)
RESULT_BEARING_ACTIONS = {
    "search_facilities",
    "search_indexed_evidence",
    "search_multilingual_comments",
    "select_comment_evidence",
}


def haversine_km(
    latitude_one: float,
    longitude_one: float,
    latitude_two: float,
    longitude_two: float,
) -> float:
    """Calculate great-circle distance without importing application code."""
    lat_one = radians(float(latitude_one))
    lon_one = radians(float(longitude_one))
    lat_two = radians(float(latitude_two))
    lon_two = radians(float(longitude_two))
    delta_lat = lat_two - lat_one
    delta_lon = lon_two - lon_one
    a = (
        sin(delta_lat / 2) ** 2
        + cos(lat_one) * cos(lat_two) * sin(delta_lon / 2) ** 2
    )
    return 6371.0 * 2 * atan2(sqrt(a), sqrt(1 - a))


def parse_kakao_transit(payload: Mapping[str, Any]) -> dict[str, Any]:
    routes = payload.get("routes")
    if not isinstance(routes, list) or not routes:
        raise ValueError("Kakao response has no transit routes")
    route = routes[0]
    if not isinstance(route, Mapping):
        raise ValueError("Kakao route is not an object")
    properties = route.get("properties")
    summary = route.get("summary")
    if isinstance(properties, Mapping):
        values = properties
    elif isinstance(summary, Mapping):
        values = summary
    else:
        values = route

    def integer(*keys: str) -> Optional[int]:
        for key in keys:
            value = values.get(key)
            if isinstance(value, (int, float)):
                return int(value)
        return None

    distance = integer("totalDistance", "distance")
    duration = integer("totalTime", "duration")
    if distance is None or duration is None:
        raise ValueError("Kakao route omits distance or duration")
    fare = values.get("fare")
    if isinstance(fare, Mapping):
        fare = fare.get("value")
    fare_krw = int(fare) if isinstance(fare, (int, float)) else integer(
        "totalFare"
    )
    return {
        "mode": str(route.get("type") or values.get("type") or "PUBLIC_TRANSIT"),
        "route_distance_m": distance,
        "duration_s": duration,
        "transfers": integer("transferCount", "transfers"),
        "fare_krw": fare_krw,
    }


def fetch_kakao_transit(
    origin: Mapping[str, float],
    destination: Mapping[str, float],
    *,
    api_key: str,
    timeout: float = 30.0,
    session: Optional[Any] = None,
) -> dict[str, Any]:
    if not api_key.strip():
        raise ValueError("Kakao API key is required")
    http = session or requests.Session()
    endpoint = "https://dapi.kakao.com/v2/routing/publictraffic"
    response = http.get(
        endpoint,
        params={
            "start_x": float(origin["longitude"]),
            "start_y": float(origin["latitude"]),
            "end_x": float(destination["longitude"]),
            "end_y": float(destination["latitude"]),
        },
        headers={"Authorization": f"KakaoAK {api_key}"},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    parsed = parse_kakao_transit(payload)
    request_facts = {
        "origin": dict(origin),
        "destination": dict(destination),
    }
    return {
        "provider": "kakao",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        **request_facts,
        **parsed,
        "request_sha256": sha256(
            json.dumps(request_facts, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "response_sha256": sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
    }


def _gate(identifier: str, passed: bool, **facts: Any) -> dict[str, Any]:
    return {"id": identifier, "passed": bool(passed), "facts": facts}


def _is_subsequence(observed: Sequence[str], required: Sequence[str]) -> bool:
    iterator = iter(observed)
    return all(any(item == wanted for item in iterator) for wanted in required)


def _required_action_records(
    trace: Sequence[Mapping[str, Any]],
    required: Sequence[str],
) -> Optional[list[Mapping[str, Any]]]:
    """Return the first ordered trace record for every required action."""
    selected: list[Mapping[str, Any]] = []
    start = 0
    for wanted in required:
        for index in range(start, len(trace)):
            record = trace[index]
            if str(record.get("action")) == wanted:
                selected.append(record)
                start = index + 1
                break
        else:
            return None
    return selected


def _turn_body(turn: Any) -> Mapping[str, Any]:
    response = turn.get("response") if isinstance(turn, Mapping) else None
    body = response.get("body") if isinstance(response, Mapping) else None
    return body if isinstance(body, Mapping) else {}


def _final_body(artifact: Mapping[str, Any]) -> Mapping[str, Any]:
    turns = artifact.get("turns")
    if not isinstance(turns, list) or not turns:
        return {}
    return _turn_body(turns[-1])


def _language_counts(text: Any) -> tuple[int, int]:
    value = str(text or "")
    hangul_ranges = (
        ("\u1100", "\u11ff"),
        ("\u3130", "\u318f"),
        ("\ua960", "\ua97f"),
        ("\uac00", "\ud7a3"),
        ("\ud7b0", "\ud7ff"),
    )
    hangul = sum(
        any(start <= character <= end for start, end in hangul_ranges)
        for character in value
    )
    latin = sum(character.isascii() and character.isalpha() for character in value)
    return hangul, latin


def _matches_language(text: Any, language: str) -> bool:
    hangul, latin = _language_counts(text)
    if language == "Korean":
        return hangul >= 4 and hangul > latin
    return latin >= 4 and latin > hangul


def _evidence_records(
    target_oracle: Mapping[str, Any],
    role: str,
) -> list[Mapping[str, Any]]:
    requirements = target_oracle.get("evidence_requirements")
    if isinstance(requirements, Mapping):
        records = requirements.get(role, [])
        if not isinstance(records, list) or not all(
            isinstance(item, Mapping)
            and isinstance(item.get("evidence_id"), str)
            and bool(item.get("evidence_id"))
            and isinstance(item.get("supports", []), list)
            for item in records
        ):
            raise ValueError(f"reverse_target evidence_requirements.{role} is invalid")
        return list(records)

    legacy_ids = target_oracle.get("evidence_ids", [])
    if not isinstance(legacy_ids, list):
        raise ValueError("reverse_target evidence_ids must be a list")
    chosen = legacy_ids[:1] if role == "decisive" else legacy_ids[1:]
    return [
        {"evidence_id": str(evidence_id), "supports": []}
        for evidence_id in chosen
        if str(evidence_id)
    ]


def _facet_text(state: Mapping[str, Any], fields: Iterable[str]) -> str:
    values: list[str] = []
    for field in fields:
        value = state.get(field)
        if isinstance(value, list):
            values.extend(str(item) for item in value)
    return " ".join(values).casefold()


def _rank(
    records: Sequence[Mapping[str, Any]],
    target: str,
    rank_field: str,
) -> Optional[int]:
    for record in records:
        if str(record.get("place_id")) == target:
            value = record.get(rank_field)
            if isinstance(value, (int, float)) and int(value) >= 1:
                return int(value)
            return None
    return None


def grade_scenario(
    scenario: Mapping[str, Any],
    artifact: Mapping[str, Any],
    transit_observation: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    oracle = scenario["oracle"]
    turns = artifact.get("turns")
    turns = turns if isinstance(turns, list) else []
    body = _final_body(artifact)
    state = body.get("state") if isinstance(body.get("state"), Mapping) else {}
    results = body.get("results") if isinstance(body.get("results"), list) else []
    health = artifact.get("health")
    health = health if isinstance(health, Mapping) else {}

    gates: list[dict[str, Any]] = []
    expected_turn_count = oracle.get("expected_turn_count")
    if not isinstance(expected_turn_count, int) or expected_turn_count < 1:
        raise ValueError("oracle.expected_turn_count must be a positive integer")
    expected_languages = oracle.get("expected_response_languages")
    if (
        not isinstance(expected_languages, list)
        or len(expected_languages) != expected_turn_count
        or any(language not in {"English", "Korean"} for language in expected_languages)
    ):
        raise ValueError(
            "oracle.expected_response_languages must define every staged turn"
        )

    journey_ok = (
        artifact.get("status") == "finished"
        and isinstance(artifact.get("finished_at"), str)
        and bool(str(artifact.get("finished_at")).strip())
        and artifact.get("language") == scenario.get("source_language")
        and len(turns) == expected_turn_count
    )
    gates.append(_gate(
        "journey_integrity",
        journey_ok,
        status=artifact.get("status"),
        source_language=scenario.get("source_language"),
        journey_language=artifact.get("language"),
        expected_turn_count=expected_turn_count,
        observed_turn_count=len(turns),
    ))

    turn_failures: list[dict[str, Any]] = []
    observed_languages: list[dict[str, Any]] = []
    run_ids: list[str] = []
    run_failures: list[dict[str, Any]] = []
    for index, turn in enumerate(turns, start=1):
        request = turn.get("request") if isinstance(turn, Mapping) else None
        response = turn.get("response") if isinstance(turn, Mapping) else None
        status_code = response.get("status_code") if isinstance(response, Mapping) else None
        turn_body = _turn_body(turn)
        message = request.get("message") if isinstance(request, Mapping) else None
        if (
            not isinstance(message, str)
            or not message.strip()
            or not isinstance(status_code, int)
            or not 200 <= status_code < 300
            or not turn_body
        ):
            turn_failures.append({
                "turn": index,
                "status_code": status_code,
                "request_present": isinstance(message, str) and bool(message.strip()),
                "body_present": bool(turn_body),
            })

        expected_language = (
            expected_languages[index - 1]
            if index <= len(expected_languages)
            else expected_languages[-1]
        )
        turn_text = turn_body.get("response")
        language_ok = _matches_language(turn_text, expected_language)
        hangul, latin = _language_counts(turn_text)
        observed_languages.append({
            "turn": index,
            "expected": expected_language,
            "passed": language_ok,
            "hangul_characters": hangul,
            "latin_characters": latin,
        })

        turn_state = turn_body.get("state")
        turn_state = turn_state if isinstance(turn_state, Mapping) else {}
        run_id = turn_state.get("last_retrieval_run_id")
        metadata = turn_state.get("last_retrieval_metadata")
        metadata = metadata if isinstance(metadata, Mapping) else {}
        if (
            not isinstance(run_id, str)
            or not run_id
            or metadata.get("run_id") != run_id
        ):
            run_failures.append({
                "turn": index,
                "run_id_present": isinstance(run_id, str) and bool(run_id),
                "metadata_run_id_matches": metadata.get("run_id") == run_id,
            })
        else:
            run_ids.append(run_id)

    gates.append(_gate(
        "turn_responses",
        len(turns) >= expected_turn_count and not turn_failures,
        failures=turn_failures,
    ))
    language_turns_ok = (
        len(turns) >= expected_turn_count
        and all(item["passed"] for item in observed_languages)
    )
    gates.append(_gate(
        "turn_languages",
        language_turns_ok,
        turns=observed_languages,
    ))
    duplicate_run_ids = sorted({run_id for run_id in run_ids if run_ids.count(run_id) > 1})
    gates.append(_gate(
        "retrieval_run_integrity",
        len(turns) >= expected_turn_count
        and len(run_ids) == len(turns)
        and not run_failures
        and not duplicate_run_ids,
        failures=run_failures,
        duplicate_run_ids=duplicate_run_ids,
    ))

    turn_radius_expectations = oracle.get("turn_radius_expectations_km")
    if turn_radius_expectations is not None:
        if (
            not isinstance(turn_radius_expectations, list)
            or len(turn_radius_expectations) != expected_turn_count
            or any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or value <= 0
                for value in turn_radius_expectations
            )
        ):
            raise ValueError(
                "oracle.turn_radius_expectations_km must define every staged turn"
            )
        radius_failures = []
        for index, expected in enumerate(turn_radius_expectations, start=1):
            turn_body = _turn_body(turns[index - 1]) if index <= len(turns) else {}
            turn_state = turn_body.get("state")
            turn_state = turn_state if isinstance(turn_state, Mapping) else {}
            observed = turn_state.get("max_distance_km")
            if (
                not isinstance(observed, (int, float))
                or isinstance(observed, bool)
                or abs(float(observed) - float(expected)) > 0.01
            ):
                radius_failures.append({
                    "turn": index,
                    "expected_km": expected,
                    "observed_km": observed,
                })
        implicit_turns = oracle.get("implicit_default_radius_turns", [])
        if (
            not isinstance(implicit_turns, list)
            or any(
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 1
                or value > expected_turn_count
                for value in implicit_turns
            )
        ):
            raise ValueError(
                "oracle.implicit_default_radius_turns must contain valid turn numbers"
            )
        gates.append(_gate(
            "staged_distance",
            len(turns) == expected_turn_count and not radius_failures,
            expectations_km=turn_radius_expectations,
            implicit_default_turns=implicit_turns,
            failures=radius_failures,
        ))

    health_ok = (
        health.get("status") == "ok"
        and health.get("model_provider") == "openrouter"
        and health.get("model") == EXPECTED_MODEL
        and health.get("agent_model") == EXPECTED_MODEL
        and isinstance(health.get("facilities"), int)
        and health.get("facilities", 0) > 0
        and health.get("facilities") == health.get("vector_documents")
        and isinstance(health.get("raw_reviews"), int)
        and health.get("raw_reviews", 0) > 0
    )
    gates.append(_gate("runtime_health", health_ok))

    response_schema_ok = (
        isinstance(body.get("response"), str)
        and isinstance(state, Mapping)
        and isinstance(results, list)
        and all(
            result.get("entity_type") == "facility"
            and result.get("final_rank") == index
            for index, result in enumerate(results, start=1)
        )
    )
    gates.append(_gate("response_schema", response_schema_ok, result_count=len(results)))

    expected_specialty = oracle.get("expected_specialty")
    specialty_ok = not expected_specialty or state.get("specialty") == expected_specialty
    expected_radius = oracle.get("maximum_distance_km")
    radius_ok = (
        expected_radius is None
        or (
            isinstance(state.get("max_distance_km"), (int, float))
            and abs(float(state["max_distance_km"]) - float(expected_radius)) <= 0.01
        )
    )
    active_text = _facet_text(state, FACET_FIELDS)
    removed = [str(term) for term in oracle.get("removed_terms", [])]
    removed_ok = all(term.casefold() not in active_text for term in removed)
    negative_text = _facet_text(
        state,
        ("negative_keywords", "negative_hard_keywords"),
    )
    negative_terms = [str(term) for term in oracle.get("negative_terms", [])]
    expected_negative_ok = (
        not negative_terms
        or any(term.casefold() in negative_text for term in negative_terms)
    )
    excluded_negative = [
        str(term) for term in oracle.get("excluded_negative_terms", [])
    ]
    excluded_negative_ok = all(
        term.casefold() not in negative_text for term in excluded_negative
    )
    gates.append(_gate(
        "intent_and_refinement",
        specialty_ok
        and radius_ok
        and removed_ok
        and expected_negative_ok
        and excluded_negative_ok,
        specialty=state.get("specialty"),
        radius_km=state.get("max_distance_km"),
        removed_terms_absent=removed_ok,
        expected_negative_present=expected_negative_ok,
        excluded_negative_absent=excluded_negative_ok,
    ))

    origin = oracle.get("origin")
    location_error = None
    location_ok = True
    if isinstance(origin, Mapping):
        if isinstance(state.get("latitude"), (int, float)) and isinstance(
            state.get("longitude"), (int, float)
        ):
            location_error = haversine_km(
                origin["latitude"],
                origin["longitude"],
                state["latitude"],
                state["longitude"],
            )
            location_ok = location_error <= 0.15
        else:
            location_ok = False

    distance_failures: list[dict[str, Any]] = []
    state_has_coordinates = all(
        isinstance(state.get(field), (int, float))
        for field in ("latitude", "longitude")
    )
    for result in results:
        result_has_coordinates = all(
            isinstance(result.get(field), (int, float))
            for field in ("lat", "lon")
        )
        if not state_has_coordinates or not result_has_coordinates:
            distance_failures.append({
                "place_id": result.get("place_id"),
                "reason": "missing origin or facility coordinates",
            })
            continue
        calculated = haversine_km(
            state["latitude"],
            state["longitude"],
            result["lat"],
            result["lon"],
        )
        reported = result.get("distance_km")
        tolerance = max(0.02, calculated * 0.005)
        if not isinstance(reported, (int, float)):
            distance_failures.append({
                "place_id": result.get("place_id"),
                "reason": "missing reported distance",
                "calculated_km": round(calculated, 6),
            })
        elif abs(calculated - float(reported)) > tolerance:
            distance_failures.append({
                "place_id": result.get("place_id"),
                "reason": "haversine mismatch",
                "calculated_km": round(calculated, 6),
                "reported_km": reported,
            })
        if (
            oracle.get("strict_radius")
            and expected_radius is not None
            and calculated > float(expected_radius) + 0.01
        ):
            distance_failures.append({
                "place_id": result.get("place_id"),
                "reason": "outside strict radius",
                "calculated_km": round(calculated, 6),
            })
    gates.append(_gate(
        "geography_and_distance",
        location_ok and not distance_failures and bool(results),
        origin_error_km=(
            round(location_error, 6) if location_error is not None else None
        ),
        failures=distance_failures,
    ))

    metadata = state.get("last_retrieval_metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    run_id = state.get("last_retrieval_run_id")
    completion_ok = (
        isinstance(run_id, str)
        and bool(run_id)
        and metadata.get("run_id") == run_id
        and metadata.get("retrieval_status") == "complete"
        and metadata.get("coverage_assessed") is True
        and metadata.get("coverage_sufficient") is True
        and metadata.get("termination_reason") == "finish_search"
    )
    gates.append(_gate(
        "retrieval_completion",
        completion_ok,
        run_id_present=bool(run_id),
        status=metadata.get("retrieval_status"),
        termination_reason=metadata.get("termination_reason"),
    ))

    trace = state.get("last_retrieval_trace")
    trace = trace if isinstance(trace, list) else []
    trace_records = [item for item in trace if isinstance(item, Mapping)]
    actions = [
        str(item.get("action"))
        for item in trace
        if isinstance(item, Mapping) and item.get("action")
    ]
    required_actions = [
        str(action) for action in oracle.get("required_trace_actions", [])
    ]
    required_records = _required_action_records(trace_records, required_actions)
    unsuccessful_required_actions = []
    if required_records is not None:
        unsuccessful_required_actions = [
            str(record.get("action"))
            for record in required_records
            if record.get("action") in RESULT_BEARING_ACTIONS
            and (
                not isinstance(record.get("result_count"), int)
                or record.get("result_count", 0) <= 0
            )
        ]
    methods = {
        str(method)
        for result in results
        if isinstance(result, Mapping)
        for method in (
            result.get("retrieval_methods", [])
            if isinstance(result.get("retrieval_methods"), list)
            else []
        )
    }
    required_methods = {
        str(method) for method in oracle.get("required_methods", [])
    }
    required_scripts = {
        str(script) for script in oracle.get("required_comment_query_scripts", [])
    }
    comment_terms = [
        str(term)
        for record in trace_records
        if record.get("action") == "search_multilingual_comments"
        for term in (
            record.get("arguments", {}).get("query_terms", [])
            if isinstance(record.get("arguments"), Mapping)
            and isinstance(record.get("arguments", {}).get("query_terms"), list)
            else []
        )
    ]
    observed_scripts = set()
    if any(any("\uac00" <= character <= "\ud7a3" for character in term) for term in comment_terms):
        observed_scripts.add("Hangul")
    if any(any(character.isascii() and character.isalpha() for character in term) for term in comment_terms):
        observed_scripts.add("Latin")
    finish_is_last = bool(required_actions) and bool(actions) and actions[-1] == "finish_search"
    orchestration_ok = (
        required_records is not None
        and not unsuccessful_required_actions
        and finish_is_last
        and required_methods.issubset(methods)
        and required_scripts.issubset(observed_scripts)
    )
    gates.append(_gate(
        "retrieval_orchestration",
        orchestration_ok,
        actions=actions,
        missing_methods=sorted(required_methods - methods),
        unsuccessful_required_actions=unsuccessful_required_actions,
        finish_is_last=finish_is_last,
        missing_comment_query_scripts=sorted(required_scripts - observed_scripts),
    ))

    if oracle.get("required_trace_actions_each_turn") is True:
        per_turn_failures = []
        for index, turn in enumerate(turns, start=1):
            turn_body = _turn_body(turn)
            turn_state = turn_body.get("state")
            turn_state = turn_state if isinstance(turn_state, Mapping) else {}
            turn_results = turn_body.get("results")
            turn_results = turn_results if isinstance(turn_results, list) else []
            turn_trace = turn_state.get("last_retrieval_trace")
            turn_trace = turn_trace if isinstance(turn_trace, list) else []
            turn_records = [
                item for item in turn_trace if isinstance(item, Mapping)
            ]
            turn_actions = [
                str(item.get("action"))
                for item in turn_records
                if item.get("action")
            ]
            selected = _required_action_records(turn_records, required_actions)
            unsuccessful = []
            if selected is not None:
                unsuccessful = [
                    str(record.get("action"))
                    for record in selected
                    if record.get("action") in RESULT_BEARING_ACTIONS
                    and (
                        not isinstance(record.get("result_count"), int)
                        or record.get("result_count", 0) <= 0
                    )
                ]
            turn_methods = {
                str(method)
                for result in turn_results
                if isinstance(result, Mapping)
                for method in (
                    result.get("retrieval_methods", [])
                    if isinstance(result.get("retrieval_methods"), list)
                    else []
                )
            }
            turn_comment_terms = [
                str(term)
                for record in turn_records
                if record.get("action") == "search_multilingual_comments"
                for term in (
                    record.get("arguments", {}).get("query_terms", [])
                    if isinstance(record.get("arguments"), Mapping)
                    and isinstance(
                        record.get("arguments", {}).get("query_terms"), list
                    )
                    else []
                )
            ]
            turn_scripts = set()
            if any(
                any("가" <= character <= "힣" for character in term)
                for term in turn_comment_terms
            ):
                turn_scripts.add("Hangul")
            if any(
                any(character.isascii() and character.isalpha() for character in term)
                for term in turn_comment_terms
            ):
                turn_scripts.add("Latin")
            if (
                selected is None
                or unsuccessful
                or not turn_actions
                or turn_actions[-1] != "finish_search"
                or not required_methods.issubset(turn_methods)
                or not required_scripts.issubset(turn_scripts)
            ):
                per_turn_failures.append({
                    "turn": index,
                    "actions": turn_actions,
                    "unsuccessful_required_actions": unsuccessful,
                    "missing_methods": sorted(required_methods - turn_methods),
                    "missing_comment_query_scripts": sorted(
                        required_scripts - turn_scripts
                    ),
                })
        gates.append(_gate(
            "retrieval_orchestration_each_turn",
            len(turns) == expected_turn_count and not per_turn_failures,
            failures=per_turn_failures,
        ))

    target_oracle = oracle.get("reverse_target")
    target_report: dict[str, Any] = {"required": False}
    if isinstance(target_oracle, Mapping):
        target_id = str(target_oracle["place_id"])
        candidates = state.get("last_retrieval_candidates")
        candidates = candidates if isinstance(candidates, list) else []
        retrieval_rank = _rank(
            candidates,
            target_id,
            "retrieval_rank_1based",
        )
        presented_rank = _rank(results, target_id, "final_rank")
        decisive_evidence = _evidence_records(target_oracle, "decisive")
        contextual_evidence = _evidence_records(target_oracle, "contextual")
        decisive_ids = [str(item["evidence_id"]) for item in decisive_evidence]
        contextual_ids = [str(item["evidence_id"]) for item in contextual_evidence]
        target_result = next(
            (
                result for result in results
                if str(result.get("place_id")) == target_id
            ),
            None,
        )
        target_evidence = (
            target_result.get("retrieval_evidence", [])
            if isinstance(target_result, Mapping)
            and isinstance(target_result.get("retrieval_evidence"), list)
            else []
        )
        observed_evidence = {
            str(evidence.get("evidence_id"))
            for evidence in target_evidence
            if isinstance(evidence, Mapping)
            and evidence.get("evidence_id")
            and str(evidence.get("place_id")) == target_id
            and evidence.get("source_type") == "verbatim_review"
            and evidence.get("is_verbatim") is True
            and isinstance(evidence.get("text"), str)
            and bool(evidence.get("text", "").strip())
        }
        missing_decisive = sorted(set(decisive_ids) - observed_evidence)
        missing_contextual = sorted(set(contextual_ids) - observed_evidence)
        evidence_ok = bool(decisive_ids) and not missing_decisive

        target_identity_ok = isinstance(target_result, Mapping)
        expected_name = target_oracle.get("name")
        if expected_name:
            target_identity_ok = target_identity_ok and target_result.get("name") == expected_name
        target_coordinate_error = None
        if all(
            isinstance(target_oracle.get(field), (int, float))
            for field in ("latitude", "longitude")
        ):
            if isinstance(target_result, Mapping) and all(
                isinstance(target_result.get(field), (int, float))
                for field in ("lat", "lon")
            ):
                target_coordinate_error = haversine_km(
                    target_oracle["latitude"],
                    target_oracle["longitude"],
                    target_result["lat"],
                    target_result["lon"],
                )
                target_identity_ok = target_identity_ok and target_coordinate_error <= 0.02
            else:
                target_identity_ok = False
        retrieval_limit = int(target_oracle.get("retrieval_rank_lte", 5))
        presented_limit = int(target_oracle.get("presented_rank_lte", 5))
        target_ok = (
            retrieval_rank is not None
            and retrieval_rank <= retrieval_limit
            and presented_rank is not None
            and presented_rank <= presented_limit
            and evidence_ok
            and target_identity_ok
        )
        target_report = {
            "required": True,
            "place_id_sha256": sha256(target_id.encode("utf-8")).hexdigest(),
            "retrieval_rank": retrieval_rank,
            "presented_rank": presented_rank,
            "required_evidence_found": evidence_ok,
            "missing_decisive_evidence_ids": missing_decisive,
            "missing_contextual_evidence_ids": missing_contextual,
            "decisive_evidence_semantics": {
                str(item["evidence_id"]): list(item.get("supports", []))
                for item in decisive_evidence
            },
            "contextual_evidence_semantics": {
                str(item["evidence_id"]): list(item.get("supports", []))
                for item in contextual_evidence
            },
            "target_identity_matches": target_identity_ok,
            "target_coordinate_error_km": (
                round(target_coordinate_error, 6)
                if target_coordinate_error is not None
                else None
            ),
        }
        gates.append(_gate("reverse_target", target_ok, **target_report))

    transit_spec = oracle.get("transit_observation")
    if isinstance(transit_spec, Mapping):
        observation = (
            transit_observation if isinstance(transit_observation, Mapping) else {}
        )
        observation_origin = observation.get("origin")
        observation_destination = observation.get("destination")
        observation_origin = (
            observation_origin if isinstance(observation_origin, Mapping) else {}
        )
        observation_destination = (
            observation_destination
            if isinstance(observation_destination, Mapping)
            else {}
        )
        target_coordinates = target_oracle if isinstance(target_oracle, Mapping) else {}
        coordinate_values_ok = all(
            isinstance(point.get(field), (int, float))
            for point in (origin, observation_origin, target_coordinates, observation_destination)
            for field in ("latitude", "longitude")
        ) if isinstance(origin, Mapping) else False
        if coordinate_values_ok:
            transit_origin_error = haversine_km(
                origin["latitude"],
                origin["longitude"],
                observation_origin["latitude"],
                observation_origin["longitude"],
            )
            transit_destination_error = haversine_km(
                target_coordinates["latitude"],
                target_coordinates["longitude"],
                observation_destination["latitude"],
                observation_destination["longitude"],
            )
            straight_line_distance = haversine_km(
                observation_origin["latitude"],
                observation_origin["longitude"],
                observation_destination["latitude"],
                observation_destination["longitude"],
            )
        else:
            transit_origin_error = None
            transit_destination_error = None
            straight_line_distance = None
        route_distance = observation.get("route_distance_m")
        duration = observation.get("duration_s")
        digests_ok = all(
            isinstance(observation.get(field), str)
            and observation.get(field) == transit_spec.get(field)
            for field in ("request_sha256", "response_sha256")
        )
        transit_ok = (
            coordinate_values_ok
            and transit_origin_error is not None
            and transit_origin_error <= 0.02
            and transit_destination_error is not None
            and transit_destination_error <= 0.02
            and observation.get("provider") == transit_spec.get("provider")
            and isinstance(observation.get("observed_at"), str)
            and observation.get("observed_at", "").startswith(
                str(transit_spec.get("observed_on", ""))
            )
            and isinstance(route_distance, (int, float))
            and route_distance > 0
            and straight_line_distance is not None
            and route_distance + 10 >= straight_line_distance * 1000
            and isinstance(duration, (int, float))
            and duration > 0
            and digests_ok
        )
        gates.append(_gate(
            "public_transit",
            transit_ok,
            provider=observation.get("provider"),
            observed_at=observation.get("observed_at"),
            origin_error_km=(
                round(transit_origin_error, 6)
                if transit_origin_error is not None
                else None
            ),
            destination_error_km=(
                round(transit_destination_error, 6)
                if transit_destination_error is not None
                else None
            ),
            straight_line_distance_km=(
                round(straight_line_distance, 6)
                if straight_line_distance is not None
                else None
            ),
            transit_route_distance_m=route_distance,
            transit_duration_s=duration,
            transit_mode=observation.get("mode"),
            request_and_response_hashes_match=digests_ok,
        ))

    response_text = str(body.get("response") or "")
    language = str(oracle.get("final_language") or scenario["source_language"])
    hangul, latin = _language_counts(response_text)
    language_ok = _matches_language(response_text, language)
    gates.append(_gate(
        "response_language",
        language_ok,
        expected=language,
        hangul_characters=hangul,
        latin_characters=latin,
    ))

    top_five = [str(result.get("place_id")) for result in results[:5]]
    return {
        "schema_version": 2,
        "scenario_id": scenario["id"],
        "pair_id": scenario.get("pair_id"),
        "counterpart_id": scenario.get("counterpart_id"),
        "source_language": scenario["source_language"],
        "intent": {
            "specialty": oracle.get("expected_specialty"),
            "max_distance_km": oracle.get("maximum_distance_km"),
            "latitude": origin.get("latitude") if isinstance(origin, Mapping) else None,
            "longitude": origin.get("longitude") if isinstance(origin, Mapping) else None,
        },
        "hard_pass": all(gate["passed"] for gate in gates),
        "gates": gates,
        "target": target_report,
        "top_five_place_ids": top_five,
    }


def grade_pair(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> dict[str, Any]:
    left_id = left.get("scenario_id")
    right_id = right.get("scenario_id")
    left_pair_id = left.get("pair_id")
    right_pair_id = right.get("pair_id")
    counterpart_ok = (
        isinstance(left_id, str)
        and isinstance(right_id, str)
        and left_id != right_id
        and isinstance(left_pair_id, str)
        and bool(left_pair_id)
        and left_pair_id == right_pair_id
        and left.get("counterpart_id") == right_id
        and right.get("counterpart_id") == left_id
    )
    language_ok = {
        left.get("source_language"),
        right.get("source_language"),
    } == {"English", "Korean"}

    left_intent = left.get("intent")
    right_intent = right.get("intent")
    intent_ok = False
    origin_distance = None
    if isinstance(left_intent, Mapping) and isinstance(right_intent, Mapping):
        coordinates_ok = all(
            isinstance(intent.get(field), (int, float))
            for intent in (left_intent, right_intent)
            for field in ("latitude", "longitude")
        )
        if coordinates_ok:
            origin_distance = haversine_km(
                left_intent["latitude"],
                left_intent["longitude"],
                right_intent["latitude"],
                right_intent["longitude"],
            )
        intent_ok = (
            left_intent.get("specialty") == right_intent.get("specialty")
            and left_intent.get("max_distance_km")
            == right_intent.get("max_distance_km")
            and origin_distance is not None
            and origin_distance <= 0.15
        )

    left_ids = set(left.get("top_five_place_ids", []))
    right_ids = set(right.get("top_five_place_ids", []))
    union = left_ids | right_ids
    jaccard = len(left_ids & right_ids) / len(union) if union else 0.0
    left_target = left.get("target")
    right_target = right.get("target")
    left_target = left_target if isinstance(left_target, Mapping) else {}
    right_target = right_target if isinstance(right_target, Mapping) else {}
    target_required = (
        left_target.get("required") is True
        and right_target.get("required") is True
    )
    target_not_applicable = (
        left_target.get("required") is False
        and right_target.get("required") is False
    )
    left_rank = left_target.get("presented_rank")
    right_rank = right_target.get("presented_rank")
    ranks_present = (
        isinstance(left_rank, int)
        and not isinstance(left_rank, bool)
        and left_rank >= 1
        and isinstance(right_rank, int)
        and not isinstance(right_rank, bool)
        and right_rank >= 1
    )
    rank_difference = abs(left_rank - right_rank) if ranks_present else None
    target_hash_ok = (
        isinstance(left_target.get("place_id_sha256"), str)
        and bool(left_target.get("place_id_sha256"))
        and left_target.get("place_id_sha256")
        == right_target.get("place_id_sha256")
    )
    if target_required:
        rank_ok = ranks_present and rank_difference is not None and rank_difference <= 2
        target_ok = rank_ok and target_hash_ok
    elif target_not_applicable:
        rank_ok = left_rank is None and right_rank is None
        target_ok = rank_ok
    else:
        rank_ok = False
        target_ok = False
    passed = (
        bool(left.get("hard_pass"))
        and bool(right.get("hard_pass"))
        and counterpart_ok
        and language_ok
        and intent_ok
        and jaccard >= 0.6
        and target_ok
    )
    return {
        "passed": passed,
        "left_scenario_id": left_id,
        "right_scenario_id": right_id,
        "counterpart_integrity": counterpart_ok,
        "language_integrity": language_ok,
        "intent_consistency": intent_ok,
        "origin_distance_km": (
            round(origin_distance, 6) if origin_distance is not None else None
        ),
        "top_five_jaccard": round(jaccard, 6),
        "target_requirement_consistency": target_required or target_not_applicable,
        "target_identity_consistency": (
            target_hash_ok if target_required else target_not_applicable
        ),
        "target_rank_consistency": rank_ok,
        "target_rank_difference": rank_difference,
    }


def _find_scenario(casebook: Mapping[str, Any], scenario_id: str) -> Mapping[str, Any]:
    for scenario in casebook.get("scenarios", []):
        if scenario.get("id") == scenario_id:
            return scenario
    raise ValueError(f"Unknown scenario ID: {scenario_id}")


def _load_transit_observation(
    casebook_path: Path,
    scenario: Mapping[str, Any],
) -> Optional[Mapping[str, Any]]:
    oracle = scenario.get("oracle")
    transit_spec = (
        oracle.get("transit_observation") if isinstance(oracle, Mapping) else None
    )
    if not isinstance(transit_spec, Mapping):
        return None
    artifact = transit_spec.get("artifact")
    if not isinstance(artifact, str) or not artifact.strip():
        raise ValueError("transit_observation.artifact must name a saved JSON file")
    path = Path(artifact)
    if not path.is_absolute():
        path = casebook_path.parent / path
    if not path.is_file():
        return None
    observation = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(observation, Mapping):
        raise ValueError("Saved transit observation must be a JSON object")
    return observation


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--casebook", type=Path, required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--journey", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    casebook = json.loads(args.casebook.read_text(encoding="utf-8"))
    journey = json.loads(args.journey.read_text(encoding="utf-8"))
    scenario = _find_scenario(casebook, args.scenario)
    report = grade_scenario(
        scenario,
        journey,
        _load_transit_observation(args.casebook, scenario),
    )
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from patient_journey import atomic_write

    atomic_write(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["hard_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
