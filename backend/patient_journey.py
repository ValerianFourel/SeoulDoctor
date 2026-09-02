"""State-custody CLI for a person conversing with a live SeoulDoc endpoint."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Mapping, Optional, Sequence
from urllib.parse import urlsplit, urlunsplit
import uuid

import requests

from models import State


EXPECTED_MODEL = "openai/gpt-oss-120b"
SAFE_HEADERS = {
    "content-type",
    "date",
    "retry-after",
    "server",
    "x-ratelimit-limit",
    "x-ratelimit-remaining",
    "x-request-id",
}
SENSITIVE_KEYS = {
    "api-key",
    "api_key",
    "authorization",
    "cookie",
    "openrouter_api_key",
    "proxy-authorization",
    "set-cookie",
    "x-api-key",
}
SECRET_PATTERNS = (
    re.compile(r"(?i)bearer\s+[a-z0-9._~+/=-]{12,}"),
    re.compile(r"(?i)sk-or-v1-[a-z0-9_-]{12,}"),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def scrub_sensitive(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): (
                "[REDACTED]"
                if str(key).casefold() in SENSITIVE_KEYS
                else scrub_sensitive(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [scrub_sensitive(item) for item in value]
    if isinstance(value, str):
        for pattern in SECRET_PATTERNS:
            value = pattern.sub("[REDACTED]", value)
    return value


def atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    serialized = json.dumps(
        scrub_sensitive(payload),
        ensure_ascii=False,
        indent=2,
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(serialized + "\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)
    os.chmod(path, 0o600)


def _health_endpoint(chat_endpoint: str) -> str:
    parsed = urlsplit(chat_endpoint)
    path = parsed.path.rstrip("/")
    if path.endswith("/chat"):
        path = f"{path[:-5]}/health"
    else:
        path = f"{path}/health"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _chat_endpoint(endpoint: str) -> str:
    parsed = urlsplit(endpoint)
    path = parsed.path.rstrip("/")
    if not path.endswith("/chat"):
        path = f"{path}/chat"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _safe_headers(headers: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key).casefold(): value
        for key, value in headers.items()
        if str(key).casefold() in SAFE_HEADERS
    }


def _health_failures(payload: Mapping[str, Any], expected_model: str) -> list[str]:
    expected = {
        "status": "ok",
        "model_provider": "openrouter",
        "model": expected_model,
        "agent_model": expected_model,
    }
    failures = [
        f"{key} mismatch"
        for key, expected_value in expected.items()
        if payload.get(key) != expected_value
    ]
    facilities = payload.get("facilities")
    vectors = payload.get("vector_documents")
    reviews = payload.get("raw_reviews")
    if not isinstance(facilities, int) or facilities <= 0:
        failures.append("facilities unavailable")
    if not isinstance(vectors, int) or vectors != facilities:
        failures.append("vector index does not match facilities")
    if not isinstance(reviews, int) or reviews <= 0:
        failures.append("raw reviews unavailable")
    return failures


def create_journey(
    path: Path,
    endpoint: str,
    *,
    language: str,
    expected_model: str = EXPECTED_MODEL,
    timeout: float = 180.0,
    session: Optional[Any] = None,
) -> dict[str, Any]:
    http = session or requests.Session()
    endpoint = _chat_endpoint(endpoint)
    response = http.get(
        _health_endpoint(endpoint),
        headers={"Accept": "application/json"},
        timeout=timeout,
    )
    try:
        health = response.json()
    except (ValueError, json.JSONDecodeError) as error:
        raise RuntimeError("SeoulDoc health response was not JSON") from error
    if not 200 <= response.status_code < 300 or not isinstance(health, Mapping):
        raise RuntimeError(f"SeoulDoc health check returned HTTP {response.status_code}")
    failures = _health_failures(health, expected_model)
    if failures:
        raise RuntimeError("SeoulDoc health gate failed: " + "; ".join(failures))

    artifact = {
        "schema_version": 1,
        "journey_id": uuid.uuid4().hex,
        "status": "active",
        "started_at": utc_now(),
        "finished_at": None,
        "finish_reason": None,
        "endpoint": endpoint,
        "language": language,
        "health": deepcopy(dict(health)),
        "current_state": State(
            location=None,
            language_pref=language,
        ).model_dump(),
        "turns": [],
    }
    atomic_write(path, artifact)
    return artifact


def _load_active(path: Path) -> dict[str, Any]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact.get("status") != "active":
        raise RuntimeError("Journey is not active")
    if not isinstance(artifact.get("current_state"), dict):
        raise RuntimeError("Journey state is invalid")
    return artifact


def say(
    path: Path,
    message: str,
    *,
    timeout: float = 180.0,
    session: Optional[Any] = None,
) -> dict[str, Any]:
    message = message.strip()
    if not message:
        raise ValueError("message must not be empty")
    artifact = _load_active(path)
    http = session or requests.Session()
    request_body = {
        "message": message,
        "current_state": deepcopy(artifact["current_state"]),
    }
    started_at = utc_now()
    started = time.perf_counter()
    try:
        response = http.post(
            artifact["endpoint"],
            json=request_body,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Origin": "https://www.seouldoc.io",
                "X-Cookie-Consent": json.dumps({
                    "necessary": True,
                    "analytics": False,
                    "advertising": False,
                }),
            },
            timeout=timeout,
        )
    except requests.RequestException as error:
        elapsed_seconds = round(time.perf_counter() - started, 6)
        artifact["turns"].append({
            "turn": len(artifact["turns"]) + 1,
            "started_at": started_at,
            "elapsed_seconds": elapsed_seconds,
            "request": request_body,
            "response": {
                "status_code": None,
                "headers": {},
                "body": None,
                "parse_error": None,
                "transport_error": {
                    "type": type(error).__name__,
                    "message": str(error)[:500],
                },
            },
        })
        artifact["status"] = "transport_error"
        artifact["finished_at"] = utc_now()
        artifact["finish_reason"] = "The live request failed before a response arrived."
        atomic_write(path, artifact)
        raise RuntimeError("SeoulDoc chat request failed before a response arrived") from error
    elapsed_seconds = round(time.perf_counter() - started, 6)
    try:
        body = response.json()
    except (ValueError, json.JSONDecodeError) as error:
        body = None
        parse_error = type(error).__name__
    else:
        parse_error = None

    turn = {
        "turn": len(artifact["turns"]) + 1,
        "started_at": started_at,
        "elapsed_seconds": elapsed_seconds,
        "request": request_body,
        "response": {
            "status_code": response.status_code,
            "headers": _safe_headers(response.headers),
            "body": body,
            "parse_error": parse_error,
        },
    }
    artifact["turns"].append(turn)
    if (
        200 <= response.status_code < 300
        and isinstance(body, Mapping)
        and isinstance(body.get("state"), Mapping)
        and isinstance(body.get("results"), list)
    ):
        artifact["current_state"] = deepcopy(dict(body["state"]))
        atomic_write(path, artifact)
        return {
            "response": body.get("response", ""),
            "results": deepcopy(body["results"]),
        }

    artifact["status"] = "app_error"
    artifact["finished_at"] = utc_now()
    atomic_write(path, artifact)
    raise RuntimeError(f"SeoulDoc chat returned an invalid HTTP {response.status_code} response")


def finish_journey(path: Path, reason: str) -> dict[str, Any]:
    artifact = _load_active(path)
    artifact["status"] = "finished"
    artifact["finished_at"] = utc_now()
    artifact["finish_reason"] = reason.strip()
    atomic_write(path, artifact)
    return artifact


def run_journey(
    path: Path,
    endpoint: str,
    *,
    language: str,
    messages: Sequence[str],
    timeout: float = 180.0,
    session: Optional[Any] = None,
) -> dict[str, Any]:
    """Run one visit serially so every turn observes the prior saved state."""
    if not messages:
        raise ValueError("at least one message is required")
    http = session or requests.Session()
    create_journey(
        path,
        endpoint,
        language=language,
        timeout=timeout,
        session=http,
    )
    for message in messages:
        say(path, message, timeout=timeout, session=http)
    return finish_journey(path, "All supplied patient messages completed.")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Talk with a live SeoulDoc app.")
    commands = parser.add_subparsers(dest="command", required=True)

    start = commands.add_parser("start")
    start.add_argument("--file", type=Path, required=True)
    start.add_argument("--endpoint", required=True)
    start.add_argument("--language", choices=("English", "Korean"), required=True)
    start.add_argument("--timeout", type=float, default=180.0)

    speak = commands.add_parser("say")
    speak.add_argument("--file", type=Path, required=True)
    speak.add_argument("--message", required=True)
    speak.add_argument("--timeout", type=float, default=180.0)

    run = commands.add_parser("run")
    run.add_argument("--file", type=Path, required=True)
    run.add_argument("--endpoint", required=True)
    run.add_argument("--language", choices=("English", "Korean"), required=True)
    run.add_argument("--message", action="append", required=True)
    run.add_argument("--timeout", type=float, default=180.0)

    finish = commands.add_parser("finish")
    finish.add_argument("--file", type=Path, required=True)
    finish.add_argument("--reason", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "start":
        artifact = create_journey(
            args.file,
            args.endpoint,
            language=args.language,
            timeout=args.timeout,
        )
        print(json.dumps({
            "journey_id": artifact["journey_id"],
            "status": artifact["status"],
            "health": artifact["health"],
        }, ensure_ascii=False, indent=2))
    elif args.command == "say":
        print(json.dumps(
            say(args.file, args.message, timeout=args.timeout),
            ensure_ascii=False,
            indent=2,
        ))
    elif args.command == "run":
        artifact = run_journey(
            args.file,
            args.endpoint,
            language=args.language,
            messages=args.message,
            timeout=args.timeout,
        )
        print(json.dumps({
            "journey_id": artifact["journey_id"],
            "status": artifact["status"],
            "turns": len(artifact["turns"]),
        }, ensure_ascii=False, indent=2))
    else:
        artifact = finish_journey(args.file, args.reason)
        print(json.dumps({
            "status": artifact["status"],
            "finish_reason": artifact["finish_reason"],
        }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
