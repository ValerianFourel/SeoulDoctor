"""Create one OpenAI-compatible LLM client for the configured provider."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Mapping, Optional, Sequence

from groq import Groq
from openai import OpenAI

from config import GROQ_API_KEY, LLM_PROVIDER, OPENROUTER_API_KEY


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
COMPLETION_TIMEOUT_SECONDS = 60.0
_RETRYABLE_ERROR_NAMES = {
    "APIConnectionError",
    "APITimeoutError",
    "InternalServerError",
    "RateLimitError",
}


def build_openrouter_client(api_key: Optional[str] = None) -> Optional[OpenAI]:
    """Build an OpenRouter client without changing the production model."""
    resolved_key = (api_key or OPENROUTER_API_KEY).strip()
    if not resolved_key:
        return None
    return OpenAI(
        api_key=resolved_key,
        base_url=OPENROUTER_BASE_URL,
        max_retries=0,
        timeout=COMPLETION_TIMEOUT_SECONDS,
        default_headers={
            "HTTP-Referer": os.getenv("APP_URL", "https://seouldoc.io"),
            "X-Title": "SeoulDoc",
        },
    )


def build_llm_client() -> Optional[Any]:
    """Build only the explicitly configured provider client."""
    if LLM_PROVIDER == "openrouter":
        return build_openrouter_client()
    if LLM_PROVIDER == "groq":
        return (
            Groq(
                api_key=GROQ_API_KEY,
                max_retries=0,
                timeout=COMPLETION_TIMEOUT_SECONDS,
            )
            if GROQ_API_KEY
            else None
        )
    raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}")


def _json_object(content: Any) -> dict[str, Any]:
    if not isinstance(content, str) or not content.strip():
        raise ValueError("model returned empty structured content")
    candidate = content.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", candidate, re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
    parsed = json.loads(candidate)
    if not isinstance(parsed, dict):
        raise ValueError("model structured content was not an object")
    return parsed


def _can_retry(error: BaseException) -> bool:
    if isinstance(error, (ValueError, TimeoutError, ConnectionError)):
        return True
    if type(error).__name__ in _RETRYABLE_ERROR_NAMES:
        return True
    status_code = getattr(error, "status_code", None)
    return isinstance(status_code, int) and (
        status_code in {408, 409, 429} or status_code >= 500
    )


def _require_keys(
    payload: Mapping[str, Any],
    required_keys: Sequence[str],
) -> None:
    missing = [key for key in required_keys if key not in payload]
    if missing:
        raise ValueError(
            "model structured content omitted required keys: "
            + ", ".join(missing)
        )


def request_json_completion(
    client: Any,
    *,
    model: str,
    messages: Sequence[Mapping[str, str]],
    max_completion_tokens: int,
    temperature: float = 0.0,
    required_keys: Sequence[str] = (),
) -> tuple[dict[str, Any], Any]:
    """Return a JSON object, retrying once with a larger low-reasoning budget."""
    last_error: Optional[BaseException] = None
    for attempt in range(2):
        token_budget = (
            max_completion_tokens
            if attempt == 0
            else max(1024, max_completion_tokens * 2)
        )
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=list(messages),
                temperature=temperature,
                max_completion_tokens=token_budget,
                response_format={"type": "json_object"},
                reasoning_effort="low",
            )
            payload = _json_object(completion.choices[0].message.content)
            _require_keys(payload, required_keys)
            return payload, completion
        except Exception as error:
            last_error = error
            if attempt == 1 or not _can_retry(error):
                raise
    assert last_error is not None
    raise last_error


def request_answer_completion(
    client: Any,
    *,
    model: str,
    messages: Sequence[Mapping[str, str]],
    max_completion_tokens: int,
    timeout_seconds: float,
) -> tuple[dict[str, Any], Any]:
    """Make one bounded request; incomplete output cannot become an answer."""
    if client is None:
        raise RuntimeError("answer_model_unavailable")
    completion = client.with_options(
        max_retries=0, timeout=timeout_seconds,
    ).chat.completions.create(
        model=model,
        messages=list(messages),
        temperature=0.0,
        max_completion_tokens=max_completion_tokens,
        response_format={"type": "json_object"},
        reasoning_effort="low",
    )
    if not completion.choices or completion.choices[0].finish_reason != "stop":
        raise ValueError("answer_completion_incomplete")
    return _json_object(completion.choices[0].message.content), completion


def request_tool_completion(
    client: Any,
    *,
    model: str,
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]],
    tool_choice: Any,
    max_completion_tokens: int,
    reasoning_effort: str,
) -> tuple[Any, Any]:
    """Return a tool-capable message; retry only an empty or transient result."""
    last_error: Optional[BaseException] = None
    for attempt in range(2):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=list(messages),
                tools=list(tools),
                tool_choice=tool_choice,
                parallel_tool_calls=False,
                reasoning_effort=(
                    reasoning_effort if attempt == 0 else "low"
                ),
                max_completion_tokens=(
                    max_completion_tokens
                    if attempt == 0
                    else max(2048, max_completion_tokens * 2)
                ),
            )
            message = completion.choices[0].message
            tool_calls = getattr(message, "tool_calls", None) or []
            content = getattr(message, "content", None)
            if not tool_calls and not (
                isinstance(content, str) and content.strip()
            ):
                raise ValueError("model returned neither text nor a tool call")
            return message, completion
        except Exception as error:
            last_error = error
            if attempt == 1 or not _can_retry(error):
                raise
    assert last_error is not None
    raise last_error
