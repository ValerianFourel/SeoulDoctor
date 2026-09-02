"""Create one OpenAI-compatible LLM client for the configured provider."""

from __future__ import annotations

import os
from typing import Any, Optional

from groq import Groq
from openai import OpenAI

from config import GROQ_API_KEY, LLM_PROVIDER, OPENROUTER_API_KEY


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def build_openrouter_client(api_key: Optional[str] = None) -> Optional[OpenAI]:
    """Build an OpenRouter client without changing the production model."""
    resolved_key = (api_key or OPENROUTER_API_KEY).strip()
    if not resolved_key:
        return None
    return OpenAI(
        api_key=resolved_key,
        base_url=OPENROUTER_BASE_URL,
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
        return Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
    raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}")
