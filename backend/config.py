"""Shared application configuration and provider selection."""

from pathlib import Path
import os

from dotenv import load_dotenv


BACKEND_DIR = Path(__file__).resolve().parent
load_dotenv(BACKEND_DIR / ".env")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

LLM_PROVIDER = os.getenv(
    "LLM_PROVIDER",
    "openrouter" if OPENROUTER_API_KEY else "groq",
).strip().casefold()
if LLM_PROVIDER not in {"openrouter", "groq"}:
    raise ValueError("LLM_PROVIDER must be 'openrouter' or 'groq'")


def _bounded_positive_int_env(
    name: str,
    default: int,
    maximum: int,
) -> int:
    """Parse a positive integer environment setting at the config boundary."""
    raw_value = os.getenv(name, str(default)).strip()
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not 1 <= value <= maximum:
        raise ValueError(f"{name} must be between 1 and {maximum}")
    return value


def _bounded_positive_float_env(
    name: str,
    default: float,
    maximum: float,
) -> float:
    """Parse a positive floating-point setting at the config boundary."""
    raw_value = os.getenv(name, str(default)).strip()
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not 0.0 < value <= maximum:
        raise ValueError(f"{name} must be greater than 0 and at most {maximum}")
    return value



CHAT_RATE_LIMIT_REQUESTS = _bounded_positive_int_env(
    "CHAT_RATE_LIMIT_REQUESTS", 12, 1_000
)
CHAT_RATE_LIMIT_WINDOW_SECONDS = _bounded_positive_int_env(
    "CHAT_RATE_LIMIT_WINDOW_SECONDS", 60, 3_600
)
ENABLE_RETRIEVAL_DEBUG = os.getenv(
    "ENABLE_RETRIEVAL_DEBUG", "false"
).strip().casefold() in {"1", "true", "yes", "on"}
RETRIEVAL_DEBUG_LIMIT = _bounded_positive_int_env(
    "RETRIEVAL_DEBUG_LIMIT", 50, 200
)

HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
RERANKER_URL = os.getenv("RERANKER_URL", "").strip().rstrip("/")
RERANKER_EXPIRES_AT = os.getenv("RERANKER_EXPIRES_AT", "").strip()
BGE_M3_RETRIEVER_EXPIRES_AT = os.getenv("BGE_M3_RETRIEVER_EXPIRES_AT", "").strip()
BGE_M3_MODEL_REVISION = os.getenv(
    "BGE_M3_MODEL_REVISION", "5617a9f61b028005a4858fdac845db406aefb181"
).strip()
RERANKER_API_TOKEN = os.getenv("RERANKER_API_TOKEN", HF_TOKEN).strip()
RERANKER_TIMEOUT_SECONDS = _bounded_positive_float_env(
    "RERANKER_TIMEOUT_SECONDS", 20.0, 120.0
)
RERANKER_MAX_CANDIDATES = _bounded_positive_int_env(
    "RERANKER_MAX_CANDIDATES", 64, 512
)

BGE_M3_RETRIEVER_URL = os.getenv(
    "BGE_M3_RETRIEVER_URL", ""
).strip().rstrip("/")
BGE_M3_RETRIEVER_API_TOKEN = os.getenv(
    "BGE_M3_RETRIEVER_API_TOKEN", HF_TOKEN
).strip()
BGE_M3_RETRIEVER_RELEASE_ID = os.getenv(
    "BGE_M3_RETRIEVER_RELEASE_ID", ""
).strip()
BGE_M3_RETRIEVER_TIMEOUT_SECONDS = _bounded_positive_float_env(
    "BGE_M3_RETRIEVER_TIMEOUT_SECONDS", 8.0, 60.0
)
if bool(BGE_M3_RETRIEVER_URL) != bool(BGE_M3_RETRIEVER_RELEASE_ID):
    raise ValueError(
        "BGE_M3_RETRIEVER_URL and BGE_M3_RETRIEVER_RELEASE_ID must be set together"
    )


OPENROUTER_CHAT_MODEL = os.getenv(
    "OPENROUTER_CHAT_MODEL", "openai/gpt-oss-120b"
)
OPENROUTER_AGENT_MODEL = os.getenv(
    "OPENROUTER_AGENT_MODEL", OPENROUTER_CHAT_MODEL
)
GROQ_DEFAULT_CHAT_MODEL = os.getenv(
    "GROQ_CHAT_MODEL", "openai/gpt-oss-120b"
)
GROQ_DEFAULT_AGENT_MODEL = os.getenv(
    "GROQ_AGENT_MODEL", GROQ_DEFAULT_CHAT_MODEL
)

CHAT_MODEL = os.getenv(
    "LLM_CHAT_MODEL",
    OPENROUTER_CHAT_MODEL if LLM_PROVIDER == "openrouter"
    else GROQ_DEFAULT_CHAT_MODEL,
)
ANSWER_MODEL = os.getenv("LLM_ANSWER_MODEL", CHAT_MODEL)
AGENT_MODEL = os.getenv(
    "LLM_AGENT_MODEL",
    OPENROUTER_AGENT_MODEL if LLM_PROVIDER == "openrouter"
    else GROQ_DEFAULT_AGENT_MODEL,
)
REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT", "medium")

# Compatibility aliases for modules that have not yet migrated their imports.
GROQ_CHAT_MODEL = CHAT_MODEL
GROQ_AGENT_MODEL = AGENT_MODEL
GROQ_REASONING_EFFORT = REASONING_EFFORT

SUPPORTED_TOOL_MODELS = {
    "qwen/qwen3.8-max",
    "qwen/qwen3.8-27b",
    "openai/gpt-5.6-sol",
    "openai/gpt-5.5",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.6-27b",
}


DISTANCE_MAPPING = {
    "Walking Distance": 0.5,
    "Nearby": 1.0,
    "Close": 2.0,
    "Moderate": 5.0,
    "Flexible": 10.0,
    "Willing to Travel": 15.0,
    "Anywhere in Seoul": 25.0,
}
