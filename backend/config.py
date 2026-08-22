"""Shared application configuration and frontend/backend contract values."""

GROQ_CHAT_MODEL = "openai/gpt-oss-20b"


DISTANCE_MAPPING = {
    "Walking Distance": 0.5,
    "Nearby": 1.0,
    "Close": 2.0,
    "Moderate": 5.0,
    "Flexible": 10.0,
    "Willing to Travel": 15.0,
    "Anywhere in Seoul": 25.0,
}
