"""Reject calls that cannot finish before a temporary endpoint expires."""

from datetime import datetime, timezone


def parse_service_expiry(value: str) -> datetime | None:
    if not value.strip():
        return None
    try:
        expiry = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        raise ValueError("service expiry must be an ISO-8601 timestamp") from None
    if expiry.tzinfo is None:
        raise ValueError("service expiry must include a timezone")
    return expiry.astimezone(timezone.utc)


def lease_allows_request(expiry: datetime | None, timeout_seconds: float) -> bool:
    return expiry is None or (
        expiry - datetime.now(timezone.utc)
    ).total_seconds() > timeout_seconds + 30.0
