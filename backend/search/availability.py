"""Structured facility-availability predicates used by scope and evidence checks."""

from __future__ import annotations

import re
import unicodedata


_TUESDAY_RANGE = re.compile(
    r"(?:^|;)\s*(?:화|화요일)(?:\([^)]*\))?\s*:\s*"
    r"\d{1,2}:\d{2}\s*-\s*(?P<hour>\d{1,2}):(?P<minute>\d{2})",
    re.IGNORECASE | re.MULTILINE,
)


def has_tuesday_evening(value: object) -> bool:
    """Return whether Tuesday service continues to at least 18:00."""
    if not isinstance(value, str):
        return False
    normalized = unicodedata.normalize("NFKC", value)
    for match in _TUESDAY_RANGE.finditer(normalized):
        hour = int(match.group("hour"))
        minute = int(match.group("minute"))
        if (hour, minute) >= (18, 0):
            return True
    return False
