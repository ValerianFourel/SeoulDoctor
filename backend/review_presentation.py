"""Prepare selected reviews for display without changing source evidence."""

from html import unescape
from itertools import zip_longest
import re
from time import monotonic
import unicodedata

import requests


TRANSLATION_URL = "https://translation.googleapis.com/language/translate/v2"
MAX_TRANSLATION_CHARACTERS = 16000
MAX_TRANSLATION_REVIEWS = 100


def response_language(message, previous="English", *, established=False, explicit=None):
    requests = (
        ("English", r"\b(?:reply|respond|answer|write|speak|continue|translate)(?:\s+to\s+me)?(?:\s+only)?\s+in\s+english\b|\benglish\s+please\b|영어로\s*(?:답|대답|응답|말|해|부탁)|^영어로[.!?\s]*$"),
        ("Korean", r"\b(?:reply|respond|answer|write|speak|continue|translate)(?:\s+to\s+me)?(?:\s+only)?\s+in\s+korean\b|\bkorean\s+please\b|한국어로\s*(?:답|대답|응답|말|해|부탁)|^한국어로[.!?\s]*$"),
    )
    choices = [(match.start(), language) for language, pattern in requests
               for match in re.finditer(pattern, message, re.IGNORECASE)]
    if choices:
        language = max(choices)[1]
        return language, language
    if explicit in {"English", "Korean"}:
        return explicit, explicit
    if established and previous in {"English", "Korean"}:
        return previous, None
    korean = len(re.findall(r"[가-힣]", message))
    english = len(re.findall(r"[A-Za-z]", message))
    return ("Korean" if korean > english else "English"), None


def useful_review(text):
    if not isinstance(text, str):
        return False
    return any(char.isalpha() and not (
        '\u3130' <= char <= '\u318f' or '\u1100' <= char <= '\u11ff'
    ) for char in text)


def needs_translation(text, language):
    if language == "English":
        return any(char.isalpha() and not char.isascii() for char in text)
    return bool(re.search(r"[A-Za-z]{2,}", text)) or not bool(re.search(r"[가-힣]", text))


def _symbols(text):
    return [char for char in text if unicodedata.category(char) in {"So", "Sk"}
            or char in {'\u200d', '\ufe0f', '\u20e3'}]


_TIME_NUMBERS = {
    word: number for number, word in enumerate((
        "zero", "one", "two", "three", "four", "five", "six", "seven",
        "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
        "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
    ))
}
_TIME_NUMBERS.update({"a": 1, "an": 1})
for tens, english, korean in (
    (20, "twenty", "스물"), (30, "thirty", "서른"), (40, "forty", "마흔"),
    (50, "fifty", "쉰"), (60, "sixty", "예순"), (70, "seventy", "일흔"),
    (80, "eighty", "여든"), (90, "ninety", "아흔"),
):
    _TIME_NUMBERS[english] = tens
    for units, word in enumerate(("", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")):
        if units:
            _TIME_NUMBERS[f"{english} {word}"] = tens + units
            _TIME_NUMBERS[f"{english}-{word}"] = tens + units
    for units, word in enumerate(("", "한", "두", "세", "네", "다섯", "여섯", "일곱", "여덟", "아홉")):
        _TIME_NUMBERS[korean + word] = tens + units
for units, word in enumerate(("영", "한", "두", "세", "네", "다섯", "여섯", "일곱", "여덟", "아홉")):
    _TIME_NUMBERS[word] = units
    if units:
        _TIME_NUMBERS["열" + word] = 10 + units
_TIME_NUMBERS.update({"열": 10, "스무": 20, "하나": 1, "둘": 2, "셋": 3, "넷": 4})
for tens, prefix in enumerate(("", "십", "이십", "삼십", "사십", "오십", "육십", "칠십", "팔십", "구십")):
    for units, word in enumerate(("", "일", "이", "삼", "사", "오", "육", "칠", "팔", "구")):
        if tens or units:
            _TIME_NUMBERS[prefix + word] = tens * 10 + units
_TIME_NUMBER_PATTERN = re.compile(
    r"(?<![A-Za-z가-힣])(" + "|".join(re.escape(word) for word in sorted(_TIME_NUMBERS, key=len, reverse=True))
    + r")\s*(?=(?:hours?|minutes?|days?|weeks?|months?|years?)\b|시간|분|일|주|개월|달|년)",
    re.IGNORECASE,
)


def _numbers(text):
    normalized = _TIME_NUMBER_PATTERN.sub(
        lambda match: str(_TIME_NUMBERS[match.group(1).lower()]) + " ", text,
    )
    normalized = re.sub(r"(?<!\d)\d{1,3}(?:,\d{3})+(?!\d)",
                        lambda match: match.group().replace(",", ""), normalized)
    return re.findall(r"\d+(?:[.,]\d+)*", normalized)


def prepare_review_presentations(cards, language, *, translation_api_key=""):
    pending = {}
    characters = 0
    trace = {"reason": None, "requested": 0, "translated": 0, "capacity_skipped": 0, "rejected": 0, "duration_ms": 0.0}
    for card in cards:
        card["review_language"] = language
    for row in zip_longest(*(card.get("retrieval_evidence", []) for card in cards)):
        for item in row:
            if item is None:
                continue
            text = item.get("text", "")
            if not item.get("is_verbatim") or not useful_review(text):
                status = "hidden"
            elif not needs_translation(text, language):
                status = "original"
            else:
                status = "unavailable"
            item["presentation"] = {"status": status, "language": language}
            if status != "unavailable":
                continue
            if text in pending:
                pending[text].append(item)
            elif len(pending) < MAX_TRANSLATION_REVIEWS and characters + len(text) <= MAX_TRANSLATION_CHARACTERS:
                pending[text] = [item]
                characters += len(text)
            else:
                trace["capacity_skipped"] += 1
    if not pending:
        trace["reason"] = "capacity_exceeded" if trace["capacity_skipped"] else "not_needed"
        return trace
    if not translation_api_key:
        trace["reason"] = "missing_credentials"
        return trace
    originals = list(pending)
    trace["requested"] = len(originals)
    started = monotonic()
    try:
        response = requests.post(
            TRANSLATION_URL,
            headers={"X-Goog-Api-Key": translation_api_key},
            json={"q": originals, "target": "ko" if language == "Korean" else "en",
                  "format": "text", "model": "nmt"},
            timeout=(3, 12),
        )
        response.raise_for_status()
        payload = response.json()
        translations = payload["data"]["translations"]
        if not isinstance(translations, list) or len(translations) != len(originals):
            trace["reason"] = "invalid_response"
            return trace
        for source, result in zip(originals, translations):
            translation = result.get("translatedText") if isinstance(result, dict) else None
            if not isinstance(translation, str) or not translation.strip():
                trace["rejected"] += 1
                continue
            translation = unescape(translation)
            if _symbols(source) != _symbols(translation):
                trace["rejected"] += 1
                continue
            if _numbers(source) != _numbers(translation):
                trace["rejected"] += 1
                continue
            if language == "English" and (re.search(r"[가-힣]", translation) or not re.search(r"[A-Za-z]", translation)):
                trace["rejected"] += 1
                continue
            if language == "Korean" and not re.search(r"[가-힣]", translation):
                trace["rejected"] += 1
                continue
            trace["translated"] += 1
            for item in pending[source]:
                item["presentation"] = {"status": "translated", "language": language, "text": translation}
    except requests.Timeout:
        trace["reason"] = "provider_timeout"
    except requests.RequestException:
        trace["reason"] = "provider_error"
    except (ValueError, KeyError, TypeError):
        trace["reason"] = "invalid_response"
    finally:
        trace["duration_ms"] = round((monotonic() - started) * 1000, 2)
    if trace["reason"] is None and trace["rejected"]:
        trace["reason"] = "translation_rejected"
    return trace
