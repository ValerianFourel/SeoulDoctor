"""Prepare selected reviews for display without changing source evidence."""

from html import unescape
import re
import unicodedata

import requests


TRANSLATION_URL = "https://translation.googleapis.com/language/translate/v2"
MAX_TRANSLATION_CHARACTERS = 16000
MAX_TRANSLATION_REVIEWS = 20


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


def prepare_review_presentations(cards, language, *, translation_api_key=""):
    pending = {}
    characters = 0
    for card in cards:
        card["review_language"] = language
        for item in card.get("retrieval_evidence", []):
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
    if not pending or not translation_api_key:
        return
    originals = list(pending)
    try:
        response = requests.post(
            TRANSLATION_URL,
            headers={"X-Goog-Api-Key": translation_api_key},
            json={"q": originals, "target": "ko" if language == "Korean" else "en",
                  "format": "text", "model": "nmt"},
            timeout=(2, 5),
        )
        response.raise_for_status()
        payload = response.json()
        translations = payload["data"]["translations"]
        if not isinstance(translations, list) or len(translations) != len(originals):
            return
        for source, result in zip(originals, translations):
            translation = result.get("translatedText") if isinstance(result, dict) else None
            if not isinstance(translation, str) or not translation.strip():
                continue
            translation = unescape(translation)
            if _symbols(source) != _symbols(translation):
                continue
            if re.findall(r"\d+(?:[.,]\d+)*", source) != re.findall(r"\d+(?:[.,]\d+)*", translation):
                continue
            if language == "English" and (re.search(r"[가-힣]", translation) or not re.search(r"[A-Za-z]", translation)):
                continue
            if language == "Korean" and not re.search(r"[가-힣]", translation):
                continue
            for item in pending[source]:
                item["presentation"] = {"status": "translated", "language": language, "text": translation}
    except (requests.RequestException, ValueError, KeyError, TypeError):
        # Provider failures must not hide originals or expose request data in logs.
        return
