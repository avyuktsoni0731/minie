from __future__ import annotations

import re

_WAKE_RE = re.compile(
    r"\bhey\s+(minie|mini|meanie|minny|meany|mimi)\b",
    re.IGNORECASE,
)

_NOISE = {
    "thank you",
    "thanks",
    "thanks for watching",
    "subscribe",
    "you",
    ".",
    "",
}


def normalize(text: str) -> str:
    text = text.lower().replace("'", "")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def is_wake_phrase(text: str) -> bool:
    cleaned = normalize(text)
    if cleaned in _NOISE:
        return False
    return _WAKE_RE.search(cleaned) is not None
