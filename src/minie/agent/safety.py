from __future__ import annotations

import re

STOP_RE = re.compile(
    r"\b(stop|cancel|abort|nevermind|never mind|forget it|minie stop|quit minie)\b",
    re.IGNORECASE,
)

DANGEROUS_RE = re.compile(
    r"\b("
    r"send(ing)?|delete|purchase|buy|order|pay|checkout|"
    r"lock (the )?screen|log ?out|password|empty trash|"
    r"shut ?down|restart( the)?( mac| computer)?|erase|format"
    r")\b",
    re.IGNORECASE,
)

CONFIRM_RE = re.compile(r"\b(yes|confirm|do it|go ahead|please do|i confirm)\b", re.IGNORECASE)

PASSWORD_RE = re.compile(r"\b(password|passwd|passcode|credit card|ssn)\b", re.IGNORECASE)


def is_stop(text: str) -> bool:
    return bool(STOP_RE.search(text or ""))


def is_dangerous(text: str) -> bool:
    return bool(DANGEROUS_RE.search(text or ""))


def is_confirmed(text: str) -> bool:
    return bool(CONFIRM_RE.search(text or ""))


def blocks_passwords(text: str) -> bool:
    return bool(PASSWORD_RE.search(text or ""))


def needs_confirm(text: str) -> bool:
    return is_dangerous(text) and not is_confirmed(text)
