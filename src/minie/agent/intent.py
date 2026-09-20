from __future__ import annotations

import re
from dataclasses import dataclass

from minie.audio.phrases import normalize

# Display name, bundle id
APP_ALIASES: dict[str, tuple[str, str]] = {
    "calculator": ("Calculator", "com.apple.calculator"),
    "calc": ("Calculator", "com.apple.calculator"),
    "safari": ("Safari", "com.apple.Safari"),
    "notes": ("Notes", "com.apple.Notes"),
    "textedit": ("TextEdit", "com.apple.TextEdit"),
    "text edit": ("TextEdit", "com.apple.TextEdit"),
    "mail": ("Mail", "com.apple.mail"),
    "messages": ("Messages", "com.apple.MobileSMS"),
    "music": ("Music", "com.apple.Music"),
    "chrome": ("Google Chrome", "com.google.Chrome"),
    "google chrome": ("Google Chrome", "com.google.Chrome"),
    "finder": ("Finder", "com.apple.finder"),
    "terminal": ("Terminal", "com.apple.Terminal"),
    "preview": ("Preview", "com.apple.Preview"),
    "calendar": ("Calendar", "com.apple.iCal"),
    "reminders": ("Reminders", "com.apple.reminders"),
    "maps": ("Maps", "com.apple.Maps"),
    "settings": ("System Settings", "com.apple.systempreferences"),
    "system settings": ("System Settings", "com.apple.systempreferences"),
    "photos": ("Photos", "com.apple.Photos"),
}

_OPEN_RE = re.compile(
    r"\b(?:open|launch|start|bring up)\s+(?:the\s+)?(.+?)(?:\s+and\b|,|\.|please|$)",
    re.IGNORECASE,
)

_NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}

_OPS = {
    "times": "*",
    "multiplied by": "*",
    "multiply": "*",
    "x": "*",
    "plus": "+",
    "add": "+",
    "minus": "-",
    "subtract": "-",
    "divided by": "/",
    "divide": "/",
    "over": "/",
}


@dataclass
class Intent:
    kind: str
    app_name: str | None = None
    bundle_id: str | None = None
    keys: list[str] | None = None
    query: str | None = None


def resolve_app(fragment: str) -> tuple[str, str] | None:
    cleaned = normalize(fragment)
    if cleaned in APP_ALIASES:
        return APP_ALIASES[cleaned]
    for alias, pair in APP_ALIASES.items():
        if cleaned.startswith(alias) or alias in cleaned.split():
            return pair
    # Last resort: title-case as a launch name.
    name = fragment.strip(" .,!?")
    if 1 <= len(name) <= 40 and re.fullmatch(r"[A-Za-z0-9 .+\-]+", name):
        return (name, "")
    return None


def parse_open(text: str) -> Intent | None:
    match = _OPEN_RE.search(text or "")
    if not match:
        return None
    resolved = resolve_app(match.group(1))
    if not resolved:
        return None
    name, bundle = resolved
    return Intent(kind="open_app", app_name=name, bundle_id=bundle or None)


def _token_to_digit(token: str) -> str | None:
    token = token.lower().strip()
    if token.isdigit() and len(token) <= 6:
        return token
    return _NUMBER_WORDS.get(token)


def parse_calculator_math(text: str) -> Intent | None:
    cleaned = normalize(text)
    if not any(w in cleaned for w in ("calculat", "compute", "times", "plus", "minus", "divid", "multiply")):
        if not re.search(r"\d+\s*([+x*/-]|times|plus)\s*\d+", cleaned):
            return None
    # six times seven / 6 * 7
    for word, op in _OPS.items():
        pattern = rf"\b([a-z0-9]+)\s+{re.escape(word)}\s+([a-z0-9]+)\b"
        match = re.search(pattern, cleaned)
        if match:
            left = _token_to_digit(match.group(1))
            right = _token_to_digit(match.group(2))
            if left and right:
                return Intent(kind="calculator", keys=_keys_for(left, op, right))
    match = re.search(r"\b(\d+)\s*([+*/x-])\s*(\d+)\b", cleaned)
    if match:
        op = "*" if match.group(2) == "x" else match.group(2)
        return Intent(kind="calculator", keys=_keys_for(match.group(1), op, match.group(3)))
    return None


def _keys_for(left: str, op: str, right: str) -> list[str]:
    keys = list(left)
    keys.append(op)
    keys.extend(list(right))
    keys.append("=")
    return keys


def parse_partial(text: str) -> list[Intent]:
    """Safe speculative intents from a still-growing transcript."""
    intents: list[Intent] = []
    opened = parse_open(text)
    if opened:
        intents.append(opened)
    math = parse_calculator_math(text)
    if math:
        intents.append(math)
    return intents
