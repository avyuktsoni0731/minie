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
    "facetime": ("FaceTime", "com.apple.FaceTime"),
    "face time": ("FaceTime", "com.apple.FaceTime"),
    "contacts": ("Contacts", "com.apple.AddressBook"),
    "address book": ("Contacts", "com.apple.AddressBook"),
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
    "multiplied": "*",
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
        if cleaned == alias or cleaned.startswith(alias + " "):
            return pair
    # Prefix match only for long aliases so "call" ≠ Calculator.
    if cleaned.startswith("calcu"):
        return APP_ALIASES["calculator"]
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
    if token.isdigit() and 1 <= len(token) <= 3:
        return token
    return _NUMBER_WORDS.get(token)


def _repair_whisper_math(cleaned: str) -> str:
    """Tiny-whisper often emits 1000 for 'six'. Don't send that to the GUI loop."""
    if "thousand" in cleaned:
        return cleaned
    return re.sub(r"\b1000+\b", "6", cleaned)


def looks_like_calculator(text: str) -> bool:
    cleaned = normalize(text)
    if any(w in cleaned for w in ("calculat", "compute", "times", "multipl", "plus", "minus", "divid")):
        return True
    opened = parse_open(text)
    return bool(opened and (opened.app_name or "").lower() == "calculator")


def parse_calculator_math(text: str) -> Intent | None:
    cleaned = _repair_whisper_math(normalize(text))
    cleaned = re.sub(r"multi[\s-]*(?:plied|ply|cloud)", "multiplied", cleaned)
    if not any(w in cleaned for w in ("calculat", "compute", "times", "plus", "minus", "divid", "multipl")):
        if not re.search(r"\d+\s*([+x*/-]|times|plus)\s*\d+", cleaned):
            return None
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
        left = _token_to_digit(match.group(1))
        right = _token_to_digit(match.group(3))
        if left and right:
            op = "*" if match.group(2) == "x" else match.group(2)
            return Intent(kind="calculator", keys=_keys_for(left, op, right))
    # Whisper often garbles the operator but keeps the two numbers.
    if any(w in cleaned for w in ("comput", "calculat", "times", "multipl")):
        nums: list[str] = []
        for tok in cleaned.split():
            digit = _token_to_digit(tok)
            if digit:
                nums.append(digit)
        if len(nums) >= 2:
            op = "*"
            if "plus" in cleaned or "add" in cleaned:
                op = "+"
            elif "minus" in cleaned or "subtract" in cleaned:
                op = "-"
            elif "divid" in cleaned:
                op = "/"
            return Intent(kind="calculator", keys=_keys_for(nums[0], op, nums[1]))
    return None


def _keys_for(left: str, op: str, right: str) -> list[str]:
    keys = list(left)
    keys.append(op)
    keys.extend(list(right))
    keys.append("=")
    return keys


_CALL_STOP = {
    "me",
    "the",
    "a",
    "an",
    "this",
    "that",
    "on",
    "up",
    "it",
    "my",
    "you",
    "him",
    "her",
    "them",
    "now",
    "back",
    "off",
    "calculator",
    "calcu",
    "app",
    "someone",
    "and",
    "or",
    "to",
    "with",
    "for",
    "from",
    "open",
    "audio",
    "video",
}


def parse_call(text: str) -> Intent | None:
    """Native FaceTime/call request. Does not place the call — adapters confirm."""
    cleaned = normalize(text)
    if not re.search(r"\b(call|facetime|face time|ring|phone)\b", cleaned):
        return None
    who = None
    for match in re.finditer(
        r"\b(?:facetime|face time|video call|phone|ring|call)\s+"
        r"(?:audio\s+)?(?:call\s+)?(?:to\s+|with\s+)?"
        r"([a-z][a-z0-9]+)\b",
        cleaned,
    ):
        candidate = match.group(1)
        if candidate not in _CALL_STOP:
            who = candidate
            break
    if not who:
        return None
    audio = any(w in cleaned for w in ("audio", "phone"))
    return Intent(
        kind="call",
        app_name="FaceTime",
        bundle_id="com.apple.FaceTime",
        query=who,
        keys=["audio"] if audio else None,
    )


def extra_action_after_open(text: str) -> bool:
    """True when 'open X and …' still has work beyond launching the app."""
    cleaned = normalize(text)
    cleaned = re.sub(
        r"\b(?:open|launch|start|bring up)\s+(?:the\s+)?[a-z0-9 ]+?(?=\s+and\b|,|$)",
        "",
        cleaned,
        count=1,
    )
    return bool(
        re.search(
            r"\b(and|then|search|type|click|compute|calculat|call|send|write|find)\b",
            cleaned,
        )
    )


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
