from __future__ import annotations

import re

_WAKE_RE = re.compile(
    r"\b(?:hey|hi|ok|okay|hay)\s+(minie|mini|minnie|meanie|minny|meany|mimi|nini|money|many|miny)\b",
    re.IGNORECASE,
)
_WAKE_GLUED = re.compile(r"h[ei]+y?m+[iy]+n+[iey]*")
_NOT_MINIE = {"man", "men", "maybe", "my", "me", "more", "mom", "mum", "make", "made", "meet"}

_NOISE = {
    "thank you",
    "thanks",
    "thanks for watching",
    "subscribe",
    "you",
    "here",
    "see you",
    "pfff",
    ".",
    "",
}

# Whisper-tiny invents YouTube outros / travel rants on silence or hiss.
_HALLUCINATION_MARKERS = (
    "subscribe",
    "thanks for watching",
    "video game",
    "gift home",
    "next big gift",
    "parents from germany",
    "honking",
    "glarm",
    "support phone",
    "launch application",
    "union with parents",
    "because their",
    "go home as",
)


def normalize(text: str) -> str:
    text = text.lower().replace("'", "")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _lev(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def is_hallucination(text: str) -> bool:
    cleaned = normalize(text)
    if not cleaned:
        return False
    if any(marker in cleaned for marker in _HALLUCINATION_MARKERS):
        return True
    words = cleaned.split()
    if len(words) > 16:
        return True
    return False


def is_wake_phrase(text: str) -> bool:
    cleaned = normalize(text)
    if not cleaned or cleaned in _NOISE:
        return False
    if is_hallucination(cleaned) or len(cleaned.split()) > 8:
        return False
    collapsed = cleaned.replace(" ", "")
    if re.search(r"\bhey\s+man\b", cleaned) or collapsed in {"heyman", "heymen", "hayman"}:
        return False
    if _WAKE_RE.search(cleaned):
        return True
    if _WAKE_GLUED.search(collapsed):
        return True
    words = cleaned.split()
    if words and words[0] in {"hey", "hi", "hay", "ok", "okay"}:
        rest = [w for w in words[1:] if w]
        if rest and rest[0] == "m":
            return True
        if rest and rest[0] not in _NOT_MINIE and rest[0].startswith("m"):
            if _lev(rest[0], "minie") <= 3 or _lev(rest[0], "minny") <= 2:
                return True
    # "Hamanin" / "heyminie" said as one mushy word — not "him and he".
    if "human" in collapsed:
        return False
    if len(cleaned.split()) > 2:
        return False
    tail = collapsed if len(collapsed) <= 14 else collapsed[-12:]
    if re.search(r"h[aeiouy]{1,3}m[aeiouy]{0,2}n[ieyain]{0,4}", tail):
        return True
    return False


def strip_wake_prefix(text: str) -> str:
    cleaned = normalize(text)
    cleaned = re.sub(
        r"^(?:(?:yes|yeah|ok|okay)\s+)*"
        r"(?:(?:hey|hi|ok|okay|hay)\s+(?:minie|mini|minnie|meanie|minny|meany|mimi|nini|money|many|miny|m)\s*)+",
        "",
        cleaned,
    )
    cleaned = re.sub(r"^(?:hemini|heyminie|heymini|himiny|heyminnie|hamanin)\s*", "", cleaned)
    return cleaned.strip()


def is_plausible_command(text: str) -> bool:
    """Reject TTS echo, wake-only clips, and Whisper-tiny garbage."""
    if is_hallucination(text):
        return False
    cleaned = strip_wake_prefix(text)
    if not cleaned or cleaned in _NOISE | {"yes", "yeah", "amen", "hey"}:
        return False
    words = re.findall(r"[a-z]{3,}", cleaned)
    if len(words) < 2:
        return False
    letters = [c for c in cleaned if c.isalpha()]
    ascii_letters = [c for c in letters if "a" <= c <= "z"]
    if letters and len(ascii_letters) / len(letters) < 0.75:
        return False
    return True
