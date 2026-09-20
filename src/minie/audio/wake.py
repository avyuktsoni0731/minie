from __future__ import annotations

import re
import time

from minie.audio.capture import MicStream
from minie.audio.vad import is_speech
from minie.audio.whisper import transcribe
from minie.config import Config
from minie.log import get_logger

_LOG = get_logger()

_WAKE_RE = re.compile(
    r"\bhey\s+(minie|mini|meanie|minny|meany|mimi)\b",
    re.IGNORECASE,
)

# Whisper-tiny silence hallucinations — never treat these as a wake.
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


class WakeListener:
    def __init__(self, config: Config) -> None:
        self.config = config
        self._cooldown_until = 0.0

    def cooldown(self, seconds: float = 3.0) -> None:
        self._cooldown_until = time.monotonic() + seconds

    def poll(self, mic: MicStream) -> bool:
        if time.monotonic() < self._cooldown_until:
            return False
        window = mic.latest(self.config.wake_window_s)
        if not is_speech(window, self.config.speech_rms):
            return False
        text = transcribe(window, self.config.wake_model)
        if is_wake_phrase(text):
            _LOG.info("wake word heard: %s", text)
            self.cooldown()
            return True
        return False
