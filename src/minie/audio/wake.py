from __future__ import annotations

import time

from minie.audio.capture import MicStream
from minie.audio.phrases import is_wake_phrase, normalize
from minie.audio.whisper import transcribe
from minie.config import Config
from minie.log import get_logger

_LOG = get_logger()

_SKIP_LOG = {"thank you", "thanks", "you", "the", "a", ""}


class WakeListener:
    """Transcribe once per real speech burst. Ambient hiss must not count as speech."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._cooldown_until = 0.0
        self._last_level_log = 0.0
        self._speech_started: float | None = None
        self._checked = False
        self._noise = 0.003

    def cooldown(self, seconds: float = 3.0) -> None:
        self._cooldown_until = time.monotonic() + seconds
        self._speech_started = None
        self._checked = False

    def _gate(self, rms: float) -> float:
        # Track the quiet baseline so MacBook hiss (~0.006) is not "speech".
        if rms < max(self._noise * 1.8, 0.012):
            self._noise = 0.9 * self._noise + 0.1 * rms
        return max(self.config.speech_rms, self._noise * 3.5, self._noise + 0.01, 0.012)

    def poll(self, mic: MicStream) -> bool:
        if time.monotonic() < self._cooldown_until:
            return False
        now = time.monotonic()
        rms = mic.rms(0.3)
        gate = self._gate(rms)
        speaking = rms >= gate
        if now - self._last_level_log >= 5.0:
            self._last_level_log = now
            _LOG.info("mic level %.4f (gate %.4f)", rms, gate)

        if speaking:
            if self._speech_started is None:
                self._speech_started = now
                self._checked = False
                _LOG.info("speech started (level %.4f)", rms)
            elif not self._checked and (now - self._speech_started) >= 1.1:
                return self._transcribe(mic, now - self._speech_started)
            return False

        if self._speech_started is None or self._checked:
            self._speech_started = None
            return False

        spoken_for = now - self._speech_started
        if spoken_for < 0.4:
            self._speech_started = None
            self._checked = False
            return False
        return self._transcribe(mic, spoken_for)

    def _transcribe(self, mic: MicStream, spoken_for: float) -> bool:
        self._checked = True
        self._speech_started = None
        window = mic.latest(max(self.config.wake_window_s, min(2.5, spoken_for + 0.4)))
        text = transcribe(window, self.config.wake_model)
        if is_wake_phrase(text):
            _LOG.info("wake word heard: %s", text)
            self.cooldown()
            return True
        cleaned = normalize(text)
        if text and cleaned not in _SKIP_LOG:
            _LOG.info("heard %r (not wake)", text)
        return False
