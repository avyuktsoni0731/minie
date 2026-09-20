from __future__ import annotations

import time

from minie.audio.capture import MicStream
from minie.audio.phrases import is_wake_phrase
from minie.audio.vad import is_speech
from minie.audio.whisper import transcribe
from minie.config import Config
from minie.log import get_logger

_LOG = get_logger()


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
