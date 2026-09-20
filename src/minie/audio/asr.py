from __future__ import annotations

import time

from minie.audio.capture import MicStream
from minie.audio.vad import is_speech, trailing_silence_s
from minie.audio.whisper import transcribe
from minie.config import Config
from minie.log import get_logger

_LOG = get_logger()


class StreamingASR:
    """Re-transcribe the growing utterance so the agent can act on partials."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.last_text = ""
        self.heard_speech = False
        self.finished = False
        self._started = 0.0
        self._last_infer = 0.0

    def reset(self) -> None:
        self.last_text = ""
        self.heard_speech = False
        self.finished = False
        self._started = time.monotonic()
        self._last_infer = 0.0

    def poll(self, mic: MicStream) -> str | None:
        """Return a new (possibly partial) transcript, or None if unchanged.

        Marks `finished` after trailing silence once some speech has been heard,
        or after a hard cap so a noisy room cannot hold the session open.
        """
        now = time.monotonic()
        elapsed = now - self._started
        window_s = min(12.0, max(1.0, elapsed + 0.4))
        audio = mic.latest(window_s)
        speaking = is_speech(audio[- int(0.25 * self.config.sample_rate) :], self.config.speech_rms) if audio.size else False
        if speaking:
            self.heard_speech = True

        if self.heard_speech:
            silence = trailing_silence_s(audio, self.config.sample_rate, self.config.speech_rms)
            if silence >= self.config.silence_end_s:
                text = transcribe(audio, self.config.asr_model)
                self.last_text = text
                self.finished = True
                _LOG.info("utterance ended: %s", text)
                return text

        if elapsed > 18.0 and self.heard_speech:
            text = transcribe(audio, self.config.asr_model)
            self.last_text = text
            self.finished = True
            _LOG.info("utterance capped: %s", text)
            return text

        if now - self._last_infer < self.config.asr_interval_s:
            return None
        if not speaking and not self.heard_speech:
            return None
        self._last_infer = now
        text = transcribe(audio, self.config.asr_model)
        if text == self.last_text:
            return None
        self.last_text = text
        _LOG.info("partial: %s", text)
        return text
