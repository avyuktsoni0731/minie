from __future__ import annotations

import time

import numpy as np

from minie.audio.capture import MicStream
from minie.audio.phrases import is_hallucination
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
        self._speech_started = 0.0
        self._last_infer = 0.0

    def reset(self) -> None:
        self.last_text = ""
        self.heard_speech = False
        self.finished = False
        self._started = time.monotonic()
        self._speech_started = 0.0
        self._last_infer = 0.0

    def poll(self, mic: MicStream) -> str | None:
        """Return a new (possibly partial) transcript, or None if unchanged.

        Only the speech burst is transcribed. A long window of hush makes
        Whisper-tiny invent YouTube outros and fake math.
        """
        now = time.monotonic()
        elapsed = now - self._started
        tail = mic.latest(0.25)
        speaking = is_speech(tail, self.config.speech_rms) if tail.size else False
        if speaking:
            if not self.heard_speech:
                self._speech_started = now
            self.heard_speech = True

        if not self.heard_speech:
            if elapsed > 6.0:
                self.finished = True
                _LOG.info("no speech in session")
            return None

        spoken_s = now - self._speech_started
        window_s = min(8.0, max(0.6, spoken_s + 0.35))
        audio = mic.latest(window_s)
        rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0

        if self.heard_speech:
            silence = trailing_silence_s(audio, self.config.sample_rate, self.config.speech_rms)
            if silence >= self.config.silence_end_s:
                return self._finish(transcribe(audio, self.config.asr_model), "ended")

        if spoken_s > 8.0:
            return self._finish(transcribe(audio, self.config.asr_model), "capped")

        if now - self._last_infer < self.config.asr_interval_s:
            return None
        if rms < self.config.speech_rms * 0.45:
            return None
        self._last_infer = now
        text = transcribe(audio, self.config.asr_model)
        if is_hallucination(text):
            _LOG.info("ignoring whisper garbage: %s", text[:90])
            return None
        if text == self.last_text:
            return None
        self.last_text = text
        _LOG.info("partial: %s", text)
        return text

    def _finish(self, text: str, why: str) -> str | None:
        if is_hallucination(text):
            _LOG.info("utterance %s was whisper garbage: %s", why, text[:90])
            self.last_text = ""
            self.finished = True
            return None
        self.last_text = text
        self.finished = True
        _LOG.info("utterance %s: %s", why, text)
        return text
