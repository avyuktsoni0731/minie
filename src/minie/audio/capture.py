from __future__ import annotations

import threading

import numpy as np
import sounddevice as sd

from minie.config import Config
from minie.log import get_logger

_LOG = get_logger()


class MicStream:
    """16 kHz mono ring buffer. Wake and ASR both read windows from here."""

    def __init__(self, config: Config, seconds: float = 12.0) -> None:
        self.sr = config.sample_rate
        self._n = int(seconds * self.sr)
        self._buf = np.zeros(self._n, dtype=np.float32)
        self._idx = 0
        self._filled = 0
        self._lock = threading.Lock()
        self._stream: sd.InputStream | None = None

    def start(self) -> None:
        if self._stream is not None:
            return

        def callback(indata, frames, time_info, status) -> None:  # noqa: ANN001
            if status:
                _LOG.debug("mic status: %s", status)
            chunk = np.asarray(indata[:, 0], dtype=np.float32)
            with self._lock:
                n = chunk.shape[0]
                end = self._idx + n
                if end <= self._n:
                    self._buf[self._idx : end] = chunk
                else:
                    split = self._n - self._idx
                    self._buf[self._idx :] = chunk[:split]
                    self._buf[: n - split] = chunk[split:]
                self._idx = end % self._n
                self._filled = min(self._n, self._filled + n)

        self._stream = sd.InputStream(
            samplerate=self.sr,
            channels=1,
            dtype="float32",
            blocksize=512,
            callback=callback,
        )
        self._stream.start()
        _LOG.info("microphone started at %d Hz", self.sr)

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def latest(self, seconds: float) -> np.ndarray:
        n = min(int(seconds * self.sr), self._n)
        with self._lock:
            n = min(n, self._filled)
            if n == 0:
                return np.zeros(0, dtype=np.float32)
            start = (self._idx - n) % self._n
            if start + n <= self._n:
                return self._buf[start : start + n].copy()
            first = self._n - start
            return np.concatenate([self._buf[start:], self._buf[: n - first]])

    def rms(self, seconds: float) -> float:
        window = self.latest(seconds)
        if window.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(window))))

    def clear(self) -> None:
        with self._lock:
            self._buf.fill(0)
            self._idx = 0
            self._filled = 0

    def __enter__(self) -> MicStream:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()
