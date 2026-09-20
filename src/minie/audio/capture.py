from __future__ import annotations

import threading

import numpy as np
import sounddevice as sd

from minie.config import Config
from minie.log import get_logger

_LOG = get_logger()


def _resample(audio: np.ndarray, src_hz: int, dst_hz: int) -> np.ndarray:
    if src_hz == dst_hz or audio.size == 0:
        return audio.astype(np.float32, copy=False)
    n = int(round(audio.size * dst_hz / src_hz))
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    x_old = np.linspace(0.0, 1.0, audio.size, endpoint=False)
    x_new = np.linspace(0.0, 1.0, n, endpoint=False)
    return np.interp(x_new, x_old, audio.astype(np.float32)).astype(np.float32)


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
        self._native_sr = self.sr

    @property
    def started(self) -> bool:
        return self._stream is not None

    def start(self) -> None:
        if self._stream is not None:
            return
        device = sd.default.device[0] if isinstance(sd.default.device, (list, tuple)) else sd.default.device
        try:
            info = sd.query_devices(device, "input")
            native = int(info.get("default_samplerate") or self.sr)
            _LOG.info("input device: %s", info.get("name") or device)
        except Exception:
            native = self.sr
        # AUHAL often rejects 16 kHz; capture native rate and resample.
        self._native_sr = native if native >= 8000 else self.sr
        _LOG.info("opening microphone at %d Hz (resample → %d)", self._native_sr, self.sr)

        def callback(indata, frames, time_info, status) -> None:  # noqa: ANN001
            if status:
                _LOG.debug("mic status: %s", status)
            chunk = np.asarray(indata[:, 0], dtype=np.float32)
            chunk = _resample(chunk, self._native_sr, self.sr)
            with self._lock:
                n = chunk.shape[0]
                if n == 0:
                    return
                end = self._idx + n
                if end <= self._n:
                    self._buf[self._idx : end] = chunk
                else:
                    split = self._n - self._idx
                    self._buf[self._idx :] = chunk[:split]
                    self._buf[: n - split] = chunk[split:]
                self._idx = end % self._n
                self._filled = min(self._n, self._filled + n)

        last_err: Exception | None = None
        for rate in (self._native_sr, 48000, 44100, self.sr):
            try:
                self._native_sr = rate
                self._stream = sd.InputStream(
                    samplerate=rate,
                    channels=1,
                    dtype="float32",
                    blocksize=0,
                    latency="high",
                    callback=callback,
                )
                self._stream.start()
                _LOG.info("microphone started at %d Hz", rate)
                return
            except Exception as exc:
                last_err = exc
                _LOG.warning("mic open at %d Hz failed: %s", rate, exc)
                self._stream = None
        raise RuntimeError(f"could not open microphone: {last_err}") from last_err

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                _LOG.debug("mic stop ignored an error", exc_info=True)
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
