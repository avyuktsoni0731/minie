from __future__ import annotations

import threading

import numpy as np

from minie.log import get_logger

_LOG = get_logger()
_LOCK = threading.Lock()
_LOADED: dict[str, str] = {}


def transcribe(audio: np.ndarray, model: str, *, language: str = "en") -> str:
    """Transcribe 16 kHz float32 audio with mlx-whisper. Serialized: MLX is not thread-safe."""
    if audio.size < 1600:  # < 100 ms
        return ""
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(audio)))
    if peak > 1.0:
        audio = audio / peak
    with _LOCK:
        first = model not in _LOADED
        if first:
            _LOG.info("loading whisper model %s (first run downloads weights)", model)
            _LOADED[model] = model
        import mlx_whisper

        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=model,
            language=language,
            verbose=False,
            condition_on_previous_text=False,
            without_timestamps=True,
        )
    text = (result.get("text") or "").strip()
    _LOG.debug("asr [%s]: %s", model.split("/")[-1], text)
    return text


def preload(model: str) -> None:
    silence = np.zeros(int(16_000 * 0.3), dtype=np.float32)
    transcribe(silence, model)
