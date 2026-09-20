from __future__ import annotations

import os
import threading

import numpy as np

from minie.log import get_logger

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")

_LOG = get_logger()
_LOCK = threading.Lock()
_CACHE_READY = False


def _silence_progress() -> None:
    try:
        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()
    except Exception:
        pass
    try:
        import tqdm

        tqdm.tqdm.disable = True
    except Exception:
        pass


def _install_model_cache() -> None:
    """mlx-whisper keeps only one model. Cache both so tiny is not reconstructed every time."""
    global _CACHE_READY
    if _CACHE_READY:
        return
    import mlx.core as mx
    from mlx_whisper.load_models import load_model
    from mlx_whisper.transcribe import ModelHolder

    cache: dict[tuple[str, mx.Dtype], object] = {}

    @classmethod  # type: ignore[misc]
    def get_model(cls, model_path: str, dtype: mx.Dtype):
        key = (str(model_path), dtype)
        if key not in cache:
            _LOG.info("loading whisper weights %s (once)", model_path)
            cache[key] = load_model(model_path, dtype=dtype)
            _LOG.info("whisper ready: %s", model_path)
        cls.model = cache[key]
        cls.model_path = model_path
        return cls.model

    ModelHolder.get_model = get_model
    _CACHE_READY = True


def transcribe(
    audio: np.ndarray,
    model: str,
    *,
    language: str = "en",
    initial_prompt: str | None = None,
) -> str:
    """Transcribe 16 kHz float32 audio with mlx-whisper. Serialized: MLX is not thread-safe."""
    if audio.size < 1600:  # < 100 ms
        return ""
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    rms = float(np.sqrt(np.mean(np.square(audio))))
    if rms < 0.004:
        return ""
    peak = float(np.max(np.abs(audio)))
    if peak > 1.0:
        audio = audio / peak
    kwargs: dict = {
        "path_or_hf_repo": model,
        "language": language,
        "verbose": None,
        "temperature": 0.0,
        "condition_on_previous_text": False,
        "no_speech_threshold": 0.75,
    }
    if initial_prompt:
        kwargs["initial_prompt"] = initial_prompt
    with _LOCK:
        _silence_progress()
        import mlx_whisper

        _install_model_cache()
        result = mlx_whisper.transcribe(audio, **kwargs)
    text = (result.get("text") or "").strip()
    segments = result.get("segments") or []
    if segments:
        avg_ns = sum(float(s.get("no_speech_prob") or 0.0) for s in segments) / len(segments)
        if avg_ns >= 0.7:
            return ""
    _LOG.debug("asr [%s]: %s", model.split("/")[-1], text)
    return text


def preload(model: str) -> None:
    silence = np.zeros(int(16_000 * 0.3), dtype=np.float32)
    transcribe(silence, model)
