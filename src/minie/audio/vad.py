from __future__ import annotations

import numpy as np


def is_speech(audio: np.ndarray, threshold: float) -> bool:
    """Energy VAD with a hangover-friendly RMS gate. No extra ML runtime."""
    if audio.size == 0:
        return False
    rms = float(np.sqrt(np.mean(np.square(audio.astype(np.float32)))))
    return rms >= threshold


def trailing_silence_s(audio: np.ndarray, sample_rate: int, threshold: float, frame_s: float = 0.03) -> float:
    """How many seconds at the end of `audio` fall below the speech gate."""
    frame = max(1, int(sample_rate * frame_s))
    if audio.size < frame:
        return 0.0 if is_speech(audio, threshold) else audio.size / sample_rate
    silent = 0.0
    i = audio.size
    while i - frame >= 0:
        chunk = audio[i - frame : i]
        if is_speech(chunk, threshold):
            break
        silent += frame_s
        i -= frame
    return silent
