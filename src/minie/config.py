from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path.cwd() / ".env")
load_dotenv(Path.home() / ".minie" / ".env")


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value is None or value == "" else value


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return default if raw is None or raw == "" else float(raw)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Config:
    sample_rate: int = 16_000
    wake_model: str = _env("MINIE_WAKE_MODEL", "mlx-community/whisper-tiny")
    asr_model: str = _env("MINIE_ASR_MODEL", "mlx-community/whisper-small-mlx")
    gemini_model: str = _env("MINIE_GEMINI_MODEL", "gemini-2.5-flash")
    gemini_api_key: str = _env("GEMINI_API_KEY", "")
    debug: bool = _env_bool("MINIE_DEBUG")
    dry_run: bool = _env_bool("MINIE_DRY_RUN")
    wake_window_s: float = _env_float("MINIE_WAKE_WINDOW_S", 1.8)
    wake_interval_s: float = _env_float("MINIE_WAKE_INTERVAL_S", 0.4)
    asr_interval_s: float = _env_float("MINIE_ASR_INTERVAL_S", 0.45)
    silence_end_s: float = _env_float("MINIE_SILENCE_END_S", 1.2)
    speech_rms: float = _env_float("MINIE_SPEECH_RMS", 0.012)
    cua_bin: str = _env("MINIE_CUA_BIN", "cua-driver")
    max_planner_steps: int = int(_env("MINIE_MAX_PLANNER_STEPS", "16"))


STATUS_IDLE = "Idle"
STATUS_LOADING = "Loading"
STATUS_LISTENING = "Listening"
STATUS_ACTING = "Acting"
