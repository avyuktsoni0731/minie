from __future__ import annotations

import os
from dataclasses import dataclass, field
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


def _env_map(name: str) -> dict[str, str]:
    raw = os.getenv(name) or ""
    out: dict[str, str] = {}
    for part in raw.split(","):
        part = part.strip()
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        key, value = key.strip().lower(), value.strip()
        if key and value:
            out[key] = value
    nick_file = Path.home() / ".minie" / "nicknames.txt"
    if nick_file.is_file():
        try:
            for line in nick_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or ":" not in line:
                    continue
                key, value = line.split(":", 1)
                key, value = key.strip().lower(), value.strip()
                if key and value:
                    out[key] = value
        except OSError:
            pass
    return out


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Config:
    sample_rate: int = 16_000
    wake_model: str = _env("MINIE_WAKE_MODEL", "mlx-community/whisper-tiny")
    # Same model as wake by default. mlx-whisper would otherwise unload tiny
    # and reconstruct small on every switch, which looks "stuck".
    asr_model: str = _env("MINIE_ASR_MODEL", "mlx-community/whisper-tiny")
    gemini_model: str = _env("MINIE_GEMINI_MODEL", "gemini-3.6-flash")
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
    nicknames: dict[str, str] = field(default_factory=lambda: _env_map("MINIE_NICKNAMES"))


STATUS_IDLE = "Idle"
STATUS_LOADING = "Loading"
STATUS_LISTENING = "Listening"
STATUS_ACTING = "Acting"
