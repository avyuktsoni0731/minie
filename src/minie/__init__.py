"""Minie — wake, speak, act on your Mac."""

from __future__ import annotations

import os

# Must be set before huggingface_hub / mlx-whisper import.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

__version__ = "0.1.0"
