from __future__ import annotations

import subprocess

from minie.log import get_logger

_LOG = get_logger()


def speak(text: str, *, wait: bool = False) -> None:
    """Talk back with macOS `say`. Non-blocking by default so barge-in still works."""
    if not text.strip():
        return
    _LOG.debug("say: %s", text)
    proc = subprocess.Popen(
        ["say", text],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if wait:
        proc.wait(timeout=15)


def chime() -> None:
    subprocess.Popen(
        ["afplay", "/System/Library/Sounds/Tink.aiff"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def acknowledge() -> None:
    chime()
    speak("yes?", wait=True)
