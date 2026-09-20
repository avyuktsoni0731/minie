from __future__ import annotations

import threading
import time
from collections.abc import Callable

from minie.agent.session import Session
from minie.audio.capture import MicStream
from minie.audio.wake import WakeListener
from minie.audio.whisper import preload
from minie.computer.cua import CuaDriver
from minie.config import STATUS_IDLE, STATUS_LOADING, Config
from minie.log import get_logger

_LOG = get_logger()


class Engine:
    def __init__(self, config: Config, set_status: Callable[[str], None]) -> None:
        self.config = config
        self.set_status = set_status
        self.running = threading.Event()
        self.running.set()

    def stop(self) -> None:
        self.running.clear()

    def run(self) -> None:
        self.set_status(STATUS_LOADING)
        _LOG.info("loading wake model")
        preload(self.config.wake_model)
        threading.Thread(
            target=lambda: preload(self.config.asr_model),
            name="minie-asr-preload",
            daemon=True,
        ).start()
        cua = CuaDriver(self.config)
        if cua.available():
            _LOG.info("Cua Driver found at %s", cua.bin)
        else:
            _LOG.warning("Cua Driver not on PATH — speech works, actions will fail until it is installed")
        session = Session(self.config, cua, self.set_status)
        wake = WakeListener(self.config)
        self.set_status(STATUS_IDLE)
        _LOG.info('listening for "Hey Minie"')
        with MicStream(self.config) as mic:
            while self.running.is_set():
                try:
                    if wake.poll(mic):
                        session.run(mic)
                        wake.cooldown(2.5)
                        self.set_status(STATUS_IDLE)
                    else:
                        time.sleep(self.config.wake_interval_s)
                except KeyboardInterrupt:
                    break
                except Exception:
                    _LOG.exception("engine loop error")
                    time.sleep(0.5)
                    self.set_status(STATUS_IDLE)
        _LOG.info("engine stopped")
