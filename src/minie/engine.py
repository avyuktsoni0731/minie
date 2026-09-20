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
        self.prepared = threading.Event()

    def stop(self) -> None:
        self.running.clear()

    def prepare(self) -> None:
        self.set_status(STATUS_LOADING)
        _LOG.info("loading wake model (this should take a few seconds, once)")
        preload(self.config.wake_model)
        if self.config.asr_model != self.config.wake_model:
            _LOG.info("command ASR will use %s on first wake (not loaded yet)", self.config.asr_model)
        cua = CuaDriver(self.config)
        if cua.available():
            _LOG.info("Cua Driver found at %s", cua.bin)
        else:
            _LOG.warning("Cua Driver not on PATH — speech works, actions will fail until it is installed")
        self._cua = cua
        self.prepared.set()

    def loop(self, mic: MicStream) -> None:
        if not hasattr(self, "_cua"):
            self.prepare()
        session = Session(self.config, self._cua, self.set_status)
        wake = WakeListener(self.config)
        self.set_status(STATUS_IDLE)
        _LOG.info('listening for "Hey Minie"')
        while self.running.is_set():
            try:
                if wake.poll(mic):
                    session.run(mic)
                    mic.clear()
                    wake.cooldown(6.0)
                    self.set_status(STATUS_IDLE)
                    _LOG.info('listening for "Hey Minie"')
                else:
                    time.sleep(self.config.wake_interval_s)
            except KeyboardInterrupt:
                break
            except Exception:
                _LOG.exception("engine loop error")
                time.sleep(0.5)
                self.set_status(STATUS_IDLE)
        _LOG.info("engine stopped")

    def run(self, mic: MicStream | None = None) -> None:
        own_mic = mic is None
        if mic is None:
            mic = MicStream(self.config)
        try:
            self.prepare()
            if not mic.started:
                mic.start()
            self.loop(mic)
        finally:
            if own_mic:
                mic.stop()
