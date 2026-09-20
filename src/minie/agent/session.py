from __future__ import annotations

import threading
import time
from collections.abc import Callable

from minie.agent.router import Router, make_context
from minie.agent.safety import blocks_passwords, is_stop, needs_confirm
from minie.agent.task import AdapterContext
from minie.audio.asr import StreamingASR
from minie.audio.capture import MicStream
from minie.audio.phrases import is_hallucination, is_plausible_command, strip_wake_prefix
from minie.computer.cua import CuaDriver
from minie.config import STATUS_LISTENING, Config
from minie.log import get_logger
from minie.voice import tts

_LOG = get_logger()


class Session:
    def __init__(
        self,
        config: Config,
        cua: CuaDriver,
        set_status: Callable[[str], None],
    ) -> None:
        self.config = config
        self.cua = cua
        self.set_status = set_status
        self._cancel = threading.Event()
        self.router = Router(config, cua)
        self.ctx: AdapterContext | None = None

    def cancel(self) -> None:
        self._cancel.set()

    def run(self, mic: MicStream) -> None:
        self._cancel.clear()
        self.ctx = make_context(self.config, self.cua, self._cancel.is_set, self.set_status)
        asr = StreamingASR(self.config)
        mic.clear()
        self.set_status(STATUS_LISTENING)
        tts.acknowledge()
        time.sleep(0.5)
        mic.clear()
        asr.reset()
        _LOG.info("session listening — say the command")

        last_partial = ""
        while not self._cancel.is_set():
            text = asr.poll(mic)
            if text:
                if is_hallucination(text):
                    _LOG.info("dropping hallucinated partial")
                    last_partial = ""
                    if asr.finished:
                        break
                    asr.reset()
                    mic.clear()
                    continue
                last_partial = text
                if is_stop(text):
                    tts.speak("stopped", wait=True)
                    return
                if blocks_passwords(text):
                    tts.speak("I will not type passwords.", wait=True)
                    return
                if _is_echo(text):
                    continue
                self.router.speculate(text, self.ctx)
            if asr.finished:
                break
            time.sleep(0.05)
        else:
            tts.speak("stopped", wait=True)
            return

        final = strip_wake_prefix(last_partial or asr.last_text)
        if is_hallucination(final) or not final.strip() or _is_echo(final) or not is_plausible_command(final):
            _LOG.info("no usable command in %r", last_partial or asr.last_text)
            tts.speak("I didn't catch that.", wait=True)
            return
        if is_stop(final):
            tts.speak("stopped", wait=True)
            return
        if needs_confirm(final):
            tts.speak("That looks destructive. Say hey Minie, then yes, to confirm.", wait=True)
            return
        self._finish(final)

    def _finish(self, instruction: str) -> None:
        assert self.ctx is not None
        result = self.router.run(instruction, self.ctx)
        tts.speak(result.speech[:140] if result.speech else "done", wait=True)


def _is_echo(text: str) -> bool:
    cleaned = " ".join(text.lower().split()).strip(" .,?!")
    return cleaned in {"yes", "yes?", "yeah", "yeah that's it", "on it", "done", "stopped", "hey minie", "hey mini"}
