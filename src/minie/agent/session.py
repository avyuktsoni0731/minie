from __future__ import annotations

import threading
import time
from collections.abc import Callable

from minie.agent.intent import Intent, parse_partial
from minie.agent.planner import Planner
from minie.agent.safety import blocks_passwords, is_stop, needs_confirm
from minie.audio.asr import StreamingASR
from minie.audio.capture import MicStream
from minie.computer.cua import CuaDriver, CuaError, WindowTarget
from minie.config import STATUS_ACTING, STATUS_LISTENING, Config
from minie.log import get_logger
from minie.voice import tts

_LOG = get_logger()

_CALC_LABELS = {
    "0": ("0",),
    "1": ("1",),
    "2": ("2",),
    "3": ("3",),
    "4": ("4",),
    "5": ("5",),
    "6": ("6",),
    "7": ("7",),
    "8": ("8",),
    "9": ("9",),
    "+": ("+", "add", "plus"),
    "-": ("-", "subtract", "minus"),
    "*": ("*", "×", "x", "multiply", "times"),
    "/": ("/", "÷", "divide"),
    "=": ("=", "equals", "equal"),
}


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
        self._opened: set[str] = set()
        self._calc_done = False
        self.target: WindowTarget | None = None

    def cancel(self) -> None:
        self._cancel.set()

    def run(self, mic: MicStream) -> None:
        self._cancel.clear()
        self._opened.clear()
        self._calc_done = False
        self.target = None
        asr = StreamingASR(self.config)
        asr.reset()
        mic.clear()
        self.set_status(STATUS_LISTENING)
        tts.acknowledge()
        time.sleep(0.55)
        mic.clear()
        _LOG.info("session listening")

        last_partial = ""
        while not self._cancel.is_set():
            text = asr.poll(mic)
            if text:
                last_partial = text
                if is_stop(text):
                    tts.speak("stopped")
                    return
                if blocks_passwords(text):
                    tts.speak("I will not type passwords.")
                    return
                if _is_echo(text):
                    continue
                self._speculate(text)
            if asr.finished:
                break
            time.sleep(0.05)
        else:
            tts.speak("stopped")
            return

        final = last_partial or asr.last_text
        if not final.strip() or _is_echo(final):
            tts.speak("I didn't catch that.")
            return
        if is_stop(final):
            tts.speak("stopped")
            return
        if needs_confirm(final):
            tts.speak("That looks destructive. Say hey Minie, then yes, to confirm.")
            return
        self._finish(final)

    def _speculate(self, text: str) -> None:
        for intent in parse_partial(text):
            if self._cancel.is_set():
                return
            if intent.kind == "open_app" and intent.app_name and intent.app_name.lower() not in self._opened:
                self._open(intent)
            elif intent.kind == "calculator" and not self._calc_done and intent.keys:
                if any(w in text.lower() for w in ("equal", "compute", "calculat", "times", "plus", "minus")):
                    self._calculator(intent)

    def _open(self, intent: Intent) -> None:
        self.set_status(STATUS_ACTING)
        try:
            self.target = self.cua.launch_app(name=intent.app_name, bundle_id=intent.bundle_id)
            self._opened.add((intent.app_name or "").lower())
            _LOG.info("opened %s pid=%s", intent.app_name, getattr(self.target, "pid", None))
            tts.speak("on it")
        except CuaError as exc:
            _LOG.warning("open failed: %s", exc)
            tts.speak("I could not open that app.")

    def _calculator(self, intent: Intent) -> None:
        if "calculator" not in self._opened:
            self._open(Intent(kind="open_app", app_name="Calculator", bundle_id="com.apple.calculator"))
        if self._cancel.is_set() or not intent.keys:
            return
        self.set_status(STATUS_ACTING)
        try:
            target = self.target or self.cua.launch_app(name="Calculator", bundle_id="com.apple.calculator")
            if target is None:
                raise CuaError("Calculator did not return a window")
            time.sleep(0.35)
            if not target.window_id:
                self.cua.snapshot(target, include_screenshot=False)
            for key in intent.keys:
                if self._cancel.is_set():
                    return
                self.cua.snapshot(target, include_screenshot=False)
                labels = _CALC_LABELS.get(key, (key,))
                idx = self.cua.find_element(target, *labels)
                if idx is None:
                    raise CuaError(f"no Calculator button for {key!r}")
                self.cua.click(target, element_index=idx)
                time.sleep(0.08)
            self._calc_done = True
            self.target = target
            _LOG.info("calculator keys %s", intent.keys)
        except CuaError as exc:
            _LOG.warning("calculator path failed, will let planner try: %s", exc)
            self._calc_done = False

    def _finish(self, instruction: str) -> None:
        if self._calc_done:
            tts.speak("done")
            return
        self.set_status(STATUS_ACTING)
        planner = Planner(self.config, self.cua, self._cancel.is_set)
        summary = planner.run(instruction, self.target)
        if summary == "stopped":
            tts.speak("stopped")
            return
        tts.speak(summary[:140] if summary else "done")


def _is_echo(text: str) -> bool:
    cleaned = " ".join(text.lower().split()).strip(" .,?!")
    return cleaned in {"yes", "yes?", "on it", "done", "stopped"}
