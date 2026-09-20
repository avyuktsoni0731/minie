from __future__ import annotations

import re
import time
from typing import Callable

from minie.agent.decide import Decision, decide
from minie.agent.intent import APP_ALIASES, parse_calculator_math, parse_open, _NUMBER_WORDS, _OPS
from minie.agent.structure import fingerprint_tree, parse_candidates, read_displayed_number, token_matches_label
from minie.agent.task import AdapterContext, Task
from minie.audio.phrases import normalize
from minie.computer.cua import CuaError, WindowTarget
from minie.config import Config
from minie.log import get_logger

_LOG = get_logger()

_SKIP = {
    "open",
    "launch",
    "start",
    "bring",
    "up",
    "the",
    "and",
    "then",
    "please",
    "compute",
    "calculat",
    "calculator",
    "a",
    "to",
    "for",
    "with",
}


class StructureLoop:
    """Offline AX ranker used by tests. Live GUI path is WorkflowLoop."""

    def __init__(self, config: Config, ctx: AdapterContext, cancelled: Callable[[], bool]) -> None:
        self.config = config
        self.ctx = ctx
        self.cancelled = cancelled
        self._stuck = 0
        self._last_fp = ""
        self._check_stuck = False
        self._typed_math = False

    def run(self, task: Task) -> str:
        remaining = _goal_tokens(task.instruction)
        _LOG.info("goal tokens %s from %r", remaining, task.instruction)
        target = self.ctx.target_for(task.app)
        launched = self._maybe_launch(task.instruction)
        if launched is not None:
            target = launched
            remaining = [tok for tok in remaining if tok not in {normalize(task.app or ""), "calculator", "safari"}]
        last_summary = "done"
        empty_snaps = 0
        for step in range(self.config.max_planner_steps):
            if self.cancelled():
                return "stopped"
            if target is None or not target.pid:
                target = self._maybe_launch(task.instruction)
                if target is None:
                    return "I couldn't find an app window to drive."
            try:
                self.ctx.cua.snapshot(target, include_screenshot=False)
            except CuaError as exc:
                _LOG.warning("snapshot failed: %s", exc)
                return f"I lost the window: {exc}"
            fp = fingerprint_tree(target)
            if self._check_stuck:
                if fp == self._last_fp:
                    self._stuck += 1
                else:
                    self._stuck = 0
            self._last_fp = fp
            self._check_stuck = False
            if self._stuck >= 2:
                return last_summary if not remaining else "I got stuck and stopped."
            candidates = parse_candidates(target)
            if not candidates:
                _LOG.info("waiting for AX tree pid=%s window=%s", target.pid, target.window_id)
                if empty_snaps >= 2:
                    typed = self._try_type_math(target, task, remaining, [])
                    if typed:
                        return typed
                empty_snaps += 1
                time.sleep(0.45)
                continue
            empty_snaps = 0
            if not self._typed_math:
                typed = self._try_type_math(target, task, remaining, candidates)
                if typed:
                    return typed
            decision = decide(
                self.config,
                goal=task.instruction,
                remaining=remaining,
                candidates=candidates,
                title=target.title or target.name,
                tree=target.tree,
            )
            _LOG.info(
                "cua step %s action=%s target=%s remaining=%s source=%s",
                step + 1,
                decision.action,
                decision.target.cid if decision.target else None,
                remaining,
                decision.source,
            )
            if decision.action == "done" or (not remaining and decision.action == "wait"):
                return self._speech(target, task, last_summary)
            if decision.done >= 0.85 and not remaining:
                return self._speech(target, task, last_summary)
            if decision.risk >= 0.6:
                return "That next click looks destructive. Say hey Minie, then yes, to confirm."
            if decision.confidence and decision.confidence < 0.25 and decision.source == "jev":
                return "I wasn't sure which control to use, so I stopped."
            if decision.action == "wait":
                typed = self._try_type_math(target, task, remaining, candidates)
                if typed:
                    return typed
                time.sleep(0.4)
                continue
            try:
                remaining, last_summary = self._apply(target, decision, remaining, task)
                self._check_stuck = True
            except CuaError as exc:
                _LOG.warning("action failed: %s", exc)
                last_summary = str(exc)
                continue
        return last_summary or "I stopped after too many steps."

    def _maybe_launch(self, instruction: str) -> WindowTarget | None:
        opened = parse_open(instruction)
        name = opened.app_name if opened else None
        bundle = opened.bundle_id if opened else None
        if not name:
            name, bundle = _guess_app(instruction, self.ctx)
        if not name:
            return self.ctx.target_for()
        existing = self.ctx.target_for(name)
        if existing is not None:
            return existing
        try:
            target = self.ctx.cua.launch_app(name=name, bundle_id=bundle)
        except CuaError as exc:
            _LOG.warning("launch %s failed: %s", name, exc)
            return self.ctx.target_for()
        self.ctx.remember(name, target)
        if target is not None:
            try:
                self.ctx.cua.bring_to_front(target)
            except CuaError:
                pass
        _LOG.info("opened %s pid=%s window=%s", name, getattr(target, "pid", None), getattr(target, "window_id", None))
        time.sleep(0.5)
        return target

    def _try_type_math(
        self,
        target: WindowTarget,
        task: Task,
        remaining: list[str],
        candidates: list,
    ) -> str | None:
        math = parse_calculator_math(task.instruction)
        if remaining and all(len(tok) == 1 and tok in "0123456789+-*/=x" for tok in remaining):
            expr = "".join(remaining)
        elif math and math.keys:
            expr = "".join(math.keys)
        else:
            return None
        self._typed_math = True
        self._click_clear(target, candidates, task)
        delivery = "foreground" if task.visible else "background"
        try:
            if task.visible:
                self.ctx.cua.bring_to_front(target)
            self.ctx.cua.type_text(target, expr, delivery_mode=delivery)
        except CuaError as exc:
            _LOG.warning("type math failed: %s", exc)
            self._typed_math = False
            return None
        _LOG.info("typed expression %s", expr)
        time.sleep(0.25)
        try:
            self.ctx.cua.snapshot(target, include_screenshot=False)
        except CuaError:
            pass
        observed = read_displayed_number(parse_candidates(target))
        expected = _finish_speech(task.instruction, expr)
        if observed and expected != "done" and _same_number(observed, expected):
            _LOG.info("calculator display %s", observed)
            return observed
        if observed:
            _LOG.warning("typed %s but display is %s; clicking instead", expr, observed)
            self._click_clear(target, parse_candidates(target), task)
            return None
        return expected

    def _click_clear(self, target: WindowTarget, candidates: list, task: Task) -> None:
        labels = {"c", "ac", "clear", "all clear"}
        clear = next((c for c in candidates if normalize(c.label) in labels or c.label.strip() in {"C", "AC"}), None)
        if clear is None:
            return
        delivery = "foreground" if task.visible else "background"
        try:
            self.ctx.cua.click(target, element_index=clear.index, delivery_mode=delivery)
            _LOG.info("cleared calculator via %s", clear.label or clear.cid)
            time.sleep(0.15)
            self.ctx.cua.snapshot(target, include_screenshot=False)
        except CuaError as exc:
            _LOG.debug("clear failed: %s", exc)

    def _speech(self, target: WindowTarget, task: Task, fallback: str) -> str:
        try:
            self.ctx.cua.snapshot(target, include_screenshot=False)
        except CuaError:
            pass
        observed = read_displayed_number(parse_candidates(target))
        if observed:
            _LOG.info("calculator display %s", observed)
            return observed
        return _finish_speech(task.instruction, fallback)

    def _apply(
        self,
        target: WindowTarget,
        decision: Decision,
        remaining: list[str],
        task: Task,
    ) -> tuple[list[str], str]:
        delivery = "foreground" if task.visible else "background"
        if task.visible:
            try:
                self.ctx.cua.bring_to_front(target)
            except CuaError:
                pass
        if decision.action == "click" and decision.target is not None:
            self.ctx.cua.click(target, element_index=decision.target.index, delivery_mode=delivery)
            label = decision.target.label
            nxt = remaining[0] if remaining else ""
            if nxt and token_matches_label(nxt, label):
                remaining = remaining[1:]
            else:
                _LOG.warning("click %s did not match next token %r; not consuming", label, nxt)
            if not remaining:
                return remaining, self._speech(target, task, f"clicked {decision.target.label or decision.target.cid}")
            return remaining, f"clicked {decision.target.label or decision.target.cid}"
        if decision.action == "type" and decision.target is not None:
            text = decision.text or " ".join(remaining)
            self.ctx.cua.type_text(target, text, element_index=decision.target.index, delivery_mode=delivery)
            return [], f"typed {text[:80]}"
        if decision.action == "press_enter":
            self.ctx.cua.press_key(target, "enter", delivery_mode=delivery)
            return remaining, "pressed enter"
        return remaining, "waited"


def _goal_tokens(instruction: str) -> list[str]:
    math = parse_calculator_math(instruction)
    if math and math.keys:
        return list(math.keys)
    cleaned = normalize(instruction)
    cleaned = re.sub(
        r"\b(?:open|launch|start|bring up)\s+(?:the\s+)?[a-z0-9 ]+?(?=\s+and\b|,|$)",
        "",
        cleaned,
        count=1,
    )
    tokens: list[str] = []
    for raw in cleaned.split():
        if raw in _SKIP or raw.startswith("calcu"):
            continue
        if raw in _NUMBER_WORDS:
            tokens.append(_NUMBER_WORDS[raw])
            continue
        if raw in _OPS:
            tokens.append(_OPS[raw])
            continue
        if raw.isdigit():
            tokens.append(raw)
            continue
        if len(raw) >= 2:
            tokens.append(raw)
    return tokens


def _guess_app(instruction: str, ctx: AdapterContext) -> tuple[str | None, str | None]:
    opened = parse_open(instruction)
    if opened:
        return opened.app_name, opened.bundle_id
    cleaned = normalize(instruction)
    if any(w.startswith("calcu") for w in cleaned.split()):
        return APP_ALIASES["calculator"]
    for alias, pair in APP_ALIASES.items():
        if alias in cleaned.split() or f" {alias} " in f" {cleaned} ":
            return pair
    try:
        apps = ctx.cua.list_apps()
    except CuaError:
        return None, None
    for app in apps:
        name = str(app.get("name") or "")
        if name and normalize(name) in cleaned:
            return name, app.get("bundle_id") or None
    return None, None


def _finish_speech(instruction: str, fallback: str) -> str:
    math = parse_calculator_math(instruction)
    if math and math.keys:
        from minie.agent.adapters.calculator import _spoken_result

        spoken = _spoken_result(math.keys)
        if spoken != "done":
            return spoken
    return fallback or "done"


def _same_number(left: str, right: str) -> bool:
    try:
        return float(left.replace(",", "")) == float(right.replace(",", ""))
    except ValueError:
        return left == right
