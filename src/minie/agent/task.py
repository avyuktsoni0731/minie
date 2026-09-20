from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from minie.computer.cua import CuaDriver, WindowTarget
from minie.config import Config


@dataclass
class Task:
    """One user request after routing, before any adapter or GUI loop runs."""

    instruction: str
    goal: str
    app: str | None = None
    success_criteria: str = ""
    risk: str = "low"  # low | confirm | dangerous
    visible: bool = True
    adapter: str = ""
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class AdapterResult:
    ok: bool
    speech: str
    summary: str = ""
    fallback_gui: bool = False


@dataclass
class AdapterContext:
    config: Config
    cua: CuaDriver
    cancelled: Callable[[], bool]
    set_status: Callable[[str], None]
    windows: dict[str, WindowTarget] = field(default_factory=dict)
    opened: set[str] = field(default_factory=set)
    completed: set[str] = field(default_factory=set)

    def remember(self, key: str, target: WindowTarget | None) -> None:
        if not key:
            return
        name = key.lower()
        self.opened.add(name)
        if target is not None and target.pid:
            self.windows[name] = target

    def target_for(self, key: str | None = None) -> WindowTarget | None:
        if key:
            found = self.windows.get(key.lower())
            if found:
                return found
        if not self.windows:
            return None
        return next(reversed(self.windows.values()))
