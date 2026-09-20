from __future__ import annotations

from minie.agent.adapters.base import Adapter
from minie.agent.intent import extra_action_after_open, parse_open
from minie.agent.task import AdapterContext, AdapterResult, Task
from minie.computer.cua import CuaError
from minie.config import STATUS_ACTING
from minie.log import get_logger

_LOG = get_logger()


class OpenAppAdapter(Adapter):
    name = "open_app"

    def claim(self, instruction: str, ctx: AdapterContext | None = None) -> Task | None:
        opened = parse_open(instruction)
        if not opened or not opened.bundle_id:
            return None
        if extra_action_after_open(instruction):
            return None
        return Task(
            instruction=instruction,
            goal=f"Open {opened.app_name}",
            app=opened.app_name,
            success_criteria=f"{opened.app_name} is running with a window",
            risk="low",
            visible=True,
            adapter=self.name,
            payload={"app_name": opened.app_name, "bundle_id": opened.bundle_id},
        )

    def speculate(self, instruction: str, ctx: AdapterContext) -> None:
        opened = parse_open(instruction)
        if not opened or not opened.bundle_id:
            return
        key = (opened.app_name or "").lower()
        if not key or key in ctx.opened or ctx.cancelled():
            return
        self._launch(opened.app_name or key, opened.bundle_id, ctx)

    def execute(self, task: Task, ctx: AdapterContext) -> AdapterResult:
        name = str(task.payload.get("app_name") or task.app or "")
        bundle = str(task.payload.get("bundle_id") or "")
        if not name or not bundle:
            return AdapterResult(ok=False, speech="I don't know that app.")
        target = self._launch(name, bundle, ctx)
        if target is None or not target.pid:
            return AdapterResult(ok=False, speech=f"I couldn't open {name}.", fallback_gui=True)
        return AdapterResult(ok=True, speech=f"opened {name}", summary=f"Opened {name}")

    def _launch(self, name: str, bundle_id: str, ctx: AdapterContext):
        ctx.set_status(STATUS_ACTING)
        try:
            target = ctx.cua.launch_app(name=name, bundle_id=bundle_id)
        except CuaError as exc:
            _LOG.warning("open %s failed: %s", name, exc)
            return None
        ctx.remember(name, target)
        if target is not None:
            try:
                ctx.cua.bring_to_front(target)
            except CuaError as exc:
                _LOG.debug("bring_to_front: %s", exc)
        _LOG.info("opened %s pid=%s window=%s", name, getattr(target, "pid", None), getattr(target, "window_id", None))
        return target
