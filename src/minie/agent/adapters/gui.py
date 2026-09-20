from __future__ import annotations

from minie.agent.adapters.base import Adapter
from minie.agent.planner import Planner
from minie.agent.task import AdapterContext, AdapterResult, Task
from minie.config import STATUS_ACTING
from minie.log import get_logger

_LOG = get_logger()


class GuiAdapter(Adapter):
    name = "gui"

    def claim(self, instruction: str, ctx: AdapterContext | None = None) -> Task:
        return Task(
            instruction=instruction,
            goal=instruction,
            app=None,
            success_criteria=f"The request is fully completed: {instruction}",
            risk="low",
            visible=True,
            adapter=self.name,
        )

    def execute(self, task: Task, ctx: AdapterContext) -> AdapterResult:
        ctx.set_status(STATUS_ACTING)
        planner = Planner(ctx.config, ctx.cua, ctx.cancelled)
        try:
            summary = planner.run(task, ctx)
        except Exception:
            _LOG.exception("planner failed")
            return AdapterResult(ok=False, speech="the planner failed")
        if summary == "stopped":
            return AdapterResult(ok=False, speech="stopped")
        text = (summary or "done").strip()
        return AdapterResult(ok=True, speech=text[:140], summary=text)
