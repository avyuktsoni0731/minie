from __future__ import annotations

from minie.agent.adapters.base import Adapter
from minie.agent.task import AdapterContext, AdapterResult, Task
from minie.agent.workflow import WorkflowLoop
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
        loop = WorkflowLoop(ctx.config, ctx, ctx.cancelled)
        try:
            summary = loop.run(task)
        except Exception:
            _LOG.exception("workflow loop failed")
            return AdapterResult(ok=False, speech="the computer-use loop failed")
        if summary == "stopped":
            return AdapterResult(ok=False, speech="stopped")
        text = (summary or "done").strip()
        if text.startswith("I need an Anthropic") or text.startswith("I couldn't") or text.startswith("I stopped"):
            return AdapterResult(ok=False, speech=text[:140], summary=text)
        return AdapterResult(ok=True, speech=text[:140], summary=text)
