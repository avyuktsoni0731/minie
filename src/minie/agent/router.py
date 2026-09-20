from __future__ import annotations

from collections.abc import Callable, Sequence

from minie.agent.adapters import native_adapters
from minie.agent.adapters.base import Adapter
from minie.agent.adapters.gui import GuiAdapter
from minie.agent.task import AdapterContext, AdapterResult, Task
from minie.computer.cua import CuaDriver
from minie.config import Config
from minie.log import get_logger

_LOG = get_logger()


class Router:
    """Native adapters first; GUI observe-act-verify only as fallback."""

    def __init__(
        self,
        config: Config,
        cua: CuaDriver,
        adapters: Sequence[Adapter] | None = None,
        gui: Adapter | None = None,
    ) -> None:
        self.config = config
        self.cua = cua
        self.adapters = list(adapters) if adapters is not None else native_adapters()
        self.gui = gui or GuiAdapter()

    def claim(self, instruction: str, ctx: AdapterContext | None = None) -> tuple[Adapter, Task]:
        for adapter in self.adapters:
            task = adapter.claim(instruction, ctx)
            if task is not None:
                _LOG.info("router native %s", adapter.name)
                return adapter, task
        task = self.gui.claim(instruction, ctx)
        _LOG.info("router gui fallback")
        return self.gui, task

    def speculate(self, instruction: str, ctx: AdapterContext) -> None:
        for adapter in self.adapters:
            if ctx.cancelled():
                return
            adapter.speculate(instruction, ctx)

    def run(self, instruction: str, ctx: AdapterContext) -> AdapterResult:
        adapter, task = self.claim(instruction, ctx)
        result = adapter.execute(task, ctx)
        if result.fallback_gui and adapter is not self.gui:
            _LOG.info("native %s asked for GUI fallback", adapter.name)
            gui_task = self.gui.claim(instruction)
            gui_task.success_criteria = task.success_criteria or gui_task.success_criteria
            gui_task.app = task.app or gui_task.app
            gui_task.visible = task.visible
            return self.gui.execute(gui_task, ctx)
        return result


def make_context(
    config: Config,
    cua: CuaDriver,
    cancelled: Callable[[], bool],
    set_status: Callable[[str], None],
) -> AdapterContext:
    return AdapterContext(config=config, cua=cua, cancelled=cancelled, set_status=set_status)
