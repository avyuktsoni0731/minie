from __future__ import annotations

from minie.agent.adapters.base import Adapter
from minie.agent.intent import Intent, looks_like_calculator, parse_calculator_math, parse_open
from minie.agent.task import AdapterContext, AdapterResult, Task
from minie.computer.cua import CuaError
from minie.config import STATUS_ACTING
from minie.log import get_logger

_LOG = get_logger()


class CalculatorAdapter(Adapter):
    name = "calculator"

    def claim(self, instruction: str, ctx: AdapterContext | None = None) -> Task | None:
        math = parse_calculator_math(instruction)
        if math is not None:
            return _task_for(instruction, math)
        calc_open = bool(ctx and "calculator" in ctx.opened)
        opened = parse_open(instruction)
        if opened and (opened.app_name or "").lower() == "calculator":
            calc_open = True
        if calc_open and looks_like_calculator(instruction):
            return _task_for(instruction, Intent(kind="calculator", keys=None))
        return None

    def speculate(self, instruction: str, ctx: AdapterContext) -> None:
        if self.name in ctx.completed or ctx.cancelled():
            return
        math = parse_calculator_math(instruction)
        if math is None or not math.keys:
            return
        if not any(w in instruction.lower() for w in ("equal", "compute", "calculat", "times", "plus", "minus")):
            return
        result = self.execute(_task_for(instruction, math), ctx)
        if result.ok:
            ctx.completed.add(self.name)

    def execute(self, task: Task, ctx: AdapterContext) -> AdapterResult:
        keys = list(task.payload.get("keys") or [])
        if not keys:
            return AdapterResult(ok=False, speech="I heard Calculator but not the numbers")
        if self.name in ctx.completed:
            return AdapterResult(ok=True, speech=_spoken_result(keys), summary="calculator already typed")
        ctx.set_status(STATUS_ACTING)
        opened = parse_open(task.instruction)
        app_name = (opened.app_name if opened else None) or "Calculator"
        bundle = (opened.bundle_id if opened else None) or "com.apple.calculator"
        expr = "".join(keys)
        if ctx.config.dry_run:
            _LOG.info("dry-run calculator %s", expr)
            ctx.completed.add(self.name)
            return AdapterResult(ok=True, speech=_spoken_result(keys), summary=expr)
        try:
            target = ctx.target_for("calculator")
            if target is None or not target.pid:
                target = ctx.cua.launch_app(name=app_name, bundle_id=bundle)
            if target is None or not target.pid:
                raise CuaError("Calculator did not launch")
            ctx.remember("calculator", target)
            if not target.window_id:
                ctx.cua.snapshot(target, include_screenshot=False)
            try:
                ctx.cua.bring_to_front(target)
            except CuaError as exc:
                _LOG.debug("bring_to_front: %s", exc)
            ctx.cua.type_text(target, expr, delivery_mode="foreground")
            ctx.completed.add(self.name)
            _LOG.info("calculator typed %s", expr)
        except CuaError as exc:
            _LOG.warning("calculator path failed: %s", exc)
            return AdapterResult(ok=False, speech="I couldn't type into Calculator", fallback_gui=True)
        spoken = _spoken_result(keys)
        if spoken == "done":
            return AdapterResult(ok=False, speech="I heard Calculator but not the numbers")
        return AdapterResult(ok=True, speech=spoken, summary=expr)


def _task_for(instruction: str, math: Intent) -> Task:
    expr = "".join(math.keys or [])
    return Task(
        instruction=instruction,
        goal=f"Compute {expr.rstrip('=')} on Calculator",
        app="Calculator",
        success_criteria=f"Calculator shows the result of {expr.rstrip('=')}",
        risk="low",
        visible=True,
        adapter="calculator",
        payload={"keys": list(math.keys or [])},
    )


def _spoken_result(keys: list[str] | None) -> str:
    if not keys:
        return "done"
    expr = "".join(keys).rstrip("=")
    if not expr or any(ch not in "0123456789+-*/" for ch in expr):
        return "done"
    try:
        value = eval(expr, {"__builtins__": {}}, {})  # noqa: S307 — digits and + - * / only
    except Exception:
        return "done"
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value)
