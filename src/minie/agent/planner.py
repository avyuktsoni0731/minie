from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from minie.agent.task import AdapterContext, Task
from minie.computer.cua import CuaDriver, CuaError, WindowTarget
from minie.config import Config
from minie.log import get_logger

_LOG = get_logger()

_MUTATING = {"click", "type_text", "press_key", "launch_app"}

_TOOLS = [
    {
        "name": "launch_app",
        "description": "Open a macOS app. Prefer bundle_id when you know it. The window is remembered by app name.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "bundle_id": {"type": "string"},
            },
        },
    },
    {
        "name": "snapshot",
        "description": "Read a window: accessibility tree plus a screenshot image attached for you. Call this to see the UI. Pass pid and window_id to switch windows.",
        "parameters": {
            "type": "object",
            "properties": {
                "pid": {"type": "integer"},
                "window_id": {"type": "integer"},
                "app": {"type": "string", "description": "Name of a previously launched app to snapshot."},
            },
        },
    },
    {
        "name": "click",
        "description": "Click an element from the last snapshot. Prefer element_index. Use x,y only if the control is missing from the tree.",
        "parameters": {
            "type": "object",
            "properties": {
                "element_index": {"type": "integer"},
                "x": {"type": "number"},
                "y": {"type": "number"},
                "app": {"type": "string"},
            },
        },
    },
    {
        "name": "type_text",
        "description": "Type into the target window or a specific element.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "element_index": {"type": "integer"},
                "app": {"type": "string"},
            },
            "required": ["text"],
        },
    },
    {
        "name": "press_key",
        "description": "Press a key such as enter, escape, tab, or a single character.",
        "parameters": {
            "type": "object",
            "properties": {
                "key": {"type": "string"},
                "app": {"type": "string"},
            },
            "required": ["key"],
        },
    },
    {
        "name": "done",
        "description": "Call when the success criteria are met, or you are stuck and cannot proceed. Summarize for the user.",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
            },
            "required": ["summary"],
        },
    },
]


def _system_prompt(task: Task) -> str:
    return f"""You are Minie, a Mac computer-use agent on the GUI fallback path.
Native OS actions already ran if they applied. You only drive the UI.

Goal: {task.goal}
Success criteria: {task.success_criteria or task.goal}

Rules:
- Snapshot to see. You will receive the accessibility tree as text and the screenshot as an image, not a file path.
- Prefer element_index over x,y. Use x,y only when the tree has no matching control.
- One mutating action per turn (click, type, key, or launch). Snapshot is not mutating.
- After each action, compare the new observation to the success criteria. If met, call done.
- If the screen does not change twice, you are stuck — call done and say so. Do not wander into other apps.
- Keep using the same app unless the task requires switching. Use the `app` argument to target a remembered window.
- Never type passwords. Never send, delete, purchase, or place a call unless the user already confirmed.
- Call done when finished. Do not keep searching after the goal is met.
"""


class Planner:
    def __init__(
        self,
        config: Config,
        cua: CuaDriver,
        cancelled: Callable[[], bool],
    ) -> None:
        self.config = config
        self.cua = cua
        self.cancelled = cancelled
        self.target: WindowTarget | None = None
        self.task: Task | None = None
        self.ctx: AdapterContext | None = None
        self._stuck = 0
        self._last_fp = ""
        self._pending_image: bytes | None = None

    def enabled(self) -> bool:
        return bool(self.config.gemini_api_key)

    def run(self, task: Task, ctx: AdapterContext) -> str:
        if self.cancelled():
            return "stopped"
        if not self.enabled():
            return "I need a Gemini API key in .env to finish that."
        self.task = task
        self.ctx = ctx
        self._stuck = 0
        self._last_fp = ""
        self.target = ctx.target_for(task.app)
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.config.gemini_api_key)
        models: list[str] = []
        for name in (self.config.gemini_model, "gemini-3.6-flash", "gemini-flash-latest"):
            if name and name not in models:
                models.append(name)
        declarations = [
            types.FunctionDeclaration(
                name=tool["name"],
                description=tool["description"],
                parameters=tool["parameters"],
            )
            for tool in _TOOLS
        ]
        tools = [types.Tool(function_declarations=declarations)]
        known = ", ".join(ctx.windows) or "none"
        contents: list[Any] = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text=(
                            f"Task: {task.instruction}\n"
                            f"Success criteria: {task.success_criteria or task.goal}\n"
                            f"Remembered windows: {known}\n"
                            "Snapshot, act once, then check success criteria."
                        )
                    )
                ],
            )
        ]
        last_summary = ""
        model = models[0]
        for _ in range(self.config.max_planner_steps):
            if self.cancelled():
                return "stopped"
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=_system_prompt(task),
                        tools=tools,
                        temperature=0.2,
                    ),
                )
            except Exception as exc:
                err = str(exc)
                if "RESOURCE_EXHAUSTED" in err or "429" in err or "spend cap" in err.lower() or "quota" in err.lower():
                    _LOG.warning("Gemini quota exhausted")
                    return (
                        "Gemini is out of quota. I can still open apps, Calculator, and FaceTime "
                        "without it. Raise the spend cap or add a new key."
                    )
                if "NOT_FOUND" in err or "404" in err:
                    nxt = next((m for m in models if m != model), None)
                    if nxt:
                        _LOG.warning("Gemini model %s unavailable, trying %s", model, nxt)
                        model = nxt
                        continue
                _LOG.warning("Gemini request failed: %s", err[:300])
                return "Gemini failed. Check the model name in .env."
            fn_calls = _function_calls(response)
            if not fn_calls:
                text = (response.text or "").strip()
                return text or last_summary or "done"
            contents.append(response.candidates[0].content)
            parts = []
            mutated = False
            self._pending_image = None
            for call in fn_calls:
                if self.cancelled():
                    return "stopped"
                if call.name in _MUTATING and mutated:
                    parts.append(
                        types.Part.from_function_response(
                            name=call.name,
                            response={"result": {"error": "ignored; one mutating action per turn"}},
                        )
                    )
                    continue
                result = self._execute(call.name, dict(call.args or {}))
                if call.name in _MUTATING:
                    mutated = True
                if call.name == "done":
                    last_summary = str((call.args or {}).get("summary") or result)
                parts.append(
                    types.Part.from_function_response(
                        name=call.name,
                        response={"result": result},
                    )
                )
            extra_parts = list(parts)
            if mutated and self.target is not None:
                obs = self._observe(self.target)
                extra_parts.append(
                    types.Part.from_text(
                        text=(
                            "Observation after action. If success criteria are met, call done. "
                            f"Stuck count: {self._stuck}.\n{json.dumps(obs, default=str)[:4000]}"
                        )
                    )
                )
                if self._stuck >= 2:
                    return last_summary or "I got stuck and stopped."
            if self._pending_image:
                extra_parts.append(types.Part.from_bytes(data=self._pending_image, mime_type="image/png"))
                self._pending_image = None
            contents.append(types.Content(role="user", parts=extra_parts))
            if any(call.name == "done" for call in fn_calls):
                return last_summary or "done"
        return last_summary or "I stopped after too many steps."

    def _delivery(self) -> str:
        if self.task is not None and self.task.visible:
            return "foreground"
        return "background"

    def _maybe_front(self) -> None:
        if self.task is None or not self.task.visible or self.target is None:
            return
        try:
            self.cua.bring_to_front(self.target)
        except CuaError as exc:
            _LOG.debug("bring_to_front: %s", exc)

    def _observe(self, target: WindowTarget, *, count_stuck: bool = True) -> dict[str, Any]:
        self.cua.snapshot(target, include_screenshot=True)
        image = _read_screenshot(target.screenshot)
        if image:
            self._pending_image = image
        payload = observation_payload(target, self.ctx.windows if self.ctx else {})
        fp = observation_fingerprint(payload)
        if count_stuck:
            if fp and fp == self._last_fp:
                self._stuck += 1
            else:
                self._stuck = 0
            self._last_fp = fp
        payload["stuck"] = self._stuck
        return payload

    def _execute(self, name: str, args: dict[str, Any]) -> Any:
        _LOG.info("planner tool %s %s", name, args)
        try:
            if name == "launch_app":
                self.target = self.cua.launch_app(
                    name=args.get("name"),
                    bundle_id=args.get("bundle_id"),
                )
                key = str(args.get("name") or args.get("bundle_id") or "app")
                if self.ctx:
                    self.ctx.remember(key, self.target)
                self._maybe_front()
                return _public_target(self.target)
            if name == "snapshot":
                self._ensure_target(args)
                assert self.target is not None
                return self._observe(self.target, count_stuck=False)
            if name == "click":
                self._ensure_target(args)
                assert self.target is not None
                self.cua.snapshot(self.target, include_screenshot=False)
                self._maybe_front()
                xy_only = args.get("element_index") is None
                return self.cua.click(
                    self.target,
                    element_index=args.get("element_index"),
                    x=args.get("x") if xy_only or args.get("x") is not None else None,
                    y=args.get("y") if xy_only or args.get("y") is not None else None,
                    delivery_mode=self._delivery(),
                )
            if name == "type_text":
                self._ensure_target(args)
                assert self.target is not None
                self._maybe_front()
                return self.cua.type_text(
                    self.target,
                    str(args.get("text") or ""),
                    element_index=args.get("element_index"),
                    delivery_mode=self._delivery(),
                )
            if name == "press_key":
                self._ensure_target(args)
                assert self.target is not None
                self._maybe_front()
                return self.cua.press_key(
                    self.target,
                    str(args.get("key") or "enter"),
                    delivery_mode=self._delivery(),
                )
            if name == "done":
                return args.get("summary") or "done"
            return {"error": f"unknown tool {name}"}
        except CuaError as exc:
            _LOG.warning("planner tool failed: %s", exc)
            return {"error": str(exc)}

    def _ensure_target(self, args: dict[str, Any]) -> None:
        app = args.get("app")
        if app and self.ctx:
            found = self.ctx.target_for(str(app))
            if found is not None:
                self.target = found
                return
        pid = args.get("pid")
        window_id = args.get("window_id")
        if pid and window_id:
            self.target = WindowTarget(pid=int(pid), window_id=int(window_id))
            return
        if self.target is None and self.ctx:
            self.target = self.ctx.target_for(self.task.app if self.task else None)
        if self.target is None:
            raise CuaError("no target window; launch_app or snapshot first")


def observation_payload(target: WindowTarget, windows: dict[str, WindowTarget]) -> dict[str, Any]:
    return {
        "pid": target.pid,
        "window_id": target.window_id,
        "title": target.title,
        "name": target.name,
        "tree": (target.tree or "")[:8000],
        "screenshot": "attached" if target.screenshot else "none",
        "windows": {
            key: {"pid": win.pid, "window_id": win.window_id, "title": win.title, "name": win.name}
            for key, win in windows.items()
        },
    }


def observation_fingerprint(payload: dict[str, Any]) -> str:
    blob = f"{payload.get('pid')}|{payload.get('window_id')}|{payload.get('title')}|{payload.get('tree') or ''}"
    return hashlib.sha256(blob.encode("utf-8", errors="ignore")).hexdigest()[:16]


def _read_screenshot(path: Path | None) -> bytes | None:
    if path is None:
        return None
    try:
        data = Path(path).read_bytes()
    except OSError:
        return None
    return data or None


def _function_calls(response: Any) -> list[Any]:
    calls: list[Any] = []
    candidates = getattr(response, "candidates", None) or []
    for cand in candidates:
        content = getattr(cand, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            fn = getattr(part, "function_call", None)
            if fn and getattr(fn, "name", None):
                calls.append(fn)
    return calls


def _public_target(target: WindowTarget | None) -> dict[str, Any]:
    if target is None:
        return {"error": "launch did not return a window"}
    return {"pid": target.pid, "window_id": target.window_id, "name": target.name, "title": target.title}


def json_preview(value: Any) -> str:
    try:
        return json.dumps(value)[:500]
    except TypeError:
        return str(value)[:500]
