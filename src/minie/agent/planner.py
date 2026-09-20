from __future__ import annotations

import json
from typing import Any, Callable

from minie.computer.cua import CuaDriver, CuaError, WindowTarget
from minie.config import Config
from minie.log import get_logger

_LOG = get_logger()

_TOOLS = [
    {
        "name": "launch_app",
        "description": "Open a macOS app in the background. Prefer bundle_id when you know it.",
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
        "description": "Read the target window: accessibility tree plus screenshot. Call before every indexed click.",
        "parameters": {
            "type": "object",
            "properties": {
                "pid": {"type": "integer"},
                "window_id": {"type": "integer"},
            },
        },
    },
    {
        "name": "click",
        "description": "Click an element in the last snapshot. Prefer element_index. Background by default.",
        "parameters": {
            "type": "object",
            "properties": {
                "element_index": {"type": "integer"},
                "x": {"type": "number"},
                "y": {"type": "number"},
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
            },
            "required": ["key"],
        },
    },
    {
        "name": "done",
        "description": "Call when the task is finished or you cannot proceed. Summarize for the user.",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
            },
            "required": ["summary"],
        },
    },
]

_SYSTEM = """You are Minie, a Mac computer-use agent.
Drive apps through Cua Driver in the BACKGROUND. Do not steal the user's cursor or ask them to click.
Always snapshot before clicking by element_index. Indexes go stale after every action — snapshot again.
Prefer element_index over x,y. Use x,y only when the control is missing from the tree.
Never type passwords, never send, delete, purchase, or lock unless the user already confirmed.
When the task is complete, call done with a short summary.
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

    def enabled(self) -> bool:
        return bool(self.config.gemini_api_key)

    def run(self, instruction: str, target: WindowTarget | None = None) -> str:
        if self.cancelled():
            return "stopped"
        if not self.enabled():
            return "I need a Gemini API key in .env to finish that."
        self.target = target
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.config.gemini_api_key)
        declarations = [
            types.FunctionDeclaration(
                name=tool["name"],
                description=tool["description"],
                parameters=tool["parameters"],
            )
            for tool in _TOOLS
        ]
        tools = [types.Tool(function_declarations=declarations)]
        contents: list[Any] = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=f"Task: {instruction}\nUse tools. Snapshot after each action.")],
            )
        ]
        last_summary = ""
        for _ in range(self.config.max_planner_steps):
            if self.cancelled():
                return "stopped"
            response = client.models.generate_content(
                model=self.config.gemini_model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=_SYSTEM,
                    tools=tools,
                    temperature=0.2,
                ),
            )
            fn_calls = _function_calls(response)
            if not fn_calls:
                text = (response.text or "").strip()
                return text or last_summary or "done"
            contents.append(response.candidates[0].content)
            parts = []
            for call in fn_calls:
                if self.cancelled():
                    return "stopped"
                result = self._execute(call.name, dict(call.args or {}))
                if call.name == "done":
                    last_summary = str((call.args or {}).get("summary") or result)
                parts.append(
                    types.Part.from_function_response(
                        name=call.name,
                        response={"result": result},
                    )
                )
            contents.append(types.Content(role="user", parts=parts))
            if any(call.name == "done" for call in fn_calls):
                return last_summary or "done"
        return last_summary or "I stopped after too many steps."

    def _execute(self, name: str, args: dict[str, Any]) -> Any:
        _LOG.info("planner tool %s %s", name, args)
        try:
            if name == "launch_app":
                self.target = self.cua.launch_app(
                    name=args.get("name"),
                    bundle_id=args.get("bundle_id"),
                )
                return _public_target(self.target)
            if name == "snapshot":
                self._ensure_target(args)
                assert self.target is not None
                self.cua.snapshot(self.target)
                return {
                    "pid": self.target.pid,
                    "window_id": self.target.window_id,
                    "title": self.target.title,
                    "tree": (self.target.tree or "")[:6000],
                    "screenshot": str(self.target.screenshot) if self.target.screenshot else None,
                }
            if name == "click":
                self._ensure_target(args)
                assert self.target is not None
                self.cua.snapshot(self.target, include_screenshot=False)
                return self.cua.click(
                    self.target,
                    element_index=args.get("element_index"),
                    x=args.get("x"),
                    y=args.get("y"),
                )
            if name == "type_text":
                self._ensure_target(args)
                assert self.target is not None
                return self.cua.type_text(
                    self.target,
                    str(args.get("text") or ""),
                    element_index=args.get("element_index"),
                )
            if name == "press_key":
                self._ensure_target(args)
                assert self.target is not None
                return self.cua.press_key(self.target, str(args.get("key") or "enter"))
            if name == "done":
                return args.get("summary") or "done"
            return {"error": f"unknown tool {name}"}
        except CuaError as exc:
            _LOG.warning("planner tool failed: %s", exc)
            return {"error": str(exc)}

    def _ensure_target(self, args: dict[str, Any]) -> None:
        pid = args.get("pid")
        window_id = args.get("window_id")
        if pid and window_id:
            self.target = WindowTarget(pid=int(pid), window_id=int(window_id))
            return
        if self.target is None:
            raise CuaError("no target window; launch_app or snapshot first")


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
