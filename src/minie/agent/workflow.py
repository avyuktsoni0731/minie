from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol

from minie.agent.intent import APP_ALIASES, resolve_app
from minie.agent.structure import Candidate, parse_candidates, public_elements, read_displayed_number
from minie.agent.task import AdapterContext, Task
from minie.computer.cua import CuaError, WindowTarget
from minie.config import Config
from minie.log import get_logger

_LOG = get_logger()

_TOOLS = ("launch_app", "open_url", "click", "type", "press_key", "wait")
_MISSING_KEY = "I need an Anthropic API key to run computer use. Add ANTHROPIC_API_KEY to .env"

_ARG_KEYS = (
    "name",
    "bundle_id",
    "app",
    "url",
    "text",
    "query",
    "key",
    "label",
    "target_id",
    "submit",
    "seconds",
    "element_index",
)

_PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "goal": {"type": "string"},
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "tool": {"type": "string", "enum": list(_TOOLS)},
                    "args": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "name": {"type": "string"},
                            "bundle_id": {"type": "string"},
                            "app": {"type": "string"},
                            "url": {"type": "string"},
                            "text": {"type": "string"},
                            "query": {"type": "string"},
                            "key": {"type": "string"},
                            "label": {"type": "string"},
                            "target_id": {"type": "string"},
                            "submit": {"type": "boolean"},
                            "seconds": {"type": "number"},
                            "element_index": {"type": "integer"},
                        },
                        "required": list(_ARG_KEYS),
                    },
                    "verify": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "window_title": {"type": "string"},
                            "url_or_title_contains": {"type": "string"},
                            "ax_contains": {"type": "string"},
                            "display_number": {"type": "string"},
                        },
                        "required": [
                            "window_title",
                            "url_or_title_contains",
                            "ax_contains",
                            "display_number",
                        ],
                    },
                },
                "required": ["tool", "args", "verify"],
            },
        },
    },
    "required": ["goal", "steps"],
}

_PICK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"target_id": {"type": "string"}},
    "required": ["target_id"],
}


@dataclass
class Verify:
    window_title: str | None = None
    url_or_title_contains: str | None = None
    ax_contains: str | None = None
    display_number: str | None = None

    def any(self) -> bool:
        return bool(self.window_title or self.url_or_title_contains or self.ax_contains or self.display_number)


@dataclass
class Step:
    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    verify: Verify = field(default_factory=Verify)


@dataclass
class Plan:
    goal: str
    steps: list[Step] = field(default_factory=list)


class PlannerClient(Protocol):
    def make_plan(
        self,
        instruction: str,
        observation: dict[str, Any],
        *,
        failed: str | None = None,
    ) -> Plan: ...

    def pick_target(
        self,
        instruction: str,
        step: Step,
        candidates: list[Candidate],
        observation: dict[str, Any],
    ) -> str | None: ...


class WorkflowLoop:
    """Plan with Haiku, execute one Cua primitive, verify in Python."""

    def __init__(
        self,
        config: Config,
        ctx: AdapterContext,
        cancelled: Any,
        client: PlannerClient | None = None,
    ) -> None:
        self.config = config
        self.ctx = ctx
        self.cancelled = cancelled
        self.client = client or AnthropicPlanner(config)

    def run(self, task: Task) -> str:
        if self._needs_key():
            _LOG.warning("ANTHROPIC_API_KEY missing")
            return _MISSING_KEY
        target = self.ctx.target_for(task.app)
        observation = self._observe(target)
        try:
            plan = self.client.make_plan(task.instruction, observation)
        except Exception as exc:
            _LOG.warning("plan failed: %s", exc)
            return f"I couldn't plan that: {exc}"
        _LOG.info("plan goal=%s steps=%s", plan.goal, [s.tool for s in plan.steps])
        if not plan.steps:
            return "I couldn't plan that: empty step list"
        return self._execute_plan(task, plan, target, replanned=False)

    def _needs_key(self) -> bool:
        if self.config.anthropic_api_key:
            return False
        return isinstance(self.client, AnthropicPlanner)

    def _execute_plan(self, task: Task, plan: Plan, target: WindowTarget | None, *, replanned: bool) -> str:
        verified: list[str] = []
        for index, step in enumerate(plan.steps):
            if self.cancelled():
                return "stopped"
            if step.tool not in _TOOLS:
                _LOG.warning("unknown tool %s", step.tool)
                continue
            try:
                target = self._act(task, step, target)
            except CuaError as exc:
                _LOG.warning("step %s %s failed: %s", index + 1, step.tool, exc)
                return self._fail_or_replan(task, plan, index, target, verified, str(exc), replanned)
            if step.tool == "wait":
                time.sleep(float(step.args.get("seconds") or 0.4))
            settle = 2.4 if step.tool == "open_url" or step.tool == "press_key" or (
                step.tool == "type" and _wants_submit(step.args)
            ) else 0.0
            deadline = time.monotonic() + settle
            while True:
                target = self._refresh_window(target)
                observation = self._observe(target)
                ok, fact = check_verify(step.verify, target, observation)
                if ok or time.monotonic() >= deadline:
                    break
                time.sleep(0.35)
            _LOG.info("verify step %s tool=%s ok=%s fact=%s", index + 1, step.tool, ok, fact)
            if ok:
                if fact:
                    verified.append(fact)
                continue
            return self._fail_or_replan(
                task,
                plan,
                index,
                target,
                verified,
                fact or "verification failed",
                replanned,
            )
        if verified:
            return verified[-1]
        if target and (target.title or target.name):
            return target.title or target.name
        return plan.goal or "done"

    def _fail_or_replan(
        self,
        task: Task,
        plan: Plan,
        index: int,
        target: WindowTarget | None,
        verified: list[str],
        reason: str,
        replanned: bool,
    ) -> str:
        if replanned:
            prefix = verified[-1] + ". " if verified else ""
            return f"{prefix}I stopped after {plan.steps[index].tool}: {reason}".strip()
        observation = self._observe(target)
        failed = f"step {index + 1} {plan.steps[index].tool} did not verify: {reason}"
        _LOG.info("replan after %s", failed)
        try:
            nxt = self.client.make_plan(task.instruction, observation, failed=failed)
        except Exception as exc:
            _LOG.warning("replan failed: %s", exc)
            prefix = verified[-1] + ". " if verified else ""
            return f"{prefix}I couldn't finish: {reason}".strip()
        if not nxt.steps:
            prefix = verified[-1] + ". " if verified else ""
            return f"{prefix}I couldn't finish: {reason}".strip()
        return self._execute_plan(task, nxt, target, replanned=True)

    def _act(self, task: Task, step: Step, target: WindowTarget | None) -> WindowTarget | None:
        delivery = "foreground" if task.visible else "background"
        if step.tool == "launch_app":
            name = str(step.args.get("name") or step.args.get("app") or "")
            bundle = str(step.args.get("bundle_id") or "") or None
            resolved = resolve_app(name) if name else None
            if resolved:
                name, bundle = resolved[0], resolved[1] or bundle
            if not name and not bundle:
                raise CuaError("launch_app needs a name")
            found = self.ctx.target_for(name) if name else None
            if found is not None and found.pid:
                target = found
            else:
                target = self.ctx.cua.launch_app(name=name or None, bundle_id=bundle)
                if name:
                    self.ctx.remember(name, target)
            if target is not None and task.visible:
                try:
                    self.ctx.cua.bring_to_front(target)
                except CuaError:
                    pass
            _LOG.info("opened %s pid=%s window=%s", name, getattr(target, "pid", None), getattr(target, "window_id", None))
            time.sleep(0.35)
            return target
        if step.tool == "open_url":
            url = _as_url(str(step.args.get("url") or step.args.get("text") or ""))
            app = (target.name if target else "") or str(step.args.get("app") or "")
            try:
                self.ctx.cua.open_url(url, app=app or None)
            except TypeError:
                self.ctx.cua.open_url(url)
            time.sleep(0.6)
            target = self._refresh_window(target) or target
            if target is None:
                target = self.ctx.target_for() or self.ctx.target_for("safari")
            return target
        if target is None or not target.pid:
            target = self.ctx.target_for()
        if target is None or not target.pid:
            raise CuaError("no window to drive")
        if task.visible:
            try:
                self.ctx.cua.bring_to_front(target)
            except CuaError:
                pass
        if step.tool in {"click", "type"}:
            self._ensure_target_id(task, step, target)
        if step.tool == "click":
            idx = _element_index(step.args)
            self.ctx.cua.click(target, element_index=idx, delivery_mode=delivery)
            return target
        if step.tool == "type":
            text = str(step.args.get("text") or step.args.get("query") or "")
            idx = _element_index(step.args)
            self.ctx.cua.type_text(target, text, element_index=idx, delivery_mode=delivery)
            if _wants_submit(step.args):
                self.ctx.cua.press_key(target, str(step.args.get("key") or "enter"), delivery_mode=delivery)
            return target
        if step.tool == "press_key":
            self.ctx.cua.press_key(target, str(step.args.get("key") or "enter"), delivery_mode=delivery)
            return target
        return target

    def _ensure_target_id(self, task: Task, step: Step, target: WindowTarget) -> None:
        if _element_index(step.args) is not None:
            return
        try:
            self.ctx.cua.snapshot(target, include_screenshot=False)
        except CuaError:
            return
        candidates = parse_candidates(target)
        if not candidates:
            return
        observation = self._observe(target)
        cid = self.client.pick_target(task.instruction, step, candidates, observation)
        pick = next((c for c in candidates if c.cid == cid), None)
        if pick is None:
            hint = str(step.args.get("label") or step.args.get("text") or "")
            pick = _match_hint(candidates, hint)
        if pick is not None:
            step.args["element_index"] = pick.index
            step.args["target_id"] = pick.cid

    def _observe(self, target: WindowTarget | None) -> dict[str, Any]:
        if target is None or not target.pid:
            return {"app": None, "title": None, "tree_head": "", "elements": []}
        try:
            self.ctx.cua.snapshot(target, include_screenshot=False)
        except CuaError as exc:
            _LOG.debug("snapshot skipped: %s", exc)
        candidates = parse_candidates(target)
        return {
            "app": target.name,
            "title": target.title,
            "tree_head": (target.tree or "")[:1500],
            "elements": public_elements(candidates),
        }

    def _refresh_window(self, target: WindowTarget | None) -> WindowTarget | None:
        if target is None or not getattr(target, "pid", None):
            return target
        try:
            windows = self.ctx.cua.list_windows(target.pid)
        except (CuaError, AttributeError, TypeError):
            return target
        if not windows:
            return target
        win = windows[0]
        target.window_id = int(win.get("window_id") or win.get("id") or target.window_id or 0)
        if win.get("title"):
            target.title = str(win.get("title"))
        return target


def check_verify(verify: Verify, target: WindowTarget | None, observation: dict[str, Any]) -> tuple[bool, str]:
    if not verify.any():
        if target is not None and target.pid:
            return True, target.title or target.name or ""
        return True, ""
    blob = " ".join(
        str(part or "")
        for part in (
            getattr(target, "title", ""),
            getattr(target, "name", ""),
            observation.get("title"),
            observation.get("tree_head"),
            " ".join(str(el.get("label") or "") for el in (observation.get("elements") or [])[:40]),
        )
    ).lower()
    facts: list[str] = []
    shown = read_displayed_number(parse_candidates(target)) if target else None
    if verify.window_title:
        if not _place_match(verify.window_title, blob):
            return False, f"window is not {verify.window_title}"
        facts.append(target.title or verify.window_title if target else verify.window_title)
    if verify.url_or_title_contains:
        if not _place_match(verify.url_or_title_contains, blob):
            return False, f"not on {verify.url_or_title_contains}"
        facts.append(_speech_place(verify.url_or_title_contains, target))
    if verify.ax_contains:
        needle = verify.ax_contains.lower()
        in_tree = needle in blob
        in_display = bool(shown and (needle in shown.lower() or _same_number(shown, verify.ax_contains)))
        if not in_tree and not in_display:
            return False, f"screen does not show {verify.ax_contains}"
        facts.append(shown if in_display and shown else verify.ax_contains)
    if verify.display_number:
        if shown is None or not _same_number(shown, verify.display_number):
            return False, f"display is {shown or 'empty'}, expected {verify.display_number}"
        facts.append(shown)
    if not facts:
        return True, (target.title or target.name or "") if target else ""
    for fact in reversed(facts):
        if fact.replace(",", "").replace(".", "", 1).replace("-", "", 1).isdigit():
            return True, fact
    return True, facts[-1]


def parse_plan(raw: dict[str, Any] | Plan) -> Plan:
    if isinstance(raw, Plan):
        return raw
    steps: list[Step] = []
    for item in raw.get("steps") or []:
        if not isinstance(item, dict):
            continue
        tool = str(item.get("tool") or "").strip()
        if tool not in _TOOLS:
            continue
        args = item.get("args") if isinstance(item.get("args"), dict) else {}
        args = _clean_args(args)
        verify_raw = item.get("verify") if isinstance(item.get("verify"), dict) else {}
        steps.append(
            Step(
                tool=tool,
                args=dict(args),
                verify=Verify(
                    window_title=_opt_str(verify_raw.get("window_title")),
                    url_or_title_contains=_opt_str(verify_raw.get("url_or_title_contains")),
                    ax_contains=_opt_str(verify_raw.get("ax_contains")),
                    display_number=_opt_str(verify_raw.get("display_number")),
                ),
            )
        )
    return Plan(goal=str(raw.get("goal") or ""), steps=steps)


class AnthropicPlanner:
    def __init__(self, config: Config) -> None:
        self.config = config

    def make_plan(self, instruction: str, observation: dict[str, Any], *, failed: str | None = None) -> Plan:
        repair = f"Previous step failed: {failed}. Plan the remaining work only.\n" if failed else ""
        user = (
            f"{repair}Transcript (may be noisy ASR): {instruction}\n"
            f"Observation: {json.dumps(observation, ensure_ascii=False)[:2500]}\n"
            f"Known apps: {', '.join(sorted(APP_ALIASES)[:24])}"
        )
        data = _post_haiku(
            self.config.anthropic_api_key,
            self.config.planner_model,
            _PLAN_SYSTEM,
            user,
            _PLAN_SCHEMA,
        )
        return parse_plan(data)

    def pick_target(
        self,
        instruction: str,
        step: Step,
        candidates: list[Candidate],
        observation: dict[str, Any],
    ) -> str | None:
        if not candidates:
            return None
        hint = str(step.args.get("label") or step.args.get("text") or "")
        local = _match_hint(candidates, hint)
        if local is not None:
            return local.cid
        criteria = {c.cid: c.brief() for c in candidates[:40]}
        criteria["none"] = "No control matches"
        user = (
            f"Goal: {instruction}\n"
            f"Next tool: {step.tool} args={json.dumps(step.args)[:400]}\n"
            f"Window: {observation.get('title')}\n"
            f"Pick target_id from: {json.dumps(criteria, ensure_ascii=False)[:2500]}"
        )
        try:
            data = _post_haiku(
                self.config.anthropic_api_key,
                self.config.planner_model,
                "Pick the single accessibility control id for the next action. Return JSON.",
                user,
                _PICK_SCHEMA,
            )
        except Exception as exc:
            _LOG.warning("pick_target failed: %s", exc)
            return None
        cid = str(data.get("target_id") or "")
        return None if cid in {"", "none"} else cid


_PLAN_SYSTEM = """You plan Mac computer-use workflows as JSON.
Tools: launch_app, open_url, click, type, press_key, wait.
For web destinations prefer open_url with https.
When typing a URL or search query, set args.submit to true so Enter is pressed. Never finish a search on type alone.
Prefer typing a full calculator expression including = instead of many digit clicks. Do not set submit on calculator type.
Do not invent success. Verification happens in code after each step.
For math, set verify.display_number to the expected result. For web, set verify.url_or_title_contains to the site name (youtube, not Calculator).
Leave unused verify strings empty. Leave unused args empty. Do not set element_index unless you know it.
args.name for launch_app is the app name. args.text for type. args.url for open_url. args.key for press_key.
"""


def _post_haiku(api_key: str, model: str, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]:
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is missing")
    attempts: list[dict[str, Any]] = [
        {
            "model": model,
            "max_tokens": 2048,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "output_config": {"format": {"type": "json_schema", "schema": schema}},
        },
        {
            "model": model,
            "max_tokens": 2048,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "tools": [
                {
                    "name": "submit_plan",
                    "description": "Submit the computer-use plan",
                    "strict": True,
                    "input_schema": schema,
                }
            ],
            "tool_choice": {"type": "tool", "name": "submit_plan"},
        },
        {
            "model": model,
            "max_tokens": 2048,
            "system": system + " Reply with a JSON object only. No markdown.",
            "messages": [{"role": "user", "content": user + "\nReturn JSON with keys goal and steps."}],
        },
    ]
    last_error = "unknown"
    for payload in attempts:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=body,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=25) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last_error = f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='ignore')[:300]}"
            _LOG.warning("haiku %s", last_error)
            continue
        parsed = payload_from_message(raw)
        if parsed:
            return parsed
        types = [b.get("type") for b in (raw.get("content") or []) if isinstance(b, dict)]
        last_error = f"empty JSON stop={raw.get('stop_reason')} types={types}"
        _LOG.warning("haiku %s text=%r", last_error, _content_text(raw)[:240])
    raise RuntimeError(f"Haiku plan failed ({last_error})")


def payload_from_message(raw: dict[str, Any]) -> dict[str, Any]:
    for block in raw.get("content") or []:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "tool_use" and isinstance(block.get("input"), dict):
            return coerce_plan_dict(block["input"])
        if block.get("type") in {"text", "output_json"}:
            parsed = extract_json(str(block.get("text") or block.get("json") or ""))
            if parsed:
                return parsed
    return extract_json(_content_text(raw))


def coerce_plan_dict(obj: Any) -> dict[str, Any]:
    if isinstance(obj, list):
        return {"goal": "", "steps": obj}
    if not isinstance(obj, dict):
        return {}
    if isinstance(obj.get("steps"), list):
        return obj
    if isinstance(obj.get("plan"), dict):
        return coerce_plan_dict(obj["plan"])
    if obj.get("tool"):
        return {"goal": str(obj.get("goal") or ""), "steps": [obj]}
    return obj


def extract_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {}
    try:
        return coerce_plan_dict(json.loads(text))
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return coerce_plan_dict(json.loads(text[start : end + 1]))
        except json.JSONDecodeError:
            pass
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        try:
            return coerce_plan_dict(json.loads(text[start : end + 1]))
        except json.JSONDecodeError:
            return {}
    return {}


def _content_text(raw: dict[str, Any]) -> str:
    blocks = raw.get("content") or []
    parts: list[str] = []
    for block in blocks:
        if isinstance(block, dict) and block.get("type") in {"text", "output_json"}:
            parts.append(str(block.get("text") or block.get("json") or ""))
    return "\n".join(parts)


def _clean_args(args: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in args.items():
        if value in ("", None):
            continue
        if key == "element_index" and value in {0, "0"}:
            continue
        if key == "seconds" and value in {0, 0.0, "0"}:
            continue
        if key == "submit" and value is False:
            continue
        cleaned[key] = value
    return cleaned


def _opt_str(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _element_index(args: dict[str, Any]) -> int | None:
    raw = args.get("element_index")
    if raw is None:
        cid = str(args.get("target_id") or "")
        match = re.fullmatch(r"e(\d+)", cid)
        return int(match.group(1)) if match else None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _wants_submit(args: dict[str, Any]) -> bool:
    flag = args.get("submit")
    if flag is True or str(flag).lower() in {"true", "enter", "yes", "1"}:
        return True
    return False


def _as_url(raw: str) -> str:
    text = raw.strip()
    if not text:
        raise CuaError("open_url needs a URL")
    if re.match(r"^[a-z][a-z0-9+.-]*:", text, re.I):
        return text
    return "https://" + text


def _place_match(needle: str, blob: str) -> bool:
    needle = (needle or "").strip().lower()
    blob = (blob or "").lower()
    if not needle:
        return True
    if needle in blob:
        return True
    host = _host_token(needle)
    if host and host in blob:
        return True
    name = host.split(".")[0] if host else needle
    if len(name) >= 3 and re.search(rf"\b{re.escape(name)}\b", blob):
        return True
    return False


def _host_token(value: str) -> str:
    text = (value or "").strip().lower()
    text = re.sub(r"^https?://", "", text)
    text = text.split("/")[0]
    text = text.split("?")[0]
    if text.startswith("www."):
        text = text[4:]
    return text


def _match_hint(candidates: list[Candidate], hint: str) -> Candidate | None:
    needle = (hint or "").strip().lower()
    if not needle:
        return None
    for cand in candidates:
        if needle == cand.label.lower() or needle in cand.label.lower():
            return cand
    return None


def _speech_place(needle: str, target: WindowTarget | None) -> str:
    app = (target.name if target else "") or "Safari"
    return f"{app} is on {needle}"


def _same_number(left: str, right: str) -> bool:
    try:
        return float(left.replace(",", "")) == float(right.replace(",", ""))
    except ValueError:
        return left.replace(",", "") == right.replace(",", "")
