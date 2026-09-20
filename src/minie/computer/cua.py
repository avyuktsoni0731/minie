from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from minie.config import Config
from minie.log import get_logger

_LOG = get_logger()


class CuaError(RuntimeError):
    pass


class CuaNotAvailable(CuaError):
    pass


@dataclass
class WindowTarget:
    pid: int
    window_id: int
    name: str = ""
    title: str = ""
    snapshot_id: str | None = None
    elements: list[dict[str, Any]] = field(default_factory=list)
    tree: str = ""
    screenshot: Path | None = None


def _extract_json(text: str) -> Any:
    text = text.strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = min((i for i in (text.find("{"), text.find("[")) if i >= 0), default=-1)
    if start < 0:
        return {"raw": text}
    decoder = json.JSONDecoder()
    try:
        obj, _ = decoder.raw_decode(text[start:])
        return obj
    except json.JSONDecodeError:
        return {"raw": text}


def _unwrap(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload
    for key in ("structuredContent", "structured_content", "result", "data"):
        inner = payload.get(key)
        if isinstance(inner, (dict, list)) and inner:
            return inner
    content = payload.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") in {None, "text", "output_text"}:
                maybe = _extract_json(str(block.get("text") or block.get("output_text") or ""))
                if maybe and maybe != {"raw": block.get("text")}:
                    return maybe
    return payload


class CuaDriver:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.bin = config.cua_bin

    def available(self) -> bool:
        return shutil.which(self.bin) is not None

    def ensure_available(self) -> None:
        if self.config.dry_run:
            return
        if not self.available():
            raise CuaNotAvailable(
                "Cua Driver is not installed. Run:\n"
                '  /bin/bash -c "$(curl -fsSL https://cua.ai/driver/install.sh)"\n'
                "  open -n -g -a CuaDriver --args serve\n"
                "  cua-driver permissions grant"
            )

    def call(self, tool: str, args: dict[str, Any] | None = None, timeout: float = 40.0) -> Any:
        self.ensure_available()
        if self.config.dry_run:
            _LOG.info("dry-run cua %s %s", tool, args or {})
            return {"ok": True, "dry_run": True, "tool": tool, "args": args or {}}
        cmd = [self.bin, "call", tool]
        payload = json.dumps(args or {})
        if args:
            cmd.append(payload)
        extra: dict[str, Any] = {}
        screenshot_out = (args or {}).get("screenshot_out_file")
        if screenshot_out:
            extra["screenshot-out-file"] = screenshot_out
            cmd.extend(["--screenshot-out-file", str(screenshot_out)])
        _LOG.debug("cua %s %s", tool, args or {})
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except FileNotFoundError as exc:
            raise CuaNotAvailable(str(exc)) from exc
        except subprocess.TimeoutExpired as exc:
            raise CuaError(f"{tool} timed out") from exc
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()
            raise CuaError(f"{tool} failed: {err[:800]}")
        return _unwrap(_extract_json(proc.stdout))

    def list_apps(self) -> list[dict[str, Any]]:
        data = self.call("list_apps")
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("apps", "items", "applications"):
                if isinstance(data.get(key), list):
                    return data[key]
        return []

    def list_windows(self, pid: int | None = None) -> list[dict[str, Any]]:
        args: dict[str, Any] = {}
        if pid is not None:
            args["pid"] = pid
        data = self.call("list_windows", args or None)
        if isinstance(data, list):
            windows = data
        elif isinstance(data, dict):
            windows = data.get("windows") or data.get("items") or []
        else:
            windows = []
        if pid is not None:
            windows = [w for w in windows if int(w.get("pid") or 0) == pid]
        return windows

    def launch_app(self, name: str | None = None, bundle_id: str | None = None) -> WindowTarget | None:
        args: dict[str, Any] = {}
        if bundle_id:
            args["bundle_id"] = bundle_id
        elif name:
            args["name"] = name
        else:
            raise ValueError("launch_app needs a name or bundle_id")
        data = self.call("launch_app", args)
        if self.config.dry_run:
            return WindowTarget(pid=0, window_id=0, name=name or bundle_id or "")
        pid = int(data.get("pid") or 0)
        windows = data.get("windows") or []
        if not windows and pid:
            windows = self.list_windows(pid)
        if not windows:
            deadline = time.monotonic() + 2.5
            while time.monotonic() < deadline and pid:
                time.sleep(0.2)
                windows = self.list_windows(pid)
                if windows:
                    break
        if not windows:
            return WindowTarget(pid=pid, window_id=0, name=str(data.get("name") or name or ""))
        win = windows[0]
        return WindowTarget(
            pid=int(win.get("pid") or pid),
            window_id=int(win.get("window_id") or win.get("id") or 0),
            name=str(data.get("name") or name or ""),
            title=str(win.get("title") or ""),
        )

    def open_url(self, url: str, app: str | None = None) -> None:
        if not url:
            raise CuaError("open_url needs a URL")
        if self.config.dry_run:
            _LOG.info("dry-run open %s %s", app or "", url)
            return
        cmd = ["open"]
        if app:
            cmd.extend(["-a", app])
        cmd.append(url)
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except FileNotFoundError as exc:
            raise CuaError("open is not available") from exc
        except subprocess.TimeoutExpired as exc:
            raise CuaError("open timed out") from exc
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()
            raise CuaError(f"open failed: {err[:400]}")

    def snapshot(self, target: WindowTarget, include_screenshot: bool = True) -> WindowTarget:
        if not target.pid or not target.window_id:
            windows = self.list_windows(target.pid) if target.pid else self.list_windows()
            if not windows:
                raise CuaError("no window to snapshot")
            win = windows[0]
            target.pid = int(win.get("pid") or target.pid)
            target.window_id = int(win.get("window_id") or win.get("id") or 0)
            target.title = str(win.get("title") or target.title)
        tmp: Path | None = None
        args: dict[str, Any] = {
            "pid": target.pid,
            "window_id": target.window_id,
            "include_screenshot": include_screenshot,
            "max_elements": 400,
            "max_dimension": 1024,
        }
        if include_screenshot:
            tmp = Path(tempfile.gettempdir()) / f"minie-{target.pid}-{target.window_id}.png"
            args["screenshot_out_file"] = str(tmp)
        data = self.call("get_window_state", args, timeout=50.0)
        if self.config.dry_run:
            target.tree = "(dry-run snapshot)"
            return target
        elements = data.get("elements") if isinstance(data, dict) else None
        target.elements = elements if isinstance(elements, list) else []
        target.tree = str(
            (data.get("tree_markdown") if isinstance(data, dict) else None)
            or (data.get("tree") if isinstance(data, dict) else None)
            or _format_elements(target.elements)
        )
        target.snapshot_id = (
            str(data.get("snapshot_id")) if isinstance(data, dict) and data.get("snapshot_id") else None
        )
        if tmp and tmp.exists():
            target.screenshot = tmp
        elif isinstance(data, dict) and data.get("screenshot_file_path"):
            target.screenshot = Path(str(data["screenshot_file_path"]))
        return target

    def click(
        self,
        target: WindowTarget,
        *,
        element_index: int | None = None,
        x: float | None = None,
        y: float | None = None,
        delivery_mode: str = "background",
    ) -> Any:
        args: dict[str, Any] = {
            "pid": target.pid,
            "window_id": target.window_id,
            "delivery_mode": delivery_mode,
        }
        if element_index is not None:
            args["element_index"] = element_index
            if target.snapshot_id:
                args["snapshot_id"] = target.snapshot_id
        if x is not None and y is not None:
            args["x"] = x
            args["y"] = y
        try:
            return self.call("click", args)
        except CuaError:
            if delivery_mode == "background":
                _LOG.warning("background click rejected; retrying foreground")
                args["delivery_mode"] = "foreground"
                return self.call("click", args)
            raise

    def type_text(
        self,
        target: WindowTarget,
        text: str,
        *,
        element_index: int | None = None,
        delivery_mode: str = "background",
    ) -> Any:
        args: dict[str, Any] = {
            "pid": target.pid,
            "window_id": target.window_id,
            "text": text,
            "delivery_mode": delivery_mode,
        }
        if element_index is not None:
            args["element_index"] = element_index
            if target.snapshot_id:
                args["snapshot_id"] = target.snapshot_id
        try:
            return self.call("type_text", args)
        except CuaError:
            if delivery_mode == "background":
                args["delivery_mode"] = "foreground"
                return self.call("type_text", args)
            raise

    def bring_to_front(self, target: WindowTarget) -> Any:
        args: dict[str, Any] = {"pid": target.pid}
        if target.window_id:
            args["window_id"] = target.window_id
        return self.call("bring_to_front", args)

    def press_key(
        self,
        target: WindowTarget,
        key: str,
        *,
        delivery_mode: str = "background",
    ) -> Any:
        args = {
            "pid": target.pid,
            "window_id": target.window_id,
            "key": key,
            "delivery_mode": delivery_mode,
        }
        try:
            return self.call("press_key", args)
        except CuaError:
            if delivery_mode == "background":
                args["delivery_mode"] = "foreground"
                return self.call("press_key", args)
            raise

    def find_element(self, target: WindowTarget, *labels: str) -> int | None:
        wanted = [label.lower().strip() for label in labels if label]
        for el in target.elements:
            idx = el.get("element_index")
            blob = " ".join(
                str(el.get(k) or "")
                for k in ("label", "title", "value", "role", "name", "description")
            ).lower()
            if any(w == blob.strip() or w in blob.split() or blob.strip() == w for w in wanted):
                return int(idx) if idx is not None else None
        tree = target.tree.lower()
        for label in wanted:
            for line in tree.splitlines():
                if label in line.lower() and "element_index" in line.lower():
                    match = re.search(r"element_index\s+(\d+)", line, re.I)
                    if match:
                        return int(match.group(1))
        return None


def _format_elements(elements: list[dict[str, Any]]) -> str:
    lines = []
    for el in elements[:400]:
        idx = el.get("element_index")
        role = el.get("role") or ""
        label = el.get("label") or el.get("title") or el.get("value") or ""
        lines.append(f"[element_index {idx}] {role} {label}".strip())
    return "\n".join(lines)
