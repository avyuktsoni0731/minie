from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass

from minie.agent.adapters.base import Adapter
from minie.agent.intent import parse_call
from minie.agent.task import AdapterContext, AdapterResult, Task
from minie.audio.phrases import normalize
from minie.computer.cua import CuaError
from minie.config import STATUS_ACTING
from minie.log import get_logger

_LOG = get_logger()

_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_PHONE = re.compile(r"^\+?[0-9][0-9\-().\s]{6,}$")


@dataclass
class ContactMatch:
    name: str
    destination: str


class FaceTimeAdapter(Adapter):
    name = "facetime"

    def claim(self, instruction: str, ctx: AdapterContext | None = None) -> Task | None:
        call = parse_call(instruction)
        if call is None or not call.query:
            return None
        audio = bool(call.keys)
        return Task(
            instruction=instruction,
            goal=f"Start FaceTime with {call.query}",
            app="FaceTime",
            success_criteria=f"FaceTime is showing a call sheet for {call.query}",
            risk="confirm",
            visible=True,
            adapter=self.name,
            payload={"who": call.query, "audio": audio},
        )

    def execute(self, task: Task, ctx: AdapterContext) -> AdapterResult:
        who = str(task.payload.get("who") or "").strip()
        audio = bool(task.payload.get("audio"))
        if not who:
            return AdapterResult(ok=False, speech="I didn't catch who to call.")
        match = resolve_contact(who, ctx.config.nicknames)
        if match is None:
            return AdapterResult(
                ok=False,
                speech=f"I couldn't find {who} in Contacts. Add a nickname in MINIE_NICKNAMES.",
            )
        url = _facetime_url(match.destination, audio=audio)
        ctx.set_status(STATUS_ACTING)
        if ctx.config.dry_run:
            _LOG.info("dry-run FaceTime %s", url)
            return AdapterResult(
                ok=True,
                speech=f"I'd open FaceTime for {match.name}. Confirm Call there.",
                summary=url,
            )
        try:
            ctx.cua.open_url(url)
        except CuaError as exc:
            _LOG.warning("FaceTime URL failed: %s", exc)
            return AdapterResult(
                ok=False,
                speech=f"I couldn't open FaceTime for {match.name}.",
                fallback_gui=True,
            )
        try:
            target = ctx.cua.launch_app(name="FaceTime", bundle_id="com.apple.FaceTime")
            ctx.remember("facetime", target)
            if target is not None:
                ctx.cua.bring_to_front(target)
        except CuaError as exc:
            _LOG.debug("FaceTime bring_to_front: %s", exc)
        _LOG.info("FaceTime sheet for %s via %s", match.name, url)
        return AdapterResult(
            ok=True,
            speech=f"FaceTime is ready for {match.name}. Confirm Call if that's right.",
            summary=f"Opened {url}",
        )


def resolve_contact(query: str, nicknames: dict[str, str] | None = None) -> ContactMatch | None:
    raw = normalize(query)
    if not raw:
        return None
    mapped = (nicknames or {}).get(raw, raw)
    if _is_destination(mapped):
        return ContactMatch(name=query, destination=_clean_dest(mapped))
    found = _contacts_lookup(mapped)
    if found:
        return found
    if mapped.lower() != raw:
        found = _contacts_lookup(raw)
        if found:
            return found
    return None


def _is_destination(value: str) -> bool:
    cleaned = value.strip()
    return bool(_EMAIL.match(cleaned) or _PHONE.match(cleaned))


def _clean_dest(value: str) -> str:
    if "@" in value:
        return value.strip()
    return re.sub(r"[^\d+]", "", value)


def _facetime_url(destination: str, *, audio: bool) -> str:
    scheme = "facetime-audio" if audio else "facetime"
    return f"{scheme}://{destination}"


def _contacts_lookup(query: str) -> ContactMatch | None:
    safe = query.replace("\\", "").replace('"', "")
    if not safe:
        return None
    script = f'''
tell application "Contacts"
    set q to "{safe}"
    set hits to (people whose name contains q)
    if (count of hits) is 0 then set hits to (people whose nickname contains q)
    if (count of hits) is 0 then return ""
    set p to item 1 of hits
    set nm to name of p
    set dest to ""
    try
        set dest to value of first email of p
    end try
    if dest is "" then
        try
            set dest to value of first phone of p
        end try
    end if
    return nm & tab & dest
end tell
'''
    try:
        proc = subprocess.run(
            ["osascript"],
            input=script,
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        _LOG.warning("Contacts lookup failed for %r", query)
        return None
    if proc.returncode != 0:
        _LOG.info("Contacts lookup denied or empty: %s", (proc.stderr or "")[:200])
        return None
    line = (proc.stdout or "").strip()
    if not line:
        return None
    if "\t" in line:
        name, dest = line.split("\t", 1)
    else:
        name, dest = line, ""
    dest = dest.strip()
    if not dest:
        return None
    return ContactMatch(name=name.strip() or query, destination=_clean_dest(dest) if not _EMAIL.match(dest) else dest.strip())
