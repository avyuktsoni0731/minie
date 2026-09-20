from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass

from minie.agent.structure import Candidate, action_for_role, is_clickable, is_typeable, rank_candidates, token_matches_label
from minie.config import Config
from minie.log import get_logger

_LOG = get_logger()

_SENSITIVE = ("delete", "send", "purchase", "buy", "pay", "trash", "erase", "call", "password")


@dataclass
class Decision:
    action: str
    target: Candidate | None = None
    text: str = ""
    done: float = 0.0
    risk: float = 0.0
    confidence: float = 0.0
    source: str = "local"


def decide(
    config: Config,
    *,
    goal: str,
    remaining: list[str],
    candidates: list[Candidate],
    title: str,
    tree: str,
) -> Decision:
    local = _decide_local(goal, remaining, candidates, title)
    if not config.typesafe_api_key:
        return local
    try:
        remote = _decide_jev(config, goal=goal, remaining=remaining, candidates=candidates, title=title, tree=tree)
    except Exception as exc:
        _LOG.warning("Jev unavailable (%s); using local structure ranker", exc)
        return local
    return _compat(remote, candidates, remaining) if remote.target or remote.action == "done" else local


def _decide_local(goal: str, remaining: list[str], candidates: list[Candidate], title: str) -> Decision:
    if not remaining:
        return Decision(action="done", done=0.95, confidence=0.9, source="local")
    ranked = rank_candidates(candidates, remaining, goal)
    nxt = remaining[0]
    exact = next((c for c in ranked if token_matches_label(nxt, c.label)), None)
    if exact is not None:
        action = action_for_role(exact.role)
        if action == "type" and len(nxt) <= 2 and is_clickable(exact.role):
            action = "click"
        return Decision(
            action=action,
            target=exact,
            text=" ".join(remaining) if action == "type" else "",
            done=0.05,
            risk=_risk(exact),
            confidence=0.85,
            source="local",
        )
    if ranked and ranked[0].score >= 2.0 and token_matches_label(nxt, ranked[0].label):
        pick = ranked[0]
        action = action_for_role(pick.role)
        text = " ".join(remaining) if action == "type" else ""
        return Decision(
            action=action,
            target=pick,
            text=text,
            done=0.05,
            risk=_risk(pick),
            confidence=min(0.8, 0.4 + pick.score / 20.0),
            source="local",
        )
    field = next((c for c in candidates if is_typeable(c.role)), None)
    if field is not None and any(len(t) > 1 for t in remaining):
        return Decision(
            action="type",
            target=field,
            text=" ".join(remaining),
            done=0.05,
            risk=_risk(field),
            confidence=0.55,
            source="local",
        )
    return Decision(action="wait", done=0.1, confidence=0.2, source="local")


def _decide_jev(
    config: Config,
    *,
    goal: str,
    remaining: list[str],
    candidates: list[Candidate],
    title: str,
    tree: str,
) -> Decision:
    shortlist = rank_candidates(candidates, remaining, goal)[:40]
    criteria = {c.cid: c.brief() for c in shortlist}
    criteria["none"] = "No control on this screen matches the next step"
    payload = {
        "model": config.typesafe_model,
        "state": {
            "goal": goal,
            "remaining": remaining,
            "window": title,
            "tree_head": (tree or "")[:1500],
            "elements": [{"id": c.cid, "role": c.role, "label": c.label[:80]} for c in shortlist],
        },
        "questions": {
            "target": {
                "type": "choice",
                "instructions": "Which control should be handled next to advance the goal?",
                "criteria": criteria,
            },
            "action": {
                "type": "choice",
                "instructions": "Which primitive action should run on that control?",
                "criteria": {
                    "click": "Activate a button, link, or cell",
                    "type": "Enter text into a field",
                    "press_enter": "Submit with the return key",
                    "wait": "The UI is still loading",
                    "done": "The goal is already complete",
                },
            },
            "done": {
                "type": "noul",
                "instructions": "The visible UI already satisfies the user's goal",
            },
            "risk": {
                "type": "noul",
                "instructions": "The next action sends, deletes, pays, or places a call",
            },
        },
    }
    data = _post_system_one(config.typesafe_api_key, payload)
    choices = data.get("choices") or {}
    nouls = data.get("nouls") or {}
    target_id = _choice_value(choices.get("target"))
    action = _choice_value(choices.get("action")) or "click"
    done = float((nouls.get("done") or {}).get("noul") or 0.0)
    risk = float((nouls.get("risk") or {}).get("noul") or 0.0)
    confidence = float((choices.get("target") or {}).get("confidence") or 0.0)
    target = next((c for c in shortlist if c.cid == target_id), None)
    text = " ".join(remaining) if action == "type" else ""
    return Decision(
        action=action,
        target=target,
        text=text,
        done=done,
        risk=risk,
        confidence=confidence,
        source="jev",
    )


def _compat(decision: Decision, candidates: list[Candidate], remaining: list[str]) -> Decision:
    if decision.action == "done" or decision.done >= 0.8:
        decision.action = "done"
        return decision
    if decision.target is None:
        return decision
    allowed = action_for_role(decision.target.role)
    if decision.action == "type" and not is_typeable(decision.target.role):
        decision.action = "click" if is_clickable(decision.target.role) else allowed
        decision.text = ""
    if decision.action == "click" and is_typeable(decision.target.role) and remaining:
        decision.action = "type"
        decision.text = " ".join(remaining)
    return decision


def _risk(candidate: Candidate) -> float:
    blob = f"{candidate.label} {candidate.role}".lower()
    return 0.85 if any(word in blob for word in _SENSITIVE) else 0.05


def _choice_value(block: dict | None) -> str | None:
    if not isinstance(block, dict):
        return None
    value = block.get("choice") or block.get("value")
    return str(value) if value not in (None, "none") else None


def _post_system_one(api_key: str, payload: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "https://api.typesafe.ai/v1/systemone",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")[:300]
        raise RuntimeError(f"Jev HTTP {exc.code}: {detail}") from exc
    data = json.loads(raw) if raw else {}
    if isinstance(data.get("data"), dict):
        return data["data"]
    return data
