from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

from minie.audio.phrases import normalize
from minie.computer.cua import WindowTarget

_CLICK_ROLES = {
    "button",
    "link",
    "checkbox",
    "radio",
    "radiobutton",
    "tab",
    "menuitem",
    "menu item",
    "pop up button",
    "popupbutton",
    "cell",
    "image",
    "disclosure triangle",
    "incrementor",
    "decrementor",
}
_TYPE_ROLES = {
    "textfield",
    "text field",
    "searchfield",
    "search text field",
    "textarea",
    "text area",
    "combobox",
    "combo box",
    "securetextfield",
    "secure text field",
}


@dataclass
class Candidate:
    cid: str
    index: int
    role: str
    label: str
    context: str = ""
    score: float = 0.0

    def brief(self) -> str:
        label = (self.label or "(unlabeled)")[:80]
        return f"{self.role}: {label}"


def parse_candidates(target: WindowTarget, limit: int = 80) -> list[Candidate]:
    found: list[Candidate] = []
    seen: set[int] = set()
    for el in target.elements or []:
        idx = el.get("element_index")
        if idx is None:
            continue
        try:
            index = int(idx)
        except (TypeError, ValueError):
            continue
        if index in seen:
            continue
        seen.add(index)
        role = str(el.get("role") or el.get("role_description") or "").strip().lower()
        label = str(
            el.get("label") or el.get("title") or el.get("value") or el.get("name") or el.get("description") or ""
        ).strip()
        found.append(
            Candidate(
                cid=f"e{index}",
                index=index,
                role=role or "unknown",
                label=label,
            )
        )
        if len(found) >= 400:
            break
    if not found and target.tree:
        found = _from_tree(target.tree)
    for i, cand in enumerate(found):
        start = max(0, i - 2)
        end = min(len(found), i + 4)
        cand.context = " | ".join(c.label for c in found[start:end] if c.label)[:160]
    return found[:limit]


def _from_tree(tree: str) -> list[Candidate]:
    found: list[Candidate] = []
    seen: set[int] = set()
    for line in tree.splitlines():
        match = re.search(r"element_index\s+(\d+)", line, re.I)
        if not match:
            continue
        index = int(match.group(1))
        if index in seen:
            continue
        seen.add(index)
        role_match = re.search(
            r"\b(button|link|text field|search text field|check box|checkbox|radio|tab|menu item|cell|combo box)\b",
            line,
            re.I,
        )
        label = re.sub(r"\[element_index\s+\d+\]", "", line, flags=re.I).strip()
        found.append(
            Candidate(
                cid=f"e{index}",
                index=index,
                role=(role_match.group(1).lower() if role_match else "unknown"),
                label=label[:80],
            )
        )
    return found


def is_clickable(role: str) -> bool:
    compact = role.replace(" ", "")
    return any(r.replace(" ", "") in compact or r in role for r in _CLICK_ROLES) or "button" in role


def is_typeable(role: str) -> bool:
    compact = role.replace(" ", "")
    return any(r.replace(" ", "") in compact or r in role for r in _TYPE_ROLES) or "field" in role


def action_for_role(role: str) -> str:
    if is_typeable(role):
        return "type"
    return "click"


def token_matches_label(token: str, label: str) -> bool:
    raw = (label or "").strip()
    tok = (token or "").strip()
    lab = normalize(label)
    if tok in {"*", "x", "X", "×"}:
        if "×" in raw or raw in {"*", "x", "X", "×"} or "multipl" in lab or lab in {"times", "x"}:
            return True
    if tok == "=":
        if raw in {"=", "equals", "equal"} or lab in {"equals", "equal"}:
            return True
    if tok == "+":
        if "+" in raw or lab in {"add", "plus", "addition"}:
            return True
    if tok in {"-", "−"}:
        if raw in {"-", "−", "–"} or lab in {"minus", "subtract", "subtraction"}:
            return True
    if tok in {"/", "÷"}:
        if raw in {"/", "÷"} or "divid" in lab:
            return True
    tok_n = normalize(token)
    if not tok_n:
        return False
    if lab == tok_n or tok_n in lab.split() or lab in tok_n:
        return True
    aliases = {
        "*": {"multiply", "times"},
        "+": {"add", "plus"},
        "-": {"minus", "subtract"},
        "/": {"divide"},
        "=": {"equals", "equal"},
    }
    return lab in aliases.get(tok, set()) or lab in aliases.get(tok_n, set())


def rank_candidates(candidates: list[Candidate], tokens: list[str], goal: str) -> list[Candidate]:
    nxt = tokens[0] if tokens else ""
    later = tokens[1:]
    goal_norm = normalize(goal)
    scored: list[Candidate] = []
    for cand in candidates:
        label = normalize(cand.label)
        score = 0.0
        if nxt and token_matches_label(nxt, cand.label):
            score += 12.0
        elif any(token_matches_label(tok, cand.label) for tok in later):
            score += 0.3
        if label and label in goal_norm:
            score += 1.0
        if nxt and not normalize(nxt) and token_matches_label(nxt, cand.label):
            score += 4.0
        elif nxt and normalize(nxt) and normalize(nxt) in label:
            score += 2.0
        if is_clickable(cand.role):
            score += 0.4
        if is_typeable(cand.role) and len(tokens) > 1:
            score += 0.8
        cand.score = score
        scored.append(cand)
    scored.sort(key=lambda c: (-c.score, c.index))
    return scored


def read_displayed_number(candidates: list[Candidate]) -> str | None:
    """Largest number in the tree that is not a 0–9 keypad button."""
    best: str | None = None
    best_len = 0
    for cand in candidates:
        compact = (cand.label or "").replace(",", "").replace(" ", "").strip()
        if not re.fullmatch(r"-?\d+(?:\.\d+)?", compact):
            continue
        if is_clickable(cand.role) and len(compact.lstrip("-")) <= 2:
            continue
        if len(compact) >= best_len:
            best = compact
            best_len = len(compact)
    return best


def fingerprint_tree(target: WindowTarget) -> str:
    blob = f"{target.pid}|{target.window_id}|{target.title}|{(target.tree or '')[:2000]}"
    return hashlib.sha256(blob.encode("utf-8", errors="ignore")).hexdigest()[:16]


def public_elements(candidates: list[Candidate], cap: int = 40) -> list[dict[str, Any]]:
    return [
        {"id": c.cid, "role": c.role, "label": c.label[:80], "context": c.context[:120]}
        for c in candidates[:cap]
    ]
