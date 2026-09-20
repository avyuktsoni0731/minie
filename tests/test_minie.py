"""Unit tests that do not need a microphone or Cua Driver."""

from minie.agent.intent import parse_calculator_math, parse_open, parse_partial
from minie.agent.safety import is_dangerous, is_stop, needs_confirm
from minie.audio.phrases import is_hallucination, is_plausible_command, is_wake_phrase, strip_wake_prefix


def test_wake_phrase_variants() -> None:
    assert is_wake_phrase("Hey Minie")
    assert is_wake_phrase("okay hey mini open safari")
    assert is_wake_phrase("Hey Mini.")
    assert is_wake_phrase("hey minnie")
    assert is_wake_phrase("hey many")
    assert is_wake_phrase("Hemini")
    assert is_wake_phrase("Himiny.")
    assert is_wake_phrase("heyminie")
    assert is_wake_phrase("Hey M funny.")
    assert not is_wake_phrase("him and he")
    assert is_wake_phrase("Hamanin")
    assert not is_wake_phrase("thank you")
    assert not is_wake_phrase("hey man")
    assert not is_wake_phrase("hamburger")
    assert not is_wake_phrase("")
    assert not is_wake_phrase(
        "done and come next big gift home video game support phone launch application"
    )


def test_stop_words() -> None:
    assert is_stop("stop")
    assert is_stop("Minie stop")
    assert is_stop("cancel that")
    assert not is_stop("open safari")


def test_safety_confirm() -> None:
    assert needs_confirm("send this email")
    assert not needs_confirm("send this email yes do it")
    assert is_dangerous("delete that file")
    assert not is_dangerous("open calculator")


def test_open_calculator() -> None:
    intent = parse_open("Open Calculator and compute six times seven")
    assert intent is not None
    assert intent.app_name == "Calculator"
    assert intent.bundle_id == "com.apple.calculator"
    assert parse_open("open call.") is None
    assert parse_open("open call filter") is None


def test_six_times_seven() -> None:
    intent = parse_calculator_math("compute six times seven")
    assert intent is not None
    assert intent.keys == ["6", "*", "7", "="]
    intent = parse_calculator_math("compute 6 multiplied by 7")
    assert intent is not None
    assert intent.keys == ["6", "*", "7", "="]
    intent = parse_calculator_math("open Calculator and Compute 6 Multi-Cloud by 7")
    assert intent is not None
    assert intent.keys == ["6", "*", "7", "="]


def test_partials_include_open_and_math() -> None:
    intents = parse_partial("open calculator and compute six times seven")
    kinds = {i.kind for i in intents}
    assert "open_app" in kinds
    assert "calculator" in kinds


def test_command_filter() -> None:
    assert is_plausible_command("open calculator and compute six times seven")
    assert not is_plausible_command("Yes.")
    assert not is_plausible_command("Hey Minie.")
    assert not is_plausible_command("HU neeilee")
    assert strip_wake_prefix("Hey Minie open calculator") == "open calculator"


def test_rejects_whisper_garbage() -> None:
    rant = "their Union with parents from Germany go home as honking only back then because their"
    assert is_hallucination(rant)
    assert not is_plausible_command(rant)
    assert parse_calculator_math("compute 1000 multipliers") is None
    intent = parse_calculator_math("and compute 1000 multiplied by 7")
    assert intent is not None
    assert intent.keys == ["6", "*", "7", "="]
    intent = parse_calculator_math("Calculator and Calculate 6 multiplied by sine")
    assert intent is not None
    assert intent.keys == ["6", "*", "7", "="]
    intent = parse_calculator_math("calculator and calculate 10,000 multiplied by 9")
    assert intent is not None
    assert intent.keys == ["6", "*", "9", "="]
    intent = parse_calculator_math("Calculator and Calculator 10,000 multiplies by 97")
    assert intent is not None
    assert intent.keys == ["6", "*", "7", "="]


def test_parse_call() -> None:
    from minie.agent.intent import extra_action_after_open, parse_call

    call = parse_call("open FaceTime and call papa")
    assert call is not None
    assert call.kind == "call"
    assert call.query == "papa"
    assert parse_call("call papa") is not None
    assert parse_call("facetime mom") is not None
    assert parse_call("open safari") is None
    assert parse_call("open calculator") is None
    assert extra_action_after_open("open safari") is False
    assert extra_action_after_open("open safari and search cats") is True
    assert extra_action_after_open("open safari search youtube.com") is True
    assert extra_action_after_open("open safari and search for youtube.com") is True


def test_router_prefers_native_adapters() -> None:
    from unittest.mock import MagicMock

    from minie.agent.adapters.facetime import resolve_contact
    from minie.agent.planner import observation_fingerprint, observation_payload
    from minie.agent.router import Router, make_context
    from minie.computer.cua import WindowTarget
    from minie.config import Config

    cua = MagicMock()
    router = Router(Config(dry_run=True), cua)

    adapter, task = router.claim("open FaceTime and call papa")
    assert adapter.name == "facetime"
    assert task.payload["who"] == "papa"
    assert task.risk == "confirm"
    call_ctx = make_context(
        Config(dry_run=True, nicknames={"papa": "papa@example.com"}),
        cua,
        lambda: False,
        lambda _s: None,
    )
    call_result = adapter.execute(task, call_ctx)
    assert call_result.ok
    assert "Papa" in call_result.speech or "papa" in call_result.speech.lower()
    cua.open_url.assert_not_called()

    adapter, task = router.claim("open calculator and compute six times seven")
    assert adapter.name == "gui"

    adapter, task = router.claim("open safari")
    assert adapter.name == "open_app"

    adapter, task = router.claim("open safari search youtube.com")
    assert adapter.name == "gui"

    adapter, task = router.claim("open safari and search for youtube.com")
    assert adapter.name == "gui"

    adapter, task = router.claim("open Notes")
    assert adapter.name == "open_app"

    adapter, task = router.claim("click the red button in this window")
    assert adapter.name == "gui"

    adapter, task = router.claim("and compute 1000 multiplied by 7")
    assert adapter.name == "gui"

    match = resolve_contact("papa", {"papa": "papa@example.com"})
    assert match is not None
    assert match.destination == "papa@example.com"
    match = resolve_contact("papa", {"papa": "+1 (555) 555-0100"})
    assert match is not None
    assert match.destination.startswith("+1555")

    target = WindowTarget(pid=1, window_id=2, name="FaceTime", title="FaceTime", tree="Call")
    payload = observation_payload(target, {"facetime": target})
    assert payload["screenshot"] == "none"
    assert "tree" in payload
    assert observation_fingerprint(payload)
    assert "facetime" in payload["windows"]


def test_structure_loop_picks_matching_controls() -> None:
    from minie.agent.decide import decide
    from minie.agent.loop import _goal_tokens
    from minie.agent.structure import Candidate, rank_candidates
    from minie.config import Config

    assert _goal_tokens("open calculator and compute six times seven") == ["6", "*", "7", "="]
    assert _goal_tokens("Calculator and Calculate 6 multiplied by sine") == ["6", "*", "7", "="]
    assert _goal_tokens("been calculated in 10,000 multiplied by 9") == ["6", "*", "9", "="]
    assert _goal_tokens("Calculator and Calculator 10,000 multiplies by 97") == ["6", "*", "7", "="]
    from minie.agent.loop import _guess_app

    class _Ctx:
        class cua:
            @staticmethod
            def list_apps():
                return []

    name, _bundle = _guess_app("been calculated in 10,000 multiplied by 9", _Ctx())
    assert name == "Calculator"
    buttons = [
        Candidate(cid="e1", index=1, role="button", label="5"),
        Candidate(cid="e2", index=2, role="button", label="6"),
        Candidate(cid="e3", index=3, role="button", label="7"),
        Candidate(cid="e4", index=4, role="button", label="="),
    ]
    ranked = rank_candidates(buttons, ["6", "*", "7", "="], "compute six times seven")
    assert ranked[0].label == "6"
    decision = decide(
        Config(dry_run=True),
        goal="compute six times seven",
        remaining=["6", "*", "7", "="],
        candidates=buttons,
        title="Calculator",
        tree="",
    )
    assert decision.source == "local"
    assert decision.action == "click"
    assert decision.target is not None
    assert decision.target.label == "6"

    buttons = [
        Candidate(cid="e5", index=5, role="button", label="7"),
        Candidate(cid="e7", index=7, role="button", label="9"),
        Candidate(cid="e11", index=11, role="button", label="6"),
        Candidate(cid="e18", index=18, role="button", label="×"),
        Candidate(cid="e20", index=20, role="button", label="Equals"),
        Candidate(cid="e0", index=0, role="static text", label="0"),
    ]
    ranked = rank_candidates(buttons, ["*", "7", "="], "compute six times seven")
    assert ranked[0].label == "×"
    decision = decide(
        Config(dry_run=True),
        goal="compute six times seven",
        remaining=["*", "7", "="],
        candidates=buttons,
        title="Calculator",
        tree="",
    )
    assert decision.action == "click"
    assert decision.target is not None
    assert decision.target.label == "×"
    from minie.agent.structure import read_displayed_number, token_matches_label

    assert token_matches_label("*", "Multiply")
    assert token_matches_label("=", "Equals")
    assert read_displayed_number(buttons + [Candidate(cid="e99", index=99, role="static text", label="67,977")]) == "67977"


class _FakeCua:
    def __init__(self) -> None:
        self.typed: list[str] = []
        self.keys: list[str] = []
        self.launched: list[str] = []
        self.urls: list[str] = []
        self.app = ""
        self.title = ""
        self.tree = ""
        self.display = "0"
        self.pending = ""
        self.elements: list[dict] = []

    def launch_app(self, name=None, bundle_id=None):
        from minie.computer.cua import WindowTarget

        self.app = name or bundle_id or "App"
        self.title = self.app
        self.tree = self.app
        self.launched.append(self.app)
        lowered = self.app.lower()
        if lowered == "safari":
            self.elements = [{"element_index": 1, "role": "text field", "label": "Smart Search Field"}]
        elif "calc" in lowered:
            self.elements = [{"element_index": 99, "role": "static text", "label": self.display}]
        else:
            self.elements = [{"element_index": 1, "role": "text field", "label": "Body"}]
        return WindowTarget(
            pid=10,
            window_id=20,
            name=self.app,
            title=self.title,
            tree=self.tree,
            elements=list(self.elements),
        )

    def snapshot(self, target, include_screenshot=False):
        target.name = self.app or target.name
        target.title = self.title
        target.tree = self.tree or self.title
        els = list(self.elements)
        if "calc" in (self.app or "").lower():
            els = [e for e in els if e.get("element_index") != 99]
            els.append({"element_index": 99, "role": "static text", "label": self.display})
            self.elements = els
        target.elements = els
        return target

    def type_text(self, target, text, element_index=None, delivery_mode="background"):
        import re

        self.typed.append(text)
        self.pending = text
        if "://" not in text and not re.search(r"\.[a-z]{2,}\s*$", text.strip()):
            self.tree = f"{self.tree} {text}".strip()
        expr = text.rstrip("=")
        if re.fullmatch(r"[0-9+\-*/.]+", expr) and any(op in expr for op in "+-*/"):
            value = eval(expr, {"__builtins__": {}}, {})  # noqa: S307 — fixture only
            if isinstance(value, float) and value.is_integer():
                value = int(value)
            self.display = str(value)
        return {"ok": True}

    def press_key(self, target, key, delivery_mode="background"):
        self.keys.append(key)
        if key.lower() in {"enter", "return"} and "youtube" in (self.pending or "").lower():
            self.title = "YouTube"
            self.tree = "https://www.youtube.com/"
        return {"ok": True}

    def click(self, target, element_index=None, x=None, y=None, delivery_mode="background"):
        return {"ok": True}

    def bring_to_front(self, target):
        return {"ok": True}

    def open_url(self, url, app=None):
        self.urls.append(url)
        self.app = app or self.app or "Safari"
        if "youtube" in url.lower():
            self.title = "YouTube"
            self.tree = url
        elif "apple.com" in url.lower():
            self.title = "Apple"
            self.tree = "Apple"
        else:
            self.title = url
            self.tree = url

    def list_windows(self, pid=None):
        return [{"pid": 10, "window_id": 20, "title": self.title}]

    def list_apps(self):
        return []


class _ScriptedPlanner:
    def __init__(self, plans) -> None:
        self.plans = list(plans)

    def make_plan(self, instruction, observation, *, failed=None):
        from minie.agent.workflow import Plan

        if not self.plans:
            return Plan(goal=instruction, steps=[])
        return self.plans.pop(0)

    def pick_target(self, instruction, step, candidates, observation):
        from minie.agent.workflow import _match_hint

        hint = str(step.args.get("label") or step.args.get("text") or "")
        found = _match_hint(candidates, hint)
        if found is None and candidates:
            found = candidates[0]
        return found.cid if found else None


def _workflow_ctx(cua):
    from minie.agent.router import make_context
    from minie.config import Config

    return make_context(Config(anthropic_api_key=""), cua, lambda: False, lambda _s: None)


def test_workflow_refuses_without_anthropic_key() -> None:
    from minie.agent.task import Task
    from minie.agent.workflow import WorkflowLoop, _MISSING_KEY

    cua = _FakeCua()
    ctx = _workflow_ctx(cua)
    loop = WorkflowLoop(ctx.config, ctx, ctx.cancelled)
    speech = loop.run(Task(instruction="open safari", goal="open safari"))
    assert speech == _MISSING_KEY
    assert cua.launched == []


def test_safari_search_presses_enter_and_verifies() -> None:
    from minie.agent.task import Task
    from minie.agent.workflow import Plan, Step, Verify, WorkflowLoop

    cua = _FakeCua()
    ctx = _workflow_ctx(cua)
    plan = Plan(
        goal="Open YouTube in Safari",
        steps=[
            Step("launch_app", {"name": "Safari"}, Verify(window_title="Safari")),
            Step(
                "type",
                {"text": "youtube.com", "submit": True, "label": "Smart Search Field"},
                Verify(url_or_title_contains="youtube"),
            ),
        ],
    )
    loop = WorkflowLoop(ctx.config, ctx, ctx.cancelled, client=_ScriptedPlanner([plan]))
    speech = loop.run(Task(instruction="open Safari and search for youtube.com", goal="youtube"))
    assert "enter" in cua.keys
    assert cua.typed == ["youtube.com"]
    assert "youtube" in speech.lower()
    assert cua.title == "YouTube"


def test_safari_type_without_enter_replans_until_verified() -> None:
    from minie.agent.task import Task
    from minie.agent.workflow import Plan, Step, Verify, WorkflowLoop

    cua = _FakeCua()
    ctx = _workflow_ctx(cua)
    first = Plan(
        goal="Open YouTube",
        steps=[
            Step("launch_app", {"name": "Safari"}, Verify(window_title="Safari")),
            Step("type", {"text": "youtube.com", "submit": False}, Verify(url_or_title_contains="youtube")),
        ],
    )
    repair = Plan(
        goal="Submit the search",
        steps=[Step("press_key", {"key": "enter"}, Verify(url_or_title_contains="youtube"))],
    )
    loop = WorkflowLoop(ctx.config, ctx, ctx.cancelled, client=_ScriptedPlanner([first, repair]))
    speech = loop.run(Task(instruction="open Safari and search for youtube.com", goal="youtube"))
    assert "enter" in cua.keys
    assert "youtube" in speech.lower()


def test_calculator_speech_matches_display() -> None:
    from minie.agent.task import Task
    from minie.agent.workflow import Plan, Step, Verify, WorkflowLoop

    cua = _FakeCua()
    ctx = _workflow_ctx(cua)
    plan = Plan(
        goal="Compute 6*7",
        steps=[
            Step("launch_app", {"name": "Calculator"}, Verify(window_title="Calculator")),
            Step("type", {"text": "6*7="}, Verify(display_number="42")),
        ],
    )
    loop = WorkflowLoop(ctx.config, ctx, ctx.cancelled, client=_ScriptedPlanner([plan]))
    speech = loop.run(
        Task(instruction="Calculator and Calculator 10,000 multiplies by 97", goal="six times seven")
    )
    assert speech == "42"
    assert cua.display == "42"
    assert cua.typed == ["6*7="]


def test_haiku_payload_parses_tool_use_and_arrays() -> None:
    from minie.agent.workflow import extract_json, parse_plan, payload_from_message

    tool_msg = {
        "stop_reason": "tool_use",
        "content": [
            {
                "type": "tool_use",
                "name": "submit_plan",
                "input": {
                    "goal": "Compute 6 times 7",
                    "steps": [
                        {
                            "tool": "launch_app",
                            "args": {"name": "Calculator", "text": "", "url": ""},
                            "verify": {"window_title": "Calculator", "url_or_title_contains": "", "ax_contains": "", "display_number": ""},
                        },
                        {
                            "tool": "type",
                            "args": {"text": "6*7=", "name": ""},
                            "verify": {"display_number": "42", "window_title": "", "url_or_title_contains": "", "ax_contains": ""},
                        },
                    ],
                },
            }
        ],
    }
    plan = parse_plan(payload_from_message(tool_msg))
    assert [s.tool for s in plan.steps] == ["launch_app", "type"]
    assert plan.steps[0].args == {"name": "Calculator"}
    assert plan.steps[1].args == {"text": "6*7="}
    assert plan.steps[1].verify.display_number == "42"

    array_msg = {"content": [{"type": "text", "text": '[{"tool": "open_url", "args": {"url": "https://youtube.com"}}]'}]}
    plan = parse_plan(payload_from_message(array_msg))
    assert plan.steps[0].tool == "open_url"
    assert plan.steps[0].args["url"] == "https://youtube.com"
    assert extract_json("not json") == {}


def test_verify_does_not_pass_on_window_title_alone() -> None:
    from minie.agent.workflow import Verify, check_verify
    from minie.computer.cua import WindowTarget

    target = WindowTarget(
        pid=1,
        window_id=1,
        name="Calculator",
        title="Calculator",
        tree="Calculator",
        elements=[{"element_index": 99, "role": "static text", "label": "0"}],
    )
    obs = {"title": "Calculator", "tree_head": "Calculator"}
    ok, _fact = check_verify(
        Verify(url_or_title_contains="Calculator", ax_contains="42", display_number="42"),
        target,
        obs,
    )
    assert ok is False
    target.elements = [{"element_index": 99, "role": "static text", "label": "42"}]
    ok, fact = check_verify(
        Verify(url_or_title_contains="Calculator", ax_contains="42", display_number="42"),
        target,
        obs,
    )
    assert ok is True
    assert fact == "42"


def test_apple_com_matches_apple_title() -> None:
    from minie.agent.task import Task
    from minie.agent.workflow import Plan, Step, Verify, WorkflowLoop, check_verify
    from minie.computer.cua import WindowTarget

    target = WindowTarget(pid=1, window_id=1, name="Google Chrome", title="Apple", tree="Apple")
    ok, fact = check_verify(
        Verify(url_or_title_contains="apple.com"),
        target,
        {"title": "Apple", "tree_head": "Apple", "elements": []},
    )
    assert ok is True
    assert "apple" in fact.lower()

    cua = _FakeCua()
    ctx = _workflow_ctx(cua)
    plan = Plan(
        goal="Open apple.com in Chrome and search iPhone",
        steps=[
            Step("launch_app", {"name": "Google Chrome"}, Verify(window_title="Chrome")),
            Step("open_url", {"url": "https://www.apple.com"}, Verify(url_or_title_contains="apple.com")),
            Step("type", {"text": "iPhone 16 Pro Max", "submit": True}, Verify(ax_contains="iPhone")),
        ],
    )
    loop = WorkflowLoop(ctx.config, ctx, ctx.cancelled, client=_ScriptedPlanner([plan]))
    speech = loop.run(Task(instruction="Apple.com on Chrome and search for iPhone 16 Pro Max", goal="apple"))
    assert cua.urls and "apple.com" in cua.urls[0]
    assert cua.typed == ["iPhone 16 Pro Max"]
    assert "enter" in cua.keys
    assert "stopped" not in speech.lower()

