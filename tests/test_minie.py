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
    intent = parse_calculator_math("and compute 6 multiplied by 7")
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


def test_router_prefers_native_adapters() -> None:
    from unittest.mock import MagicMock

    from minie.agent.adapters.calculator import CalculatorAdapter
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
    assert adapter.name == "calculator"
    ctx = make_context(Config(dry_run=True), cua, lambda: False, lambda _s: None)
    result = CalculatorAdapter().execute(task, ctx)
    assert result.ok
    assert result.speech == "42"

    adapter, task = router.claim("open safari")
    assert adapter.name == "open_app"

    adapter, task = router.claim("click the red button in this window")
    assert adapter.name == "gui"

    adapter, task = router.claim("and compute 1000 multiplied by 7")
    assert adapter.name == "calculator"
    assert task.payload["keys"] == ["6", "*", "7", "="]

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

