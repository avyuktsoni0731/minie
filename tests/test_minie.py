"""Unit tests that do not need a microphone or Cua Driver."""

from minie.agent.intent import parse_calculator_math, parse_open, parse_partial
from minie.agent.safety import is_dangerous, is_stop, needs_confirm
from minie.audio.phrases import is_wake_phrase


def test_wake_phrase_variants() -> None:
    assert is_wake_phrase("Hey Minie")
    assert is_wake_phrase("okay hey mini open safari")
    assert is_wake_phrase("Hey Mini.")
    assert not is_wake_phrase("thank you")
    assert not is_wake_phrase("hey man")
    assert not is_wake_phrase("")


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


def test_six_times_seven() -> None:
    intent = parse_calculator_math("compute six times seven")
    assert intent is not None
    assert intent.keys == ["6", "*", "7", "="]


def test_partials_include_open_and_math() -> None:
    intents = parse_partial("open calculator and compute six times seven")
    kinds = {i.kind for i in intents}
    assert "open_app" in kinds
    assert "calculator" in kinds
