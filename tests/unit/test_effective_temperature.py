from __future__ import annotations

from rune.engine.graph import _effective_temperature
from rune.engine.policy import _with_target


def test_zeroshot_floor_is_greedy() -> None:
    code_action = _with_target("code", "f")  # first code attempt for subtask "f"
    # escalate mode, no prior code for "f" => zero-shot floor => temperature 0
    assert _effective_temperature("escalate", code_action, {}, 0.8) == 0.0


def test_escalation_uses_configured_temperature() -> None:
    code_action = _with_target("code", "f")
    # "f" already has code => this is an escalation re-code => keep configured temp
    assert (
        _effective_temperature("escalate", code_action, {"f": "def f(): ..."}, 0.8)
        == 0.8
    )


def test_non_escalate_mode_unchanged() -> None:
    code_action = _with_target("code", "f")
    assert _effective_temperature("full", code_action, {}, 0.3) == 0.3
