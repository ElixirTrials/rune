"""Escalation floor: first code attempt is zero-shot base, then adapter on repair.

Issue #52: the engine (model+adapter) must never be worse than the model alone.
In ``escalate`` mode the first code attempt for a subtask runs at adapter scaling
0 (base, spec in prompt); repairs/re-codes engage the adapter. keep-best then
ships the strongest candidate, so the floor is base.
"""

from __future__ import annotations

from dataclasses import replace

from rune.engine.graph import _effective_scaling
from rune.engine.policy import ACTIONS


def _code(target: str):
    return replace(ACTIONS["code"], target_subtask=target)


def _repair(target: str):
    return replace(ACTIONS["repair"], target_subtask=target)


class TestEffectiveScaling:
    def test_first_code_attempt_is_base_in_escalate_mode(self) -> None:
        # no prior code for "f" -> zero-shot base (scaling 0)
        assert _effective_scaling("escalate", _code("f"), {}, 0.627) == 0.0

    def test_re_code_after_prior_attempt_uses_adapter(self) -> None:
        # "f" already has code -> escalated, adapter on
        assert _effective_scaling("escalate", _code("f"), {"f": "def f(): ..."}, 0.627) == 0.627

    def test_repair_always_uses_adapter(self) -> None:
        assert _effective_scaling("escalate", _repair("f"), {}, 0.627) == 0.627

    def test_other_modes_unaffected(self) -> None:
        # full / episodic keep their configured scaling on the first attempt
        assert _effective_scaling("full", _code("f"), {}, 0.627) == 0.627
        assert _effective_scaling("episodic", _code("f"), {}, 0.627) == 0.627
