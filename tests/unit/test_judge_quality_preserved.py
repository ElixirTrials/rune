"""Regression for issue #52 P0-3: a model-judge flip must drive repair ROUTING
without destroying the retained QUALITY of a verified public-passing candidate.

A candidate that passed the sandbox (exit 0) but is later judge-flipped to a
synthetic failing Feedback (exit 1, stderr without AssertionError) must keep
quality 3 in ``best_quality`` so a subsequent bare-running repair can't clobber
it in ``best_code``; routing (``code_passed``) still reflects the flipped result.
"""

from __future__ import annotations

from rune.engine.parse import _parse_code_action, parse_output
from rune.engine.state import Action, Feedback

CODE = "def f(x):\n    return x + 1\n"


def _state() -> dict:
    return {
        "code_results": {},
        "code_passed": {},
        "code_solved": {},
        "best_code": {},
        "best_quality": {},
        "retries": {},
        "feedback": {},
        "diagnosis": {},
        "subtasks": [],
        "public_checks": "",
    }


def test_judge_flip_preserves_quality_via_parse_code_action() -> None:
    routing_fb = Feedback(
        stdout="",
        stderr="Correctness judge: wrong on input (3,). off by one",
        exit_code=1,
    )
    quality_fb = Feedback(stdout="", stderr="", exit_code=0)
    out = _parse_code_action(
        "f",
        "",
        routing_fb,
        _state(),
        retries_delta=0,
        code=CODE,
        quality_feedback=quality_fb,
    )
    assert out["best_quality"]["f"] == 3
    assert out["code_passed"]["f"] is False


def test_judge_flip_preserves_quality_via_parse_output() -> None:
    action = Action("code", "code", "prompt_code", "", None, True, "f")
    routing_fb = Feedback(
        stdout="",
        stderr="Correctness judge: wrong on input (3,). off by one",
        exit_code=1,
    )
    quality_fb = Feedback(stdout="", stderr="", exit_code=0)
    out = parse_output(
        action,
        "",
        routing_fb,
        _state(),
        code=CODE,
        quality_feedback=quality_fb,
    )
    assert out["best_quality"]["f"] == 3
    assert out["code_passed"]["f"] is False


def test_no_quality_feedback_is_identical_to_today() -> None:
    routing_fb = Feedback(stdout="", stderr="", exit_code=0)
    out = _parse_code_action("f", "", routing_fb, _state(), retries_delta=0, code=CODE)
    assert out["best_quality"]["f"] == 3
    assert out["code_passed"]["f"] is True
