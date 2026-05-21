"""Tests for the reasoning loop LangGraph subgraph."""

from __future__ import annotations

from typing import Any

from rune_agent.reasoning_loop import (
    ReasoningLoopState,
    recover_node,
    route_after_reason,
    select_best_output,
    should_continue,
)


def _make_state(**overrides: Any) -> ReasoningLoopState:
    defaults: dict[str, Any] = {
        "task_description": "test task",
        "phase": "code",
        "adapter_ids": [],
        "session_id": "test-session",
        "generated_code": "",
        "stdout": "",
        "stderr": "",
        "exit_code": 0,
        "tests_passed": False,
        "test_count": 0,
        "tests_ran": False,
        "finish_reason": "length",
        "outcome": None,
        "prompt_context": None,
        "artifact": None,
        "trajectory_state": None,
        "turn_count": 0,
        "max_turns": 20,
        "sliding_window": "",
        "sliding_window_size": 1024,
        "base_sliding_window_size": 1024,
        "current_adapter_path": None,
        "current_adapter_weights": None,
        "prior_adapter_weights": None,
        "first_adapter_norm": None,
        "scaling_factor": 0.16,
        "enable_chunk_composition": False,
        "chunk_threshold": 1024,
        "phase_executes": True,
        "code_scaling_boost": 1.2,
        "default_merge_method": "ties",
        "turn_history": [],
        "adapter_cosine_sim": 0.0,
        "adapter_norm_ratio": 1.0,
        "output_repetition": 0.0,
        "consecutive_high_similarity": 0,
        "recovery_attempted": False,
    }
    defaults.update(overrides)
    return defaults  # type: ignore[return-value]


def test_reasoning_loop_state_structure():
    annotations = ReasoningLoopState.__annotations__
    required_keys = {
        "task_description", "phase", "adapter_ids", "session_id",
        "generated_code", "stdout", "stderr", "exit_code",
        "tests_passed", "test_count", "tests_ran", "finish_reason",
        "outcome", "prompt_context",
        "artifact", "trajectory_state",
        "turn_count", "max_turns", "sliding_window", "sliding_window_size",
        "base_sliding_window_size", "current_adapter_path",
        "current_adapter_weights", "prior_adapter_weights", "first_adapter_norm",
        "scaling_factor", "enable_chunk_composition", "chunk_threshold",
        "phase_executes", "code_scaling_boost", "default_merge_method",
        "turn_history",
        "adapter_cosine_sim", "adapter_norm_ratio", "output_repetition",
        "consecutive_high_similarity", "recovery_attempted",
    }
    for key in required_keys:
        assert key in annotations, f"Missing key: {key}"


def test_should_continue_stop():
    state = _make_state(finish_reason="stop")
    assert should_continue(state) == "__end__"


def test_should_continue_max_turns():
    state = _make_state(turn_count=20, max_turns=20)
    assert should_continue(state) == "__end__"


def test_should_continue_healthy():
    state = _make_state(
        turn_count=3, max_turns=20,
        adapter_cosine_sim=0.5, adapter_norm_ratio=1.0,
        output_repetition=0.2, finish_reason="length",
    )
    assert should_continue(state) == "reason"


def test_should_continue_collapse_triggers_recovery():
    state = _make_state(
        turn_count=3, max_turns=20,
        adapter_cosine_sim=0.97, adapter_norm_ratio=1.0,
        output_repetition=0.85, finish_reason="length",
        recovery_attempted=False,
    )
    assert should_continue(state) == "recover"


def test_should_continue_collapse_after_recovery_halts():
    state = _make_state(
        turn_count=3, max_turns=20,
        adapter_cosine_sim=0.97, adapter_norm_ratio=1.0,
        output_repetition=0.85, finish_reason="length",
        recovery_attempted=True,
    )
    assert should_continue(state) == "__end__"


def test_should_continue_norm_collapse_immediate_halt():
    state = _make_state(adapter_norm_ratio=0.05, finish_reason="length")
    assert should_continue(state) == "__end__"


def test_route_after_reason_code_phase():
    state = _make_state(phase_executes=True)
    assert route_after_reason(state) == "execute_reason"


def test_route_after_reason_text_phase():
    state = _make_state(phase_executes=False)
    assert route_after_reason(state) == "reflect"


async def test_recover_node_doubles_window():
    state = _make_state(sliding_window_size=1024)
    result = await recover_node(state)
    assert result["sliding_window_size"] == 2048
    assert result["recovery_attempted"] is True


def test_reasoning_loop_graph_compiles():
    from rune_agent.reasoning_loop import create_reasoning_loop_graph

    graph = create_reasoning_loop_graph()
    assert graph is not None


def test_best_passing_turn_tracking():
    history = [
        {"turn": 0, "tests_passed": False, "generated_code_len": 100},
        {"turn": 1, "tests_passed": True, "generated_code_len": 200},
        {"turn": 2, "tests_passed": False, "generated_code_len": 300},
    ]
    best_turn = select_best_output(history)
    assert best_turn == 1


def test_best_passing_turn_none():
    history = [
        {"turn": 0, "tests_passed": False},
        {"turn": 1, "tests_passed": False},
    ]
    assert select_best_output(history) is None


# Task 12: integration tests
def test_should_continue_stop_on_natural_end():
    state = _make_state(finish_reason="stop", turn_count=2)
    assert should_continue(state) == "__end__"


def test_should_continue_stop_on_max_turns():
    state = _make_state(turn_count=3, max_turns=3)
    assert should_continue(state) == "__end__"


def test_should_continue_continues_when_healthy():
    state = _make_state(
        turn_count=1, max_turns=10,
        adapter_cosine_sim=0.3, adapter_norm_ratio=1.0,
        output_repetition=0.1,
    )
    assert should_continue(state) == "reason"
