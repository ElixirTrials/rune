"""Tests for artifact_state and adapter_strategy modules."""

from __future__ import annotations

import pytest

from rune_agent.artifact_state import ArtifactState, PatchRecord, TrajectoryState
from rune_agent.adapter_strategy import (
    AdapterPlacement,
    AdapterStrategy,
    ChunkComposition,
    SinglePass,
    resolve_adapter_strategy,
)


def test_artifact_state_creation():
    patch = PatchRecord(turn=1, description="added helper", diff_summary="+def helper()")
    state = ArtifactState(
        file_contents="def main(): pass",
        interface_summary="def main()",
        import_block="import os",
        patches=[patch],
        test_results="1 passed",
        stderr_summary="",
        tests_passed=True,
        todos=[],
    )
    assert state.file_contents == "def main(): pass"
    assert len(state.patches) == 1
    assert state.patches[0].turn == 1
    assert state.tests_passed is True


def test_trajectory_state_creation():
    ts = TrajectoryState(
        turn=3,
        output="decomposed into 4 subtasks",
        feedback="",
        diagnosis="good decomposition",
    )
    assert ts.turn == 3
    assert ts.output == "decomposed into 4 subtasks"


def test_artifact_state_round_trip():
    patch = PatchRecord(turn=1, description="init", diff_summary="+def main()")
    original = ArtifactState(
        file_contents="import os\ndef main(): pass",
        interface_summary="def main()",
        import_block="import os",
        patches=[patch],
        test_results="1 passed",
        stderr_summary="",
        tests_passed=True,
        todos=["implement body"],
    )
    d = original.to_dict()
    restored = ArtifactState.from_dict(d)
    assert restored.file_contents == original.file_contents
    assert restored.patches[0].turn == 1
    assert restored.todos == ["implement body"]


def test_trajectory_state_round_trip():
    original = TrajectoryState(turn=2, output="plan text", feedback="ok", diagnosis="good")
    d = original.to_dict()
    restored = TrajectoryState.from_dict(d)
    assert restored.turn == 2
    assert restored.output == "plan text"


def test_adapter_placement_defaults():
    p = AdapterPlacement()
    assert p.target_modules is None
    assert p.layer_indices is None
    assert p.layer_selection == "all"


def test_single_pass_strategy():
    s = SinglePass(scaling=0.16)
    assert isinstance(s, AdapterStrategy)
    assert s.scaling == 0.16
    assert s.truncate is False


def test_chunk_composition_strategy():
    c = ChunkComposition(scaling=0.192, merge_method="ties")
    assert isinstance(c, AdapterStrategy)
    assert c.scaling == 0.192
    assert c.merge_method == "ties"


def test_resolve_strategy_small_artifact():
    s = resolve_adapter_strategy("code", 500, 1024, 0.16)
    assert isinstance(s, SinglePass)
    assert s.scaling == 0.16
    assert s.truncate is False


def test_resolve_strategy_large_code_no_chunk():
    s = resolve_adapter_strategy("code", 2000, 1024, 0.16)
    assert isinstance(s, SinglePass)
    assert s.truncate is True


def test_resolve_strategy_large_code_with_chunk():
    s = resolve_adapter_strategy(
        "code", 2000, 1024, 0.16,
        enable_chunk_composition=True,
    )
    assert isinstance(s, ChunkComposition)
    assert s.scaling == pytest.approx(0.192)
    assert s.merge_method == "ties"


def test_resolve_strategy_large_text_phase():
    s = resolve_adapter_strategy(
        "plan", 2000, 1024, 0.16,
        enable_chunk_composition=True,
    )
    assert isinstance(s, SinglePass)
    assert s.truncate is True


def test_resolve_strategy_custom_boost():
    s = resolve_adapter_strategy(
        "code", 2000, 1024, 0.16,
        enable_chunk_composition=True,
        code_scaling_boost=1.5,
    )
    assert isinstance(s, ChunkComposition)
    assert s.scaling == pytest.approx(0.24)
