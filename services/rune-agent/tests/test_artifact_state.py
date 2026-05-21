"""Tests for artifact_state and adapter_strategy modules."""

from __future__ import annotations

import pytest
from rune_agent.adapter_strategy import (
    AdapterStrategy,
    ChunkComposition,
    SinglePass,
    resolve_adapter_strategy,
)
from rune_agent.artifact_state import (
    ArtifactState,
    PatchRecord,
    TrajectoryState,
    build_artifact_state,
    chunk_code_state,
)


def test_artifact_state_creation():
    patch = PatchRecord(
        turn=1, description="added helper", diff_summary="+def helper()",
    )
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
    original = TrajectoryState(
        turn=2, output="plan text", feedback="ok", diagnosis="good",
    )
    d = original.to_dict()
    restored = TrajectoryState.from_dict(d)
    assert restored.turn == 2
    assert restored.output == "plan text"


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


def test_build_artifact_state_first_turn():
    art = build_artifact_state(
        generated_code="import os\n\ndef main():\n    print('hello')\n",
        stdout="1 passed",
        stderr="",
        tests_passed=True,
        turn=0,
        previous_artifact=None,
    )
    assert "import os" in art.import_block
    assert "main" in art.interface_summary
    assert art.tests_passed is True
    assert len(art.patches) == 1
    assert art.patches[0].turn == 0


def test_build_artifact_state_with_previous():
    prev = ArtifactState(
        file_contents="def old(): pass",
        interface_summary="def old()",
        import_block="",
        patches=[PatchRecord(turn=0, description="initial", diff_summary="+def old()")],
        test_results="1 passed",
        stderr_summary="",
        tests_passed=True,
        todos=[],
    )
    art = build_artifact_state(
        generated_code="import os\n\ndef old(): pass\ndef new(): pass\n",
        stdout="2 passed",
        stderr="",
        tests_passed=True,
        turn=1,
        previous_artifact=prev,
    )
    assert len(art.patches) == 2
    assert art.patches[1].turn == 1
    assert "new" in art.patches[1].diff_summary


def test_chunk_code_state_small_artifact():
    art = ArtifactState(
        file_contents="import os\ndef main(): pass",
        interface_summary="def main()",
        import_block="import os",
        patches=[],
        test_results="",
        stderr_summary="",
        tests_passed=True,
        todos=[],
    )
    chunks = chunk_code_state(art, max_chunk_tokens=5000)
    assert len(chunks) >= 2
    types = {c.chunk_type for c in chunks}
    assert "imports" in types
    assert "interfaces" in types


def test_chunk_code_state_priority_ordering():
    art = ArtifactState(
        file_contents=(
            "import os\nimport sys\n\nclass Foo:\n    pass\n\ndef bar():\n    pass\n"
        ),
        interface_summary="class Foo\ndef bar()",
        import_block="import os\nimport sys",
        patches=[PatchRecord(turn=0, description="init", diff_summary="+Foo, +bar")],
        test_results="2 passed",
        stderr_summary="",
        tests_passed=True,
        todos=[],
    )
    chunks = chunk_code_state(art, max_chunk_tokens=100)
    assert chunks[0].chunk_type == "imports"
    assert chunks[0].priority == 1.0
    assert chunks[1].chunk_type == "interfaces"
    assert chunks[1].priority == 0.95
