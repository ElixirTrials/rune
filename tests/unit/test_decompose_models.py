"""Unit tests for Subtask and DecomposeResult Pydantic models.

Verifies the structural validation that guards against decompose phase
over-splitting (thinking-block leakage producing 16–30 subtasks) and
degenerate single-subtask results.
"""

import pytest
from pydantic import ValidationError
from shared.rune_models import DecomposeResult, Subtask

# ---------------------------------------------------------------------------
# Subtask tests
# ---------------------------------------------------------------------------


def test_subtask_valid_creation() -> None:
    """Subtask can be created with only the required name field."""
    st = Subtask(name="parse_input")
    assert st.name == "parse_input"
    assert st.description == ""
    assert st.depends_on == []


def test_subtask_all_fields() -> None:
    """Subtask accepts all fields when explicitly provided."""
    st = Subtask(
        name="write_output",
        description="Emit JSON to stdout",
        depends_on=["parse_input"],
    )
    assert st.description == "Emit JSON to stdout"
    assert st.depends_on == ["parse_input"]


def test_subtask_name_min_length_enforced() -> None:
    """Subtask rejects an empty string for name (min_length=1)."""
    with pytest.raises(ValidationError) as exc_info:
        Subtask(name="")
    errors = exc_info.value.errors()
    assert any(e["loc"] == ("name",) for e in errors)


def test_subtask_depends_on_defaults_to_empty_list() -> None:
    """depends_on defaults to an empty list when not supplied."""
    st = Subtask(name="standalone")
    assert st.depends_on == []


def test_subtask_depends_on_independent_default_per_instance() -> None:
    """Each Subtask gets its own depends_on list (no shared mutable)."""
    a = Subtask(name="a")
    b = Subtask(name="b")
    a.depends_on.append("extra")
    assert b.depends_on == []


# ---------------------------------------------------------------------------
# DecomposeResult tests
# ---------------------------------------------------------------------------


def _make_subtasks(n: int) -> list[Subtask]:
    return [Subtask(name=f"task_{i}") for i in range(n)]


def test_decompose_result_valid_two_subtasks() -> None:
    """DecomposeResult accepts exactly 2 subtasks (min boundary)."""
    result = DecomposeResult(subtasks=_make_subtasks(2))
    assert len(result.subtasks) == 2


def test_decompose_result_valid_eight_subtasks() -> None:
    """DecomposeResult accepts exactly 8 subtasks (max boundary)."""
    result = DecomposeResult(subtasks=_make_subtasks(8))
    assert len(result.subtasks) == 8


def test_decompose_result_valid_mid_range() -> None:
    """DecomposeResult accepts a typical 4-subtask decomposition."""
    result = DecomposeResult(subtasks=_make_subtasks(4))
    assert len(result.subtasks) == 4


def test_decompose_result_rejects_single_subtask() -> None:
    """DecomposeResult rejects a single subtask (min_length=2 not met)."""
    with pytest.raises(ValidationError) as exc_info:
        DecomposeResult(subtasks=_make_subtasks(1))
    errors = exc_info.value.errors()
    assert any(e["loc"] == ("subtasks",) for e in errors)


def test_decompose_result_rejects_empty_list() -> None:
    """DecomposeResult rejects an empty subtask list."""
    with pytest.raises(ValidationError):
        DecomposeResult(subtasks=[])


def test_decompose_result_rejects_nine_subtasks() -> None:
    """DecomposeResult rejects 9 subtasks (max_length=8 exceeded)."""
    with pytest.raises(ValidationError) as exc_info:
        DecomposeResult(subtasks=_make_subtasks(9))
    errors = exc_info.value.errors()
    assert any(e["loc"] == ("subtasks",) for e in errors)


def test_decompose_result_rejects_thinking_block_overflow() -> None:
    """DecomposeResult rejects 16+ subtasks (thinking-block leakage scenario)."""
    with pytest.raises(ValidationError):
        DecomposeResult(subtasks=_make_subtasks(16))


def test_decompose_result_subtasks_are_subtask_instances() -> None:
    """DecomposeResult.subtasks contains Subtask objects."""
    result = DecomposeResult(subtasks=_make_subtasks(3))
    assert all(isinstance(st, Subtask) for st in result.subtasks)


def test_decompose_result_round_trip_serialization() -> None:
    """DecomposeResult round-trips through model_dump and reconstruction."""
    original = DecomposeResult(
        subtasks=[
            Subtask(name="parse", description="Parse input", depends_on=[]),
            Subtask(name="emit", description="Emit output", depends_on=["parse"]),
        ]
    )
    data = original.model_dump()
    restored = DecomposeResult(**data)
    assert len(restored.subtasks) == 2
    assert restored.subtasks[1].depends_on == ["parse"]
