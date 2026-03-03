"""TDD wireframe tests for model_training.trajectory module."""

import pytest

from model_training.trajectory import format_for_sft, load_trajectory, record_trajectory


def test_record_trajectory_raises_not_implemented() -> None:
    """record_trajectory raises NotImplementedError with function name in message."""
    with pytest.raises(NotImplementedError, match="record_trajectory"):
        record_trajectory("session-001", [])


def test_load_trajectory_raises_not_implemented() -> None:
    """load_trajectory raises NotImplementedError with function name in message."""
    with pytest.raises(NotImplementedError, match="load_trajectory"):
        load_trajectory("traj-001")


def test_format_for_sft_raises_not_implemented() -> None:
    """format_for_sft raises NotImplementedError with function name in message."""
    with pytest.raises(NotImplementedError, match="format_for_sft"):
        format_for_sft({})
