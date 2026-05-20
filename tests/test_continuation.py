"""Tests for runner-managed continuation loop (_run_continuation_loop)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


@pytest.fixture()
def _patch_runner_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub heavy imports so rune_runner can be imported in CPU-only CI."""
    import types

    # mlflow needs start_span to not raise
    mlflow_stub = types.ModuleType("mlflow")
    mlflow_stub.start_span = lambda **kw: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlflow", mlflow_stub)

    for mod_name in (
        "torch",
        "peft",
        "transformers",
        "safetensors",
        "llama_cpp",
        "model_training",
        "model_training.trajectory",
        "model_training.adapter_generator",
        "model_training.hypernetwork",
        "model_training.merging",
        "model_training.kill_switch",
    ):
        if mod_name not in sys.modules:
            monkeypatch.setitem(sys.modules, mod_name, types.ModuleType(mod_name))

    mt = sys.modules["model_training"]
    if not hasattr(mt, "trajectory"):
        mt.trajectory = types.ModuleType("model_training.trajectory")  # type: ignore[attr-defined]
    if not hasattr(mt.trajectory, "record_trajectory"):  # type: ignore[attr-defined]
        mt.trajectory.record_trajectory = lambda **kw: None  # type: ignore[attr-defined]


async def _noop(*_a: Any, **_kw: Any) -> None:
    return None


@pytest.mark.asyncio()
@pytest.mark.usefixtures("_patch_runner_imports")
async def test_continuation_accumulates_code() -> None:
    """Continuation loop concatenates output from multiple turns."""
    from scripts.rune_runner import _run_continuation_loop

    call_count = 0

    class FakeGraph:
        async def ainvoke(self, state: dict[str, Any]) -> dict[str, Any]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {
                    "generated_code": "# part 2",
                    "finish_reason": "length",
                    "tests_passed": False,
                }
            return {
                "generated_code": "# part 3",
                "finish_reason": "stop",
                "tests_passed": False,
            }

    code_state: dict[str, Any] = {
        "generated_code": "# part 1",
        "finish_reason": "length",
        "tests_passed": False,
    }

    final_state, accumulated = await _run_continuation_loop(
        code_state,
        "",
        graph=FakeGraph(),
        project_prompt="test task",
        session_id="test-session",
        iteration_base=100,
        subtask={"name": "test_subtask"},
        project_label="test",
        run_hypernetwork_fn=lambda **kw: None,
        adapter_dir=Path("/tmp/test_adapters"),
        base_model_id="test-model",
        checkpoint_path=None,
        device="cpu",
        adapter_scaling=0.1,
        adapter_max_length=2048,
        pool=None,
        render_trajectory_fn=lambda *a, **kw: "traj",
        load_adapter_fn=_noop,
        eager_unload_fn=_noop,
    )

    assert "# part 1" in accumulated
    assert "# part 2" in accumulated
    assert "# part 3" in accumulated
    assert final_state["finish_reason"] == "stop"


@pytest.mark.asyncio()
@pytest.mark.usefixtures("_patch_runner_imports")
async def test_continuation_skipped_when_not_truncated() -> None:
    """No continuation turns when finish_reason != 'length'."""
    from scripts.rune_runner import _run_continuation_loop

    code_state: dict[str, Any] = {
        "generated_code": "def solve(): pass",
        "finish_reason": "stop",
        "tests_passed": True,
    }

    final_state, accumulated = await _run_continuation_loop(
        code_state,
        "def solve(): pass",
        graph=None,  # Should never be called
        project_prompt="test task",
        session_id="test-session",
        iteration_base=100,
        subtask={"name": "test_subtask"},
        project_label="test",
        run_hypernetwork_fn=lambda **kw: None,
        adapter_dir=Path("/tmp/test_adapters"),
        base_model_id="test-model",
        checkpoint_path=None,
        device="cpu",
        adapter_scaling=0.1,
        adapter_max_length=2048,
        pool=None,
        render_trajectory_fn=lambda *a, **kw: "traj",
        load_adapter_fn=_noop,
        eager_unload_fn=_noop,
    )

    assert accumulated == "def solve(): pass"
    assert final_state is code_state


@pytest.mark.asyncio()
@pytest.mark.usefixtures("_patch_runner_imports")
async def test_continuation_respects_max_turns() -> None:
    """Continuation loop caps at _MAX_CONTINUATIONS turns."""
    from scripts.rune_runner import _MAX_CONTINUATIONS, _run_continuation_loop

    call_count = 0

    class AlwaysTruncatedGraph:
        async def ainvoke(self, state: dict[str, Any]) -> dict[str, Any]:
            nonlocal call_count
            call_count += 1
            return {
                "generated_code": f"# chunk {call_count}",
                "finish_reason": "length",
                "tests_passed": False,
            }

    code_state: dict[str, Any] = {
        "generated_code": "# initial",
        "finish_reason": "length",
        "tests_passed": False,
    }

    _, accumulated = await _run_continuation_loop(
        code_state,
        "",
        graph=AlwaysTruncatedGraph(),
        project_prompt="test task",
        session_id="test-session",
        iteration_base=100,
        subtask={"name": "test_subtask"},
        project_label="test",
        run_hypernetwork_fn=lambda **kw: None,
        adapter_dir=Path("/tmp/test_adapters"),
        base_model_id="test-model",
        checkpoint_path=None,
        device="cpu",
        adapter_scaling=0.1,
        adapter_max_length=2048,
        pool=None,
        render_trajectory_fn=lambda *a, **kw: "traj",
        load_adapter_fn=_noop,
        eager_unload_fn=_noop,
    )

    assert call_count == _MAX_CONTINUATIONS
    assert _MAX_CONTINUATIONS == 3
