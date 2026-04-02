"""Tests for model_training.trainer module.

Tests are split into two categories:
- CPU tests: validate trajectory loading, parameter resolution, and wiring
  logic that doesn't require GPU libraries
- GPU tests: marked with @requires_gpu, skipped when torch/peft are not
  available. These test real training behavior.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from model_training.model_configs import ModelRegistry


def _gpu_available() -> bool:
    """Check if real GPU training libraries are importable."""
    try:
        import peft  # noqa: F401
        import torch  # noqa: F401
        import trl  # noqa: F401

        return True
    except ImportError:
        return False


requires_gpu = pytest.mark.skipif(
    not _gpu_available(),
    reason="GPU libraries (torch, peft, trl) not available",
)


def _make_trajectory(tmp_path: Path, session_id: str, outcome: str) -> Path:
    """Write a trajectory JSON file to tmp_path and return its path."""
    traj = {
        "session_id": session_id,
        "task_description": "Write a hello world function",
        "task_type": "code-gen",
        "adapter_ids": [],
        "outcome": outcome,
        "timestamp": "2026-03-05T00:00:00Z",
        "steps": [
            {
                "attempt": 1,
                "generated_code": "def hello(): return 'hello'",
                "tests_passed": True,
            }
        ],
    }
    traj_file = tmp_path / f"{session_id}.json"
    traj_file.write_text(json.dumps(traj))
    return traj_file


# ---------------------------------------------------------------------------
# CPU tests: no GPU libraries needed
# ---------------------------------------------------------------------------


def test_train_qlora_function_importable_without_gpu() -> None:
    """train_qlora is importable without GPU libs (all GPU imports are deferred)."""
    from model_training.trainer import train_qlora  # noqa: F401

    assert callable(train_qlora)


def test_train_qlora_rejects_missing_trajectory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """train_qlora raises FileNotFoundError for a non-existent session_id."""
    monkeypatch.setenv("RUNE_TRAJECTORY_DIR", str(tmp_path))

    from model_training.trajectory import load_trajectory

    with pytest.raises(FileNotFoundError):
        load_trajectory("nonexistent-session")


def test_train_qlora_rejects_unsuccessful_trajectory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """format_for_sft returns empty for a trajectory with outcome='exhausted'."""
    monkeypatch.setenv("RUNE_TRAJECTORY_DIR", str(tmp_path))
    _make_trajectory(tmp_path, "sess-exhausted", outcome="exhausted")

    from model_training.trajectory import format_for_sft, load_trajectory

    trajectory = load_trajectory("sess-exhausted")
    messages = format_for_sft(trajectory)
    assert not messages


# ---------------------------------------------------------------------------
# _resolve_training_params tests: pure Python, no GPU deps
# ---------------------------------------------------------------------------


def test_resolve_params_defaults_without_registry() -> None:
    """Without model_config_name, uses hardcoded defaults."""
    from model_training.trainer import _resolve_training_params

    params = _resolve_training_params(
        model_config_name=None,
        base_model_id=None,
        warm_start_adapter_id=None,
        rank=None,
        alpha=None,
        epochs=None,
        gradient_accumulation_steps=None,
        lr_scheduler_type=None,
    )
    assert params["rank"] == 64
    assert params["alpha"] == 128
    assert params["epochs"] == 3
    assert params["grad_accum"] == 16
    assert params["lr_sched"] == "constant"
    assert params["attn_impl"] is None
    assert params["warm_start"] is None


def test_resolve_params_with_registry() -> None:
    """model_config_name populates defaults from registry."""
    from model_training.trainer import _resolve_training_params

    params = _resolve_training_params(
        model_config_name="qwen3.5-9b",
        base_model_id=None,
        warm_start_adapter_id=None,
        rank=None,
        alpha=None,
        epochs=None,
        gradient_accumulation_steps=None,
        lr_scheduler_type=None,
    )
    mc = ModelRegistry.default().get("qwen3.5-9b")
    assert params["base_model_id"] == mc.model_id
    assert params["warm_start"] == mc.warm_start_adapter_id
    assert params["rank"] == mc.default_lora_rank
    assert params["alpha"] == mc.default_lora_alpha
    assert params["epochs"] == mc.epochs
    assert params["grad_accum"] == mc.gradient_accumulation_steps
    assert params["lr_sched"] == mc.lr_scheduler_type
    assert params["attn_impl"] == mc.attn_implementation


def test_resolve_params_explicit_overrides_registry() -> None:
    """Explicit args take precedence over registry defaults."""
    from model_training.trainer import _resolve_training_params

    params = _resolve_training_params(
        model_config_name="qwen3.5-9b",
        base_model_id="custom/model",
        warm_start_adapter_id="custom/adapter",
        rank=32,
        alpha=16,
        epochs=5,
        gradient_accumulation_steps=8,
        lr_scheduler_type="cosine",
    )
    assert params["base_model_id"] == "custom/model"
    assert params["warm_start"] == "custom/adapter"
    assert params["rank"] == 32
    assert params["alpha"] == 16
    assert params["epochs"] == 5
    assert params["grad_accum"] == 8
    assert params["lr_sched"] == "cosine"
    # attn_impl always comes from registry when model_config_name is set
    assert params["attn_impl"] == "eager"


def test_resolve_params_unknown_registry_raises() -> None:
    """Unknown model_config_name raises KeyError."""
    from model_training.trainer import _resolve_training_params

    with pytest.raises(KeyError, match="nonexistent"):
        _resolve_training_params(
            model_config_name="nonexistent",
            base_model_id=None,
            warm_start_adapter_id=None,
            rank=None,
            alpha=None,
            epochs=None,
            gradient_accumulation_steps=None,
            lr_scheduler_type=None,
        )


def test_resolve_params_qwen3_coder_next() -> None:
    """qwen3-coder-next has different defaults than qwen3.5-9b."""
    from model_training.trainer import _resolve_training_params

    params = _resolve_training_params(
        model_config_name="qwen3-coder-next",
        base_model_id=None,
        warm_start_adapter_id=None,
        rank=None,
        alpha=None,
        epochs=None,
        gradient_accumulation_steps=None,
        lr_scheduler_type=None,
    )
    mc = ModelRegistry.default().get("qwen3-coder-next")
    assert params["rank"] == mc.default_lora_rank  # 8, not 64
    assert params["warm_start"] is None  # no warm-start for this model
    assert params["attn_impl"] is None  # no attention override


# ---------------------------------------------------------------------------
# train_and_register wiring test: mocks train_qlora (GPU boundary) only
# ---------------------------------------------------------------------------


def test_train_and_register_creates_adapter_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """train_and_register creates the adapter dir and stores the record."""
    monkeypatch.setenv("RUNE_TRAJECTORY_DIR", str(tmp_path))
    monkeypatch.setenv("RUNE_ADAPTER_DIR", str(tmp_path / "adapters"))

    session_id = "sess-success"
    adapter_id = "adapter-001"
    _make_trajectory(tmp_path, session_id, outcome="success")

    # Build a fake safetensors file that train_qlora would write
    fake_adapter_dir = tmp_path / "adapters" / adapter_id
    fake_adapter_dir.mkdir(parents=True)
    fake_safetensors = fake_adapter_dir / "adapter_model.safetensors"
    fake_safetensors.write_bytes(b"fake_weights")

    with patch("model_training.trainer.train_qlora") as mock_train:
        mock_train.return_value = str(fake_adapter_dir)

        mock_registry = MagicMock()
        mock_registry_cls = MagicMock(return_value=mock_registry)
        monkeypatch.setattr(
            "adapter_registry.registry.AdapterRegistry", mock_registry_cls
        )

        from model_training.trainer import train_and_register

        result = train_and_register(
            session_id=session_id,
            adapter_id=adapter_id,
            database_url="sqlite:///:memory:",
        )

    assert result == adapter_id
    assert fake_adapter_dir.exists()
    mock_registry.store.assert_called_once()


# ---------------------------------------------------------------------------
# GPU integration tests: require real torch/peft/trl
# ---------------------------------------------------------------------------

# To add GPU integration tests, mark them with @requires_gpu:
#
# @requires_gpu
# def test_train_qlora_warm_start_loads_deltacoder(tmp_path, monkeypatch):
#     """DeltaCoder adapter loads via PeftModel.from_pretrained on real GPU."""
#     ...
