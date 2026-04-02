"""Tests for model_training.trainer module."""

import json
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest


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


def test_train_qlora_function_importable_without_gpu() -> None:
    """train_qlora is importable without GPU libs (all GPU imports are deferred)."""
    from model_training.trainer import train_qlora  # noqa: F401

    assert callable(train_qlora)


def test_train_qlora_rejects_missing_trajectory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """train_qlora raises FileNotFoundError for a non-existent session_id."""
    monkeypatch.setenv("RUNE_TRAJECTORY_DIR", str(tmp_path))

    # Inject fake GPU modules so deferred imports succeed without real GPU libs
    _inject_fake_gpu_modules()
    try:
        from model_training.trainer import train_qlora

        with pytest.raises(FileNotFoundError):
            train_qlora(
                session_id="nonexistent-session",
                adapter_id="test-id",
                output_dir=str(tmp_path / "out"),
            )
    finally:
        _remove_fake_gpu_modules()


def test_train_qlora_rejects_unsuccessful_trajectory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """train_qlora raises ValueError for a trajectory with outcome='exhausted'."""
    monkeypatch.setenv("RUNE_TRAJECTORY_DIR", str(tmp_path))
    _make_trajectory(tmp_path, "sess-exhausted", outcome="exhausted")

    _inject_fake_gpu_modules()
    try:
        from model_training.trainer import train_qlora

        with pytest.raises(ValueError, match="not successful"):
            train_qlora(
                session_id="sess-exhausted",
                adapter_id="test-id",
                output_dir=str(tmp_path / "out"),
            )
    finally:
        _remove_fake_gpu_modules()


@pytest.mark.xfail(reason="Full GPU mocking chain is fragile in CPU CI", strict=False)
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

    _inject_fake_gpu_modules()
    try:
        with (
            patch("model_training.trainer.train_qlora") as mock_train,
            patch("model_training.trainer.AdapterRegistry") as mock_registry_cls,
        ):
            mock_train.return_value = str(fake_adapter_dir)
            mock_registry = MagicMock()
            mock_registry_cls.return_value = mock_registry

            from model_training.trainer import train_and_register

            result = train_and_register(
                session_id=session_id,
                adapter_id=adapter_id,
                database_url="sqlite:///:memory:",
            )

        assert result == adapter_id
        assert fake_adapter_dir.exists()
        mock_registry.store.assert_called_once()
    finally:
        _remove_fake_gpu_modules()


# ---------------------------------------------------------------------------
# Helpers: inject/remove fake GPU modules into sys.modules so deferred imports
# inside train_qlora resolve without requiring real GPU packages.
# ---------------------------------------------------------------------------

_FAKE_GPU_MODULES = [
    "datasets",
    "torch",
    "transformers",
    "transformers.AutoModelForCausalLM",
    "transformers.AutoTokenizer",
    "transformers.BitsAndBytesConfig",
    "trl",
    "trl.SFTConfig",
    "trl.SFTTrainer",
    "peft",
]


def _inject_fake_gpu_modules() -> None:
    """Inject minimal fake modules into sys.modules for CPU-only test runs."""
    fake_torch = ModuleType("torch")
    fake_torch.bfloat16 = MagicMock()  # type: ignore[attr-defined]

    fake_datasets = ModuleType("datasets")

    def _fake_from_list(records: list) -> MagicMock:  # type: ignore[type-arg]
        return MagicMock()

    fake_dataset_cls = MagicMock()
    fake_dataset_cls.from_list = staticmethod(_fake_from_list)
    fake_datasets.Dataset = fake_dataset_cls  # type: ignore[attr-defined]

    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoModelForCausalLM = MagicMock()  # type: ignore[attr-defined]
    fake_transformers.AutoTokenizer = MagicMock()  # type: ignore[attr-defined]
    fake_transformers.BitsAndBytesConfig = MagicMock()  # type: ignore[attr-defined]

    fake_trl = ModuleType("trl")
    fake_trl.SFTConfig = MagicMock()  # type: ignore[attr-defined]
    fake_trl.SFTTrainer = MagicMock()  # type: ignore[attr-defined]

    fake_peft = ModuleType("peft")
    fake_peft.LoraConfig = MagicMock()  # type: ignore[attr-defined]
    fake_peft.get_peft_model = MagicMock()  # type: ignore[attr-defined]
    fake_peft.PeftModel = MagicMock()  # type: ignore[attr-defined]

    fake_bnb = ModuleType("bitsandbytes")

    sys.modules.setdefault("torch", fake_torch)
    sys.modules.setdefault("datasets", fake_datasets)
    sys.modules.setdefault("transformers", fake_transformers)
    sys.modules.setdefault("trl", fake_trl)
    sys.modules.setdefault("peft", fake_peft)
    sys.modules.setdefault("bitsandbytes", fake_bnb)


def _remove_fake_gpu_modules() -> None:
    """Remove injected fake modules (only if we injected them)."""
    for mod in ("torch", "datasets", "transformers", "trl", "peft", "bitsandbytes"):
        if mod in sys.modules and isinstance(sys.modules[mod], ModuleType):
            # Only remove if it's actually a fake (not the real package)
            m = sys.modules[mod]
            if getattr(m, "__file__", "FAKE") is None:
                del sys.modules[mod]


# ---------------------------------------------------------------------------
# New tests: warm-start, registry config, loss masking, hyperparameters
# ---------------------------------------------------------------------------


def _force_inject_fake_gpu_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force-inject fake GPU modules via monkeypatch for test isolation.

    Unlike _inject_fake_gpu_modules (which uses setdefault and won't override
    real modules loaded by other test files), this uses monkeypatch.setitem
    for reliable test-level isolation even when real torch/peft are loaded.
    """
    fake_torch = ModuleType("torch")
    fake_torch.bfloat16 = MagicMock()  # type: ignore[attr-defined]

    fake_datasets = ModuleType("datasets")
    fake_dataset_cls = MagicMock()
    fake_dataset_cls.from_list = staticmethod(lambda records: MagicMock())
    fake_datasets.Dataset = fake_dataset_cls  # type: ignore[attr-defined]

    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoModelForCausalLM = MagicMock()  # type: ignore[attr-defined]
    fake_transformers.AutoTokenizer = MagicMock()  # type: ignore[attr-defined]
    fake_transformers.BitsAndBytesConfig = MagicMock()  # type: ignore[attr-defined]

    fake_trl = ModuleType("trl")
    fake_trl.SFTConfig = MagicMock()  # type: ignore[attr-defined]
    fake_trl.SFTTrainer = MagicMock()  # type: ignore[attr-defined]

    fake_peft = ModuleType("peft")
    fake_peft.LoraConfig = MagicMock()  # type: ignore[attr-defined]
    fake_peft.get_peft_model = MagicMock()  # type: ignore[attr-defined]
    fake_peft.PeftModel = MagicMock()  # type: ignore[attr-defined]

    fake_bnb = ModuleType("bitsandbytes")

    for name, mod in [
        ("torch", fake_torch),
        ("datasets", fake_datasets),
        ("transformers", fake_transformers),
        ("trl", fake_trl),
        ("peft", fake_peft),
        ("bitsandbytes", fake_bnb),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)


def test_train_qlora_with_warm_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Warm-start loads PeftModel.from_pretrained and skips fresh LoRA config."""
    monkeypatch.setenv("RUNE_TRAJECTORY_DIR", str(tmp_path))
    _make_trajectory(tmp_path, "sess-warm", outcome="success")
    _force_inject_fake_gpu_modules(monkeypatch)

    # Configure fake PeftModel to return a model with named_parameters
    import peft as fake_peft

    fake_wrapped = MagicMock()
    fake_wrapped.named_parameters.return_value = [
        ("lora_A.weight", MagicMock()),
        ("lora_B.weight", MagicMock()),
    ]
    fake_peft.PeftModel.from_pretrained.return_value = fake_wrapped

    # Configure fake SFTTrainer
    import trl as fake_trl

    fake_trainer_instance = MagicMock()
    fake_trl.SFTTrainer.return_value = fake_trainer_instance

    from model_training.trainer import train_qlora

    train_qlora(
        session_id="sess-warm",
        adapter_id="warm-test",
        output_dir=str(tmp_path / "out"),
        warm_start_adapter_id="test/warm-adapter",
    )

    # PeftModel.from_pretrained was called
    fake_peft.PeftModel.from_pretrained.assert_called_once()

    # SFTTrainer was called with peft_config=None (warm-start skips it)
    call_kwargs = fake_trl.SFTTrainer.call_args[1]
    assert call_kwargs["peft_config"] is None


def test_train_qlora_registry_config_sets_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """model_config_name populates defaults from registry."""
    monkeypatch.setenv("RUNE_TRAJECTORY_DIR", str(tmp_path))
    _make_trajectory(tmp_path, "sess-reg", outcome="success")
    _force_inject_fake_gpu_modules(monkeypatch)

    import transformers as fake_xf
    import trl as fake_trl

    fake_trainer_instance = MagicMock()
    fake_trl.SFTTrainer.return_value = fake_trainer_instance

    # Configure PeftModel for warm-start (qwen3.5-9b has warm_start)
    import peft as fake_peft

    fake_wrapped = MagicMock()
    fake_wrapped.named_parameters.return_value = []
    fake_peft.PeftModel.from_pretrained.return_value = fake_wrapped

    from model_training.trainer import train_qlora

    train_qlora(
        session_id="sess-reg",
        adapter_id="reg-test",
        output_dir=str(tmp_path / "out"),
        model_config_name="qwen3.5-9b",
    )

    # Model loaded with Qwen3.5-9B model ID and eager attention
    load_call = fake_xf.AutoModelForCausalLM.from_pretrained
    call_args = load_call.call_args
    assert call_args[0][0] == "Qwen/Qwen3.5-9B"
    assert call_args[1].get("attn_implementation") == "eager"

    # SFTConfig called with strategy-aligned defaults
    sft_config_call = fake_trl.SFTConfig.call_args[1]
    assert sft_config_call["gradient_accumulation_steps"] == 16
    assert sft_config_call["lr_scheduler_type"] == "constant"
    assert sft_config_call["assistant_only_loss"] is True


def test_train_qlora_assistant_only_loss_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """assistant_only_loss=True is always set in SFTConfig."""
    monkeypatch.setenv("RUNE_TRAJECTORY_DIR", str(tmp_path))
    _make_trajectory(tmp_path, "sess-loss", outcome="success")
    _force_inject_fake_gpu_modules(monkeypatch)

    import trl as fake_trl

    fake_trainer_instance = MagicMock()
    fake_trl.SFTTrainer.return_value = fake_trainer_instance

    from model_training.trainer import train_qlora

    train_qlora(
        session_id="sess-loss",
        adapter_id="loss-test",
        output_dir=str(tmp_path / "out"),
    )

    sft_config_call = fake_trl.SFTConfig.call_args[1]
    assert sft_config_call["assistant_only_loss"] is True


def test_train_qlora_fresh_init_backward_compat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without model_config_name, fresh LoRA init with explicit params."""
    monkeypatch.setenv("RUNE_TRAJECTORY_DIR", str(tmp_path))
    _make_trajectory(tmp_path, "sess-fresh", outcome="success")
    _force_inject_fake_gpu_modules(monkeypatch)

    import trl as fake_trl

    fake_trainer_instance = MagicMock()
    fake_trl.SFTTrainer.return_value = fake_trainer_instance

    from model_training.trainer import train_qlora

    train_qlora(
        session_id="sess-fresh",
        adapter_id="fresh-test",
        output_dir=str(tmp_path / "out"),
        rank=32,
        alpha=64,
    )

    # SFTTrainer was called with a peft_config (not None)
    call_kwargs = fake_trl.SFTTrainer.call_args[1]
    assert call_kwargs["peft_config"] is not None
