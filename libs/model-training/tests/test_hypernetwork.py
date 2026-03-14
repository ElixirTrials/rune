"""Tests for model_training.hypernetwork module.

CPU-only tests using sys.modules injection pattern established in test_trainer.py.
"""

import json
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Test 1: CPU-only importability
# ---------------------------------------------------------------------------


def test_hypernetwork_importable_without_gpu() -> None:
    """hypernetwork module is importable without GPU libs installed."""
    # Skip sys.modules cleanup if torch is already fully loaded — clearing
    # model_training.hypernetwork after real torch import triggers a reimport
    # chain that hits torch's _TritonLibrary double-registration crash in
    # pytest-xdist forked workers.
    torch_mod = sys.modules.get("torch")
    torch_fully_loaded = torch_mod is not None and hasattr(torch_mod, "Tensor")
    if not torch_fully_loaded:
        for key in list(sys.modules.keys()):
            if "model_training.hypernetwork" in key:
                del sys.modules[key]

    # Should not raise even without torch/safetensors
    from model_training.hypernetwork import (  # noqa: F401
        DocToLoraHypernetwork,
        save_hypernetwork_adapter,
    )

    assert callable(DocToLoraHypernetwork)
    assert callable(save_hypernetwork_adapter)


# ---------------------------------------------------------------------------
# Helpers: inject/remove fake GPU modules for hypernetwork tests
# ---------------------------------------------------------------------------

_FAKE_HN_MODULES = [
    "torch",
    "torch.nn",
    "safetensors",
    "safetensors.torch",
]


def _inject_fake_hn_modules() -> None:
    """Inject torch and safetensors fakes for hypernetwork CPU tests."""
    import torch  # noqa: PLC0415  # imported here intentionally — may be real

    # If real torch is available, no need to inject fakes
    if hasattr(torch, "Tensor") and not isinstance(getattr(torch, "Tensor"), MagicMock):
        return  # real torch is available — tests will use it directly

    fake_torch = ModuleType("torch")
    fake_torch.Tensor = MagicMock()  # type: ignore[attr-defined]
    fake_torch.randn = MagicMock()  # type: ignore[attr-defined]

    fake_nn = ModuleType("torch.nn")
    fake_nn.Module = MagicMock()  # type: ignore[attr-defined]
    fake_nn.Embedding = MagicMock()  # type: ignore[attr-defined]
    fake_nn.MultiheadAttention = MagicMock()  # type: ignore[attr-defined]
    fake_nn.TransformerEncoder = MagicMock()  # type: ignore[attr-defined]
    fake_nn.TransformerEncoderLayer = MagicMock()  # type: ignore[attr-defined]
    fake_nn.Linear = MagicMock()  # type: ignore[attr-defined]
    fake_nn.Parameter = MagicMock()  # type: ignore[attr-defined]
    fake_torch.nn = fake_nn  # type: ignore[attr-defined]

    fake_safetensors = ModuleType("safetensors")
    fake_safetensors_torch = ModuleType("safetensors.torch")
    fake_safetensors_torch.save_file = MagicMock()  # type: ignore[attr-defined]
    fake_safetensors.torch = fake_safetensors_torch  # type: ignore[attr-defined]

    sys.modules.setdefault("torch", fake_torch)
    sys.modules.setdefault("torch.nn", fake_nn)
    sys.modules.setdefault("safetensors", fake_safetensors)
    sys.modules.setdefault("safetensors.torch", fake_safetensors_torch)


# ---------------------------------------------------------------------------
# Fixtures: skip tests when torch is not really available
# ---------------------------------------------------------------------------

try:
    import torch  # noqa: PLC0415
    import torch.nn  # noqa: PLC0415

    _TORCH_AVAILABLE = not isinstance(torch, MagicMock) and hasattr(torch.nn, "Module")
except ImportError:
    _TORCH_AVAILABLE = False

requires_torch = pytest.mark.skipif(
    not _TORCH_AVAILABLE, reason="torch not installed in this environment"
)


# ---------------------------------------------------------------------------
# Test 2: DocToLoraHypernetwork latents parameter shape
# ---------------------------------------------------------------------------


@requires_torch
def test_hypernetwork_latents_shape() -> None:
    """DocToLoraHypernetwork.__init__ creates latents param with shape (32, 256)."""
    import torch  # noqa: PLC0415
    from model_training.hypernetwork import DocToLoraHypernetwork  # noqa: PLC0415

    # Use small hidden_dim and num_layers to keep construction fast on CPU
    model = DocToLoraHypernetwork(input_dim=1000, num_layers=1, hidden_dim=32)
    assert hasattr(model, "latents"), "model must have latents attribute"
    assert isinstance(model.latents, torch.nn.Parameter), "latents must be nn.Parameter"
    # Default num_latents=32, latent_dim=256 — shape must always be (32, 256)
    assert model.latents.shape == (32, 256), (
        f"Expected (32, 256) but got {model.latents.shape}"
    )


# ---------------------------------------------------------------------------
# Test 3: forward() returns dict with correct PEFT state_dict keys
# ---------------------------------------------------------------------------


@requires_torch
def test_hypernetwork_forward_peft_keys() -> None:
    """forward() returns dict with PEFT-compatible state_dict key pattern."""
    import torch  # noqa: PLC0415
    from model_training.hypernetwork import DocToLoraHypernetwork  # noqa: PLC0415

    # Use small hidden_dim and few layers to keep the test fast on CPU
    model = DocToLoraHypernetwork(
        input_dim=1000, num_layers=2, hidden_dim=32, num_latents=4, latent_dim=32
    )
    token_ids = torch.zeros(1, 8, dtype=torch.long)
    result = model(token_ids)

    assert isinstance(result, dict), "forward() must return a dict"

    # Check PEFT key naming convention
    for i in range(2):  # num_layers=2
        for module in ("q_proj", "v_proj"):
            for ab in ("A", "B"):
                prefix = "base_model.model.model.layers"
                key = f"{prefix}.{i}.self_attn.{module}.lora_{ab}.weight"
                assert key in result, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# Test 4: forward() output lora_A/B tensor shapes
# ---------------------------------------------------------------------------


@requires_torch
def test_hypernetwork_forward_lora_shapes() -> None:
    """forward() lora_A is (rank, hidden) and lora_B is (hidden, rank)."""
    import torch  # noqa: PLC0415
    from model_training.hypernetwork import DocToLoraHypernetwork  # noqa: PLC0415

    rank = 8
    hidden_dim = 64  # small for fast CPU test
    model = DocToLoraHypernetwork(
        input_dim=32000, num_layers=1, rank=rank, hidden_dim=hidden_dim
    )
    token_ids = torch.zeros(1, 8, dtype=torch.long)
    result = model(token_ids)

    for module in ("q_proj", "v_proj"):
        key_a = f"base_model.model.model.layers.0.self_attn.{module}.lora_A.weight"
        key_b = f"base_model.model.model.layers.0.self_attn.{module}.lora_B.weight"
        assert result[key_a].shape == (rank, hidden_dim), (
            f"lora_A mismatch: got {result[key_a].shape}"
        )
        assert result[key_b].shape == (hidden_dim, rank), (
            f"lora_B mismatch: got {result[key_b].shape}"
        )


# ---------------------------------------------------------------------------
# Test 5: save_hypernetwork_adapter() writes expected files
# ---------------------------------------------------------------------------


def _ensure_safetensors_module() -> None:
    """Ensure safetensors.torch is in sys.modules so patch() can target it."""
    if "safetensors" not in sys.modules:
        fake_st = ModuleType("safetensors")
        fake_st_torch = ModuleType("safetensors.torch")
        fake_st_torch.save_file = MagicMock()  # type: ignore[attr-defined]
        fake_st.torch = fake_st_torch  # type: ignore[attr-defined]
        sys.modules["safetensors"] = fake_st
        sys.modules["safetensors.torch"] = fake_st_torch


def test_save_hypernetwork_adapter_writes_files() -> None:
    """save_hypernetwork_adapter() writes safetensors and config."""
    _ensure_safetensors_module()
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Mock weights dict
        lora_key = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
        mock_weights: dict[str, MagicMock] = {lora_key: MagicMock()}

        with patch("safetensors.torch.save_file") as mock_save:
            from model_training.hypernetwork import (  # noqa: PLC0415
                save_hypernetwork_adapter,
            )

            save_hypernetwork_adapter(
                weights=mock_weights,
                output_dir=str(output_dir),
                base_model_id="Qwen/Qwen2.5-Coder-7B",
            )

        # safetensors.save_file was called
        mock_save.assert_called_once()
        args = mock_save.call_args
        assert str(output_dir / "adapter_model.safetensors") in str(args)

        # adapter_config.json was written
        config_path = output_dir / "adapter_config.json"
        assert config_path.exists(), "adapter_config.json must be written"


# ---------------------------------------------------------------------------
# Test 6: adapter_config.json contains correct fields
# ---------------------------------------------------------------------------


def test_save_hypernetwork_adapter_config_fields() -> None:
    """adapter_config.json has correct PEFT fields."""
    _ensure_safetensors_module()
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        mock_weights: dict[str, MagicMock] = {}

        with patch("safetensors.torch.save_file"):
            from model_training.hypernetwork import (  # noqa: PLC0415
                save_hypernetwork_adapter,
            )

            save_hypernetwork_adapter(
                weights=mock_weights,
                output_dir=str(output_dir),
                base_model_id="Qwen/Qwen2.5-Coder-7B",
                rank=8,
                target_modules=["q_proj", "v_proj"],
            )

        config_path = output_dir / "adapter_config.json"
        config = json.loads(config_path.read_text())

        assert config["peft_type"] == "LORA"
        assert config["r"] == 8
        assert config["target_modules"] == ["q_proj", "v_proj"]
        assert config["inference_mode"] is True
        assert config.get("modules_to_save") is None, (
            "modules_to_save must be null to avoid vLLM rejection"
        )
        assert config["task_type"] == "CAUSAL_LM"
        assert config["base_model_name_or_path"] == "Qwen/Qwen2.5-Coder-7B"


# ---------------------------------------------------------------------------
# Test 7: forward pass completes without loading 7B model
# ---------------------------------------------------------------------------


@requires_torch
def test_hypernetwork_forward_no_base_model() -> None:
    """forward() uses hypernetwork's own embedding — no 7B model loading required."""
    import time  # noqa: PLC0415

    import torch  # noqa: PLC0415
    from model_training.hypernetwork import DocToLoraHypernetwork  # noqa: PLC0415

    # Tiny config for fast CPU test
    model = DocToLoraHypernetwork(
        input_dim=1000,
        num_latents=4,
        latent_dim=32,
        depth=1,
        heads=2,
        rank=4,
        hidden_dim=16,
        num_layers=1,
    )
    token_ids = torch.zeros(1, 4, dtype=torch.long)

    start = time.time()
    result = model(token_ids)
    elapsed = time.time() - start

    assert isinstance(result, dict)
    assert elapsed < 30.0, f"forward() took too long: {elapsed:.1f}s (expected <30s)"
