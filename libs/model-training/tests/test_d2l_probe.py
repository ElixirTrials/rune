"""Tests for model_training.d2l_probe module.

CPU-only tests using sys.modules injection and tiny mock nn.Module objects.
Tests cover:
- probe_model discovers attention layers and skips DeltaNet layers
- probe_model captures correct feature (weight) dimensions
- Probe cache round-trips through JSON
- load_probe_cache returns None on miss (never raises)
- extract_activations_with_model returns correct shape from mock model
- extract_activations_with_model auto-detects layer indices from cache
- extract_activations_with_model raises RuntimeError without cache
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Fake model helpers for probe tests
# ---------------------------------------------------------------------------


class _FakeAttnLayer(nn.Module):
    """Mimics a Qwen3 full-attention layer — has q/k/v/o_proj children."""

    def __init__(self, hidden: int = 16) -> None:
        super().__init__()
        self.q_proj = nn.Linear(hidden, hidden * 2)
        self.k_proj = nn.Linear(hidden, hidden)
        self.v_proj = nn.Linear(hidden, hidden)
        self.o_proj = nn.Linear(hidden * 2, hidden)


class _FakeGDNLayer(nn.Module):
    """Mimics a Qwen3.5 GDN layer with GDN projection children."""

    def __init__(self, hidden: int = 16) -> None:
        super().__init__()
        self.in_proj_qkv = nn.Linear(hidden, hidden * 3)
        self.in_proj_z = nn.Linear(hidden, hidden)
        self.in_proj_b = nn.Linear(hidden, hidden)
        self.in_proj_a = nn.Linear(hidden, hidden)
        self.out_proj = nn.Linear(hidden, hidden)


class _FakeMLPLayer(nn.Module):
    """Mimics a transformer MLP block with gate/up/down projections."""

    def __init__(self, hidden: int = 16) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden, hidden * 4)
        self.up_proj = nn.Linear(hidden, hidden * 4)
        self.down_proj = nn.Linear(hidden * 4, hidden)


class _FakeDeltaNetLayer(nn.Module):
    """Mimics a DeltaNet linear-attention layer — no q/k/v/o_proj."""

    def __init__(self, hidden: int = 16) -> None:
        super().__init__()
        self.fc = nn.Linear(hidden, hidden)


class _FakeFullLayer(nn.Module):
    """Layer with attention + MLP children (standard transformer block)."""

    def __init__(self, hidden: int = 16) -> None:
        super().__init__()
        self.self_attn = _FakeAttnLayer(hidden)
        self.mlp = _FakeMLPLayer(hidden)


class _FakeModel(nn.Module):
    """Hybrid model with 3 DeltaNet layers + 1 Attention layer at index 3."""

    def __init__(self, hidden: int = 16) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                _FakeDeltaNetLayer(hidden),
                _FakeDeltaNetLayer(hidden),
                _FakeDeltaNetLayer(hidden),
                _FakeAttnLayer(hidden),
            ]
        )


class _FakeModelWithGDN(nn.Module):
    """Model with GDN layers for target module discovery testing."""

    def __init__(self, hidden: int = 16) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_FakeGDNLayer(hidden), _FakeGDNLayer(hidden)])


class _FakeModelFull(nn.Module):
    """Model with attention + MLP layers for full module discovery."""

    def __init__(self, hidden: int = 16) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_FakeFullLayer(hidden), _FakeFullLayer(hidden)])


class _FakeModelMultiAttn(nn.Module):
    """Model with attention layers at indices 1 and 3, DeltaNet at 0 and 2."""

    def __init__(self, hidden: int = 16) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                _FakeDeltaNetLayer(hidden),
                _FakeAttnLayer(hidden),
                _FakeDeltaNetLayer(hidden),
                _FakeAttnLayer(hidden),
            ]
        )


class _FakeModelWithForward(nn.Module):
    """Mock model that returns fake hidden_states on forward."""

    def __init__(self, hidden: int = 16, seq_len: int = 8) -> None:
        super().__init__()
        # Minimal parameter so .device works
        self._dummy = nn.Linear(1, 1)
        self._hidden = hidden
        self._seq_len = seq_len

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def forward(self, **kwargs: Any) -> Any:
        """Return an output-like object with hidden_states attribute."""
        hidden_states = tuple(
            torch.zeros(1, self._seq_len, self._hidden) for _ in range(6)
        )
        result = MagicMock()
        result.hidden_states = hidden_states
        return result


# ---------------------------------------------------------------------------
# Test 1: probe_model finds attention layers
# ---------------------------------------------------------------------------


def test_probe_model_finds_attention_layers() -> None:
    """probe_model on _FakeModel (3 DeltaNet + 1 Attn at idx 3) returns [3]."""
    from model_training.d2l_probe import probe_model

    model = _FakeModel()
    result = probe_model(model)
    assert result["attention_layer_indices"] == [3]


# ---------------------------------------------------------------------------
# Test 2: probe_model skips DeltaNet layers
# ---------------------------------------------------------------------------


def test_probe_model_skips_deltanet_layers() -> None:
    """probe_model on mixed model returns only indices with q/k/v/o_proj."""
    from model_training.d2l_probe import probe_model

    model = _FakeModelMultiAttn()
    result = probe_model(model)
    assert result["attention_layer_indices"] == [1, 3]


# ---------------------------------------------------------------------------
# Test 3: probe_model captures feature sizes
# ---------------------------------------------------------------------------


def test_probe_model_captures_feature_sizes() -> None:
    """probe_model returns feature_sizes with correct q_proj and v_proj shapes."""
    from model_training.d2l_probe import probe_model

    hidden = 16
    model = _FakeModel(hidden=hidden)
    result = probe_model(model)

    fs = result["feature_sizes"]
    # q_proj is nn.Linear(hidden, hidden*2) → out=hidden*2, in=hidden
    assert "q_proj" in fs
    assert fs["q_proj"]["out"] == hidden * 2
    assert fs["q_proj"]["in"] == hidden

    # v_proj is nn.Linear(hidden, hidden) → out=hidden, in=hidden
    assert "v_proj" in fs
    assert fs["v_proj"]["out"] == hidden
    assert fs["v_proj"]["in"] == hidden


# ---------------------------------------------------------------------------
# Test 4: save and load probe cache
# ---------------------------------------------------------------------------


def test_save_and_load_probe_cache(tmp_path: Any, monkeypatch: Any) -> None:
    """save_probe_cache writes JSON; load_probe_cache reads back identically."""
    import model_training.d2l_probe as probe_module
    from model_training.d2l_probe import load_probe_cache, save_probe_cache

    monkeypatch.setattr(probe_module, "PROBE_CACHE_DIR", tmp_path)

    model_name = "test-model-abc"
    probe_data: dict[str, Any] = {
        "attention_layer_indices": [3, 7],
        "feature_sizes": {
            "q_proj": {"in": 16, "out": 32},
            "v_proj": {"in": 16, "out": 16},
        },
    }

    saved_path = save_probe_cache(model_name, probe_data)
    assert saved_path.exists()

    loaded = load_probe_cache(model_name)
    assert loaded is not None
    assert loaded["model_name"] == model_name
    assert loaded["attention_layer_indices"] == probe_data["attention_layer_indices"]
    assert loaded["feature_sizes"] == probe_data["feature_sizes"]


# ---------------------------------------------------------------------------
# Test 5: load_probe_cache returns None on miss
# ---------------------------------------------------------------------------


def test_load_probe_cache_returns_none_on_miss(tmp_path: Any, monkeypatch: Any) -> None:
    """load_probe_cache for non-existent model returns None, never raises."""
    import model_training.d2l_probe as probe_module
    from model_training.d2l_probe import load_probe_cache

    monkeypatch.setattr(probe_module, "PROBE_CACHE_DIR", tmp_path)

    result = load_probe_cache("nonexistent-model-xyz-999")
    assert result is None


# ---------------------------------------------------------------------------
# Test 6: extract_activations_with_model returns correct shape
# ---------------------------------------------------------------------------


def test_extract_activations_with_model_returns_correct_shape() -> None:
    """extract_activations_with_model returns (1, N_layers, seq_len, hidden)."""
    from model_training.d2l_probe import extract_activations_with_model

    hidden = 16
    seq_len = 8
    layer_indices = [0, 2, 4]

    model = _FakeModelWithForward(hidden=hidden, seq_len=seq_len)

    # Minimal fake tokenizer that produces a fixed-length input
    class _FakeTokenizer:
        def __call__(
            self, text: str, return_tensors: str, truncation: bool, max_length: int
        ) -> dict[str, torch.Tensor]:
            return {
                "input_ids": torch.ones(1, seq_len, dtype=torch.long),
                "attention_mask": torch.ones(1, seq_len, dtype=torch.long),
            }

    tokenizer = _FakeTokenizer()
    features, attn_mask = extract_activations_with_model(
        text="hello world",
        model=model,
        tokenizer=tokenizer,
        layer_indices=layer_indices,
    )

    assert features.shape == (1, len(layer_indices), seq_len, hidden)
    assert attn_mask.shape == (1, seq_len)


# ---------------------------------------------------------------------------
# Test 7: auto-detect layer_indices from cache
# ---------------------------------------------------------------------------


def test_extract_activations_with_model_auto_detects_from_cache(
    tmp_path: Any, monkeypatch: Any
) -> None:
    """extract_activations_with_model uses cached layer indices when None given."""
    import model_training.d2l_probe as probe_module
    from model_training.d2l_probe import (
        extract_activations_with_model,
        save_probe_cache,
    )

    monkeypatch.setattr(probe_module, "PROBE_CACHE_DIR", tmp_path)

    model_name = "cached-model-for-extraction"
    probe_data: dict[str, Any] = {
        "attention_layer_indices": [0, 2],
        "feature_sizes": {"q_proj": {"in": 16, "out": 16}},
    }
    save_probe_cache(model_name, probe_data)

    hidden = 16
    seq_len = 5
    model = _FakeModelWithForward(hidden=hidden, seq_len=seq_len)

    class _FakeTokenizer:
        def __call__(
            self, text: str, return_tensors: str, truncation: bool, max_length: int
        ) -> dict[str, torch.Tensor]:
            return {
                "input_ids": torch.ones(1, seq_len, dtype=torch.long),
                "attention_mask": torch.ones(1, seq_len, dtype=torch.long),
            }

    features, _ = extract_activations_with_model(
        text="test",
        model=model,
        tokenizer=_FakeTokenizer(),
        layer_indices=None,
        model_name=model_name,
    )
    # Should have used [0, 2] → 2 layers
    assert features.shape[1] == 2


# ---------------------------------------------------------------------------
# Test 8: raises RuntimeError without cache
# ---------------------------------------------------------------------------


def test_extract_activations_with_model_raises_without_cache(
    tmp_path: Any, monkeypatch: Any
) -> None:
    """extract_activations_with_model raises RuntimeError without cache."""
    import model_training.d2l_probe as probe_module
    from model_training.d2l_probe import extract_activations_with_model

    monkeypatch.setattr(probe_module, "PROBE_CACHE_DIR", tmp_path)

    model = _FakeModelWithForward()

    class _FakeTokenizer:
        def __call__(self, *args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
            return {
                "input_ids": torch.ones(1, 4, dtype=torch.long),
                "attention_mask": torch.ones(1, 4, dtype=torch.long),
            }

    with pytest.raises(RuntimeError, match="layer_indices"):
        extract_activations_with_model(
            text="test",
            model=model,
            tokenizer=_FakeTokenizer(),
            layer_indices=None,
            model_name="uncached-model-xyz",
        )


# ---------------------------------------------------------------------------
# Test 9: discover_target_modules finds attention projections
# ---------------------------------------------------------------------------


def test_discover_target_modules_finds_attn() -> None:
    """discover_target_modules finds q/k/v/o_proj in attention layers."""
    from model_training.d2l_probe import discover_target_modules

    model = _FakeModel()
    modules = discover_target_modules(model)
    assert "q_proj" in modules
    assert "k_proj" in modules
    assert "v_proj" in modules
    assert "o_proj" in modules


# ---------------------------------------------------------------------------
# Test 10: discover_target_modules finds GDN projections
# ---------------------------------------------------------------------------


def test_discover_target_modules_finds_gdn() -> None:
    """discover_target_modules finds GDN projections (in_proj_qkv, etc.)."""
    from model_training.d2l_probe import discover_target_modules

    model = _FakeModelWithGDN()
    modules = discover_target_modules(model)
    assert "in_proj_qkv" in modules
    assert "in_proj_z" in modules
    assert "in_proj_b" in modules
    assert "in_proj_a" in modules
    assert "out_proj" in modules


# ---------------------------------------------------------------------------
# Test 11: discover_target_modules finds MLP projections
# ---------------------------------------------------------------------------


def test_discover_target_modules_finds_mlp() -> None:
    """discover_target_modules finds MLP projections (gate/up/down_proj)."""
    from model_training.d2l_probe import discover_target_modules

    model = _FakeModelFull()
    modules = discover_target_modules(model)
    assert "gate_proj" in modules
    assert "up_proj" in modules
    assert "down_proj" in modules


# ---------------------------------------------------------------------------
# Test 12: discover_target_modules returns sorted unique list
# ---------------------------------------------------------------------------


def test_discover_target_modules_sorted_unique() -> None:
    """discover_target_modules returns sorted, deduplicated module names."""
    from model_training.d2l_probe import discover_target_modules

    model = _FakeModelFull()
    modules = discover_target_modules(model)
    assert modules == sorted(set(modules))


# ---------------------------------------------------------------------------
# Test 13: probe_model includes target_modules key
# ---------------------------------------------------------------------------


def test_probe_model_includes_target_modules() -> None:
    """probe_model result includes target_modules from discover_target_modules."""
    from model_training.d2l_probe import probe_model

    model = _FakeModel()
    result = probe_model(model)
    assert "target_modules" in result
    assert "q_proj" in result["target_modules"]


# ---------------------------------------------------------------------------
# Test 14: discover on model with no known projections returns empty
# ---------------------------------------------------------------------------


def test_discover_target_modules_empty_for_unknown() -> None:
    """discover_target_modules returns [] for model with no known projections."""
    from model_training.d2l_probe import discover_target_modules

    model = _FakeDeltaNetLayer()
    modules = discover_target_modules(model)
    assert modules == []
