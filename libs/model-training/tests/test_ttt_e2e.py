"""Tests for TTT-E2E (test-time training) baseline."""

from __future__ import annotations

from model_training.ttt_e2e import (
    TTTConfig,
    select_mlp_layers,
)


def test_ttt_config_defaults() -> None:
    cfg = TTTConfig()
    assert cfg.mlp_fraction == 0.25
    assert cfg.inner_lr > 0
    assert cfg.inner_steps > 0


def test_select_mlp_layers_25_percent() -> None:
    """Selects ~25% of MLP layers from a mock model."""
    layer_names = [f"model.layers.{i}.mlp.gate_proj" for i in range(32)]
    selected = select_mlp_layers(layer_names, fraction=0.25)
    assert len(selected) == 8


def test_select_mlp_layers_fraction_bounds() -> None:
    """Fraction clamped to [0, 1]."""
    layer_names = [f"layer.{i}.mlp" for i in range(10)]
    assert len(select_mlp_layers(layer_names, fraction=0.0)) == 0
    assert len(select_mlp_layers(layer_names, fraction=1.0)) == 10
