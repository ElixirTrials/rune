"""Tests for across-corpus z-score statistics."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _per_record(
    a: "torch.Tensor", b: "torch.Tensor"
) -> dict[str, dict[str, "torch.Tensor"]]:
    return {"q_proj": {"A": a, "B": b}}


def test_mean_and_std_shapes_match_single_record_shape() -> None:
    from model_training.reconstruction.stats import compute_zscore_stats

    a = torch.randn(4, 8, 16)  # (n_layers, r, in_features)
    b = torch.randn(4, 32, 8)
    stats = compute_zscore_stats([_per_record(a, b)])
    assert stats["q_proj"]["avg_A"].shape == (4, 8, 16)
    assert stats["q_proj"]["std_A"].shape == (4, 8, 16)
    assert stats["q_proj"]["avg_B"].shape == (4, 32, 8)
    assert stats["q_proj"]["std_B"].shape == (4, 32, 8)


def test_mean_matches_elementwise_average_across_records() -> None:
    from model_training.reconstruction.stats import compute_zscore_stats

    a1 = torch.ones(2, 2, 4)
    a2 = torch.full((2, 2, 4), 3.0)
    b1 = torch.ones(2, 4, 2)
    b2 = torch.full((2, 4, 2), 3.0)
    stats = compute_zscore_stats([_per_record(a1, b1), _per_record(a2, b2)])
    assert torch.allclose(stats["q_proj"]["avg_A"], torch.full((2, 2, 4), 2.0))
    assert torch.allclose(stats["q_proj"]["avg_B"], torch.full((2, 4, 2), 2.0))


def test_std_floored_to_minimum() -> None:
    from model_training.reconstruction.stats import (
        STD_FLOOR,
        compute_zscore_stats,
    )

    a = torch.zeros(2, 2, 4)  # zero variance
    b = torch.zeros(2, 4, 2)
    stats = compute_zscore_stats([_per_record(a, b), _per_record(a, b)])
    assert torch.all(stats["q_proj"]["std_A"] >= STD_FLOOR)
    assert torch.all(stats["q_proj"]["std_B"] >= STD_FLOOR)


def test_save_load_roundtrip(tmp_path: Path) -> None:
    from model_training.reconstruction.stats import (
        compute_zscore_stats,
        load_zscore_stats,
        save_zscore_stats,
    )

    a = torch.randn(3, 4, 8)
    b = torch.randn(3, 16, 4)
    stats = compute_zscore_stats([_per_record(a, b), _per_record(a * 0.5, b * 2)])
    path = tmp_path / "zscore.pt"
    save_zscore_stats(stats, path)
    loaded = load_zscore_stats(path)
    for module in stats:
        for key in ("avg_A", "std_A", "avg_B", "std_B"):
            assert torch.allclose(loaded[module][key], stats[module][key])


def test_compute_raises_on_empty_input() -> None:
    from model_training.reconstruction.stats import compute_zscore_stats

    with pytest.raises(ValueError, match="at least one"):
        compute_zscore_stats([])


def test_compute_raises_on_inconsistent_modules() -> None:
    from model_training.reconstruction.stats import compute_zscore_stats

    a = torch.randn(2, 2, 4)
    b = torch.randn(2, 4, 2)
    r1 = {"q_proj": {"A": a, "B": b}}
    r2 = {"v_proj": {"A": a, "B": b}}
    with pytest.raises(ValueError, match="inconsistent modules"):
        compute_zscore_stats([r1, r2])


def test_stats_module_is_cpu_importable() -> None:
    import importlib

    mod = importlib.import_module("model_training.reconstruction.stats")
    assert hasattr(mod, "compute_zscore_stats")
