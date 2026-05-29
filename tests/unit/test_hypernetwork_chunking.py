"""Chunked gated-MLP forward caps peak memory regardless of batch dim.

Regression: the modality_projection MLP receives (n_layers, seq_len, hidden).
The 4x gated intermediate is sized by n_layers * seq_len, not seq_len alone, so
chunking must span the flattened token count — chunking only the seq dim leaves
the n_layers (=32) multiplier uncapped and OOMs on long trajectories.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from rune.model.hypernetwork import _chunk_gated_mlp  # noqa: E402


class _GatedMLP(nn.Module):
    def __init__(self, hidden: int, intermediate: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def test_chunks_over_flattened_tokens_not_just_seq() -> None:

    hidden = 8
    mlp = _GatedMLP(hidden, hidden * 4)
    chunk = 2048

    calls = {"n": 0, "max_tokens": 0}

    def counting_fwd(module: object, x: torch.Tensor) -> torch.Tensor:
        calls["n"] += 1
        calls["max_tokens"] = max(calls["max_tokens"], x.reshape(-1, hidden).shape[0])
        return _GatedMLP.forward(module, x)  # type: ignore[arg-type]

    # seq_len=100 (<= chunk) but n_layers*seq_len = 3200 (> chunk): must chunk.
    x = torch.randn(32, 100, hidden)
    out = _chunk_gated_mlp(counting_fwd, mlp, x, chunk)

    ref = _GatedMLP.forward(mlp, x)
    assert out.shape == x.shape
    assert torch.allclose(out, ref, atol=1e-5)
    assert calls["n"] > 1, "batched input above chunk size was not chunked"
    assert calls["max_tokens"] <= chunk


def test_small_input_runs_in_one_pass() -> None:

    hidden = 8
    mlp = _GatedMLP(hidden, hidden * 4)

    calls = {"n": 0}

    def counting_fwd(module: object, x: torch.Tensor) -> torch.Tensor:
        calls["n"] += 1
        return _GatedMLP.forward(module, x)  # type: ignore[arg-type]

    x = torch.randn(1, 16, hidden)  # 16 tokens, well under threshold
    out = _chunk_gated_mlp(counting_fwd, mlp, x, 2048)

    assert out.shape == x.shape
    assert calls["n"] == 1


class _GatedMLPProj(nn.Module):
    """Gated MLP whose output feature dim differs from the input (like the real
    modality_projection: hidden -> out_dim)."""

    def __init__(self, hidden: int, intermediate: int, out_dim: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, out_dim, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def test_chunks_preserve_output_dim_when_mlp_changes_feature_dim() -> None:
    hidden, out_dim = 8, 4
    mlp = _GatedMLPProj(hidden, hidden * 4, out_dim)
    chunk = 2048

    def fwd(module: object, x: torch.Tensor) -> torch.Tensor:
        return _GatedMLPProj.forward(module, x)  # type: ignore[arg-type]

    # n_layers*seq_len = 32*100 = 3200 > chunk -> chunked path.
    x = torch.randn(32, 100, hidden)
    out = _chunk_gated_mlp(fwd, mlp, x, chunk)
    ref = _GatedMLPProj.forward(mlp, x)

    assert out.shape == (32, 100, out_dim)        # leading dims preserved, OUTPUT feature dim
    assert torch.allclose(out, ref, atol=1e-5)
