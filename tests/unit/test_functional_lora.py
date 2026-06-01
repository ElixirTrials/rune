"""Equivalence contract for the custom functional-LoRA forward (issue #49 reviewer).

The training student forward applies the adapter by patching each target Linear to
add a LoRA delta on top of the layer's original output. This is used with a 4-bit
(bnb Linear4bit) base, so the math must be verified independently of layer dtype.
"""
import torch

from rune.training.hypernet_distill import _lora_delta


def test_lora_delta_matches_manual_reference() -> None:
    torch.manual_seed(0)
    r, d_in, d_out, seq = 4, 6, 8, 5
    a = torch.randn(1, r, d_in)
    b = torch.randn(1, r, d_out)
    x = torch.randn(1, seq, d_in)
    scaling = 2.0

    got = _lora_delta(x, a, b, scaling)
    # manual: for each position s, delta = scaling * (x_s @ Aᵀ) @ B
    expected = scaling * (x[0] @ a[0].T) @ b[0]  # [seq, d_out]
    assert got.shape == (1, seq, d_out)
    assert torch.allclose(got[0], expected, atol=1e-5)


def test_patched_forward_equals_orig_plus_delta() -> None:
    # Simulate the patched forward: a plain Linear's output + the LoRA delta.
    torch.manual_seed(1)
    r, d_in, d_out, seq = 2, 4, 4, 3
    lin = torch.nn.Linear(d_in, d_out)
    a = torch.randn(1, r, d_in)
    b = torch.randn(1, r, d_out)
    x = torch.randn(1, seq, d_in)
    scaling = 1.5

    base_out = lin(x)
    patched = base_out + _lora_delta(x, a, b, scaling).to(base_out.dtype)
    manual = lin(x) + scaling * (x[0] @ a[0].T) @ b[0]
    assert torch.allclose(patched[0], manual, atol=1e-5)


def test_lora_delta_zero_when_b_zero() -> None:
    a = torch.randn(1, 3, 5)
    b = torch.zeros(1, 3, 7)
    x = torch.randn(1, 4, 5)
    assert float(_lora_delta(x, a, b, 2.0).abs().max()) == 0.0
