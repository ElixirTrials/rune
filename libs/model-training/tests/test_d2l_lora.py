"""Tests for apply_functional_lora context manager.

Verifies LORA-01 (F.linear patching), LORA-02 (autograd continuity),
and LORA-03 (forward restoration on exit).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn

from model_training.d2l_lora import apply_functional_lora

# ---------------------------------------------------------------------------
# Fake model hierarchy matching real Qwen3-style module paths:
#   model.layers.{i}.self_attn.{q,k,v,o}_proj
# ---------------------------------------------------------------------------

D = 16
R = 2
N_LAYERS = 3
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
LAYER_INDICES = [0, 1, 2]


class _FakeSelfAttn(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)


class _FakeLayer(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.self_attn = _FakeSelfAttn(d)


class _FakeModel(nn.Module):
    def __init__(self, d: int, n_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_FakeLayer(d) for _ in range(n_layers)])

    def forward(self, x: Any) -> Any:
        for layer in self.layers:
            x = layer.self_attn.q_proj(x)
        return x


def _make_fake_lora_dict(
    target_modules: list[str], n_layers: int, r: int, d: int
) -> dict[str, dict[str, torch.Tensor]]:
    return {
        mod: {
            "A": torch.randn(1, n_layers, r, d, requires_grad=True),
            "B": torch.randn(1, n_layers, r, d, requires_grad=True),
        }
        for mod in target_modules
    }


def _make_fake_hc(
    target_modules: list[str],
    layer_indices: list[int],
    r: int = R,
    lora_alpha: float = 1.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        lora_config=SimpleNamespace(
            target_modules=target_modules,
            r=r,
            lora_alpha=lora_alpha,
        ),
        layer_indices=layer_indices,
    )


# ---------------------------------------------------------------------------
# LORA-02: Autograd continuity — A/B leaf tensors get gradients after backward
# ---------------------------------------------------------------------------


class TestAutogradContinuity:
    def test_ab_grads_after_backward(self) -> None:
        model = _FakeModel(D, N_LAYERS)
        lora_dict = _make_fake_lora_dict(TARGET_MODULES, N_LAYERS, R, D)
        hc = _make_fake_hc(TARGET_MODULES, LAYER_INDICES)

        x = torch.randn(1, 4, D)
        with apply_functional_lora(model, lora_dict, hc):
            out = model(x)

        loss = out.sum()
        loss.backward()

        for mod_name in TARGET_MODULES:
            assert lora_dict[mod_name]["A"].grad is not None, (
                f"{mod_name} A.grad is None after backward"
            )
            assert lora_dict[mod_name]["B"].grad is not None, (
                f"{mod_name} B.grad is None after backward"
            )


# ---------------------------------------------------------------------------
# LORA-01: Inside context, forward is patched (not original)
# ---------------------------------------------------------------------------


class TestForwardPatching:
    def test_forward_replaced_inside_context(self) -> None:
        model = _FakeModel(D, N_LAYERS)
        lora_dict = _make_fake_lora_dict(TARGET_MODULES, N_LAYERS, R, D)
        hc = _make_fake_hc(TARGET_MODULES, LAYER_INDICES)

        orig_forward = model.layers[0].self_attn.q_proj.forward

        with apply_functional_lora(model, lora_dict, hc):
            patched_forward = model.layers[0].self_attn.q_proj.forward
            assert patched_forward is not orig_forward, (
                "forward should be patched inside context"
            )

    def test_patched_output_equals_base_plus_lora(self) -> None:
        """Numerically verify: patched_forward(x) == base_out + lora_out."""
        model = _FakeModel(D, N_LAYERS)
        hc = _make_fake_hc(TARGET_MODULES, LAYER_INDICES, lora_alpha=2.0)
        scale = hc.lora_config.lora_alpha / hc.lora_config.r

        # Fixed tensors for reproducibility
        torch.manual_seed(42)
        lora_dict = _make_fake_lora_dict(TARGET_MODULES, N_LAYERS, R, D)
        x = torch.randn(1, 4, D)

        # Get the q_proj at layer 0
        q_proj = model.layers[0].self_attn.q_proj
        W = q_proj.weight.detach()

        # Expected: base + lora
        base_out = torch.nn.functional.linear(x, W)
        A = lora_dict["q_proj"]["A"][0, 0]  # (r, d_in)
        B = lora_dict["q_proj"]["B"][0, 0]  # (r, d_out)
        lora_Ax = torch.nn.functional.linear(x, A)
        lora_out = torch.nn.functional.linear(lora_Ax, B.t()) * scale

        expected = base_out + lora_out

        with apply_functional_lora(model, lora_dict, hc):
            actual = q_proj.forward(x)

        assert torch.allclose(actual, expected, atol=1e-5), (
            f"Output mismatch: max diff = {(actual - expected).abs().max().item()}"
        )


# ---------------------------------------------------------------------------
# LORA-03: Forward restored after context exit
# ---------------------------------------------------------------------------


class TestForwardRestoration:
    def test_forward_restored_after_exit(self) -> None:
        model = _FakeModel(D, N_LAYERS)
        lora_dict = _make_fake_lora_dict(TARGET_MODULES, N_LAYERS, R, D)
        hc = _make_fake_hc(TARGET_MODULES, LAYER_INDICES)

        orig_forward = model.layers[0].self_attn.q_proj.forward

        with apply_functional_lora(model, lora_dict, hc):
            pass  # just enter and exit

        assert model.layers[0].self_attn.q_proj.forward is orig_forward, (
            "forward should be restored after context exit"
        )

    def test_forward_restored_after_exception(self) -> None:
        model = _FakeModel(D, N_LAYERS)
        lora_dict = _make_fake_lora_dict(TARGET_MODULES, N_LAYERS, R, D)
        hc = _make_fake_hc(TARGET_MODULES, LAYER_INDICES)

        orig_forward = model.layers[0].self_attn.q_proj.forward

        try:
            with apply_functional_lora(model, lora_dict, hc):
                raise ValueError("deliberate error")
        except ValueError:
            pass

        assert model.layers[0].self_attn.q_proj.forward is orig_forward, (
            "forward should be restored even after exception"
        )


# ---------------------------------------------------------------------------
# Shape mismatch raises RuntimeError
# ---------------------------------------------------------------------------


class TestShapeMismatch:
    def test_shape_mismatch_raises_runtime_error(self) -> None:
        model = _FakeModel(D, N_LAYERS)
        hc = _make_fake_hc(TARGET_MODULES, LAYER_INDICES)

        # Create A with wrong d_in (D+1 instead of D)
        bad_lora_dict = {
            mod: {
                "A": torch.randn(1, N_LAYERS, R, D + 1, requires_grad=True),
                "B": torch.randn(1, N_LAYERS, R, D, requires_grad=True),
            }
            for mod in TARGET_MODULES
        }

        import pytest

        with pytest.raises(RuntimeError, match="Shape mismatch"):
            with apply_functional_lora(model, bad_lora_dict, hc):
                pass


# ---------------------------------------------------------------------------
# Skip behavior: layers not in hc.layer_indices are not patched
# ---------------------------------------------------------------------------


class TestSkipBehavior:
    def test_non_target_layers_not_patched(self) -> None:
        model = _FakeModel(D, N_LAYERS)
        lora_dict = _make_fake_lora_dict(TARGET_MODULES, 1, R, D)
        # Only layer index 1 in layer_indices — layers 0 and 2 should be skipped
        hc = _make_fake_hc(TARGET_MODULES, [1])

        orig_fwd_layer0 = model.layers[0].self_attn.q_proj.forward
        orig_fwd_layer2 = model.layers[2].self_attn.q_proj.forward

        with apply_functional_lora(model, lora_dict, hc):
            assert model.layers[0].self_attn.q_proj.forward is orig_fwd_layer0, (
                "layer 0 should NOT be patched (not in layer_indices)"
            )
            assert model.layers[2].self_attn.q_proj.forward is orig_fwd_layer2, (
                "layer 2 should NOT be patched (not in layer_indices)"
            )
            # But layer 1 SHOULD be patched
            assert model.layers[1].self_attn.q_proj.forward is not orig_fwd_layer0
