"""CPU tests for ``_override_lora_alpha`` and ``_override_lora_dropout``.

PEFT's ``LoraLayer`` caches scaling and dropout modules per-adapter, so
mutating ``peft_config`` after ``PeftModel.from_pretrained`` does not
propagate to the module tree. These helpers walk the tree and patch the
cached values in place so HPO trials can vary alpha/dropout without
discarding the DeltaCoder warm-start.

Tests use a minimal fake PEFT model (no torch, no transformers) that
mimics the attributes the helpers rely on: ``.modules()``,
``.peft_config``, per-layer ``.scaling`` / ``.r`` dicts, and a
``lora_dropout`` object that behaves like a ModuleDict.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from model_training.trainer import (
    _override_lora_alpha,
    _override_lora_dropout,
)


class _FakeDropout:
    """Stand-in for ``torch.nn.Dropout`` — only ``.p`` matters for override."""

    def __init__(self, p: float) -> None:
        self.p = p


@dataclass
class _FakeLoraLayer:
    """A minimal PEFT-like layer carrying per-adapter scaling + dropout."""

    scaling: dict[str, float] = field(default_factory=dict)
    r: dict[str, int] = field(default_factory=dict)
    lora_dropout: dict[str, _FakeDropout] = field(default_factory=dict)


@dataclass
class _FakePeftConfig:
    lora_alpha: int = 32
    lora_dropout: float = 0.1


class _FakePeftModel:
    """Enough surface for the override helpers to work against in tests."""

    def __init__(self, layers: list[_FakeLoraLayer], adapter_name: str) -> None:
        self._layers = layers
        self.peft_config = {adapter_name: _FakePeftConfig()}

    def modules(self) -> list[_FakeLoraLayer]:
        return list(self._layers)


def _make_model(
    adapter_name: str = "default",
    *,
    saved_alpha: int = 32,
    saved_rank: int = 64,
    saved_dropout: float = 0.1,
    n_layers: int = 3,
) -> _FakePeftModel:
    layers = [
        _FakeLoraLayer(
            scaling={adapter_name: saved_alpha / saved_rank},
            r={adapter_name: saved_rank},
            lora_dropout={adapter_name: _FakeDropout(saved_dropout)},
        )
        for _ in range(n_layers)
    ]
    return _FakePeftModel(layers, adapter_name)


def test_override_alpha_updates_every_matching_layer() -> None:
    model = _make_model(saved_alpha=32, saved_rank=64)
    _override_lora_alpha(model, "default", new_alpha=128)
    for layer in model._layers:  # noqa: SLF001 — reaching into test fake
        assert layer.scaling["default"] == 128 / 64
    assert model.peft_config["default"].lora_alpha == 128


def test_override_alpha_noop_on_unknown_adapter() -> None:
    """Layers without the named adapter are left untouched."""
    model = _make_model(adapter_name="default")
    before = [dict(layer.scaling) for layer in model._layers]  # noqa: SLF001
    _override_lora_alpha(model, "not-this-one", new_alpha=256)
    after = [dict(layer.scaling) for layer in model._layers]  # noqa: SLF001
    assert before == after


def test_override_alpha_ignores_nonlora_modules() -> None:
    """Modules missing scaling/r dicts must be skipped silently."""
    model = _make_model()
    # Inject a non-LoRA module — the helper must not crash.
    model._layers.append(object())  # type: ignore[arg-type]
    _override_lora_alpha(model, "default", new_alpha=64)
    # Surviving layers updated; the non-LoRA object is ignored.
    for layer in model._layers[:-1]:  # noqa: SLF001
        assert layer.scaling["default"] == 64 / 64


def test_override_dropout_updates_every_matching_layer() -> None:
    model = _make_model(saved_dropout=0.1)
    _override_lora_dropout(model, "default", new_p=0.05)
    for layer in model._layers:  # noqa: SLF001
        assert layer.lora_dropout["default"].p == pytest.approx(0.05)
    assert model.peft_config["default"].lora_dropout == pytest.approx(0.05)


def test_override_dropout_accepts_zero() -> None:
    """0.0 is a valid probability — effectively disables LoRA dropout."""
    model = _make_model()
    _override_lora_dropout(model, "default", new_p=0.0)
    for layer in model._layers:  # noqa: SLF001
        assert layer.lora_dropout["default"].p == 0.0


def test_override_dropout_rejects_out_of_range() -> None:
    model = _make_model()
    with pytest.raises(ValueError, match="in \\[0, 1\\]"):
        _override_lora_dropout(model, "default", new_p=-0.1)
    with pytest.raises(ValueError, match="in \\[0, 1\\]"):
        _override_lora_dropout(model, "default", new_p=1.01)


def test_override_dropout_ignores_nonlora_modules() -> None:
    model = _make_model()
    model._layers.append(object())  # type: ignore[arg-type]
    _override_lora_dropout(model, "default", new_p=0.25)
    for layer in model._layers[:-1]:  # noqa: SLF001
        assert layer.lora_dropout["default"].p == pytest.approx(0.25)


def test_train_qlora_exposes_override_kwargs() -> None:
    """Trainer signature must accept override_lora_alpha and override_lora_dropout."""
    import inspect

    from model_training.trainer import train_and_register, train_qlora

    sig_q = inspect.signature(train_qlora)
    sig_r = inspect.signature(train_and_register)
    for sig in (sig_q, sig_r):
        assert "override_lora_alpha" in sig.parameters
        assert "override_lora_dropout" in sig.parameters
        assert sig.parameters["override_lora_alpha"].default is None
        assert sig.parameters["override_lora_dropout"].default is None
