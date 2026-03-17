"""Tests for TransformersProvider adapter lifecycle.

Verifies that PEFT adapters are loaded, activated, deactivated, and unloaded
correctly -- without accumulating wrappers or leaking GPU memory.

Uses sys.modules patching to avoid importing the real peft/torch stack
in CPU-only CI (INFRA-05 pattern).
"""

from __future__ import annotations

import asyncio
import sys
import types
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _fake_peft(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Inject a fake ``peft`` module so ``from peft import PeftModel`` works."""
    mock_peft_cls = MagicMock(name="PeftModel")
    fake_peft = types.ModuleType("peft")
    fake_peft.PeftModel = mock_peft_cls  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "peft", fake_peft)
    return mock_peft_cls


@pytest.fixture()
def peft_cls(_fake_peft: MagicMock) -> MagicMock:
    """Return the mock PeftModel class."""
    return _fake_peft


@pytest.fixture()
def provider() -> "TransformersProvider":  # noqa: F821
    """Create a provider with a fake base model pre-loaded."""
    from inference.transformers_provider import TransformersProvider

    p = TransformersProvider(model_name="test-model", device="cpu")
    base = MagicMock(name="base_model")
    p._model = base
    p._base_model = base
    p._tokenizer = MagicMock()
    return p


def _make_wrapped() -> MagicMock:
    wrapped = MagicMock(name="peft_model")
    wrapped.to.return_value = wrapped
    return wrapped


def test_first_adapter_creates_peft_wrapper(
    provider: "TransformersProvider",  # noqa: F821
    peft_cls: MagicMock,
) -> None:
    """First adapter wraps base model via from_pretrained."""
    wrapped = _make_wrapped()
    peft_cls.from_pretrained.return_value = wrapped

    provider._loaded_adapters["a1"] = "/path/a1"
    provider._activate_adapter("a1")

    peft_cls.from_pretrained.assert_called_once_with(
        provider._base_model, "/path/a1", adapter_name="a1"
    )
    assert provider._is_peft_wrapped is True
    assert provider._active_adapter == "a1"
    assert provider._model is wrapped


def test_second_adapter_reuses_wrapper(
    provider: "TransformersProvider",  # noqa: F821
    peft_cls: MagicMock,
) -> None:
    """Second adapter reuses wrapper via load_adapter."""
    wrapped = _make_wrapped()
    wrapped.peft_config = {"a1": MagicMock()}
    peft_cls.from_pretrained.return_value = wrapped

    provider._loaded_adapters["a1"] = "/path/a1"
    provider._loaded_adapters["a2"] = "/path/a2"

    provider._activate_adapter("a1")
    provider._activate_adapter("a2")

    assert peft_cls.from_pretrained.call_count == 1
    wrapped.load_adapter.assert_called_once_with("/path/a2", adapter_name="a2")
    wrapped.set_adapter.assert_called_with("a2")
    wrapped.enable_adapter_layers.assert_called()


def test_deactivate_disables_layers_keeps_wrapper(
    provider: "TransformersProvider",  # noqa: F821
    peft_cls: MagicMock,
) -> None:
    """Deactivate disables layers but keeps the wrapper."""
    wrapped = _make_wrapped()
    peft_cls.from_pretrained.return_value = wrapped

    provider._loaded_adapters["a1"] = "/path/a1"

    provider._activate_adapter("a1")
    provider._deactivate_adapter()

    wrapped.disable_adapter_layers.assert_called_once()
    assert provider._active_adapter is None
    assert provider._is_peft_wrapped is True
    assert provider._model is wrapped


def test_unload_calls_delete_adapter(
    provider: "TransformersProvider",  # noqa: F821
    peft_cls: MagicMock,
) -> None:
    """Unload calls delete_adapter to free GPU memory."""
    wrapped = _make_wrapped()
    wrapped.peft_config = {"a1": MagicMock(), "a2": MagicMock()}
    peft_cls.from_pretrained.return_value = wrapped

    provider._loaded_adapters["a1"] = "/path/a1"
    provider._loaded_adapters["a2"] = "/path/a2"

    provider._activate_adapter("a1")
    asyncio.run(provider.unload_adapter("a1"))

    wrapped.delete_adapter.assert_called_once_with("a1")
    assert provider._is_peft_wrapped is True


def test_unload_last_adapter_reverts_to_base(
    provider: "TransformersProvider",  # noqa: F821
    peft_cls: MagicMock,
) -> None:
    """Unloading the last adapter reverts to base model."""
    wrapped = _make_wrapped()
    wrapped.peft_config = {"a1": MagicMock()}
    peft_cls.from_pretrained.return_value = wrapped

    provider._loaded_adapters["a1"] = "/path/a1"

    provider._activate_adapter("a1")

    base = provider._base_model
    asyncio.run(provider.unload_adapter("a1"))

    assert provider._is_peft_wrapped is False
    assert provider._model is base


def test_full_lifecycle_no_accumulation(
    provider: "TransformersProvider",  # noqa: F821
    peft_cls: MagicMock,
) -> None:
    """Full lifecycle: load/activate/unload without accumulation."""
    wrapped = _make_wrapped()
    wrapped.peft_config = {}
    peft_cls.from_pretrained.return_value = wrapped

    def _track(path: str, adapter_name: str) -> None:
        wrapped.peft_config[adapter_name] = MagicMock()

    wrapped.load_adapter.side_effect = _track

    asyncio.run(provider.load_adapter("a1", "/path/a1"))
    asyncio.run(provider.load_adapter("a2", "/path/a2"))

    provider._activate_adapter("a1")
    wrapped.peft_config["a1"] = MagicMock()
    assert peft_cls.from_pretrained.call_count == 1

    provider._activate_adapter("a2")
    assert peft_cls.from_pretrained.call_count == 1

    asyncio.run(provider.unload_adapter("a1"))
    wrapped.delete_adapter.assert_called_with("a1")
    assert provider._is_peft_wrapped is True

    asyncio.run(provider.unload_adapter("a2"))
    assert provider._is_peft_wrapped is False
    assert provider._model is provider._base_model
