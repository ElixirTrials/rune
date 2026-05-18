"""Unit tests for model_pool module (CPU-only, no GPU required)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from model_training.model_pool import ModelPool, get_pool, set_pool


@pytest.fixture(autouse=True)
def _reset_pool() -> Any:
    """Reset the module-level singleton before and after every test."""
    import model_training.model_pool as _mp

    _mp._POOL = None
    yield
    _mp._POOL = None


def test_create_pool() -> None:
    """create() factory returns ModelPool with correct properties."""
    pool = ModelPool.create(
        model_name="Qwen/Qwen3.5-9B",
        device="cpu",
        hypernet_checkpoint_path="/tmp/hn.pt",
        hypernet_variant="qwen_4b_d2l",
    )

    assert isinstance(pool, ModelPool)
    assert pool.model_name == "Qwen/Qwen3.5-9B"
    assert pool.device == "cpu"


def test_create_pool_defaults() -> None:
    """create() uses sensible defaults when optional args are omitted."""
    pool = ModelPool.create(model_name="test-model")

    assert pool.model_name == "test-model"
    assert pool.device == "cuda"


def test_base_model_lazy_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    """base_model() calls from_pretrained only on the first invocation."""
    fake_model = MagicMock()
    fake_model.to.return_value = fake_model
    fake_tokenizer = MagicMock()
    fake_tokenizer.pad_token = "pad"

    fake_dtype = MagicMock()

    pool = ModelPool.create(model_name="test-model", device="cpu")

    with (
        patch("transformers.AutoModelForCausalLM") as mock_amc,
        patch("transformers.AutoTokenizer") as mock_at,
    ):
        mock_amc.from_pretrained.return_value = fake_model
        mock_at.from_pretrained.return_value = fake_tokenizer

        monkeypatch.setattr(pool, "_resolve_dtype", lambda: fake_dtype)

        model1, tok1 = pool.base_model()
        model2, tok2 = pool.base_model()

    assert model1 is model2
    assert tok1 is tok2
    mock_amc.from_pretrained.assert_called_once_with(
        "test-model", torch_dtype=fake_dtype
    )
    mock_at.from_pretrained.assert_called_once_with("test-model")


def test_base_model_sets_pad_token_when_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """base_model() sets pad_token = eos_token when pad_token is None."""
    fake_model = MagicMock()
    fake_model.to.return_value = fake_model

    fake_tokenizer = MagicMock()
    fake_tokenizer.pad_token = None
    fake_tokenizer.eos_token = "<eos>"

    pool = ModelPool.create(model_name="test-model", device="cpu")
    monkeypatch.setattr(pool, "_resolve_dtype", lambda: MagicMock())

    with (
        patch("transformers.AutoModelForCausalLM") as mock_amc,
        patch("transformers.AutoTokenizer") as mock_at,
    ):
        mock_amc.from_pretrained.return_value = fake_model
        mock_at.from_pretrained.return_value = fake_tokenizer

        pool.base_model()

    assert fake_tokenizer.pad_token == "<eos>"


def test_hypernetwork_lazy_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    """hypernetwork() calls load_sakana_checkpoint only on the first invocation."""
    fake_hypernet = MagicMock()
    fake_hc = MagicMock()

    import model_training.hypernetwork as _hn_mod

    load_calls: list[dict[str, Any]] = []

    def _fake_load(
        checkpoint_path: Any = None,
        variant: str = "gemma_demo",
        device: str = "cpu",
    ) -> tuple[MagicMock, MagicMock]:
        load_calls.append(
            {"checkpoint_path": checkpoint_path, "variant": variant, "device": device}
        )
        return fake_hypernet, fake_hc

    monkeypatch.setattr(_hn_mod, "load_hypernetwork", _fake_load)

    pool = ModelPool.create(
        model_name="test-model",
        device="cpu",
        hypernet_checkpoint_path="/tmp/hn.pt",
        hypernet_variant="qwen_4b_d2l",
    )

    hn1, hc1 = pool.hypernetwork()
    hn2, hc2 = pool.hypernetwork()

    assert hn1 is hn2
    assert hc1 is hc2
    assert len(load_calls) == 1
    assert load_calls[0]["checkpoint_path"] == "/tmp/hn.pt"
    assert load_calls[0]["variant"] == "qwen_4b_d2l"
    assert load_calls[0]["device"] == "cpu"


def test_get_set_pool_singleton() -> None:
    """set_pool/get_pool round-trips the same ModelPool instance."""
    pool = ModelPool.create(model_name="test-model", device="cpu")
    set_pool(pool)
    retrieved = get_pool()
    assert retrieved is pool


def test_get_pool_uninitialized_raises() -> None:
    """get_pool() raises RuntimeError before set_pool() is called."""
    with pytest.raises(RuntimeError, match="not initialised"):
        get_pool()


def test_release_clears_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """After release(), next base_model() call reloads from transformers."""
    fake_model = MagicMock()
    fake_model.to.return_value = fake_model
    fake_tokenizer = MagicMock()
    fake_tokenizer.pad_token = "pad"

    pool = ModelPool.create(model_name="test-model", device="cpu")
    monkeypatch.setattr(pool, "_resolve_dtype", lambda: MagicMock())

    with (
        patch("transformers.AutoModelForCausalLM") as mock_amc,
        patch("transformers.AutoTokenizer") as mock_at,
    ):
        mock_amc.from_pretrained.return_value = fake_model
        mock_at.from_pretrained.return_value = fake_tokenizer

        pool.base_model()
        pool.release()
        pool.base_model()

    assert mock_amc.from_pretrained.call_count == 2


def test_release_clears_hypernetwork_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """After release(), next hypernetwork() call reloads."""
    fake_hypernet = MagicMock()
    fake_hc = MagicMock()

    import model_training.hypernetwork as _hn_mod

    load_calls: list[int] = []

    def _fake_load(**kwargs: Any) -> tuple[MagicMock, MagicMock]:
        load_calls.append(1)
        return fake_hypernet, fake_hc

    monkeypatch.setattr(_hn_mod, "load_hypernetwork", _fake_load)

    pool = ModelPool.create(model_name="test-model", device="cpu")
    pool.hypernetwork()
    pool.release()
    pool.hypernetwork()

    assert len(load_calls) == 2
