from contextlib import contextmanager
from unittest.mock import MagicMock

import torch

from rune.model.hypernetwork import extract_activations_with_model


def _fake_model(with_disable: bool):
    m = MagicMock()
    m.parameters.return_value = iter([torch.zeros(1)])
    out = MagicMock()
    out.hidden_states = [torch.zeros(1, 3, 8) for _ in range(4)]
    m.return_value = out
    m.__call__ = lambda **kw: out
    if with_disable:
        called = {"n": 0}

        @contextmanager
        def _dis():
            called["n"] += 1
            yield

        m.disable_adapter = _dis
        m._disable_calls = called
    else:
        del m.disable_adapter
    return m


def test_disable_adapter_used_when_available() -> None:
    tok = MagicMock()
    tok.return_value = {
        "input_ids": torch.zeros(1, 3, dtype=torch.long),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
    }
    model = _fake_model(with_disable=True)
    extract_activations_with_model("ctx", model, tok, layer_indices=[0, 1])
    assert model._disable_calls["n"] == 1


def test_non_peft_model_still_extracts() -> None:
    tok = MagicMock()
    tok.return_value = {
        "input_ids": torch.zeros(1, 3, dtype=torch.long),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
    }
    model = _fake_model(with_disable=False)
    feats, mask = extract_activations_with_model(
        "ctx", model, tok, layer_indices=[0, 1]
    )
    assert feats is not None and mask is not None
