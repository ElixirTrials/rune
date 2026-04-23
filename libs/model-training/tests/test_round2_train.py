"""CPU-only unit tests for round2_train module."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from model_training.round2_train import (
    _teacher_forward_with_oracle,
)


class _StubLogits:
    """Stand-in for the ``.logits`` attribute of an HF model output."""

    def __init__(self, marker: str) -> None:
        self.marker = marker


class _StubOutput:
    def __init__(self, marker: str) -> None:
        self.logits = _StubLogits(marker)


class _FakeCtxMgr:
    """Context manager that tracks enter/exit counts via outer closure."""

    def __init__(self, enter_log: list, exit_log: list, name: str) -> None:
        self._enter_log = enter_log
        self._exit_log = exit_log
        self._name = name

    def __enter__(self) -> None:
        self._enter_log.append(self._name)

    def __exit__(self, *exc: object) -> None:
        self._exit_log.append(self._name)


def test_teacher_forward_applies_oracle_lora_dict_via_functional_lora(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """oracle_lora_dict is not None → apply_functional_lora wraps the base forward."""
    from model_training import round2_train

    enter_log: list = []
    exit_log: list = []
    applied_dicts: list = []

    def _fake_apply(base: object, lora_dict: object, hc: object) -> _FakeCtxMgr:
        applied_dicts.append(lora_dict)
        return _FakeCtxMgr(enter_log, exit_log, "functional_lora")

    monkeypatch.setattr(round2_train, "_apply_functional_lora", _fake_apply)

    base = MagicMock(name="base")
    base.return_value = _StubOutput("from_base_with_oracle_patch")
    oracle_dict = {"q_proj": {"A": MagicMock(), "B": MagicMock()}}
    hc = MagicMock(name="hc")
    inputs = {"input_ids": MagicMock(), "attention_mask": MagicMock()}

    logits = _teacher_forward_with_oracle(
        base_model=base,
        oracle_lora_dict=oracle_dict,
        hc=hc,
        inputs=inputs,
    )

    assert logits.marker == "from_base_with_oracle_patch"
    assert applied_dicts == [oracle_dict]
    assert enter_log == ["functional_lora"]
    assert exit_log == ["functional_lora"]
    base.assert_called_once_with(**inputs, output_hidden_states=False)


def test_teacher_forward_bypasses_functional_lora_when_oracle_is_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """oracle_lora_dict=None → bare base model forward, functional_lora NOT called."""
    from model_training import round2_train

    def _must_not_call(*a: object, **kw: object) -> None:
        raise AssertionError(
            "apply_functional_lora must not be called when oracle is None"
        )

    monkeypatch.setattr(round2_train, "_apply_functional_lora", _must_not_call)

    base = MagicMock(name="base")
    base.return_value = _StubOutput("from_bare_base")
    hc = MagicMock(name="hc")
    inputs = {"input_ids": MagicMock(), "attention_mask": MagicMock()}

    logits = _teacher_forward_with_oracle(
        base_model=base,
        oracle_lora_dict=None,
        hc=hc,
        inputs=inputs,
    )

    assert logits.marker == "from_bare_base"
    base.assert_called_once_with(**inputs, output_hidden_states=False)
