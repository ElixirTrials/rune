"""Tests for _ThinkingBudgetProcessor logits processor."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest
import torch

from inference.transformers_provider import _ThinkingBudgetProcessor


class TestThinkingBudgetProcessor:
    """Unit tests using synthetic tensors — no model load."""

    def _make_input_ids(self, prompt: list[int], generated: list[int]) -> torch.Tensor:
        return torch.tensor([prompt + generated], dtype=torch.long)

    def _make_scores(self, vocab_size: int = 100) -> torch.Tensor:
        return torch.randn(1, vocab_size)

    def test_no_op_under_budget(self) -> None:
        proc = _ThinkingBudgetProcessor(end_think_token_id=5, budget=10, prompt_len=3)
        ids = self._make_input_ids([1, 2, 3], [10, 11, 12])
        scores = self._make_scores()
        original = scores.clone()
        result = proc(ids, scores)
        assert torch.equal(result, original)

    def test_forces_end_think_at_budget(self) -> None:
        proc = _ThinkingBudgetProcessor(end_think_token_id=5, budget=4, prompt_len=3)
        ids = self._make_input_ids([1, 2, 3], [10, 11, 12, 13])
        scores = self._make_scores()
        result = proc(ids, scores)
        assert result[0, 5].item() == 0.0
        mask = torch.arange(100) != 5
        assert (result[0, mask] == float("-inf")).all()

    def test_no_op_after_end_think_emitted(self) -> None:
        etid = 5
        proc = _ThinkingBudgetProcessor(end_think_token_id=etid, budget=4, prompt_len=3)
        ids = self._make_input_ids([1, 2, 3], [10, etid, 12, 13, 14])
        scores = self._make_scores()
        original = scores.clone()
        result = proc(ids, scores)
        assert torch.equal(result, original)
        assert proc._done is True

    def test_done_flag_persists(self) -> None:
        etid = 5
        proc = _ThinkingBudgetProcessor(end_think_token_id=etid, budget=2, prompt_len=3)
        ids_with_etid = self._make_input_ids([1, 2, 3], [10, etid])
        proc(ids_with_etid, self._make_scores())
        assert proc._done is True

        ids_over_budget = self._make_input_ids([1, 2, 3], [10, etid, 12, 13, 14])
        scores = self._make_scores()
        original = scores.clone()
        result = proc(ids_over_budget, scores)
        assert torch.equal(result, original)

    def test_forces_at_exact_budget(self) -> None:
        proc = _ThinkingBudgetProcessor(end_think_token_id=7, budget=3, prompt_len=2)
        ids = self._make_input_ids([1, 2], [10, 11, 12])
        scores = self._make_scores(50)
        result = proc(ids, scores)
        assert result[0, 7].item() == 0.0
        assert (result[0, :7] == float("-inf")).all()
        assert (result[0, 8:] == float("-inf")).all()


class TestThinkingBudgetWiring:
    """Verify the processor is wired into generate() correctly."""

    def test_logits_processor_passed_when_thinking_enabled(self) -> None:
        from inference.transformers_provider import TransformersProvider

        p = TransformersProvider(model_name="test", device="cpu")
        model = MagicMock(name="model")
        model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        p._model = model
        p._base_model = model
        tok = MagicMock()
        tok.pad_token = "<pad>"
        tok.pad_token_id = 0
        tok.eos_token = "<eos>"
        tok.unk_token_id = 99
        tok.return_value = {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.ones(1, 2)}
        tok.convert_tokens_to_ids.side_effect = lambda t: {
            "<think>": 50, "</think>": 51
        }.get(t, 99)
        tok.decode.return_value = "output"
        tok.apply_chat_template.return_value = "formatted"
        p._tokenizer = tok

        with patch("torch.no_grad"):
            asyncio.run(
                p.generate(
                    prompt="test",
                    model="m",
                    enable_thinking=True,
                    thinking_budget=512,
                    temperature=0.6,
                )
            )

        call_kwargs = model.generate.call_args
        assert "logits_processor" in call_kwargs.kwargs
        processors = call_kwargs.kwargs["logits_processor"]
        assert len(processors) == 1
        assert isinstance(processors[0], _ThinkingBudgetProcessor)

    def test_no_processor_when_thinking_disabled(self) -> None:
        from inference.transformers_provider import TransformersProvider

        p = TransformersProvider(model_name="test", device="cpu")
        model = MagicMock(name="model")
        model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        p._model = model
        p._base_model = model
        tok = MagicMock()
        tok.pad_token = "<pad>"
        tok.pad_token_id = 0
        tok.eos_token = "<eos>"
        tok.unk_token_id = 99
        tok.return_value = {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.ones(1, 2)}
        tok.convert_tokens_to_ids.side_effect = lambda t: {
            "<think>": 50, "</think>": 51
        }.get(t, 99)
        tok.decode.return_value = "output"
        tok.apply_chat_template.return_value = "formatted"
        p._tokenizer = tok

        with patch("torch.no_grad"):
            asyncio.run(
                p.generate(
                    prompt="test",
                    model="m",
                    enable_thinking=False,
                    thinking_budget=0,
                )
            )

        call_kwargs = model.generate.call_args
        assert "logits_processor" not in call_kwargs.kwargs
