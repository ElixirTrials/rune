"""Unit tests for ModelWrapper."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rune.config import PipelineConfig
from rune.model.adapter import AdapterResult
from rune.model.inference import GenerationResult
from rune.model.wrapper import ModelWrapper


class TestModelWrapper:
    def _make_wrapper(self) -> Any:
        cfg = PipelineConfig()
        base_model = MagicMock()
        tokenizer = MagicMock()
        hypernet = MagicMock()
        hypernet.config = MagicMock()
        hypernet.config.layer_indices = [0, 1, 2]
        return ModelWrapper(base_model, tokenizer, hypernet, config=cfg)

    def test_generate_adapter_returns_adapter_result(self) -> None:
        fake_state_dict = {"weight": MagicMock()}
        with patch(
            "rune.model.wrapper.generate_adapter_weights",
            return_value=fake_state_dict,
        ):
            wrapper = self._make_wrapper()
            result = wrapper.generate_adapter("some trajectory")

        assert isinstance(result, AdapterResult)
        assert result.state_dict is fake_state_dict
        assert isinstance(result.adapter_id, str)
        assert len(result.adapter_id) > 0

    def test_generate_adapter_unique_ids(self) -> None:
        fake_state_dict: dict[str, Any] = {}
        with patch(
            "rune.model.wrapper.generate_adapter_weights",
            return_value=fake_state_dict,
        ):
            wrapper = self._make_wrapper()
            r1 = wrapper.generate_adapter("traj1")
            r2 = wrapper.generate_adapter("traj2")

        assert r1.adapter_id != r2.adapter_id

    def test_hotswap_adapter_delegates(self) -> None:
        with patch("rune.model.wrapper.hotswap_adapter_fn") as mock_swap:
            wrapper = self._make_wrapper()
            state_dict: dict[str, Any] = {"k": MagicMock()}
            wrapper.hotswap_adapter(state_dict)

        mock_swap.assert_called_once_with(wrapper._base_model, state_dict)

    def test_generate_delegates(self) -> None:
        expected = GenerationResult(text="hello", thinking="", tokens_used=5)
        with patch(
            "rune.model.wrapper.inference_generate",
            new=AsyncMock(return_value=expected),
        ):
            wrapper = self._make_wrapper()
            result = asyncio.run(
                wrapper.generate(
                    prompt="p",
                    system_prompt="s",
                    output_schema=None,
                    max_tokens=512,
                )
            )

        assert result is expected

    def test_generate_default_args(self) -> None:
        expected = GenerationResult(text="out", thinking="", tokens_used=3)
        with patch(
            "rune.model.wrapper.inference_generate",
            new=AsyncMock(return_value=expected),
        ) as mock_gen:
            wrapper = self._make_wrapper()
            asyncio.run(wrapper.generate(prompt="x"))
            call_kwargs = mock_gen.call_args.kwargs
            assert call_kwargs["system_prompt"] == ""
            assert call_kwargs["output_schema"] is None
            assert call_kwargs["max_tokens"] == 2048

    def test_layer_indices_from_hypernet_config(self) -> None:
        cfg = PipelineConfig()
        base_model = MagicMock()
        tokenizer = MagicMock()
        hypernet = MagicMock()
        hypernet.config = MagicMock()
        hypernet.config.layer_indices = [3, 7, 15]
        wrapper = ModelWrapper(base_model, tokenizer, hypernet, config=cfg)
        assert wrapper._layer_indices == [3, 7, 15]

    def test_layer_indices_missing_attr_falls_back_to_empty(self) -> None:
        cfg = PipelineConfig()
        base_model = MagicMock()
        tokenizer = MagicMock()
        hypernet = MagicMock(spec=[])  # no .config
        wrapper = ModelWrapper(base_model, tokenizer, hypernet, config=cfg)
        assert wrapper._layer_indices == []

    def test_generate_adapter_passes_layer_indices(self) -> None:
        with patch(
            "rune.model.wrapper.generate_adapter_weights",
            return_value={},
        ) as mock_gen:
            wrapper = self._make_wrapper()
            wrapper.generate_adapter("traj")
            call_kwargs = mock_gen.call_args
            assert call_kwargs.kwargs["layer_indices"] == [0, 1, 2]

    def test_from_config_raises_on_empty_checkpoint(self) -> None:
        cfg = PipelineConfig(checkpoint_path="")
        with pytest.raises(ValueError, match="checkpoint_path"):
            ModelWrapper.from_config(cfg)


class TestGenerateContinuation:
    def _make_wrapper(self) -> Any:
        cfg = PipelineConfig()
        base_model = MagicMock()
        tokenizer = MagicMock()
        hypernet = MagicMock()
        hypernet.config = MagicMock()
        hypernet.config.layer_indices = [0, 1, 2]
        return ModelWrapper(base_model, tokenizer, hypernet, config=cfg)

    def test_delegates_to_inference(self) -> None:
        expected = GenerationResult(
            text="    return self.data\n", thinking="", tokens_used=10,
        )
        with patch(
            "rune.model.wrapper.inference_generate_continuation",
            new=AsyncMock(return_value=expected),
        ) as mock_gen:
            wrapper = self._make_wrapper()
            result = asyncio.run(
                wrapper.generate_continuation(
                    system_prompt="Output only Python code.",
                    user_prompt="Write a class",
                    assistant_prefix="class Node:\n    def __init__(self):\n",
                    max_tokens=512,
                )
            )
            assert result is expected
            call_kwargs = mock_gen.call_args.kwargs
            assert call_kwargs["system_prompt"] == "Output only Python code."
            assert call_kwargs["assistant_prefix"] == "class Node:\n    def __init__(self):\n"
            assert call_kwargs["max_tokens"] == 512
