"""Tests for thinking_budget parameter across the inference provider stack."""

from __future__ import annotations

import inspect

import pytest
from inference.provider import InferenceProvider


class TestThinkingBudgetABC:
    """Verify thinking_budget exists in the provider ABC signature."""

    def test_provider_abc_has_thinking_budget_param(self) -> None:
        sig = inspect.signature(InferenceProvider.generate)
        assert "thinking_budget" in sig.parameters
        param = sig.parameters["thinking_budget"]
        assert param.default == 0

    @pytest.mark.parametrize(
        "provider_cls",
        [
            "inference.transformers_provider.TransformersProvider",
            "inference.vllm_provider.VLLMProvider",
            "inference.ollama_provider.OllamaProvider",
            "inference.llamacpp_provider.LlamaCppProvider",
        ],
    )
    def test_concrete_providers_accept_thinking_budget(self, provider_cls: str) -> None:
        module_path, cls_name = provider_cls.rsplit(".", 1)
        import importlib

        mod = importlib.import_module(module_path)
        cls = getattr(mod, cls_name)
        sig = inspect.signature(cls.generate)
        assert "thinking_budget" in sig.parameters
        assert sig.parameters["thinking_budget"].default == 0
