"""Tests for InferenceProvider ABC, GenerationResult dataclass, and UnsupportedOperationError."""

import pytest

from inference.exceptions import UnsupportedOperationError
from inference.provider import GenerationResult, InferenceProvider


class TestInferenceProviderABC:
    """Tests for InferenceProvider abstract base class."""

    def test_cannot_instantiate_directly(self) -> None:
        """Test 1: InferenceProvider cannot be instantiated directly (TypeError from ABC)."""
        with pytest.raises(TypeError):
            InferenceProvider()  # type: ignore[abstract]

    def test_concrete_subclass_missing_method_cannot_be_instantiated(self) -> None:
        """Test 2: A concrete subclass missing any abstract method cannot be instantiated."""

        class IncompleteProvider(InferenceProvider):
            async def generate(self, prompt: str, model: str, adapter_id: str | None = None, max_tokens: int = 1024) -> GenerationResult:  # type: ignore[override]
                raise NotImplementedError

            # Missing load_adapter, unload_adapter, list_adapters

        with pytest.raises(TypeError):
            IncompleteProvider()  # type: ignore[abstract]


class TestGenerationResult:
    """Tests for GenerationResult dataclass."""

    def test_can_be_constructed_with_all_fields(self) -> None:
        """Test 3: GenerationResult dataclass can be constructed with all required fields."""
        result = GenerationResult(
            text="def hello(): pass",
            model="Qwen/Qwen2.5-Coder-7B",
            adapter_id="adapter-001",
            token_count=10,
            finish_reason="stop",
        )
        assert result.text == "def hello(): pass"
        assert result.model == "Qwen/Qwen2.5-Coder-7B"
        assert result.adapter_id == "adapter-001"
        assert result.token_count == 10
        assert result.finish_reason == "stop"

    def test_adapter_id_can_be_none(self) -> None:
        """Test 4: GenerationResult.adapter_id can be None."""
        result = GenerationResult(
            text="output",
            model="some-model",
            adapter_id=None,
            token_count=5,
            finish_reason="stop",
        )
        assert result.adapter_id is None


class TestUnsupportedOperationError:
    """Tests for UnsupportedOperationError exception."""

    def test_is_subclass_of_exception(self) -> None:
        """Test 5: UnsupportedOperationError is a subclass of Exception and carries message."""
        err = UnsupportedOperationError("operation not supported")
        assert isinstance(err, Exception)
        assert str(err) == "operation not supported"

    def test_can_be_raised_and_caught(self) -> None:
        """Test that UnsupportedOperationError can be raised and caught."""
        with pytest.raises(UnsupportedOperationError, match="not supported"):
            raise UnsupportedOperationError("not supported")
