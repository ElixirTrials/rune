"""TDD wireframe tests for inference.completion module.

All three functions are stubs that raise NotImplementedError.
Tests assert the TDD red-phase contract.
"""

import pytest

from inference.completion import (
    batch_generate,
    generate_completion,
    generate_with_adapter,
)


def test_generate_completion_raises_not_implemented() -> None:
    """generate_completion raises NotImplementedError with function name in message."""
    with pytest.raises(NotImplementedError, match="generate_completion"):
        generate_completion("def hello():")


def test_generate_with_adapter_raises_not_implemented() -> None:
    """generate_with_adapter raises NotImplementedError with function name in message."""
    with pytest.raises(NotImplementedError, match="generate_with_adapter"):
        generate_with_adapter("def hello():", "adapter-001")


def test_batch_generate_raises_not_implemented() -> None:
    """batch_generate raises NotImplementedError with function name in message."""
    with pytest.raises(NotImplementedError, match="batch_generate"):
        batch_generate(["def foo():", "def bar():"])
