"""TDD wireframe tests for evaluation.metrics module.

All 6 evaluation functions are stubs that raise NotImplementedError.
Tests assert the TDD red-phase contract: each function raises
NotImplementedError with its name in the error message.
"""

import pytest
from evaluation.metrics import (
    calculate_pass_at_k,
    compare_adapters,
    evaluate_fitness,
    run_humaneval_subset,
    score_adapter_quality,
)
from evaluation.metrics import (
    test_generalization as _test_generalization,
)


def test_run_humaneval_subset_raises_not_implemented() -> None:
    """run_humaneval_subset raises NotImplementedError with function name."""
    with pytest.raises(NotImplementedError, match="run_humaneval_subset"):
        run_humaneval_subset("adapter-001")


def test_calculate_pass_at_k_raises_not_implemented() -> None:
    """calculate_pass_at_k raises NotImplementedError with function name."""
    with pytest.raises(NotImplementedError, match="calculate_pass_at_k"):
        calculate_pass_at_k(100, 85)


def test_score_adapter_quality_raises_not_implemented() -> None:
    """score_adapter_quality raises NotImplementedError with function name."""
    with pytest.raises(NotImplementedError, match="score_adapter_quality"):
        score_adapter_quality("adapter-001", 0.85)


def test_compare_adapters_raises_not_implemented() -> None:
    """compare_adapters raises NotImplementedError with function name."""
    with pytest.raises(NotImplementedError, match="compare_adapters"):
        compare_adapters(["adapter-001", "adapter-002"])


def test_test_generalization_raises_not_implemented() -> None:
    """test_generalization raises NotImplementedError with function name."""
    with pytest.raises(NotImplementedError, match="test_generalization"):
        _test_generalization("adapter-001")


def test_evaluate_fitness_raises_not_implemented() -> None:
    """evaluate_fitness raises NotImplementedError with function name."""
    with pytest.raises(NotImplementedError, match="evaluate_fitness"):
        evaluate_fitness("adapter-001", 0.85)
