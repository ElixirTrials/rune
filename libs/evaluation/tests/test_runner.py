"""Integration test: run_benchmark() end-to-end with mock InferenceProvider.

Uses HumanEvalAdapter with 5-problem fixture and a mock provider that
returns a trivially wrong answer for each problem. Verifies that:
- run_benchmark() returns a BenchmarkResult
- pass_at_1 < 1.0 when mock returns wrong completions
- ThreadPoolExecutor fan-out works correctly
- problem_ids filter limits evaluation to specified IDs
- Unknown benchmark_id raises ValueError
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from evaluation.benchmarks.protocol import BenchmarkConfig, BenchmarkResult
from evaluation.benchmarks.runner import run_benchmark

FIXTURE = Path(__file__).parent / "fixtures" / "humaneval_mini.parquet"


class _MockProvider:
    """Minimal mock InferenceProvider that always returns a wrong answer."""

    async def generate(
        self,
        prompt: str,
        model: str,
        adapter_id: str | None = None,
        max_tokens: int = 512,
        system_prompt: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
    ) -> MagicMock:
        """Return a trivially wrong completion (deterministic, no GPU needed)."""
        from inference.provider import GenerationResult

        return GenerationResult(
            text="    return []",
            model=model,
            adapter_id=adapter_id,
        )

    async def load_adapter(self, adapter_id: str, adapter_path: str) -> None:
        """No-op."""

    async def unload_adapter(self, adapter_id: str) -> None:
        """No-op."""

    async def list_adapters(self) -> list[str]:
        """Return empty list."""
        return []


@pytest.fixture()
def mock_adapter_stack(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """AdapterStack wrapping a mock provider, pointing at HumanEval fixture."""
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    monkeypatch.setattr(
        "evaluation.benchmarks.humaneval.HumanEvalAdapter._fixture_path",
        FIXTURE,
    )
    from evaluation.benchmarks.adapter_stack import AdapterStack

    return AdapterStack(  # type: ignore[return-value]
        base_model="mock-model",
        adapter_ids=[],
        adapter_paths={},
        provider=_MockProvider(),  # type: ignore[arg-type]
    )


def test_run_benchmark_returns_benchmark_result(mock_adapter_stack: MagicMock) -> None:
    """run_benchmark() returns a BenchmarkResult instance."""
    result = run_benchmark(
        adapter_stack=mock_adapter_stack,
        benchmark_id="humaneval",
        max_samples=3,
    )
    assert isinstance(result, BenchmarkResult)
    assert result.benchmark_id == "humaneval"


def test_run_benchmark_verdicts_count(mock_adapter_stack: MagicMock) -> None:
    """run_benchmark() with max_samples=3 produces exactly 3 verdicts."""
    result = run_benchmark(
        adapter_stack=mock_adapter_stack,
        benchmark_id="humaneval",
        max_samples=3,
    )
    assert result.n_problems == 3


def test_run_benchmark_problem_ids_filter(mock_adapter_stack: MagicMock) -> None:
    """problem_ids filter limits evaluation to specified problem IDs."""
    from evaluation.benchmarks.humaneval import HumanEvalAdapter

    adapter = HumanEvalAdapter()
    problems = adapter.load_problems(max_samples=3)
    target_ids = [problems[0].problem_id]

    result = run_benchmark(
        adapter_stack=mock_adapter_stack,
        benchmark_id="humaneval",
        problem_ids=target_ids,
    )
    assert result.n_problems == 1
    assert result.verdicts[0].problem_id == target_ids[0]


def test_run_benchmark_wrong_completions_produce_failures(
    mock_adapter_stack: MagicMock,
) -> None:
    """Mock returning 'return []' causes failures — pass_at_1 < 1.0."""
    result = run_benchmark(
        adapter_stack=mock_adapter_stack,
        benchmark_id="humaneval",
        max_samples=5,
    )
    # Mock always returns "    return []" which is wrong for HumanEval
    assert result.pass_at_1 < 1.0


def test_run_benchmark_unknown_benchmark_raises(mock_adapter_stack: MagicMock) -> None:
    """run_benchmark() raises ValueError for unknown benchmark_id."""
    with pytest.raises(ValueError, match="unknown_bench"):
        run_benchmark(
            adapter_stack=mock_adapter_stack,
            benchmark_id="unknown_bench",
        )


def test_run_benchmark_config_passed_through(mock_adapter_stack: MagicMock) -> None:
    """BenchmarkConfig timeout_s and max_workers are accepted without error."""
    cfg = BenchmarkConfig(timeout_s=5, max_workers=2, max_samples=2, seed=0)
    result = run_benchmark(
        adapter_stack=mock_adapter_stack,
        benchmark_id="humaneval",
        config=cfg,
    )
    assert isinstance(result, BenchmarkResult)
    assert result.n_problems <= 2
