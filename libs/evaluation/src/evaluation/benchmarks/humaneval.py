"""HumanEval benchmark adapter.

Loads problems from openai/openai_humaneval via HuggingFace datasets.
Falls back to a local parquet fixture when HF_DATASETS_OFFLINE=1.

Scoring: concatenate prompt + generation + test_code and execute via
SubprocessBackend. HumanEval test harnesses use `assert` statements and
a `check(candidate)` call pattern.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

from evaluation.benchmarks.protocol import PassVerdict, Problem

_DEFAULT_FIXTURE = (
    Path(__file__).parent.parent.parent.parent.parent.parent
    / "tests"
    / "fixtures"
    / "humaneval_mini.parquet"
)


class HumanEvalAdapter:
    """Benchmark adapter for the OpenAI HumanEval dataset.

    Attributes:
        benchmark_id: Identifier string "humaneval".
        _fixture_path: Path to the local parquet fixture used in offline mode.

    Example:
        >>> adapter = HumanEvalAdapter()
        >>> problems = adapter.load_problems(max_samples=10)
        >>> verdict = adapter.score(problems[0], "    return sorted(lst)")
    """

    benchmark_id: str = "humaneval"
    _fixture_path: Path = _DEFAULT_FIXTURE

    def load_problems(
        self,
        max_samples: int | None = None,
        seed: int = 42,
    ) -> list[Problem]:
        """Load HumanEval problems from HF or local fixture.

        Args:
            max_samples: Cap on problems returned. None returns all 164.
            seed: Random seed for subsampling when max_samples < total.

        Returns:
            List of Problem instances.
        """
        rows = self._load_rows()
        if max_samples is not None and max_samples < len(rows):
            rng = random.Random(seed)
            rows = rng.sample(rows, max_samples)
        return [self._row_to_problem(r) for r in rows]

    def score(
        self,
        problem: Problem,
        generation: str,
        timeout_s: int = 30,
    ) -> PassVerdict:
        r"""Score a generation against HumanEval test harness.

        Constructs: ``{prompt}\n{generation}\n\n{test_code}\ncheck({entry_point})``
        and executes via SubprocessBackend.

        Args:
            problem: Problem instance from load_problems().
            generation: Model completion (body of the function).
            timeout_s: Sandbox timeout in seconds.

        Returns:
            PassVerdict with passed=True iff exit_code==0 and not timed_out.
        """
        from shared.sandbox import SubprocessBackend  # deferred: INFRA-05

        entry = problem.entry_point or ""
        code = (
            f"{problem.prompt}\n"
            f"{generation}\n\n"
            f"{problem.test_code}\n"
            f"check({entry})\n"
        )
        backend = SubprocessBackend()
        result = backend.run(code, timeout=timeout_s)
        passed = result.exit_code == 0 and not result.is_timed_out
        error: str | None = None
        if not passed:
            error = result.stderr or result.stdout or "unknown error"
        return PassVerdict(
            problem_id=problem.problem_id,
            passed=passed,
            generation=generation,
            error=error,
            timed_out=result.is_timed_out,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_rows(self) -> list[dict[str, Any]]:
        """Load rows from HF dataset or local fixture.

        Checks HF_DATASETS_OFFLINE env var. If set to "1", loads from the
        local parquet fixture at _fixture_path. Otherwise attempts HF download
        and falls back to fixture on any exception.

        Returns:
            List of raw row dicts.
        """
        offline = os.environ.get("HF_DATASETS_OFFLINE", "0") == "1"
        if offline:
            return self._load_from_fixture()
        try:
            return self._load_from_hf()
        except Exception:
            return self._load_from_fixture()

    def _load_from_hf(self) -> list[dict[str, Any]]:
        """Load from HuggingFace datasets."""
        import datasets as hf_datasets  # deferred: may require network

        ds = hf_datasets.load_dataset("openai/openai_humaneval", split="test")
        return list(ds)

    def _load_from_fixture(self) -> list[dict[str, Any]]:
        """Load from local parquet fixture."""
        import pandas as pd  # noqa: PLC0415

        df = pd.read_parquet(self._fixture_path)
        records: list[dict[str, Any]] = df.to_dict(orient="records")
        return records

    def _row_to_problem(self, row: dict[str, Any]) -> Problem:
        """Convert a raw HF row to a Problem instance."""
        return Problem(
            problem_id=str(row.get("task_id", "")),
            prompt=str(row.get("prompt", "")),
            test_code=str(row.get("test", "")),
            entry_point=str(row.get("entry_point", "")) or None,
            metadata={"canonical_solution": row.get("canonical_solution", "")},
        )
