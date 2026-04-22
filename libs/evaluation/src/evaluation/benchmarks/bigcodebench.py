"""BigCodeBench benchmark adapter.

Loads from bigcode/bigcodebench (v0.1.2 split, default config).
Each problem has a complete_prompt and a test field with unittest assertions.

HF dataset: bigcode/bigcodebench
Split: v0.1.2 (verified: splits are v0.1.0_hf, v0.1.1, v0.1.2, v0.1.3, v0.1.4)
Config: default (no config argument needed)
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
    / "bigcodebench_mini.parquet"
)


class BigCodeBenchAdapter:
    """Benchmark adapter for BigCodeBench.

    Attributes:
        benchmark_id: "bigcodebench".
    """

    benchmark_id: str = "bigcodebench"
    _fixture_path: Path = _DEFAULT_FIXTURE

    def load_problems(
        self,
        max_samples: int | None = None,
        seed: int = 42,
    ) -> list[Problem]:
        """Load BigCodeBench problems.

        Args:
            max_samples: Cap on returned problems.
            seed: Subsampling seed.

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
        """Score a generation against BigCodeBench test assertions.

        Args:
            problem: Problem instance.
            generation: Model's completion of the prompt.
            timeout_s: Sandbox timeout.

        Returns:
            PassVerdict.
        """
        from shared.sandbox import SubprocessBackend  # deferred: INFRA-05

        code = f"{problem.prompt}\n{generation}\n\n{problem.test_code}\n"
        backend = SubprocessBackend()
        result = backend.run(code, timeout=timeout_s)
        passed = result.exit_code == 0 and not result.is_timed_out
        return PassVerdict(
            problem_id=problem.problem_id,
            passed=passed,
            generation=generation,
            error=(result.stderr or result.stdout or None) if not passed else None,
            timed_out=result.is_timed_out,
        )

    def _load_rows(self) -> list[dict[str, Any]]:
        """Load rows from HF or local fixture."""
        offline = os.environ.get("HF_DATASETS_OFFLINE", "0") == "1"
        if offline:
            return self._load_from_fixture()
        try:
            return self._load_from_hf()
        except Exception:
            return self._load_from_fixture()

    def _load_from_hf(self) -> list[dict[str, Any]]:
        """Load from HuggingFace datasets (v0.1.2 split, default config)."""
        import datasets as hf_datasets  # deferred

        ds = hf_datasets.load_dataset("bigcode/bigcodebench", split="v0.1.2")
        return list(ds)

    def _load_from_fixture(self) -> list[dict[str, Any]]:
        """Load from local parquet fixture."""
        import pandas as pd

        records: list[dict[str, Any]] = pd.read_parquet(self._fixture_path).to_dict(
            orient="records"
        )
        return records

    def _row_to_problem(self, row: dict[str, Any]) -> Problem:
        """Convert a raw BigCodeBench row to a Problem instance."""
        return Problem(
            problem_id=f"bigcodebench/{row.get('task_id', '')}",
            prompt=str(row.get("complete_prompt", row.get("instruct_prompt", ""))),
            test_code=str(row.get("test", "")),
            metadata={"libs": row.get("libs", [])},
        )
