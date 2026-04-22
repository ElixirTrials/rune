"""MBPP benchmark adapter.

Loads from google-research-datasets/mbpp (full config, train split).
MBPP test_list field contains assert statements; we join and execute them.
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
    / "mbpp_mini.parquet"
)


class MBPPAdapter:
    """Benchmark adapter for the MBPP dataset.

    Attributes:
        benchmark_id: "mbpp".
    """

    benchmark_id: str = "mbpp"
    _fixture_path: Path = _DEFAULT_FIXTURE

    def load_problems(
        self,
        max_samples: int | None = None,
        seed: int = 42,
    ) -> list[Problem]:
        """Load MBPP problems.

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
        """Score a generation against MBPP test assertions.

        Args:
            problem: Problem instance from load_problems().
            generation: Model completion (function body or full function).
            timeout_s: Sandbox timeout in seconds.

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
        """Load rows from HF or fixture."""
        offline = os.environ.get("HF_DATASETS_OFFLINE", "0") == "1"
        if offline:
            return self._load_from_fixture()
        try:
            return self._load_from_hf()
        except Exception:
            return self._load_from_fixture()

    def _load_from_hf(self) -> list[dict[str, Any]]:
        """Load from HuggingFace datasets."""
        import datasets as hf_datasets  # deferred

        ds = hf_datasets.load_dataset(
            "google-research-datasets/mbpp", "full", split="train"
        )
        return list(ds)

    def _load_from_fixture(self) -> list[dict[str, Any]]:
        """Load from local parquet fixture."""
        import pandas as pd

        records: list[dict[str, Any]] = pd.read_parquet(self._fixture_path).to_dict(
            orient="records"
        )
        return records

    def _row_to_problem(self, row: dict[str, Any]) -> Problem:
        """Convert a raw MBPP row to a Problem instance."""
        test_list = row.get("test_list", [])
        if isinstance(test_list, list):
            test_code = "\n".join(str(t) for t in test_list)
        else:
            test_code = str(test_list)
        return Problem(
            problem_id=f"mbpp/{row.get('task_id', '')}",
            prompt=str(row.get("text", "")),
            test_code=test_code,
            metadata={"source_file": row.get("source_file", "")},
        )
