"""DS-1000 benchmark adapter.

Loads from xlangai/DS-1000. Each problem has a prompt and a
reference_code solution; test harness uses code_context with exec().
The metadata.library column indicates which data-science library is tested.

HF dataset: xlangai/DS-1000
Config: default
Split: test
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
    / "ds1000_mini.parquet"
)


class DS1000Adapter:
    """Benchmark adapter for DS-1000.

    Attributes:
        benchmark_id: "ds_1000".
    """

    benchmark_id: str = "ds_1000"
    _fixture_path: Path = _DEFAULT_FIXTURE

    def load_problems(
        self,
        max_samples: int | None = None,
        seed: int = 42,
    ) -> list[Problem]:
        """Load DS-1000 problems.

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
        """Score a generation against DS-1000 test code.

        DS-1000 uses an exec-based harness via code_context. We run:
            {generation}
            {test_code}

        Args:
            problem: Problem instance.
            generation: Model completion.
            timeout_s: Sandbox timeout.

        Returns:
            PassVerdict.
        """
        from shared.sandbox import SubprocessBackend  # deferred: INFRA-05

        code = f"{generation}\n\n{problem.test_code}\n"
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
        """Load from HuggingFace datasets."""
        import datasets as hf_datasets  # deferred

        ds = hf_datasets.load_dataset("xlangai/DS-1000", split="test")
        return list(ds)

    def _load_from_fixture(self) -> list[dict[str, Any]]:
        """Load from local parquet fixture."""
        import pandas as pd

        records: list[dict[str, Any]] = pd.read_parquet(self._fixture_path).to_dict(
            orient="records"
        )
        return records

    def _row_to_problem(self, row: dict[str, Any]) -> Problem:
        """Convert a raw DS-1000 row to a Problem instance."""
        meta = row.get("metadata", {})
        if isinstance(meta, str):
            import json

            try:
                meta = json.loads(meta)
            except (json.JSONDecodeError, ValueError):
                meta = {}
        problem_id = str(meta.get("problem_id", row.get("problem_id", "")))
        library = str(meta.get("library", row.get("lib", "")))
        return Problem(
            problem_id=f"ds1000/{problem_id}",
            prompt=str(row.get("prompt", "")),
            test_code=str(row.get("reference_code", row.get("test", ""))),
            metadata={"library": library},
        )
