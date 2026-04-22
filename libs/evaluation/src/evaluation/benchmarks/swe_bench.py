"""SWE-Bench-Lite benchmark adapter.

load_problems() is fully implemented — loads the 300-problem Lite split
from princeton-nlp/SWE-bench_Lite.

score() raises NotImplementedError. Multi-file repo checkout, git apply,
and pytest orchestration are required for scoring and are deferred to a
follow-on plan. See docs/superpowers/plans/ for the follow-on spec.
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
    / "swe_bench_lite_mini.parquet"
)


class SWEBenchLiteAdapter:
    """Benchmark adapter for SWE-Bench-Lite (held-out generalization set).

    Attributes:
        benchmark_id: "swe_bench_lite".

    Note:
        score() is intentionally not implemented. Scoring SWE-Bench requires
        cloning the target repository at the base commit, applying the
        generated patch, and running the problem's test suite — a multi-step
        preflight that is deferred to a follow-on plan.
    """

    benchmark_id: str = "swe_bench_lite"
    _fixture_path: Path = _DEFAULT_FIXTURE

    def load_problems(
        self,
        max_samples: int | None = None,
        seed: int = 42,
    ) -> list[Problem]:
        """Load SWE-Bench-Lite problem metadata.

        Does NOT clone repositories or set up execution environments.
        Returns problem statements and metadata only.

        Args:
            max_samples: Cap on returned problems.
            seed: Subsampling seed.

        Returns:
            List of Problem instances (test_code is empty; scoring not supported).
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
        """Not implemented — requires repo checkout preflight.

        Args:
            problem: Problem instance.
            generation: Proposed patch (unified diff format).
            timeout_s: Unused.

        Raises:
            NotImplementedError: Always. Scoring SWE-Bench-Lite requires
                cloning the target repo at base_commit, applying the patch,
                and running the fail-to-pass test suite. This preflight is
                deferred to a follow-on plan.
        """
        raise NotImplementedError(
            "preflight clone/apply not yet implemented — see follow-on plan. "
            "SWEBenchLiteAdapter.load_problems() works; score() requires "
            "repo checkout, git-apply, and pytest orchestration."
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

        ds = hf_datasets.load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
        return list(ds)

    def _load_from_fixture(self) -> list[dict[str, Any]]:
        """Load from local parquet fixture."""
        import pandas as pd

        records: list[dict[str, Any]] = pd.read_parquet(self._fixture_path).to_dict(
            orient="records"
        )
        return records

    def _row_to_problem(self, row: dict[str, Any]) -> Problem:
        """Convert a raw SWE-Bench-Lite row to a Problem instance."""
        return Problem(
            problem_id=f"swe_bench_lite/{row.get('instance_id', '')}",
            prompt=str(row.get("problem_statement", "")),
            test_code="",  # scoring deferred; test_code unused
            metadata={
                "repo": str(row.get("repo", "")),
                "base_commit": str(row.get("base_commit", "")),
                "issue_url": str(row.get("issue_url", row.get("instance_id", ""))),
            },
        )
