"""LiveCodeBench benchmark adapter.

Loads from livecodebench/code_generation_lite, pinned to release_v4 split.
Test harness uses the public_tests field (input/output pairs).

HF dataset: livecodebench/code_generation_lite
Note: This dataset uses a legacy loading script. The fixture-based offline path
is the primary CI path. Online loading may fail with current datasets>=2.x.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

from evaluation.benchmarks.protocol import PassVerdict, Problem

_DEFAULT_FIXTURE = (
    Path(__file__).parent.parent.parent.parent.parent.parent
    / "tests"
    / "fixtures"
    / "livecodebench_mini.parquet"
)


class LiveCodeBenchAdapter:
    """Benchmark adapter for LiveCodeBench (release_v4).

    Attributes:
        benchmark_id: "livecodebench".
    """

    benchmark_id: str = "livecodebench"
    _fixture_path: Path = _DEFAULT_FIXTURE
    _SPLIT: str = "release_v4"

    def load_problems(
        self,
        max_samples: int | None = None,
        seed: int = 42,
    ) -> list[Problem]:
        """Load LiveCodeBench problems from release_v4.

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
        """Score a generation using LiveCodeBench public test cases.

        Runs the generation as a complete Python program, feeding each
        public_test input via stdin and comparing stdout to expected output.

        Args:
            problem: Problem instance.
            generation: Model's complete Python solution.
            timeout_s: Per-test-case timeout.

        Returns:
            PassVerdict.
        """
        from shared.sandbox import SubprocessBackend  # deferred: INFRA-05

        backend = SubprocessBackend()
        public_tests_raw = problem.metadata.get("public_tests", "[]")
        try:
            public_tests: list[dict[str, str]] = (
                json.loads(public_tests_raw)
                if isinstance(public_tests_raw, str)
                else list(public_tests_raw)
            )
        except (json.JSONDecodeError, TypeError):
            public_tests = []

        if not public_tests:
            return PassVerdict(
                problem_id=problem.problem_id,
                passed=False,
                generation=generation,
                error="No public test cases available",
                timed_out=False,
            )

        for tc in public_tests:
            inp = str(tc.get("input", ""))
            expected = str(tc.get("output", "")).strip()
            harness = (
                f"import sys\nfrom io import StringIO\n"
                f"sys.stdin = StringIO({inp!r})\n"
                f"{generation}\n"
            )
            result = backend.run(harness, timeout=timeout_s)
            if result.is_timed_out:
                return PassVerdict(
                    problem_id=problem.problem_id,
                    passed=False,
                    generation=generation,
                    error=f"Timed out on input: {inp!r}",
                    timed_out=True,
                )
            if result.exit_code != 0 or result.stdout.strip() != expected:
                return PassVerdict(
                    problem_id=problem.problem_id,
                    passed=False,
                    generation=generation,
                    error=(
                        result.stderr or f"Wrong answer: got {result.stdout.strip()!r}"
                    ),
                    timed_out=False,
                )

        return PassVerdict(
            problem_id=problem.problem_id,
            passed=True,
            generation=generation,
            error=None,
            timed_out=False,
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
        """Load from HuggingFace datasets (release_v4 config, test split)."""
        import datasets as hf_datasets  # deferred

        ds = hf_datasets.load_dataset(
            "livecodebench/code_generation_lite",
            "release_v4",
            split="test",
        )
        return list(ds)

    def _load_from_fixture(self) -> list[dict[str, Any]]:
        """Load from local parquet fixture."""
        import pandas as pd

        records: list[dict[Any, Any]] = pd.read_parquet(self._fixture_path).to_dict(
            orient="records"
        )
        return records

    def _row_to_problem(self, row: dict[str, Any]) -> Problem:
        """Convert a raw LiveCodeBench row to a Problem instance."""
        public_tests = row.get("public_tests", [])
        if not isinstance(public_tests, str):
            public_tests = json.dumps(public_tests)
        return Problem(
            problem_id=f"livecodebench/{row.get('question_id', '')}",
            prompt=str(row.get("question_content", "")),
            test_code="",  # LCB uses public_tests I/O pairs, not assert code
            metadata={
                "public_tests": public_tests,
                "difficulty": str(row.get("difficulty", "")),
                "platform": str(row.get("platform", "")),
            },
        )
