"""CodeContests benchmark adapter (held-out generalization set).

Loads from deepmind/code_contests. Uses public_tests and private_tests
input/output pairs for scoring. Held-out from training; used only for
generalization evaluation.

HF dataset: deepmind/code_contests
Config: default
Split: test
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
    / "codecontests_mini.parquet"
)


def _extract_io_pairs(tests: Any) -> list[tuple[str, str]]:
    """Extract (input, output) pairs from a CodeContests test dict or list.

    CodeContests stores tests as a dict with "input" and "output" keys that
    hold numpy arrays (when loaded from parquet). This function normalises
    all representations to a flat list of (input_str, output_str) pairs.

    Args:
        tests: Raw test data from the row. May be a dict, list, or string.

    Returns:
        List of (input_str, output_str) tuples.
    """
    if isinstance(tests, dict):
        inputs = list(tests.get("input", []))
        outputs = list(tests.get("output", []))
        return [(str(i), str(o)) for i, o in zip(inputs, outputs)]
    if isinstance(tests, list):
        return [(str(tc.get("input", "")), str(tc.get("output", ""))) for tc in tests]
    if isinstance(tests, str):
        try:
            parsed = json.loads(tests)
            return _extract_io_pairs(parsed)
        except (json.JSONDecodeError, ValueError):
            return []
    return []


class CodeContestsAdapter:
    """Benchmark adapter for DeepMind CodeContests (held-out generalization set).

    Attributes:
        benchmark_id: "codecontests".
    """

    benchmark_id: str = "codecontests"
    _fixture_path: Path = _DEFAULT_FIXTURE

    def load_problems(
        self,
        max_samples: int | None = None,
        seed: int = 42,
    ) -> list[Problem]:
        """Load CodeContests problems.

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
        """Score a generation against CodeContests public + private test cases.

        Runs the generation as a complete Python program, feeding each
        test input via stdin and comparing stdout to expected output.

        Args:
            problem: Problem instance.
            generation: Model's complete Python solution.
            timeout_s: Per-test-case timeout.

        Returns:
            PassVerdict — passed only if all test cases pass.
        """
        from shared.sandbox import SubprocessBackend  # deferred: INFRA-05

        backend = SubprocessBackend()
        all_pairs: list[tuple[str, str]] = []

        for key in ("public_tests", "private_tests"):
            raw = problem.metadata.get(key, [])
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw)
                except (json.JSONDecodeError, ValueError):
                    raw = []
            all_pairs.extend(_extract_io_pairs(raw))

        if not all_pairs:
            return PassVerdict(
                problem_id=problem.problem_id,
                passed=False,
                generation=generation,
                error="No test cases found",
                timed_out=False,
            )

        for inp, expected in all_pairs:
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
            if result.exit_code != 0 or result.stdout.strip() != expected.strip():
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
        """Load from HuggingFace datasets."""
        import datasets as hf_datasets  # deferred

        ds = hf_datasets.load_dataset("deepmind/code_contests", split="test")
        return list(ds)

    def _load_from_fixture(self) -> list[dict[str, Any]]:
        """Load from local parquet fixture."""
        import pandas as pd

        records: list[dict[Any, Any]] = pd.read_parquet(self._fixture_path).to_dict(
            orient="records"
        )
        return records

    def _row_to_problem(self, row: dict[str, Any]) -> Problem:
        """Convert a raw CodeContests row to a Problem instance."""
        return Problem(
            problem_id=f"codecontests/{row.get('name', '')}",
            prompt=str(row.get("description", "")),
            test_code="",  # uses I/O pairs stored in metadata
            metadata={
                "public_tests": row.get("public_tests", []),
                "private_tests": row.get("private_tests", []),
                "difficulty": str(row.get("difficulty", "")),
            },
        )
