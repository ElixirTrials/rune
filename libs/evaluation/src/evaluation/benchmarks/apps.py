"""APPS benchmark adapter.

Loads from codeparrot/apps (all config). Applies stratified-random
subsampling by difficulty when max_samples is set. Seed=42 by default.
Max cap: 5000 problems.

Note: The APPS dataset uses a legacy loading script incompatible with
current datasets>=2.x. Offline mode (parquet fixture) is the default
path for CI. Online mode falls back gracefully if the dataset loads.
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
    / "apps_mini.parquet"
)

_APPS_MAX_SAMPLES = 5000
_DIFFICULTY_LEVELS = ("introductory", "interview", "competition")


class APPSAdapter:
    """Benchmark adapter for the APPS dataset.

    Applies stratified-random subsampling by difficulty level.
    Hard max cap: 5000 problems.

    Attributes:
        benchmark_id: "apps".
    """

    benchmark_id: str = "apps"
    _fixture_path: Path = _DEFAULT_FIXTURE

    def load_problems(
        self,
        max_samples: int | None = None,
        seed: int = 42,
    ) -> list[Problem]:
        """Load APPS problems with stratified subsampling by difficulty.

        Args:
            max_samples: Cap applied after stratified sampling.
                Hard-capped at 5000 regardless.
            seed: Random seed for reproducibility.

        Returns:
            List of Problem instances.
        """
        rows = self._load_rows()
        cap = min(max_samples or _APPS_MAX_SAMPLES, _APPS_MAX_SAMPLES)
        if cap < len(rows):
            rows = self._stratified_sample(rows, cap, seed)
        return [self._row_to_problem(r) for r in rows]

    def score(
        self,
        problem: Problem,
        generation: str,
        timeout_s: int = 30,
    ) -> PassVerdict:
        """Score a generation against APPS input/output test cases.

        APPS provides input_output JSON with "inputs" and "outputs" lists.
        Runs the generation for each input via stdin and compares stdout.

        Note: Strict I/O equality is used; multi-valid-output problems may
        be undercounted (APPS open question #3 from plan).

        Args:
            problem: Problem instance.
            generation: Model's complete Python program.
            timeout_s: Per-test-case timeout.

        Returns:
            PassVerdict — passed only if all test cases produce correct output.
        """
        from shared.sandbox import SubprocessBackend  # deferred: INFRA-05

        backend = SubprocessBackend()
        io_data_raw = problem.metadata.get("input_output", "")
        try:
            io_data = json.loads(io_data_raw) if io_data_raw else {}
        except (json.JSONDecodeError, TypeError):
            io_data = {}

        inputs: list[str] = io_data.get("inputs", [])
        outputs: list[str] = io_data.get("outputs", [])

        if not inputs:
            return PassVerdict(
                problem_id=problem.problem_id,
                passed=False,
                generation=generation,
                error="No test cases found in problem metadata",
                timed_out=False,
            )

        for inp, expected_out in zip(inputs, outputs):
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
            if result.exit_code != 0:
                return PassVerdict(
                    problem_id=problem.problem_id,
                    passed=False,
                    generation=generation,
                    error=result.stderr or result.stdout,
                    timed_out=False,
                )
            actual = result.stdout.strip()
            if actual != str(expected_out).strip():
                return PassVerdict(
                    problem_id=problem.problem_id,
                    passed=False,
                    generation=generation,
                    error=f"Expected {expected_out!r}, got {actual!r}",
                    timed_out=False,
                )

        return PassVerdict(
            problem_id=problem.problem_id,
            passed=True,
            generation=generation,
            error=None,
            timed_out=False,
        )

    # ------------------------------------------------------------------

    def _stratified_sample(
        self,
        rows: list[dict[str, Any]],
        cap: int,
        seed: int,
    ) -> list[dict[str, Any]]:
        """Stratified-random sample by difficulty, proportional allocation.

        Args:
            rows: All problem rows.
            cap: Total desired sample size.
            seed: Random seed.

        Returns:
            Sampled rows, shuffled.
        """
        rng = random.Random(seed)
        buckets: dict[str, list[dict[str, Any]]] = {d: [] for d in _DIFFICULTY_LEVELS}
        for row in rows:
            diff = str(row.get("difficulty", "interview"))
            bucket = buckets.get(diff, buckets["interview"])
            bucket.append(row)
        n_per_bucket = cap // len(_DIFFICULTY_LEVELS)
        sampled: list[dict[str, Any]] = []
        for diff in _DIFFICULTY_LEVELS:
            bucket = buckets[diff]
            k = min(n_per_bucket, len(bucket))
            sampled.extend(rng.sample(bucket, k))
        # Fill remaining slots from all rows
        remaining = cap - len(sampled)
        if remaining > 0:
            sampled_ids = {id(r) for r in sampled}
            unsampled = [r for r in rows if id(r) not in sampled_ids]
            extra = rng.sample(unsampled, min(remaining, len(unsampled)))
            sampled.extend(extra)
        rng.shuffle(sampled)
        return sampled[:cap]

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

        ds = hf_datasets.load_dataset("codeparrot/apps", "all", split="train")
        return list(ds)

    def _load_from_fixture(self) -> list[dict[str, Any]]:
        """Load from local parquet fixture."""
        import pandas as pd

        records: list[dict[Any, Any]] = pd.read_parquet(self._fixture_path).to_dict(
            orient="records"
        )
        return records

    def _row_to_problem(self, row: dict[str, Any]) -> Problem:
        """Convert a raw APPS row to a Problem instance."""
        return Problem(
            problem_id=f"apps/{row.get('problem_id', '')}",
            prompt=str(row.get("question", "")),
            test_code="",  # APPS uses input/output pairs, not assert statements
            metadata={
                "difficulty": str(row.get("difficulty", "interview")),
                "input_output": row.get("input_output", ""),
                "solutions": row.get("solutions", ""),
            },
        )
