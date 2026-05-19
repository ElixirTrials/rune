"""MBPP benchmark adapter.

Loads from google-research-datasets/mbpp (full config, train split).
MBPP test_list field contains assert statements; we join and execute them.

Prompt construction follows the standard MBPP evaluation protocol
(bigcode-evaluation-harness, EvalPlus): the natural-language description
is wrapped with example assertions so the model can infer the expected
function name and signature.
"""

from __future__ import annotations

import logging
import os
import random
import re
from pathlib import Path
from typing import Any

from evaluation.benchmarks.protocol import PassVerdict, Problem

logger = logging.getLogger(__name__)

_DEFAULT_FIXTURE = (
    Path(__file__).parent.parent.parent.parent.parent.parent
    / "tests"
    / "fixtures"
    / "mbpp_mini.parquet"
)

_FUNC_NAME_RE = re.compile(r"assert\s+(\w+)\s*\(")


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

        setup = problem.metadata.get("test_setup_code", "")
        setup_block = f"{setup}\n" if setup else ""
        code = f"{setup_block}{generation}\n\n{problem.test_code}\n"
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
            logger.warning(
                "HuggingFace load failed, falling back to fixture",
                exc_info=True,
            )
            return self._load_from_fixture()

    def _load_from_hf(self) -> list[dict[str, Any]]:
        """Load from HuggingFace datasets."""
        import datasets as hf_datasets  # deferred

        ds = hf_datasets.load_dataset(
            "google-research-datasets/mbpp", "sanitized", split="test"
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
        """Convert a raw MBPP row to a Problem instance."""
        test_list = row.get("test_list", [])
        if not isinstance(test_list, str):
            test_code = "\n".join(str(t) for t in test_list)
        else:
            test_code = test_list

        description = str(row.get("text", "") or row.get("prompt", ""))

        # Standard MBPP eval protocol (bigcode-evaluation-harness):
        # include assertions so the model sees the expected function name.
        test_lines = test_code.split("\n")
        test_hint = test_lines[0] if test_lines else ""
        prompt = f'"""\n{description}\n\n>>> {test_hint}\n"""\n'

        entry_point: str | None = None
        match = _FUNC_NAME_RE.search(test_code)
        if match:
            entry_point = match.group(1)

        setup_code = str(row.get("test_setup_code", "") or "")

        return Problem(
            problem_id=f"mbpp/{row.get('task_id', '')}",
            prompt=prompt,
            test_code=test_code,
            entry_point=entry_point,
            metadata={
                "source_file": row.get("source_file", ""),
                "test_setup_code": setup_code,
                "description": description,
            },
        )
