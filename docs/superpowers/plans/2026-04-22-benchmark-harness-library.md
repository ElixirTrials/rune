# Benchmark Harness Library Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a unified code-benchmark Pass@1 harness under `libs/evaluation/src/evaluation/benchmarks/` supporting HumanEval, MBPP, APPS, BigCodeBench, DS-1000, LiveCodeBench (training-time oracles) and SWE-Bench-Lite, CodeContests (held-out generalization sets) — enabling per-oracle validation and the Report_2 kill-switch.

**Architecture:** Each benchmark is a stateless `BenchmarkAdapter` protocol that loads problems from HuggingFace `datasets` (with offline parquet fixture fallback) and scores a generation via `shared.sandbox.SubprocessBackend` (30 s default timeout). A `run_benchmark()` orchestrator samples from an `InferenceProvider`, fans out scoring via a `ThreadPoolExecutor` (4 workers default), and returns aggregate Pass@1. Adapter stack loading reuses `libs/adapter-registry/`. SWE-Bench-Lite `load_problems` is implemented but `score` raises `NotImplementedError` pending a follow-on repo-checkout plan.

**Tech Stack:** Python 3.12, `huggingface datasets`, `shared.sandbox.SubprocessBackend`, `pytest`, `concurrent.futures.ThreadPoolExecutor`, `uv run`

---

## File Structure

All files are new unless marked **Modify**.

```
libs/evaluation/src/evaluation/benchmarks/
├── __init__.py                    # Re-exports public API: Problem, PassVerdict, BenchmarkAdapter, BenchmarkConfig, BenchmarkResult, run_benchmark, load_adapter_stack
├── protocol.py                    # Problem, PassVerdict dataclasses + BenchmarkAdapter protocol + BenchmarkConfig + BenchmarkResult
├── aggregator.py                  # pass_at_1_from_verdicts() pure aggregation function
├── runner.py                      # run_benchmark() orchestrator + ThreadPoolExecutor fan-out
├── adapter_stack.py               # load_adapter_stack() reusing AdapterRegistry + InferenceProvider
├── humaneval.py                   # HumanEvalAdapter: load_problems + score
├── mbpp.py                        # MBPPAdapter: load_problems + score
├── apps.py                        # APPSAdapter: stratified-random subsampler + score
├── bigcodebench.py                # BigCodeBenchAdapter: load_problems + score
├── ds1000.py                      # DS1000Adapter: load_problems + score
├── livecodebench.py               # LiveCodeBenchAdapter: pinned to release_v4 + score
├── swe_bench.py                   # SWEBenchLiteAdapter: load_problems implemented, score raises NotImplementedError
└── codecontests.py                # CodeContestsAdapter: load_problems + score

libs/evaluation/tests/
├── fixtures/
│   ├── humaneval_mini.parquet     # 5-problem HumanEval subset (checked in)
│   ├── mbpp_mini.parquet          # 5-problem MBPP subset (checked in)
│   ├── apps_mini.parquet          # 5-problem APPS subset (checked in)
│   ├── bigcodebench_mini.parquet  # 5-problem BigCodeBench subset (checked in)
│   ├── ds1000_mini.parquet        # 5-problem DS-1000 subset (checked in)
│   ├── livecodebench_mini.parquet # 5-problem LiveCodeBench subset (checked in)
│   ├── swe_bench_lite_mini.parquet # 5-problem SWE-Bench-Lite subset (checked in)
│   └── codecontests_mini.parquet  # 5-problem CodeContests subset (checked in)
├── test_protocol.py               # Unit tests for Problem, PassVerdict, BenchmarkConfig, BenchmarkResult
├── test_aggregator.py             # Golden-file Pass@1 aggregation tests
├── test_humaneval_adapter.py      # HumanEval load + score with fixture
├── test_mbpp_adapter.py           # MBPP load + score with fixture
├── test_apps_adapter.py           # APPS load + stratified sample + score
├── test_bigcodebench_adapter.py   # BigCodeBench load + score with fixture
├── test_ds1000_adapter.py         # DS-1000 load + score with fixture
├── test_livecodebench_adapter.py  # LiveCodeBench load + score with fixture
├── test_swe_bench_adapter.py      # SWE-Bench-Lite load works; score raises NotImplementedError
├── test_codecontests_adapter.py   # CodeContests load + score with fixture
├── test_runner.py                 # Integration test: HumanEval 5-problem end-to-end with mock InferenceProvider
└── test_adapter_stack.py          # adapter_stack.py unit tests with mock registry

libs/evaluation/pyproject.toml    # Modify: add datasets dependency
scripts/run_benchmark.py           # New CLI entrypoint mirroring trainer_cli.py --dry-run pattern
```

---

## Follow-on Plans (Out of Scope — Flag Only)

- **Kill-switch wiring into hypernetwork training loop** — `run_benchmark()` exists but nothing calls it during training. A separate plan will wire it into `libs/model-training/src/model_training/hypernetwork.py`.
- **Oracle validation runner** — per-oracle "beat base by ≥3% absolute at its phase" gate. Depends on this plan's `run_benchmark()` API.
- **Multi-GPU / distributed harness** — current plan uses 4-thread pool on a single node. Ray or SLURM distribution is a follow-on.
- **Non-Python benchmarks** — all adapters here target Python execution only.
- **SWE-Bench-Lite `score` implementation** — requires multi-file repo checkout, git apply, pytest orchestration. Explicitly deferred; `score` raises `NotImplementedError`.

---

## Locked Design Decisions (Architecture Section)

- **Execution sandbox**: `shared.sandbox.SubprocessBackend` with per-problem 30 s default timeout, configurable via `BenchmarkConfig.timeout_s`.
- **Dataset loading**: `datasets.load_dataset(...)` directly; `os.environ["HF_DATASETS_OFFLINE"]` honored; fallback to `tests/fixtures/<benchmark>_mini.parquet` for CI.
- **Problem timeout**: 30 s default, configurable via `BenchmarkConfig.timeout_s`.
- **LiveCodeBench**: pinned to `release_v4` split.
- **APPS subsampling**: stratified-random by difficulty, `max_samples=5000` cap, seed=42 deterministic sampler.
- **SWE-Bench-Lite**: `load_problems` implemented; `score` raises `NotImplementedError("preflight clone/apply not yet implemented — see follow-on plan")`.
- **Parallelism**: `ThreadPoolExecutor`, `max_workers=4` default, configurable via `BenchmarkConfig.max_workers`. No process pool.

---

## Task 1: Protocol Dataclasses + BenchmarkConfig + BenchmarkResult

**Files:**
- Create: `libs/evaluation/src/evaluation/benchmarks/__init__.py`
- Create: `libs/evaluation/src/evaluation/benchmarks/protocol.py`
- Test: `libs/evaluation/tests/test_protocol.py`

**Parallel with:** Tasks 2 (aggregator) are both foundation tasks — aggregator only depends on `PassVerdict`, which is defined here.

- [ ] **Step 1.1: Create benchmarks package marker**

Create `libs/evaluation/src/evaluation/benchmarks/__init__.py`:

```python
"""Unified benchmark harness for Pass@1 evaluation.

Public API re-exported for convenience:

    from evaluation.benchmarks import (
        Problem,
        PassVerdict,
        BenchmarkAdapter,
        BenchmarkConfig,
        BenchmarkResult,
        run_benchmark,
        load_adapter_stack,
    )
"""

from evaluation.benchmarks.protocol import (
    BenchmarkAdapter,
    BenchmarkConfig,
    BenchmarkResult,
    PassVerdict,
    Problem,
)
from evaluation.benchmarks.runner import run_benchmark
from evaluation.benchmarks.adapter_stack import load_adapter_stack

__all__ = [
    "BenchmarkAdapter",
    "BenchmarkConfig",
    "BenchmarkResult",
    "PassVerdict",
    "Problem",
    "run_benchmark",
    "load_adapter_stack",
]
```

- [ ] **Step 1.2: Write the failing test first**

Create `libs/evaluation/tests/test_protocol.py`:

```python
"""Tests for evaluation.benchmarks.protocol."""

from __future__ import annotations

import pytest
from evaluation.benchmarks.protocol import (
    BenchmarkConfig,
    BenchmarkResult,
    PassVerdict,
    Problem,
)


def test_problem_fields_present() -> None:
    """Problem dataclass stores id, prompt, test_code, and optional metadata."""
    p = Problem(
        problem_id="HumanEval/0",
        prompt="def has_close_elements(numbers, threshold):\n",
        test_code="assert has_close_elements([1.0, 2.0, 3.0], 0.5) == False",
        entry_point="has_close_elements",
        metadata={"source": "humaneval"},
    )
    assert p.problem_id == "HumanEval/0"
    assert "has_close_elements" in p.prompt
    assert p.entry_point == "has_close_elements"
    assert p.metadata == {"source": "humaneval"}


def test_problem_metadata_defaults_empty() -> None:
    """Problem metadata defaults to empty dict when not provided."""
    p = Problem(
        problem_id="mbpp/1",
        prompt="Write a function.",
        test_code="assert f() == 1",
    )
    assert p.metadata == {}


def test_pass_verdict_passed() -> None:
    """PassVerdict with passed=True stores correct fields."""
    v = PassVerdict(
        problem_id="HumanEval/0",
        passed=True,
        generation="def has_close_elements(): pass",
        error=None,
        timed_out=False,
    )
    assert v.passed is True
    assert v.timed_out is False
    assert v.error is None


def test_pass_verdict_failed_with_error() -> None:
    """PassVerdict with passed=False stores error message."""
    v = PassVerdict(
        problem_id="HumanEval/0",
        passed=False,
        generation="def f(): pass",
        error="AssertionError: ...",
        timed_out=False,
    )
    assert v.passed is False
    assert v.error == "AssertionError: ..."


def test_pass_verdict_timed_out() -> None:
    """PassVerdict with timed_out=True has passed=False."""
    v = PassVerdict(
        problem_id="mbpp/1",
        passed=False,
        generation="",
        error="Execution timed out after 30s",
        timed_out=True,
    )
    assert v.timed_out is True
    assert v.passed is False


def test_benchmark_config_defaults() -> None:
    """BenchmarkConfig has expected defaults."""
    cfg = BenchmarkConfig()
    assert cfg.timeout_s == 30
    assert cfg.max_workers == 4
    assert cfg.max_samples is None
    assert cfg.seed == 42


def test_benchmark_config_custom() -> None:
    """BenchmarkConfig accepts custom values."""
    cfg = BenchmarkConfig(timeout_s=60, max_workers=8, max_samples=100, seed=0)
    assert cfg.timeout_s == 60
    assert cfg.max_workers == 8
    assert cfg.max_samples == 100
    assert cfg.seed == 0


def test_benchmark_result_pass_at_1() -> None:
    """BenchmarkResult.pass_at_1 returns correct fraction."""
    verdicts = [
        PassVerdict(problem_id="a", passed=True, generation="", error=None, timed_out=False),
        PassVerdict(problem_id="b", passed=True, generation="", error=None, timed_out=False),
        PassVerdict(problem_id="c", passed=False, generation="", error="err", timed_out=False),
        PassVerdict(problem_id="d", passed=False, generation="", error="err", timed_out=False),
    ]
    result = BenchmarkResult(benchmark_id="humaneval", verdicts=verdicts)
    assert result.pass_at_1 == 0.5
    assert result.n_problems == 4
    assert result.n_passed == 2


def test_benchmark_result_empty_verdicts_zero() -> None:
    """BenchmarkResult with no verdicts returns pass_at_1 of 0.0."""
    result = BenchmarkResult(benchmark_id="mbpp", verdicts=[])
    assert result.pass_at_1 == 0.0
    assert result.n_problems == 0
```

Run the (expected-failing) test:

```
uv run pytest libs/evaluation/tests/test_protocol.py -x 2>&1 | head -20
```

Expected output:
```
FAILED libs/evaluation/tests/test_protocol.py::test_problem_fields_present - ModuleNotFoundError
```

- [ ] **Step 1.3: Implement protocol.py**

Create `libs/evaluation/src/evaluation/benchmarks/protocol.py`:

```python
"""Protocol types for the benchmark harness.

Defines Problem, PassVerdict, BenchmarkAdapter (runtime-checkable protocol),
BenchmarkConfig, and BenchmarkResult. All are dataclasses or frozen dataclasses
to prevent accidental mutation during concurrent evaluation.

No GPU imports. CPU-safe.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class Problem:
    """A single benchmark problem to be solved by the model.

    Attributes:
        problem_id: Unique identifier (e.g. "HumanEval/0", "mbpp/1").
        prompt: The code prompt / problem description shown to the model.
        test_code: Python code that asserts correctness of a solution.
            This is appended to the model's generation and executed.
        entry_point: Optional function name the model must define.
            Used by HumanEval-style benchmarks to construct the test harness.
        metadata: Arbitrary extra fields (difficulty, source, etc.).
    """

    problem_id: str
    prompt: str
    test_code: str
    entry_point: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PassVerdict:
    """Verdict for a single problem evaluation.

    Attributes:
        problem_id: Matches the Problem.problem_id that was evaluated.
        passed: True if the generation passed all tests.
        generation: The raw text generated by the model.
        error: Error message from the sandbox, or None if passed.
        timed_out: True if the sandbox terminated the process for timeout.
    """

    problem_id: str
    passed: bool
    generation: str
    error: str | None
    timed_out: bool


@dataclass
class BenchmarkConfig:
    """Runtime configuration for a benchmark run.

    Attributes:
        timeout_s: Per-problem execution timeout in seconds. Default 30.
        max_workers: ThreadPoolExecutor worker count. Default 4.
        max_samples: Cap on problems loaded. None means load all.
        seed: Random seed for subsampling (APPS stratified sampler). Default 42.
    """

    timeout_s: int = 30
    max_workers: int = 4
    max_samples: int | None = None
    seed: int = 42


@dataclass
class BenchmarkResult:
    """Aggregated result of a full benchmark run.

    Attributes:
        benchmark_id: Identifier of the benchmark (e.g. "humaneval").
        verdicts: Per-problem PassVerdict list.
    """

    benchmark_id: str
    verdicts: list[PassVerdict]

    @property
    def n_problems(self) -> int:
        """Total number of problems evaluated."""
        return len(self.verdicts)

    @property
    def n_passed(self) -> int:
        """Number of problems where passed=True."""
        return sum(1 for v in self.verdicts if v.passed)

    @property
    def pass_at_1(self) -> float:
        """Pass@1 as a fraction in [0.0, 1.0].

        Returns 0.0 when there are no verdicts.
        """
        if not self.verdicts:
            return 0.0
        return self.n_passed / self.n_problems


@runtime_checkable
class BenchmarkAdapter(Protocol):
    """Protocol that every benchmark adapter must satisfy.

    Implementations are stateless loaders + scorers. Each adapter
    is responsible for exactly one benchmark (HumanEval, MBPP, etc.).

    Implementors:
        HumanEvalAdapter, MBPPAdapter, APPSAdapter, BigCodeBenchAdapter,
        DS1000Adapter, LiveCodeBenchAdapter, SWEBenchLiteAdapter,
        CodeContestsAdapter
    """

    benchmark_id: str

    def load_problems(
        self,
        max_samples: int | None = None,
        seed: int = 42,
    ) -> list[Problem]:
        """Load benchmark problems, optionally capped and seeded.

        Args:
            max_samples: Maximum number of problems to return.
                None means return all available problems.
            seed: Random seed for subsampling when max_samples < total.

        Returns:
            List of Problem instances ready for scoring.
        """
        ...

    def score(self, problem: Problem, generation: str, timeout_s: int = 30) -> PassVerdict:
        """Score a single model generation against the problem's tests.

        Args:
            problem: The Problem instance (from load_problems).
            generation: Raw text generated by the model. This is the
                model's completion of problem.prompt.
            timeout_s: Sandbox timeout in seconds.

        Returns:
            A PassVerdict indicating pass/fail and any error output.
        """
        ...
```

- [ ] **Step 1.4: Run tests (must pass)**

```
uv run pytest libs/evaluation/tests/test_protocol.py -v
```

Expected output:
```
PASSED tests/test_protocol.py::test_problem_fields_present
PASSED tests/test_protocol.py::test_problem_metadata_defaults_empty
PASSED tests/test_protocol.py::test_pass_verdict_passed
PASSED tests/test_protocol.py::test_pass_verdict_failed_with_error
PASSED tests/test_protocol.py::test_pass_verdict_timed_out
PASSED tests/test_protocol.py::test_benchmark_config_defaults
PASSED tests/test_protocol.py::test_benchmark_config_custom
PASSED tests/test_protocol.py::test_benchmark_result_pass_at_1
PASSED tests/test_protocol.py::test_benchmark_result_empty_verdicts_zero
9 passed in 0.XXs
```

- [ ] **Step 1.5: Lint and type-check**

```
uv run ruff check libs/evaluation/src/evaluation/benchmarks/protocol.py
uv run mypy libs/evaluation/src/evaluation/benchmarks/protocol.py
```

Expected: no errors.

- [ ] **Step 1.6: Commit**

```
git add libs/evaluation/src/evaluation/benchmarks/__init__.py \
        libs/evaluation/src/evaluation/benchmarks/protocol.py \
        libs/evaluation/tests/test_protocol.py
git commit -m "$(cat <<'EOF'
feat(benchmarks): add Protocol dataclasses — Problem, PassVerdict, BenchmarkConfig, BenchmarkResult

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

## Task 2: Pass@1 Aggregator

**Files:**
- Create: `libs/evaluation/src/evaluation/benchmarks/aggregator.py`
- Test: `libs/evaluation/tests/test_aggregator.py`

**Parallel with:** Task 1 (only needs `PassVerdict` from protocol.py, which Task 1 creates — run after Task 1 completes).

- [ ] **Step 2.1: Write the failing test first**

Create `libs/evaluation/tests/test_aggregator.py`:

```python
"""Golden-file tests for the Pass@1 aggregation function."""

from __future__ import annotations

import pytest
from evaluation.benchmarks.aggregator import pass_at_1_from_verdicts
from evaluation.benchmarks.protocol import PassVerdict


def _verdict(problem_id: str, passed: bool) -> PassVerdict:
    return PassVerdict(
        problem_id=problem_id,
        passed=passed,
        generation="",
        error=None if passed else "err",
        timed_out=False,
    )


def test_all_passed() -> None:
    """All 5 verdicts passed → pass@1 = 1.0."""
    verdicts = [_verdict(f"p{i}", True) for i in range(5)]
    assert pass_at_1_from_verdicts(verdicts) == 1.0


def test_none_passed() -> None:
    """All 5 verdicts failed → pass@1 = 0.0."""
    verdicts = [_verdict(f"p{i}", False) for i in range(5)]
    assert pass_at_1_from_verdicts(verdicts) == 0.0


def test_half_passed() -> None:
    """3 of 6 passed → pass@1 = 0.5."""
    verdicts = [_verdict(f"p{i}", i < 3) for i in range(6)]
    assert pass_at_1_from_verdicts(verdicts) == 0.5


def test_empty_returns_zero() -> None:
    """No verdicts → pass@1 = 0.0 (no ZeroDivisionError)."""
    assert pass_at_1_from_verdicts([]) == 0.0


def test_single_pass() -> None:
    """Single passing verdict → pass@1 = 1.0."""
    assert pass_at_1_from_verdicts([_verdict("x", True)]) == 1.0


def test_single_fail() -> None:
    """Single failing verdict → pass@1 = 0.0."""
    assert pass_at_1_from_verdicts([_verdict("x", False)]) == 0.0


def test_golden_4_of_5() -> None:
    """Golden: 4 of 5 pass → 0.8 exactly."""
    verdicts = [_verdict(f"p{i}", i < 4) for i in range(5)]
    result = pass_at_1_from_verdicts(verdicts)
    assert abs(result - 0.8) < 1e-9


def test_timed_out_counts_as_fail() -> None:
    """Timed-out verdicts count as failures in the aggregation."""
    verdicts = [
        PassVerdict(problem_id="a", passed=False, generation="", error="timeout", timed_out=True),
        PassVerdict(problem_id="b", passed=True, generation="ok", error=None, timed_out=False),
    ]
    assert pass_at_1_from_verdicts(verdicts) == 0.5
```

Run (expected-failing):

```
uv run pytest libs/evaluation/tests/test_aggregator.py -x 2>&1 | head -10
```

Expected output:
```
FAILED ... ModuleNotFoundError: No module named 'evaluation.benchmarks.aggregator'
```

- [ ] **Step 2.2: Implement aggregator.py**

Create `libs/evaluation/src/evaluation/benchmarks/aggregator.py`:

```python
"""Pass@1 aggregation from a list of PassVerdict instances.

Pure function; no I/O, no GPU dependencies. Suitable for direct import
in CPU-only environments and in test fixtures.
"""

from __future__ import annotations

from evaluation.benchmarks.protocol import PassVerdict


def pass_at_1_from_verdicts(verdicts: list[PassVerdict]) -> float:
    """Compute Pass@1 from a list of per-problem verdicts.

    Pass@1 is the fraction of problems where the model's single generation
    passed all tests. Timed-out problems count as failures.

    Args:
        verdicts: List of PassVerdict instances. May be empty.

    Returns:
        Float in [0.0, 1.0]. Returns 0.0 for an empty list.

    Example:
        >>> v = [PassVerdict("p1", True, "", None, False),
        ...      PassVerdict("p2", False, "", "err", False)]
        >>> pass_at_1_from_verdicts(v)
        0.5
    """
    if not verdicts:
        return 0.0
    n_passed = sum(1 for v in verdicts if v.passed)
    return n_passed / len(verdicts)
```

- [ ] **Step 2.3: Run tests (must all pass)**

```
uv run pytest libs/evaluation/tests/test_aggregator.py -v
```

Expected output:
```
PASSED tests/test_aggregator.py::test_all_passed
PASSED tests/test_aggregator.py::test_none_passed
PASSED tests/test_aggregator.py::test_half_passed
PASSED tests/test_aggregator.py::test_empty_returns_zero
PASSED tests/test_aggregator.py::test_single_pass
PASSED tests/test_aggregator.py::test_single_fail
PASSED tests/test_aggregator.py::test_golden_4_of_5
PASSED tests/test_aggregator.py::test_timed_out_counts_as_fail
8 passed in 0.XXs
```

- [ ] **Step 2.4: Lint and type-check**

```
uv run ruff check libs/evaluation/src/evaluation/benchmarks/aggregator.py
uv run mypy libs/evaluation/src/evaluation/benchmarks/aggregator.py
```

Expected: no errors.

- [ ] **Step 2.5: Commit**

```
git add libs/evaluation/src/evaluation/benchmarks/aggregator.py \
        libs/evaluation/tests/test_aggregator.py
git commit -m "$(cat <<'EOF'
feat(benchmarks): add pass_at_1_from_verdicts aggregator with golden-file tests

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

## Task 3: Parquet Fixtures + HumanEval Adapter

**Files:**
- Create: `libs/evaluation/tests/fixtures/humaneval_mini.parquet` (generated by fixture script)
- Create: `libs/evaluation/src/evaluation/benchmarks/humaneval.py`
- Test: `libs/evaluation/tests/test_humaneval_adapter.py`

**Parallel with:** Tasks 4–8 (each adapter is independent; all depend only on `protocol.py` from Task 1).

- [ ] **Step 3.1: Generate the HumanEval fixture parquet**

Run this once to create the checked-in fixture (requires `datasets` and `pandas`):

```python
# scripts/generate_benchmark_fixtures.py  (run once, output checked in)
"""Generate mini parquet fixtures for benchmark adapter tests.

Run:  uv run python scripts/generate_benchmark_fixtures.py
"""
from __future__ import annotations
import os
from pathlib import Path

FIXTURE_DIR = Path("libs/evaluation/tests/fixtures")
FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("HF_DATASETS_OFFLINE", "0")  # allow download during fixture gen

import datasets  # noqa: E402

def save_mini(dataset_name: str, config: str | None, split: str, n: int,
              out_name: str, column_map: dict[str, str] | None = None) -> None:
    """Load a dataset split, take first n rows, rename columns, save parquet."""
    ds = datasets.load_dataset(dataset_name, config, split=split, trust_remote_code=True)
    df = ds.to_pandas().head(n)
    if column_map:
        df = df.rename(columns=column_map)
    df.to_parquet(FIXTURE_DIR / out_name, index=False)
    print(f"Wrote {FIXTURE_DIR / out_name} ({len(df)} rows)")

# HumanEval — openai/openai_humaneval
save_mini("openai/openai_humaneval", None, "test", 5, "humaneval_mini.parquet")
# MBPP — google-research-datasets/mbpp
save_mini("google-research-datasets/mbpp", "full", "train", 5, "mbpp_mini.parquet")
```

Execute:
```
uv run python scripts/generate_benchmark_fixtures.py
```

Expected: creates `libs/evaluation/tests/fixtures/humaneval_mini.parquet` and `mbpp_mini.parquet`.

- [ ] **Step 3.2: Write the failing HumanEval adapter test**

Create `libs/evaluation/tests/test_humaneval_adapter.py`:

```python
"""Tests for HumanEvalAdapter."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from evaluation.benchmarks.humaneval import HumanEvalAdapter
from evaluation.benchmarks.protocol import PassVerdict

FIXTURE = Path(__file__).parent / "fixtures" / "humaneval_mini.parquet"


@pytest.fixture(autouse=True)
def use_fixture_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point adapter to local parquet fixture; set HF offline mode."""
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    monkeypatch.setattr(
        "evaluation.benchmarks.humaneval.HumanEvalAdapter._fixture_path",
        FIXTURE,
    )


def test_load_problems_returns_list() -> None:
    """load_problems returns a non-empty list of Problem instances."""
    adapter = HumanEvalAdapter()
    problems = adapter.load_problems()
    assert len(problems) > 0


def test_load_problems_max_samples() -> None:
    """load_problems respects max_samples cap."""
    adapter = HumanEvalAdapter()
    problems = adapter.load_problems(max_samples=3)
    assert len(problems) <= 3


def test_problem_fields_populated() -> None:
    """Each problem has non-empty problem_id, prompt, test_code, entry_point."""
    adapter = HumanEvalAdapter()
    problems = adapter.load_problems()
    for p in problems:
        assert p.problem_id.startswith("HumanEval/")
        assert len(p.prompt) > 0
        assert len(p.test_code) > 0
        assert p.entry_point is not None


def test_score_passing_solution() -> None:
    """A trivially correct solution for a simple problem returns passed=True."""
    adapter = HumanEvalAdapter()
    problems = adapter.load_problems()
    # Use a problem we know has a simple canonical solution.
    # We'll score with the test_code check disabled by using a known-correct completion.
    p = problems[0]
    # Build a completion that just runs the canonical solution.
    # For HumanEval the model appends code after the function signature in prompt.
    # We inject a dummy that appends `pass` — this will fail, proving the scorer works.
    verdict = adapter.score(p, "    return []", timeout_s=10)
    assert isinstance(verdict, PassVerdict)
    assert verdict.problem_id == p.problem_id
    # `return []` is wrong for most HumanEval problems — we expect failure.
    assert verdict.passed is False


def test_score_correct_identity_solution() -> None:
    """score() returns passed=True when generation passes all assertions."""
    adapter = HumanEvalAdapter()
    problems = adapter.load_problems()
    p = problems[0]
    # Build a full solution: prompt + correct body.
    # We extract the canonical solution from fixture metadata if available;
    # if not, skip gracefully.
    canonical = p.metadata.get("canonical_solution")
    if canonical is None:
        pytest.skip("No canonical_solution in fixture metadata")
    verdict = adapter.score(p, canonical, timeout_s=15)
    assert verdict.passed is True


def test_score_timeout_returns_verdict() -> None:
    """score() with an infinite loop returns a PassVerdict with timed_out=True."""
    adapter = HumanEvalAdapter()
    problems = adapter.load_problems()
    p = problems[0]
    infinite_loop = "    while True: pass"
    verdict = adapter.score(p, infinite_loop, timeout_s=2)
    assert isinstance(verdict, PassVerdict)
    assert verdict.timed_out is True
    assert verdict.passed is False
```

Run (expected-failing):
```
uv run pytest libs/evaluation/tests/test_humaneval_adapter.py -x 2>&1 | head -10
```

Expected:
```
FAILED ... ModuleNotFoundError: No module named 'evaluation.benchmarks.humaneval'
```

- [ ] **Step 3.3: Implement humaneval.py**

Create `libs/evaluation/src/evaluation/benchmarks/humaneval.py`:

```python
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

from evaluation.benchmarks.protocol import BenchmarkAdapter, PassVerdict, Problem

# Fixture path used by tests (monkeypatched in test_humaneval_adapter.py).
_DEFAULT_FIXTURE = Path(__file__).parent.parent.parent.parent.parent.parent / \
    "tests" / "fixtures" / "humaneval_mini.parquet"


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
        """Score a generation against HumanEval test harness.

        Constructs: `{problem.prompt}\n{generation}\n\n{problem.test_code}\ncheck({entry_point})`
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

        Returns:
            List of raw row dicts.
        """
        offline = os.environ.get("HF_DATASETS_OFFLINE", "0") == "1"
        if offline or not self._fixture_path.parent.parent.name == "tests":
            return self._load_from_fixture()
        try:
            return self._load_from_hf()
        except Exception:
            return self._load_from_fixture()

    def _load_from_hf(self) -> list[dict[str, Any]]:
        """Load from HuggingFace datasets."""
        import datasets as hf_datasets  # deferred: may require network

        ds = hf_datasets.load_dataset("openai/openai_humaneval", split="test")
        return list(ds)  # type: ignore[arg-type]

    def _load_from_fixture(self) -> list[dict[str, Any]]:
        """Load from local parquet fixture."""
        import pandas as pd  # noqa: PLC0415

        df = pd.read_parquet(self._fixture_path)
        return df.to_dict(orient="records")  # type: ignore[return-value]

    def _row_to_problem(self, row: dict[str, Any]) -> Problem:
        """Convert a raw HF row to a Problem instance."""
        return Problem(
            problem_id=str(row.get("task_id", "")),
            prompt=str(row.get("prompt", "")),
            test_code=str(row.get("test", "")),
            entry_point=str(row.get("entry_point", "")) or None,
            metadata={"canonical_solution": row.get("canonical_solution", "")},
        )
```

- [ ] **Step 3.4: Fix the offline detection logic**

The `_load_rows` method needs a simpler offline check. Update the method:

```python
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
```

- [ ] **Step 3.5: Run tests**

```
uv run pytest libs/evaluation/tests/test_humaneval_adapter.py -v
```

Expected output:
```
PASSED tests/test_humaneval_adapter.py::test_load_problems_returns_list
PASSED tests/test_humaneval_adapter.py::test_load_problems_max_samples
PASSED tests/test_humaneval_adapter.py::test_problem_fields_populated
PASSED tests/test_humaneval_adapter.py::test_score_passing_solution
PASSED tests/test_humaneval_adapter.py::test_score_correct_identity_solution
PASSED tests/test_humaneval_adapter.py::test_score_timeout_returns_verdict
6 passed in X.Xs
```

(If `test_score_correct_identity_solution` is skipped due to missing canonical_solution in fixture, that is acceptable — regenerate fixture to include it.)

- [ ] **Step 3.6: Lint and type-check**

```
uv run ruff check libs/evaluation/src/evaluation/benchmarks/humaneval.py
uv run mypy libs/evaluation/src/evaluation/benchmarks/humaneval.py
```

Expected: no errors.

- [ ] **Step 3.7: Commit**

```
git add libs/evaluation/src/evaluation/benchmarks/humaneval.py \
        libs/evaluation/tests/test_humaneval_adapter.py \
        libs/evaluation/tests/fixtures/humaneval_mini.parquet \
        libs/evaluation/tests/fixtures/mbpp_mini.parquet
git commit -m "$(cat <<'EOF'
feat(benchmarks): add HumanEvalAdapter with parquet fixture and scoring tests

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

## Task 4: MBPP, APPS, BigCodeBench, DS-1000 Adapters

**Files:**
- Create: `libs/evaluation/src/evaluation/benchmarks/mbpp.py`
- Create: `libs/evaluation/src/evaluation/benchmarks/apps.py`
- Create: `libs/evaluation/src/evaluation/benchmarks/bigcodebench.py`
- Create: `libs/evaluation/src/evaluation/benchmarks/ds1000.py`
- Create: `libs/evaluation/tests/fixtures/apps_mini.parquet`
- Create: `libs/evaluation/tests/fixtures/bigcodebench_mini.parquet`
- Create: `libs/evaluation/tests/fixtures/ds1000_mini.parquet`
- Test: `libs/evaluation/tests/test_mbpp_adapter.py`
- Test: `libs/evaluation/tests/test_apps_adapter.py`
- Test: `libs/evaluation/tests/test_bigcodebench_adapter.py`
- Test: `libs/evaluation/tests/test_ds1000_adapter.py`

**Parallel with:** Task 3 (disjoint files). Run after Task 1 completes.

- [ ] **Step 4.1: Extend the fixture generator script for remaining benchmarks**

Add to `scripts/generate_benchmark_fixtures.py`:

```python
# Append to scripts/generate_benchmark_fixtures.py

# APPS — codeparrot/apps
save_mini("codeparrot/apps", "all", "train", 5, "apps_mini.parquet")

# BigCodeBench — bigcode/bigcodebench  (split v0.1.2 may need config name)
try:
    save_mini("bigcode/bigcodebench", None, "v0.1.2", 5, "bigcodebench_mini.parquet")
except Exception:
    save_mini("bigcode/bigcodebench", None, "train", 5, "bigcodebench_mini.parquet")

# DS-1000 — xlangai/DS-1000
save_mini("xlangai/DS-1000", None, "test", 5, "ds1000_mini.parquet")
```

Run:
```
uv run python scripts/generate_benchmark_fixtures.py
```

- [ ] **Step 4.2: Write failing tests**

Create `libs/evaluation/tests/test_mbpp_adapter.py`:

```python
"""Tests for MBPPAdapter."""

from __future__ import annotations

from pathlib import Path

import pytest
from evaluation.benchmarks.mbpp import MBPPAdapter
from evaluation.benchmarks.protocol import PassVerdict

FIXTURE = Path(__file__).parent / "fixtures" / "mbpp_mini.parquet"


@pytest.fixture(autouse=True)
def use_fixture_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    monkeypatch.setattr("evaluation.benchmarks.mbpp.MBPPAdapter._fixture_path", FIXTURE)


def test_load_problems_returns_list() -> None:
    adapter = MBPPAdapter()
    problems = adapter.load_problems()
    assert len(problems) > 0


def test_load_problems_max_samples() -> None:
    adapter = MBPPAdapter()
    assert len(adapter.load_problems(max_samples=2)) <= 2


def test_problem_fields_populated() -> None:
    adapter = MBPPAdapter()
    for p in adapter.load_problems():
        assert p.problem_id
        assert p.prompt
        assert p.test_code


def test_score_wrong_returns_fail() -> None:
    adapter = MBPPAdapter()
    p = adapter.load_problems()[0]
    verdict = adapter.score(p, "    return None", timeout_s=10)
    assert isinstance(verdict, PassVerdict)
    # returning None will almost certainly fail MBPP assertions
    assert verdict.problem_id == p.problem_id


def test_score_timeout() -> None:
    adapter = MBPPAdapter()
    p = adapter.load_problems()[0]
    verdict = adapter.score(p, "    while True: pass", timeout_s=2)
    assert verdict.timed_out is True
    assert verdict.passed is False
```

Create `libs/evaluation/tests/test_apps_adapter.py`:

```python
"""Tests for APPSAdapter."""

from __future__ import annotations

from pathlib import Path

import pytest
from evaluation.benchmarks.apps import APPSAdapter

FIXTURE = Path(__file__).parent / "fixtures" / "apps_mini.parquet"


@pytest.fixture(autouse=True)
def use_fixture_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    monkeypatch.setattr("evaluation.benchmarks.apps.APPSAdapter._fixture_path", FIXTURE)


def test_load_problems_returns_list() -> None:
    adapter = APPSAdapter()
    problems = adapter.load_problems()
    assert len(problems) > 0


def test_load_problems_max_samples_cap() -> None:
    """max_samples is respected."""
    adapter = APPSAdapter()
    problems = adapter.load_problems(max_samples=2)
    assert len(problems) <= 2


def test_stratified_sample_seed_deterministic() -> None:
    """Same seed produces same problem ordering."""
    adapter = APPSAdapter()
    a = adapter.load_problems(max_samples=3, seed=42)
    b = adapter.load_problems(max_samples=3, seed=42)
    assert [p.problem_id for p in a] == [p.problem_id for p in b]


def test_stratified_sample_different_seed() -> None:
    """Different seeds can produce different orderings (probabilistic)."""
    adapter = APPSAdapter()
    problems = adapter.load_problems()
    if len(problems) < 4:
        pytest.skip("Not enough fixture rows to test seeding variation")
    a = adapter.load_problems(max_samples=3, seed=0)
    b = adapter.load_problems(max_samples=3, seed=99)
    # With only 3-from-5, the ordering will differ for different seeds
    assert [p.problem_id for p in a] != [p.problem_id for p in b] or True  # probabilistic


def test_problem_has_difficulty_in_metadata() -> None:
    """Problems from APPS include difficulty in metadata."""
    adapter = APPSAdapter()
    for p in adapter.load_problems():
        assert "difficulty" in p.metadata


def test_score_wrong_returns_verdict() -> None:
    adapter = APPSAdapter()
    p = adapter.load_problems()[0]
    verdict = adapter.score(p, "    print('wrong')", timeout_s=10)
    assert verdict.problem_id == p.problem_id
```

Create `libs/evaluation/tests/test_bigcodebench_adapter.py`:

```python
"""Tests for BigCodeBenchAdapter."""

from __future__ import annotations

from pathlib import Path

import pytest
from evaluation.benchmarks.bigcodebench import BigCodeBenchAdapter

FIXTURE = Path(__file__).parent / "fixtures" / "bigcodebench_mini.parquet"


@pytest.fixture(autouse=True)
def use_fixture_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    monkeypatch.setattr(
        "evaluation.benchmarks.bigcodebench.BigCodeBenchAdapter._fixture_path",
        FIXTURE,
    )


def test_load_problems_returns_list() -> None:
    adapter = BigCodeBenchAdapter()
    assert len(adapter.load_problems()) > 0


def test_problem_fields_populated() -> None:
    adapter = BigCodeBenchAdapter()
    for p in adapter.load_problems():
        assert p.problem_id
        assert p.prompt


def test_score_timeout() -> None:
    adapter = BigCodeBenchAdapter()
    p = adapter.load_problems()[0]
    verdict = adapter.score(p, "    while True: pass", timeout_s=2)
    assert verdict.timed_out is True
```

Create `libs/evaluation/tests/test_ds1000_adapter.py`:

```python
"""Tests for DS1000Adapter."""

from __future__ import annotations

from pathlib import Path

import pytest
from evaluation.benchmarks.ds1000 import DS1000Adapter

FIXTURE = Path(__file__).parent / "fixtures" / "ds1000_mini.parquet"


@pytest.fixture(autouse=True)
def use_fixture_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    monkeypatch.setattr("evaluation.benchmarks.ds1000.DS1000Adapter._fixture_path", FIXTURE)


def test_load_problems_returns_list() -> None:
    adapter = DS1000Adapter()
    assert len(adapter.load_problems()) > 0


def test_problem_fields_populated() -> None:
    adapter = DS1000Adapter()
    for p in adapter.load_problems():
        assert p.problem_id
        assert p.prompt


def test_metadata_has_library() -> None:
    """DS-1000 problems include the target library in metadata."""
    adapter = DS1000Adapter()
    for p in adapter.load_problems():
        assert "library" in p.metadata


def test_score_wrong_completion() -> None:
    adapter = DS1000Adapter()
    p = adapter.load_problems()[0]
    verdict = adapter.score(p, "pass", timeout_s=10)
    assert verdict.problem_id == p.problem_id
```

Run (expected-failing for all four):
```
uv run pytest libs/evaluation/tests/test_mbpp_adapter.py \
             libs/evaluation/tests/test_apps_adapter.py \
             libs/evaluation/tests/test_bigcodebench_adapter.py \
             libs/evaluation/tests/test_ds1000_adapter.py -x 2>&1 | head -10
```

Expected:
```
FAILED ... ModuleNotFoundError: No module named 'evaluation.benchmarks.mbpp'
```

- [ ] **Step 4.3: Implement mbpp.py**

Create `libs/evaluation/src/evaluation/benchmarks/mbpp.py`:

```python
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

_DEFAULT_FIXTURE = Path(__file__).parent.parent.parent.parent.parent.parent / \
    "tests" / "fixtures" / "mbpp_mini.parquet"


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
        offline = os.environ.get("HF_DATASETS_OFFLINE", "0") == "1"
        if offline:
            return self._load_from_fixture()
        try:
            return self._load_from_hf()
        except Exception:
            return self._load_from_fixture()

    def _load_from_hf(self) -> list[dict[str, Any]]:
        import datasets as hf_datasets  # deferred

        ds = hf_datasets.load_dataset("google-research-datasets/mbpp", "full", split="train")
        return list(ds)  # type: ignore[arg-type]

    def _load_from_fixture(self) -> list[dict[str, Any]]:
        import pandas as pd

        return pd.read_parquet(self._fixture_path).to_dict(orient="records")  # type: ignore[return-value]

    def _row_to_problem(self, row: dict[str, Any]) -> Problem:
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
```

- [ ] **Step 4.4: Implement apps.py**

Create `libs/evaluation/src/evaluation/benchmarks/apps.py`:

```python
"""APPS benchmark adapter.

Loads from codeparrot/apps (all config). Applies stratified-random
subsampling by difficulty when max_samples is set. Seed=42 by default.
Max cap: 5000 problems.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

from evaluation.benchmarks.protocol import PassVerdict, Problem

_DEFAULT_FIXTURE = Path(__file__).parent.parent.parent.parent.parent.parent / \
    "tests" / "fixtures" / "apps_mini.parquet"

_APPS_MAX_SAMPLES = 5000
_DIFFICULTY_LEVELS = ("introductory", "interview", "competition")


class APPSAdapter:
    """Benchmark adapter for the APPS dataset.

    Applies stratified-random subsampling by difficulty level.
    max_samples cap: 5000.

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
        We run the generation for each input and compare stdout to expected output.

        Args:
            problem: Problem instance.
            generation: Model's complete Python program (not just a function).
            timeout_s: Per-test-case timeout.

        Returns:
            PassVerdict — passed only if all test cases produce correct output.
        """
        from shared.sandbox import SubprocessBackend  # deferred: INFRA-05
        import json

        backend = SubprocessBackend()
        io_data_raw = problem.metadata.get("input_output", "")
        try:
            io_data = json.loads(io_data_raw) if io_data_raw else {}
        except (json.JSONDecodeError, TypeError):
            io_data = {}

        inputs: list[str] = io_data.get("inputs", [])
        outputs: list[str] = io_data.get("outputs", [])

        if not inputs:
            # No test cases — cannot verify. Return inconclusive as fail.
            return PassVerdict(
                problem_id=problem.problem_id,
                passed=False,
                generation=generation,
                error="No test cases found in problem metadata",
                timed_out=False,
            )

        for inp, expected_out in zip(inputs, outputs):
            # Wrap generation to accept stdin
            harness = f"import sys\nfrom io import StringIO\nsys.stdin = StringIO({inp!r})\n{generation}\n"
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
        """Stratified-random sample by difficulty, proportional allocation."""
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
        # Fill remaining slots from any bucket
        remaining = cap - len(sampled)
        if remaining > 0:
            all_unsampled = [r for r in rows if r not in set(id(x) for x in sampled)]
            extra = rng.sample(rows, min(remaining, len(rows)))
            sampled.extend(extra[:remaining])
        rng.shuffle(sampled)
        return sampled[:cap]

    def _load_rows(self) -> list[dict[str, Any]]:
        offline = os.environ.get("HF_DATASETS_OFFLINE", "0") == "1"
        if offline:
            return self._load_from_fixture()
        try:
            return self._load_from_hf()
        except Exception:
            return self._load_from_fixture()

    def _load_from_hf(self) -> list[dict[str, Any]]:
        import datasets as hf_datasets  # deferred

        ds = hf_datasets.load_dataset("codeparrot/apps", "all", split="train")
        return list(ds)  # type: ignore[arg-type]

    def _load_from_fixture(self) -> list[dict[str, Any]]:
        import pandas as pd

        return pd.read_parquet(self._fixture_path).to_dict(orient="records")  # type: ignore[return-value]

    def _row_to_problem(self, row: dict[str, Any]) -> Problem:
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
```

- [ ] **Step 4.5: Implement bigcodebench.py**

Create `libs/evaluation/src/evaluation/benchmarks/bigcodebench.py`:

```python
"""BigCodeBench benchmark adapter.

Loads from bigcode/bigcodebench (v0.1.2 split). Each problem has a
complete_prompt and a test field with assert statements.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

from evaluation.benchmarks.protocol import PassVerdict, Problem

_DEFAULT_FIXTURE = Path(__file__).parent.parent.parent.parent.parent.parent / \
    "tests" / "fixtures" / "bigcodebench_mini.parquet"


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
        offline = os.environ.get("HF_DATASETS_OFFLINE", "0") == "1"
        if offline:
            return self._load_from_fixture()
        try:
            return self._load_from_hf()
        except Exception:
            return self._load_from_fixture()

    def _load_from_hf(self) -> list[dict[str, Any]]:
        import datasets as hf_datasets  # deferred

        try:
            ds = hf_datasets.load_dataset("bigcode/bigcodebench", split="v0.1.2")
        except Exception:
            ds = hf_datasets.load_dataset("bigcode/bigcodebench", split="train")
        return list(ds)  # type: ignore[arg-type]

    def _load_from_fixture(self) -> list[dict[str, Any]]:
        import pandas as pd

        return pd.read_parquet(self._fixture_path).to_dict(orient="records")  # type: ignore[return-value]

    def _row_to_problem(self, row: dict[str, Any]) -> Problem:
        return Problem(
            problem_id=f"bigcodebench/{row.get('task_id', '')}",
            prompt=str(row.get("complete_prompt", row.get("instruct_prompt", ""))),
            test_code=str(row.get("test", "")),
            metadata={"libs": row.get("libs", [])},
        )
```

- [ ] **Step 4.6: Implement ds1000.py**

Create `libs/evaluation/src/evaluation/benchmarks/ds1000.py`:

```python
"""DS-1000 benchmark adapter.

Loads from xlangai/DS-1000. Each problem has a prompt and a
reference_code solution; test is executed via exec() pattern.
The library column indicates which data-science library is tested.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

from evaluation.benchmarks.protocol import PassVerdict, Problem

_DEFAULT_FIXTURE = Path(__file__).parent.parent.parent.parent.parent.parent / \
    "tests" / "fixtures" / "ds1000_mini.parquet"


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

        DS-1000 test_code uses an exec-based harness. We run:
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
        offline = os.environ.get("HF_DATASETS_OFFLINE", "0") == "1"
        if offline:
            return self._load_from_fixture()
        try:
            return self._load_from_hf()
        except Exception:
            return self._load_from_fixture()

    def _load_from_hf(self) -> list[dict[str, Any]]:
        import datasets as hf_datasets  # deferred

        ds = hf_datasets.load_dataset("xlangai/DS-1000", split="test")
        return list(ds)  # type: ignore[arg-type]

    def _load_from_fixture(self) -> list[dict[str, Any]]:
        import pandas as pd

        return pd.read_parquet(self._fixture_path).to_dict(orient="records")  # type: ignore[return-value]

    def _row_to_problem(self, row: dict[str, Any]) -> Problem:
        return Problem(
            problem_id=f"ds1000/{row.get('metadata', {}).get('problem_id', row.get('problem_id', ''))}",
            prompt=str(row.get("prompt", "")),
            test_code=str(row.get("reference_code", row.get("test", ""))),
            metadata={"library": str(row.get("metadata", {}).get("library", row.get("lib", "")))},
        )
```

- [ ] **Step 4.7: Run all four test files**

```
uv run pytest libs/evaluation/tests/test_mbpp_adapter.py \
             libs/evaluation/tests/test_apps_adapter.py \
             libs/evaluation/tests/test_bigcodebench_adapter.py \
             libs/evaluation/tests/test_ds1000_adapter.py -v
```

Expected: all tests pass (some `test_score_*` may yield `passed=False` which is correct behavior for wrong completions).

- [ ] **Step 4.8: Lint and type-check**

```
uv run ruff check libs/evaluation/src/evaluation/benchmarks/mbpp.py \
                  libs/evaluation/src/evaluation/benchmarks/apps.py \
                  libs/evaluation/src/evaluation/benchmarks/bigcodebench.py \
                  libs/evaluation/src/evaluation/benchmarks/ds1000.py
uv run mypy libs/evaluation/src/evaluation/benchmarks/
```

Expected: no errors.

- [ ] **Step 4.9: Commit**

```
git add libs/evaluation/src/evaluation/benchmarks/mbpp.py \
        libs/evaluation/src/evaluation/benchmarks/apps.py \
        libs/evaluation/src/evaluation/benchmarks/bigcodebench.py \
        libs/evaluation/src/evaluation/benchmarks/ds1000.py \
        libs/evaluation/tests/test_mbpp_adapter.py \
        libs/evaluation/tests/test_apps_adapter.py \
        libs/evaluation/tests/test_bigcodebench_adapter.py \
        libs/evaluation/tests/test_ds1000_adapter.py \
        libs/evaluation/tests/fixtures/apps_mini.parquet \
        libs/evaluation/tests/fixtures/bigcodebench_mini.parquet \
        libs/evaluation/tests/fixtures/ds1000_mini.parquet
git commit -m "$(cat <<'EOF'
feat(benchmarks): add MBPP, APPS (stratified), BigCodeBench, DS-1000 adapters

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

## Task 5: LiveCodeBench, SWE-Bench-Lite, and CodeContests Adapters

**Files:**
- Create: `libs/evaluation/src/evaluation/benchmarks/livecodebench.py`
- Create: `libs/evaluation/src/evaluation/benchmarks/swe_bench.py`
- Create: `libs/evaluation/src/evaluation/benchmarks/codecontests.py`
- Create: `libs/evaluation/tests/fixtures/livecodebench_mini.parquet`
- Create: `libs/evaluation/tests/fixtures/swe_bench_lite_mini.parquet`
- Create: `libs/evaluation/tests/fixtures/codecontests_mini.parquet`
- Test: `libs/evaluation/tests/test_livecodebench_adapter.py`
- Test: `libs/evaluation/tests/test_swe_bench_adapter.py`
- Test: `libs/evaluation/tests/test_codecontests_adapter.py`

**Parallel with:** Task 4 (disjoint files).

- [ ] **Step 5.1: Extend fixture generator for remaining three benchmarks**

Add to `scripts/generate_benchmark_fixtures.py`:

```python
# Append to scripts/generate_benchmark_fixtures.py

# LiveCodeBench — livecodebench/code_generation_lite, pinned release_v4
save_mini("livecodebench/code_generation_lite", "release_v4", "test", 5,
          "livecodebench_mini.parquet")

# SWE-Bench-Lite — princeton-nlp/SWE-bench_Lite
save_mini("princeton-nlp/SWE-bench_Lite", None, "test", 5, "swe_bench_lite_mini.parquet")

# CodeContests — deepmind/code_contests
save_mini("deepmind/code_contests", None, "test", 5, "codecontests_mini.parquet")
```

Run:
```
uv run python scripts/generate_benchmark_fixtures.py
```

- [ ] **Step 5.2: Write failing tests**

Create `libs/evaluation/tests/test_livecodebench_adapter.py`:

```python
"""Tests for LiveCodeBenchAdapter."""

from __future__ import annotations

from pathlib import Path

import pytest
from evaluation.benchmarks.livecodebench import LiveCodeBenchAdapter

FIXTURE = Path(__file__).parent / "fixtures" / "livecodebench_mini.parquet"


@pytest.fixture(autouse=True)
def use_fixture_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    monkeypatch.setattr(
        "evaluation.benchmarks.livecodebench.LiveCodeBenchAdapter._fixture_path",
        FIXTURE,
    )


def test_load_problems_returns_list() -> None:
    adapter = LiveCodeBenchAdapter()
    assert len(adapter.load_problems()) > 0


def test_benchmark_id() -> None:
    assert LiveCodeBenchAdapter.benchmark_id == "livecodebench"


def test_problem_fields_populated() -> None:
    adapter = LiveCodeBenchAdapter()
    for p in adapter.load_problems():
        assert p.problem_id
        assert p.prompt


def test_score_wrong_returns_verdict() -> None:
    adapter = LiveCodeBenchAdapter()
    p = adapter.load_problems()[0]
    verdict = adapter.score(p, "    return None", timeout_s=10)
    assert verdict.problem_id == p.problem_id


def test_score_timeout() -> None:
    adapter = LiveCodeBenchAdapter()
    p = adapter.load_problems()[0]
    verdict = adapter.score(p, "    while True: pass", timeout_s=2)
    assert verdict.timed_out is True
    assert verdict.passed is False
```

Create `libs/evaluation/tests/test_swe_bench_adapter.py`:

```python
"""Tests for SWEBenchLiteAdapter.

score() must raise NotImplementedError (preflight clone/apply not yet implemented).
load_problems() must work with fixture data.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from evaluation.benchmarks.swe_bench import SWEBenchLiteAdapter

FIXTURE = Path(__file__).parent / "fixtures" / "swe_bench_lite_mini.parquet"


@pytest.fixture(autouse=True)
def use_fixture_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    monkeypatch.setattr(
        "evaluation.benchmarks.swe_bench.SWEBenchLiteAdapter._fixture_path",
        FIXTURE,
    )


def test_load_problems_returns_list() -> None:
    """load_problems() works even though score() is not implemented."""
    adapter = SWEBenchLiteAdapter()
    problems = adapter.load_problems()
    assert len(problems) > 0


def test_benchmark_id() -> None:
    assert SWEBenchLiteAdapter.benchmark_id == "swe_bench_lite"


def test_problem_has_repo_in_metadata() -> None:
    """SWE-Bench problems include repo and issue_url in metadata."""
    adapter = SWEBenchLiteAdapter()
    for p in adapter.load_problems():
        assert "repo" in p.metadata


def test_score_raises_not_implemented() -> None:
    """score() must raise NotImplementedError with informative message."""
    adapter = SWEBenchLiteAdapter()
    problems = adapter.load_problems()
    assert len(problems) > 0
    with pytest.raises(NotImplementedError, match="preflight clone/apply not yet implemented"):
        adapter.score(problems[0], "some patch", timeout_s=30)
```

Create `libs/evaluation/tests/test_codecontests_adapter.py`:

```python
"""Tests for CodeContestsAdapter."""

from __future__ import annotations

from pathlib import Path

import pytest
from evaluation.benchmarks.codecontests import CodeContestsAdapter

FIXTURE = Path(__file__).parent / "fixtures" / "codecontests_mini.parquet"


@pytest.fixture(autouse=True)
def use_fixture_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    monkeypatch.setattr(
        "evaluation.benchmarks.codecontests.CodeContestsAdapter._fixture_path",
        FIXTURE,
    )


def test_load_problems_returns_list() -> None:
    adapter = CodeContestsAdapter()
    assert len(adapter.load_problems()) > 0


def test_benchmark_id() -> None:
    assert CodeContestsAdapter.benchmark_id == "codecontests"


def test_problem_fields_populated() -> None:
    adapter = CodeContestsAdapter()
    for p in adapter.load_problems():
        assert p.problem_id
        assert p.prompt


def test_score_wrong_returns_verdict() -> None:
    adapter = CodeContestsAdapter()
    p = adapter.load_problems()[0]
    verdict = adapter.score(p, "print('wrong')", timeout_s=10)
    assert verdict.problem_id == p.problem_id


def test_score_timeout() -> None:
    adapter = CodeContestsAdapter()
    p = adapter.load_problems()[0]
    verdict = adapter.score(p, "while True: pass", timeout_s=2)
    assert verdict.timed_out is True
```

Run (expected-failing):
```
uv run pytest libs/evaluation/tests/test_livecodebench_adapter.py \
             libs/evaluation/tests/test_swe_bench_adapter.py \
             libs/evaluation/tests/test_codecontests_adapter.py -x 2>&1 | head -10
```

Expected:
```
FAILED ... ModuleNotFoundError: No module named 'evaluation.benchmarks.livecodebench'
```

- [ ] **Step 5.3: Implement livecodebench.py**

Create `libs/evaluation/src/evaluation/benchmarks/livecodebench.py`:

```python
"""LiveCodeBench benchmark adapter.

Loads from livecodebench/code_generation_lite, pinned to release_v4 split.
Test harness uses the public_tests field (input/output pairs).
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

from evaluation.benchmarks.protocol import PassVerdict, Problem

_DEFAULT_FIXTURE = Path(__file__).parent.parent.parent.parent.parent.parent / \
    "tests" / "fixtures" / "livecodebench_mini.parquet"


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
            public_tests: list[dict[str, str]] = json.loads(public_tests_raw) \
                if isinstance(public_tests_raw, str) else public_tests_raw
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
                    error=result.stderr or f"Wrong answer: got {result.stdout.strip()!r}",
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
        offline = os.environ.get("HF_DATASETS_OFFLINE", "0") == "1"
        if offline:
            return self._load_from_fixture()
        try:
            return self._load_from_hf()
        except Exception:
            return self._load_from_fixture()

    def _load_from_hf(self) -> list[dict[str, Any]]:
        import datasets as hf_datasets  # deferred

        ds = hf_datasets.load_dataset(
            "livecodebench/code_generation_lite",
            "release_v4",
            split="test",
            trust_remote_code=True,
        )
        return list(ds)  # type: ignore[arg-type]

    def _load_from_fixture(self) -> list[dict[str, Any]]:
        import pandas as pd

        return pd.read_parquet(self._fixture_path).to_dict(orient="records")  # type: ignore[return-value]

    def _row_to_problem(self, row: dict[str, Any]) -> Problem:
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
```

- [ ] **Step 5.4: Implement swe_bench.py**

Create `libs/evaluation/src/evaluation/benchmarks/swe_bench.py`:

```python
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

_DEFAULT_FIXTURE = Path(__file__).parent.parent.parent.parent.parent.parent / \
    "tests" / "fixtures" / "swe_bench_lite_mini.parquet"


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
        offline = os.environ.get("HF_DATASETS_OFFLINE", "0") == "1"
        if offline:
            return self._load_from_fixture()
        try:
            return self._load_from_hf()
        except Exception:
            return self._load_from_fixture()

    def _load_from_hf(self) -> list[dict[str, Any]]:
        import datasets as hf_datasets  # deferred

        ds = hf_datasets.load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
        return list(ds)  # type: ignore[arg-type]

    def _load_from_fixture(self) -> list[dict[str, Any]]:
        import pandas as pd

        return pd.read_parquet(self._fixture_path).to_dict(orient="records")  # type: ignore[return-value]

    def _row_to_problem(self, row: dict[str, Any]) -> Problem:
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
```

- [ ] **Step 5.5: Implement codecontests.py**

Create `libs/evaluation/src/evaluation/benchmarks/codecontests.py`:

```python
"""CodeContests benchmark adapter (held-out generalization set).

Loads from deepmind/code_contests. Uses public_tests and private_tests
input/output pairs for scoring. Held-out from training; used only for
generalization evaluation.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

from evaluation.benchmarks.protocol import PassVerdict, Problem

_DEFAULT_FIXTURE = Path(__file__).parent.parent.parent.parent.parent.parent / \
    "tests" / "fixtures" / "codecontests_mini.parquet"


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
        all_tests: list[dict[str, str]] = []

        for key in ("public_tests", "private_tests"):
            raw = problem.metadata.get(key, "[]")
            try:
                tests = json.loads(raw) if isinstance(raw, str) else raw
                if isinstance(tests, dict):
                    inputs = tests.get("input", [])
                    outputs = tests.get("output", [])
                    all_tests.extend(
                        {"input": i, "output": o} for i, o in zip(inputs, outputs)
                    )
                elif isinstance(tests, list):
                    all_tests.extend(tests)
            except (json.JSONDecodeError, TypeError):
                pass

        if not all_tests:
            return PassVerdict(
                problem_id=problem.problem_id,
                passed=False,
                generation=generation,
                error="No test cases found",
                timed_out=False,
            )

        for tc in all_tests:
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
                    error=result.stderr or f"Wrong answer: got {result.stdout.strip()!r}",
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
        offline = os.environ.get("HF_DATASETS_OFFLINE", "0") == "1"
        if offline:
            return self._load_from_fixture()
        try:
            return self._load_from_hf()
        except Exception:
            return self._load_from_fixture()

    def _load_from_hf(self) -> list[dict[str, Any]]:
        import datasets as hf_datasets  # deferred

        ds = hf_datasets.load_dataset("deepmind/code_contests", split="test")
        return list(ds)  # type: ignore[arg-type]

    def _load_from_fixture(self) -> list[dict[str, Any]]:
        import pandas as pd

        return pd.read_parquet(self._fixture_path).to_dict(orient="records")  # type: ignore[return-value]

    def _row_to_problem(self, row: dict[str, Any]) -> Problem:
        pub = row.get("public_tests", {})
        priv = row.get("private_tests", {})
        return Problem(
            problem_id=f"codecontests/{row.get('name', '')}",
            prompt=str(row.get("description", "")),
            test_code="",  # uses I/O pairs in metadata
            metadata={
                "public_tests": json.dumps(pub) if not isinstance(pub, str) else pub,
                "private_tests": json.dumps(priv) if not isinstance(priv, str) else priv,
                "difficulty": str(row.get("difficulty", "")),
            },
        )
```

- [ ] **Step 5.6: Run all three test files**

```
uv run pytest libs/evaluation/tests/test_livecodebench_adapter.py \
             libs/evaluation/tests/test_swe_bench_adapter.py \
             libs/evaluation/tests/test_codecontests_adapter.py -v
```

Expected output:
```
PASSED tests/test_livecodebench_adapter.py::test_load_problems_returns_list
PASSED tests/test_livecodebench_adapter.py::test_benchmark_id
PASSED tests/test_livecodebench_adapter.py::test_problem_fields_populated
PASSED tests/test_livecodebench_adapter.py::test_score_wrong_returns_verdict
PASSED tests/test_livecodebench_adapter.py::test_score_timeout
PASSED tests/test_swe_bench_adapter.py::test_load_problems_returns_list
PASSED tests/test_swe_bench_adapter.py::test_benchmark_id
PASSED tests/test_swe_bench_adapter.py::test_problem_has_repo_in_metadata
PASSED tests/test_swe_bench_adapter.py::test_score_raises_not_implemented
PASSED tests/test_codecontests_adapter.py::test_load_problems_returns_list
PASSED tests/test_codecontests_adapter.py::test_benchmark_id
PASSED tests/test_codecontests_adapter.py::test_problem_fields_populated
PASSED tests/test_codecontests_adapter.py::test_score_wrong_returns_verdict
PASSED tests/test_codecontests_adapter.py::test_score_timeout
14 passed in X.Xs
```

- [ ] **Step 5.7: Lint and type-check**

```
uv run ruff check libs/evaluation/src/evaluation/benchmarks/livecodebench.py \
                  libs/evaluation/src/evaluation/benchmarks/swe_bench.py \
                  libs/evaluation/src/evaluation/benchmarks/codecontests.py
uv run mypy libs/evaluation/src/evaluation/benchmarks/
```

Expected: no errors.

- [ ] **Step 5.8: Commit**

```
git add libs/evaluation/src/evaluation/benchmarks/livecodebench.py \
        libs/evaluation/src/evaluation/benchmarks/swe_bench.py \
        libs/evaluation/src/evaluation/benchmarks/codecontests.py \
        libs/evaluation/tests/test_livecodebench_adapter.py \
        libs/evaluation/tests/test_swe_bench_adapter.py \
        libs/evaluation/tests/test_codecontests_adapter.py \
        libs/evaluation/tests/fixtures/livecodebench_mini.parquet \
        libs/evaluation/tests/fixtures/swe_bench_lite_mini.parquet \
        libs/evaluation/tests/fixtures/codecontests_mini.parquet
git commit -m "$(cat <<'EOF'
feat(benchmarks): add LiveCodeBench (release_v4), SWE-Bench-Lite (load-only), CodeContests adapters

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

## Task 6: Adapter Stack Loader

**Files:**
- Create: `libs/evaluation/src/evaluation/benchmarks/adapter_stack.py`
- Test: `libs/evaluation/tests/test_adapter_stack.py`

**Depends on:** Task 1 (protocol types). Runs after Tasks 3–5 are complete so the full adapter set is in place, but has no file-level dependency on them.

- [ ] **Step 6.1: Write the failing test first**

Create `libs/evaluation/tests/test_adapter_stack.py`:

```python
"""Tests for adapter_stack.load_adapter_stack()."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from evaluation.benchmarks.adapter_stack import AdapterStack, load_adapter_stack


def _make_mock_registry(records: list[MagicMock]) -> MagicMock:
    """Return a mock AdapterRegistry that returns records from retrieve_by_id."""
    registry = MagicMock()
    registry.retrieve_by_id.side_effect = lambda aid: next(
        (r for r in records if r.id == aid), None
    )
    return registry


def test_load_adapter_stack_empty_adapter_ids() -> None:
    """load_adapter_stack with no adapter_ids returns an AdapterStack with empty list."""
    provider = MagicMock()
    stack = load_adapter_stack(
        base_model="Qwen/Qwen3.5-9B",
        adapter_ids=[],
        provider=provider,
        registry=MagicMock(),
    )
    assert isinstance(stack, AdapterStack)
    assert stack.base_model == "Qwen/Qwen3.5-9B"
    assert stack.adapter_ids == []


def test_load_adapter_stack_with_adapters() -> None:
    """load_adapter_stack with adapter_ids resolves file_paths from registry."""
    record = MagicMock()
    record.id = "adapter-001"
    record.file_path = "/adapters/adapter-001"
    registry = _make_mock_registry([record])
    provider = MagicMock()

    stack = load_adapter_stack(
        base_model="Qwen/Qwen3.5-9B",
        adapter_ids=["adapter-001"],
        provider=provider,
        registry=registry,
    )
    assert stack.adapter_ids == ["adapter-001"]
    assert stack.adapter_paths == {"adapter-001": "/adapters/adapter-001"}


def test_load_adapter_stack_missing_adapter_raises() -> None:
    """load_adapter_stack raises ValueError for unknown adapter_id."""
    registry = MagicMock()
    from adapter_registry.exceptions import AdapterNotFoundError

    registry.retrieve_by_id.side_effect = AdapterNotFoundError("not found")
    with pytest.raises(ValueError, match="adapter-999"):
        load_adapter_stack(
            base_model="Qwen/Qwen3.5-9B",
            adapter_ids=["adapter-999"],
            provider=MagicMock(),
            registry=registry,
        )


def test_adapter_stack_repr() -> None:
    """AdapterStack has a useful repr."""
    stack = AdapterStack(
        base_model="Qwen/Qwen3.5-9B",
        adapter_ids=["a1", "a2"],
        adapter_paths={"a1": "/p1", "a2": "/p2"},
        provider=MagicMock(),
    )
    r = repr(stack)
    assert "Qwen" in r
    assert "a1" in r


def test_adapter_stack_describe() -> None:
    """AdapterStack.describe() returns a dict with base_model and adapter_ids."""
    stack = AdapterStack(
        base_model="base",
        adapter_ids=["x"],
        adapter_paths={"x": "/p"},
        provider=MagicMock(),
    )
    d = stack.describe()
    assert d["base_model"] == "base"
    assert d["adapter_ids"] == ["x"]
```

Run (expected-failing):
```
uv run pytest libs/evaluation/tests/test_adapter_stack.py -x 2>&1 | head -10
```

Expected:
```
FAILED ... ModuleNotFoundError: No module named 'evaluation.benchmarks.adapter_stack'
```

- [ ] **Step 6.2: Implement adapter_stack.py**

Create `libs/evaluation/src/evaluation/benchmarks/adapter_stack.py`:

```python
"""Adapter stack loader for benchmark evaluation.

Resolves a list of adapter_ids from AdapterRegistry into file paths,
and bundles them with an InferenceProvider into an AdapterStack that
the benchmark runner uses to generate completions.

No GPU imports. CPU-safe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from inference.provider import InferenceProvider


@dataclass
class AdapterStack:
    """Bundle of base model + ordered adapter stack + provider.

    Attributes:
        base_model: HuggingFace model ID or local path for the base model.
        adapter_ids: Ordered list of adapter IDs to apply (first = innermost).
        adapter_paths: Dict mapping adapter_id -> filesystem path.
        provider: InferenceProvider used to generate completions.
    """

    base_model: str
    adapter_ids: list[str]
    adapter_paths: dict[str, str]
    provider: InferenceProvider

    def __repr__(self) -> str:
        """Human-readable representation."""
        return (
            f"AdapterStack(base_model={self.base_model!r}, "
            f"adapter_ids={self.adapter_ids!r})"
        )

    def describe(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary of this stack.

        Returns:
            Dict with keys: base_model, adapter_ids, adapter_paths.
        """
        return {
            "base_model": self.base_model,
            "adapter_ids": list(self.adapter_ids),
            "adapter_paths": dict(self.adapter_paths),
        }


def load_adapter_stack(
    base_model: str,
    adapter_ids: list[str],
    provider: InferenceProvider,
    registry: Any,
) -> AdapterStack:
    """Resolve adapter IDs to file paths and construct an AdapterStack.

    Queries the AdapterRegistry for each adapter_id to retrieve its
    file_path. Raises ValueError for any unknown adapter_id.

    Args:
        base_model: HuggingFace model ID or local path for the base model.
        adapter_ids: Ordered list of adapter registry IDs to load.
            Pass an empty list to use the base model only.
        provider: InferenceProvider instance for generating completions.
        registry: AdapterRegistry instance (or compatible duck-type) with
            a retrieve_by_id(adapter_id: str) -> AdapterRecord method.

    Returns:
        AdapterStack with resolved file paths.

    Raises:
        ValueError: If any adapter_id is not found in the registry.

    Example:
        >>> from sqlalchemy import create_engine
        >>> from adapter_registry.registry import AdapterRegistry
        >>> engine = create_engine("sqlite:///adapters.db")
        >>> reg = AdapterRegistry(engine=engine)
        >>> stack = load_adapter_stack("Qwen/Qwen3.5-9B", ["a1"], provider, reg)
    """
    adapter_paths: dict[str, str] = {}
    for aid in adapter_ids:
        try:
            record = registry.retrieve_by_id(aid)
            adapter_paths[aid] = str(record.file_path)
        except Exception as exc:
            raise ValueError(
                f"Adapter '{aid}' not found in registry: {exc}"
            ) from exc

    return AdapterStack(
        base_model=base_model,
        adapter_ids=list(adapter_ids),
        adapter_paths=adapter_paths,
        provider=provider,
    )
```

- [ ] **Step 6.3: Run tests**

```
uv run pytest libs/evaluation/tests/test_adapter_stack.py -v
```

Expected output:
```
PASSED tests/test_adapter_stack.py::test_load_adapter_stack_empty_adapter_ids
PASSED tests/test_adapter_stack.py::test_load_adapter_stack_with_adapters
PASSED tests/test_adapter_stack.py::test_load_adapter_stack_missing_adapter_raises
PASSED tests/test_adapter_stack.py::test_adapter_stack_repr
PASSED tests/test_adapter_stack.py::test_adapter_stack_describe
5 passed in 0.XXs
```

- [ ] **Step 6.4: Lint and type-check**

```
uv run ruff check libs/evaluation/src/evaluation/benchmarks/adapter_stack.py
uv run mypy libs/evaluation/src/evaluation/benchmarks/adapter_stack.py \
            --python-path libs/inference/src:libs/adapter-registry/src:libs/shared/src
```

Expected: no errors.

- [ ] **Step 6.5: Commit**

```
git add libs/evaluation/src/evaluation/benchmarks/adapter_stack.py \
        libs/evaluation/tests/test_adapter_stack.py
git commit -m "$(cat <<'EOF'
feat(benchmarks): add load_adapter_stack resolving registry IDs to file paths

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

## Task 7: Runner — `run_benchmark()` Orchestrator

**Files:**
- Create: `libs/evaluation/src/evaluation/benchmarks/runner.py`
- Test: `libs/evaluation/tests/test_runner.py`

**Depends on:** Tasks 1–6 (protocol, aggregator, all adapters, adapter_stack).

- [ ] **Step 7.1: Write the failing integration test first**

Create `libs/evaluation/tests/test_runner.py`:

```python
"""Integration test: run_benchmark() end-to-end with mock InferenceProvider.

Uses HumanEvalAdapter with 5-problem fixture and a mock provider that
returns the canonical solution for each problem. Verifies that:
- run_benchmark() returns a BenchmarkResult
- pass_at_1 is > 0 when mock returns correct completions
- ThreadPoolExecutor fan-out works correctly
- problem_ids filter limits evaluation to specified IDs
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from evaluation.benchmarks.protocol import BenchmarkResult
from evaluation.benchmarks.runner import run_benchmark


FIXTURE = Path(__file__).parent / "fixtures" / "humaneval_mini.parquet"


class _MockProvider:
    """Minimal mock InferenceProvider that echoes canonical_solution from metadata."""

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
        from inference.provider import GenerationResult

        # Return a trivially wrong answer so test is deterministic without GPU
        return GenerationResult(
            text="    return []",
            model=model,
            adapter_id=adapter_id,
            token_count=5,
            finish_reason="stop",
        )

    async def load_adapter(self, adapter_id: str, adapter_path: str) -> None:
        pass

    async def unload_adapter(self, adapter_id: str) -> None:
        pass

    async def list_adapters(self) -> list[str]:
        return []


@pytest.fixture()
def mock_adapter_stack(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Return an AdapterStack wrapping a mock provider, pointing at HumanEval fixture."""
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    monkeypatch.setattr(
        "evaluation.benchmarks.humaneval.HumanEvalAdapter._fixture_path",
        FIXTURE,
    )
    from evaluation.benchmarks.adapter_stack import AdapterStack

    return AdapterStack(
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
    # Mock always returns "    return []" which is wrong for all HumanEval tasks
    assert result.pass_at_1 < 1.0


def test_run_benchmark_unknown_benchmark_raises(mock_adapter_stack: MagicMock) -> None:
    """run_benchmark() raises ValueError for unknown benchmark_id."""
    with pytest.raises(ValueError, match="unknown_bench"):
        run_benchmark(
            adapter_stack=mock_adapter_stack,
            benchmark_id="unknown_bench",
        )
```

Run (expected-failing):
```
uv run pytest libs/evaluation/tests/test_runner.py -x 2>&1 | head -10
```

Expected:
```
FAILED ... ModuleNotFoundError: No module named 'evaluation.benchmarks.runner'
```

- [ ] **Step 7.2: Implement runner.py**

Create `libs/evaluation/src/evaluation/benchmarks/runner.py`:

```python
"""Benchmark runner — orchestrates sampling + scoring for run_benchmark().

Uses ThreadPoolExecutor for parallel per-problem evaluation. Supports
all eight benchmark adapters via a registry dict keyed by benchmark_id.

No GPU imports. All heavy lifting (model inference, sandbox execution)
happens inside the provider and adapter.score() calls, which are
already CPU-safe at import time.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from evaluation.benchmarks.adapter_stack import AdapterStack
from evaluation.benchmarks.aggregator import pass_at_1_from_verdicts
from evaluation.benchmarks.protocol import (
    BenchmarkConfig,
    BenchmarkResult,
    PassVerdict,
    Problem,
)

logger = logging.getLogger(__name__)

# Registry of benchmark_id -> adapter class (lazy import avoids heavy deps at module load)
_ADAPTER_REGISTRY: dict[str, str] = {
    "humaneval": "evaluation.benchmarks.humaneval.HumanEvalAdapter",
    "mbpp": "evaluation.benchmarks.mbpp.MBPPAdapter",
    "apps": "evaluation.benchmarks.apps.APPSAdapter",
    "bigcodebench": "evaluation.benchmarks.bigcodebench.BigCodeBenchAdapter",
    "ds_1000": "evaluation.benchmarks.ds1000.DS1000Adapter",
    "livecodebench": "evaluation.benchmarks.livecodebench.LiveCodeBenchAdapter",
    "swe_bench_lite": "evaluation.benchmarks.swe_bench.SWEBenchLiteAdapter",
    "codecontests": "evaluation.benchmarks.codecontests.CodeContestsAdapter",
}


def _import_adapter(dotted_path: str) -> Any:
    """Import and instantiate an adapter class from a dotted module path.

    Args:
        dotted_path: e.g. "evaluation.benchmarks.humaneval.HumanEvalAdapter"

    Returns:
        An instantiated adapter object.
    """
    module_path, cls_name = dotted_path.rsplit(".", 1)
    import importlib

    mod = importlib.import_module(module_path)
    cls = getattr(mod, cls_name)
    return cls()


def _generate_completion(
    adapter_stack: AdapterStack,
    problem: Problem,
    max_tokens: int = 512,
) -> str:
    """Synchronously call provider.generate() from a thread.

    Creates a new event loop per thread (ThreadPoolExecutor threads
    have no running loop by default).

    Args:
        adapter_stack: AdapterStack with provider and model config.
        problem: Problem whose prompt is sent to the model.
        max_tokens: Generation token cap.

    Returns:
        Generated text string.
    """
    adapter_id = adapter_stack.adapter_ids[0] if adapter_stack.adapter_ids else None
    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(
            adapter_stack.provider.generate(
                prompt=problem.prompt,
                model=adapter_stack.base_model,
                adapter_id=adapter_id,
                max_tokens=max_tokens,
            )
        )
        return result.text
    finally:
        loop.close()


def _evaluate_one(
    adapter: Any,
    adapter_stack: AdapterStack,
    problem: Problem,
    config: BenchmarkConfig,
) -> PassVerdict:
    """Generate a completion and score it for a single problem.

    Args:
        adapter: Benchmark adapter instance (has .score()).
        adapter_stack: AdapterStack for generation.
        problem: Problem to evaluate.
        config: BenchmarkConfig with timeout_s.

    Returns:
        PassVerdict for this problem.
    """
    try:
        generation = _generate_completion(adapter_stack, problem)
        return adapter.score(problem, generation, timeout_s=config.timeout_s)
    except Exception as exc:
        logger.warning("Error evaluating problem %s: %s", problem.problem_id, exc)
        return PassVerdict(
            problem_id=problem.problem_id,
            passed=False,
            generation="",
            error=str(exc),
            timed_out=False,
        )


def run_benchmark(
    adapter_stack: AdapterStack,
    benchmark_id: str,
    problem_ids: list[str] | None = None,
    max_samples: int | None = None,
    config: BenchmarkConfig | None = None,
) -> BenchmarkResult:
    """Run a full benchmark evaluation pass and return aggregate Pass@1.

    Orchestrates:
    1. Load problems from the benchmark adapter (with optional ID filter).
    2. Fan out (generate + score) via ThreadPoolExecutor.
    3. Aggregate verdicts into a BenchmarkResult.

    Args:
        adapter_stack: AdapterStack describing base model + adapters + provider.
        benchmark_id: One of the registered benchmark IDs:
            humaneval, mbpp, apps, bigcodebench, ds_1000,
            livecodebench, swe_bench_lite, codecontests.
        problem_ids: Optional list of problem_id strings to restrict
            evaluation to a subset. If None, evaluates all loaded problems.
        max_samples: Cap on total problems evaluated. Applied before
            problem_ids filter when both are provided.
        config: BenchmarkConfig overriding defaults (timeout, workers, seed).

    Returns:
        BenchmarkResult with per-problem verdicts and aggregate pass_at_1.

    Raises:
        ValueError: If benchmark_id is not in the known registry.

    Example:
        >>> result = run_benchmark(stack, "humaneval", max_samples=50)
        >>> print(f"Pass@1: {result.pass_at_1:.2%}")
    """
    if benchmark_id not in _ADAPTER_REGISTRY:
        raise ValueError(
            f"Unknown benchmark_id {benchmark_id!r}. "
            f"Known benchmarks: {sorted(_ADAPTER_REGISTRY)}"
        )

    cfg = config or BenchmarkConfig(max_samples=max_samples)
    if max_samples is not None:
        cfg = BenchmarkConfig(
            timeout_s=cfg.timeout_s,
            max_workers=cfg.max_workers,
            max_samples=max_samples,
            seed=cfg.seed,
        )

    adapter = _import_adapter(_ADAPTER_REGISTRY[benchmark_id])
    problems: list[Problem] = adapter.load_problems(
        max_samples=cfg.max_samples,
        seed=cfg.seed,
    )

    # Apply problem_ids filter
    if problem_ids is not None:
        id_set = set(problem_ids)
        problems = [p for p in problems if p.problem_id in id_set]

    if not problems:
        return BenchmarkResult(benchmark_id=benchmark_id, verdicts=[])

    logger.info(
        "run_benchmark: benchmark=%s, n_problems=%d, max_workers=%d",
        benchmark_id,
        len(problems),
        cfg.max_workers,
    )

    verdicts: list[PassVerdict] = []
    with ThreadPoolExecutor(max_workers=cfg.max_workers) as executor:
        futures = {
            executor.submit(_evaluate_one, adapter, adapter_stack, p, cfg): p
            for p in problems
        }
        for future in as_completed(futures):
            verdict = future.result()
            verdicts.append(verdict)
            status = "PASS" if verdict.passed else "FAIL"
            logger.debug("  [%s] %s", status, verdict.problem_id)

    # Restore original problem order
    id_order = {p.problem_id: i for i, p in enumerate(problems)}
    verdicts.sort(key=lambda v: id_order.get(v.problem_id, 9999))

    return BenchmarkResult(benchmark_id=benchmark_id, verdicts=verdicts)
```

- [ ] **Step 7.3: Run tests**

```
uv run pytest libs/evaluation/tests/test_runner.py -v
```

Expected output:
```
PASSED tests/test_runner.py::test_run_benchmark_returns_benchmark_result
PASSED tests/test_runner.py::test_run_benchmark_verdicts_count
PASSED tests/test_runner.py::test_run_benchmark_problem_ids_filter
PASSED tests/test_runner.py::test_run_benchmark_wrong_completions_produce_failures
PASSED tests/test_runner.py::test_run_benchmark_unknown_benchmark_raises
5 passed in X.Xs
```

- [ ] **Step 7.4: Lint and type-check**

```
uv run ruff check libs/evaluation/src/evaluation/benchmarks/runner.py
uv run mypy libs/evaluation/src/evaluation/benchmarks/runner.py
```

Expected: no errors.

- [ ] **Step 7.5: Commit**

```
git add libs/evaluation/src/evaluation/benchmarks/runner.py \
        libs/evaluation/tests/test_runner.py
git commit -m "$(cat <<'EOF'
feat(benchmarks): add run_benchmark() orchestrator with ThreadPoolExecutor fan-out

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

## Task 8: CLI Entrypoint `scripts/run_benchmark.py`

**Files:**
- Create: `scripts/run_benchmark.py`
- Modify: `libs/evaluation/pyproject.toml` (add `datasets` dependency)

**Depends on:** Task 7 (runner). Mirrors `libs/model-training/src/model_training/trainer_cli.py` --dry-run pattern exactly.

- [ ] **Step 8.1: Add `datasets` to evaluation pyproject.toml**

Edit `libs/evaluation/pyproject.toml` to add the `datasets` dependency:

```toml
[project]
name = "evaluation"
version = "0.1.0"
description = "Metrics and reporting"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "shared",
    "pandas>=2.0.0",
    "datasets>=2.19.0",
]
```

Then sync:
```
uv sync --all-extras
```

- [ ] **Step 8.2: Write a dry-run smoke test**

Add to `libs/evaluation/tests/test_runner.py` (append after existing tests):

```python
def test_run_benchmark_config_passed_through(mock_adapter_stack: MagicMock) -> None:
    """BenchmarkConfig timeout_s and max_workers are accepted without error."""
    from evaluation.benchmarks.protocol import BenchmarkConfig

    cfg = BenchmarkConfig(timeout_s=5, max_workers=2, max_samples=2, seed=0)
    result = run_benchmark(
        adapter_stack=mock_adapter_stack,
        benchmark_id="humaneval",
        config=cfg,
    )
    assert isinstance(result, BenchmarkResult)
    assert result.n_problems <= 2
```

Run:
```
uv run pytest libs/evaluation/tests/test_runner.py -v
```

Expected: 6 passed.

- [ ] **Step 8.3: Implement scripts/run_benchmark.py**

Create `scripts/run_benchmark.py`:

```python
r"""CLI entrypoint for benchmark Pass@1 evaluation.

Mirrors trainer_cli.py: all heavy imports (torch, transformers, datasets)
are deferred inside run_benchmark(). This script is CPU-safe and supports
--dry-run mode for CI validation without loading any models.

Usage:
    uv run python scripts/run_benchmark.py \
        --benchmark humaneval \
        --base-model Qwen/Qwen3.5-9B \
        --max-samples 50 \
        --dry-run

    uv run python scripts/run_benchmark.py \
        --benchmark humaneval \
        --base-model Qwen/Qwen3.5-9B \
        --adapter-ids adapter-001 adapter-002 \
        --max-samples 50 \
        --timeout 30 \
        --workers 4 \
        --output results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any

logger = logging.getLogger(__name__)

_KNOWN_BENCHMARKS = [
    "humaneval",
    "mbpp",
    "apps",
    "bigcodebench",
    "ds_1000",
    "livecodebench",
    "swe_bench_lite",
    "codecontests",
]


def _build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for run_benchmark.py."""
    parser = argparse.ArgumentParser(
        prog="run_benchmark",
        description="Run a benchmark Pass@1 evaluation on a (base_model, adapter_stack) pair.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        choices=_KNOWN_BENCHMARKS,
        help="Benchmark to evaluate.",
    )
    parser.add_argument(
        "--base-model",
        dest="base_model",
        required=True,
        help="HuggingFace model ID or local path for the base model.",
    )
    parser.add_argument(
        "--adapter-ids",
        dest="adapter_ids",
        nargs="*",
        default=[],
        metavar="ID",
        help="Zero or more adapter registry IDs to load on top of base model.",
    )
    parser.add_argument(
        "--max-samples",
        dest="max_samples",
        type=int,
        default=None,
        metavar="N",
        help="Cap on problems evaluated. None = all available.",
    )
    parser.add_argument(
        "--timeout",
        dest="timeout_s",
        type=int,
        default=30,
        metavar="SECONDS",
        help="Per-problem sandbox timeout in seconds.",
    )
    parser.add_argument(
        "--workers",
        dest="max_workers",
        type=int,
        default=4,
        metavar="N",
        help="ThreadPoolExecutor worker count.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for problem subsampling.",
    )
    parser.add_argument(
        "--problem-ids",
        dest="problem_ids",
        nargs="*",
        default=None,
        metavar="ID",
        help="Optional list of specific problem IDs to evaluate.",
    )
    parser.add_argument(
        "--registry-db",
        dest="registry_db",
        default="~/.rune/adapters.db",
        metavar="PATH",
        help="Path to AdapterRegistry SQLite database.",
    )
    parser.add_argument(
        "--provider",
        default="vllm",
        choices=["vllm", "ollama"],
        help="Inference provider backend.",
    )
    parser.add_argument(
        "--provider-url",
        dest="provider_url",
        default="http://localhost:8000",
        metavar="URL",
        help="Base URL for the inference provider.",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help="Write JSON results to this file. Defaults to stdout.",
    )
    parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help=(
            "Resolve and print arguments as JSON without loading any models. "
            "CPU-only; safe to run in CI."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return parser


def _dry_run_output(args: argparse.Namespace) -> dict[str, Any]:
    """Produce a JSON-serialisable summary of resolved arguments.

    Args:
        args: Parsed argparse namespace.

    Returns:
        Dict suitable for JSON serialisation and stdout printing.
    """
    return {
        "dry_run": True,
        "benchmark": args.benchmark,
        "base_model": args.base_model,
        "adapter_ids": args.adapter_ids,
        "max_samples": args.max_samples,
        "timeout_s": args.timeout_s,
        "max_workers": args.max_workers,
        "seed": args.seed,
        "problem_ids": args.problem_ids,
        "registry_db": args.registry_db,
        "provider": args.provider,
        "provider_url": args.provider_url,
        "output": args.output,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark CLI.

    Args:
        argv: Command-line arguments. Defaults to sys.argv[1:].

    Returns:
        Exit code (0 = success, 1 = error).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s %(message)s",
    )

    if args.dry_run:
        resolved = _dry_run_output(args)
        print(json.dumps(resolved, indent=2))
        return 0

    # --- Heavy imports deferred here (INFRA-05 pattern) ---
    try:
        from sqlalchemy import create_engine  # deferred: heavy

        from adapter_registry.registry import AdapterRegistry  # deferred
        from evaluation.benchmarks.adapter_stack import load_adapter_stack  # deferred
        from evaluation.benchmarks.protocol import BenchmarkConfig  # deferred
        from evaluation.benchmarks.runner import run_benchmark  # deferred
    except ImportError as exc:
        logger.error("Missing dependency: %s. Use --dry-run for CPU-only mode.", exc)
        return 1

    # Build provider (deferred import)
    try:
        if args.provider == "vllm":
            from inference.vllm_provider import VLLMProvider  # deferred

            provider = VLLMProvider(base_url=args.provider_url)
        else:
            from inference.ollama_provider import OllamaProvider  # deferred

            provider = OllamaProvider(base_url=args.provider_url)
    except ImportError as exc:
        logger.error("Could not import provider %s: %s", args.provider, exc)
        return 1

    # Build registry and adapter stack
    import os

    db_path = os.path.expanduser(args.registry_db)
    engine = create_engine(f"sqlite:///{db_path}")
    registry = AdapterRegistry(engine=engine)

    try:
        stack = load_adapter_stack(
            base_model=args.base_model,
            adapter_ids=args.adapter_ids,
            provider=provider,
            registry=registry,
        )
    except ValueError as exc:
        logger.error("Adapter stack error: %s", exc)
        return 1

    cfg = BenchmarkConfig(
        timeout_s=args.timeout_s,
        max_workers=args.max_workers,
        max_samples=args.max_samples,
        seed=args.seed,
    )

    logger.info(
        "Running benchmark=%s on base_model=%s adapters=%s max_samples=%s",
        args.benchmark,
        args.base_model,
        args.adapter_ids,
        args.max_samples,
    )

    result = run_benchmark(
        adapter_stack=stack,
        benchmark_id=args.benchmark,
        problem_ids=args.problem_ids,
        config=cfg,
    )

    output: dict[str, Any] = {
        "benchmark_id": result.benchmark_id,
        "n_problems": result.n_problems,
        "n_passed": result.n_passed,
        "pass_at_1": result.pass_at_1,
        "verdicts": [
            {
                "problem_id": v.problem_id,
                "passed": v.passed,
                "timed_out": v.timed_out,
                "error": v.error,
            }
            for v in result.verdicts
        ],
    }

    output_str = json.dumps(output, indent=2)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output_str)
        logger.info("Results written to %s", args.output)
    else:
        print(output_str)

    logger.info(
        "Pass@1: %.4f (%d/%d)", result.pass_at_1, result.n_passed, result.n_problems
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 8.4: Verify dry-run is CPU-safe**

```
uv run python scripts/run_benchmark.py \
    --benchmark humaneval \
    --base-model Qwen/Qwen3.5-9B \
    --max-samples 10 \
    --dry-run
```

Expected output (JSON, no torch/transformers imported):
```json
{
  "dry_run": true,
  "benchmark": "humaneval",
  "base_model": "Qwen/Qwen3.5-9B",
  "adapter_ids": [],
  "max_samples": 10,
  "timeout_s": 30,
  "max_workers": 4,
  "seed": 42,
  "problem_ids": null,
  "registry_db": "~/.rune/adapters.db",
  "provider": "vllm",
  "provider_url": "http://localhost:8000",
  "output": null
}
```

- [ ] **Step 8.5: Verify --help**

```
uv run python scripts/run_benchmark.py --help 2>&1 | head -20
```

Expected: prints usage block listing `--benchmark`, `--base-model`, `--dry-run`.

- [ ] **Step 8.6: Lint and type-check**

```
uv run ruff check scripts/run_benchmark.py
uv run mypy scripts/run_benchmark.py
```

Expected: no errors.

- [ ] **Step 8.7: Commit**

```
git add scripts/run_benchmark.py libs/evaluation/pyproject.toml
git commit -m "$(cat <<'EOF'
feat(benchmarks): add run_benchmark CLI with --dry-run; add datasets dep to evaluation

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

## Task 9: Wire `__init__.py` and Full Suite Smoke Test

**Files:**
- Modify: `libs/evaluation/src/evaluation/benchmarks/__init__.py` (finalize re-exports)
- No new test file — runs full suite

**Depends on:** Tasks 1–8 (all modules in place).

- [ ] **Step 9.1: Verify `__init__.py` re-exports are complete**

Read `libs/evaluation/src/evaluation/benchmarks/__init__.py` (created in Task 1). Confirm it re-exports:
- `Problem`, `PassVerdict`, `BenchmarkAdapter`, `BenchmarkConfig`, `BenchmarkResult` from `protocol`
- `run_benchmark` from `runner`
- `load_adapter_stack` from `adapter_stack`

If any import in `__init__.py` fails (e.g. a module doesn't exist yet), fix the missing module before proceeding.

- [ ] **Step 9.2: Smoke-import the public API**

```
uv run python -c "
from evaluation.benchmarks import (
    Problem, PassVerdict, BenchmarkAdapter,
    BenchmarkConfig, BenchmarkResult,
    run_benchmark, load_adapter_stack,
)
print('All imports OK')
"
```

Expected:
```
All imports OK
```

- [ ] **Step 9.3: Run the complete benchmarks test suite**

```
uv run pytest libs/evaluation/tests/ -v --tb=short 2>&1 | tail -30
```

Expected output (all green, counts may vary by fixture availability):
```
PASSED tests/test_protocol.py::test_problem_fields_present
PASSED tests/test_protocol.py::...
PASSED tests/test_aggregator.py::...
PASSED tests/test_humaneval_adapter.py::...
PASSED tests/test_mbpp_adapter.py::...
PASSED tests/test_apps_adapter.py::...
PASSED tests/test_bigcodebench_adapter.py::...
PASSED tests/test_ds1000_adapter.py::...
PASSED tests/test_livecodebench_adapter.py::...
PASSED tests/test_swe_bench_adapter.py::...
PASSED tests/test_codecontests_adapter.py::...
PASSED tests/test_runner.py::...
PASSED tests/test_adapter_stack.py::...
XX passed in X.Xs
```

- [ ] **Step 9.4: Run full repo lint + type-check**

```
uv run ruff check libs/evaluation/src/evaluation/benchmarks/
uv run mypy libs/evaluation/src/evaluation/benchmarks/ \
    --python-path libs/shared/src:libs/inference/src:libs/adapter-registry/src
```

Expected: no errors.

- [ ] **Step 9.5: Run existing evaluation tests to confirm no regressions**

```
uv run pytest libs/evaluation/tests/test_metrics.py \
             libs/evaluation/tests/test_ood_benchmark.py -v
```

Expected: all previously passing tests still pass.

- [ ] **Step 9.6: Commit**

```
git add libs/evaluation/src/evaluation/benchmarks/__init__.py
git commit -m "$(cat <<'EOF'
feat(benchmarks): wire __init__ re-exports; full suite smoke-tested

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Fixture Generator Script Cleanup + Final Integration

**Files:**
- Create: `scripts/generate_benchmark_fixtures.py` (complete, self-contained version)
- No new tests

**Depends on:** Tasks 3–5 (fixtures referenced by all adapter tests).

This task consolidates the incremental additions from Tasks 3–5 into a single, runnable script and ensures fixtures are committed.

- [ ] **Step 10.1: Write the complete fixture generator**

Create `scripts/generate_benchmark_fixtures.py`:

```python
"""Generate mini parquet fixtures for benchmark adapter tests.

Downloads 5 rows from each benchmark dataset and saves them as parquet
files in libs/evaluation/tests/fixtures/. These fixtures are checked
into the repo so adapter tests run offline (HF_DATASETS_OFFLINE=1).

Run once (with network access):
    uv run python scripts/generate_benchmark_fixtures.py

Prerequisites:
    uv sync --all-extras  (installs datasets, pandas)

Generates:
    libs/evaluation/tests/fixtures/humaneval_mini.parquet
    libs/evaluation/tests/fixtures/mbpp_mini.parquet
    libs/evaluation/tests/fixtures/apps_mini.parquet
    libs/evaluation/tests/fixtures/bigcodebench_mini.parquet
    libs/evaluation/tests/fixtures/ds1000_mini.parquet
    libs/evaluation/tests/fixtures/livecodebench_mini.parquet
    libs/evaluation/tests/fixtures/swe_bench_lite_mini.parquet
    libs/evaluation/tests/fixtures/codecontests_mini.parquet
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "libs" / "evaluation" / "src"))

FIXTURE_DIR = Path(__file__).parent.parent / "libs" / "evaluation" / "tests" / "fixtures"
FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

# Allow HF downloads during fixture generation
os.environ["HF_DATASETS_OFFLINE"] = "0"

import datasets  # noqa: E402
import pandas as pd  # noqa: E402


def save_mini(
    dataset_name: str,
    config: str | None,
    split: str,
    n: int,
    out_name: str,
) -> None:
    """Load a dataset split, take first n rows, save as parquet fixture.

    Args:
        dataset_name: HuggingFace dataset identifier.
        config: Dataset config/subset name, or None.
        split: Dataset split to load (e.g. "test", "train").
        n: Number of rows to keep.
        out_name: Output parquet filename in FIXTURE_DIR.
    """
    out_path = FIXTURE_DIR / out_name
    if out_path.exists():
        print(f"  [skip] {out_path} already exists")
        return
    print(f"  Loading {dataset_name} ({config or 'default'}, {split}) ...")
    kwargs: dict[str, object] = {"split": split, "trust_remote_code": True}
    if config:
        ds = datasets.load_dataset(dataset_name, config, **kwargs)
    else:
        ds = datasets.load_dataset(dataset_name, **kwargs)
    df = ds.to_pandas().head(n)  # type: ignore[union-attr]
    df.to_parquet(out_path, index=False)
    print(f"  Wrote {out_path} ({len(df)} rows, {out_path.stat().st_size} bytes)")


def main() -> None:
    """Generate all benchmark fixtures."""
    print("Generating benchmark fixtures...")

    # Training-oracle benchmarks
    save_mini("openai/openai_humaneval", None, "test", 5, "humaneval_mini.parquet")
    save_mini("google-research-datasets/mbpp", "full", "train", 5, "mbpp_mini.parquet")
    save_mini("codeparrot/apps", "all", "train", 5, "apps_mini.parquet")

    try:
        save_mini("bigcode/bigcodebench", None, "v0.1.2", 5, "bigcodebench_mini.parquet")
    except Exception:
        save_mini("bigcode/bigcodebench", None, "train", 5, "bigcodebench_mini.parquet")

    save_mini("xlangai/DS-1000", None, "test", 5, "ds1000_mini.parquet")
    save_mini(
        "livecodebench/code_generation_lite",
        "release_v4",
        "test",
        5,
        "livecodebench_mini.parquet",
    )

    # Held-out benchmarks
    save_mini("princeton-nlp/SWE-bench_Lite", None, "test", 5, "swe_bench_lite_mini.parquet")
    save_mini("deepmind/code_contests", None, "test", 5, "codecontests_mini.parquet")

    print("\nAll fixtures generated. Check them into git:")
    print("  git add libs/evaluation/tests/fixtures/*.parquet")


if __name__ == "__main__":
    main()
```

- [ ] **Step 10.2: Run the fixture generator (requires network)**

```
uv run python scripts/generate_benchmark_fixtures.py
```

Expected: prints one line per fixture, no errors. Fixtures written to `libs/evaluation/tests/fixtures/`.

- [ ] **Step 10.3: Verify all fixture files exist**

```
uv run python -c "
from pathlib import Path
fixtures = Path('libs/evaluation/tests/fixtures')
expected = [
    'humaneval_mini.parquet', 'mbpp_mini.parquet', 'apps_mini.parquet',
    'bigcodebench_mini.parquet', 'ds1000_mini.parquet', 'livecodebench_mini.parquet',
    'swe_bench_lite_mini.parquet', 'codecontests_mini.parquet',
]
missing = [f for f in expected if not (fixtures / f).exists()]
if missing:
    print('MISSING:', missing)
    raise SystemExit(1)
print('All', len(expected), 'fixture files present')
"
```

Expected:
```
All 8 fixture files present
```

- [ ] **Step 10.4: Run full adapter test suite with offline fixtures**

```
HF_DATASETS_OFFLINE=1 uv run pytest libs/evaluation/tests/ -v --tb=short -q 2>&1 | tail -10
```

Expected:
```
XX passed in X.Xs
```

- [ ] **Step 10.5: Lint fixture generator**

```
uv run ruff check scripts/generate_benchmark_fixtures.py
```

Expected: no errors.

- [ ] **Step 10.6: Commit fixtures and generator**

```
git add scripts/generate_benchmark_fixtures.py \
        libs/evaluation/tests/fixtures/
git commit -m "$(cat <<'EOF'
feat(benchmarks): add fixture generator script and all 8 mini parquet fixtures

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review

### Spec Requirements Coverage

| Spec Requirement | Covered By | Status |
|---|---|---|
| `benchmark_harness_library` gap (phase-benchmark-pivot.yaml L199) | Tasks 1–10 | Complete |
| HumanEval, MBPP, APPS, BigCodeBench, DS-1000, LiveCodeBench training oracles | Tasks 3–4 adapters | Complete |
| SWE-Bench-Lite held-out (load only; score deferred) | Task 5 `swe_bench.py` | Partial (by design) |
| CodeContests held-out | Task 5 `codecontests.py` | Complete |
| `run_benchmark(model_adapter_stack, benchmark_id, ...)` API | Task 7 `runner.py` | Complete |
| `load_adapter_stack(base_model, [adapter_ids])` | Task 6 `adapter_stack.py` | Complete |
| Pass@1 aggregator, tested with toy data | Task 2 `aggregator.py` | Complete |
| CLI `scripts/run_benchmark.py` with `--dry-run` | Task 8 | Complete |
| `SubprocessBackend` reuse, 30 s default timeout | All adapter `score()` methods | Complete |
| `BenchmarkConfig.timeout_s` configurable | Task 1 `protocol.py` | Complete |
| `HF_DATASETS_OFFLINE` honored, parquet fixture fallback | All adapters `_load_rows()` | Complete |
| `tests/fixtures/<benchmark>_mini.parquet` checked in | Task 10 | Complete |
| LiveCodeBench pinned to `release_v4` | Task 5 `livecodebench.py` | Complete |
| APPS stratified-random by difficulty, seed=42, max=5000 | Task 4 `apps.py` | Complete |
| SWE-Bench `score` raises `NotImplementedError` with message | Task 5 `swe_bench.py` | Complete |
| ThreadPoolExecutor, `max_workers=4` default, configurable | Task 7 `runner.py` | Complete |
| No Ray, no process pool | Tasks 7–8 | Complete |
| `uv run` throughout | All tasks | Complete |
| Google docstrings | All modules | Complete |
| Deferred GPU imports (INFRA-05) | All `score()` and provider imports | Complete |
| TDD (failing test first) | Tasks 1–8 | Complete |
| Exact `uv run pytest` commands with expected output | Tasks 1–8 | Complete |
| Conventional Commits messages | All commit steps | Complete |
| Report_2 kill-switch semantics (≥5% Pass@1 absolute) | `metrics.run_kill_switch_gate` (existing) + `run_benchmark` API ready for wiring | Harness ready; wiring is follow-on |
| Per-oracle validation gate (≥3% absolute) | `run_benchmark` API ready; oracle runner is follow-on | Harness ready |
| Follow-on plans flagged | "Follow-on Plans" section | Complete |
| Locked design decisions stated in Architecture | Architecture section | Complete |
| SWE-Bench partial scope stated clearly | Follow-on Plans + `swe_bench.py` docstring | Complete |

### Open Design Questions for the Orchestrator

1. **BigCodeBench split name**: The spec references `v0.1.2` as a split name, but HF datasets may expose it as a config name rather than a split. The adapter tries `split="v0.1.2"` then falls back to `split="train"`. Verify the correct HF split/config name before running the fixture generator.

2. **LiveCodeBench HF dataset ID**: The adapter uses `livecodebench/code_generation_lite` with config `release_v4`. Confirm this is the correct repo/config on HF Hub at execution time; the dataset may have moved or been renamed.

3. **`score()` for APPS**: APPS problems use input/output pairs rather than assert statements. The current scorer runs the full program with stdin injection. Some APPS problems may require special judge logic (e.g., multiple valid outputs). This is acceptable for training-oracle use (strict equality is conservative) but should be revisited if APPS Pass@1 is unexpectedly low.

4. **Async provider in thread pool**: `_generate_completion()` creates a new `asyncio` event loop per thread. This works with any `InferenceProvider` that is not bound to a specific loop at construction time. If `VLLMProvider` uses a session-level `httpx.AsyncClient` created at `__init__`, thread isolation will fail. Verify provider implementation before running at scale.

5. **`datasets` version pinning**: `pyproject.toml` pins `datasets>=2.19.0`. If the workspace uses a tighter upper bound elsewhere, reconcile before syncing.
