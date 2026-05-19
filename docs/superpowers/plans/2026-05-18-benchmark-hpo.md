# Benchmark HPO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an Optuna HPO runner that tunes Rune pipeline hyperparameters on failed MBPP problems by driving the real `run_phased_pipeline()`, and replace the bespoke condition-(v) iterative codepath in `run_all_conditions.py` with one that calls the same pipeline.

**Architecture:** A new standalone script `scripts/optimization/run_benchmark_hpo.py` loads failed MBPP problems, splits them 70/30 (tuning/validation), and runs an Optuna TPE study. Each trial samples 4 hyperparameters, applies them to the pipeline via env vars + a temp `PipelineConfig`, runs `run_phased_pipeline()` per sampled problem, scores the output with `MBPPAdapter.score()`, and logs per-trial/per-problem metrics to MLflow. After the study, best params are validated once on the held-out set and saved to the `PipelineConfig`. Separately, `run_all_conditions.py` condition (v) is rewritten to call `run_phased_pipeline()` directly, deleting ~470 lines of reimplemented pipeline logic.

**Tech Stack:** Python 3.12, `uv` workspace, Optuna (TPE sampler, SQLite persistence), MLflow, FastAPI-adjacent libs (`evaluation`, `shared`). The real pipeline lives in `scripts/rune_runner.py` and is **not modified**.

---

## Background & Key Facts (read before starting)

The implementer has zero context for this codebase. These facts are load-bearing:

1. **`run_phased_pipeline()` is `async`.** Signature (`scripts/rune_runner.py:451`):
   ```python
   async def run_phased_pipeline(
       project_prompt: str,
       max_iterations: int = 10,
       checkpoint_path: str | None = None,
       base_model_id: str = DEFAULT_BASE_MODEL,
       device: str = "cpu",
       population_size: int = 2,
       max_phase_iterations: int | None = None,
   ) -> dict[str, Any]: ...
   ```
   Call it with `asyncio.run(run_phased_pipeline(...))` from sync code.

2. **`run_phased_pipeline()` return dict** has keys: `session_id`, `total_iterations`, `project_prompt`, `final_tests_passed`, `phases`, `adapter_dir`, `subtasks` (list of subtask name strings), `adapters` (list of dicts), `accumulated_code` (the final code string), `evolution`.
   - `phases` is a dict with keys `decompose`, `plan`, `code`, `integrate`, and optionally `repair`:
     - `phases["decompose"]` → `{"subtasks": [...], "adapter_id": ..., "iterations": int, "best_score": float}`
     - `phases["plan"]` → `{"plans": {...}, "plan_lengths": {...}, "iterations": int, "best_score": float}`
     - `phases["code"]` → `{"outputs": {...}, "subtask_results": [...], "iterations": int, "passed": int, "total": int}`
     - `phases["integrate"]` → `{"adapter_id": ..., "tests_passed": bool, "iterations": int, "best_score": float}`
     - `phases["repair"]` → present **only if** the diagnose→repair phase ran: `{"iterations": int, "best_score": float, "diagnosed_total": int}`
   - `evolution` → `{"phase_iterations": {...}, "sweeps": {...}, "best_adapters": {...}}`

3. **Hyperparameter wiring** — `run_phased_pipeline()` reads its parameters as follows (it is NOT modified by this plan):
   - `adapter.scaling` — read **only** from `PipelineConfig` via `load_config()` (`rune_runner.py:534,542`). `load_config()` honors the `RUNE_PIPELINE_CONFIG` env var pointing at a JSON file. There is no kwarg and no env var for scaling. **The only way to vary scaling per trial is to write a temp `PipelineConfig` JSON and point `RUNE_PIPELINE_CONFIG` at it.** The spec says scaling is "passed to `run_hypernetwork()`"; that is loose wording — the spec also mandates `rune_runner.py` stays untouched, so the temp-config mechanism is the resolution.
   - `temperature` — `run_phased_pipeline()` does `os.environ.setdefault("RUNE_TEMPERATURE", ...)`. Setting `RUNE_TEMPERATURE` **before** the call wins (setdefault is a no-op when already set).
   - `repetition_penalty` — same pattern, `RUNE_REPETITION_PENALTY`.
   - `max_phase_iterations` — set the `RUNE_MAX_PHASE_ITERATIONS` env var (resolved by `_get_phase_iterations()` in `rune_runner.py`).

4. **Problem selection.** The list of MBPP problems that failed the prior 3-attempt Rune run *was* recovered from MLflow: experiment `paper-table2`, run `a4329a0c64d8419fbab3645616bee90c`, by reconstructing per-problem pass/fail from the step-indexed `v_iter_mbpp_running_pass_at_1` metric (`n_passed(k) = round(running_p1(k) * k)`; deltas are all 0/1; problem order is the deterministic `MBPPAdapter().load_problems()` order). **125 of 257** problems failed — written to `evaluation_results/paper/mbpp_failed_ids.json` (a JSON list of `"mbpp/<task_id>"` strings). The HPO script's working set is selected as:
   - **Default:** the problems listed in `--failed-ids` (defaults to `evaluation_results/paper/mbpp_failed_ids.json`).
   - **Fallback:** if that file does not exist, a seed-deterministic random subset of `--n-problems` (default 127) MBPP problems.
   The working set is then split 70/30 into tuning/validation. Do **not** build a "compute failures" mode — out of scope. Note: only *which* problems failed was recovered, not *why* (per-problem errors were never logged) — that gap is what the new JSONL artifacts in fact 11 close for future runs.

5. **`PipelineConfig`** (`libs/shared/src/shared/pipeline_config.py`): frozen dataclass. `PipelineConfig().override(**{"adapter.scaling": 0.1})` returns a new config with dotted-key overrides. `.save(path)` writes JSON; `.save()` with no arg writes `~/.rune/pipeline_config.json`. `load_config(path)` reads it; with no arg it checks `RUNE_PIPELINE_CONFIG` then the default path.

6. **`MBPPAdapter`** (`libs/evaluation/src/evaluation/benchmarks/mbpp.py`): `load_problems(max_samples=None, seed=42)` returns `list[Problem]`; `score(problem, generation, timeout_s=30)` returns `PassVerdict`. `Problem.problem_id` is `"mbpp/<task_id>"`. Scoring runs the model code + the problem's test asserts in a subprocess sandbox (CPU, no GPU).

7. **`Problem`** dataclass (`evaluation/benchmarks/protocol.py`): `problem_id: str`, `prompt: str`, `test_code: str`, `entry_point: str | None = None`, `metadata: dict = {}`.
   **`PassVerdict`** (frozen): `problem_id: str`, `passed: bool`, `generation: str`, `error: str | None`, `timed_out: bool`.
   **`BenchmarkResult`**: constructed `BenchmarkResult(benchmark_id=..., verdicts=[...])`, exposes a `.pass_at_1` property.

8. **Standalone-script preamble.** Every script under `scripts/` starts with this exact preamble to make workspace packages importable (copied from `scripts/optimization/run_optimization.py`):
   ```python
   import sys
   from pathlib import Path

   sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
   from bootstrap import setup_path  # type: ignore[import-not-found]

   setup_path()
   ```
   For a file in `scripts/optimization/`, `parent.parent` is `scripts/`, which makes both `bootstrap` and `rune_runner` importable.

9. **Test layout.** Tests for optimization scripts live in `scripts/optimization/tests/test_*.py`. The test file adds the script dir to `sys.path` then imports symbols directly. Pattern (from `scripts/optimization/tests/test_training_hpo.py`):
   ```python
   REPO_ROOT = Path(__file__).resolve().parents[3]
   SCRIPT_DIR = REPO_ROOT / "scripts" / "optimization"
   if str(SCRIPT_DIR) not in sys.path:
       sys.path.insert(0, str(SCRIPT_DIR))
   from run_benchmark_hpo import (...)  # noqa: E402
   ```
   MBPP tests should set `HF_DATASETS_OFFLINE=1` (uses the local parquet fixture) — but the tests in this plan construct `Problem` objects directly and do not need MBPP loading.

10. **Quality gates** (run after every task): `uv run ruff check`, `uv run mypy libs/ services/ scripts/`, `uv run pytest scripts/optimization/tests/ scripts/paper/tests/`. Google-style docstrings. ruff line-length 88. No comments unless the *why* is non-obvious. Deferred GPU/heavy imports inside function bodies (INFRA-05).

11. **Per-problem JSONL artifacts go to MLflow, not local disk.** For post-hoc investigation ("what failed and why") every trial — and the validation pass — writes a JSONL where each line is one problem's full record (generated code, untruncated error, attempts, phase metrics). To avoid filling the local disk over a 30-trial run, each JSONL is written to a temp dir, logged via `mlflow.log_artifact()`, then the temp dir is deleted. Nothing per-problem accumulates on the local filesystem. The small summary artifacts (`best_params.json`, `trial_summary.csv`, `validation_results.json`, `study.db`) still live in `--output-dir` — they are bounded in size.
    **Precondition:** `mlflow.log_artifact()` uploads to whatever artifact store the run's tracking server is configured with. For the artifacts to land in S3 (as the user requires), `MLFLOW_TRACKING_URI` must point at the team's tracking server whose artifact root is S3 (per `infra/docker-compose.yml` and `run_all_conditions.py`'s `HPO_S3_PREFIX`). If MLflow falls back to the local SQLite default, artifacts go to `./mlruns/` on local disk instead. `main()` therefore checks `mlflow.get_artifact_uri()` and logs a loud warning if it is not an `s3://` URI — see Task 7.

---

## File Structure

| File | Responsibility | Action |
|------|----------------|--------|
| `scripts/optimization/run_benchmark_hpo.py` | The HPO runner: data split, trial config, pipeline driver, Optuna objective, MLflow logging, artifacts, CLI. | Create |
| `scripts/optimization/tests/test_benchmark_hpo.py` | CPU unit tests for the pure helpers and monkeypatched orchestration. | Create |
| `scripts/paper/run_all_conditions.py` | Replace condition (v): delete the iterative + one-shot codepaths, add `run_condition_rune_phased()`. | Modify |
| `scripts/paper/tests/test_run_all_conditions.py` | Add a test for `run_condition_rune_phased()`. | Modify |

`run_benchmark_hpo.py` is one cohesive file (~350 lines) — it mirrors the existing single-file pattern of `run_optimization.py` and the spec explicitly names one script. Functions are ordered: constants → dataclass → data helpers → trial-config helpers → pipeline driver → Optuna objective + logging → artifact writers → `_build_parser` → `main`.

---

### Task 1: Module skeleton — problem split, subsample & failed-ID loading

**Files:**
- Create: `scripts/optimization/run_benchmark_hpo.py`
- Create: `scripts/optimization/tests/test_benchmark_hpo.py`

- [ ] **Step 1: Write the failing tests**

Create `scripts/optimization/tests/test_benchmark_hpo.py`:

```python
"""CPU unit tests for ``scripts/optimization/run_benchmark_hpo.py``.

GPU paths (run_phased_pipeline) are monkeypatched; pure helpers are
tested directly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = REPO_ROOT / "scripts" / "optimization"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_benchmark_hpo import (  # noqa: E402
    load_failed_ids,
    split_problems,
    subsample_problems,
)


def _problem(pid: str):
    """Build a minimal Problem stand-in with a problem_id."""
    from evaluation.benchmarks.protocol import Problem

    return Problem(problem_id=pid, prompt="p", test_code="assert True")


class TestSplitProblems:
    def test_split_is_seventy_thirty(self) -> None:
        problems = [_problem(f"mbpp/{i}") for i in range(100)]
        tuning, validation = split_problems(problems, seed=42)
        assert len(tuning) == 70
        assert len(validation) == 30

    def test_split_is_deterministic(self) -> None:
        problems = [_problem(f"mbpp/{i}") for i in range(50)]
        a_tune, a_val = split_problems(problems, seed=42)
        b_tune, b_val = split_problems(problems, seed=42)
        assert [p.problem_id for p in a_tune] == [p.problem_id for p in b_tune]
        assert [p.problem_id for p in a_val] == [p.problem_id for p in b_val]

    def test_split_is_disjoint_and_complete(self) -> None:
        problems = [_problem(f"mbpp/{i}") for i in range(127)]
        tuning, validation = split_problems(problems, seed=42)
        ids = {p.problem_id for p in tuning} | {p.problem_id for p in validation}
        assert ids == {p.problem_id for p in problems}
        assert len(tuning) + len(validation) == 127

    def test_custom_fraction(self) -> None:
        problems = [_problem(f"mbpp/{i}") for i in range(10)]
        tuning, validation = split_problems(problems, seed=1, tuning_fraction=0.5)
        assert len(tuning) == 5
        assert len(validation) == 5


class TestLoadFailedIds:
    def test_loads_json_list(self, tmp_path: Path) -> None:
        path = tmp_path / "failed.json"
        path.write_text(json.dumps(["mbpp/1", "mbpp/2", "mbpp/3"]))
        assert load_failed_ids(path) == {"mbpp/1", "mbpp/2", "mbpp/3"}

    def test_missing_file_raises_clear_error(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Failed-ID file not found"):
            load_failed_ids(tmp_path / "nope.json")


class TestSubsampleProblems:
    def test_returns_requested_count(self) -> None:
        problems = [_problem(f"mbpp/{i}") for i in range(200)]
        subset = subsample_problems(problems, n=127, seed=42)
        assert len(subset) == 127

    def test_is_deterministic(self) -> None:
        problems = [_problem(f"mbpp/{i}") for i in range(200)]
        a = subsample_problems(problems, n=50, seed=42)
        b = subsample_problems(problems, n=50, seed=42)
        assert [p.problem_id for p in a] == [p.problem_id for p in b]

    def test_n_larger_than_pool_returns_all(self) -> None:
        problems = [_problem(f"mbpp/{i}") for i in range(10)]
        subset = subsample_problems(problems, n=99, seed=42)
        assert len(subset) == 10
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest scripts/optimization/tests/test_benchmark_hpo.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'run_benchmark_hpo'`.

- [ ] **Step 3: Create the module skeleton with the three helpers**

Create `scripts/optimization/run_benchmark_hpo.py`:

```python
"""Benchmark HPO: Optuna over Rune pipeline parameters on failed MBPP problems.

Tunes ``scaling_factor``, ``temperature``, ``repetition_penalty``, and
``max_phase_iterations`` by running the real ``run_phased_pipeline()`` from
``rune_runner.py`` per problem and measuring MBPP pass rate. Best params are
validated once on a held-out set and saved to the ``PipelineConfig``.

Usage:
    uv run python scripts/optimization/run_benchmark_hpo.py \\
        --hypernet-checkpoint s3://.../checkpoint.pt \\
        --failed-ids evaluation_results/paper/mbpp_failed_ids.json \\
        --n-trials 30 --problems-per-trial 8
"""

from __future__ import annotations

import json
import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bootstrap import setup_path  # type: ignore[import-not-found]

setup_path()

from evaluation.benchmarks.protocol import Problem  # noqa: E402

logger = logging.getLogger("benchmark_hpo")

EXPERIMENT_NAME = "benchmark-hpo-mbpp"
DEFAULT_SEED = 42
TUNING_FRACTION = 0.70
DEFAULT_BASE_MODEL = "Qwen/Qwen3.5-9B"


def split_problems(
    problems: list[Problem],
    seed: int = DEFAULT_SEED,
    tuning_fraction: float = TUNING_FRACTION,
) -> tuple[list[Problem], list[Problem]]:
    """Split problems into (tuning, validation) sets, seed-deterministic.

    Problems are sorted by ``problem_id`` for a stable starting order, then
    shuffled with a seeded RNG. The first ``tuning_fraction`` become the
    tuning set; the remainder are held out for validation.

    Args:
        problems: Problems to split.
        seed: RNG seed controlling the shuffle.
        tuning_fraction: Fraction assigned to the tuning set.

    Returns:
        A ``(tuning, validation)`` tuple of disjoint problem lists.
    """
    ordered = sorted(problems, key=lambda p: p.problem_id)
    random.Random(seed).shuffle(ordered)
    n_tuning = round(len(ordered) * tuning_fraction)
    return ordered[:n_tuning], ordered[n_tuning:]


def load_failed_ids(path: Path) -> set[str]:
    """Load the set of failed MBPP problem IDs from a JSON file.

    The file must be a JSON list of ``problem_id`` strings (e.g.
    ``"mbpp/123"``). Recovering this list from a prior run is the caller's
    responsibility — it is not computed here.

    Args:
        path: Path to the JSON list of failed problem IDs.

    Returns:
        The failed problem IDs as a set.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Failed-ID file not found: {path}. Supply --failed-ids pointing "
            f"at a JSON list of failed MBPP problem_ids (e.g. ['mbpp/12', ...])."
        )
    return {str(i) for i in json.loads(path.read_text())}


def subsample_problems(
    problems: list[Problem],
    n: int,
    seed: int = DEFAULT_SEED,
) -> list[Problem]:
    """Return a seed-deterministic random subset of ``n`` problems.

    If ``n`` is at least the pool size, all problems are returned. Problems
    are sorted by ``problem_id`` before sampling so the result depends only
    on the pool contents and seed, not on input order.

    Args:
        problems: The pool of problems to sample from.
        n: Number of problems to return.
        seed: RNG seed controlling the sample.

    Returns:
        A list of ``min(n, len(problems))`` problems.
    """
    ordered = sorted(problems, key=lambda p: p.problem_id)
    if n >= len(ordered):
        return ordered
    return random.Random(seed).sample(ordered, n)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest scripts/optimization/tests/test_benchmark_hpo.py -v`
Expected: PASS — 9 tests.

- [ ] **Step 5: Lint and type-check**

Run: `uv run ruff check scripts/optimization/run_benchmark_hpo.py scripts/optimization/tests/test_benchmark_hpo.py && uv run mypy scripts/optimization/run_benchmark_hpo.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add scripts/optimization/run_benchmark_hpo.py scripts/optimization/tests/test_benchmark_hpo.py
git commit -m "feat(hpo): add benchmark HPO module skeleton with problem selection"
```

---

### Task 2: `ProblemVerdict` dataclass & `extract_phase_metrics`

**Files:**
- Modify: `scripts/optimization/run_benchmark_hpo.py`
- Modify: `scripts/optimization/tests/test_benchmark_hpo.py`

- [ ] **Step 1: Write the failing tests**

Append to `scripts/optimization/tests/test_benchmark_hpo.py` (and add `extract_phase_metrics`, `ProblemVerdict` to the `from run_benchmark_hpo import (...)` block):

```python
class TestExtractPhaseMetrics:
    def test_full_result_all_phases(self) -> None:
        from run_benchmark_hpo import extract_phase_metrics

        result = {
            "phases": {
                "decompose": {"subtasks": [{"name": "a"}], "best_score": 1.0},
                "plan": {"plans": {"a": "plan text"}},
                "code": {"iterations": 3},
                "integrate": {"tests_passed": True},
                "repair": {"iterations": 1},
            },
            "evolution": {"sweeps": {"phase1": {}, "final": {}}},
            "adapters": [{"id": "x"}, {"id": "y"}, {"id": "z"}],
        }
        m = extract_phase_metrics(result)
        assert m["phase_decompose_ok"] == 1.0
        assert m["phase_plan_ok"] == 1.0
        assert m["phase_code_attempts"] == 3.0
        assert m["phase_integrate_ok"] == 1.0
        assert m["evolution_sweeps"] == 2.0
        assert m["adapters_generated"] == 3.0

    def test_empty_result_defaults_to_zero(self) -> None:
        from run_benchmark_hpo import extract_phase_metrics

        m = extract_phase_metrics({})
        assert m["phase_decompose_ok"] == 0.0
        assert m["phase_plan_ok"] == 0.0
        assert m["phase_code_attempts"] == 0.0
        assert m["phase_integrate_ok"] == 0.0
        assert m["evolution_sweeps"] == 0.0
        assert m["adapters_generated"] == 0.0

    def test_failed_integrate_is_zero(self) -> None:
        from run_benchmark_hpo import extract_phase_metrics

        m = extract_phase_metrics({"phases": {"integrate": {"tests_passed": False}}})
        assert m["phase_integrate_ok"] == 0.0


class TestProblemVerdict:
    def test_fields_default_phase_metrics_to_empty_dict(self) -> None:
        from run_benchmark_hpo import ProblemVerdict

        v = ProblemVerdict(
            problem_id="mbpp/1",
            passed=True,
            code_attempts=2,
            diagnose_fired=False,
            n_subtasks=1,
            wall_time_s=1.5,
            accumulated_code_len=42,
            error="",
        )
        assert v.phase_metrics == {}
        assert v.passed is True
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest scripts/optimization/tests/test_benchmark_hpo.py -v -k "PhaseMetrics or ProblemVerdict"`
Expected: FAIL — `ImportError: cannot import name 'extract_phase_metrics'`.

- [ ] **Step 3: Add the dataclass and helper**

Add `from dataclasses import dataclass, field` and `from typing import Any` to the imports at the top of `run_benchmark_hpo.py`. Append after `load_failed_ids`:

```python
@dataclass
class ProblemVerdict:
    """Outcome of running the pipeline on a single problem.

    Attributes:
        problem_id: The MBPP problem_id evaluated.
        passed: True if the generated code passed all MBPP test asserts.
        code_attempts: Code-phase iterations used (``phases.code.iterations``).
        diagnose_fired: True if the diagnose->repair phase ran.
        n_subtasks: Number of subtasks the decompose phase produced.
        wall_time_s: Wall-clock duration of the pipeline run.
        accumulated_code_len: Length of the final accumulated code string.
        error: Full failure message, or "" if passed. Truncated only at the
            point it is logged as an MLflow param (kept whole in JSONL).
        generation: The final accumulated code the pipeline produced. Kept so
            the per-problem JSONL artifact can show exactly what was generated.
        phase_metrics: Flat phase/evolution metrics from extract_phase_metrics.
    """

    problem_id: str
    passed: bool
    code_attempts: int
    diagnose_fired: bool
    n_subtasks: int
    wall_time_s: float
    accumulated_code_len: int
    error: str
    generation: str = ""
    phase_metrics: dict[str, float] = field(default_factory=dict)


def extract_phase_metrics(result: dict[str, Any]) -> dict[str, float]:
    """Flatten a ``run_phased_pipeline()`` result into MLflow metric values.

    Missing phases default to 0.0 (the phase did not run or did not succeed).

    Args:
        result: The dict returned by ``run_phased_pipeline()``.

    Returns:
        A dict of float metric values keyed by metric name.
    """
    phases = result.get("phases", {})
    decompose = phases.get("decompose", {})
    plan = phases.get("plan", {})
    code = phases.get("code", {})
    integrate = phases.get("integrate", {})
    evolution = result.get("evolution", {})
    return {
        "phase_decompose_ok": float(bool(decompose.get("subtasks"))),
        "phase_plan_ok": float(bool(plan.get("plans"))),
        "phase_code_attempts": float(code.get("iterations", 0)),
        "phase_integrate_ok": float(bool(integrate.get("tests_passed"))),
        "evolution_sweeps": float(len(evolution.get("sweeps", {}))),
        "adapters_generated": float(len(result.get("adapters", []))),
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest scripts/optimization/tests/test_benchmark_hpo.py -v`
Expected: PASS — all tests (13 total).

- [ ] **Step 5: Lint and type-check**

Run: `uv run ruff check scripts/optimization/ && uv run mypy scripts/optimization/run_benchmark_hpo.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add scripts/optimization/run_benchmark_hpo.py scripts/optimization/tests/test_benchmark_hpo.py
git commit -m "feat(hpo): add ProblemVerdict and phase-metric extraction"
```

---

### Task 3: Trial config — `write_trial_pipeline_config` & `apply_trial_env`

**Files:**
- Modify: `scripts/optimization/run_benchmark_hpo.py`
- Modify: `scripts/optimization/tests/test_benchmark_hpo.py`

- [ ] **Step 1: Write the failing tests**

Append to `test_benchmark_hpo.py`:

```python
class TestTrialConfig:
    def test_write_trial_pipeline_config_round_trips_scaling(
        self, tmp_path: Path
    ) -> None:
        from run_benchmark_hpo import write_trial_pipeline_config
        from shared.pipeline_config import load_config

        cfg_path = write_trial_pipeline_config(0.37, tmp_path)
        assert cfg_path.exists()
        loaded = load_config(cfg_path)
        assert loaded.adapter.scaling == pytest.approx(0.37)

    def test_apply_trial_env_sets_all_vars(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import os

        from run_benchmark_hpo import apply_trial_env
        from shared.pipeline_config import load_config

        for var in (
            "RUNE_PIPELINE_CONFIG",
            "RUNE_TEMPERATURE",
            "RUNE_REPETITION_PENALTY",
            "RUNE_MAX_PHASE_ITERATIONS",
        ):
            monkeypatch.delenv(var, raising=False)

        apply_trial_env(
            scaling_factor=0.2,
            temperature=0.45,
            repetition_penalty=1.07,
            max_phase_iterations=4,
            config_dir=tmp_path,
        )
        assert os.environ["RUNE_TEMPERATURE"] == "0.45"
        assert os.environ["RUNE_REPETITION_PENALTY"] == "1.07"
        assert os.environ["RUNE_MAX_PHASE_ITERATIONS"] == "4"
        # RUNE_PIPELINE_CONFIG must point at a config carrying the scaling.
        loaded = load_config(Path(os.environ["RUNE_PIPELINE_CONFIG"]))
        assert loaded.adapter.scaling == pytest.approx(0.2)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest scripts/optimization/tests/test_benchmark_hpo.py -v -k TrialConfig`
Expected: FAIL — `ImportError: cannot import name 'write_trial_pipeline_config'`.

- [ ] **Step 3: Add the two helpers**

Add `import os` to the imports at the top of `run_benchmark_hpo.py`. Append after `extract_phase_metrics`:

```python
def write_trial_pipeline_config(scaling_factor: float, dest_dir: Path) -> Path:
    """Write a temp ``PipelineConfig`` JSON carrying the trial's adapter scaling.

    ``run_phased_pipeline()`` reads ``adapter.scaling`` from the PipelineConfig
    (via ``load_config()``, which honors ``RUNE_PIPELINE_CONFIG``). This temp
    file is the only way to vary scaling per trial without modifying
    ``rune_runner.py``.

    Args:
        scaling_factor: Adapter scaling value for this trial.
        dest_dir: Directory the config file is written into (created if absent).

    Returns:
        Path to the written ``pipeline_config.json``.
    """
    from shared.pipeline_config import PipelineConfig

    dest_dir.mkdir(parents=True, exist_ok=True)
    cfg = PipelineConfig().override(**{"adapter.scaling": scaling_factor})
    return cfg.save(dest_dir / "pipeline_config.json")


def apply_trial_env(
    scaling_factor: float,
    temperature: float,
    repetition_penalty: float,
    max_phase_iterations: int,
    config_dir: Path,
) -> None:
    """Set env vars + temp PipelineConfig so ``run_phased_pipeline()`` sees the trial params.

    - ``scaling_factor`` -> temp PipelineConfig + ``RUNE_PIPELINE_CONFIG``
    - ``temperature`` -> ``RUNE_TEMPERATURE``
    - ``repetition_penalty`` -> ``RUNE_REPETITION_PENALTY``
    - ``max_phase_iterations`` -> ``RUNE_MAX_PHASE_ITERATIONS``

    Env vars are set unconditionally (overwriting any prior trial's values);
    ``run_phased_pipeline()`` uses ``os.environ.setdefault`` so values set here
    take precedence.

    Args:
        scaling_factor: Adapter scaling for this trial.
        temperature: Generation temperature.
        repetition_penalty: Generation repetition penalty.
        max_phase_iterations: Per-phase iteration cap.
        config_dir: Directory for the trial's temp PipelineConfig.
    """
    cfg_path = write_trial_pipeline_config(scaling_factor, config_dir)
    os.environ["RUNE_PIPELINE_CONFIG"] = str(cfg_path)
    os.environ["RUNE_TEMPERATURE"] = str(temperature)
    os.environ["RUNE_REPETITION_PENALTY"] = str(repetition_penalty)
    os.environ["RUNE_MAX_PHASE_ITERATIONS"] = str(max_phase_iterations)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest scripts/optimization/tests/test_benchmark_hpo.py -v`
Expected: PASS — all tests (15 total).

- [ ] **Step 5: Lint and type-check**

Run: `uv run ruff check scripts/optimization/ && uv run mypy scripts/optimization/run_benchmark_hpo.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add scripts/optimization/run_benchmark_hpo.py scripts/optimization/tests/test_benchmark_hpo.py
git commit -m "feat(hpo): add trial pipeline-config and env-var wiring"
```

---

### Task 4: Pipeline driver — `score_pipeline_result`, `run_pipeline_on_problem`, `evaluate_problem_set`

**Files:**
- Modify: `scripts/optimization/run_benchmark_hpo.py`
- Modify: `scripts/optimization/tests/test_benchmark_hpo.py`

`score_pipeline_result` is pure (CPU sandbox only) and fully TDD-tested. `run_pipeline_on_problem` and `evaluate_problem_set` are thin GPU wrappers — they are written but verified by lint/mypy only (no GPU in CI).

- [ ] **Step 1: Write the failing tests**

Append to `test_benchmark_hpo.py`:

```python
class TestScorePipelineResult:
    def test_passing_code_yields_passed_verdict(self) -> None:
        from evaluation.benchmarks.protocol import Problem
        from run_benchmark_hpo import score_pipeline_result

        problem = Problem(
            problem_id="mbpp/add",
            prompt='"""Add two numbers."""',
            test_code="assert add(1, 2) == 3\nassert add(0, 0) == 0",
        )
        result = {
            "accumulated_code": "def add(a, b):\n    return a + b\n",
            "phases": {
                "code": {"iterations": 2},
                "integrate": {"tests_passed": True},
            },
            "subtasks": ["add"],
            "evolution": {"sweeps": {"final": {}}},
            "adapters": [{"id": "a"}],
        }
        verdict = score_pipeline_result(problem, result, wall_time_s=3.0)
        assert verdict.passed is True
        assert verdict.problem_id == "mbpp/add"
        assert verdict.code_attempts == 2
        assert verdict.diagnose_fired is False
        assert verdict.n_subtasks == 1
        assert verdict.wall_time_s == pytest.approx(3.0)
        assert verdict.accumulated_code_len == len(result["accumulated_code"])
        assert verdict.error == ""
        assert verdict.generation == result["accumulated_code"]
        assert verdict.phase_metrics["phase_integrate_ok"] == 1.0

    def test_failing_code_yields_failed_verdict_with_error(self) -> None:
        from evaluation.benchmarks.protocol import Problem
        from run_benchmark_hpo import score_pipeline_result

        problem = Problem(
            problem_id="mbpp/bad",
            prompt='"""Add."""',
            test_code="assert add(1, 2) == 3",
        )
        result = {
            "accumulated_code": "def add(a, b):\n    return a - b\n",
            "phases": {"code": {"iterations": 1}, "repair": {"iterations": 1}},
        }
        verdict = score_pipeline_result(problem, result, wall_time_s=1.0)
        assert verdict.passed is False
        assert verdict.error != ""
        assert verdict.diagnose_fired is True
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest scripts/optimization/tests/test_benchmark_hpo.py -v -k ScorePipelineResult`
Expected: FAIL — `ImportError: cannot import name 'score_pipeline_result'`.

- [ ] **Step 3: Add the three functions**

Add `import asyncio`, `import shutil`, and `import time` to the imports at the top of `run_benchmark_hpo.py`. Append after `apply_trial_env`:

```python
def score_pipeline_result(
    problem: Problem,
    result: dict[str, Any],
    wall_time_s: float,
) -> ProblemVerdict:
    """Score a ``run_phased_pipeline()`` result against the problem's MBPP tests.

    Args:
        problem: The MBPP problem that was run.
        result: The dict returned by ``run_phased_pipeline()``.
        wall_time_s: Wall-clock duration of the pipeline run.

    Returns:
        A ProblemVerdict capturing pass/fail and per-problem diagnostics.
    """
    from evaluation.benchmarks.mbpp import MBPPAdapter

    code = result.get("accumulated_code", "")
    verdict = MBPPAdapter().score(problem, code)
    phase_metrics = extract_phase_metrics(result)
    return ProblemVerdict(
        problem_id=problem.problem_id,
        passed=verdict.passed,
        code_attempts=int(phase_metrics["phase_code_attempts"]),
        diagnose_fired="repair" in result.get("phases", {}),
        n_subtasks=len(result.get("subtasks", [])),
        wall_time_s=wall_time_s,
        accumulated_code_len=len(code),
        error="" if verdict.passed else (verdict.error or ""),
        generation=code,
        phase_metrics=phase_metrics,
    )


def run_pipeline_on_problem(
    problem: Problem,
    hypernet_checkpoint: str,
    base_model: str,
    device: str,
) -> ProblemVerdict:
    """Run the full 5-phase pipeline on one problem and return its verdict.

    Any pipeline exception is caught and converted into a failed verdict so a
    single bad problem cannot abort an Optuna trial. The per-session adapter
    directory is removed afterwards to bound disk usage.

    Args:
        problem: The MBPP problem to solve.
        hypernet_checkpoint: Path (local or ``s3://``) to the hypernetwork.
        base_model: Base model HuggingFace ID.
        device: Device for pipeline computation (e.g. ``"cuda"``).

    Returns:
        A ProblemVerdict for the run.
    """
    from rune_runner import run_phased_pipeline

    start = time.time()
    adapter_dir: str | None = None
    try:
        result = asyncio.run(
            run_phased_pipeline(
                project_prompt=problem.prompt,
                checkpoint_path=hypernet_checkpoint,
                base_model_id=base_model,
                device=device,
            )
        )
        adapter_dir = result.get("adapter_dir")
        return score_pipeline_result(problem, result, time.time() - start)
    except Exception as exc:  # noqa: BLE001 - pipeline failure -> failed verdict
        logger.warning("Pipeline failed on %s: %s", problem.problem_id, exc)
        return ProblemVerdict(
            problem_id=problem.problem_id,
            passed=False,
            code_attempts=0,
            diagnose_fired=False,
            n_subtasks=0,
            wall_time_s=time.time() - start,
            accumulated_code_len=0,
            error=str(exc),
            generation="",
        )
    finally:
        if adapter_dir:
            shutil.rmtree(adapter_dir, ignore_errors=True)


def evaluate_problem_set(
    problems: list[Problem],
    hypernet_checkpoint: str,
    base_model: str,
    device: str,
) -> list[ProblemVerdict]:
    """Run the pipeline on every problem, returning verdicts in input order.

    Args:
        problems: Problems to evaluate.
        hypernet_checkpoint: Path to the hypernetwork checkpoint.
        base_model: Base model HuggingFace ID.
        device: Device for pipeline computation.

    Returns:
        A list of ProblemVerdict, one per input problem.
    """
    return [
        run_pipeline_on_problem(p, hypernet_checkpoint, base_model, device)
        for p in problems
    ]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest scripts/optimization/tests/test_benchmark_hpo.py -v`
Expected: PASS — all tests (17 total).

- [ ] **Step 5: Lint and type-check**

Run: `uv run ruff check scripts/optimization/ && uv run mypy scripts/optimization/run_benchmark_hpo.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add scripts/optimization/run_benchmark_hpo.py scripts/optimization/tests/test_benchmark_hpo.py
git commit -m "feat(hpo): add pipeline driver and result scoring"
```

---

### Task 5: Optuna objective & MLflow logging — `write_verdicts_jsonl`, `log_verdicts_artifact`, `log_trial_metrics`, `make_objective`

**Files:**
- Modify: `scripts/optimization/run_benchmark_hpo.py`
- Modify: `scripts/optimization/tests/test_benchmark_hpo.py`

- [ ] **Step 1: Write the failing tests**

Append to `test_benchmark_hpo.py`. The `_FakeMlflow` helper captures MLflow calls so logging math can be asserted without a tracking server:

```python
class _FakeMlflow:
    """Captures mlflow log calls for assertions (no tracking server needed)."""

    def __init__(self) -> None:
        self.metrics: dict[str, float] = {}
        self.params: dict[str, object] = {}
        self.artifacts: list[tuple[str, str | None]] = []

    def log_metrics(self, d: dict[str, float], step: int | None = None) -> None:
        self.metrics.update(d)

    def log_metric(self, k: str, v: float, step: int | None = None) -> None:
        self.metrics[k] = v

    def log_param(self, k: str, v: object) -> None:
        self.params[k] = v

    def log_params(self, d: dict[str, object]) -> None:
        self.params.update(d)

    def log_artifact(self, path: str, artifact_path: str | None = None) -> None:
        self.artifacts.append((path, artifact_path))


def _verdict(pid: str, passed: bool, attempts: int, diagnosed: bool):
    from run_benchmark_hpo import ProblemVerdict

    return ProblemVerdict(
        problem_id=pid,
        passed=passed,
        code_attempts=attempts,
        diagnose_fired=diagnosed,
        n_subtasks=1,
        wall_time_s=1.0,
        accumulated_code_len=10,
        error="" if passed else "boom",
        phase_metrics={"phase_decompose_ok": 1.0},
    )


class TestVerdictsJsonl:
    def test_write_verdicts_jsonl_one_record_per_line(
        self, tmp_path: Path
    ) -> None:
        from run_benchmark_hpo import write_verdicts_jsonl

        verdicts = [
            _verdict("mbpp/1", True, 2, False),
            _verdict("mbpp/2", False, 1, True),
        ]
        path = tmp_path / "v.jsonl"
        write_verdicts_jsonl(verdicts, path)
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 2
        rec = json.loads(lines[0])
        assert rec["problem_id"] == "mbpp/1"
        assert rec["passed"] is True
        assert "generation" in rec
        assert "phase_metrics" in rec

    def test_log_verdicts_artifact_logs_then_cleans_up(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import run_benchmark_hpo as mod

        captured: dict[str, object] = {}

        def _capture(path: str, artifact_path: str | None = None) -> None:
            captured["existed_during_call"] = Path(path).exists()
            captured["temp_path"] = path
            captured["artifact_path"] = artifact_path

        fake = _FakeMlflow()
        fake.log_artifact = _capture  # type: ignore[method-assign]
        monkeypatch.setattr(mod, "_mlflow", lambda: fake)

        mod.log_verdicts_artifact(
            [_verdict("mbpp/1", True, 1, False)], "trial-001"
        )
        # The artifact must exist at log time and be gone afterwards.
        assert captured["existed_during_call"] is True
        assert captured["artifact_path"] == "verdicts"
        assert not Path(str(captured["temp_path"])).exists()


class TestLogTrialMetrics:
    def test_aggregate_metrics(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import run_benchmark_hpo as mod

        fake = _FakeMlflow()
        monkeypatch.setattr(mod, "_mlflow", lambda: fake)
        verdicts = [
            _verdict("mbpp/1", True, 2, False),
            _verdict("mbpp/2", True, 4, True),
            _verdict("mbpp/3", False, 3, False),
            _verdict("mbpp/4", False, 1, True),
        ]
        mod.log_trial_metrics(verdicts, wall_time_s=12.0)
        assert fake.metrics["pass_rate"] == pytest.approx(0.5)
        assert fake.metrics["n_passed"] == 2
        assert fake.metrics["n_problems"] == 4
        assert fake.metrics["wall_time_s"] == pytest.approx(12.0)
        assert fake.metrics["mean_attempts_used"] == pytest.approx(2.5)
        assert fake.metrics["diagnose_fire_rate"] == pytest.approx(0.5)

    def test_per_problem_metrics_and_error_param(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import run_benchmark_hpo as mod

        fake = _FakeMlflow()
        monkeypatch.setattr(mod, "_mlflow", lambda: fake)
        mod.log_trial_metrics([_verdict("mbpp/9", False, 1, False)], wall_time_s=1.0)
        assert fake.metrics["problem/mbpp/9/passed"] == 0.0
        assert fake.metrics["problem/mbpp/9/code_attempts"] == 1
        assert fake.metrics["problem/mbpp/9/phase_decompose_ok"] == 1.0
        assert fake.params["problem/mbpp/9/error"] == "boom"


class TestMakeObjective:
    def test_objective_returns_pass_rate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import contextlib

        import optuna

        import run_benchmark_hpo as mod

        fake = _FakeMlflow()
        monkeypatch.setattr(mod, "_mlflow", lambda: fake)
        monkeypatch.setattr(
            mod, "_mlflow_run", lambda **kw: contextlib.nullcontext()
        )
        monkeypatch.setattr(
            mod, "apply_trial_env", lambda **kw: None
        )
        monkeypatch.setattr(
            mod,
            "evaluate_problem_set",
            lambda problems, *a, **k: [
                _verdict(p.problem_id, True, 1, False) for p in problems
            ],
        )
        problems = [_problem(f"mbpp/{i}") for i in range(8)]
        objective = mod.make_objective(
            problems,
            hypernet_checkpoint="ckpt",
            base_model="m",
            device="cpu",
            problems_per_trial=4,
            seed=42,
            work_dir=tmp_path,
        )
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=1)
        assert study.best_value == pytest.approx(1.0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest scripts/optimization/tests/test_benchmark_hpo.py -v -k "VerdictsJsonl or LogTrialMetrics or MakeObjective"`
Expected: FAIL — `ImportError: cannot import name 'write_verdicts_jsonl'` / `'log_trial_metrics'`.

- [ ] **Step 3: Add the MLflow helpers, `log_trial_metrics`, and `make_objective`**

Add `from typing import Any, Callable` (merge with the existing `Any` import), `import optuna`, and add `asdict` to the existing `from dataclasses import dataclass, field` line (-> `from dataclasses import asdict, dataclass, field`). Append after `evaluate_problem_set`. The `_mlflow` / `_mlflow_run` indirection exists purely so tests can monkeypatch MLflow without a tracking server:

```python
def _mlflow() -> Any:
    """Return the imported ``mlflow`` module (indirection for test monkeypatching)."""
    import mlflow

    return mlflow


def _mlflow_run(**kwargs: Any) -> Any:
    """Open an MLflow run context (indirection for test monkeypatching)."""
    return _mlflow().start_run(**kwargs)


def write_verdicts_jsonl(verdicts: list[ProblemVerdict], path: Path) -> None:
    """Write one JSON record per verdict to a JSONL file.

    Each line is a full per-problem record — generated code and untruncated
    error included — so a failed run can be investigated after the fact.

    Args:
        verdicts: Verdicts to serialize.
        path: Destination ``.jsonl`` file.
    """
    with path.open("w") as f:
        for v in verdicts:
            f.write(json.dumps(asdict(v)) + "\n")


def log_verdicts_artifact(verdicts: list[ProblemVerdict], label: str) -> None:
    """Write verdicts to a temp JSONL, log it to MLflow, then delete the temp file.

    The JSONL lands in the active MLflow run's artifact store (S3-backed via
    the tracking server) under ``verdicts/``. Nothing per-problem is left on
    the local filesystem — this is what keeps a 30-trial run from filling the
    local disk.

    Args:
        verdicts: Per-problem verdicts to persist for investigation.
        label: Filename stem for the artifact (e.g. ``"trial-007"``).
    """
    import tempfile

    tmp_dir = Path(tempfile.mkdtemp(prefix="hpo_verdicts_"))
    try:
        jsonl_path = tmp_dir / f"{label}.jsonl"
        write_verdicts_jsonl(verdicts, jsonl_path)
        _mlflow().log_artifact(str(jsonl_path), artifact_path="verdicts")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def log_trial_metrics(verdicts: list[ProblemVerdict], wall_time_s: float) -> None:
    """Log aggregate and per-problem metrics for one trial to the active MLflow run.

    Args:
        verdicts: Per-problem verdicts from this trial.
        wall_time_s: Total wall-clock duration of the trial.
    """
    mlflow = _mlflow()
    n = len(verdicts)
    n_passed = sum(v.passed for v in verdicts)
    mlflow.log_metrics(
        {
            "pass_rate": n_passed / n if n else 0.0,
            "n_passed": n_passed,
            "n_problems": n,
            "wall_time_s": wall_time_s,
            "mean_attempts_used": (
                sum(v.code_attempts for v in verdicts) / n if n else 0.0
            ),
            "diagnose_fire_rate": (
                sum(v.diagnose_fired for v in verdicts) / n if n else 0.0
            ),
        }
    )
    for idx, v in enumerate(verdicts):
        pid = v.problem_id
        per_problem = {
            f"problem/{pid}/passed": float(v.passed),
            f"problem/{pid}/code_attempts": v.code_attempts,
            f"problem/{pid}/diagnose_fired": float(v.diagnose_fired),
            f"problem/{pid}/n_subtasks": v.n_subtasks,
            f"problem/{pid}/wall_time_s": v.wall_time_s,
            f"problem/{pid}/accumulated_code_len": v.accumulated_code_len,
        }
        for key, val in v.phase_metrics.items():
            per_problem[f"problem/{pid}/{key}"] = val
        mlflow.log_metrics(per_problem, step=idx)
        if v.error:
            mlflow.log_param(f"problem/{pid}/error", v.error)


def make_objective(
    tuning_problems: list[Problem],
    *,
    hypernet_checkpoint: str,
    base_model: str,
    device: str,
    problems_per_trial: int,
    seed: int,
    work_dir: Path,
) -> Callable[[optuna.Trial], float]:
    """Build the Optuna objective: sample params, run the pipeline, log, score.

    Args:
        tuning_problems: Problems trials sample from.
        hypernet_checkpoint: Path to the hypernetwork checkpoint.
        base_model: Base model HuggingFace ID.
        device: Device for pipeline computation.
        problems_per_trial: Problems sampled (without replacement) per trial.
        seed: Base seed; trial ``k`` samples with ``Random(seed + k)``.
        work_dir: Directory for per-trial temp PipelineConfig files.

    Returns:
        An objective callable returning the trial's MBPP pass rate.
    """

    def objective(trial: optuna.Trial) -> float:
        scaling_factor = trial.suggest_float("scaling_factor", 0.02, 0.50, log=True)
        temperature = trial.suggest_float("temperature", 0.1, 0.7)
        repetition_penalty = trial.suggest_float("repetition_penalty", 1.0, 1.3)
        max_phase_iterations = trial.suggest_int("max_phase_iterations", 2, 6)

        n = min(problems_per_trial, len(tuning_problems))
        trial_problems = random.Random(seed + trial.number).sample(
            tuning_problems, n
        )
        apply_trial_env(
            scaling_factor=scaling_factor,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            max_phase_iterations=max_phase_iterations,
            config_dir=work_dir / f"trial_{trial.number}",
        )

        start = time.time()
        with _mlflow_run(run_name=f"trial-{trial.number:03d}", nested=True):
            _mlflow().log_params(
                {
                    "trial_number": trial.number,
                    "scaling_factor": scaling_factor,
                    "temperature": temperature,
                    "repetition_penalty": repetition_penalty,
                    "max_phase_iterations": max_phase_iterations,
                }
            )
            verdicts = evaluate_problem_set(
                trial_problems, hypernet_checkpoint, base_model, device
            )
            log_trial_metrics(verdicts, time.time() - start)
            log_verdicts_artifact(verdicts, f"trial-{trial.number:03d}")

        n_passed = sum(v.passed for v in verdicts)
        pass_rate = n_passed / len(verdicts) if verdicts else 0.0
        logger.info(
            "Trial %d: pass_rate=%.3f (%d/%d)",
            trial.number,
            pass_rate,
            n_passed,
            len(verdicts),
        )
        return pass_rate

    return objective
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest scripts/optimization/tests/test_benchmark_hpo.py -v`
Expected: PASS — all tests (22 total).

- [ ] **Step 5: Lint and type-check**

Run: `uv run ruff check scripts/optimization/ && uv run mypy scripts/optimization/run_benchmark_hpo.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add scripts/optimization/run_benchmark_hpo.py scripts/optimization/tests/test_benchmark_hpo.py
git commit -m "feat(hpo): add Optuna objective and MLflow trial logging"
```

---

### Task 6: Artifact writers — `save_best_params`, `write_validation_results`, `write_trial_summary`

**Files:**
- Modify: `scripts/optimization/run_benchmark_hpo.py`
- Modify: `scripts/optimization/tests/test_benchmark_hpo.py`

- [ ] **Step 1: Write the failing tests**

Append to `test_benchmark_hpo.py`:

```python
def _study_with_two_trials():
    """Build an in-memory Optuna study with two completed trials."""
    import optuna

    study = optuna.create_study(direction="maximize")

    def obj(trial: "optuna.Trial") -> float:
        trial.suggest_float("scaling_factor", 0.02, 0.5, log=True)
        trial.suggest_float("temperature", 0.1, 0.7)
        trial.suggest_float("repetition_penalty", 1.0, 1.3)
        trial.suggest_int("max_phase_iterations", 2, 6)
        return 0.5 + 0.1 * trial.number

    study.optimize(obj, n_trials=2)
    return study


class TestArtifactWriters:
    def test_save_best_params_writes_json_and_config(self, tmp_path: Path) -> None:
        from run_benchmark_hpo import save_best_params
        from shared.pipeline_config import load_config

        study = _study_with_two_trials()
        config_path = tmp_path / "pipeline_config.json"
        params_path = save_best_params(
            study, out_dir=tmp_path, config_path=config_path
        )
        assert params_path == tmp_path / "best_params.json"
        best = json.loads(params_path.read_text())
        assert set(best) == {
            "scaling_factor",
            "temperature",
            "repetition_penalty",
            "max_phase_iterations",
        }
        loaded = load_config(config_path)
        assert loaded.adapter.scaling == pytest.approx(best["scaling_factor"])
        assert loaded.generation.temperature == pytest.approx(best["temperature"])

    def test_write_validation_results(self, tmp_path: Path) -> None:
        from run_benchmark_hpo import write_validation_results

        verdicts = [
            _verdict("mbpp/1", True, 2, False),
            _verdict("mbpp/2", False, 1, True),
        ]
        path = tmp_path / "validation_results.json"
        write_validation_results(verdicts, path)
        data = json.loads(path.read_text())
        assert data["pass_rate"] == pytest.approx(0.5)
        assert data["problems"]["mbpp/1"]["passed"] is True
        assert data["problems"]["mbpp/2"]["passed"] is False
        assert data["problems"]["mbpp/2"]["error"] == "boom"

    def test_write_trial_summary_csv(self, tmp_path: Path) -> None:
        from run_benchmark_hpo import write_trial_summary

        study = _study_with_two_trials()
        path = tmp_path / "trial_summary.csv"
        write_trial_summary(study, path)
        lines = path.read_text().strip().splitlines()
        assert lines[0].startswith("trial_number,state,pass_rate")
        assert len(lines) == 3  # header + 2 trials
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest scripts/optimization/tests/test_benchmark_hpo.py -v -k ArtifactWriters`
Expected: FAIL — `ImportError: cannot import name 'save_best_params'`.

- [ ] **Step 3: Add the three artifact writers**

Add `import csv` to the imports at the top of `run_benchmark_hpo.py`. Append after `make_objective`:

```python
def save_best_params(
    study: optuna.Study,
    out_dir: Path,
    config_path: Path | None = None,
) -> Path:
    """Write ``best_params.json`` and update the PipelineConfig with best params.

    ``max_phase_iterations`` is recorded in ``best_params.json`` only — the
    PipelineConfig has no field for it.

    Args:
        study: The completed Optuna study.
        out_dir: Directory for ``best_params.json`` (created if absent).
        config_path: PipelineConfig destination; ``None`` writes the default
            ``~/.rune/pipeline_config.json``.

    Returns:
        Path to the written ``best_params.json``.
    """
    from shared.pipeline_config import PipelineConfig

    best = study.best_params
    out_dir.mkdir(parents=True, exist_ok=True)
    params_path = out_dir / "best_params.json"
    params_path.write_text(json.dumps(best, indent=2))

    config = PipelineConfig().override(
        **{
            "adapter.scaling": best["scaling_factor"],
            "generation.temperature": best["temperature"],
            "generation.repetition_penalty": best["repetition_penalty"],
        }
    )
    config.save(config_path)
    return params_path


def write_validation_results(verdicts: list[ProblemVerdict], path: Path) -> None:
    """Write per-problem validation pass/fail results as JSON.

    Args:
        verdicts: Verdicts from the validation-set evaluation.
        path: Destination JSON file.
    """
    n = len(verdicts)
    n_passed = sum(v.passed for v in verdicts)
    data = {
        "pass_rate": n_passed / n if n else 0.0,
        "n_passed": n_passed,
        "n_problems": n,
        "problems": {
            v.problem_id: {
                "passed": v.passed,
                "code_attempts": v.code_attempts,
                "diagnose_fired": v.diagnose_fired,
                "error": v.error,
            }
            for v in verdicts
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def write_trial_summary(study: optuna.Study, path: Path) -> None:
    """Write a CSV of every trial: number, state, pass rate, and params.

    Args:
        study: The completed Optuna study.
        path: Destination CSV file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "trial_number",
                "state",
                "pass_rate",
                "scaling_factor",
                "temperature",
                "repetition_penalty",
                "max_phase_iterations",
            ]
        )
        for t in study.trials:
            writer.writerow(
                [
                    t.number,
                    t.state.name,
                    t.value if t.value is not None else "",
                    t.params.get("scaling_factor", ""),
                    t.params.get("temperature", ""),
                    t.params.get("repetition_penalty", ""),
                    t.params.get("max_phase_iterations", ""),
                ]
            )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest scripts/optimization/tests/test_benchmark_hpo.py -v`
Expected: PASS — all tests (25 total).

- [ ] **Step 5: Lint and type-check**

Run: `uv run ruff check scripts/optimization/ && uv run mypy scripts/optimization/run_benchmark_hpo.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add scripts/optimization/run_benchmark_hpo.py scripts/optimization/tests/test_benchmark_hpo.py
git commit -m "feat(hpo): add best-params, validation-results, and trial-summary writers"
```

---

### Task 7: CLI parser & `main()`

**Files:**
- Modify: `scripts/optimization/run_benchmark_hpo.py`
- Modify: `scripts/optimization/tests/test_benchmark_hpo.py`

- [ ] **Step 1: Write the failing tests**

Append to `test_benchmark_hpo.py` (add `_build_parser` to the top-of-file import block):

```python
class TestBuildParser:
    def test_required_hypernet_checkpoint(self) -> None:
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_defaults(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["--hypernet-checkpoint", "ckpt.pt"])
        assert args.failed_ids == Path("evaluation_results/paper/mbpp_failed_ids.json")
        assert args.n_problems == 127
        assert args.n_trials == 30
        assert args.problems_per_trial == 8
        assert args.seed == 42
        assert args.base_model == "Qwen/Qwen3.5-9B"
        assert args.device == "cuda"
        assert args.study_name.startswith("mbpp-hpo-")
        assert args.db is None

    def test_overrides(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(
            [
                "--hypernet-checkpoint", "ckpt.pt",
                "--failed-ids", "/tmp/f.json",
                "--n-problems", "50",
                "--n-trials", "5",
                "--problems-per-trial", "3",
                "--device", "cpu",
            ]
        )
        assert args.failed_ids == Path("/tmp/f.json")
        assert args.n_problems == 50
        assert args.n_trials == 5
        assert args.problems_per_trial == 3
        assert args.device == "cpu"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest scripts/optimization/tests/test_benchmark_hpo.py -v -k BuildParser`
Expected: FAIL — `ImportError: cannot import name '_build_parser'`.

- [ ] **Step 3: Add `_build_parser` and `main`**

Add `import argparse` and `import time` (if not already present) to the top of `run_benchmark_hpo.py`. Append at the end of the file:

```python
def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for the benchmark HPO runner."""
    from datetime import date

    parser = argparse.ArgumentParser(
        description="Optuna HPO over Rune pipeline params on failed MBPP problems"
    )
    parser.add_argument(
        "--hypernet-checkpoint",
        required=True,
        help="Path (local or s3://) to the trained hypernetwork checkpoint",
    )
    parser.add_argument(
        "--failed-ids",
        type=Path,
        default=Path("evaluation_results/paper/mbpp_failed_ids.json"),
        help="JSON list of MBPP problem_ids to use as the working set. "
        "Defaults to the recovered failed-problem list. If the file is "
        "absent, a random subset of --n-problems is used instead.",
    )
    parser.add_argument(
        "--n-problems",
        type=int,
        default=127,
        help="Size of the random MBPP subset used when --failed-ids is absent",
    )
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--problems-per-trial", type=int, default=8)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--study-name",
        default=f"mbpp-hpo-{date.today():%Y%m%d}",
        help="Optuna study name (also the MLflow parent run name)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation_results/benchmark_hpo"),
        help="Directory for best_params.json, study.db, and result artifacts",
    )
    parser.add_argument(
        "--db",
        default=None,
        help="Optuna storage URI (default: sqlite:///<output-dir>/study.db)",
    )
    return parser


def main() -> None:
    """Entry point: load problems, run the Optuna study, validate, save artifacts."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = _build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    db_uri = args.db or f"sqlite:///{args.output_dir / 'study.db'}"
    work_dir = args.output_dir / "trials"

    from evaluation.benchmarks.mbpp import MBPPAdapter

    all_problems = MBPPAdapter().load_problems()
    if args.failed_ids.exists():
        failed = load_failed_ids(args.failed_ids)
        problems = [p for p in all_problems if p.problem_id in failed]
        if not problems:
            raise SystemExit(
                f"No MBPP problems matched the {len(failed)} failed IDs in "
                f"{args.failed_ids}. Check the file's problem_id format."
            )
        logger.info("Using %d problems from %s", len(problems), args.failed_ids)
    else:
        problems = subsample_problems(all_problems, args.n_problems, args.seed)
        logger.info(
            "%s not found - using a random subset of %d MBPP problems",
            args.failed_ids,
            len(problems),
        )
    tuning, validation = split_problems(problems, seed=args.seed)
    logger.info(
        "Loaded %d failed problems: tuning=%d validation=%d",
        len(problems),
        len(tuning),
        len(validation),
    )

    mlflow = _mlflow()
    mlflow.set_experiment(EXPERIMENT_NAME)

    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        storage=db_uri,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )
    objective = make_objective(
        tuning,
        hypernet_checkpoint=args.hypernet_checkpoint,
        base_model=args.base_model,
        device=args.device,
        problems_per_trial=args.problems_per_trial,
        seed=args.seed,
        work_dir=work_dir,
    )

    t0 = time.time()
    with _mlflow_run(run_name=args.study_name):
        mlflow.log_params(
            {
                "study_name": args.study_name,
                "sampler": "TPE",
                "n_trials": args.n_trials,
                "problems_per_trial": args.problems_per_trial,
                "tuning_set_size": len(tuning),
                "validation_set_size": len(validation),
                "seed": args.seed,
            }
        )
        artifact_uri = mlflow.get_artifact_uri()
        if not artifact_uri.startswith("s3://"):
            logger.warning(
                "MLflow artifact store is %s (not s3://). Per-problem JSONL "
                "artifacts will be written to local disk, not S3. Point "
                "MLFLOW_TRACKING_URI at the S3-backed tracking server.",
                artifact_uri,
            )
        # catch=(Exception,) marks a failed trial FAILED in Optuna (excluded
        # from best-trial selection) rather than scoring it 0.0.
        study.optimize(objective, n_trials=args.n_trials, catch=(Exception,))

        completed = [t for t in study.trials if t.state.name == "COMPLETE"]
        if not completed:
            raise SystemExit(
                "Benchmark HPO produced no successful trials. "
                "Refusing to save a best config."
            )

        best = study.best_params
        logger.info("Best trial #%d: %s", study.best_trial.number, best)

        # Validation pass: best params, evaluated once on the held-out set.
        apply_trial_env(
            scaling_factor=best["scaling_factor"],
            temperature=best["temperature"],
            repetition_penalty=best["repetition_penalty"],
            max_phase_iterations=best["max_phase_iterations"],
            config_dir=work_dir / "validation",
        )
        val_verdicts = evaluate_problem_set(
            validation, args.hypernet_checkpoint, args.base_model, args.device
        )
        val_pass_rate = (
            sum(v.passed for v in val_verdicts) / len(val_verdicts)
            if val_verdicts
            else 0.0
        )
        log_verdicts_artifact(val_verdicts, "validation")

        params_path = save_best_params(study, args.output_dir)
        validation_path = args.output_dir / "validation_results.json"
        summary_path = args.output_dir / "trial_summary.csv"
        write_validation_results(val_verdicts, validation_path)
        write_trial_summary(study, summary_path)

        mlflow.log_metrics(
            {
                "best_trial_number": study.best_trial.number,
                "best_pass_rate": study.best_value,
                "validation_pass_rate": val_pass_rate,
                "tuning_vs_validation_gap": study.best_value - val_pass_rate,
                "total_wall_time_s": time.time() - t0,
            }
        )
        for key, val in best.items():
            mlflow.log_param(f"best/{key}", val)
        for artifact in (params_path, validation_path, summary_path):
            mlflow.log_artifact(str(artifact))
        db_file = db_uri.replace("sqlite:///", "")
        if Path(db_file).exists():
            mlflow.log_artifact(db_file)

    logger.info(
        "HPO complete: best_pass_rate=%.3f validation_pass_rate=%.3f",
        study.best_value,
        val_pass_rate,
    )
    print(f"Best params: {best}")
    print(f"Artifacts written to {args.output_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest scripts/optimization/tests/test_benchmark_hpo.py -v`
Expected: PASS — all tests (28 total).

- [ ] **Step 5: Lint and type-check the whole module**

Run: `uv run ruff check scripts/optimization/ && uv run mypy scripts/optimization/run_benchmark_hpo.py`
Expected: no errors.

- [ ] **Step 6: Smoke-test the CLI surface**

Run: `uv run python scripts/optimization/run_benchmark_hpo.py --help`
Expected: usage text prints, exit 0, lists `--hypernet-checkpoint`, `--failed-ids`, `--n-trials`, etc.

- [ ] **Step 7: Commit**

```bash
git add scripts/optimization/run_benchmark_hpo.py scripts/optimization/tests/test_benchmark_hpo.py
git commit -m "feat(hpo): add CLI parser and main study runner"
```

---

### Task 8: Replace condition (v) in `run_all_conditions.py`

Delete the reimplemented iterative pipeline and the one-shot codepath; add `run_condition_rune_phased()` which calls the real `run_phased_pipeline()`. This is one cohesive change — the file must be import-clean and test-passing at the end of the task.

**Files:**
- Modify: `scripts/paper/run_all_conditions.py`
  - Delete: `run_condition_rune()` (lines 406–423)
  - Delete: `run_condition_rune_iterative()` (lines 426–734)
  - Delete: `_rune_eval_with_pregenerated()` (lines 859–902)
  - Delete: `--rune-one-shot` and `--rune-max-attempts` argparse args
  - Modify: the `cond == "v"` branch in `main()` (lines 1153–1180)
  - Add: `run_condition_rune_phased()`
- Modify: `scripts/paper/tests/test_run_all_conditions.py` (add a test)

- [ ] **Step 1: Write the failing test**

Append to `scripts/paper/tests/test_run_all_conditions.py`:

```python
# ── run_condition_rune_phased ────────────────────────────────────────


class TestRuneConditionPhased:
    def test_scores_each_problem_with_adapter(self) -> None:
        """run_condition_rune_phased drives run_phased_pipeline + adapter.score."""
        from evaluation.benchmarks.protocol import PassVerdict, Problem

        from scripts.paper.run_all_conditions import run_condition_rune_phased

        problems = [
            Problem(problem_id="mbpp/1", prompt="p1", test_code="assert True"),
            Problem(problem_id="mbpp/2", prompt="p2", test_code="assert True"),
        ]

        class _FakeAdapter:
            def load_problems(self, max_samples=None, seed=42):  # noqa: ANN001
                return problems

            def score(self, problem, generation, timeout_s=30):  # noqa: ANN001
                passed = problem.problem_id == "mbpp/1"
                return PassVerdict(
                    problem_id=problem.problem_id,
                    passed=passed,
                    generation=generation,
                    error=None if passed else "fail",
                    timed_out=False,
                )

        async def _fake_pipeline(**kwargs):  # noqa: ANN001, ANN003
            return {"accumulated_code": "code", "adapter_dir": ""}

        with (
            patch(
                "evaluation.benchmarks.runner._import_adapter",
                return_value=_FakeAdapter(),
            ),
            patch.dict(
                "evaluation.benchmarks.runner._ADAPTER_REGISTRY",
                {"mbpp": "x"},
                clear=False,
            ),
            patch("rune_runner.run_phased_pipeline", _fake_pipeline),
        ):
            results = run_condition_rune_phased(
                benchmarks=["mbpp"],
                model="m",
                hypernet_checkpoint="ckpt",
                device="cpu",
            )
        assert results["mbpp"] == pytest.approx(0.5)

    def test_pipeline_exception_is_a_failed_verdict(self) -> None:
        from evaluation.benchmarks.protocol import Problem

        from scripts.paper.run_all_conditions import run_condition_rune_phased

        problems = [Problem(problem_id="mbpp/9", prompt="p", test_code="assert True")]

        class _FakeAdapter:
            def load_problems(self, max_samples=None, seed=42):  # noqa: ANN001
                return problems

            def score(self, problem, generation, timeout_s=30):  # noqa: ANN001
                raise AssertionError("score must not be called on crash path")

        async def _boom(**kwargs):  # noqa: ANN001, ANN003
            raise RuntimeError("pipeline exploded")

        with (
            patch(
                "evaluation.benchmarks.runner._import_adapter",
                return_value=_FakeAdapter(),
            ),
            patch.dict(
                "evaluation.benchmarks.runner._ADAPTER_REGISTRY",
                {"mbpp": "x"},
                clear=False,
            ),
            patch("rune_runner.run_phased_pipeline", _boom),
        ):
            results = run_condition_rune_phased(
                benchmarks=["mbpp"],
                model="m",
                hypernet_checkpoint="ckpt",
                device="cpu",
            )
        assert results["mbpp"] == pytest.approx(0.0)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest scripts/paper/tests/test_run_all_conditions.py -v -k RuneConditionPhased`
Expected: FAIL — `ImportError: cannot import name 'run_condition_rune_phased'`.

- [ ] **Step 3: Delete `run_condition_rune` and `run_condition_rune_iterative`**

In `scripts/paper/run_all_conditions.py`, delete the entire `run_condition_rune()` function (the `def run_condition_rune(` block, lines 406–423) and the entire `run_condition_rune_iterative()` function (the `def run_condition_rune_iterative(` block including all nested closures `_generate_text`, `_generate_adapter`, `_extract_error_summary`, `_safe_id`, and the `HyperLoRA.forward` monkey-patch, lines 426–734).

- [ ] **Step 4: Delete `_rune_eval_with_pregenerated`**

Delete the entire `_rune_eval_with_pregenerated()` function (the `def _rune_eval_with_pregenerated(` block, lines 859–902). Keep `pregenerate_rune_adapters()` and `assemble_table2()` untouched.

- [ ] **Step 5: Add `run_condition_rune_phased`**

Insert this function where `run_condition_rune` was (immediately before `pregenerate_rune_adapters`):

```python
def run_condition_rune_phased(
    benchmarks: list[str],
    model: str,
    hypernet_checkpoint: str,
    device: str = "cuda",
    max_samples: int | None = None,
    checkpoint_dir: Path | str | None = None,
) -> dict[str, float | None]:
    """Condition (v) — Rune: the full 5-phase pipeline per benchmark problem.

    Runs the real ``run_phased_pipeline()`` from ``rune_runner.py`` on each
    problem and scores the accumulated code with the benchmark adapter's
    ``score()`` method. A pipeline exception on one problem becomes a failed
    verdict so the run continues.

    Args:
        benchmarks: Benchmark IDs to evaluate.
        model: Base model HuggingFace ID.
        hypernet_checkpoint: Path (local or ``s3://``) to the hypernetwork.
        device: Device for pipeline computation.
        max_samples: Cap on problems per benchmark.
        checkpoint_dir: Unused; accepted for signature parity with other
            conditions.

    Returns:
        Dict of ``{benchmark: pass_at_1}``.
    """
    import asyncio
    import shutil

    from evaluation.benchmarks.protocol import BenchmarkResult, PassVerdict
    from evaluation.benchmarks.runner import _ADAPTER_REGISTRY, _import_adapter
    from rune_runner import run_phased_pipeline

    try:
        import mlflow

        mlflow_active = mlflow.active_run() is not None
    except Exception:
        mlflow_active = False

    results: dict[str, float | None] = {}
    for bench_id in benchmarks:
        if bench_id not in _ADAPTER_REGISTRY:
            logger.error("Unknown benchmark %s", bench_id)
            results[bench_id] = None
            continue

        adapter = _import_adapter(_ADAPTER_REGISTRY[bench_id])
        problems = adapter.load_problems(max_samples=max_samples, seed=42)
        verdicts: list[PassVerdict] = []
        n_passed = 0

        logger.info("Rune phased: %s — %d problems", bench_id, len(problems))
        for pi, problem in enumerate(problems):
            try:
                result = asyncio.run(
                    run_phased_pipeline(
                        project_prompt=problem.prompt,
                        checkpoint_path=hypernet_checkpoint,
                        base_model_id=model,
                        device=device,
                    )
                )
                verdict = adapter.score(
                    problem, result.get("accumulated_code", "")
                )
                shutil.rmtree(result.get("adapter_dir", ""), ignore_errors=True)
            except Exception as exc:  # noqa: BLE001 - failure -> failed verdict
                logger.error(
                    "Pipeline failed on %s: %s", problem.problem_id, exc
                )
                verdict = PassVerdict(
                    problem_id=problem.problem_id,
                    passed=False,
                    generation="",
                    error=str(exc)[:500],
                    timed_out=False,
                )
            verdicts.append(verdict)
            if verdict.passed:
                n_passed += 1
            running_p1 = n_passed / (pi + 1)
            if mlflow_active:
                mlflow.log_metric(
                    f"v_{bench_id}_running_pass_at_1", running_p1, step=pi + 1
                )
            if (pi + 1) % 10 == 0 or (pi + 1) == len(problems):
                print(
                    f"  [rune/{bench_id}] {pi + 1}/{len(problems)} "
                    f"running Pass@1={running_p1:.2%}"
                )

        bench_result = BenchmarkResult(benchmark_id=bench_id, verdicts=verdicts)
        results[bench_id] = bench_result.pass_at_1
        logger.info(
            "Rune phased %s: Pass@1=%.2f%%", bench_id, bench_result.pass_at_1 * 100
        )
        if mlflow_active:
            mlflow.log_metric(f"v_{bench_id}_pass_at_1", bench_result.pass_at_1)

    return results
```

- [ ] **Step 6: Delete the `--rune-one-shot` and `--rune-max-attempts` CLI args**

In `main()`, delete the two `parser.add_argument(...)` blocks for `--rune-max-attempts` and `--rune-one-shot`. Keep `--rune-adapter-dir` (still used by the `--pregenerate` path) and `--pregenerate`.

- [ ] **Step 7: Rewrite the `cond == "v"` branch in `main()`**

Replace the entire `elif cond == "v":` block (the `if args.rune_one_shot:` / `else:` logic) with:

```python
        elif cond == "v":
            if not args.hypernet_checkpoint:
                print("  SKIPPED: --hypernet-checkpoint required for Rune condition")
                continue
            results = run_condition_rune_phased(
                args.benchmarks,
                args.model,
                hypernet_checkpoint=args.hypernet_checkpoint,
                device=args.device,
                max_samples=args.max_samples,
                checkpoint_dir=cond_ckpt,
            )
```

- [ ] **Step 8: Verify no dangling references**

Run: `grep -rn "run_condition_rune_iterative\|_rune_eval_with_pregenerated\|rune_one_shot\|rune_max_attempts\|run_condition_rune\b" scripts/ tests/ libs/`
Expected: no matches except the new `run_condition_rune_phased` definition and call. If anything else references the deleted symbols, fix that reference (it should not — the earlier grep found only `run_all_conditions.py` itself uses them).

- [ ] **Step 9: Run the tests**

Run: `uv run pytest scripts/paper/tests/test_run_all_conditions.py -v`
Expected: PASS — existing tests plus the 2 new `RuneConditionPhased` tests.

- [ ] **Step 10: Lint and type-check**

Run: `uv run ruff check scripts/paper/run_all_conditions.py scripts/paper/tests/test_run_all_conditions.py && uv run mypy scripts/paper/run_all_conditions.py`
Expected: no errors.

- [ ] **Step 11: Smoke-test the CLI surface**

Run: `uv run python scripts/paper/run_all_conditions.py --help`
Expected: usage prints, exit 0, no `--rune-one-shot` / `--rune-max-attempts` listed, `--pregenerate` and `--rune-adapter-dir` still present.

- [ ] **Step 12: Commit**

```bash
git add scripts/paper/run_all_conditions.py scripts/paper/tests/test_run_all_conditions.py
git commit -m "refactor(paper): replace condition (v) iterative codepath with run_phased_pipeline"
```

---

## Final Verification

After all tasks, run the full gate suite:

```bash
uv run ruff check
uv run mypy libs/ services/ scripts/
uv run pytest scripts/optimization/tests/ scripts/paper/tests/ tests/ -v
```

Expected: ruff clean, mypy clean, all tests pass. Confirm `scripts/rune_runner.py` is unchanged (`git diff --stat main -- scripts/rune_runner.py` shows nothing).

## Spec Coverage Notes

- **Search space** (4 params, log-uniform scaling, int max_phase_iterations) — Task 5 `objective`.
- **70/30 seed-deterministic split** — Task 1 `split_problems`.
- **MLflow study-level + per-trial + per-problem + phase-level metrics** — Tasks 2, 5, 7. Phase keys are emitted under `problem/{id}/phase_*` via `phase_metrics`.
- **Artifacts** (`best_params.json`, `study.db`, `validation_results.json`, `trial_summary.csv`) — Tasks 6, 7. Small, bounded — kept in `--output-dir` and logged to MLflow.
- **Per-problem JSONL artifacts** — Tasks 5, 7. Each trial and the validation pass write a full per-problem JSONL (generated code + untruncated error) to a temp file, log it to MLflow (S3-backed artifact store), then delete the temp file. Keeps the local disk from filling over a long run while preserving the data needed to investigate failures.
- **Deletions** (`run_condition_rune_iterative` + closures + monkey-patch + `--rune-one-shot` + one-shot codepath) — Task 8. `run_condition_rune` and `_rune_eval_with_pregenerated` are also deleted (orphaned once `--rune-one-shot` is removed; user-confirmed).
- **What stays** — `pregenerate_rune_adapters()`, `--pregenerate`, `--rune-adapter-dir`, `run_optimization.py`, `rune_runner.py`, the MBPP description fix — all untouched.
- **Known spec gap** — the spec says scaling is "passed to `run_hypernetwork()`"; because `rune_runner.py` must stay untouched and reads scaling only from `PipelineConfig`, the implementation uses a temp `PipelineConfig` + `RUNE_PIPELINE_CONFIG` (Task 3). Documented in "Background" fact 3.
- **Problem selection** — defaults to the 125 recovered failed problems (`--failed-ids` → `evaluation_results/paper/mbpp_failed_ids.json`, Task 7); falls back to a seed-deterministic random subset of `--n-problems` (`subsample_problems`, Task 1) when that file is absent. The script never computes failures itself (out of scope).
