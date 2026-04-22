# Phase Corpus Producer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `scripts/phase_corpus_producer.py` — a self-distillation harness that runs Rune's full 5-phase pipeline over benchmark problems, filters by Pass@1, bins successful traces into 25 oracle bins (4 phases × 6 benchmarks + 1 pooled diagnose), emits JSONL training manifests, and kicks off per-bin QLoRA training via `train_and_register`, with SQLite resume/checkpoint and STaR rationalization fallback.

**Architecture:** Subprocess-mode invocation of `rune_runner.py` gives clean GPU state and process-level parallelism per problem. The benchmark harness (`evaluation.benchmarks.run_benchmark`) determines Pass@1 on integrated output; its import path resolves after Plan A (`2026-04-22-benchmark-harness-library.md`) lands — all tests in this plan mock it. Per-bin JSONL manifests are drop-in compatible with `trainer.py` / `d2l_data.py` pair schema (`task_id`, `activation_text`, `teacher_text`, optional `metadata`). Adapter registration reuses `train_and_register` with `warm_start="deltacoder"` and `task_type="<phase>_<benchmark>"` or `"diagnose_pooled"`.

**Tech Stack:** Python 3.12, `uv`, `sqlite3` (stdlib), `subprocess`, existing `trainer.train_and_register`, `adapter_registry.registry.AdapterRegistry`, `model_training.d2l_data.save_jsonl`, `pytest` + `unittest.mock`.

**Coordination:** This plan depends on the output interface of `docs/superpowers/plans/2026-04-22-benchmark-harness-library.md` (Plan A). The `run_benchmark` signature is locked by the spec; exact import path resolves after Plan A's first task lands (`from evaluation.benchmarks import run_benchmark`). All tests mock `run_benchmark` so implementation of tasks 1–9 does not block on Plan A.

---

## File Structure

All new files unless noted. No edits to existing modules except where a line range is specified.

```
scripts/
├── phase_corpus_producer.py          # NEW — CLI entrypoint + orchestrator
└── run_phase_corpus.sh               # NEW — batch runner for all 6 benchmarks

libs/shared/src/shared/rune_models.py # EDIT — confirm DIAGNOSE in PipelinePhase (line ~49)

tests/
└── test_phase_corpus_producer.py     # NEW — unit + integration tests (mocked)

libs/corpus-producer/                 # NEW library subpackage (importable by tests without scripts/ on sys.path)
├── pyproject.toml
└── src/corpus_producer/
    ├── __init__.py
    ├── models.py                     # PhaseArtifact dataclass
    ├── pipeline_runner.py            # run_pipeline_for_problem() subprocess wrapper
    ├── success_filter.py             # filter_artifacts() — calls run_benchmark
    ├── binning.py                    # bin_artifacts()
    ├── manifest.py                   # emit_bin_manifest()
    ├── trainer_bridge.py             # invoke_bin_training() — calls train_and_register
    ├── rationalization.py            # star_rationalize() — STaR fallback
    └── progress_db.py                # SQLite resume/checkpoint table

tests/corpus_producer/
├── test_models.py
├── test_pipeline_runner.py
├── test_success_filter.py
├── test_binning.py
├── test_manifest.py
├── test_trainer_bridge.py
├── test_rationalization.py
├── test_progress_db.py
└── test_integration.py               # end-to-end with mocked pipeline + run_benchmark
```

### Dependency Graph

```
Task 1 (models.py) ──┬──► Task 2 (pipeline_runner.py)
                     ├──► Task 3 (success_filter.py)
                     ├──► Task 4 (binning.py)
                     └──► Task 5 (manifest.py)
                              │
Task 6 (progress_db.py) ──┐  │
Task 7 (rationalization)──┤  ▼
                          └► Task 8 (trainer_bridge.py) ──► Task 9 (CLI)
                                                              │
                                                              ▼
                                                         Task 10 (batch shell)
                                                              │
                                                              ▼
                                                         Task 11 (integration test)
```

Tasks 2–5 are **parallel-safe** (disjoint files, depend only on Task 1 types).
Tasks 6–7 are **parallel-safe** with each other and with Tasks 2–5.

---

## Constants (referenced throughout)

```python
MIN_EXAMPLES_PER_BIN = 60
PIPELINE_TIMEOUT_SECS = 300
BENCHMARKS = ["humaneval", "mbpp", "apps", "bigcodebench", "ds_1000", "livecodebench"]
PHASES = ["decompose", "plan", "code", "integrate"]   # non-diagnose phases
DIAGNOSE_BIN_KEY = "diagnose_pooled"
BASE_MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
WARM_START_ALIAS = "deltacoder"
```

---

## Task 1 — `PhaseArtifact` dataclass + library scaffold (sequential)

**Files:**
- `libs/corpus-producer/pyproject.toml` (new)
- `libs/corpus-producer/src/corpus_producer/__init__.py` (new)
- `libs/corpus-producer/src/corpus_producer/models.py` (new)
- `tests/corpus_producer/test_models.py` (new)

### Steps

- [ ] 1.1 Create `libs/corpus-producer/pyproject.toml` wiring the package into the uv workspace.
- [ ] 1.2 Create `libs/corpus-producer/src/corpus_producer/__init__.py` (empty, docstring only).
- [ ] 1.3 Implement `PhaseArtifact` in `models.py`.
- [ ] 1.4 Write failing test.
- [ ] 1.5 Verify test passes.
- [ ] 1.6 Run ruff + mypy.
- [ ] 1.7 Commit.

### `libs/corpus-producer/pyproject.toml`

```toml
[project]
name = "corpus-producer"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "shared",
]

[tool.uv.sources]
shared = { workspace = true }

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

### `libs/corpus-producer/src/corpus_producer/__init__.py`

```python
"""Corpus producer: self-distillation pipeline for phase-aware oracle training."""
```

### `libs/corpus-producer/src/corpus_producer/models.py`

```python
"""Core data models for the phase corpus producer.

PhaseArtifact is the unit of data flowing from a single pipeline run
through filtering, binning, manifest emission, and training invocation.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PhaseArtifact:
    """One phase-boundary output from a single Rune pipeline run.

    Attributes:
        phase: Pipeline phase name: "decompose" | "plan" | "code" | "integrate"
            | "diagnose".
        benchmark: Benchmark identifier (e.g. "humaneval").
        problem_id: Benchmark-specific problem identifier.
        pipeline_run_id: UUID of the pipeline run that produced this artifact.
        input_text: The prompt / trajectory input fed to the model for this phase.
        output_text: The model output for this phase.
        pass_at_1: True if the integrated final code passed the benchmark test suite.
            None if not yet evaluated.
        rationalized: True if this artifact was produced by STaR rationalization
            (ground-truth test hints), not a live pipeline run.
        metadata: Optional extra key/value pairs (e.g. subtask name, layer index).
    """

    phase: str
    benchmark: str
    problem_id: str
    pipeline_run_id: str
    input_text: str
    output_text: str
    pass_at_1: bool | None = None
    rationalized: bool = False
    metadata: dict[str, str] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def bin_key(self) -> str:
        """Return the training bin key for this artifact.

        Returns:
            "diagnose_pooled" for diagnose phase; "<phase>_<benchmark>" otherwise.
        """
        if self.phase == "diagnose":
            return "diagnose_pooled"
        return f"{self.phase}_{self.benchmark}"

    def to_manifest_record(self) -> dict[str, object]:
        """Serialize to a JSONL training manifest record.

        Schema is drop-in compatible with ``model_training.d2l_data`` pair
        records consumed by ``trainer.train_and_register``:
        - ``task_id``: ``"<benchmark>/<problem_id>/<phase>"``
        - ``activation_text``: model input for this phase
        - ``teacher_text``: activation + model output (what the trainer supervises)
        - ``metadata``: provenance fields

        Returns:
            Dict suitable for ``json.dumps`` and JSONL emission.
        """
        task_id = f"{self.benchmark}/{self.problem_id}/{self.phase}"
        activation = self.input_text
        teacher = f"{activation}\n\n{self.output_text}".strip()
        return {
            "task_id": task_id,
            "activation_text": activation,
            "teacher_text": teacher,
            "metadata": {
                "phase": self.phase,
                "benchmark": self.benchmark,
                "problem_id": self.problem_id,
                "pipeline_run_id": self.pipeline_run_id,
                "pass_at_1": self.pass_at_1,
                "rationalized": self.rationalized,
                **self.metadata,
            },
        }
```

### `tests/corpus_producer/test_models.py`

```python
"""Tests for corpus_producer.models.PhaseArtifact."""

import pytest
from corpus_producer.models import PhaseArtifact


def _make(phase: str = "decompose", benchmark: str = "humaneval") -> PhaseArtifact:
    return PhaseArtifact(
        phase=phase,
        benchmark=benchmark,
        problem_id="HumanEval/0",
        pipeline_run_id="run-abc",
        input_text="## Task\nWrite a function.",
        output_text="1. subtask_a — parse input",
        pass_at_1=True,
    )


def test_bin_key_non_diagnose():
    art = _make(phase="decompose", benchmark="humaneval")
    assert art.bin_key() == "decompose_humaneval"


def test_bin_key_diagnose_always_pooled():
    art = _make(phase="diagnose", benchmark="mbpp")
    assert art.bin_key() == "diagnose_pooled"


def test_to_manifest_record_keys():
    art = _make()
    rec = art.to_manifest_record()
    assert "task_id" in rec
    assert "activation_text" in rec
    assert "teacher_text" in rec
    assert "metadata" in rec


def test_to_manifest_record_task_id_format():
    art = _make(phase="plan", benchmark="mbpp")
    art.problem_id = "MBPP/1"
    rec = art.to_manifest_record()
    assert rec["task_id"] == "mbpp/MBPP/1/plan"


def test_to_manifest_record_teacher_contains_output():
    art = _make()
    rec = art.to_manifest_record()
    assert art.output_text in rec["teacher_text"]


def test_to_manifest_record_metadata_provenance():
    art = _make(phase="code", benchmark="apps")
    art.rationalized = True
    rec = art.to_manifest_record()
    assert rec["metadata"]["rationalized"] is True
    assert rec["metadata"]["phase"] == "code"


def test_rationalized_defaults_false():
    art = _make()
    assert art.rationalized is False


def test_metadata_extra_fields_forwarded():
    art = _make()
    art.metadata["subtask"] = "parse_input"
    rec = art.to_manifest_record()
    assert rec["metadata"]["subtask"] == "parse_input"
```

### Expected test output

```
uv run pytest tests/corpus_producer/test_models.py -v
...
8 passed in 0.xx s
```

### Commit

```bash
git commit -m "$(cat <<'EOF'
feat(corpus-producer): add PhaseArtifact dataclass and library scaffold

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2 — Pipeline runner wrapper (parallel-safe after Task 1)

**Files:**
- `libs/corpus-producer/src/corpus_producer/pipeline_runner.py` (new)
- `tests/corpus_producer/test_pipeline_runner.py` (new)

### Steps

- [ ] 2.1 Implement `PipelineRunResult` dataclass.
- [ ] 2.2 Implement `run_pipeline_for_problem()` — subprocess call to `rune_runner.py`.
- [ ] 2.3 Implement `FakePipelineRunner` protocol for test injection.
- [ ] 2.4 Write failing tests with `FakePipelineRunner`.
- [ ] 2.5 Verify tests pass.
- [ ] 2.6 Run ruff + mypy.
- [ ] 2.7 Commit.

### `libs/corpus-producer/src/corpus_producer/pipeline_runner.py`

```python
"""Subprocess wrapper around scripts/rune_runner.py.

Runs one (benchmark, problem) pair through the full 5-phase Rune pipeline
and parses per-phase artifacts from the JSON output file written by the
subprocess.

Design rationale: subprocess mode (vs in-process import) gives clean GPU
state, prevents adapter registry cross-contamination between runs, and
enables future process-level parallelism.
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from corpus_producer.models import PhaseArtifact

logger = logging.getLogger(__name__)

# Path to the rune_runner.py entrypoint, resolved relative to this file.
# Walks up: corpus_producer/pipeline_runner.py -> corpus_producer/ -> src/
#   -> corpus-producer/ -> libs/ -> rune/ -> scripts/
_RUNE_ROOT = Path(__file__).resolve().parents[5]
_RUNE_RUNNER = _RUNE_ROOT / "scripts" / "rune_runner.py"


@dataclass
class PipelineRunResult:
    """Output of a single pipeline run for one (benchmark, problem) pair.

    Attributes:
        run_id: UUID of this pipeline run.
        benchmark: Benchmark identifier.
        problem_id: Problem identifier within the benchmark.
        artifacts: Per-phase PhaseArtifacts, in pipeline order.
        final_code: Integrated final code (Phase 4 output).
        success: True if the subprocess exited 0 and produced artifacts.
        error: Subprocess stderr, populated on failure.
    """

    run_id: str
    benchmark: str
    problem_id: str
    artifacts: list[PhaseArtifact] = field(default_factory=list)
    final_code: str = ""
    success: bool = False
    error: str = ""


@runtime_checkable
class PipelineRunnerProtocol(Protocol):
    """Protocol for injectable pipeline runner (enables test doubles)."""

    def __call__(
        self,
        benchmark: str,
        problem_id: str,
        problem_prompt: str,
        *,
        timeout: int = 300,
        base_model_id: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
    ) -> PipelineRunResult: ...


def _parse_artifacts_from_output(
    run_id: str,
    benchmark: str,
    problem_id: str,
    output_data: dict[str, object],
) -> list[PhaseArtifact]:
    """Extract per-phase PhaseArtifacts from rune_runner JSON output.

    rune_runner returns a dict with keys:
      phase_results.decompose.{output}
      phase_results.plan.{outputs: {subtask: text}}
      phase_results.code.{outputs: {subtask: text}}
      phase_results.integrate.{generated_code}
      phase_results.repair (optional, if diagnose fired)

    We flatten to one artifact per phase. For code/plan we concatenate
    subtask outputs into a single text with subtask headers.

    Args:
        run_id: Pipeline run UUID.
        benchmark: Benchmark identifier.
        problem_id: Problem identifier.
        output_data: Parsed JSON dict from rune_runner.

    Returns:
        List of PhaseArtifact, one per phase that produced output.
    """
    artifacts: list[PhaseArtifact] = []
    phase_results: dict[str, object] = output_data.get("phase_results", {})  # type: ignore[assignment]

    def _art(phase: str, inp: str, out: str, meta: dict[str, str] | None = None) -> PhaseArtifact:
        return PhaseArtifact(
            phase=phase,
            benchmark=benchmark,
            problem_id=problem_id,
            pipeline_run_id=run_id,
            input_text=inp,
            output_text=out,
            metadata=meta or {},
        )

    # DECOMPOSE
    decompose = phase_results.get("decompose", {})  # type: ignore[union-attr]
    if isinstance(decompose, dict) and decompose.get("output"):
        artifacts.append(_art("decompose", inp=output_data.get("project_prompt", ""), out=str(decompose["output"])))  # type: ignore[arg-type]

    # PLAN (concatenate subtask plans)
    plan = phase_results.get("plan", {})  # type: ignore[union-attr]
    if isinstance(plan, dict) and plan.get("plans"):
        plans_dict: dict[str, str] = plan["plans"]  # type: ignore[assignment]
        out_parts = [f"### {k}\n{v}" for k, v in plans_dict.items()]
        artifacts.append(_art("plan", inp=str(decompose.get("output", "")), out="\n\n".join(out_parts)))

    # CODE (concatenate subtask code)
    code = phase_results.get("code", {})  # type: ignore[union-attr]
    if isinstance(code, dict) and code.get("outputs"):
        code_dict: dict[str, str] = code["outputs"]  # type: ignore[assignment]
        out_parts = [f"### {k}\n{v}" for k, v in code_dict.items()]
        artifacts.append(_art("code", inp=str(plan.get("plans", "")), out="\n\n".join(out_parts)))

    # INTEGRATE
    integrate = phase_results.get("integrate", {})  # type: ignore[union-attr]
    if isinstance(integrate, dict):
        final = str(output_data.get("final_code", ""))
        artifacts.append(_art("integrate", inp=str(code.get("outputs", "")), out=final))

    # DIAGNOSE (optional — only if repair loop fired)
    repair = phase_results.get("repair", {})  # type: ignore[union-attr]
    if isinstance(repair, dict) and repair.get("diagnosis"):
        artifacts.append(_art("diagnose", inp=str(integrate.get("stderr", "")), out=str(repair["diagnosis"])))

    return artifacts


def run_pipeline_for_problem(
    benchmark: str,
    problem_id: str,
    problem_prompt: str,
    *,
    timeout: int = 300,
    base_model_id: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
) -> PipelineRunResult:
    """Run the full 5-phase Rune pipeline for one problem via subprocess.

    Writes a temporary JSON output file that rune_runner populates, then
    reads and parses artifacts from it.

    Args:
        benchmark: Benchmark identifier (used for artifact tagging only).
        problem_id: Problem identifier (used for artifact tagging).
        problem_prompt: Full problem text passed to rune_runner --project.
        timeout: Process timeout in seconds (default 300).
        base_model_id: Base model HF repo id.

    Returns:
        PipelineRunResult with artifacts and success flag.
    """
    run_id = str(uuid.uuid4())

    with tempfile.TemporaryDirectory(prefix="rune_run_") as tmpdir:
        output_json = Path(tmpdir) / "pipeline_output.json"

        cmd = [
            "uv",
            "run",
            str(_RUNE_RUNNER),
            "--project",
            problem_prompt,
            "--output-json",
            str(output_json),
            "--base-model",
            base_model_id,
            "--max-phase-iterations",
            "1",  # one iteration per phase for corpus production
        ]

        logger.info("Running pipeline for %s/%s (run_id=%s)", benchmark, problem_id, run_id)
        try:
            proc = subprocess.run(
                cmd,
                timeout=timeout,
                capture_output=True,
                text=True,
                cwd=str(_RUNE_ROOT),
            )
        except subprocess.TimeoutExpired:
            logger.warning("Pipeline timed out for %s/%s", benchmark, problem_id)
            return PipelineRunResult(
                run_id=run_id,
                benchmark=benchmark,
                problem_id=problem_id,
                error=f"Timed out after {timeout}s",
            )
        except Exception as exc:
            logger.exception("Subprocess error for %s/%s", benchmark, problem_id)
            return PipelineRunResult(
                run_id=run_id,
                benchmark=benchmark,
                problem_id=problem_id,
                error=str(exc),
            )

        if proc.returncode != 0:
            logger.warning(
                "Pipeline failed for %s/%s (exit %d): %s",
                benchmark,
                problem_id,
                proc.returncode,
                proc.stderr[-500:],
            )
            return PipelineRunResult(
                run_id=run_id,
                benchmark=benchmark,
                problem_id=problem_id,
                error=proc.stderr[-500:],
            )

        if not output_json.exists():
            return PipelineRunResult(
                run_id=run_id,
                benchmark=benchmark,
                problem_id=problem_id,
                error="rune_runner did not write output JSON",
            )

        try:
            output_data: dict[str, object] = json.loads(output_json.read_text())
        except json.JSONDecodeError as exc:
            return PipelineRunResult(
                run_id=run_id,
                benchmark=benchmark,
                problem_id=problem_id,
                error=f"JSON parse error: {exc}",
            )

        artifacts = _parse_artifacts_from_output(run_id, benchmark, problem_id, output_data)
        final_code = str(output_data.get("final_code", ""))

        return PipelineRunResult(
            run_id=run_id,
            benchmark=benchmark,
            problem_id=problem_id,
            artifacts=artifacts,
            final_code=final_code,
            success=True,
        )
```

### `tests/corpus_producer/test_pipeline_runner.py`

```python
"""Tests for corpus_producer.pipeline_runner."""

from __future__ import annotations

from corpus_producer.models import PhaseArtifact
from corpus_producer.pipeline_runner import (
    PipelineRunResult,
    PipelineRunnerProtocol,
    _parse_artifacts_from_output,
    run_pipeline_for_problem,
)


def _fake_output(with_diagnose: bool = False) -> dict[str, object]:
    out: dict[str, object] = {
        "project_prompt": "Write a sort function.",
        "final_code": "def solution(): pass",
        "phase_results": {
            "decompose": {"output": "1. parse — parse input"},
            "plan": {"plans": {"parse": "Use argparse"}},
            "code": {"outputs": {"parse": "def parse(): pass"}},
            "integrate": {"generated_code": "def solution(): pass", "stderr": ""},
        },
    }
    if with_diagnose:
        out["phase_results"]["repair"] = {"diagnosis": "NameError on line 3"}  # type: ignore[index]
    return out


def test_parse_artifacts_returns_four_phases():
    arts = _parse_artifacts_from_output("run-1", "humaneval", "HumanEval/0", _fake_output())
    phases = [a.phase for a in arts]
    assert "decompose" in phases
    assert "plan" in phases
    assert "code" in phases
    assert "integrate" in phases


def test_parse_artifacts_diagnose_present_when_repair_fired():
    arts = _parse_artifacts_from_output("run-1", "humaneval", "HumanEval/0", _fake_output(with_diagnose=True))
    phases = [a.phase for a in arts]
    assert "diagnose" in phases


def test_parse_artifacts_no_diagnose_when_no_repair():
    arts = _parse_artifacts_from_output("run-1", "humaneval", "HumanEval/0", _fake_output())
    phases = [a.phase for a in arts]
    assert "diagnose" not in phases


def test_parse_artifacts_benchmark_and_problem_tagged():
    arts = _parse_artifacts_from_output("run-1", "mbpp", "MBPP/42", _fake_output())
    for art in arts:
        assert art.benchmark == "mbpp"
        assert art.problem_id == "MBPP/42"


def test_parse_artifacts_pipeline_run_id_set():
    arts = _parse_artifacts_from_output("run-xyz", "humaneval", "HumanEval/0", _fake_output())
    for art in arts:
        assert art.pipeline_run_id == "run-xyz"


class FakeRunner:
    """Test double satisfying PipelineRunnerProtocol."""

    def __init__(self, result: PipelineRunResult) -> None:
        self._result = result

    def __call__(
        self,
        benchmark: str,
        problem_id: str,
        problem_prompt: str,
        *,
        timeout: int = 300,
        base_model_id: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
    ) -> PipelineRunResult:
        return self._result


def test_fake_runner_satisfies_protocol():
    result = PipelineRunResult(run_id="r", benchmark="humaneval", problem_id="HE/0")
    runner = FakeRunner(result)
    assert isinstance(runner, PipelineRunnerProtocol)


def test_pipeline_run_result_defaults():
    r = PipelineRunResult(run_id="r", benchmark="b", problem_id="p")
    assert r.success is False
    assert r.artifacts == []
    assert r.final_code == ""
```

### Expected test output

```
uv run pytest tests/corpus_producer/test_pipeline_runner.py -v
...
9 passed in 0.xx s
```

### Commit

```bash
git commit -m "$(cat <<'EOF'
feat(corpus-producer): add pipeline runner subprocess wrapper + FakePipelineRunner protocol

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3 — Success filter (parallel-safe after Task 1)

**Files:**
- `libs/corpus-producer/src/corpus_producer/success_filter.py` (new)
- `tests/corpus_producer/test_success_filter.py` (new)

### Steps

- [ ] 3.1 Implement `filter_artifacts()` calling `run_benchmark`.
- [ ] 3.2 Write failing tests with `unittest.mock.patch` on `run_benchmark`.
- [ ] 3.3 Verify tests pass.
- [ ] 3.4 Run ruff + mypy.
- [ ] 3.5 Commit.

### `libs/corpus-producer/src/corpus_producer/success_filter.py`

```python
"""Pass@1 success filter for phase corpus artifacts.

Calls the benchmark harness to evaluate the integrated final code and
marks every artifact in the run with the result. Keeps artifacts according
to the filtering policy:

  Pass@1 == 1.0  ->  keep ALL phase artifacts (they all contributed to success)
  Pass@1 == 0.0  ->  keep ONLY diagnose artifacts where diagnose->repair SUCCEEDED
  Any other rate ->  discard (partial pass means the code is not fully correct)

This implements the STaR-style self-distillation principle: only traces that
produced correct outputs are used as training data.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from corpus_producer.models import PhaseArtifact

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def filter_artifacts(
    artifacts: list[PhaseArtifact],
    final_code: str,
    benchmark: str,
    problem_id: str,
) -> list[PhaseArtifact]:
    """Evaluate Pass@1 and return only training-worthy artifacts.

    Imports ``run_benchmark`` lazily so this module is CPU-importable before
    Plan A lands. In tests the import is patched via
    ``unittest.mock.patch("corpus_producer.success_filter.run_benchmark")``.

    Args:
        artifacts: All per-phase artifacts from a single pipeline run.
        final_code: Integrated output (Phase 4) to evaluate.
        benchmark: Benchmark identifier.
        problem_id: Problem identifier used to scope the benchmark run.

    Returns:
        Filtered list of PhaseArtifact with ``pass_at_1`` field set.
    """
    # Lazy import — resolves after Plan A lands; patched in tests.
    from evaluation.benchmarks import run_benchmark  # type: ignore[import]

    result = run_benchmark(
        model_adapter_stack=("__direct_eval__", []),
        benchmark_id=benchmark,
        problem_ids=[problem_id],
        max_samples=1,
    )

    verdict = result.per_problem.get(problem_id)
    passed = verdict.passed if verdict is not None else False

    logger.info(
        "Pass@1 for %s/%s: %s",
        benchmark,
        problem_id,
        "PASS" if passed else "FAIL",
    )

    kept: list[PhaseArtifact] = []

    if passed:
        # All phases contributed to a successful outcome.
        for art in artifacts:
            art.pass_at_1 = True
            kept.append(art)
    else:
        # Only keep diagnose artifacts where the repair succeeded.
        # The repair success flag is encoded in the artifact metadata
        # by the pipeline runner when the diagnose->repair loop recovers.
        for art in artifacts:
            art.pass_at_1 = False
            if art.phase == "diagnose" and art.metadata.get("repair_succeeded") == "true":
                kept.append(art)

    return kept
```

### `tests/corpus_producer/test_success_filter.py`

```python
"""Tests for corpus_producer.success_filter."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from corpus_producer.models import PhaseArtifact
from corpus_producer.success_filter import filter_artifacts


def _arts(phases: list[str], benchmark: str = "humaneval") -> list[PhaseArtifact]:
    return [
        PhaseArtifact(
            phase=p,
            benchmark=benchmark,
            problem_id="HE/0",
            pipeline_run_id="run-1",
            input_text="prompt",
            output_text=f"output for {p}",
        )
        for p in phases
    ]


def _mock_verdict(passed: bool) -> MagicMock:
    verdict = MagicMock()
    verdict.passed = passed
    result = MagicMock()
    result.per_problem = {"HE/0": verdict}
    return result


@patch("corpus_producer.success_filter.run_benchmark")
def test_pass_keeps_all_phases(mock_rb: MagicMock) -> None:
    mock_rb.return_value = _mock_verdict(True)
    arts = _arts(["decompose", "plan", "code", "integrate"])
    kept = filter_artifacts(arts, "def f(): pass", "humaneval", "HE/0")
    assert len(kept) == 4
    assert all(a.pass_at_1 is True for a in kept)


@patch("corpus_producer.success_filter.run_benchmark")
def test_fail_drops_non_diagnose(mock_rb: MagicMock) -> None:
    mock_rb.return_value = _mock_verdict(False)
    arts = _arts(["decompose", "plan", "code", "integrate"])
    kept = filter_artifacts(arts, "", "humaneval", "HE/0")
    assert len(kept) == 0


@patch("corpus_producer.success_filter.run_benchmark")
def test_fail_keeps_diagnose_when_repair_succeeded(mock_rb: MagicMock) -> None:
    mock_rb.return_value = _mock_verdict(False)
    arts = _arts(["decompose", "diagnose"])
    arts[1].metadata["repair_succeeded"] = "true"
    kept = filter_artifacts(arts, "", "humaneval", "HE/0")
    assert len(kept) == 1
    assert kept[0].phase == "diagnose"


@patch("corpus_producer.success_filter.run_benchmark")
def test_fail_drops_diagnose_when_repair_not_succeeded(mock_rb: MagicMock) -> None:
    mock_rb.return_value = _mock_verdict(False)
    arts = _arts(["diagnose"])
    # no repair_succeeded metadata
    kept = filter_artifacts(arts, "", "humaneval", "HE/0")
    assert len(kept) == 0


@patch("corpus_producer.success_filter.run_benchmark")
def test_run_benchmark_called_with_correct_args(mock_rb: MagicMock) -> None:
    mock_rb.return_value = _mock_verdict(True)
    arts = _arts(["integrate"])
    filter_artifacts(arts, "code", "mbpp", "MBPP/5")
    mock_rb.assert_called_once()
    kwargs = mock_rb.call_args
    assert kwargs[1]["benchmark_id"] == "mbpp" or kwargs[0][1] == "mbpp"
```

### Expected test output

```
uv run pytest tests/corpus_producer/test_success_filter.py -v
...
5 passed in 0.xx s
```

### Commit

```bash
git commit -m "$(cat <<'EOF'
feat(corpus-producer): add Pass@1 success filter with run_benchmark integration

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4 — Binning (parallel-safe after Task 1)

**Files:**
- `libs/corpus-producer/src/corpus_producer/binning.py` (new)
- `tests/corpus_producer/test_binning.py` (new)

### Steps

- [ ] 4.1 Implement `bin_artifacts()`.
- [ ] 4.2 Write failing tests.
- [ ] 4.3 Verify tests pass.
- [ ] 4.4 Run ruff + mypy.
- [ ] 4.5 Commit.

### `libs/corpus-producer/src/corpus_producer/binning.py`

```python
"""Bin PhaseArtifacts into per-(phase, benchmark) oracle bins.

Bins are keyed:
  "<phase>_<benchmark>"  for decompose / plan / code / integrate
  "diagnose_pooled"      for diagnose (all benchmarks pooled per spec q4 decision)
"""

from __future__ import annotations

from collections import defaultdict

from corpus_producer.models import PhaseArtifact

DIAGNOSE_BIN_KEY = "diagnose_pooled"


def bin_artifacts(
    artifacts: list[PhaseArtifact],
) -> dict[str, list[PhaseArtifact]]:
    """Group artifacts into oracle training bins.

    Args:
        artifacts: Any mix of PhaseArtifacts (may span multiple runs / benchmarks).

    Returns:
        Dict mapping bin key -> list of artifacts in that bin. Diagnose
        artifacts from all benchmarks share the key ``"diagnose_pooled"``.
        Empty bins are not included.
    """
    bins: dict[str, list[PhaseArtifact]] = defaultdict(list)
    for art in artifacts:
        bins[art.bin_key()].append(art)
    return dict(bins)


def expected_bin_keys(
    benchmarks: list[str] | None = None,
) -> list[str]:
    """Return the complete list of expected bin keys for the 25-oracle target.

    Args:
        benchmarks: Override list of benchmark ids. Defaults to the 6 spec benchmarks.

    Returns:
        List of 25 bin keys: 4 phases × len(benchmarks) + 1 diagnose_pooled.
    """
    if benchmarks is None:
        benchmarks = ["humaneval", "mbpp", "apps", "bigcodebench", "ds_1000", "livecodebench"]
    keys = [
        f"{phase}_{bm}"
        for phase in ("decompose", "plan", "code", "integrate")
        for bm in benchmarks
    ]
    keys.append(DIAGNOSE_BIN_KEY)
    return keys
```

### `tests/corpus_producer/test_binning.py`

```python
"""Tests for corpus_producer.binning."""

from corpus_producer.binning import DIAGNOSE_BIN_KEY, bin_artifacts, expected_bin_keys
from corpus_producer.models import PhaseArtifact


def _art(phase: str, benchmark: str = "humaneval") -> PhaseArtifact:
    return PhaseArtifact(
        phase=phase,
        benchmark=benchmark,
        problem_id="P/0",
        pipeline_run_id="r",
        input_text="in",
        output_text="out",
    )


def test_bin_artifacts_groups_by_phase_benchmark():
    arts = [_art("decompose", "humaneval"), _art("plan", "humaneval"), _art("decompose", "mbpp")]
    bins = bin_artifacts(arts)
    assert "decompose_humaneval" in bins
    assert "plan_humaneval" in bins
    assert "decompose_mbpp" in bins
    assert len(bins["decompose_humaneval"]) == 1


def test_bin_artifacts_diagnose_pooled():
    arts = [_art("diagnose", "humaneval"), _art("diagnose", "mbpp")]
    bins = bin_artifacts(arts)
    assert DIAGNOSE_BIN_KEY in bins
    assert len(bins[DIAGNOSE_BIN_KEY]) == 2
    assert "diagnose_humaneval" not in bins


def test_bin_artifacts_empty_list():
    assert bin_artifacts([]) == {}


def test_expected_bin_keys_count():
    keys = expected_bin_keys()
    assert len(keys) == 25  # 4 phases × 6 benchmarks + 1 diagnose_pooled


def test_expected_bin_keys_contains_diagnose_pooled():
    assert DIAGNOSE_BIN_KEY in expected_bin_keys()


def test_expected_bin_keys_custom_benchmarks():
    keys = expected_bin_keys(["humaneval", "mbpp"])
    # 4 phases × 2 benchmarks + 1 diagnose_pooled = 9
    assert len(keys) == 9
```

### Expected test output

```
uv run pytest tests/corpus_producer/test_binning.py -v
...
6 passed in 0.xx s
```

### Commit

```bash
git commit -m "$(cat <<'EOF'
feat(corpus-producer): add artifact binning into 25 oracle bins

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5 — Manifest emission (parallel-safe after Task 1)

**Files:**
- `libs/corpus-producer/src/corpus_producer/manifest.py` (new)
- `tests/corpus_producer/test_manifest.py` (new)

### Steps

- [ ] 5.1 Implement `emit_bin_manifest()`.
- [ ] 5.2 Write failing tests.
- [ ] 5.3 Verify tests pass.
- [ ] 5.4 Run ruff + mypy.
- [ ] 5.5 Commit.

### `libs/corpus-producer/src/corpus_producer/manifest.py`

```python
"""JSONL training manifest emission for oracle bins.

Each bin's manifest is a JSONL file where every line is a training record
compatible with ``model_training.d2l_data.pairs_to_chat_messages`` and
the ``trainer.train_and_register(dataset_path=...)`` entry point.

Schema (per record):
  task_id         str   "<benchmark>/<problem_id>/<phase>"
  activation_text str   phase input (what the model sees as context)
  teacher_text    str   activation + phase output (supervised target)
  metadata        dict  provenance: phase, benchmark, problem_id, pipeline_run_id,
                        pass_at_1, rationalized, + any extra fields from artifact
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from corpus_producer.models import PhaseArtifact

logger = logging.getLogger(__name__)


def emit_bin_manifest(
    bin_key: str,
    artifacts: list[PhaseArtifact],
    out_dir: Path | str,
) -> Path:
    """Write a JSONL training manifest for one oracle bin.

    Args:
        bin_key: Oracle bin identifier (e.g. "decompose_humaneval",
            "diagnose_pooled").
        artifacts: All PhaseArtifacts for this bin.
        out_dir: Directory to write the manifest into. Created if absent.

    Returns:
        Path to the written ``.jsonl`` file.

    Raises:
        ValueError: If ``artifacts`` is empty.
    """
    if not artifacts:
        raise ValueError(f"Cannot emit manifest for empty bin {bin_key!r}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / f"{bin_key}.jsonl"

    records = [art.to_manifest_record() for art in artifacts]

    with manifest_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info(
        "Wrote manifest %s: %d records -> %s",
        bin_key,
        len(records),
        manifest_path,
    )
    return manifest_path


def load_bin_manifest(path: Path | str) -> list[dict[str, object]]:
    """Read a previously-emitted manifest back into memory.

    Args:
        path: Path to the ``.jsonl`` manifest file.

    Returns:
        List of record dicts, one per non-empty line.
    """
    src = Path(path)
    records: list[dict[str, object]] = []
    for line in src.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records
```

### `tests/corpus_producer/test_manifest.py`

```python
"""Tests for corpus_producer.manifest."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from corpus_producer.manifest import emit_bin_manifest, load_bin_manifest
from corpus_producer.models import PhaseArtifact


def _art(phase: str = "decompose", benchmark: str = "humaneval", pid: str = "HE/0") -> PhaseArtifact:
    return PhaseArtifact(
        phase=phase,
        benchmark=benchmark,
        problem_id=pid,
        pipeline_run_id="run-1",
        input_text="## Task\nSort a list.",
        output_text="1. parse — read input",
        pass_at_1=True,
    )


def test_emit_creates_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = emit_bin_manifest("decompose_humaneval", [_art()], tmpdir)
        assert p.exists()
        assert p.name == "decompose_humaneval.jsonl"


def test_emit_correct_line_count():
    with tempfile.TemporaryDirectory() as tmpdir:
        arts = [_art(pid=f"HE/{i}") for i in range(5)]
        p = emit_bin_manifest("decompose_humaneval", arts, tmpdir)
        lines = [l for l in p.read_text().splitlines() if l.strip()]
        assert len(lines) == 5


def test_emit_record_schema():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = emit_bin_manifest("decompose_humaneval", [_art()], tmpdir)
        rec = json.loads(p.read_text().splitlines()[0])
        assert "task_id" in rec
        assert "activation_text" in rec
        assert "teacher_text" in rec
        assert "metadata" in rec


def test_emit_raises_on_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="empty bin"):
            emit_bin_manifest("decompose_humaneval", [], tmpdir)


def test_emit_diagnose_pooled_bin_key():
    with tempfile.TemporaryDirectory() as tmpdir:
        art = _art(phase="diagnose")
        p = emit_bin_manifest("diagnose_pooled", [art], tmpdir)
        assert p.name == "diagnose_pooled.jsonl"


def test_load_bin_manifest_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        arts = [_art(pid=f"HE/{i}") for i in range(3)]
        p = emit_bin_manifest("decompose_humaneval", arts, tmpdir)
        records = load_bin_manifest(p)
        assert len(records) == 3
        assert all("task_id" in r for r in records)


def test_emit_creates_parent_dirs():
    with tempfile.TemporaryDirectory() as tmpdir:
        nested = Path(tmpdir) / "a" / "b" / "c"
        p = emit_bin_manifest("plan_mbpp", [_art(phase="plan", benchmark="mbpp")], nested)
        assert p.exists()
```

### Expected test output

```
uv run pytest tests/corpus_producer/test_manifest.py -v
...
8 passed in 0.xx s
```

### Commit

```bash
git commit -m "$(cat <<'EOF'
feat(corpus-producer): add JSONL manifest emission compatible with trainer_cli dataset schema

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6 — SQLite progress/checkpoint table (parallel-safe after Task 1)

**Files:**
- `libs/corpus-producer/src/corpus_producer/progress_db.py` (new)
- `tests/corpus_producer/test_progress_db.py` (new)

### Steps

- [ ] 6.1 Implement `ProgressDB` with `phase_corpus_progress` table.
- [ ] 6.2 Write failing tests.
- [ ] 6.3 Verify tests pass.
- [ ] 6.4 Run ruff + mypy.
- [ ] 6.5 Commit.

### `libs/corpus-producer/src/corpus_producer/progress_db.py`

```python
"""SQLite resume/checkpoint table for phase corpus production.

Schema
------
Table: phase_corpus_progress
  benchmark     TEXT  NOT NULL
  problem_id    TEXT  NOT NULL
  phase         TEXT  NOT NULL
  status        TEXT  NOT NULL   -- "pending" | "running" | "done" | "failed"
  completed_at  TEXT             -- ISO-8601 UTC, NULL until status="done"
  PRIMARY KEY (benchmark, problem_id, phase)

A separate table tracks per-bin training status:
  bin_key       TEXT  PRIMARY KEY
  status        TEXT  NOT NULL   -- "pending" | "training" | "done" | "failed"
  adapter_id    TEXT             -- set once training completes
  completed_at  TEXT
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path


_DDL_PROGRESS = """
CREATE TABLE IF NOT EXISTS phase_corpus_progress (
    benchmark    TEXT NOT NULL,
    problem_id   TEXT NOT NULL,
    phase        TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'pending',
    completed_at TEXT,
    PRIMARY KEY (benchmark, problem_id, phase)
);
"""

_DDL_BIN_TRAINING = """
CREATE TABLE IF NOT EXISTS bin_training_progress (
    bin_key      TEXT PRIMARY KEY,
    status       TEXT NOT NULL DEFAULT 'pending',
    adapter_id   TEXT,
    completed_at TEXT
);
"""


class ProgressDB:
    """Lightweight SQLite wrapper for phase corpus producer checkpointing.

    Args:
        db_path: Path to the SQLite database file. Created if absent.
    """

    def __init__(self, db_path: Path | str) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(_DDL_PROGRESS)
        self._conn.execute(_DDL_BIN_TRAINING)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Problem-level tracking
    # ------------------------------------------------------------------

    def mark_running(self, benchmark: str, problem_id: str, phase: str) -> None:
        """Upsert a (benchmark, problem_id, phase) row as 'running'."""
        self._conn.execute(
            """
            INSERT INTO phase_corpus_progress (benchmark, problem_id, phase, status)
            VALUES (?, ?, ?, 'running')
            ON CONFLICT(benchmark, problem_id, phase) DO UPDATE SET status='running'
            """,
            (benchmark, problem_id, phase),
        )
        self._conn.commit()

    def mark_done(self, benchmark: str, problem_id: str, phase: str) -> None:
        """Mark a (benchmark, problem_id, phase) row as 'done'."""
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """
            INSERT INTO phase_corpus_progress (benchmark, problem_id, phase, status, completed_at)
            VALUES (?, ?, ?, 'done', ?)
            ON CONFLICT(benchmark, problem_id, phase) DO UPDATE
            SET status='done', completed_at=excluded.completed_at
            """,
            (benchmark, problem_id, phase, now),
        )
        self._conn.commit()

    def mark_failed(self, benchmark: str, problem_id: str, phase: str) -> None:
        """Mark a (benchmark, problem_id, phase) row as 'failed'."""
        self._conn.execute(
            """
            INSERT INTO phase_corpus_progress (benchmark, problem_id, phase, status)
            VALUES (?, ?, ?, 'failed')
            ON CONFLICT(benchmark, problem_id, phase) DO UPDATE SET status='failed'
            """,
            (benchmark, problem_id, phase),
        )
        self._conn.commit()

    def is_done(self, benchmark: str, problem_id: str, phase: str) -> bool:
        """Return True if the (benchmark, problem_id, phase) row has status='done'."""
        row = self._conn.execute(
            "SELECT status FROM phase_corpus_progress WHERE benchmark=? AND problem_id=? AND phase=?",
            (benchmark, problem_id, phase),
        ).fetchone()
        return row is not None and row[0] == "done"

    # ------------------------------------------------------------------
    # Bin-level training tracking
    # ------------------------------------------------------------------

    def mark_bin_training(self, bin_key: str) -> None:
        """Mark a bin as currently being trained."""
        self._conn.execute(
            """
            INSERT INTO bin_training_progress (bin_key, status)
            VALUES (?, 'training')
            ON CONFLICT(bin_key) DO UPDATE SET status='training'
            """,
            (bin_key,),
        )
        self._conn.commit()

    def mark_bin_done(self, bin_key: str, adapter_id: str) -> None:
        """Mark a bin's training as complete with the resulting adapter_id."""
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """
            INSERT INTO bin_training_progress (bin_key, status, adapter_id, completed_at)
            VALUES (?, 'done', ?, ?)
            ON CONFLICT(bin_key) DO UPDATE
            SET status='done', adapter_id=excluded.adapter_id, completed_at=excluded.completed_at
            """,
            (bin_key, adapter_id, now),
        )
        self._conn.commit()

    def is_bin_done(self, bin_key: str) -> bool:
        """Return True if the bin's training has status='done'."""
        row = self._conn.execute(
            "SELECT status FROM bin_training_progress WHERE bin_key=?",
            (bin_key,),
        ).fetchone()
        return row is not None and row[0] == "done"

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()
```

### `tests/corpus_producer/test_progress_db.py`

```python
"""Tests for corpus_producer.progress_db.ProgressDB."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from corpus_producer.progress_db import ProgressDB


def _db(tmp: str) -> ProgressDB:
    return ProgressDB(Path(tmp) / "progress.db")


def test_is_done_false_before_mark():
    with tempfile.TemporaryDirectory() as tmp:
        db = _db(tmp)
        assert db.is_done("humaneval", "HE/0", "decompose") is False
        db.close()


def test_mark_done_then_is_done():
    with tempfile.TemporaryDirectory() as tmp:
        db = _db(tmp)
        db.mark_done("humaneval", "HE/0", "decompose")
        assert db.is_done("humaneval", "HE/0", "decompose") is True
        db.close()


def test_mark_running_not_done():
    with tempfile.TemporaryDirectory() as tmp:
        db = _db(tmp)
        db.mark_running("humaneval", "HE/0", "plan")
        assert db.is_done("humaneval", "HE/0", "plan") is False
        db.close()


def test_mark_failed_not_done():
    with tempfile.TemporaryDirectory() as tmp:
        db = _db(tmp)
        db.mark_failed("humaneval", "HE/0", "code")
        assert db.is_done("humaneval", "HE/0", "code") is False
        db.close()


def test_is_done_different_phases_independent():
    with tempfile.TemporaryDirectory() as tmp:
        db = _db(tmp)
        db.mark_done("humaneval", "HE/0", "decompose")
        assert db.is_done("humaneval", "HE/0", "plan") is False
        db.close()


def test_bin_done_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        db = _db(tmp)
        assert db.is_bin_done("decompose_humaneval") is False
        db.mark_bin_training("decompose_humaneval")
        assert db.is_bin_done("decompose_humaneval") is False
        db.mark_bin_done("decompose_humaneval", "oracle_decompose_humaneval")
        assert db.is_bin_done("decompose_humaneval") is True
        db.close()


def test_db_persists_across_reopen():
    with tempfile.TemporaryDirectory() as tmp:
        db1 = _db(tmp)
        db1.mark_done("mbpp", "MBPP/1", "integrate")
        db1.close()
        db2 = _db(tmp)
        assert db2.is_done("mbpp", "MBPP/1", "integrate") is True
        db2.close()
```

### Expected test output

```
uv run pytest tests/corpus_producer/test_progress_db.py -v
...
7 passed in 0.xx s
```

### Commit

```bash
git commit -m "$(cat <<'EOF'
feat(corpus-producer): add SQLite progress/checkpoint table for resume support

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7 — STaR rationalization fallback (parallel-safe after Task 1)

**Files:**
- `libs/corpus-producer/src/corpus_producer/rationalization.py` (new)
- `tests/corpus_producer/test_rationalization.py` (new)

### Steps

- [ ] 7.1 Implement `star_rationalize()`.
- [ ] 7.2 Write failing tests with mocked pipeline runner.
- [ ] 7.3 Verify tests pass.
- [ ] 7.4 Run ruff + mypy.
- [ ] 7.5 Commit.

### `libs/corpus-producer/src/corpus_producer/rationalization.py`

```python
"""STaR-style rationalization fallback for thin oracle bins.

When a bin has fewer than MIN_EXAMPLES_PER_BIN examples after the primary
self-distillation pass, this module re-runs the pipeline for the *failing*
problems, augmenting the prompt with ground-truth test outcome hints. This
gives the model extra signal to produce a passing trace, which is then
filtered, binned, and added to the manifest — tagged with rationalized=True.

Reference: Zelikman et al. 2022 (STaR) arxiv:2203.14465 — "hint-based
rationalization": inject the ground-truth answer as a hint, generate a
rationale that arrives at it, discard the hint at inference time.

Here the "answer" is the set of test outcomes (which tests pass/fail). We
inject them as a hint comment prepended to the problem prompt. The pipeline
runs again; if it now succeeds, we keep the artifacts as rationalized training
data.
"""

from __future__ import annotations

import logging
from typing import Callable

from corpus_producer.models import PhaseArtifact

logger = logging.getLogger(__name__)

MIN_EXAMPLES_PER_BIN = 60


def _build_hint_prompt(problem_prompt: str, test_hints: list[str]) -> str:
    """Prepend ground-truth test outcome hints to a problem prompt.

    Args:
        problem_prompt: Original problem text.
        test_hints: List of test outcome strings, e.g.
            ["test_add_two_numbers: PASS", "test_edge_case: FAIL"].

    Returns:
        Augmented prompt with hints block prepended.
    """
    hints_block = "\n".join(f"  # {h}" for h in test_hints)
    return (
        f"# Ground-truth test hints (use to guide your solution):\n"
        f"{hints_block}\n\n"
        f"{problem_prompt}"
    )


def star_rationalize(
    bin_key: str,
    existing_artifacts: list[PhaseArtifact],
    failing_problems: list[tuple[str, str, str]],  # (benchmark, problem_id, prompt)
    test_hints_by_problem: dict[str, list[str]],
    pipeline_runner: Callable[..., object],
    success_filter_fn: Callable[..., list[PhaseArtifact]],
    *,
    timeout: int = 300,
    base_model_id: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
) -> list[PhaseArtifact]:
    """Attempt rationalization on failing problems to pad a thin bin.

    Runs the pipeline with hint-augmented prompts for each failing problem.
    Keeps artifacts that pass the success filter, marks them rationalized=True.
    Stops once the bin reaches MIN_EXAMPLES_PER_BIN.

    Args:
        bin_key: The oracle bin that needs padding (for logging).
        existing_artifacts: Already-collected artifacts in this bin.
        failing_problems: List of (benchmark, problem_id, prompt) triples that
            failed in the primary pass.
        test_hints_by_problem: Map from problem_id to list of hint strings.
        pipeline_runner: Callable matching PipelineRunnerProtocol.
        success_filter_fn: Callable matching filter_artifacts signature.
        timeout: Per-run subprocess timeout.
        base_model_id: Base model for pipeline runs.

    Returns:
        New rationalized artifacts (not including existing_artifacts).
        Each returned artifact has rationalized=True.
    """
    new_artifacts: list[PhaseArtifact] = []
    current_count = len(existing_artifacts)

    for benchmark, problem_id, prompt in failing_problems:
        if current_count + len(new_artifacts) >= MIN_EXAMPLES_PER_BIN:
            logger.info(
                "Bin %s reached %d examples; stopping rationalization.",
                bin_key,
                MIN_EXAMPLES_PER_BIN,
            )
            break

        hints = test_hints_by_problem.get(problem_id, [])
        if not hints:
            logger.debug("No hints for %s/%s; skipping rationalization.", benchmark, problem_id)
            continue

        hint_prompt = _build_hint_prompt(prompt, hints)
        logger.info("Rationalizing %s/%s for bin %s", benchmark, problem_id, bin_key)

        from corpus_producer.pipeline_runner import PipelineRunResult

        result: PipelineRunResult = pipeline_runner(  # type: ignore[assignment]
            benchmark,
            problem_id,
            hint_prompt,
            timeout=timeout,
            base_model_id=base_model_id,
        )

        if not result.success:
            logger.debug("Rationalization pipeline failed for %s/%s", benchmark, problem_id)
            continue

        filtered = success_filter_fn(
            result.artifacts,
            result.final_code,
            benchmark,
            problem_id,
        )

        for art in filtered:
            art.rationalized = True
        new_artifacts.extend(filtered)

    logger.info(
        "Rationalization for bin %s produced %d new artifacts.",
        bin_key,
        len(new_artifacts),
    )
    return new_artifacts
```

### `tests/corpus_producer/test_rationalization.py`

```python
"""Tests for corpus_producer.rationalization."""

from __future__ import annotations

from corpus_producer.models import PhaseArtifact
from corpus_producer.pipeline_runner import PipelineRunResult
from corpus_producer.rationalization import (
    MIN_EXAMPLES_PER_BIN,
    _build_hint_prompt,
    star_rationalize,
)


def _art(phase: str = "decompose", rationalized: bool = False) -> PhaseArtifact:
    a = PhaseArtifact(
        phase=phase,
        benchmark="humaneval",
        problem_id="HE/0",
        pipeline_run_id="r",
        input_text="in",
        output_text="out",
        pass_at_1=True,
    )
    a.rationalized = rationalized
    return a


def _make_runner(success: bool, artifacts: list[PhaseArtifact]) -> object:
    """Return a callable that mimics PipelineRunnerProtocol."""

    def runner(benchmark: str, problem_id: str, prompt: str, **kw: object) -> PipelineRunResult:
        return PipelineRunResult(
            run_id="r2",
            benchmark=benchmark,
            problem_id=problem_id,
            artifacts=artifacts,
            final_code="def f(): pass",
            success=success,
        )

    return runner


def _filter_all_pass(arts: list[PhaseArtifact], code: str, bm: str, pid: str) -> list[PhaseArtifact]:
    for a in arts:
        a.pass_at_1 = True
    return arts


def _filter_none_pass(arts: list[PhaseArtifact], code: str, bm: str, pid: str) -> list[PhaseArtifact]:
    return []


def test_build_hint_prompt_includes_hints():
    result = _build_hint_prompt("Sort a list.", ["test_sort: PASS"])
    assert "test_sort: PASS" in result
    assert "Sort a list." in result


def test_rationalize_marks_artifacts_rationalized():
    failing = [("humaneval", "HE/1", "prompt")]
    hints = {"HE/1": ["test_a: FAIL"]}
    runner = _make_runner(True, [_art()])
    new = star_rationalize(
        "decompose_humaneval",
        existing_artifacts=[],
        failing_problems=failing,
        test_hints_by_problem=hints,
        pipeline_runner=runner,
        success_filter_fn=_filter_all_pass,
    )
    assert all(a.rationalized is True for a in new)


def test_rationalize_stops_when_bin_full():
    # existing_artifacts already at MIN - 1; one rationalization should fill it
    existing = [_art() for _ in range(MIN_EXAMPLES_PER_BIN - 1)]
    failing = [("humaneval", f"HE/{i}", "p") for i in range(10)]
    hints = {f"HE/{i}": ["test: PASS"] for i in range(10)}
    runner = _make_runner(True, [_art()])

    calls: list[str] = []

    def counting_runner(bm: str, pid: str, prompt: str, **kw: object) -> PipelineRunResult:
        calls.append(pid)
        return PipelineRunResult(
            run_id="r", benchmark=bm, problem_id=pid,
            artifacts=[_art()], final_code="", success=True,
        )

    new = star_rationalize(
        "decompose_humaneval",
        existing_artifacts=existing,
        failing_problems=failing,
        test_hints_by_problem=hints,
        pipeline_runner=counting_runner,
        success_filter_fn=_filter_all_pass,
    )
    # Should stop after 1 call (existing 59 + 1 new = 60)
    assert len(calls) == 1


def test_rationalize_skips_problem_with_no_hints():
    failing = [("humaneval", "HE/no_hint", "prompt")]
    hints: dict[str, list[str]] = {}
    runner = _make_runner(True, [_art()])
    calls: list[str] = []

    def counting_runner(bm: str, pid: str, prompt: str, **kw: object) -> PipelineRunResult:
        calls.append(pid)
        return PipelineRunResult(run_id="r", benchmark=bm, problem_id=pid, success=False)

    new = star_rationalize(
        "decompose_humaneval",
        existing_artifacts=[],
        failing_problems=failing,
        test_hints_by_problem=hints,
        pipeline_runner=counting_runner,
        success_filter_fn=_filter_all_pass,
    )
    assert calls == []
    assert new == []


def test_rationalize_returns_empty_when_filter_rejects():
    failing = [("humaneval", "HE/1", "p")]
    hints = {"HE/1": ["test: FAIL"]}
    runner = _make_runner(True, [_art()])
    new = star_rationalize(
        "decompose_humaneval",
        existing_artifacts=[],
        failing_problems=failing,
        test_hints_by_problem=hints,
        pipeline_runner=runner,
        success_filter_fn=_filter_none_pass,
    )
    assert new == []
```

### Expected test output

```
uv run pytest tests/corpus_producer/test_rationalization.py -v
...
5 passed in 0.xx s
```

### Commit

```bash
git commit -m "$(cat <<'EOF'
feat(corpus-producer): add STaR rationalization fallback for thin oracle bins

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 8 — Trainer bridge (sequential: after Tasks 5, 6, 7)

**Files:**
- `libs/corpus-producer/src/corpus_producer/trainer_bridge.py` (new)
- `tests/corpus_producer/test_trainer_bridge.py` (new)

### Steps

- [ ] 8.1 Implement `invoke_bin_training()` with Report_2 QLoRA defaults.
- [ ] 8.2 Write failing tests mocking `train_and_register`.
- [ ] 8.3 Verify tests pass.
- [ ] 8.4 Run ruff + mypy.
- [ ] 8.5 Commit.

### `libs/corpus-producer/src/corpus_producer/trainer_bridge.py`

```python
"""Bridge between the corpus producer and the QLoRA trainer.

Calls ``model_training.trainer.train_and_register`` for a single oracle bin
with Report_2-compliant defaults (rank=64 from DeltaCoder, alpha=32,
lr=2e-4, constant LR schedule, diff_aware_loss=True, warm_start=deltacoder).

GPU imports are deferred inside ``invoke_bin_training`` per INFRA-05 so the
module stays importable in CPU-only CI.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Report_2 / DeltaCoder warm-start defaults (Section 2.2)
_DEFAULT_RANK = 64
_DEFAULT_ALPHA = 32           # alpha = rank/2 per DeltaCoder convention
_DEFAULT_LR = 2e-4
_DEFAULT_EPOCHS = 3
_DEFAULT_LR_SCHED = "constant"
_DEFAULT_GRAD_ACCUM = 16
_DEFAULT_WARMUP_RATIO = 0.03
_WARM_START = "danielcherubini/Qwen3.5-DeltaCoder-9B"
_MODEL_CONFIG = "qwen3.5-9b"


def invoke_bin_training(
    bin_key: str,
    manifest_path: Path | str,
    *,
    dry_run: bool = False,
    database_url: str | None = None,
    mlflow_experiment: str = "rune-qlora",
    diff_aware_loss: bool = True,
    epochs: int | None = None,
    learning_rate: float = _DEFAULT_LR,
) -> str:
    """Train a QLoRA adapter for one oracle bin and register it.

    Calls ``train_and_register`` with DeltaCoder warm-start and Report_2
    hyperparameter defaults. The adapter_id is deterministic:
    ``oracle_<bin_key>``.

    Args:
        bin_key: Oracle bin identifier (e.g. "decompose_humaneval",
            "diagnose_pooled").
        manifest_path: Path to the JSONL manifest for this bin.
        dry_run: If True, log parameters and return without training.
        database_url: SQLAlchemy URL for AdapterRegistry. Defaults to
            env/default path.
        mlflow_experiment: MLflow experiment name.
        diff_aware_loss: Whether to enable diff-aware loss weighting.
            Default True per Report_2 recommendation.
        epochs: Override training epochs. Defaults to ``_DEFAULT_EPOCHS``.
        learning_rate: Override learning rate.

    Returns:
        The adapter_id registered in AdapterRegistry.

    Raises:
        FileNotFoundError: If ``manifest_path`` does not exist.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    adapter_id = f"oracle_{bin_key}"
    resolved_epochs = epochs if epochs is not None else _DEFAULT_EPOCHS

    logger.info(
        "Training oracle adapter %r from %s (dry_run=%s)",
        adapter_id,
        manifest_path,
        dry_run,
    )

    if dry_run:
        logger.info(
            "DRY RUN — would call train_and_register("
            "adapter_id=%r, dataset_path=%s, warm_start=%r, "
            "rank=%d, alpha=%d, epochs=%d, lr=%g, diff_aware_loss=%s)",
            adapter_id,
            manifest_path,
            _WARM_START,
            _DEFAULT_RANK,
            _DEFAULT_ALPHA,
            resolved_epochs,
            learning_rate,
            diff_aware_loss,
        )
        return adapter_id

    # Deferred GPU import (INFRA-05)
    from model_training.trainer import train_and_register  # type: ignore[import]

    train_and_register(
        session_id=None,
        adapter_id=adapter_id,
        dataset_path=str(manifest_path),
        task_type=bin_key,
        model_config_name=_MODEL_CONFIG,
        warm_start_adapter_id=_WARM_START,
        rank=_DEFAULT_RANK,
        alpha=_DEFAULT_ALPHA,
        epochs=resolved_epochs,
        learning_rate=learning_rate,
        gradient_accumulation_steps=_DEFAULT_GRAD_ACCUM,
        lr_scheduler_type=_DEFAULT_LR_SCHED,
        warmup_ratio=_DEFAULT_WARMUP_RATIO,
        diff_aware_loss=diff_aware_loss,
        database_url=database_url,
        mlflow_experiment=mlflow_experiment,
        encoding_mode="single_turn",
    )

    logger.info("Registered oracle adapter %r", adapter_id)
    return adapter_id
```

### `tests/corpus_producer/test_trainer_bridge.py`

```python
"""Tests for corpus_producer.trainer_bridge."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from corpus_producer.trainer_bridge import _WARM_START, invoke_bin_training


def _write_manifest(tmpdir: str, n: int = 3) -> Path:
    p = Path(tmpdir) / "decompose_humaneval.jsonl"
    records = [
        {"task_id": f"humaneval/HE/{i}/decompose", "activation_text": "in", "teacher_text": "in\nout"}
        for i in range(n)
    ]
    with p.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return p


def test_dry_run_returns_adapter_id_without_training():
    with tempfile.TemporaryDirectory() as tmp:
        mp = _write_manifest(tmp)
        adapter_id = invoke_bin_training("decompose_humaneval", mp, dry_run=True)
        assert adapter_id == "oracle_decompose_humaneval"


def test_adapter_id_format():
    with tempfile.TemporaryDirectory() as tmp:
        mp = _write_manifest(tmp)
        adapter_id = invoke_bin_training("diagnose_pooled", mp, dry_run=True)
        assert adapter_id == "oracle_diagnose_pooled"


def test_raises_if_manifest_missing():
    with pytest.raises(FileNotFoundError, match="Manifest not found"):
        invoke_bin_training("decompose_humaneval", "/nonexistent/path.jsonl", dry_run=True)


@patch("corpus_producer.trainer_bridge.train_and_register")
def test_train_called_with_correct_args(mock_t: MagicMock) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        mp = _write_manifest(tmp)
        invoke_bin_training("decompose_humaneval", mp, dry_run=False)
        mock_t.assert_called_once()
        kwargs = mock_t.call_args[1]
        assert kwargs["adapter_id"] == "oracle_decompose_humaneval"
        assert kwargs["task_type"] == "decompose_humaneval"
        assert kwargs["warm_start_adapter_id"] == _WARM_START
        assert kwargs["rank"] == 64
        assert kwargs["diff_aware_loss"] is True


@patch("corpus_producer.trainer_bridge.train_and_register")
def test_train_called_with_dataset_path(mock_t: MagicMock) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        mp = _write_manifest(tmp)
        invoke_bin_training("plan_mbpp", mp, dry_run=False)
        kwargs = mock_t.call_args[1]
        assert kwargs["dataset_path"] == str(mp)
```

### Expected test output

```
uv run pytest tests/corpus_producer/test_trainer_bridge.py -v
...
5 passed in 0.xx s
```

### Commit

```bash
git commit -m "$(cat <<'EOF'
feat(corpus-producer): add trainer bridge with Report_2 QLoRA defaults and DeltaCoder warm-start

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 9 — CLI orchestrator `scripts/phase_corpus_producer.py` (sequential: after Tasks 2–8)

**Files:**
- `scripts/phase_corpus_producer.py` (new)
- No test file for this task — covered by Task 11 integration test.

### Steps

- [ ] 9.1 Implement CLI argument parser mirroring `trainer_cli.py` patterns.
- [ ] 9.2 Implement `produce_corpus()` main orchestration loop.
- [ ] 9.3 Wire `--force` to skip `is_done` check; `--dry-run` to `invoke_bin_training(dry_run=True)`.
- [ ] 9.4 Run `uv run ruff check scripts/phase_corpus_producer.py` and `uv run mypy scripts/phase_corpus_producer.py`.
- [ ] 9.5 Commit.

### `scripts/phase_corpus_producer.py`

```python
"""Phase Corpus Producer — self-distillation oracle corpus for phase-aware training.

For each (benchmark, problem), runs the full 5-phase Rune pipeline, filters
by Pass@1=1.0, bins per-phase artifacts into 25 oracle bins, emits JSONL
manifests, and invokes QLoRA training per bin.

Usage:
    uv run scripts/phase_corpus_producer.py \\
        --benchmark humaneval \\
        --out-dir data/phase_corpus \\
        --max-problems 20

    uv run scripts/phase_corpus_producer.py \\
        --benchmark humaneval mbpp apps \\
        --out-dir data/phase_corpus \\
        --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bootstrap import setup_path  # type: ignore[import]

setup_path()

from corpus_producer.binning import bin_artifacts, expected_bin_keys
from corpus_producer.manifest import emit_bin_manifest
from corpus_producer.models import PhaseArtifact
from corpus_producer.pipeline_runner import run_pipeline_for_problem
from corpus_producer.progress_db import ProgressDB
from corpus_producer.rationalization import MIN_EXAMPLES_PER_BIN, star_rationalize
from corpus_producer.success_filter import filter_artifacts
from corpus_producer.trainer_bridge import invoke_bin_training

logger = logging.getLogger(__name__)

BENCHMARKS = [
    "humaneval",
    "mbpp",
    "apps",
    "bigcodebench",
    "ds_1000",
    "livecodebench",
]
PIPELINE_TIMEOUT_DEFAULT = 300


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="phase_corpus_producer",
        description="Self-distillation oracle corpus producer for Rune phase training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--benchmark",
        nargs="+",
        choices=BENCHMARKS,
        default=BENCHMARKS,
        metavar="BENCHMARK",
        help="One or more benchmark ids to run.",
    )
    parser.add_argument(
        "--problems",
        nargs="*",
        metavar="PROBLEM_ID",
        default=None,
        help="Explicit problem ids to run (overrides --max-problems).",
    )
    parser.add_argument(
        "--max-problems",
        type=int,
        default=None,
        metavar="N",
        help="Maximum problems per benchmark (None = all).",
    )
    parser.add_argument(
        "--out-dir",
        default="data/phase_corpus",
        metavar="DIR",
        help="Output directory for JSONL manifests and progress DB.",
    )
    parser.add_argument(
        "--pipeline-timeout",
        type=int,
        default=PIPELINE_TIMEOUT_DEFAULT,
        dest="pipeline_timeout",
        metavar="SECS",
        help="Per-problem pipeline subprocess timeout (seconds).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if progress DB marks a problem as done.",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        dest="skip_training",
        help="Emit manifests but do not invoke train_and_register.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Run the pipeline and emit manifests; pass dry_run=True to trainer.",
    )
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen2.5-Coder-7B-Instruct",
        dest="base_model",
        metavar="MODEL_ID",
        help="Base model HF repo id for pipeline runs.",
    )
    parser.add_argument(
        "--database-url",
        default=None,
        dest="database_url",
        metavar="URL",
        help="SQLAlchemy URL for AdapterRegistry (defaults to env/default).",
    )
    parser.add_argument(
        "--mlflow-experiment",
        default="rune-qlora",
        dest="mlflow_experiment",
        metavar="NAME",
    )
    return parser


def _load_problems(benchmark: str, problem_ids: list[str] | None, max_problems: int | None) -> list[tuple[str, str]]:
    """Return list of (problem_id, prompt) pairs for the given benchmark.

    Imports benchmark dataset loader lazily. Falls back to a stub list when
    the evaluation package is not yet installed (CPU-only CI).

    Args:
        benchmark: Benchmark identifier.
        problem_ids: Explicit list of problem ids (overrides max_problems).
        max_problems: Cap on number of problems; None = all.

    Returns:
        List of (problem_id, prompt) tuples.
    """
    try:
        from evaluation.benchmarks import load_problems  # type: ignore[import]
        problems = load_problems(benchmark, problem_ids=problem_ids, max_samples=max_problems)
        return [(p.problem_id, p.prompt) for p in problems]
    except ImportError:
        logger.warning(
            "evaluation.benchmarks not available — using stub problem list for %s",
            benchmark,
        )
        n = max_problems or 1
        return [(f"{benchmark.upper()}/{i}", f"Stub problem {i} for {benchmark}.") for i in range(n)]


def produce_corpus(
    benchmarks: list[str],
    out_dir: Path,
    *,
    problem_ids: list[str] | None = None,
    max_problems: int | None = None,
    pipeline_timeout: int = PIPELINE_TIMEOUT_DEFAULT,
    force: bool = False,
    skip_training: bool = False,
    dry_run: bool = False,
    base_model: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
    database_url: str | None = None,
    mlflow_experiment: str = "rune-qlora",
) -> dict[str, int]:
    """Main orchestration loop for phase corpus production.

    For each (benchmark, problem):
      1. Skip if already done in progress DB (unless --force).
      2. Run 5-phase pipeline via subprocess.
      3. Filter artifacts by Pass@1.
      4. Accumulate into per-bin artifact lists.

    After all problems:
      5. For each bin with < MIN_EXAMPLES_PER_BIN, run STaR rationalization.
      6. Emit JSONL manifests per bin.
      7. Invoke QLoRA training per bin (unless --skip-training / --dry-run).

    Args:
        benchmarks: Benchmark ids to process.
        out_dir: Root output directory.
        problem_ids: Explicit problem ids (applied across all benchmarks).
        max_problems: Cap per benchmark.
        pipeline_timeout: Per-problem subprocess timeout (seconds).
        force: Ignore progress DB and re-run all problems.
        skip_training: Emit manifests but skip training.
        dry_run: Pass dry_run=True to trainer; still emits manifests.
        base_model: HF repo id for pipeline subprocess.
        database_url: AdapterRegistry DB URL.
        mlflow_experiment: MLflow experiment name.

    Returns:
        Dict mapping bin_key -> number of training records in that bin.
    """
    out_dir = Path(out_dir)
    db = ProgressDB(out_dir / "progress.db")

    # Accumulated artifacts keyed by bin
    bin_artifacts_map: dict[str, list[PhaseArtifact]] = {}
    # Track failing problems per benchmark for rationalization
    failing_problems: list[tuple[str, str, str]] = []

    for benchmark in benchmarks:
        problems = _load_problems(benchmark, problem_ids, max_problems)
        logger.info("Processing %d problems for benchmark %s", len(problems), benchmark)

        for problem_id, prompt in problems:
            # Resume: skip if all phases are already done
            if not force and db.is_done(benchmark, problem_id, "integrate"):
                logger.debug("Skipping done problem %s/%s", benchmark, problem_id)
                continue

            db.mark_running(benchmark, problem_id, "pipeline")
            result = run_pipeline_for_problem(
                benchmark,
                problem_id,
                prompt,
                timeout=pipeline_timeout,
                base_model_id=base_model,
            )

            if not result.success:
                db.mark_failed(benchmark, problem_id, "pipeline")
                failing_problems.append((benchmark, problem_id, prompt))
                continue

            kept = filter_artifacts(result.artifacts, result.final_code, benchmark, problem_id)

            if kept:
                for art in kept:
                    key = art.bin_key()
                    bin_artifacts_map.setdefault(key, []).append(art)
                db.mark_done(benchmark, problem_id, "integrate")
            else:
                db.mark_failed(benchmark, problem_id, "integrate")
                failing_problems.append((benchmark, problem_id, prompt))

    # STaR rationalization for thin bins
    for bin_key, arts in list(bin_artifacts_map.items()):
        if len(arts) < MIN_EXAMPLES_PER_BIN:
            logger.info(
                "Bin %s has %d examples (< %d); attempting rationalization.",
                bin_key,
                len(arts),
                MIN_EXAMPLES_PER_BIN,
            )
            new_arts = star_rationalize(
                bin_key,
                existing_artifacts=arts,
                failing_problems=failing_problems,
                test_hints_by_problem={},  # hints sourced from benchmark harness when available
                pipeline_runner=lambda bm, pid, p, **kw: run_pipeline_for_problem(bm, pid, p, **kw),
                success_filter_fn=filter_artifacts,
                timeout=pipeline_timeout,
                base_model_id=base_model,
            )
            bin_artifacts_map[bin_key].extend(new_arts)

    # Emit manifests and train
    manifests_dir = out_dir / "manifests"
    bin_record_counts: dict[str, int] = {}

    for bin_key, arts in bin_artifacts_map.items():
        if not arts:
            continue
        if db.is_bin_done(bin_key) and not force:
            logger.info("Bin %s already trained; skipping.", bin_key)
            bin_record_counts[bin_key] = len(arts)
            continue

        manifest_path = emit_bin_manifest(bin_key, arts, manifests_dir)
        bin_record_counts[bin_key] = len(arts)

        if skip_training:
            logger.info("--skip-training: manifest written, skipping training for %s.", bin_key)
            continue

        db.mark_bin_training(bin_key)
        adapter_id = invoke_bin_training(
            bin_key,
            manifest_path,
            dry_run=dry_run,
            database_url=database_url,
            mlflow_experiment=mlflow_experiment,
        )
        db.mark_bin_done(bin_key, adapter_id)
        logger.info("Bin %s -> adapter %s", bin_key, adapter_id)

    db.close()
    return bin_record_counts


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = _build_parser()
    args = parser.parse_args()

    counts = produce_corpus(
        benchmarks=args.benchmark,
        out_dir=Path(args.out_dir),
        problem_ids=args.problems,
        max_problems=args.max_problems,
        pipeline_timeout=args.pipeline_timeout,
        force=args.force,
        skip_training=args.skip_training,
        dry_run=args.dry_run,
        base_model=args.base_model,
        database_url=args.database_url,
        mlflow_experiment=args.mlflow_experiment,
    )

    total = sum(counts.values())
    print(f"Done. {len(counts)} bins, {total} total training records.")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v} records")


if __name__ == "__main__":
    main()
```

### Commit

```bash
git commit -m "$(cat <<'EOF'
feat(corpus-producer): add phase_corpus_producer.py CLI orchestrator

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 10 — Batch shell runner `scripts/run_phase_corpus.sh` (parallel-safe after Task 9)

**Files:**
- `scripts/run_phase_corpus.sh` (new)

### Steps

- [ ] 10.1 Write the batch shell script.
- [ ] 10.2 Make executable: `chmod +x scripts/run_phase_corpus.sh`.
- [ ] 10.3 Commit.

### `scripts/run_phase_corpus.sh`

```bash
#!/usr/bin/env bash
# run_phase_corpus.sh — batch oracle corpus production across all 6 benchmarks.
#
# APPS is subsampled to N=500 with seed=42, stratified by difficulty.
# All other benchmarks use their full split (or capped by --max-problems).
#
# Usage:
#   ./scripts/run_phase_corpus.sh                    # full run
#   ./scripts/run_phase_corpus.sh --dry-run          # dry run (no GPU)
#   ./scripts/run_phase_corpus.sh --skip-training    # emit manifests only
#   OUT_DIR=data/phase_corpus_v2 ./scripts/run_phase_corpus.sh
#
# Environment:
#   OUT_DIR          Output directory (default: data/phase_corpus)
#   BASE_MODEL       HF repo id (default: Qwen/Qwen2.5-Coder-7B-Instruct)
#   PIPELINE_TIMEOUT Per-problem timeout in seconds (default: 300)

set -euo pipefail

OUT_DIR="${OUT_DIR:-data/phase_corpus}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}"
PIPELINE_TIMEOUT="${PIPELINE_TIMEOUT:-300}"
EXTRA_FLAGS="${@}"

SCRIPT="uv run scripts/phase_corpus_producer.py"
COMMON_FLAGS="--out-dir ${OUT_DIR} --base-model ${BASE_MODEL} --pipeline-timeout ${PIPELINE_TIMEOUT} ${EXTRA_FLAGS}"

echo "=== Phase Corpus Production ==="
echo "    OUT_DIR=${OUT_DIR}"
echo "    BASE_MODEL=${BASE_MODEL}"
echo "    PIPELINE_TIMEOUT=${PIPELINE_TIMEOUT}s"
echo ""

# HumanEval — 164 problems (elementary)
echo "--- humaneval ---"
${SCRIPT} --benchmark humaneval ${COMMON_FLAGS}

# MBPP — 374 problems (basic)
echo "--- mbpp ---"
${SCRIPT} --benchmark mbpp ${COMMON_FLAGS}

# APPS — subsampled N=500, seed=42, stratified by difficulty
# The corpus producer's _load_problems() calls load_problems(benchmark,
# max_samples=500) which applies stratified sampling internally.
echo "--- apps (N=500, seed=42, stratified) ---"
${SCRIPT} --benchmark apps --max-problems 500 ${COMMON_FLAGS}

# BigCodeBench — 1140 problems (applied)
echo "--- bigcodebench ---"
${SCRIPT} --benchmark bigcodebench ${COMMON_FLAGS}

# DS-1000 — 1000 problems (data science)
echo "--- ds_1000 ---"
${SCRIPT} --benchmark ds_1000 ${COMMON_FLAGS}

# LiveCodeBench — 500 problems (competitive)
echo "--- livecodebench ---"
${SCRIPT} --benchmark livecodebench ${COMMON_FLAGS}

echo ""
echo "=== All benchmarks complete. Manifests in ${OUT_DIR}/manifests/ ==="
```

### Commit

```bash
git commit -m "$(cat <<'EOF'
feat(corpus-producer): add run_phase_corpus.sh batch runner for all 6 benchmarks

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 11 — Integration test (sequential: after Task 9)

**Files:**
- `tests/corpus_producer/test_integration.py` (new)

### Steps

- [ ] 11.1 Write end-to-end integration test with mocked `run_benchmark` and `FakePipelineRunner`.
- [ ] 11.2 Verify test passes.
- [ ] 11.3 Run full test suite: `uv run pytest tests/corpus_producer/ -v`.
- [ ] 11.4 Run ruff + mypy across the new library.
- [ ] 11.5 Commit.

### `tests/corpus_producer/test_integration.py`

```python
"""End-to-end integration test for the phase corpus producer.

Uses:
  - FakePipelineRunner (returns synthetic artifacts for 1 HumanEval problem)
  - Mocked run_benchmark returning Pass@1=1.0
  - Real ProgressDB, bin_artifacts, emit_bin_manifest, invoke_bin_training(dry_run=True)

Verifies the full produce_corpus() flow produces the expected manifest files
and bin record counts without touching GPU or the real benchmark harness.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from corpus_producer.models import PhaseArtifact
from corpus_producer.pipeline_runner import PipelineRunResult
from corpus_producer.rationalization import MIN_EXAMPLES_PER_BIN

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BENCHMARK = "humaneval"
_PROBLEM_ID = "HumanEval/0"
_PHASES = ["decompose", "plan", "code", "integrate"]


def _make_artifacts(benchmark: str = _BENCHMARK, problem_id: str = _PROBLEM_ID) -> list[PhaseArtifact]:
    return [
        PhaseArtifact(
            phase=p,
            benchmark=benchmark,
            problem_id=problem_id,
            pipeline_run_id="run-test",
            input_text=f"input for {p}",
            output_text=f"output for {p}",
        )
        for p in _PHASES
    ]


def _fake_pipeline_runner(
    benchmark: str,
    problem_id: str,
    prompt: str,
    *,
    timeout: int = 300,
    base_model_id: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
) -> PipelineRunResult:
    return PipelineRunResult(
        run_id="run-test",
        benchmark=benchmark,
        problem_id=problem_id,
        artifacts=_make_artifacts(benchmark, problem_id),
        final_code="def solution(): pass",
        success=True,
    )


def _mock_run_benchmark_pass(model_adapter_stack, benchmark_id, problem_ids=None, max_samples=1):
    verdict = MagicMock()
    verdict.passed = True
    result = MagicMock()
    result.per_problem = {pid: verdict for pid in (problem_ids or [_PROBLEM_ID])}
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@patch("corpus_producer.success_filter.run_benchmark", side_effect=_mock_run_benchmark_pass)
@patch("corpus_producer.pipeline_runner.run_pipeline_for_problem", side_effect=_fake_pipeline_runner)
@patch("corpus_producer.phase_corpus_producer._load_problems")
def test_produce_corpus_one_problem_emits_manifests(
    mock_load: MagicMock,
    mock_runner: MagicMock,
    mock_rb: MagicMock,
) -> None:
    """One HumanEval problem, Pass@1=1.0 → 4 manifest files (one per phase)."""
    mock_load.return_value = [(_PROBLEM_ID, "Sort a list of integers.")]

    # Import here so patches are applied
    from corpus_producer import phase_corpus_producer as pcp  # noqa: PLC0415 (deferred)
    pcp.run_pipeline_for_problem = _fake_pipeline_runner  # type: ignore[attr-defined]

    with tempfile.TemporaryDirectory() as tmpdir:
        counts = pcp.produce_corpus(
            benchmarks=[_BENCHMARK],
            out_dir=Path(tmpdir),
            skip_training=True,
        )

    assert "decompose_humaneval" in counts
    assert "plan_humaneval" in counts
    assert "code_humaneval" in counts
    assert "integrate_humaneval" in counts
    assert "diagnose_pooled" not in counts  # no diagnose phase in this run


@patch("corpus_producer.success_filter.run_benchmark", side_effect=_mock_run_benchmark_pass)
@patch("corpus_producer.pipeline_runner.run_pipeline_for_problem", side_effect=_fake_pipeline_runner)
@patch("corpus_producer.phase_corpus_producer._load_problems")
def test_produce_corpus_dry_run_does_not_train(
    mock_load: MagicMock,
    mock_runner: MagicMock,
    mock_rb: MagicMock,
) -> None:
    """dry_run=True should emit manifests but call train with dry_run=True."""
    mock_load.return_value = [(_PROBLEM_ID, "Sort a list.")]

    from corpus_producer import phase_corpus_producer as pcp  # noqa: PLC0415
    pcp.run_pipeline_for_problem = _fake_pipeline_runner  # type: ignore[attr-defined]

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("corpus_producer.trainer_bridge.train_and_register") as mock_train:
            counts = pcp.produce_corpus(
                benchmarks=[_BENCHMARK],
                out_dir=Path(tmpdir),
                dry_run=True,
            )
            # dry_run=True means trainer is called with dry_run=True (not real GPU)
            # trainer_bridge.invoke_bin_training(dry_run=True) returns early
            mock_train.assert_not_called()


@patch("corpus_producer.success_filter.run_benchmark", side_effect=_mock_run_benchmark_pass)
@patch("corpus_producer.pipeline_runner.run_pipeline_for_problem", side_effect=_fake_pipeline_runner)
@patch("corpus_producer.phase_corpus_producer._load_problems")
def test_produce_corpus_resume_skips_done_problems(
    mock_load: MagicMock,
    mock_runner: MagicMock,
    mock_rb: MagicMock,
) -> None:
    """Second run with same out_dir skips already-done problem."""
    mock_load.return_value = [(_PROBLEM_ID, "prompt")]

    from corpus_producer import phase_corpus_producer as pcp  # noqa: PLC0415
    pcp.run_pipeline_for_problem = _fake_pipeline_runner  # type: ignore[attr-defined]

    with tempfile.TemporaryDirectory() as tmpdir:
        # First run
        pcp.produce_corpus(benchmarks=[_BENCHMARK], out_dir=Path(tmpdir), skip_training=True)
        first_call_count = mock_runner.call_count

        # Second run (same out_dir, force=False)
        mock_runner.reset_mock()
        pcp.produce_corpus(benchmarks=[_BENCHMARK], out_dir=Path(tmpdir), skip_training=True)
        second_call_count = mock_runner.call_count

    assert second_call_count == 0, "Resume should skip already-done problems"


@patch("corpus_producer.success_filter.run_benchmark", side_effect=_mock_run_benchmark_pass)
@patch("corpus_producer.pipeline_runner.run_pipeline_for_problem", side_effect=_fake_pipeline_runner)
@patch("corpus_producer.phase_corpus_producer._load_problems")
def test_produce_corpus_force_reruns_done_problems(
    mock_load: MagicMock,
    mock_runner: MagicMock,
    mock_rb: MagicMock,
) -> None:
    """--force re-runs even done problems."""
    mock_load.return_value = [(_PROBLEM_ID, "prompt")]

    from corpus_producer import phase_corpus_producer as pcp  # noqa: PLC0415
    pcp.run_pipeline_for_problem = _fake_pipeline_runner  # type: ignore[attr-defined]

    with tempfile.TemporaryDirectory() as tmpdir:
        pcp.produce_corpus(benchmarks=[_BENCHMARK], out_dir=Path(tmpdir), skip_training=True)
        mock_runner.reset_mock()
        pcp.produce_corpus(benchmarks=[_BENCHMARK], out_dir=Path(tmpdir), skip_training=True, force=True)
        assert mock_runner.call_count > 0
```

### Expected test output

```
uv run pytest tests/corpus_producer/ -v
...
35+ passed in 0.xx s
```

### Full suite check

```bash
uv run ruff check libs/corpus-producer/ scripts/phase_corpus_producer.py scripts/run_phase_corpus.sh
uv run mypy libs/corpus-producer/src/ scripts/phase_corpus_producer.py
uv run pytest tests/corpus_producer/ -v
```

### Commit

```bash
git commit -m "$(cat <<'EOF'
test(corpus-producer): add end-to-end integration test with mocked pipeline and run_benchmark

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review

### Coverage against scope requirements

| Requirement | Covered | Task |
|---|---|---|
| `PhaseArtifact` dataclass with all specified fields | Yes | Task 1 |
| Pipeline runner wrapper (subprocess-mode) + `FakePipelineRunner` protocol | Yes | Task 2 |
| Success filter calling `run_benchmark`; Pass@1=1.0 keeps all; 0.0 keeps diagnose-repair-succeeded only | Yes | Task 3 |
| `bin_artifacts()` — 25 bins: 4 phases × 6 benchmarks + `diagnose_pooled` | Yes | Task 4 |
| `emit_bin_manifest()` — JSONL with `task_id` / `activation_text` / `teacher_text` / `metadata` | Yes | Task 5 |
| Per-bin `train_and_register(...)` with DeltaCoder warm-start + Report_2 defaults | Yes | Task 8 |
| STaR rationalization with `MIN_EXAMPLES_PER_BIN = 60`, `rationalized=True` tag | Yes | Task 7 |
| SQLite `phase_corpus_progress` table, PK `(benchmark, problem_id, phase)` | Yes | Task 6 |
| CLI `--benchmark`, `--problems`, `--max-problems`, `--out-dir`, `--force`, `--skip-training`, `--dry-run`, `--pipeline-timeout` | Yes | Task 9 |
| Batch runner iterating 6 benchmarks, APPS N=500 seed=42 | Yes | Task 10 |
| Integration test: 1 HumanEval problem, mocked `run_benchmark` Pass@1=1.0, mocked pipeline | Yes | Task 11 |
| Unit tests per component | Yes | Tasks 1–8 |

### Architecture decisions confirmed

- **Subprocess-mode** for `rune_runner.py`: clean GPU state, no adapter registry cross-contamination between runs, future process-level parallelism.
- **`diagnose_pooled`** is the literal bin key; the `bin_key()` method on `PhaseArtifact` hardcodes this for `phase == "diagnose"`.
- **`MIN_EXAMPLES_PER_BIN = 60`** constant lives in `rationalization.py`; imported by CLI.
- **Manifest schema** is `task_id` / `activation_text` / `teacher_text` / `metadata` — identical to `d2l_data.normalize_mined_pairs` output, drop-in with `trainer_cli.py --dataset PATH`.
- **DeltaCoder warm-start**: `trainer_bridge.py` uses `"danielcherubini/Qwen3.5-DeltaCoder-9B"` explicitly, matching `trainer_cli.py` alias resolution and Report_2 Section 2.1.
- **Per-pipeline timeout**: 300 s default, `--pipeline-timeout` configurable.
- **Resume**: `ProgressDB.is_done("integrate")` is the gating check per-problem; `is_bin_done()` gates per-bin training. `--force` bypasses both.
- **APPS stratified sampling**: delegated to `evaluation.benchmarks.load_problems(benchmark, max_samples=500)` which is expected to apply stratified-by-difficulty sampling internally (Plan A's responsibility); the shell script passes `--max-problems 500`.

### Locked design decisions (from spec)

All locked decisions from the spec are reflected:
- `DIAGNOSE_BIN_KEY = "diagnose_pooled"` (q4 decision)
- `MIN_EXAMPLES_PER_BIN = 60` (spec constant)
- Resume table schema: `benchmark TEXT, problem_id TEXT, phase TEXT, status TEXT, completed_at TEXT`, PK `(benchmark, problem_id, phase)` — implemented verbatim in `progress_db.py`
- Manifest compatible with `trainer.py` / `d2l_data.py` pair schema — verified against `_make_pair_record` and `pairs_to_chat_messages`
- `run_benchmark` mock in all tests via `unittest.mock.patch`
- `FakePipelineRunner` protocol for test injection

### Follow-on items (out of scope for this plan)

1. **Round-2 hypernetwork-in-the-loop redistribution** — after hypernetwork v1 is trained, re-run corpus production with the hypernetwork in the loop, mix 50/50 base/hypernet traces, retrain. Tracked in spec `corpus_bootstrap.distribution_shift_mitigation.round_2`.
2. **Hypernetwork architecture swap** — T2L MLP + per-module heads replacing the Perceiver; separate plan required.
3. **GPU-distributed corpus generation** — `run_phase_corpus.sh` currently runs benchmarks serially; a future `--parallel` flag could fan out across GPUs using `asyncio` + process pools.
4. **S3 upload path for manifests** — `emit_bin_manifest` writes locally; adding an `--upload-s3` flag is deferred.
5. **`load_problems` stratified APPS sampling** — the exact stratification logic (seed=42, by difficulty bucket) is delegated to Plan A's `evaluation.benchmarks.load_problems` interface; if Plan A does not implement stratification, `produce_corpus` will need a wrapper.
6. **`rune_runner.py --output-json` flag** — Task 2's `run_pipeline_for_problem` calls `rune_runner.py --output-json <path>`. This flag does not exist in the current `rune_runner.py`. It must be added as a follow-on patch to `rune_runner.py` (trivially: serialize `run_phased_pipeline` return dict to the path). This is a one-line addition and is blocked on nothing else in this plan.

### Test count summary

| Task | Test file | Expected passing tests |
|---|---|---|
| 1 | `test_models.py` | 8 |
| 2 | `test_pipeline_runner.py` | 9 |
| 3 | `test_success_filter.py` | 5 |
| 4 | `test_binning.py` | 6 |
| 5 | `test_manifest.py` | 8 |
| 6 | `test_progress_db.py` | 7 |
| 7 | `test_rationalization.py` | 5 |
| 8 | `test_trainer_bridge.py` | 5 |
| 11 | `test_integration.py` | 4 |
| **Total** | | **57** |

