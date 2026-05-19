# Pipeline Speed & Reliability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix three compounding failures (adapter OOM, decomposition explosion, length stops) that cause every HPO trial to score 0.0.

**Architecture:** Five independent fixes applied in risk order: (P1) eager adapter unload after every `run_iteration()` call, (P2) decompose prompt improvement with few-shot examples and chain-of-thought suppression, (P3) task-complexity gating to skip decompose for simple tasks, (P4) thinking token budget so `<think>` blocks don't starve response tokens, (P5) multi-turn continuation routing so truncated outputs accumulate instead of wasting retry attempts. P6 (structured JSON decompose) deferred to a separate plan.

**Tech Stack:** Python 3.12, LangGraph, PEFT, PyTorch, Jinja2, frozen dataclasses, pytest

**Spec:** `docs/superpowers/specs/2026-05-19-pipeline-speed-reliability-design.md`

---

## File Structure

| Action | Path | Responsibility |
|--------|------|----------------|
| Modify | `scripts/rune_runner.py` | P1: eager unload at 9 call sites; P3: `_should_skip_decompose()`; P5: continuation loop in code phase |
| Modify | `libs/shared/src/shared/templates/decompose.j2` | P2: chain-of-thought suppression, simple-task examples, negative example |
| Modify | `libs/shared/src/shared/templates/prompt_decompose.j2` | P2: formatting constraints, negative example |
| Modify | `libs/shared/src/shared/templates/prompt_decompose_concise.j2` | P2: same constraints |
| Modify | `libs/shared/src/shared/pipeline_config.py` | P3: `DecomposeConfig`; P4: `thinking_budget` field |
| Modify | `libs/inference/src/inference/provider.py` | P4: `thinking_budget` param on ABC |
| Modify | `libs/inference/src/inference/transformers_provider.py` | P4: thinking budget logic in `generate()` |
| Modify | `libs/inference/src/inference/vllm_provider.py` | P4: accept `thinking_budget` param |
| Modify | `libs/inference/src/inference/ollama_provider.py` | P4: accept `thinking_budget` param |
| Modify | `libs/inference/src/inference/llamacpp_provider.py` | P4: accept `thinking_budget` param |
| Modify | `services/rune-agent/src/rune_agent/nodes.py` | P4: pass `thinking_budget` per phase |
| Create | `tests/test_eager_unload.py` | P1: integration test for adapter unload |
| Create | `libs/shared/tests/test_decompose_templates.py` | P2: template rendering tests |
| Create | `tests/test_skip_decompose.py` | P3: gating logic tests |
| Create | `libs/inference/tests/test_thinking_budget.py` | P4: thinking budget tests |
| Create | `tests/test_continuation.py` | P5: runner-managed continuation tests |

---

## Task 1: Eager Adapter Unload (P1)

**Files:**
- Modify: `scripts/rune_runner.py` (9 call sites)
- Test: `tests/test_eager_unload.py`

This task adds `provider.unload_adapter()` + `torch.cuda.empty_cache()` after every `run_iteration()` call site that uses an adapter. Prevents 60+ adapters accumulating within a phase and causing CUDA OOM.

- [ ] **Step 1: Write the failing test**

Create `tests/test_eager_unload.py`:

```python
"""Tests that adapter unload is called after every run_iteration call site."""

from __future__ import annotations

import ast
import re

from pathlib import Path


RUNNER_PATH = Path("scripts/rune_runner.py")


def _get_run_iteration_call_sites() -> list[int]:
    """Find all line numbers where run_iteration() is called."""
    source = RUNNER_PATH.read_text()
    tree = ast.parse(source)
    lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "run_iteration":
                lines.append(node.lineno)
            elif isinstance(func, ast.Attribute) and func.attr == "run_iteration":
                lines.append(node.lineno)
    return sorted(lines)


def _find_unload_after_line(source_lines: list[str], call_line: int) -> bool:
    """Check that unload_adapter appears within 30 lines after a run_iteration call."""
    start = call_line  # 0-indexed: call_line is 1-indexed
    end = min(start + 30, len(source_lines))
    window = "\n".join(source_lines[start:end])
    return "unload_adapter" in window


def test_every_run_iteration_has_unload() -> None:
    """Every run_iteration() call site must have unload_adapter within 30 lines."""
    source = RUNNER_PATH.read_text()
    source_lines = source.splitlines()
    call_sites = _get_run_iteration_call_sites()

    assert len(call_sites) >= 9, f"Expected >=9 call sites, found {len(call_sites)}"

    missing = []
    for line_no in call_sites:
        if not _find_unload_after_line(source_lines, line_no):
            missing.append(line_no)

    assert not missing, (
        f"run_iteration() call sites missing unload_adapter within 30 lines: "
        f"lines {missing}"
    )


def test_cleanup_phase_adapters_still_exists() -> None:
    """_cleanup_phase_adapters() remains as a safety net."""
    source = RUNNER_PATH.read_text()
    assert "_cleanup_phase_adapters" in source
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_eager_unload.py -v`
Expected: FAIL on `test_every_run_iteration_has_unload` — most call sites lack `unload_adapter`

- [ ] **Step 3: Add eager unload helper function**

Add a helper near `_cleanup_phase_adapters()` (around line 548) in `scripts/rune_runner.py`:

```python
async def _eager_unload(adapter_id: str | None) -> None:
    """Unload a single adapter immediately after use to prevent VRAM accumulation."""
    if not adapter_id:
        return
    from inference import get_provider  # noqa: PLC0415

    provider = get_provider()
    try:
        await provider.unload_adapter(adapter_id)
    except Exception:
        logger.warning("Failed to eager-unload adapter %s", adapter_id, exc_info=True)
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        pass
    else:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
```

**Performance note:** `torch.cuda.empty_cache()` triggers a CUDA sync, so calling it 60+ times per phase adds latency. This is acceptable — adapter load is ~100ms vs ~120s for generation — but a future optimization could call `empty_cache()` only between subtasks (not after every unload) by splitting `_eager_unload` into unload-only and unload+flush variants. Out of scope for P1.

- [ ] **Step 4: Add unload after decompose call site (line ~793)**

After the `state = await run_iteration(...)` call at line 793, add:

```python
            await _eager_unload(adapter_id)
```

- [ ] **Step 5: Add unload after decompose cycle-fix call site (line ~932)**

After the `retry_state = await run_iteration(...)` call at line 932, add:

```python
                await _eager_unload(cycle_adapter_id)
```

- [ ] **Step 6: Add unload after plan call site (line ~1057)**

Inside `_plan_subtask()`, after the `plan_state = await run_iteration(...)` call at line 1057. This already has manual unload at lines 1083-1089, but replace with `_eager_unload` for consistency. Remove the manual loop at 1083-1089 and instead add after the `return` block:

Actually, the plan phase does parallel `asyncio.gather()` then unloads all plan adapters in a loop (lines 1080-1089). This pattern is correct — unloading inside the parallel coroutines could race. Keep the existing loop but replace its body with `_eager_unload`:

```python
            for _name, _text, _paid in plan_results:
                await _eager_unload(_paid)
```

Replace lines 1083-1089 (the existing manual unload code).

- [ ] **Step 7: Add unload after diagnose call site within code retry (line ~1253)**

After the `diag_state = await run_iteration(...)` call at line 1253, add:

```python
                            await _eager_unload(diag_aid)
```

- [ ] **Step 8: Add unload after code attempt call site (line ~1400)**

After the `code_state = await run_iteration(...)` call at line 1400, add:

```python
                await _eager_unload(code_adapter_id)
```

- [ ] **Step 9: Add unload after integrate call site (line ~1715)**

After the `integrate_state = await run_iteration(...)` call at line 1715, add:

```python
            await _eager_unload(integrate_aid)
```

- [ ] **Step 10: Add unload after phase 5 diagnose call site (line ~1850)**

After the `diagnose_state = await run_iteration(...)` call at line 1850, add:

```python
            await _eager_unload(diagnose_aid)
```

- [ ] **Step 11: Add unload after phase 5 repair call site (line ~1937)**

After the `repair_state = await run_iteration(...)` call at line 1937, add:

```python
                await _eager_unload(repair_aid)
```

- [ ] **Step 12: Add unload after phase 5 reintegrate call site (line ~2005)**

After the `reintegrate_state = await run_iteration(...)` call at line 2005, add:

```python
            await _eager_unload(reintegrate_aid)
```

- [ ] **Step 13: Run test to verify it passes**

Run: `uv run pytest tests/test_eager_unload.py -v`
Expected: PASS

- [ ] **Step 14: Run existing tests to check for regressions**

Run: `uv run pytest tests/ services/rune-agent/tests/ -x -v`
Expected: All pass

- [ ] **Step 15: Lint and type check**

Run: `uv run ruff check scripts/rune_runner.py && uv run mypy scripts/rune_runner.py`
Expected: Clean

- [ ] **Step 16: Commit**

```bash
git add scripts/rune_runner.py tests/test_eager_unload.py
git commit -m "fix(runner): eager adapter unload after every run_iteration call

Prevents 60+ adapters accumulating within a phase and causing CUDA OOM.
Adds _eager_unload() helper called at all 9 run_iteration() call sites."
```

---

## Task 2: Decompose Prompt Improvement (P2)

**Files:**
- Modify: `libs/shared/src/shared/templates/decompose.j2`
- Modify: `libs/shared/src/shared/templates/prompt_decompose.j2`
- Modify: `libs/shared/src/shared/templates/prompt_decompose_concise.j2`
- Create: `libs/shared/tests/test_decompose_templates.py`

Adds chain-of-thought suppression, simple-task few-shot examples, and a negative example to prevent the model from leaking reasoning steps as subtask entries.

- [ ] **Step 1: Write the failing test**

Create `libs/shared/tests/test_decompose_templates.py`:

```python
"""Tests for decompose template chain-of-thought suppression."""

from __future__ import annotations

from pathlib import Path

TEMPLATE_DIR = Path("libs/shared/src/shared/templates")


def test_decompose_trajectory_has_cot_suppression() -> None:
    """decompose.j2 must contain chain-of-thought suppression."""
    content = (TEMPLATE_DIR / "decompose.j2").read_text()
    assert "Do NOT include your chain-of-thought" in content or "do NOT include" in content.lower()


def test_decompose_trajectory_has_simple_task_example() -> None:
    """decompose.j2 must have a simple single-function example."""
    content = (TEMPLATE_DIR / "decompose.j2").read_text()
    assert "Write a function" in content or "write a function" in content


def test_decompose_trajectory_has_negative_example() -> None:
    """decompose.j2 must have a BAD/negative example."""
    content = (TEMPLATE_DIR / "decompose.j2").read_text()
    assert "BAD" in content


def test_prompt_decompose_has_cot_suppression() -> None:
    """prompt_decompose.j2 must contain chain-of-thought suppression."""
    content = (TEMPLATE_DIR / "prompt_decompose.j2").read_text()
    assert "No preamble" in content or "no preamble" in content


def test_prompt_decompose_has_negative_example() -> None:
    """prompt_decompose.j2 must have a negative example."""
    content = (TEMPLATE_DIR / "prompt_decompose.j2").read_text()
    assert "BAD" in content


def test_prompt_decompose_concise_has_cot_suppression() -> None:
    """prompt_decompose_concise.j2 must contain chain-of-thought suppression."""
    content = (TEMPLATE_DIR / "prompt_decompose_concise.j2").read_text()
    assert "No preamble" in content or "no preamble" in content or "ONLY" in content
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest libs/shared/tests/test_decompose_templates.py -v`
Expected: FAIL on multiple assertions (templates lack these elements)

- [ ] **Step 3: Update `decompose.j2` trajectory template**

Replace the full content of `libs/shared/src/shared/templates/decompose.j2` with:

```
ROLE: project-decomposer
PROJECT: {{ project[:1200] }}
METHODOLOGY: Decompose into 3-6 independent subtasks that can be implemented
and tested in isolation. Each subtask should:
- Have a clear, focused scope (one layer or feature)
- Produce testable output independently
- Minimize dependencies on other subtasks
Order by dependency: data/models first, then logic, then interface, then integration.

Output ONLY a numbered list of subtasks. No preamble, no analysis, no reasoning.
Do NOT include your chain-of-thought as subtask entries.
Each line: "N. name — description [depends: none]"

EXAMPLE 1 — Web API (5 subtasks):
1. Data models — Dataclasses for User, Post, Comment with validation and serialization [depends: none]
2. Storage layer — In-memory repository with CRUD, filtering, and pagination [depends: 1]
3. Route handlers — GET/POST/PUT/DELETE endpoints with request parsing and error responses [depends: 2]
4. Input validation — Schema validators for create/update payloads with typed error messages [depends: 1]
5. CLI entry point — argparse-based launcher with host/port/debug flags [depends: 3, 4]

EXAMPLE 2 — Single function (2 subtasks):
1. implement_is_prime — Core primality check with edge cases [depends: none]
2. add_tests — Unit tests for primes, non-primes, edge cases [depends: 1]

EXAMPLE 3 — Merge two sorted lists (3 subtasks):
1. implement_merge — Two-pointer merge of sorted lists [depends: none]
2. handle_edge_cases — Empty lists, single-element, duplicates [depends: 1]
3. add_tests — Comprehensive test cases [depends: 1, 2]

BAD (do not do this):
1. Analyze the Request — ...
2. Numbered list? Yes
3. Never code? Yes
These are reasoning steps, NOT subtasks. Never output these.

ANTI-PATTERNS (do NOT produce these):
- Single monolithic subtask covering everything (too coarse, untestable in isolation)
- 10+ micro-subtasks with heavy overlap (too fine, creates dependency spaghetti)
- Chain-of-thought or reasoning steps disguised as subtask entries

Output a numbered list with dependency declarations:
1. name — one-line description [depends: none]
2. name — one-line description [depends: 1]
3. name — one-line description [depends: 1, 2]
```

- [ ] **Step 4: Update `prompt_decompose.j2`**

Replace the full content of `libs/shared/src/shared/templates/prompt_decompose.j2` with:

```
Decompose this project into 3-6 subtasks with dependencies.
Do NOT write code. Output ONLY a numbered list in this exact format.
No preamble, no analysis, no reasoning. Just the list.
1. name — one-line description [depends: none]
2. name — one-line description [depends: 1]
3. name — one-line description [depends: 1, 2]

Example output:
1. Data models — Dataclasses with validation and serialization [depends: none]
2. Storage layer — SQLite-backed repository with CRUD operations [depends: 1]
3. Business logic — Service class enforcing domain rules [depends: 1, 2]
4. CLI interface — argparse subcommands for user interaction [depends: 3]

BAD (do not do this):
1. Analyze the request — determine requirements
2. Numbered list? Yes
These are reasoning steps, NOT subtasks. Never output these.

Project: {{ task_description }}
```

- [ ] **Step 5: Update `prompt_decompose_concise.j2`**

Replace the full content of `libs/shared/src/shared/templates/prompt_decompose_concise.j2` with:

```
Decompose into 3-6 subtasks with dependencies. Output ONLY a numbered list.
No preamble, no analysis, no reasoning.
1. name — description [depends: none]
2. name — description [depends: 1]

BAD: "Analyze the request" / "Numbered list? Yes" — these are reasoning, NOT subtasks.

Project: {{ task_description }}
```

- [ ] **Step 6: Run test to verify it passes**

Run: `uv run pytest libs/shared/tests/test_decompose_templates.py -v`
Expected: PASS

- [ ] **Step 7: Run existing template tests**

Run: `uv run pytest libs/shared/tests/ -x -v`
Expected: All pass

- [ ] **Step 8: Commit**

```bash
git add libs/shared/src/shared/templates/decompose.j2 \
       libs/shared/src/shared/templates/prompt_decompose.j2 \
       libs/shared/src/shared/templates/prompt_decompose_concise.j2 \
       libs/shared/tests/test_decompose_templates.py
git commit -m "feat(decompose): add chain-of-thought suppression and few-shot examples

Prevents model from leaking reasoning steps as subtask entries.
Adds simple-task examples (2-3 subtasks) and negative examples."
```

---

## Task 3: Task-Complexity Gating (P3)

**Files:**
- Modify: `scripts/rune_runner.py` — add `_should_skip_decompose()` + skip logic
- Modify: `libs/shared/src/shared/pipeline_config.py` — add `DecomposeConfig`
- Test: `tests/test_skip_decompose.py`
- Test: `libs/shared/tests/test_pipeline_config.py` (existing, add new assertion)

Simple tasks (short prompt, single-function signals) skip decompose entirely, producing a single `implementation` subtask.

- [ ] **Step 1: Write the failing test for `DecomposeConfig`**

Add to `libs/shared/tests/test_pipeline_config.py`:

```python
def test_decompose_config_defaults() -> None:
    cfg = default_config()
    assert hasattr(cfg, "decompose")
    assert cfg.decompose.skip_threshold == 200


def test_decompose_config_override() -> None:
    cfg = default_config()
    updated = cfg.override(**{"decompose.skip_threshold": 100})
    assert updated.decompose.skip_threshold == 100


def test_decompose_config_round_trip(tmp_path: Path) -> None:
    cfg = default_config()
    path = cfg.save(tmp_path / "test_decompose.json")
    loaded = load_config(path)
    assert loaded.decompose.skip_threshold == 200
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest libs/shared/tests/test_pipeline_config.py::test_decompose_config_defaults -v`
Expected: FAIL — `PipelineConfig` has no `decompose` attribute

- [ ] **Step 3: Add `DecomposeConfig` to `pipeline_config.py`**

In `libs/shared/src/shared/pipeline_config.py`, add after `CalibrationConfig`:

```python
@dataclass(frozen=True)
class DecomposeConfig:
    """Decompose phase settings."""

    skip_threshold: int = 200
```

Update `PipelineConfig` to include it:

```python
@dataclass(frozen=True)
class PipelineConfig:
    """Top-level pipeline configuration."""

    adapter: AdapterConfig = field(default_factory=AdapterConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    trajectory: TrajectoryConfig = field(default_factory=TrajectoryConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    decompose: DecomposeConfig = field(default_factory=DecomposeConfig)
```

Update `_from_dict()` to reconstruct it:

```python
def _from_dict(d: dict[str, Any]) -> PipelineConfig:
    cal = d.get("calibration", {})
    if "scaling_range" in cal and isinstance(cal["scaling_range"], list):
        cal["scaling_range"] = tuple(cal["scaling_range"])
    return PipelineConfig(
        adapter=AdapterConfig(**d.get("adapter", {})),
        generation=GenerationConfig(**d.get("generation", {})),
        prompt=PromptConfig(**d.get("prompt", {})),
        trajectory=TrajectoryConfig(**d.get("trajectory", {})),
        calibration=CalibrationConfig(**cal),
        decompose=DecomposeConfig(**d.get("decompose", {})),
    )
```

- [ ] **Step 4: Run pipeline_config tests to verify they pass**

Run: `uv run pytest libs/shared/tests/test_pipeline_config.py -v`
Expected: PASS

- [ ] **Step 5: Write the failing test for `_should_skip_decompose()`**

Create `tests/test_skip_decompose.py`:

```python
"""Tests for task-complexity gating (_should_skip_decompose)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from rune_runner import _should_skip_decompose  # type: ignore[import-not-found]


def test_short_function_prompt_skips() -> None:
    """Short 'write a function' prompt should skip decompose."""
    assert _should_skip_decompose("Write a function to check if a number is prime") is True


def test_long_prompt_does_not_skip() -> None:
    """Prompt over threshold words should not skip."""
    long_prompt = "Build a web application that " + " ".join(["word"] * 250)
    assert _should_skip_decompose(long_prompt) is False


def test_short_prompt_without_function_signal_does_not_skip() -> None:
    """Short prompt without function signals should not skip."""
    assert _should_skip_decompose("Build a REST API with three endpoints") is False


def test_implement_signal_skips() -> None:
    """'implement a function' signal should skip for short prompts."""
    assert _should_skip_decompose("Implement a function that returns fibonacci numbers") is True


def test_custom_threshold() -> None:
    """Custom threshold should be respected."""
    prompt = "Write a function to sort a list"  # ~7 words
    assert _should_skip_decompose(prompt, threshold=5) is False
    assert _should_skip_decompose(prompt, threshold=50) is True
```

- [ ] **Step 6: Run test to verify it fails**

Run: `uv run pytest tests/test_skip_decompose.py -v`
Expected: FAIL — `_should_skip_decompose` doesn't exist

- [ ] **Step 7: Implement `_should_skip_decompose()` in `rune_runner.py`**

Add near line 237 (before `_parse_subtask_list`):

```python
def _should_skip_decompose(project_prompt: str, threshold: int = 200) -> bool:
    """Return True if the task is simple enough to skip decomposition.

    Simple tasks are short prompts (under threshold words) that contain
    single-function signals like 'write a function' or 'implement a method'.
    """
    word_count = len(project_prompt.split())
    if word_count > threshold:
        return False
    single_fn_signals = [
        "write a function",
        "implement a function",
        "write a method",
        "implement a method",
        "create a function",
        "def ",
    ]
    return any(s in project_prompt.lower() for s in single_fn_signals)
```

- [ ] **Step 8: Wire skip logic into `run_phased_pipeline()`**

In `run_phased_pipeline()`, before the decompose evolution loop (around line 720), add the skip check. Find the line where the decompose loop starts and add before it:

```python
    # Task-complexity gating: skip decompose for simple single-function tasks
    config = load_config()
    if _should_skip_decompose(project_prompt, threshold=config.decompose.skip_threshold):
        logger.info("Simple task detected — skipping decompose phase")
        subtasks = [{"name": "implementation", "description": project_prompt[:200], "depends_on": []}]
        phase_results["decompose"] = {
            "subtasks": subtasks,
            "adapter_id": None,
            "iterations": 0,
            "best_score": 1.0,
        }
    else:
        # ... existing decompose loop (indent existing code into this else block)
```

Add the import near the top of the function:

```python
    from shared.pipeline_config import load_config  # noqa: PLC0415
```

- [ ] **Step 9: Run tests to verify they pass**

Run: `uv run pytest tests/test_skip_decompose.py -v`
Expected: PASS

- [ ] **Step 10: Run full test suite for regressions**

Run: `uv run pytest tests/ libs/shared/tests/ -x -v`
Expected: All pass

- [ ] **Step 11: Lint and type check**

Run: `uv run ruff check scripts/rune_runner.py libs/shared/src/shared/pipeline_config.py`
Expected: Clean

- [ ] **Step 12: Commit**

```bash
git add scripts/rune_runner.py \
       libs/shared/src/shared/pipeline_config.py \
       libs/shared/tests/test_pipeline_config.py \
       tests/test_skip_decompose.py
git commit -m "feat(decompose): skip decomposition for simple single-function tasks

Adds _should_skip_decompose() gating with configurable word threshold
(default 200). Simple tasks produce a single 'implementation' subtask
instead of triggering 16-30 subtask decomposition explosion."
```

---

## Task 4: Thinking Token Budget (P4)

**Files:**
- Modify: `libs/inference/src/inference/provider.py` — add `thinking_budget` to ABC
- Modify: `libs/inference/src/inference/transformers_provider.py` — implement budget logic
- Modify: `libs/inference/src/inference/vllm_provider.py` — accept param
- Modify: `libs/inference/src/inference/ollama_provider.py` — accept param
- Modify: `libs/inference/src/inference/llamacpp_provider.py` — accept param
- Modify: `libs/shared/src/shared/pipeline_config.py` — add `thinking_budget` to `GenerationConfig`
- Modify: `services/rune-agent/src/rune_agent/nodes.py` — pass budget per phase
- Create: `libs/inference/tests/test_thinking_budget.py`

Gives thinking tokens their own allocation so `<think>` blocks don't starve response tokens.

- [ ] **Step 1: Write the failing test for `thinking_budget` in `GenerationConfig`**

Add to `libs/shared/tests/test_pipeline_config.py`:

```python
def test_thinking_budget_default() -> None:
    cfg = default_config()
    assert cfg.generation.thinking_budget == 512


def test_thinking_budget_override() -> None:
    cfg = default_config()
    updated = cfg.override(**{"generation.thinking_budget": 256})
    assert updated.generation.thinking_budget == 256
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest libs/shared/tests/test_pipeline_config.py::test_thinking_budget_default -v`
Expected: FAIL — `GenerationConfig` has no `thinking_budget`

- [ ] **Step 3: Add `thinking_budget` to `GenerationConfig`**

In `libs/shared/src/shared/pipeline_config.py`, update `GenerationConfig`:

```python
@dataclass(frozen=True)
class GenerationConfig:
    """LLM generation settings."""

    temperature: float = 0.3
    max_tokens: int = 2048
    repetition_penalty: float = 1.1
    top_p: float = 0.9
    thinking_budget: int = 512
```

- [ ] **Step 4: Run config test to verify it passes**

Run: `uv run pytest libs/shared/tests/test_pipeline_config.py -v`
Expected: PASS

- [ ] **Step 5: Write the failing test for provider ABC**

Create `libs/inference/tests/test_thinking_budget.py`:

```python
"""Tests for thinking token budget in inference providers."""

from __future__ import annotations

import inspect

from inference.provider import InferenceProvider


def test_provider_abc_has_thinking_budget_param() -> None:
    """InferenceProvider.generate() must accept thinking_budget."""
    sig = inspect.signature(InferenceProvider.generate)
    assert "thinking_budget" in sig.parameters
    assert sig.parameters["thinking_budget"].default == 0
```

- [ ] **Step 6: Run test to verify it fails**

Run: `uv run pytest libs/inference/tests/test_thinking_budget.py::test_provider_abc_has_thinking_budget_param -v`
Expected: FAIL — `generate()` has no `thinking_budget` param

- [ ] **Step 7: Add `thinking_budget` to provider ABC**

In `libs/inference/src/inference/provider.py`, update the `generate()` signature:

```python
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        model: str,
        adapter_id: str | None = None,
        max_tokens: int = 4096,
        system_prompt: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
        enable_thinking: bool = True,
        thinking_budget: int = 0,
    ) -> GenerationResult:
```

Add to the docstring:

```
            thinking_budget: Extra token allocation for thinking/reasoning.
                When > 0, effective max_new_tokens = max_tokens + thinking_budget.
                Only meaningful when enable_thinking=True.
```

- [ ] **Step 8: Run ABC test to verify it passes**

Run: `uv run pytest libs/inference/tests/test_thinking_budget.py -v`
Expected: PASS

- [ ] **Step 9: Implement thinking budget in `TransformersProvider.generate()`**

In `libs/inference/src/inference/transformers_provider.py`, update the `generate()` signature to add `thinking_budget: int = 0`:

```python
    async def generate(
        self,
        prompt: str,
        model: str,
        adapter_id: str | None = None,
        max_tokens: int = 4096,
        system_prompt: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
        enable_thinking: bool = True,
        thinking_budget: int = 0,
    ) -> GenerationResult:
```

Update `max_new_tokens` assignment (line ~210):

Change:
```python
            "max_new_tokens": max_tokens,
```

To:
```python
            "max_new_tokens": max_tokens + thinking_budget,
```

Update `finish_reason` logic (line ~240):

Change:
```python
        finish_reason = "length" if new_token_count >= max_tokens else "stop"
```

To:
```python
        # When thinking_budget is active, only count non-thinking tokens for truncation
        if thinking_budget > 0 and thinking:
            thinking_token_count = sum(
                len(self._tokenizer.encode(p, add_special_tokens=False))
                for p in thinking_parts
            )
            response_token_count = new_token_count - thinking_token_count
            finish_reason = "length" if response_token_count >= max_tokens else "stop"
        else:
            finish_reason = "length" if new_token_count >= (max_tokens + thinking_budget) else "stop"
```

- [ ] **Step 10: Update `vllm_provider.py` signature**

In `libs/inference/src/inference/vllm_provider.py`, add `thinking_budget: int = 0` to `generate()`:

```python
    async def generate(
        self,
        prompt: str,
        model: str,
        adapter_id: str | None = None,
        max_tokens: int = 4096,
        system_prompt: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
        enable_thinking: bool = True,
        thinking_budget: int = 0,
    ) -> GenerationResult:
```

- [ ] **Step 11: Update `ollama_provider.py` signature**

Same change — add `thinking_budget: int = 0` to `generate()`.

- [ ] **Step 12: Update `llamacpp_provider.py` signature**

Same change — add `thinking_budget: int = 0` to `generate()`.

- [ ] **Step 13: Write the failing test for `generate_node` passing thinking_budget**

Add to `libs/inference/tests/test_thinking_budget.py`:

```python
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any

from inference import GenerationResult


def _make_result(text: str, finish: str = "stop") -> GenerationResult:
    return GenerationResult(
        text=text, model="test", adapter_id=None, token_count=10,
        finish_reason=finish,
    )


async def test_generate_node_passes_thinking_budget_for_text_phase() -> None:
    """generate_node passes thinking_budget=512 for decompose phase."""
    provider = MagicMock()
    provider.generate = AsyncMock(return_value=_make_result("1. subtask"))

    state: dict[str, Any] = {
        "task_description": "Write something",
        "task_type": "project",
        "test_suite": "",
        "adapter_ids": [],
        "attempt_count": 0,
        "generated_code": "",
        "stdout": "",
        "stderr": "",
        "exit_code": 0,
        "phase": "decompose",
        "prompt_context": None,
        "finish_reason": None,
    }

    with patch("rune_agent.nodes.get_provider", return_value=provider):
        await generate_node(state)

    call_kwargs = provider.generate.call_args.kwargs
    assert call_kwargs.get("thinking_budget", 0) == 512


async def test_generate_node_passes_zero_budget_for_code_phase() -> None:
    """generate_node passes thinking_budget=0 for code phase."""
    provider = MagicMock()
    provider.generate = AsyncMock(return_value=_make_result("def f(): pass"))

    state: dict[str, Any] = {
        "task_description": "Write something",
        "task_type": "project",
        "test_suite": "",
        "adapter_ids": [],
        "attempt_count": 0,
        "generated_code": "",
        "stdout": "",
        "stderr": "",
        "exit_code": 0,
        "phase": "code",
        "prompt_context": None,
        "finish_reason": None,
    }

    with patch("rune_agent.nodes.get_provider", return_value=provider):
        await generate_node(state)

    call_kwargs = provider.generate.call_args.kwargs
    assert call_kwargs.get("thinking_budget", 0) == 0
```

Import `generate_node` at top of this test file:

```python
from rune_agent.nodes import generate_node
```

- [ ] **Step 14: Run test to verify it fails**

Run: `uv run pytest libs/inference/tests/test_thinking_budget.py::test_generate_node_passes_thinking_budget_for_text_phase -v`
Expected: FAIL — `generate_node` doesn't pass `thinking_budget`

- [ ] **Step 15: Update `generate_node` to pass `thinking_budget` per phase**

In `services/rune-agent/src/rune_agent/nodes.py`, update `generate_node()`. After line 239 (`enable_thinking = phase in _TEXT_ONLY_PHASES`), add:

```python
    thinking_budget = 512 if enable_thinking else 0
```

Then update the `provider.generate()` call to include it:

```python
    result: GenerationResult = await provider.generate(
        prompt=user_prompt,
        model=model,
        adapter_id=adapter_id,
        max_tokens=max_tokens,
        system_prompt=system_prompt,
        enable_thinking=enable_thinking,
        thinking_budget=thinking_budget,
    )
```

- [ ] **Step 16: Run all thinking budget tests**

Run: `uv run pytest libs/inference/tests/test_thinking_budget.py -v`
Expected: PASS

- [ ] **Step 17: Run existing inference and node tests**

Run: `uv run pytest libs/inference/tests/ services/rune-agent/tests/ -x -v`
Expected: All pass

- [ ] **Step 18: Lint and type check**

Run: `uv run ruff check libs/inference/ services/rune-agent/ libs/shared/ && uv run mypy libs/inference/ services/rune-agent/ libs/shared/`
Expected: Clean

- [ ] **Step 19: Commit**

```bash
git add libs/inference/src/inference/provider.py \
       libs/inference/src/inference/transformers_provider.py \
       libs/inference/src/inference/vllm_provider.py \
       libs/inference/src/inference/ollama_provider.py \
       libs/inference/src/inference/llamacpp_provider.py \
       libs/shared/src/shared/pipeline_config.py \
       libs/shared/tests/test_pipeline_config.py \
       services/rune-agent/src/rune_agent/nodes.py \
       libs/inference/tests/test_thinking_budget.py
git commit -m "feat(inference): add thinking token budget to prevent response starvation

Thinking tokens get their own allocation (default 512) so <think> blocks
don't consume the response budget. effective_max = max_tokens + thinking_budget.
finish_reason='length' only fires when response tokens exceed max_tokens."
```

---

## Task 5: Runner-Managed Continuation for Truncated Outputs (P5)

**Files:**
- Modify: `scripts/rune_runner.py` — inner continuation loop after code generation, replaces existing `is_truncated` branch
- Create: `tests/test_continuation.py`

When `generate` produces a truncated output (`finish_reason == "length"`), the runner accumulates partial output locally and re-invokes `run_iteration()` with a fresh adapter encoding the accumulated context. Max 3 continuations per subtask. Continuations do NOT count as retry attempts.

**Design note:** Continuation is managed entirely in the runner's outer loop, not via new LangGraph nodes. `run_iteration()` builds `initial_state` from scratch every call (line 379), so graph-internal state like `accumulated_code` would be lost between invocations. The runner already owns the hypernetwork call cycle and is the natural place for this loop. `finish_reason` is already returned by `generate_node()` (line 269 of `nodes.py`) and propagated through the graph state, so the runner can read it from the returned state dict.

- [ ] **Step 1: Write the failing test for `_run_continuation_loop()`**

Create `tests/test_continuation.py`. The tests mock `run_iteration_fn`, `render_trajectory_fn`, `run_hypernetwork_fn`, `load_adapter_fn`, and `unload_adapter_fn` — the injectable callables that `_run_continuation_loop()` will accept:

```python
"""Tests for runner-managed multi-turn continuation."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from rune_runner import _run_continuation_loop  # type: ignore[import-not-found]


def _make_state(
    finish_reason: str = "stop",
    generated_code: str = "def f(): pass",
) -> dict[str, Any]:
    """Build a minimal state dict as returned by run_iteration."""
    return {
        "finish_reason": finish_reason,
        "generated_code": generated_code,
        "tests_passed": finish_reason == "stop",
        "stdout": "",
        "stderr": "",
        "exit_code": 0,
        "test_count": 1 if finish_reason == "stop" else 0,
        "tests_ran": finish_reason == "stop",
    }


@pytest.mark.asyncio
async def test_no_continuation_on_stop() -> None:
    """finish_reason='stop' returns immediately with no continuation calls."""
    initial = _make_state(finish_reason="stop", generated_code="def f(): pass")

    async def mock_run_iteration(**kw: Any) -> dict:
        pytest.fail("run_iteration should not be called when finish_reason='stop'")

    final_state, accumulated = await _run_continuation_loop(
        initial_state=initial,
        run_iteration_fn=mock_run_iteration,
        render_trajectory_fn=lambda **kw: "traj",
        run_hypernetwork_fn=lambda **kw: None,
        load_adapter_fn=lambda *a: None,
        unload_adapter_fn=lambda *a: None,
    )
    assert accumulated == "def f(): pass"
    assert final_state["finish_reason"] == "stop"


@pytest.mark.asyncio
async def test_single_continuation_then_stop() -> None:
    """One truncation followed by a complete generation."""
    initial = _make_state(finish_reason="length", generated_code="import os\n")
    call_count = 0

    async def mock_run_iteration(**kw: Any) -> dict:
        nonlocal call_count
        call_count += 1
        return _make_state(finish_reason="stop", generated_code="def main(): pass")

    final_state, accumulated = await _run_continuation_loop(
        initial_state=initial,
        run_iteration_fn=mock_run_iteration,
        render_trajectory_fn=lambda **kw: "traj",
        run_hypernetwork_fn=lambda **kw: "/fake/path",
        load_adapter_fn=lambda *a: None,
        unload_adapter_fn=lambda *a: None,
    )
    assert call_count == 1
    assert "import os" in accumulated
    assert "def main(): pass" in accumulated


@pytest.mark.asyncio
async def test_continuation_caps_at_three() -> None:
    """After 3 continuations, loop stops even if still truncated."""
    initial = _make_state(finish_reason="length", generated_code="# part 0")
    call_count = 0

    async def mock_run_iteration(**kw: Any) -> dict:
        nonlocal call_count
        call_count += 1
        return _make_state(finish_reason="length", generated_code=f"# part {call_count}")

    final_state, accumulated = await _run_continuation_loop(
        initial_state=initial,
        run_iteration_fn=mock_run_iteration,
        render_trajectory_fn=lambda **kw: "traj",
        run_hypernetwork_fn=lambda **kw: "/fake/path",
        load_adapter_fn=lambda *a: None,
        unload_adapter_fn=lambda *a: None,
    )
    assert call_count == 3
    assert "# part 0" in accumulated
    assert "# part 3" in accumulated


@pytest.mark.asyncio
async def test_continuation_passes_accumulated_to_trajectory() -> None:
    """render_trajectory_fn receives accumulated code so far."""
    initial = _make_state(finish_reason="length", generated_code="import os")
    captured_existing: list[str] = []

    def mock_render(**kw: Any) -> str:
        captured_existing.append(kw.get("existing_code", ""))
        return "traj"

    call_count = 0

    async def mock_run_iteration(**kw: Any) -> dict:
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            return _make_state(finish_reason="length", generated_code="def f(): pass")
        return _make_state(finish_reason="stop", generated_code="# done")

    await _run_continuation_loop(
        initial_state=initial,
        run_iteration_fn=mock_run_iteration,
        render_trajectory_fn=mock_render,
        run_hypernetwork_fn=lambda **kw: "/fake/path",
        load_adapter_fn=lambda *a: None,
        unload_adapter_fn=lambda *a: None,
    )
    # First continuation sees initial code
    assert "import os" in captured_existing[0]
    # Second continuation sees accumulated code
    assert "def f(): pass" in captured_existing[1]


@pytest.mark.asyncio
async def test_unload_called_after_each_continuation() -> None:
    """unload_adapter_fn is called after every continuation iteration."""
    initial = _make_state(finish_reason="length", generated_code="# start")
    unload_calls: list[str] = []
    call_count = 0

    async def mock_run_iteration(**kw: Any) -> dict:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            return _make_state(finish_reason="length", generated_code=f"# part {call_count}")
        return _make_state(finish_reason="stop", generated_code="# end")

    async def mock_unload(aid: str | None) -> None:
        unload_calls.append(aid or "none")

    await _run_continuation_loop(
        initial_state=initial,
        run_iteration_fn=mock_run_iteration,
        render_trajectory_fn=lambda **kw: "traj",
        run_hypernetwork_fn=lambda **kw: "/fake/path",
        load_adapter_fn=lambda *a: None,
        unload_adapter_fn=mock_unload,
    )
    assert len(unload_calls) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_continuation.py -v`
Expected: FAIL — `_run_continuation_loop` does not exist in `rune_runner`

- [ ] **Step 3: Implement `_run_continuation_loop()` in `rune_runner.py`**

Add near line 350 (before `run_iteration()`):

```python
_MAX_CONTINUATIONS = 3


async def _run_continuation_loop(
    initial_state: dict[str, Any],
    *,
    run_iteration_fn: Any,
    render_trajectory_fn: Any,
    run_hypernetwork_fn: Any,
    load_adapter_fn: Any,
    unload_adapter_fn: Any,
    subtask: dict[str, Any] | None = None,
    attempt: int = 0,
    plan: str = "",
    dep_interfaces: str = "",
    project_prompt: str = "",
    adapter_dir: Any = None,
    base_model_id: str = "",
    checkpoint_path: str | None = None,
    device: str = "cuda",
    adapter_scaling: float = 0.16,
    adapter_max_length: int = 512,
    pool: Any = None,
    session_id: str = "",
    iteration_base: int = 0,
    project_label: str = "",
    graph: Any = None,
) -> tuple[dict[str, Any], str]:
    """Run continuation loop for truncated code generation.

    Returns (final_state, accumulated_code). Does not count as retry attempts.
    """
    accumulated = initial_state.get("generated_code", "")
    state = initial_state
    cont_count = 0

    while state.get("finish_reason") == "length" and cont_count < _MAX_CONTINUATIONS:
        cont_count += 1
        logger.info(
            "  Subtask '%s' truncated (cont %d/%d), accumulating...",
            (subtask or {}).get("name", "?"),
            cont_count,
            _MAX_CONTINUATIONS,
        )
        traj = render_trajectory_fn(
            phase="code_continue",
            subtask=subtask,
            attempt=cont_count,
            max_retries=_MAX_CONTINUATIONS,
            plan=plan,
            existing_code=accumulated[:1200],
            project=project_prompt,
            dependency_interfaces=dep_interfaces,
        )
        subtask_name = (subtask or {}).get("name", "unknown")
        cont_adapter_dir = (
            str(adapter_dir / f"phase3_cont_{_safe_adapter_id(subtask_name)}_v{attempt}_c{cont_count}")
            if adapter_dir
            else None
        )
        cont_adapter_path = run_hypernetwork_fn(
            trajectory_text=traj,
            output_dir=cont_adapter_dir,
            base_model_id=base_model_id,
            checkpoint_path=checkpoint_path,
            device=device,
            scaling_factor=adapter_scaling,
            max_length=adapter_max_length,
            pool=pool,
        )
        cont_aid: str | None = None
        if cont_adapter_path:
            cont_aid = f"phase3-cont-{_safe_adapter_id(subtask_name)}-v{attempt}-c{cont_count}"
            await load_adapter_fn(cont_aid, cont_adapter_path)

        state = await run_iteration_fn(
            graph=graph,
            project_prompt=project_prompt,
            adapter_id=cont_aid,
            session_id=session_id,
            iteration=iteration_base + cont_count,
            phase="code_continue",
            prompt_context={
                "subtask_name": subtask_name,
                "project_label": project_label,
            },
        )
        await unload_adapter_fn(cont_aid)

        new_fragment = state.get("generated_code", "")
        accumulated = accumulated + "\n" + new_fragment
        logger.info(
            "  Subtask '%s' continuation %d: +%d chars (total %d)",
            subtask_name,
            cont_count,
            len(new_fragment),
            len(accumulated),
        )

    return state, accumulated
```

- [ ] **Step 4: Run continuation tests to verify they pass**

Run: `uv run pytest tests/test_continuation.py -v`
Expected: PASS

- [ ] **Step 5: Wire `_run_continuation_loop()` into the code phase**

In `scripts/rune_runner.py`, the code phase attempt loop (starting around line 1181) currently handles continuation inside the `for attempt in range(iters_code)` loop. When `attempt > 0` and `is_truncated` is true, it uses `code_continue` phase. This conflates continuation with retries — a continuation wastes a retry attempt.

**Three changes needed:**

**Change A:** Replace the `code_state = await run_iteration(...)` call (line ~1406) and the old concat block (lines ~1416-1429) with:

```python
                code_state = await run_iteration(
                    graph=graph,
                    project_prompt=project_prompt,
                    adapter_id=code_adapter_id,
                    session_id=session_id,
                    iteration=iteration_counter + idx * iters_code + attempt + 1,
                    phase=code_phase,
                    prompt_context=code_ctx,
                )
                await _eager_unload(code_adapter_id)

                # Continuation loop — accumulates truncated output without
                # consuming retry attempts.
                code_state, existing_code = await _run_continuation_loop(
                    initial_state=code_state,
                    run_iteration_fn=run_iteration,
                    render_trajectory_fn=lambda **kw: render_trajectory(kw.pop("phase"), **kw),
                    run_hypernetwork_fn=run_hypernetwork,
                    load_adapter_fn=lambda aid, path: _load_adapter(aid, path, loaded_code_adapter),
                    unload_adapter_fn=_eager_unload,
                    subtask=subtask,
                    attempt=attempt,
                    plan=plan,
                    dep_interfaces=dep_interfaces,
                    project_prompt=project_prompt,
                    adapter_dir=adapter_dir,
                    base_model_id=base_model_id,
                    checkpoint_path=checkpoint_path,
                    device=device,
                    adapter_scaling=adapter_scaling,
                    adapter_max_length=adapter_max_length,
                    pool=pool,
                    session_id=session_id,
                    iteration_base=iteration_counter + idx * iters_code + attempt + 1,
                    project_label=project_label,
                    graph=graph,
                )
                last_state = code_state
```

**Change B:** Remove the `is_truncated` branch from the `attempt > 0` block. The block at lines ~1194-1209 currently checks `is_truncated = last_state.get("finish_reason") == "length"` and, when true, renders a `code_continue` trajectory. Delete this entire `if is_truncated:` branch. When `attempt > 0`, the code should always go to the diagnose→repair path. The continuation loop above handles truncation transparently.

Delete:
```python
                    is_truncated = last_state.get("finish_reason") == "length"

                    if is_truncated:
                        # Continuation: prior output was cut off at max_tokens.
                        traj = render_trajectory(
                            "code_continue",
                            subtask=subtask,
                            attempt=attempt + 1,
                            max_retries=iters_code,
                            plan=plan,
                            existing_code=existing_code,
                            project=project_prompt,
                            dependency_interfaces=dep_interfaces,
                        )
                    else:
```

And outdent the retry (diagnose→repair) block that follows.

**Change C:** Remove the old continuation concat at lines ~1419-1427:

```python
                # For continuation: concatenate prior code + new output
                if attempt > 0 and is_truncated:
                    existing_code = existing_code + "\n" + new_code
                    ...
                else:
                    existing_code = new_code
```

This is replaced by `existing_code` being set by `_run_continuation_loop()` in Change A.

Also remove the `is_truncated` reference in the `code_phase` selection block (~lines 1370-1375) — the `elif is_truncated:` branch that sets `code_phase = "code_continue"` should be deleted. Retries always use `code_retry` phase.

**Known limitation:** After continuation, `code_state["tests_passed"]` reflects only the last fragment's execution (the fragment alone likely fails tests). The outer retry loop will treat this as a failed attempt and trigger diagnose→repair — which is correct behavior, since the accumulated code still needs to be tested as a whole. A future optimization could inject the accumulated code into `generated_code` before the graph's execute_node runs, but this is out of scope for P5.

- [ ] **Step 6: Run all tests**

Run: `uv run pytest tests/test_continuation.py services/rune-agent/tests/ tests/ -x -v`
Expected: All pass

- [ ] **Step 7: Lint and type check**

Run: `uv run ruff check scripts/rune_runner.py && uv run mypy scripts/rune_runner.py`
Expected: Clean

- [ ] **Step 8: Commit**

```bash
git add scripts/rune_runner.py \
       tests/test_continuation.py
git commit -m "feat(runner): runner-managed continuation loop for truncated outputs

Extracts _run_continuation_loop() for testable truncation recovery.
Truncated outputs (finish_reason='length') accumulate without consuming
retry attempts. Max 3 continuations per subtask, each with a fresh adapter
encoding the accumulated code context. Replaces the old is_truncated branch
that conflated continuation with retries."
```

---

## Integration Verification

After all 5 tasks are committed:

- [ ] **Step 1: Run full test suite**

```bash
uv run pytest -x -v
```

Expected: All 776+ tests pass

- [ ] **Step 2: Lint and type check entire codebase**

```bash
uv run ruff check libs/ services/ scripts/
uv run mypy libs/ services/
```

Expected: Clean

- [ ] **Step 3: Verify changes compile together**

```bash
python -c "
from shared.pipeline_config import load_config, default_config
from inference.provider import InferenceProvider
import inspect

# Verify thinking_budget in provider signature
sig = inspect.signature(InferenceProvider.generate)
assert 'thinking_budget' in sig.parameters

# Verify decompose config
cfg = default_config()
assert cfg.decompose.skip_threshold == 200
assert cfg.generation.thinking_budget == 512

print('All integration checks passed')
"
```

Expected: "All integration checks passed"
