# Rune v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild rune as a ~5K-line single-loop LangGraph engine on a fresh branch, TDD-first, then optimize templates empirically against MBPP/Pass@1.

**Architecture:** Single `StateGraph` with one node + conditional self-edge. Policy selects batched actions, hypernetwork generates LoRA adapters from trajectories, outlines constrains structured output via two-stage generation. Training pipeline (oracle → hypernetwork distillation → success gate) preserved from v1.

**Tech Stack:** Python 3.12, uv, LangGraph, outlines, PEFT, transformers, tree-sitter, Optuna, MLflow, typer, pytest, Qwen/Qwen3.5-9B

**Spec:** `docs/superpowers/specs/2026-05-21-rune-v2-simplification-design.md`

---

## Phase 0: Scaffolding

### Task 0: Create branch and nuke everything

**Files:**
- Delete: everything except `.git/`
- Create: fresh directory structure

- [ ] **Step 1: Create fresh branch**

```bash
git checkout -b feat/rune-v2 main
```

- [ ] **Step 2: Remove all files (preserve git history)**

```bash
git rm -rf .
git commit -m "chore: clean slate for rune v2"
```

- [ ] **Step 3: Create directory structure**

```bash
mkdir -p src/rune/engine src/rune/model src/rune/training src/rune/registry src/rune/sandbox src/rune/templates src/rune/bench
mkdir -p tests/unit tests/integration tests/gpu
mkdir -p docs/superpowers/specs docs/superpowers/plans
mkdir -p infra scripts .claude/hooks .claude/skills .claude/commands .github/workflows
```

- [ ] **Step 4: Create all `__init__.py` files**

```bash
touch src/rune/__init__.py src/rune/engine/__init__.py src/rune/model/__init__.py src/rune/training/__init__.py src/rune/registry/__init__.py src/rune/sandbox/__init__.py src/rune/bench/__init__.py
touch tests/__init__.py tests/unit/__init__.py tests/integration/__init__.py tests/gpu/__init__.py
```

- [ ] **Step 5: Commit skeleton**

```bash
git add -A
git commit -m "chore: create rune v2 directory skeleton"
```

---

### Task 1: Copy and adapt Template repo infrastructure

**Files:**
- Create: `pyproject.toml`, `Makefile`, `.gitignore`, `.claudeignore`, `PRODUCT.md`, `CLAUDE.md`, `mkdocs.yml`, `infra/docker-compose.yml`
- Source: `/Users/noahdolevelixir/Code/Template/`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "rune"
version = "0.1.0"
description = "Local-first coding agent with hypernetwork-generated LoRA adapters"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "langgraph>=0.4.0",
    "outlines>=0.2.0",
    "transformers>=4.48.0",
    "peft>=0.14.0",
    "torch>=2.5.0",
    "safetensors>=0.4.0",
    "tree-sitter>=0.24.0",
    "tree-sitter-python>=0.23.0",
    "jinja2>=3.1.0",
    "typer>=0.15.0",
    "mlflow>=2.18.0",
    "optuna>=4.1.0",
    "optuna-integration[mlflow]>=4.1.0",
    "pydantic>=2.10.0",
    "httpx>=0.28.0",
    "python-dotenv>=1.2.0",
    "bitsandbytes>=0.45.0",
]

[project.scripts]
rune = "rune.cli:app"

[dependency-groups]
dev = [
    "mypy>=1.19.0",
    "pytest>=9.0.2",
    "pytest-asyncio>=1.2.0",
    "pytest-cov>=6.0.0",
    "pytest-xdist>=3.8.0",
    "ruff>=0.14.0",
]

[tool.uv]
package = true

[tool.uv.sources]

[tool.mypy]
exclude = "^(site|tests)/"
strict = true

[[tool.mypy.overrides]]
module = [
    "langgraph", "langgraph.*",
    "langchain_core", "langchain_core.*",
    "outlines", "outlines.*",
    "peft", "peft.*",
    "torch", "torch.*",
    "transformers", "transformers.*",
    "safetensors", "safetensors.*",
    "tree_sitter", "tree_sitter.*",
    "tree_sitter_python", "tree_sitter_python.*",
    "mlflow", "mlflow.*",
    "optuna", "optuna.*",
    "optuna_integration", "optuna_integration.*",
    "bitsandbytes", "bitsandbytes.*",
    "ctx_to_lora", "ctx_to_lora.*",
]
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
norecursedirs = ["site", "docs", ".venv", "venv", ".git"]
markers = [
    "gpu: requires GPU (deselect with '-m \"not gpu\"')",
]
filterwarnings = [
    "ignore:unclosed database:ResourceWarning",
]
asyncio_mode = "auto"

[tool.coverage.run]
source = ["src/rune"]
omit = ["*/tests/*"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "raise NotImplementedError",
    "if TYPE_CHECKING:",
]

[tool.ruff]
line-length = 88
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM", "PLC", "PLE", "PLW"]
```

- [ ] **Step 2: Create `Makefile`**

```makefile
SHELL := /bin/bash
.PHONY: help lint lint-fix typecheck test check clean

lint:
	@uv run ruff check .

lint-fix:
	@uv run ruff check . --fix
	@uv run ruff format .

typecheck:
	@uv run mypy src/

test:
	@uv run pytest tests/ -q

test-unit:
	@uv run pytest tests/unit/ -q

test-integration:
	@uv run pytest tests/integration/ -q

test-gpu:
	@uv run pytest tests/gpu/ -q -m gpu

check: lint typecheck test-unit

clean:
	@rm -rf site/ .cache/ .mypy_cache/ .pytest_cache/ .ruff_cache/ htmlcov/

help:
	@echo "Rune v2"
	@echo ""
	@echo "  make check          lint + typecheck + unit tests"
	@echo "  make test           all tests"
	@echo "  make test-unit      unit tests only"
	@echo "  make test-gpu       GPU tests only"
	@echo "  make lint-fix       auto-fix lint issues"
```

- [ ] **Step 3: Copy `.gitignore` from Template**

```bash
cp /Users/noahdolevelixir/Code/Template/.gitignore .gitignore
```

- [ ] **Step 4: Copy `.claudeignore` from Template**

```bash
cp /Users/noahdolevelixir/Code/Template/.claudeignore .claudeignore
```

- [ ] **Step 5: Copy `PRODUCT.md` from Template**

```bash
cp /Users/noahdolevelixir/Code/Template/PRODUCT.md PRODUCT.md
```

- [ ] **Step 6: Create `CLAUDE.md` for rune v2**

```markdown
# Rune

Local-first coding agent that encodes coding trajectories into LoRA adapters via a perceiver hypernetwork. Single-loop LangGraph engine with four concerns: mine, train, run, benchmark.

## Read first
- **`PRODUCT.md`** — read before any non-trivial change. If missing or contains `<!-- TODO -->` stubs, stop and ask the user to fill it in.

## Stack
- `uv` Python 3.12 single package.
- Engine: LangGraph single-loop (`src/rune/engine/`).
- Model: outlines + PEFT + transformers (`src/rune/model/`).
- Training: oracle + hypernetwork distillation + DiffAwareSFTTrainer (`src/rune/training/`).
- Quality: ruff, mypy (strict), pytest.

## Hard rules
**Long-running ops** — never execute. Ask the user to run and log.
**Deploy / install** — never.
**GPU imports** — deferred inside function bodies (importable in CPU-only CI).

## Running Tests
```bash
uv sync
uv run pytest tests/unit/ -q       # fast, no GPU
uv run pytest tests/ -q             # all tests
uv run pytest tests/gpu/ -m gpu -q  # GPU only
uv run ruff check .                 # lint
uv run mypy src/                    # type check
```

## Style
- No preamble. No restating the question.
- Diff-style edits over rewrites.
- No emoji unless asked. No comments unless the *why* is non-obvious.
- Always use `uv run` to launch Python.

## Key Entry Points
- `src/rune/cli.py` — typer CLI: `rune run`, `rune train`, `rune mine`, `rune bench`
- `src/rune/engine/graph.py` — LangGraph StateGraph with single step_node
- `src/rune/engine/policy.py` — deterministic action selection + DAG layer grouping
- `src/rune/config.py` — PipelineConfig frozen dataclass
```

- [ ] **Step 7: Copy `infra/docker-compose.yml` from Template**

```bash
cp /Users/noahdolevelixir/Code/Template/infra/docker-compose.yml infra/docker-compose.yml
```

- [ ] **Step 8: Copy spec and plan to docs**

```bash
cp /Users/noahdolevelixir/Code/rune/docs/superpowers/specs/2026-05-21-rune-v2-simplification-design.md docs/superpowers/specs/
cp /Users/noahdolevelixir/Code/rune/docs/superpowers/plans/2026-05-22-rune-v2-implementation.md docs/superpowers/plans/
```

Note: These paths reference the current branch. The files should be read from the current branch before switching, or committed to the v2 branch after creation.

- [ ] **Step 9: Install dependencies**

Run: `uv sync`
Expected: Clean install, no errors.

- [ ] **Step 10: Commit infrastructure**

```bash
git add -A
git commit -m "chore: add Template-based infrastructure (pyproject, Makefile, CLAUDE.md)"
```

---

### Task 2: Copy Jinja2 templates from v1

**Files:**
- Create: `src/rune/templates/*.j2`
- Source: current `rune` branch `libs/shared/src/shared/templates/`

- [ ] **Step 1: Copy templates from v1 branch**

Before switching branches, copy templates to a temp location, or use git show:

```bash
git show main:libs/shared/src/shared/templates/decompose.j2 > src/rune/templates/decompose.j2
git show main:libs/shared/src/shared/templates/prompt_decompose.j2 > src/rune/templates/prompt_decompose.j2
git show main:libs/shared/src/shared/templates/prompt_decompose_concise.j2 > src/rune/templates/prompt_decompose_concise.j2
git show main:libs/shared/src/shared/templates/plan.j2 > src/rune/templates/plan.j2
git show main:libs/shared/src/shared/templates/prompt_plan.j2 > src/rune/templates/prompt_plan.j2
git show main:libs/shared/src/shared/templates/code.j2 > src/rune/templates/code.j2
git show main:libs/shared/src/shared/templates/prompt_code.j2 > src/rune/templates/prompt_code.j2
git show main:libs/shared/src/shared/templates/code_retry.j2 > src/rune/templates/code_retry.j2
git show main:libs/shared/src/shared/templates/prompt_code_retry.j2 > src/rune/templates/prompt_code_retry.j2
git show main:libs/shared/src/shared/templates/code_repair.j2 > src/rune/templates/code_repair.j2
git show main:libs/shared/src/shared/templates/prompt_code_repair.j2 > src/rune/templates/prompt_code_repair.j2
git show main:libs/shared/src/shared/templates/integrate.j2 > src/rune/templates/integrate.j2
git show main:libs/shared/src/shared/templates/prompt_integrate.j2 > src/rune/templates/prompt_integrate.j2
git show main:libs/shared/src/shared/templates/diagnose.j2 > src/rune/templates/diagnose.j2
git show main:libs/shared/src/shared/templates/prompt_diagnose.j2 > src/rune/templates/prompt_diagnose.j2
```

- [ ] **Step 2: Commit templates**

```bash
git add src/rune/templates/
git commit -m "feat: copy v1 Jinja2 templates as Mode 2 starting point"
```

---

## Phase 1: Foundation — Types, Config, Registry, Sandbox

### Task 3: Engine state types (`src/rune/engine/state.py`)

**Files:**
- Create: `src/rune/engine/state.py`
- Test: `tests/unit/test_state.py`

- [ ] **Step 1: Write tests**

```python
# tests/unit/test_state.py
from rune.engine.state import Action, Subtask, StepRecord, Feedback, RunState


class TestSubtask:
    def test_create_subtask(self) -> None:
        s = Subtask(name="parse_input", description="Parse user input", depends_on=[])
        assert s.name == "parse_input"
        assert s.depends_on == []

    def test_subtask_with_dependencies(self) -> None:
        s = Subtask(name="validate", description="Validate parsed input", depends_on=["parse_input"])
        assert s.depends_on == ["parse_input"]


class TestAction:
    def test_create_action(self) -> None:
        a = Action(
            name="decompose",
            trajectory_template="decompose",
            prompt_template="prompt_decompose",
            system_prompt="You are a decomposer.",
            output_schema=None,
            executes_code=False,
            target_subtask=None,
        )
        assert a.name == "decompose"
        assert a.executes_code is False

    def test_action_with_target(self) -> None:
        a = Action(
            name="code",
            trajectory_template="code",
            prompt_template="prompt_code",
            system_prompt="You are a coder.",
            output_schema=None,
            executes_code=True,
            target_subtask="parse_input",
        )
        assert a.target_subtask == "parse_input"


class TestFeedback:
    def test_passing_feedback(self) -> None:
        f = Feedback(stdout="ok", stderr="", exit_code=0)
        assert f.exit_code == 0

    def test_failing_feedback(self) -> None:
        f = Feedback(stdout="", stderr="NameError", exit_code=1)
        assert f.exit_code == 1


class TestRunState:
    def test_empty_initial_state(self) -> None:
        state: RunState = {
            "task": "build a calculator",
            "subtasks": [],
            "interfaces": {},
            "plans": {},
            "code_results": {},
            "code_passed": {},
            "retries": {},
            "integrated_code": "",
            "current_adapter": None,
            "feedback": None,
            "diagnosis": None,
            "actions": [],
            "trajectory": [],
            "step": 0,
            "budget_remaining": 20,
        }
        assert state["task"] == "build a calculator"
        assert state["budget_remaining"] == 20
        assert state["actions"] == []
```

- [ ] **Step 2: Run tests — expect FAIL**

Run: `uv run pytest tests/unit/test_state.py -v`
Expected: ImportError — `rune.engine.state` does not exist yet.

- [ ] **Step 3: Implement `state.py`**

```python
# src/rune/engine/state.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict


@dataclass(frozen=True)
class Subtask:
    name: str
    description: str
    depends_on: list[str]


@dataclass(frozen=True)
class Action:
    name: str
    trajectory_template: str
    prompt_template: str
    system_prompt: str
    output_schema: type[Any] | None
    executes_code: bool
    target_subtask: str | None


@dataclass(frozen=True)
class Feedback:
    stdout: str
    stderr: str
    exit_code: int


@dataclass(frozen=True)
class StepRecord:
    step: int
    action_name: str
    target_subtask: str | None
    adapter_id: str | None
    feedback: Feedback | None


class RunState(TypedDict):
    task: str
    subtasks: list[Subtask]
    interfaces: dict[str, str]
    plans: dict[str, str]
    code_results: dict[str, str]
    code_passed: dict[str, bool]
    retries: dict[str, int]
    integrated_code: str
    current_adapter: str | None
    feedback: Feedback | None
    diagnosis: str | None
    actions: list[Action]
    trajectory: list[StepRecord]
    step: int
    budget_remaining: int
```

- [ ] **Step 4: Run tests — expect PASS**

Run: `uv run pytest tests/unit/test_state.py -v`
Expected: All 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/rune/engine/state.py tests/unit/test_state.py
git commit -m "feat: add RunState, Action, Subtask, Feedback types"
```

---

### Task 4: Config (`src/rune/config.py`)

**Files:**
- Create: `src/rune/config.py`
- Test: `tests/unit/test_config.py`

- [ ] **Step 1: Write tests**

```python
# tests/unit/test_config.py
import json
import os
from pathlib import Path

from rune.config import PipelineConfig, load_config


class TestPipelineConfig:
    def test_defaults(self) -> None:
        cfg = PipelineConfig()
        assert cfg.model_id == "Qwen/Qwen3.5-9B"
        assert cfg.adapter_scaling == 0.075
        assert cfg.temperature == 0.3
        assert cfg.max_tokens == 2048
        assert cfg.thinking_budget == 1024
        assert cfg.max_phase_iterations == 5
        assert cfg.prompt_style == "skeleton"
        assert cfg.trajectory_style == "prose"
        assert cfg.adapter_ttl_days == 7

    def test_frozen(self) -> None:
        cfg = PipelineConfig()
        try:
            cfg.temperature = 0.5  # type: ignore[misc]
            assert False, "Should raise"
        except AttributeError:
            pass

    def test_to_dict_roundtrip(self) -> None:
        cfg = PipelineConfig(temperature=0.5, adapter_scaling=0.1)
        d = cfg.to_dict()
        assert d["temperature"] == 0.5
        assert d["adapter_scaling"] == 0.1

    def test_override(self) -> None:
        cfg = PipelineConfig()
        new = cfg.override(temperature=0.8, max_tokens=4096)
        assert new.temperature == 0.8
        assert new.max_tokens == 4096
        assert cfg.temperature == 0.3  # original unchanged

    def test_save_and_load(self, tmp_path: Path) -> None:
        cfg = PipelineConfig(temperature=0.42)
        path = cfg.save(tmp_path / "config.json")
        loaded = load_config(path)
        assert loaded.temperature == 0.42

    def test_env_var_override(self, monkeypatch: object) -> None:
        os.environ["RUNE_TEMPERATURE"] = "0.99"
        try:
            cfg = PipelineConfig.from_env()
            assert cfg.temperature == 0.99
        finally:
            del os.environ["RUNE_TEMPERATURE"]

    def test_phase_max_tokens(self) -> None:
        cfg = PipelineConfig(phase_max_tokens={"plan": 512, "code": 2048})
        assert cfg.phase_max_tokens["plan"] == 512
```

- [ ] **Step 2: Run tests — expect FAIL**

Run: `uv run pytest tests/unit/test_config.py -v`

- [ ] **Step 3: Implement `config.py`**

```python
# src/rune/config.py
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PipelineConfig:
    model_id: str = "Qwen/Qwen3.5-9B"
    adapter_scaling: float = 0.075
    temperature: float = 0.3
    max_tokens: int = 2048
    repetition_penalty: float = 1.1
    top_p: float = 0.9
    thinking_budget: int = 1024
    phase_max_tokens: dict[str, int] = field(default_factory=dict)
    max_phase_iterations: int = 5
    prompt_style: str = "skeleton"
    trajectory_style: str = "prose"
    adapter_ttl_days: int = 7
    checkpoint_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    def override(self, **kwargs: Any) -> PipelineConfig:
        d = self.to_dict()
        d.update(kwargs)
        return PipelineConfig(**d)

    @classmethod
    def from_env(cls) -> PipelineConfig:
        overrides: dict[str, Any] = {}
        env_map: dict[str, tuple[str, type]] = {
            "RUNE_TEMPERATURE": ("temperature", float),
            "RUNE_MAX_TOKENS": ("max_tokens", int),
            "RUNE_REPETITION_PENALTY": ("repetition_penalty", float),
            "RUNE_TOP_P": ("top_p", float),
            "RUNE_THINKING_BUDGET": ("thinking_budget", int),
            "RUNE_MAX_PHASE_ITERATIONS": ("max_phase_iterations", int),
            "RUNE_ADAPTER_SCALING": ("adapter_scaling", float),
        }
        for env_key, (field_name, converter) in env_map.items():
            val = os.environ.get(env_key)
            if val is not None:
                overrides[field_name] = converter(val)
        if not overrides:
            return cls()
        return cls(**overrides)


def load_config(path: Path) -> PipelineConfig:
    if path.exists():
        d = json.loads(path.read_text())
        return PipelineConfig(**d)
    return PipelineConfig()
```

- [ ] **Step 4: Run tests — expect PASS**

Run: `uv run pytest tests/unit/test_config.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/rune/config.py tests/unit/test_config.py
git commit -m "feat: add PipelineConfig with env var overrides"
```

---

### Task 5: Sandbox executor (`src/rune/sandbox/executor.py`)

**Files:**
- Create: `src/rune/sandbox/executor.py`
- Test: `tests/unit/test_sandbox.py` (unit), `tests/integration/test_sandbox.py` (integration)

- [ ] **Step 1: Write unit test**

```python
# tests/unit/test_sandbox.py
from rune.sandbox.executor import ExecutionResult, extract_code


class TestExtractCode:
    def test_extract_fenced_python(self) -> None:
        raw = "Here is code:\n```python\nprint('hello')\n```\nDone."
        assert extract_code(raw) == "print('hello')"

    def test_extract_unfenced(self) -> None:
        raw = "print('hello')"
        assert extract_code(raw) == "print('hello')"

    def test_extract_multiple_blocks_takes_longest(self) -> None:
        raw = "```python\nx = 1\n```\n\n```python\nx = 1\ny = 2\nz = 3\n```"
        assert "z = 3" in extract_code(raw)


class TestExecutionResult:
    def test_passed(self) -> None:
        r = ExecutionResult(stdout="ok", stderr="", exit_code=0)
        assert r.exit_code == 0
```

- [ ] **Step 2: Write integration test**

```python
# tests/integration/test_sandbox.py
from rune.sandbox.executor import run_in_sandbox


class TestRunInSandbox:
    def test_passing_code(self) -> None:
        result = run_in_sandbox("print('hello')")
        assert result.exit_code == 0
        assert "hello" in result.stdout

    def test_failing_code(self) -> None:
        result = run_in_sandbox("raise ValueError('boom')")
        assert result.exit_code == 1
        assert "ValueError" in result.stderr

    def test_timeout(self) -> None:
        result = run_in_sandbox("import time; time.sleep(10)", timeout=1)
        assert result.exit_code != 0

    def test_syntax_error(self) -> None:
        result = run_in_sandbox("def (broken")
        assert result.exit_code != 0
```

- [ ] **Step 3: Run tests — expect FAIL**

Run: `uv run pytest tests/unit/test_sandbox.py tests/integration/test_sandbox.py -v`

- [ ] **Step 4: Implement `executor.py`**

```python
# src/rune/sandbox/executor.py
from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExecutionResult:
    stdout: str
    stderr: str
    exit_code: int


def extract_code(raw: str) -> str:
    blocks = re.findall(r"```(?:python)?\n(.*?)```", raw, re.DOTALL)
    if blocks:
        return max(blocks, key=len).strip()
    return raw.strip()


def run_in_sandbox(code: str, *, timeout: int = 30) -> ExecutionResult:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        f.flush()
        path = Path(f.name)
    try:
        proc = subprocess.run(
            ["python", str(path)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return ExecutionResult(
            stdout=proc.stdout,
            stderr=proc.stderr,
            exit_code=proc.returncode,
        )
    except subprocess.TimeoutExpired:
        return ExecutionResult(stdout="", stderr="Timeout", exit_code=-1)
    finally:
        path.unlink(missing_ok=True)
```

- [ ] **Step 5: Run tests — expect PASS**

Run: `uv run pytest tests/unit/test_sandbox.py tests/integration/test_sandbox.py -v`

- [ ] **Step 6: Commit**

```bash
git add src/rune/sandbox/executor.py tests/unit/test_sandbox.py tests/integration/test_sandbox.py
git commit -m "feat: add sandbox executor with code extraction"
```

---

### Task 6: Adapter registry (`src/rune/registry/store.py`)

**Files:**
- Create: `src/rune/registry/store.py`
- Test: `tests/unit/test_registry.py`

- [ ] **Step 1: Write tests**

```python
# tests/unit/test_registry.py
import time
from pathlib import Path

from rune.registry.store import AdapterRecord, AdapterRegistry


class TestAdapterRegistry:
    def test_register_and_get(self) -> None:
        reg = AdapterRegistry.create(":memory:")
        reg.register(
            adapter_id="a1",
            disk_path="/tmp/a1.safetensors",
            parent_id=None,
            action="decompose",
            session_id="s1",
            generation=0,
        )
        record = reg.get("a1")
        assert record is not None
        assert record.adapter_id == "a1"
        assert record.action == "decompose"
        assert record.parent_id is None

    def test_get_missing_returns_none(self) -> None:
        reg = AdapterRegistry.create(":memory:")
        assert reg.get("nonexistent") is None

    def test_lineage(self) -> None:
        reg = AdapterRegistry.create(":memory:")
        reg.register("a1", "/tmp/a1", None, "decompose", "s1", 0)
        reg.register("a2", "/tmp/a2", "a1", "plan", "s1", 1)
        reg.register("a3", "/tmp/a3", "a2", "code", "s1", 2)
        lineage = reg.lineage("a3")
        assert [r.adapter_id for r in lineage] == ["a3", "a2", "a1"]

    def test_list_by_session(self) -> None:
        reg = AdapterRegistry.create(":memory:")
        reg.register("a1", "/tmp/a1", None, "decompose", "s1", 0)
        reg.register("a2", "/tmp/a2", None, "decompose", "s2", 0)
        records = reg.list_by_session("s1")
        assert len(records) == 1
        assert records[0].adapter_id == "a1"

    def test_prune_by_age(self, tmp_path: Path) -> None:
        reg = AdapterRegistry.create(":memory:")
        # Insert with manually backdated created_at
        reg.register("old", str(tmp_path / "old.st"), None, "code", "s1", 0)
        reg._backdate("old", days=10)
        reg.register("new", str(tmp_path / "new.st"), None, "code", "s1", 1)
        (tmp_path / "old.st").write_bytes(b"\x00")
        pruned = reg.prune(max_age_days=7)
        assert pruned == 1
        assert reg.get("old") is None
        assert reg.get("new") is not None
        assert not (tmp_path / "old.st").exists()
```

- [ ] **Step 2: Run tests — expect FAIL**

Run: `uv run pytest tests/unit/test_registry.py -v`

- [ ] **Step 3: Implement `store.py`**

```python
# src/rune/registry/store.py
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AdapterRecord:
    adapter_id: str
    disk_path: str
    parent_id: str | None
    action: str
    session_id: str
    generation: int
    created_at: float


class AdapterRegistry:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS adapters (
                adapter_id TEXT PRIMARY KEY,
                disk_path TEXT NOT NULL,
                parent_id TEXT,
                action TEXT NOT NULL,
                session_id TEXT NOT NULL,
                generation INTEGER NOT NULL,
                created_at REAL NOT NULL
            )"""
        )
        self._conn.commit()

    @classmethod
    def create(cls, db_path: str | Path) -> AdapterRegistry:
        conn = sqlite3.connect(str(db_path))
        return cls(conn)

    def register(
        self,
        adapter_id: str,
        disk_path: str,
        parent_id: str | None,
        action: str,
        session_id: str,
        generation: int,
    ) -> None:
        self._conn.execute(
            "INSERT INTO adapters VALUES (?, ?, ?, ?, ?, ?, ?)",
            (adapter_id, disk_path, parent_id, action, session_id, generation, time.time()),
        )
        self._conn.commit()

    def get(self, adapter_id: str) -> AdapterRecord | None:
        row = self._conn.execute(
            "SELECT * FROM adapters WHERE adapter_id = ?", (adapter_id,)
        ).fetchone()
        return AdapterRecord(*row) if row else None

    def lineage(self, adapter_id: str) -> list[AdapterRecord]:
        chain: list[AdapterRecord] = []
        current = adapter_id
        while current:
            record = self.get(current)
            if record is None:
                break
            chain.append(record)
            current = record.parent_id
        return chain

    def list_by_session(self, session_id: str) -> list[AdapterRecord]:
        rows = self._conn.execute(
            "SELECT * FROM adapters WHERE session_id = ? ORDER BY generation",
            (session_id,),
        ).fetchall()
        return [AdapterRecord(*r) for r in rows]

    def prune(self, max_age_days: int = 7) -> int:
        cutoff = time.time() - (max_age_days * 86400)
        rows = self._conn.execute(
            "SELECT disk_path FROM adapters WHERE created_at < ?", (cutoff,)
        ).fetchall()
        for (disk_path,) in rows:
            Path(disk_path).unlink(missing_ok=True)
        cursor = self._conn.execute(
            "DELETE FROM adapters WHERE created_at < ?", (cutoff,)
        )
        self._conn.commit()
        return cursor.rowcount

    def _backdate(self, adapter_id: str, days: int) -> None:
        self._conn.execute(
            "UPDATE adapters SET created_at = ? WHERE adapter_id = ?",
            (time.time() - days * 86400, adapter_id),
        )
        self._conn.commit()
```

- [ ] **Step 4: Run tests — expect PASS**

Run: `uv run pytest tests/unit/test_registry.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/rune/registry/store.py tests/unit/test_registry.py
git commit -m "feat: add SQLite adapter registry with lineage and pruning"
```

---

## Phase 2: Engine — Policy, Parse, Interfaces, Graph

### Task 7: Template rendering + parse_output (`src/rune/engine/parse.py`)

**Files:**
- Create: `src/rune/engine/parse.py`
- Test: `tests/unit/test_parse.py`

- [ ] **Step 1: Write tests**

```python
# tests/unit/test_parse.py
from rune.engine.parse import parse_output, render_template, DecomposeResult, DiagnoseResult
from rune.engine.state import Action, Feedback


class TestRenderTemplate:
    def test_renders_jinja2(self) -> None:
        text = render_template("decompose", task="build a calculator", subtasks=[])
        assert "calculator" in text


class TestDecomposeResult:
    def test_parse_valid_json(self) -> None:
        raw = '{"subtasks": [{"name": "parse", "description": "Parse input", "depends_on": []}]}'
        result = DecomposeResult.model_validate_json(raw)
        assert len(result.subtasks) == 1
        assert result.subtasks[0].name == "parse"


class TestDiagnoseResult:
    def test_parse_valid_json(self) -> None:
        raw = '{"fix_guidance": "Add missing import for os module"}'
        result = DiagnoseResult.model_validate_json(raw)
        assert "import" in result.fix_guidance


class TestParseOutput:
    def test_decompose_action(self) -> None:
        action = Action("decompose", "decompose", "prompt_decompose", "", DecomposeResult, False, None)
        raw = '{"subtasks": [{"name": "a", "description": "do a", "depends_on": []}]}'
        state_stub: dict = {"plans": {}, "code_results": {}, "code_passed": {}, "retries": {}, "subtasks": []}
        updates = parse_output(action, raw, None, state_stub)
        assert len(updates["subtasks"]) == 1

    def test_code_action_passing(self) -> None:
        action = Action("code", "code", "prompt_code", "", None, True, "task_a")
        fb = Feedback(stdout="ok", stderr="", exit_code=0)
        state_stub: dict = {"code_results": {}, "code_passed": {}, "retries": {}}
        updates = parse_output(action, "```python\nprint(1)\n```", fb, state_stub)
        assert updates["code_passed"]["task_a"] is True
        assert "print(1)" in updates["code_results"]["task_a"]

    def test_code_retry_increments_retries(self) -> None:
        action = Action("code_retry", "code_retry", "prompt_code_retry", "", None, True, "task_a")
        fb = Feedback(stdout="", stderr="err", exit_code=1)
        state_stub: dict = {"code_results": {}, "code_passed": {}, "retries": {"task_a": 1}}
        updates = parse_output(action, "```python\npass\n```", fb, state_stub)
        assert updates["retries"]["task_a"] == 2

    def test_diagnose_action(self) -> None:
        action = Action("diagnose", "diagnose", "prompt_diagnose", "", DiagnoseResult, False, None)
        raw = '{"fix_guidance": "fix the bug"}'
        updates = parse_output(action, raw, None, {})
        assert updates["diagnosis"] == "fix the bug"
```

- [ ] **Step 2: Run tests — expect FAIL**

Run: `uv run pytest tests/unit/test_parse.py -v`

- [ ] **Step 3: Implement `parse.py`**

```python
# src/rune/engine/parse.py
from __future__ import annotations

from typing import Any

from jinja2 import Environment, PackageLoader
from pydantic import BaseModel

from rune.engine.state import Action, Feedback, Subtask
from rune.sandbox.executor import extract_code

_env = Environment(loader=PackageLoader("rune", "templates"))


def render_template(template_name: str, **kwargs: Any) -> str:
    return _env.get_template(f"{template_name}.j2").render(**kwargs)


class SubtaskSchema(BaseModel):
    name: str
    description: str
    depends_on: list[str] = []


class DecomposeResult(BaseModel):
    subtasks: list[SubtaskSchema]


class DiagnoseResult(BaseModel):
    fix_guidance: str


def parse_output(
    action: Action,
    raw: str,
    feedback: Feedback | None,
    state: dict[str, Any],
) -> dict[str, Any]:
    match action.name:
        case "decompose":
            result = DecomposeResult.model_validate_json(raw)
            return {
                "subtasks": [
                    Subtask(name=s.name, description=s.description, depends_on=s.depends_on)
                    for s in result.subtasks
                ]
            }
        case "plan":
            target = action.target_subtask
            return {"plans": {**state.get("plans", {}), target: raw}}
        case "code" | "code_retry":
            target = action.target_subtask
            passed = feedback is not None and feedback.exit_code == 0
            retries = dict(state.get("retries", {}))
            if action.name == "code_retry":
                retries[target] = retries.get(target, 0) + 1
            return {
                "code_results": {**state.get("code_results", {}), target: extract_code(raw)},
                "code_passed": {**state.get("code_passed", {}), target: passed},
                "retries": retries,
                "feedback": feedback,
            }
        case "integrate":
            passed = feedback is not None and feedback.exit_code == 0
            return {
                "integrated_code": extract_code(raw) if passed else "",
                "feedback": feedback,
                "diagnosis": None,
            }
        case "diagnose":
            result = DiagnoseResult.model_validate_json(raw)
            return {"diagnosis": result.fix_guidance}
    return {}
```

- [ ] **Step 4: Run tests — expect PASS**

Run: `uv run pytest tests/unit/test_parse.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/rune/engine/parse.py tests/unit/test_parse.py
git commit -m "feat: add parse_output dispatcher and template rendering"
```

---

### Task 8: Policy — action selection + DAG layering (`src/rune/engine/policy.py`)

**Files:**
- Create: `src/rune/engine/policy.py`
- Test: `tests/unit/test_policy.py`

- [ ] **Step 1: Write tests**

```python
# tests/unit/test_policy.py
from rune.engine.policy import select_action, build_execution_layers, ACTIONS
from rune.engine.state import RunState, Subtask


def _make_state(**overrides: object) -> dict:
    base: dict = {
        "task": "test",
        "subtasks": [],
        "interfaces": {},
        "plans": {},
        "code_results": {},
        "code_passed": {},
        "retries": {},
        "integrated_code": "",
        "current_adapter": None,
        "feedback": None,
        "diagnosis": None,
        "actions": [],
        "trajectory": [],
        "step": 0,
        "budget_remaining": 20,
    }
    base.update(overrides)
    return base


class TestSelectAction:
    def test_empty_subtasks_returns_decompose(self) -> None:
        actions = select_action(_make_state())
        assert len(actions) == 1
        assert actions[0].name == "decompose"

    def test_unplanned_subtasks_returns_plan(self) -> None:
        subtasks = [Subtask("a", "do a", []), Subtask("b", "do b", [])]
        actions = select_action(_make_state(subtasks=subtasks))
        assert all(a.name == "plan" for a in actions)
        assert len(actions) == 2  # both are independent

    def test_uncoded_subtask_returns_code(self) -> None:
        subtasks = [Subtask("a", "do a", [])]
        actions = select_action(_make_state(subtasks=subtasks, plans={"a": "plan a"}))
        assert len(actions) == 1
        assert actions[0].name == "code"

    def test_failed_code_returns_code_retry(self) -> None:
        subtasks = [Subtask("a", "do a", [])]
        state = _make_state(
            subtasks=subtasks,
            plans={"a": "plan"},
            code_results={"a": "bad code"},
            code_passed={"a": False},
            retries={"a": 1},
        )
        actions = select_action(state)
        assert actions[0].name == "code_retry"

    def test_max_retries_returns_empty(self) -> None:
        subtasks = [Subtask("a", "do a", [])]
        state = _make_state(
            subtasks=subtasks,
            plans={"a": "plan"},
            code_results={"a": "bad"},
            code_passed={"a": False},
            retries={"a": 3},
        )
        actions = select_action(state)
        assert actions == []

    def test_all_passing_returns_integrate(self) -> None:
        subtasks = [Subtask("a", "do a", [])]
        state = _make_state(
            subtasks=subtasks,
            plans={"a": "plan"},
            code_results={"a": "good"},
            code_passed={"a": True},
        )
        actions = select_action(state)
        assert actions[0].name == "integrate"

    def test_done_returns_empty(self) -> None:
        subtasks = [Subtask("a", "do a", [])]
        state = _make_state(
            subtasks=subtasks,
            plans={"a": "plan"},
            code_results={"a": "good"},
            code_passed={"a": True},
            integrated_code="final code",
        )
        actions = select_action(state)
        assert actions == []


class TestBuildExecutionLayers:
    def test_no_deps_single_layer(self) -> None:
        subtasks = [Subtask("a", "", []), Subtask("b", "", [])]
        layers = build_execution_layers(subtasks)
        assert len(layers) == 1
        assert set(layers[0]) == {"a", "b"}

    def test_chain_dependency(self) -> None:
        subtasks = [Subtask("a", "", []), Subtask("b", "", ["a"]), Subtask("c", "", ["b"])]
        layers = build_execution_layers(subtasks)
        assert len(layers) == 3
        assert layers[0] == ["a"]
        assert layers[1] == ["b"]
        assert layers[2] == ["c"]

    def test_diamond_dependency(self) -> None:
        subtasks = [
            Subtask("a", "", []),
            Subtask("b", "", ["a"]),
            Subtask("c", "", ["a"]),
            Subtask("d", "", ["b", "c"]),
        ]
        layers = build_execution_layers(subtasks)
        assert layers[0] == ["a"]
        assert set(layers[1]) == {"b", "c"}
        assert layers[2] == ["d"]
```

- [ ] **Step 2: Run tests — expect FAIL**

Run: `uv run pytest tests/unit/test_policy.py -v`

- [ ] **Step 3: Implement `policy.py`**

```python
# src/rune/engine/policy.py
from __future__ import annotations

from graphlib import TopologicalSorter

from rune.engine.parse import DecomposeResult, DiagnoseResult
from rune.engine.state import Action, Subtask

MAX_RETRIES = 3

ACTIONS: dict[str, Action] = {
    "decompose": Action("decompose", "decompose", "prompt_decompose", "You are a project decomposer.", DecomposeResult, False, None),
    "plan": Action("plan", "plan", "prompt_plan", "You are a project planner.", None, False, None),
    "code": Action("code", "code", "prompt_code", "You are a code generator.", None, True, None),
    "code_retry": Action("code_retry", "code_retry", "prompt_code_retry", "You are a code generator.", None, True, None),
    "integrate": Action("integrate", "integrate", "prompt_integrate", "You are a code integrator.", None, True, None),
    "diagnose": Action("diagnose", "diagnose", "prompt_diagnose", "You are a code diagnostician.", DiagnoseResult, False, None),
}


def build_execution_layers(subtasks: list[Subtask]) -> list[list[str]]:
    graph: dict[str, set[str]] = {}
    for s in subtasks:
        graph[s.name] = set(s.depends_on)
    sorter = TopologicalSorter(graph)
    sorter.prepare()
    layers: list[list[str]] = []
    while sorter.is_active():
        ready = sorted(sorter.get_ready())
        layers.append(ready)
        for node in ready:
            sorter.done(node)
    return layers


def _with_target(action_name: str, target: str) -> Action:
    base = ACTIONS[action_name]
    return Action(
        name=base.name,
        trajectory_template=base.trajectory_template,
        prompt_template=base.prompt_template,
        system_prompt=base.system_prompt,
        output_schema=base.output_schema,
        executes_code=base.executes_code,
        target_subtask=target,
    )


def select_action(state: dict) -> list[Action]:
    subtasks: list[Subtask] = state["subtasks"]
    if not subtasks:
        return [ACTIONS["decompose"]]

    # Plan unplanned subtasks
    unplanned = [s for s in subtasks if s.name not in state["plans"]]
    if unplanned:
        layers = build_execution_layers(unplanned)
        return [_with_target("plan", name) for name in layers[0]]

    # Code uncoded or failing subtasks
    failing = [
        s for s in subtasks
        if not state["code_passed"].get(s.name)
    ]
    if failing:
        layers = build_execution_layers(failing)
        ready_names = set(layers[0])
        # Only subtasks whose deps all pass
        ready = [
            s for s in failing
            if s.name in ready_names
            and all(state["code_passed"].get(d, False) for d in s.depends_on)
        ]
        actions: list[Action] = []
        for s in ready:
            if state["retries"].get(s.name, 0) >= MAX_RETRIES:
                return []  # stuck
            action_name = "code_retry" if s.name in state["code_results"] else "code"
            actions.append(_with_target(action_name, s.name))
        return actions if actions else []

    # All passing — integrate or done
    if state["integrated_code"]:
        return []

    if state.get("diagnosis"):
        return [ACTIONS["integrate"]]
    if state.get("feedback") and state["feedback"].exit_code != 0:
        return [ACTIONS["diagnose"]]
    return [ACTIONS["integrate"]]
```

- [ ] **Step 4: Run tests — expect PASS**

Run: `uv run pytest tests/unit/test_policy.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/rune/engine/policy.py tests/unit/test_policy.py
git commit -m "feat: add policy with DAG-layered action selection"
```

---

### Task 9: Interface extraction (`src/rune/engine/interfaces.py`)

**Files:**
- Create: `src/rune/engine/interfaces.py`
- Test: `tests/unit/test_interfaces.py`

- [ ] **Step 1: Write tests**

```python
# tests/unit/test_interfaces.py
from rune.engine.interfaces import extract_interfaces


class TestExtractInterfaces:
    def test_extract_function_signature(self) -> None:
        code = "def add(a: int, b: int) -> int:\n    return a + b\n"
        interfaces = extract_interfaces(code)
        assert "def add(a: int, b: int) -> int" in interfaces

    def test_extract_class_definition(self) -> None:
        code = "class Calculator:\n    def __init__(self) -> None:\n        self.value = 0\n    def add(self, x: int) -> None:\n        self.value += x\n"
        interfaces = extract_interfaces(code)
        assert "class Calculator" in interfaces
        assert "def add" in interfaces

    def test_empty_code(self) -> None:
        assert extract_interfaces("") == ""

    def test_no_definitions(self) -> None:
        code = "x = 1\ny = 2\nprint(x + y)\n"
        interfaces = extract_interfaces(code)
        # Should return empty or minimal — no function/class definitions
        assert "def " not in interfaces
```

- [ ] **Step 2: Run tests — expect FAIL**

Run: `uv run pytest tests/unit/test_interfaces.py -v`

- [ ] **Step 3: Implement `interfaces.py`**

Uses tree-sitter to extract function signatures and class definitions from Python code.

```python
# src/rune/engine/interfaces.py
from __future__ import annotations

import tree_sitter_python as tspython
from tree_sitter import Language, Parser

PY_LANGUAGE = Language(tspython.language())
_parser = Parser(PY_LANGUAGE)


def extract_interfaces(code: str) -> str:
    if not code.strip():
        return ""

    tree = _parser.parse(code.encode())
    lines: list[str] = []

    for node in tree.root_node.children:
        if node.type == "function_definition":
            # Extract just the signature line
            first_line = code[node.start_byte : node.end_byte].split("\n")[0]
            lines.append(first_line)
        elif node.type == "class_definition":
            # Extract class line + method signatures
            class_code = code[node.start_byte : node.end_byte]
            class_lines = class_code.split("\n")
            lines.append(class_lines[0])
            for child in node.children:
                if child.type == "block":
                    for stmt in child.children:
                        if stmt.type == "function_definition":
                            method_line = code[stmt.start_byte : stmt.end_byte].split("\n")[0]
                            lines.append(method_line)

    return "\n".join(lines)
```

- [ ] **Step 4: Run tests — expect PASS**

Run: `uv run pytest tests/unit/test_interfaces.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/rune/engine/interfaces.py tests/unit/test_interfaces.py
git commit -m "feat: add tree-sitter interface extraction"
```

---

### Task 10: LangGraph engine (`src/rune/engine/graph.py`)

**Files:**
- Create: `src/rune/engine/graph.py`
- Test: `tests/integration/test_graph.py`

- [ ] **Step 1: Write integration test with mock model**

```python
# tests/integration/test_graph.py
import json
from unittest.mock import AsyncMock, MagicMock

from rune.engine.graph import create_engine, should_continue
from rune.engine.state import RunState


def _initial_state(task: str = "add two numbers", budget: int = 10) -> RunState:
    return {
        "task": task,
        "subtasks": [],
        "interfaces": {},
        "plans": {},
        "code_results": {},
        "code_passed": {},
        "retries": {},
        "integrated_code": "",
        "current_adapter": None,
        "feedback": None,
        "diagnosis": None,
        "actions": [MagicMock()],  # non-empty so first step runs
        "trajectory": [],
        "step": 0,
        "budget_remaining": budget,
    }


class TestShouldContinue:
    def test_empty_actions_returns_done(self) -> None:
        state = _initial_state()
        state["actions"] = []
        assert should_continue(state) == "done"

    def test_budget_zero_returns_done(self) -> None:
        state = _initial_state(budget=0)
        assert should_continue(state) == "done"

    def test_has_actions_and_budget_returns_continue(self) -> None:
        state = _initial_state()
        assert should_continue(state) == "continue"


class TestCreateEngine:
    def test_engine_compiles(self) -> None:
        engine = create_engine()
        assert engine is not None
```

- [ ] **Step 2: Run tests — expect FAIL**

Run: `uv run pytest tests/integration/test_graph.py -v`

- [ ] **Step 3: Implement `graph.py`**

```python
# src/rune/engine/graph.py
from __future__ import annotations

import asyncio
from typing import Any

from langgraph.graph import END, StateGraph
from langgraph.graph.graph import CompiledGraph

from rune.engine.parse import parse_output, render_template
from rune.engine.policy import select_action
from rune.engine.state import Action, Feedback, RunState, StepRecord
from rune.sandbox.executor import run_in_sandbox


def state_to_ctx(state: RunState) -> dict[str, Any]:
    return {
        "task": state["task"],
        "subtasks": state["subtasks"],
        "plans": state["plans"],
        "code": state.get("code_results", {}),
        "integrated_code": state["integrated_code"],
        "feedback": state["feedback"],
        "diagnosis": state["diagnosis"],
        "interfaces": state["interfaces"],
    }


async def step_node(state: RunState, config: dict) -> dict:
    configurable = config.get("configurable", {})
    model = configurable["model"]
    registry = configurable.get("registry")
    run_config = configurable.get("run_config", {})

    actions = select_action(state)
    if not actions:
        return {"actions": [], "budget_remaining": state["budget_remaining"]}

    results: list[tuple[Action, str, str]] = []
    for action in actions:
        ctx = state_to_ctx(state)
        trajectory_text = render_template(action.trajectory_template, **ctx)
        prompt_text = render_template(action.prompt_template, **ctx)

        adapter = model.generate_adapter(trajectory_text)
        model.hotswap_adapter(adapter.state_dict)
        result = await model.generate(
            prompt=prompt_text,
            system_prompt=action.system_prompt,
            output_schema=action.output_schema,
            max_tokens=run_config.get("max_tokens", 2048),
        )
        target_name = action.target_subtask or ""
        results.append((action, target_name, result.text))

    code_actions = [(a, name, text) for a, name, text in results if a.executes_code]
    sandbox_results = await asyncio.gather(*[
        asyncio.to_thread(run_in_sandbox, text) for _, _, text in code_actions
    ])
    feedback_map = {
        name: Feedback(stdout=fb.stdout, stderr=fb.stderr, exit_code=fb.exit_code)
        for (_, name, _), fb in zip(code_actions, sandbox_results)
    }

    updates: dict[str, Any] = {}
    for action, target_name, raw in results:
        fb = feedback_map.get(target_name)
        partial = parse_output(action, raw, fb, state)
        for k, v in partial.items():
            if isinstance(v, dict) and isinstance(updates.get(k), dict):
                updates[k] = {**updates[k], **v}
            else:
                updates[k] = v

    records = [
        StepRecord(
            step=state["step"],
            action_name=a.name,
            target_subtask=name,
            adapter_id=state["current_adapter"],
            feedback=feedback_map.get(name),
        )
        for a, name, _ in results
    ]
    updates["actions"] = actions
    updates["trajectory"] = state["trajectory"] + records
    updates["step"] = state["step"] + 1
    updates["budget_remaining"] = state["budget_remaining"] - 1
    return updates


def should_continue(state: RunState) -> str:
    if not state["actions"] or state["budget_remaining"] <= 0:
        return "done"
    return "continue"


def create_engine() -> CompiledGraph:
    graph = StateGraph(RunState)
    graph.add_node("step", step_node)
    graph.set_entry_point("step")
    graph.add_conditional_edges("step", should_continue, {
        "continue": "step",
        "done": END,
    })
    return graph.compile()
```

- [ ] **Step 4: Run tests — expect PASS**

Run: `uv run pytest tests/integration/test_graph.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/rune/engine/graph.py tests/integration/test_graph.py
git commit -m "feat: add LangGraph engine with step_node and should_continue"
```

---

## Phase 3: Model Layer — Stubs for GPU-Dependent Code

### Task 11: Model layer stubs (`src/rune/model/`)

These modules depend on GPU libraries (torch, transformers, PEFT, outlines). Implement as stubs with clear interfaces. Full implementation requires GPU and happens during subagent-driven execution.

**Files:**
- Create: `src/rune/model/inference.py`, `src/rune/model/hypernetwork.py`, `src/rune/model/adapter.py`

- [ ] **Step 1: Create `adapter.py`**

```python
# src/rune/model/adapter.py
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AdapterResult:
    adapter_id: str
    state_dict: dict[str, Any]


async def persist_adapter(
    state_dict: dict[str, Any],
    adapter_id: str,
    output_dir: Path,
) -> Path:
    path = output_dir / f"{adapter_id}.safetensors"

    def _write() -> None:
        from safetensors.torch import save_file  # noqa: PLC0415
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file(state_dict, str(path))

    await asyncio.to_thread(_write)
    return path


def hotswap_adapter(model: Any, state_dict: dict[str, Any]) -> None:
    from peft import set_peft_model_state_dict  # noqa: PLC0415
    set_peft_model_state_dict(model, state_dict)
```

- [ ] **Step 2: Create `hypernetwork.py`**

```python
# src/rune/model/hypernetwork.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class HypernetworkConfig:
    checkpoint_path: str
    model_config_name: str = "qwen3.5-9b"


def load_hypernetwork(config: HypernetworkConfig) -> Any:
    import torch  # noqa: PLC0415

    logger.info("Loading hypernetwork from %s", config.checkpoint_path)
    sd = torch.load(config.checkpoint_path, map_location="cpu", weights_only=False)

    from ctx_to_lora.modeling.hypernet import HyperLoRA  # noqa: PLC0415

    hc = sd.get("hypernet_config") or sd.get("config")
    hypernet = HyperLoRA(hc)
    weights = sd.get("hypernet_state_dict") or sd.get("model_state_dict", sd)
    hypernet.load_state_dict(weights, strict=False)
    return hypernet.eval()


def generate_adapter_weights(
    hypernet: Any,
    trajectory_text: str,
    base_model: Any,
    tokenizer: Any,
    layer_indices: list[int],
    max_length: int = 2048,
) -> dict[str, Any]:
    import torch  # noqa: PLC0415
    from model_training.d2l_activations import extract_activations_with_model  # noqa: PLC0415

    features, attn_mask = extract_activations_with_model(
        text=trajectory_text,
        model=base_model,
        tokenizer=tokenizer,
        layer_indices=layer_indices,
        max_length=max_length,
    )
    with torch.no_grad():
        lora_dict, _ = hypernet.generate_weights(features, attn_mask, None)
    return lora_dict
```

- [ ] **Step 3: Create `inference.py`**

```python
# src/rune/model/inference.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    text: str
    thinking: str
    tokens_used: int


async def generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    system_prompt: str = "",
    output_schema: type[Any] | None = None,
    max_tokens: int = 2048,
    temperature: float = 0.3,
    thinking_budget: int = 1024,
) -> GenerationResult:
    import torch  # noqa: PLC0415

    if output_schema is not None:
        return await _generate_structured(
            model, tokenizer, prompt,
            system_prompt=system_prompt,
            schema=output_schema,
            max_tokens=max_tokens,
            thinking_budget=thinking_budget,
        )
    return await _generate_freeform(
        model, tokenizer, prompt,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )


async def _generate_freeform(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    system_prompt: str,
    max_tokens: int,
    temperature: float,
) -> GenerationResult:
    import asyncio  # noqa: PLC0415

    def _run() -> GenerationResult:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
        import torch  # noqa: PLC0415
        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=max_tokens, temperature=temperature, do_sample=True)
        text = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
        return GenerationResult(text=text, thinking="", tokens_used=output.shape[1])

    return await asyncio.to_thread(_run)


async def _generate_structured(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    system_prompt: str,
    schema: type[Any],
    max_tokens: int,
    thinking_budget: int,
) -> GenerationResult:
    import asyncio  # noqa: PLC0415

    def _run() -> GenerationResult:
        # Two-stage: free-form thinking → outlines constrained
        # Stage 1: generate thinking with stop at </think>
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)

        think_token_id = tokenizer.encode("</think>", add_special_tokens=False)
        import torch  # noqa: PLC0415
        with torch.no_grad():
            thinking_output = model.generate(
                input_ids,
                max_new_tokens=thinking_budget,
                eos_token_id=think_token_id,
                do_sample=False,
            )
        thinking_text = tokenizer.decode(thinking_output[0][input_ids.shape[1]:], skip_special_tokens=False)

        # Stage 2: constrained output via outlines
        import outlines  # noqa: PLC0415
        generator = outlines.generate.json(model, schema)
        full_prefix = prompt + thinking_text
        if not full_prefix.endswith("</think>\n"):
            full_prefix += "</think>\n"
        structured_text = generator(full_prefix)
        result_json = structured_text if isinstance(structured_text, str) else structured_text.model_dump_json()
        return GenerationResult(text=result_json, thinking=thinking_text, tokens_used=len(thinking_text.split()))

    return await asyncio.to_thread(_run)
```

- [ ] **Step 4: Commit stubs**

```bash
git add src/rune/model/
git commit -m "feat: add model layer stubs (inference, hypernetwork, adapter)"
```

---

## Phase 4: Training — Carry v1 Files

### Task 12: Carry training files from v1

**Files:**
- Create: `src/rune/training/diff_loss.py`, `src/rune/training/oracle_cache.py`, `src/rune/training/config.py`, `src/rune/training/gate.py`

- [ ] **Step 1: Copy `diff_loss.py` from v1 and update imports**

```bash
git show main:libs/model-training/src/model_training/diff_loss.py > src/rune/training/diff_loss.py
```

Then find-and-replace all `model_training.` imports with `rune.training.`:

```bash
sed -i '' 's/from model_training\./from rune.training./g; s/import model_training\./import rune.training./g' src/rune/training/diff_loss.py
```

- [ ] **Step 2: Copy `oracle_cache.py` and update imports**

```bash
git show main:libs/model-training/src/model_training/oracle_cache.py > src/rune/training/oracle_cache.py
sed -i '' 's/from model_training\./from rune.training./g; s/import model_training\./import rune.training./g' src/rune/training/oracle_cache.py
```

- [ ] **Step 3: Copy `round2_config.py` → `config.py` and update**

```bash
git show main:libs/model-training/src/model_training/round2_config.py > src/rune/training/config.py
sed -i '' 's/from model_training\./from rune.training./g; s/import model_training\./import rune.training./g; s/sakana_checkpoint_path/checkpoint_path/g' src/rune/training/config.py
```

- [ ] **Step 4: Create `gate.py` — success gate**

```python
# src/rune/training/gate.py
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

MIN_BENCHMARKS_PASSING = 4
MIN_IMPROVEMENT = 2.0
MAX_REGRESSION = 1.0


@dataclass(frozen=True)
class GateResult:
    passed: bool
    passing_benchmarks: int
    total_benchmarks: int
    improvements: dict[str, float]
    regressions: dict[str, float]


def evaluate_gate(
    baseline_scores: dict[str, float],
    new_scores: dict[str, float],
) -> GateResult:
    improvements: dict[str, float] = {}
    regressions: dict[str, float] = {}

    for bench, new_score in new_scores.items():
        base_score = baseline_scores.get(bench, 0.0)
        delta = new_score - base_score
        if delta >= MIN_IMPROVEMENT:
            improvements[bench] = delta
        elif delta < -MAX_REGRESSION:
            regressions[bench] = delta

    passed = (
        len(improvements) >= MIN_BENCHMARKS_PASSING
        and len(regressions) == 0
    )
    return GateResult(
        passed=passed,
        passing_benchmarks=len(improvements),
        total_benchmarks=len(new_scores),
        improvements=improvements,
        regressions=regressions,
    )
```

- [ ] **Step 5: Write gate test**

```python
# tests/unit/test_gate.py
from rune.training.gate import evaluate_gate


class TestEvaluateGate:
    def test_passes_with_sufficient_improvements(self) -> None:
        baseline = {"a": 10.0, "b": 10.0, "c": 10.0, "d": 10.0, "e": 10.0, "f": 10.0}
        new = {"a": 13.0, "b": 14.0, "c": 12.5, "d": 15.0, "e": 10.5, "f": 10.0}
        result = evaluate_gate(baseline, new)
        assert result.passed is True
        assert result.passing_benchmarks >= 4

    def test_fails_with_regression(self) -> None:
        baseline = {"a": 10.0, "b": 10.0, "c": 10.0, "d": 10.0}
        new = {"a": 13.0, "b": 14.0, "c": 12.5, "d": 7.0}  # d regresses
        result = evaluate_gate(baseline, new)
        assert result.passed is False
        assert "d" in result.regressions

    def test_fails_with_too_few_improvements(self) -> None:
        baseline = {"a": 10.0, "b": 10.0, "c": 10.0, "d": 10.0}
        new = {"a": 13.0, "b": 10.5, "c": 10.0, "d": 10.0}  # only 1 improves enough
        result = evaluate_gate(baseline, new)
        assert result.passed is False
```

- [ ] **Step 6: Run gate tests — expect PASS**

Run: `uv run pytest tests/unit/test_gate.py -v`

- [ ] **Step 7: Commit**

```bash
git add src/rune/training/ tests/unit/test_gate.py
git commit -m "feat: carry v1 training files + add success gate"
```

---

## Phase 5: CLI + Benchmark

### Task 13: CLI entry point (`src/rune/cli.py`)

**Files:**
- Create: `src/rune/cli.py`
- Test: `tests/unit/test_cli.py`

- [ ] **Step 1: Write tests**

```python
# tests/unit/test_cli.py
from typer.testing import CliRunner

from rune.cli import app

runner = CliRunner()


class TestCLI:
    def test_help(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "run" in result.output
        assert "train" in result.output
        assert "bench" in result.output
        assert "mine" in result.output

    def test_run_requires_task(self) -> None:
        result = runner.invoke(app, ["run"])
        assert result.exit_code != 0

    def test_bench_help(self) -> None:
        result = runner.invoke(app, ["bench", "--help"])
        assert result.exit_code == 0
        assert "hpo" in result.output.lower() or "n-trials" in result.output.lower()
```

- [ ] **Step 2: Run tests — expect FAIL**

Run: `uv run pytest tests/unit/test_cli.py -v`

- [ ] **Step 3: Implement `cli.py`**

```python
# src/rune/cli.py
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(name="rune", help="Local-first coding agent with hypernetwork LoRA adapters")


@app.command()
def run(
    task: str = typer.Argument(..., help="Task description"),
    config: Optional[Path] = typer.Option(None, help="Path to config JSON"),
    checkpoint: Optional[str] = typer.Option(None, help="Hypernetwork checkpoint path"),
) -> None:
    """Run a single task through the engine."""
    from rune.config import PipelineConfig, load_config  # noqa: PLC0415

    cfg = load_config(config) if config else PipelineConfig()
    if checkpoint:
        cfg = cfg.override(checkpoint_path=checkpoint)

    typer.echo(f"Running task: {task}")
    # Engine invocation happens here during implementation
    raise NotImplementedError("Engine invocation not yet implemented")


@app.command()
def train(
    corpus_dir: Optional[Path] = typer.Option(None, help="Training corpus directory"),
    config: Optional[Path] = typer.Option(None, help="Config JSON path"),
    hpo: bool = typer.Option(False, help="Run Optuna HPO"),
    n_trials: int = typer.Option(50, help="Number of HPO trials"),
) -> None:
    """Train hypernetwork (oracle → distillation → gate)."""
    typer.echo(f"Training {'with HPO' if hpo else 'single run'}")
    raise NotImplementedError("Training not yet implemented")


@app.command()
def mine(
    sessions_dir: Path = typer.Option(..., help="Directory of coding sessions"),
    output_dir: Path = typer.Option(..., help="Output corpus directory"),
) -> None:
    """Mine coding sessions into training corpus."""
    typer.echo(f"Mining {sessions_dir} → {output_dir}")
    raise NotImplementedError("Mining not yet implemented")


@app.command()
def bench(
    tasks_file: Optional[Path] = typer.Option(None, help="Benchmark tasks JSON"),
    config: Optional[Path] = typer.Option(None, help="Config JSON path"),
    hpo: bool = typer.Option(False, help="Run Optuna HPO"),
    n_trials: int = typer.Option(50, help="Number of HPO trials"),
) -> None:
    """Run benchmark suite, optionally with HPO."""
    typer.echo(f"Benchmarking {'with HPO' if hpo else 'single pass'}")
    raise NotImplementedError("Benchmarking not yet implemented")


if __name__ == "__main__":
    app()
```

- [ ] **Step 4: Run tests — expect PASS**

Run: `uv run pytest tests/unit/test_cli.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/rune/cli.py tests/unit/test_cli.py
git commit -m "feat: add typer CLI with run/train/mine/bench commands"
```

---

### Task 14: Benchmark runner stub (`src/rune/bench/runner.py`)

**Files:**
- Create: `src/rune/bench/runner.py`

- [ ] **Step 1: Create benchmark runner**

```python
# src/rune/bench/runner.py
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BenchTask:
    task_id: str
    description: str
    test_code: str
    entry_point: str = "solution"


@dataclass(frozen=True)
class TaskResult:
    task_id: str
    passed: bool
    code: str
    stderr: str


@dataclass
class BenchResult:
    pass_at_1: float
    total_tasks: int
    passed_tasks: int
    per_task: list[TaskResult] = field(default_factory=list)


def load_tasks(path: Path) -> list[BenchTask]:
    data = json.loads(path.read_text())
    return [BenchTask(**t) for t in data]


def run_benchmark(
    tasks: list[BenchTask],
    engine: Any,
    config: dict[str, Any],
) -> BenchResult:
    results: list[TaskResult] = []
    for task in tasks:
        # Each task is run through the engine
        # Implementation filled in during subagent execution
        raise NotImplementedError("Benchmark execution not yet implemented")
    passed = sum(1 for r in results if r.passed)
    return BenchResult(
        pass_at_1=passed / len(results) if results else 0.0,
        total_tasks=len(results),
        passed_tasks=passed,
        per_task=results,
    )
```

- [ ] **Step 2: Commit**

```bash
git add src/rune/bench/runner.py
git commit -m "feat: add benchmark runner stub"
```

---

## Phase 6: Lint, Typecheck, Verify

### Task 15: Quality gate

- [ ] **Step 1: Run ruff**

Run: `uv run ruff check .`
Expected: No errors (fix any that appear).

- [ ] **Step 2: Run mypy**

Run: `uv run mypy src/`
Expected: Clean or only expected GPU library ignores.

- [ ] **Step 3: Run all unit tests**

Run: `uv run pytest tests/unit/ -v`
Expected: All tests PASS.

- [ ] **Step 4: Run integration tests**

Run: `uv run pytest tests/integration/ -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit any fixes**

```bash
git add -A
git commit -m "fix: resolve lint and type errors from quality gate"
```

---

## Summary: What's Built After This Plan

After all 15 tasks, you have:

| Layer | Status | Key files |
|---|---|---|
| Infrastructure | Complete | pyproject.toml, Makefile, CLAUDE.md, PRODUCT.md |
| Templates | Copied from v1 | src/rune/templates/*.j2 |
| State types | Complete + tested | src/rune/engine/state.py |
| Config | Complete + tested | src/rune/config.py |
| Sandbox | Complete + tested | src/rune/sandbox/executor.py |
| Registry | Complete + tested | src/rune/registry/store.py |
| Policy | Complete + tested | src/rune/engine/policy.py |
| Parse | Complete + tested | src/rune/engine/parse.py |
| Interfaces | Complete + tested | src/rune/engine/interfaces.py |
| Graph | Complete + tested | src/rune/engine/graph.py |
| Model layer | Stubs (GPU-dependent) | src/rune/model/*.py |
| Training | Carried from v1 + gate tested | src/rune/training/*.py |
| CLI | Complete + tested | src/rune/cli.py |
| Benchmark | Stub | src/rune/bench/runner.py |

**Next steps after this plan:**
1. **Fill PRODUCT.md** — required before any further development
2. **Subagent-driven implementation** — fill in model layer stubs (requires GPU), wire CLI to engine, implement benchmark runner
3. **Mode 2: Empirical template optimization** — MBPP → Pass@1 → additional benchmarks
