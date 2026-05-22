# Rune v2: Simplified Single-Loop Architecture

## Overview

Rewrite of the rune codebase from a 72K-line multi-service monorepo to a single ~3-4K line Python package (plus ~1.5K lines of training code carried from v1 with modifications). The core insight: the paper describes one recursive loop (state → H(state) → adapter → generate → observe → repeat), but the current implementation has 5 overlapping iteration mechanisms across 32 entry-point scripts, 7 libs, and 5 FastAPI services.

The simplified version has four concerns: **mine, train, run, benchmark**. Everything else is cut.

**All cuts are provisional until PRODUCT.md is filled in.** PRODUCT.md is currently empty (`<!-- TODO -->` stubs). Before writing any v2 code, PRODUCT.md must define `do-not-break:`, `out-of-scope:`, and `regulatory-surface:` sections. Any cut below may be reversed if PRODUCT.md reveals the feature is load-bearing.

## Development Approach

**Fresh branch on `rune`.** Delete everything, rebuild from scratch.

**From ElixirTrials Template repo** (`/Users/noahdolevelixir/Code/Template`) — copy + modify:
- `pyproject.toml` — uv workspace structure, quality tooling config
- `Makefile` — workspace-aware lint/test/typecheck/docs targets
- `PRODUCT.md` — interview-mode template (fill in before building)
- `.claude/` — skills, hooks, commands infrastructure
- `infra/docker-compose.yml` — Postgres + MLflow
- `.claudeignore`, `.gitignore`, `mkdocs.yml`

**From current `rune`** — copy + modify (not verbatim — imports change):
- `diff_loss.py` (~1250 lines) — DiffAwareSFTTrainer + DiffWeightedDataCollator. Hunk-weighted token loss is load-bearing. Modify: update import paths from `model_training.*` to `rune.training.*`
- `oracle_cache.py` — LRU adapter cache (max 4). Modify: update imports
- `round2_config.py` — Round2TrainConfig. Modify: update imports, rename `sakana_checkpoint_path` → `checkpoint_path` (points to HPO-selected hypernetwork)
- `d2l_train.py` — KL+CE distillation loop. Modify: update imports, replace all `sakana_checkpoint_path` references with `checkpoint_path`. Architecture config and weights both load from the HPO checkpoint — Sakana is no longer in the chain
- `templates/*.j2` — all Jinja2 templates as starting point for empirical optimization
- SQLite registry schema

**From MLflow** — the one bootstrap artifact:
- HPO-selected hypernetwork checkpoint — already trained through oracle → distillation → gate pipeline

**From HuggingFace** — downloaded at runtime:
- `Qwen/Qwen3.5-9B` base model (configurable via `engine.model_id`)

### Two Development Modes

**Mode 1: TDD (deterministic, planned)** — build the machinery that templates plug into. Every module has tests before implementation.
1. Config, state, registry, sandbox
2. Model layer (inference, adapter hotswap, hypernetwork loading)
3. Engine (graph, policy, parse, interfaces)
4. Training pipeline (oracle, hypernetwork, diff_loss, gate)
5. CLI entry points
6. Done when: `uv run rune run "trivial task"` completes end-to-end using the MLflow checkpoint

**Mode 2: Empirical (goal-driven, results-based)** — optimize Jinja2 prompt and adapter templates. Iterate based on benchmark scores.
- Benchmark progression (each gates the next):
  1. **MBPP** — get good scores first (simplest coding benchmark, validates core loop)
  2. **Pass@1** — broader coding task performance
  3. **Additional benchmarks** — expand once fundamentals are solid
- Experiment loop: hypothesize template change → `uv run rune bench` → compare scores → keep or revert
- HPO: `uv run rune train --hpo` (training params) + `uv run rune bench --hpo` (engine params)
- MLflow logs every experiment
- Done when: good absolute MBPP scores, then good Pass@1

## Package Structure

```
rune/
├── .claude/                    # From ElixirTrials Template
│   ├── hooks/guard.py          # Adapted: drop FastAPI/Alembic, keep git/security
│   ├── hooks/post-edit-lint.sh
│   ├── hooks/session-start.sh
│   ├── settings.json
│   ├── skills/                 # Subset: plan-first, debug-systematic, commit-prep,
│   │                           #   long-run-fallback, pr-description
│   ├── skills-index.md
│   └── commands/
├── .github/workflows/ci.yml   # Simplified: no frontend, no multi-service discovery
├── .claudeignore               # From Template
├── CLAUDE.md                   # Rewritten for rune (no services, no medtech)
├── PRODUCT.md                  # From Template structure, filled in for rune
├── Makefile                    # Simplified: lint, test, typecheck, docs
├── pyproject.toml              # Single package, uv managed
├── src/
│   └── rune/
│       ├── __init__.py
│       ├── cli.py              # typer CLI: rune mine|train|run|bench
│       ├── config.py           # PipelineConfig (flattened dataclass) + env var overrides
│       ├── engine/
│       │   ├── __init__.py
│       │   ├── graph.py        # StateGraph, step_node, conditional edge (~150 lines)
│       │   ├── policy.py       # select_action() → list[Action], DAG layer grouping (~100 lines)
│       │   ├── state.py        # RunState TypedDict, Subtask, StepRecord, Action (~60 lines)
│       │   ├── parse.py        # parse_output match-case dispatcher (~80 lines)
│       │   └── interfaces.py   # tree-sitter interface extraction (~60 lines)
│       ├── model/
│       │   ├── __init__.py
│       │   ├── inference.py    # Two-stage generation: free-form thinking → outlines constrained (~120 lines)
│       │   ├── hypernetwork.py # Perceiver loading from HPO checkpoint, trajectory→adapter (~100 lines)
│       │   └── adapter.py      # hotswap_adapter_from_state_dict, async serialization (~80 lines)
│       ├── training/
│       │   ├── __init__.py
│       │   ├── oracle.py       # Round-1: per-bin QLoRA, oracle validation (~200 lines)
│       │   ├── hypernetwork.py # Round-2: distillation, functional LoRA, KL+CE loss (~300 lines)
│       │   ├── diff_loss.py    # FROM V1 + MODIFIED IMPORTS (~1250 lines)
│       │   ├── oracle_cache.py # FROM V1 + MODIFIED IMPORTS (~80 lines)
│       │   ├── gate.py         # Success gate evaluation (~60 lines)
│       │   └── config.py       # Round2TrainConfig, engine config (~80 lines)
│       ├── registry/
│       │   ├── __init__.py
│       │   └── store.py        # SQLite CRUD, age-based pruning (~150 lines)
│       ├── sandbox/
│       │   ├── __init__.py
│       │   └── executor.py     # Subprocess code execution (~80 lines)
│       ├── templates/          # Jinja2 trajectory + prompt templates (FROM V1, optimized in Mode 2)
│       │   ├── decompose.j2
│       │   ├── prompt_decompose.j2
│       │   ├── plan.j2
│       │   ├── prompt_plan.j2
│       │   ├── code.j2
│       │   ├── prompt_code.j2
│       │   ├── code_retry.j2
│       │   ├── prompt_code_retry.j2
│       │   ├── integrate.j2
│       │   ├── prompt_integrate.j2
│       │   ├── diagnose.j2
│       │   ├── prompt_diagnose.j2
│       │   ├── code_repair.j2
│       │   └── prompt_code_repair.j2
│       └── bench/
│           ├── __init__.py
│           └── runner.py       # Benchmark harness, Optuna HPO (~200 lines)
├── tests/
│   ├── unit/                   # No GPU, no disk, fast
│   │   ├── test_policy.py
│   │   ├── test_state.py
│   │   ├── test_parse.py
│   │   ├── test_interfaces.py
│   │   ├── test_config.py
│   │   ├── test_registry.py
│   │   ├── test_gate.py
│   │   └── test_cli.py
│   ├── integration/            # Mocked model, real sandbox
│   │   ├── test_graph.py
│   │   ├── test_sandbox.py
│   │   ├── test_adapter.py
│   │   ├── test_inference.py
│   │   └── test_dag_e2e.py
│   └── gpu/                    # @pytest.mark.gpu, CI-optional
│       ├── test_hypernetwork.py
│       ├── test_oracle.py
│       ├── test_training.py
│       └── test_bench.py
├── docs/
├── infra/
│   └── docker-compose.yml      # Postgres + MLflow
├── mkdocs.yml
└── scripts/                    # Minimal: check-all.sh, setup-git-hooks.sh
```

~25 source files, ~3-4K new lines + ~1.5K carried from v1 (with import modifications). One `pyproject.toml`, one `uv sync`, one `uv run pytest`.

### Template Repo Alignment

The `.claude/` directory, `CLAUDE.md`, `PRODUCT.md`, `Makefile`, `.claudeignore`, `.github/workflows/ci.yml`, and `mkdocs.yml` are adapted from the ElixirTrials Template repo.

Key adaptations:
- **Guard hooks**: drop Alembic, FastAPI, frontend blocks; keep git mutability guards, security, long-run blocking
- **Skills**: subset relevant to research (plan-first, debug-systematic, commit-prep, long-run-fallback, pr-description); drop compliance, ux-pathway, frontend-design, alembic-safe, event-trace
- **CI**: single job (lint + typecheck + test), no path-filter, no frontend
- **Makefile**: `make lint`, `make test`, `make typecheck`, `make docs-build`

## Engine: Single-Loop LangGraph

### The Core Loop

One LangGraph `StateGraph` with a single node and a conditional self-edge:

```
step ──→ should_continue? ─── continue ──→ step
                           └── done ──────→ END
```

Each iteration: policy picks action(s) from the library → render trajectory + prompt → generate adapter → generate output → execute if needed → update state.

```python
from langgraph.graph import StateGraph, END

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

### RunState

```python
class RunState(TypedDict):
    task: str
    subtasks: list[Subtask]          # each has depends_on: list[str] (subtask names)
    interfaces: dict[str, str]       # subtask_name → extracted interfaces (tree-sitter)
    plans: dict[str, str]
    code_results: dict[str, str]     # subtask_name → generated code
    code_passed: dict[str, bool]     # subtask_name → last exit_code == 0
    retries: dict[str, int]          # subtask_name → retry count
    integrated_code: str
    current_adapter: str | None
    feedback: Feedback | None
    diagnosis: str | None            # consumed by next integrate retry
    actions: list[Action]            # always written by step_node — empty = done
    trajectory: list[StepRecord]
    step: int
    budget_remaining: int
```

State grows unboundedly — by design. Templates select which slice of state to render for each step, so trajectory text stays bounded per step while full history is preserved for the registry and debugging.

### Action Template Library

```python
@dataclass(frozen=True)
class Action:
    name: str
    trajectory_template: str
    prompt_template: str
    system_prompt: str
    output_schema: type[BaseModel] | None   # outlines constrained when not None
    executes_code: bool

@dataclass(frozen=True)
class Subtask:
    name: str
    description: str
    depends_on: list[str]           # subtask names, not indices — stable under reordering
```

Adding a new action = one entry in `ACTIONS` dict + one Jinja2 template pair.

### Policy: Batched Action Selection

```python
def select_action(state: RunState) -> list[Action]:
```

Returns a **list** of actions. Independent subtasks (those whose dependencies are all satisfied) are returned as a batch. The engine processes them in a single step:

- **Inference is sequential** — one adapter in GPU at a time, each action gets its own hypernetwork pass + adapter hotswap + LLM generation
- **Sandbox execution is parallel** — after all code is generated for a batch, `asyncio.gather` executes sandbox runs concurrently
- **Hypernetwork forward passes** may overlap with sandbox execution (CPU-bound, no GPU contention)

This avoids the false promise of "parallel inference" while still benefiting from batched sandbox execution for independent subtasks.

Returning an empty list signals completion — `should_continue` routes to `END`.

### DAG Dependencies (No Blackboard)

Dependencies flow through RunState, not a separate Blackboard class:

1. Decompose phase emits `Subtask` objects with `depends_on` names (not indices — stable under reordering)
2. `select_action()` groups subtasks into execution layers via topological sort (~5 lines, uses `graphlib.TopologicalSorter`)
3. Layer 0 (no deps) subtasks are batched for the current step
4. After each subtask completes, tree-sitter extracts interfaces → `state["interfaces"][name]`
5. Layer N subtasks get predecessor interfaces rendered into their trajectory template via Jinja2: `{{ interfaces[dep_name] }}`

Interface extraction uses **tree-sitter** AST parsing (not regex). The project already has tree-sitter via the code-review-graph tool. This extracts function signatures, class definitions, and type annotations reliably.

### step_node

Injected dependencies via LangGraph's `configurable` dict (not stored in RunState):

```python
async def step_node(state: RunState, config: RunnableConfig) -> dict:
    model = config["configurable"]["model"]
    registry = config["configurable"]["registry"]
    run_config = config["configurable"]["run_config"]

    actions = select_action(state)
    if not actions:
        return {"actions": [], "budget_remaining": state["budget_remaining"]}

    results: list[tuple[Action, str, str]] = []  # (action, subtask_name, raw_output)
    for action in actions:
        ctx = state_to_ctx(state)
        trajectory_text = render_template(action.trajectory_template, **ctx)
        prompt_text = render_template(action.prompt_template, **ctx)

        # Sequential: one adapter in GPU at a time
        adapter = model.generate_adapter(trajectory_text)
        model.hotswap_adapter(adapter.state_dict)
        result = model.generate(
            prompt=prompt_text,
            system_prompt=action.system_prompt,
            output_schema=action.output_schema,
            max_tokens=run_config.max_tokens,
        )
        # Background: async disk serialization (no GPU contention)
        asyncio.create_task(registry.persist_adapter(adapter))
        target_name = action.target_subtask  # set by policy
        results.append((action, target_name, result.text))

    # Parallel sandbox execution — results aligned by index to code_actions
    code_actions = [(a, name, text) for a, name, text in results if a.executes_code]
    sandbox_results = await asyncio.gather(*[
        asyncio.to_thread(run_in_sandbox, text) for _, _, text in code_actions
    ])
    # Zip back: sandbox_results[i] corresponds to code_actions[i]
    feedback_map: dict[str, ExecutionResult] = {
        name: fb for (_, name, _), fb in zip(code_actions, sandbox_results)
    }

    updates = merge_action_results(results, feedback_map, state)
    updates["actions"] = actions
    updates["step"] = state["step"] + 1
    updates["budget_remaining"] = state["budget_remaining"] - 1
    return updates

def should_continue(state: RunState) -> str:
    if not state["actions"] or state["budget_remaining"] <= 0:
        return "done"
    return "continue"
```

**Stale-state prevention**: `step_node` always writes `actions` (empty list or populated). `should_continue` checks truthiness of `actions`, not a nullable field that could persist from a prior step.

**Sandbox alignment**: `code_actions` and `sandbox_results` are aligned by construction — same list, same indices. `feedback_map` keys by subtask name for unambiguous lookup in `merge_action_results`.

### Structured Output: Two-Stage Generation

Actions that need structured responses (decompose, diagnose) set `output_schema` to a Pydantic model class. Constrained decoding uses **`outlines`** library (not XGrammar, not vLLM).

**Problem**: Constrained decoding and thinking tokens are incompatible — outlines applies token masks to the entire output, corrupting `<think>...</think>` blocks.

**Solution**: Two-stage generation:
1. **Stage 1 (thinking)**: Free-form generation with `<think>` token. Stop generation at `</think>` token.
2. **Stage 2 (structured output)**: Use outlines Pydantic schema constraint on the response portion only.

```python
def generate_structured(
    model, tokenizer, prompt: str, schema: type[BaseModel],
    thinking_budget: int = 1024,
) -> BaseModel:
    # Stage 1: free-form thinking — returns output + KV cache
    thinking_output, past_kv = generate_with_stop(
        model, tokenizer, prompt,
        stop_token="</think>",
        max_tokens=thinking_budget,
        return_past_key_values=True,
    )
    # Stage 2: constrained output — continues from KV cache, no re-encoding
    structured_output = outlines.generate.json(
        model, schema,
        past_key_values=past_kv,
    )(thinking_output + "</think>\n")
    return structured_output
```

**KV-cache continuation**: Stage 2 extends generation from the `</think>` position using `past_key_values` from Stage 1. The full prompt + thinking tokens are not re-encoded. Without this, latency roughly doubles for every structured action. If outlines does not support `past_key_values` passthrough, fall back to re-encoding with an explicit performance note — this is an optimization target, not a blocker.

Code actions leave `output_schema=None` — single-stage free-form generation with thinking enabled.

### Model: Qwen/Qwen3.5-9B

Default base model is `Qwen/Qwen3.5-9B` (9.6B params, Qwen3.5 hybrid-attention architecture with `text_config` wrapped in a VL config). Configurable via `model_id` in config. The **exact HuggingFace model ID** must match what the HPO checkpoint was trained against — architecture mismatch will silently produce wrong adapter shapes.

The model ID is pinned in the model registry (`model_configs.py` line 116: `model_id="Qwen/Qwen3.5-9B"`). The warm-start adapter is `danielcherubini/Qwen3.5-DeltaCoder-9B`.

Rationale for keeping this model:
- Existing hypernetwork checkpoint is trained against this exact architecture
- Demonstrates adapter approach works with models not specifically trained for coding
- Hybrid attention (full + linear layers) is the architecture the perceiver targets

<!-- TODO: evaluate swapping to a top-tier coding model once v2 is stable -->

### Templates

Trajectory templates (fed to hypernetwork) and prompt templates (fed to LLM) are Jinja2 files in `src/rune/templates/`. The distinction is encoded in the `Action` dataclass.

```python
from jinja2 import Environment, PackageLoader

_env = Environment(loader=PackageLoader("rune", "templates"))

def render_template(template_name: str, **kwargs) -> str:
    return _env.get_template(f"{template_name}.j2").render(**kwargs)
```

Templates are the primary lever for Mode 2 empirical optimization. Template quality directly determines retry rate — the #1 performance bottleneck.

## Adapter Lifecycle: Single-Slot GPU

**One adapter in GPU memory at a time.** No pooling, no concurrent loading.

```
Per action in step_node:
  1. hypernetwork(trajectory_text) → LoRA state_dict (CPU tensors)
  2. hotswap_adapter_from_state_dict(model, state_dict)  # in-place, constant VRAM
  3. LLM generates output
  4. background: asyncio.create_task(persist_adapter(state_dict, adapter_id))
```

**Hard constraint**: All generated adapters share the same `target_modules` list. The hypernetwork architecture guarantees this — perceiver output dimensions are fixed at init. `hotswap_adapter_from_state_dict` always works without reloading the base model.

**Registry**: SQLite via `registry/store.py`. Each adapter gets:
- `adapter_id` (UUID)
- `task_id`, `phase`, `iteration`
- `parent_id` (lineage tracking — preserved from v1)
- `created_at` timestamp
- `disk_path` (async-written safetensors)

**Pruning**: Age-based. On startup and every N iterations, delete adapters older than configurable TTL (default 7 days). Registry DELETE + disk unlink.

**What's cut**: ModelPool class (implied concurrent adapters), TIES/DARE merging, evolution/fitness scoring, kill switch.

**What's preserved**: `hotswap_adapter_from_state_dict` (PEFT API), SQLite registry with lineage, adapter serialization to disk.

## Training Pipeline: Oracle + Hypernetwork + DiffAware

**Three trainers, not one.** The actual pipeline is load-bearing and must be preserved.

```
Round 1: Oracle Training (per-bin)
  ├── 25-bin corpus (produced by rune mine)
  ├── Per-bin QLoRA fine-tuning → oracle adapters
  ├── Validation gate: ≥3% Pass@1 improvement per oracle
  └── Output: oracle_<bin_key> adapters in registry

Round 2: Hypernetwork Distillation
  ├── DiffAwareSFTTrainer (hunk-weighted token loss) — PRESERVED
  ├── apply_functional_lora + KL+CE loss
  ├── Oracle adapter cache (LRU max 4)
  └── Output: trained hypernetwork checkpoint

Success Gate:
  ├── evaluate_round2_gate
  ├── ≥4/6 benchmarks ≥2.0% Pass@1 improvement
  ├── No regression >1.0% on any benchmark
  └── Exit code: 0 PASS / 1 FAIL
```

### Files carried from v1 (with import modifications)

These files are carried because they encode hard-won training logic. They are **not** verbatim copies — import paths change from `model_training.*` to `rune.training.*`:

| File | Lines | What changes |
|---|---|---|
| `diff_loss.py` | ~1250 | Import paths only. Core logic untouched. |
| `oracle_cache.py` | ~80 | Import paths only. |
| `round2_config.py` | ~67 | Import paths. `sakana_checkpoint_path` → `checkpoint_path` (HPO-selected hypernetwork only). |
| `d2l_train.py` | ~844 | Import paths. Replace `sakana_checkpoint_path` with `checkpoint_path`. Remove Sakana-specific logic — architecture config and weights both come from the HPO checkpoint. |

### Training objective: Doc2LoRA vs trajectory-aware

The original Sakana Doc2LoRA trains a hypernetwork to internalize **document content** via NIAH/QA reconstruction objectives. Rune's hypernetwork generates **task-solving adapters from trajectory text** — a different input distribution. The carried `d2l_train.py` has already been adapted for Rune's trajectory-aware training: it uses `compute_kl_ce_loss` to distill oracle adapter behavior into the hypernetwork, with the oracle providing the teacher signal (not document reconstruction). The training data is trajectory JSONL, not document QA pairs.

However, this assumption must be **validated during Mode 1**: confirm that the loss function in `d2l_train.py`'s `_training_step` operates on trajectory inputs and that the data loader (`d2l_data.load_jsonl`) expects trajectory records. If the loss function still contains document-reconstruction terms from the original Sakana code, replace them before any training run — the hypernetwork will silently underfit trajectory inputs without erroring.

### HPO

Two Optuna surfaces, both with CLI access:

```bash
uv run rune train --hpo --n-trials 50    # training HPO
uv run rune bench --hpo --n-trials 50    # engine HPO
```

Both use `MLflowCallback(mlflow_kwargs={"nested": True})` for nested trial logging.

**Training HPO tunes**: learning rate, warmup ratio, LoRA rank, NEFTune alpha
**Engine HPO tunes**: adapter_scaling, temperature, max_tokens, max_phase_iterations, template selection

**Required dependency**: `optuna-integration[mlflow]` — must be in `pyproject.toml`.

## Observability: MLflow

Single tracing spine. MLflow is the only observability backend.

```python
mlflow.langchain.autolog(run_tracer_inline=True)
```

LangGraph node traces for free — every `step_node` invocation logged as a span. Thread isolation via LangGraph's `configurable` dict:

```python
graph.invoke(state, config={"configurable": {"thread_id": run_id}})
```

**Per-run logging**: action selected, adapter_id, token counts, latency, sandbox pass/fail, total iterations, final status, benchmark score.

**Training logging**: per-epoch loss, eval metrics, gate pass/fail (via existing `mlflow_log_params` helper).

**HPO logging**: Optuna trials nest under parent MLflow run. Each trial is a child run.

**What's cut**: Custom OpenTelemetry, Prometheus, distributed tracing, Grafana dashboards.

## Config: Flattened

```python
@dataclass(frozen=True)
class PipelineConfig:
    model_id: str = "Qwen/Qwen3.5-9B"
    adapter_scaling: float = 0.075       # HPO-tunable, current best from v1
    temperature: float = 0.3
    max_tokens: int = 2048
    repetition_penalty: float = 1.1
    top_p: float = 0.9
    thinking_budget: int = 1024
    phase_max_tokens: dict[str, int] = field(default_factory=dict)
    max_phase_iterations: int = 5        # HPO-tunable
    prompt_style: str = "skeleton"
    trajectory_style: str = "prose"
    adapter_ttl_days: int = 7
```

Env var overrides preserved: `RUNE_TEMPERATURE`, `RUNE_MAX_TOKENS`, `RUNE_REPETITION_PENALTY`, `RUNE_TOP_P`, `RUNE_THINKING_BUDGET`, `RUNE_MAX_TOKENS_{PHASE}`, `RUNE_MAX_PHASE_ITERATIONS`.

**What's cut**: Separate `AdapterConfig`, `GenerationConfig`, `PromptConfig`, `TrajectoryConfig`, `CalibrationConfig`, `DecomposeConfig` dataclasses. `ReasoningLoopConfig` entirely (collapse detection, sliding windows, merge methods — all cut with the reasoning loop).

## CLI: Four Commands

Built with **`typer`**, single `cli.py` entry point:

```bash
uv run rune mine --sessions-dir ./sessions --output-dir ./corpus
# → bins trajectories, produces JSONL shards

uv run rune train --corpus-dir ./corpus --config ./config.json
# → Round-1 oracles → Round-2 hypernetwork → success gate
# → exits 0/1
uv run rune train --hpo --n-trials 50
# → Optuna tunes learning rate, warmup, LoRA rank, NEFTune alpha

uv run rune run "implement a binary search tree" --config ./config.json
# → single LangGraph invocation, prints final code to stdout
# → MLflow run logged

uv run rune bench --tasks-file ./tasks.json
# → runs benchmark suite, prints Pass@1 scores
uv run rune bench --hpo --n-trials 50
# → Optuna tunes engine params (scaling, temperature, max_iterations)
```

## Test Surface: Fresh Suite

No migration from v1. Clean test suite designed for v2 architecture.

**Tier 1 — Unit tests (no GPU, no disk, fast)**:
- `test_policy.py`: action selection logic, DAG layer grouping, topological sort
- `test_state.py`: RunState construction, Subtask dependency validation
- `test_parse.py`: output parsing for each action type
- `test_interfaces.py`: tree-sitter extraction on known Python snippets
- `test_config.py`: PipelineConfig load/save/override, env var resolution
- `test_registry.py`: SQLite CRUD, age-based pruning (in-memory `:memory:` DB)
- `test_gate.py`: success gate pass/fail logic on synthetic scores
- `test_cli.py`: typer command parsing, flag validation

**Tier 2 — Integration tests (mocked model, real sandbox)**:
- `test_graph.py`: full graph invocation with mock LLM, verify state transitions, iteration limits, conditional edge termination
- `test_sandbox.py`: real subprocess execution, timeout handling, error capture
- `test_adapter.py`: hotswap with synthetic state_dicts, async serialization to temp dir
- `test_inference.py`: two-stage generation with tiny test model or mock
- `test_dag_e2e.py`: multi-subtask with dependencies, verify execution order and interface passing

**Tier 3 — GPU tests (marked, CI-optional)**:
- `test_hypernetwork.py`: real perceiver forward pass, verify output shapes match target_modules
- `test_oracle.py`: single-bin micro training run
- `test_training.py`: Round-2 micro run with DiffAwareSFTTrainer
- `test_bench.py`: single-task benchmark pass on trivial task

**Markers**: `@pytest.mark.gpu` for Tier 3 (skipped in CPU-only CI). No marker for Tiers 1 & 2 (must pass on every commit).

**Target**: ~200-300 focused tests covering the actual architecture.

## What Gets Cut

| Current | New |
|---|---|
| 7 libs (22K lines) | `src/rune/` (~25 files) |
| 5 FastAPI services (2.9K lines) | Gone |
| 32 entry-point scripts (15K+ lines) | 1 CLI, 4 subcommands |
| 4 inference providers + ABC | Single model class with hotswap |
| 5 overlapping retry mechanisms | 1 LangGraph conditional edge |
| Phase-scoped trajectories (discarded between phases) | 1 growing `RunState` |
| `rune_runner.py` (2699 lines) | `engine/graph.py` + `engine/policy.py` (~250 lines) |
| Swarm orchestration (swarm.py, workers, evolution) | Gone |
| Evolution / TIES/DARE merging | Gone |
| Kill switch (239 lines) | Gone (YAGNI) |
| SageMaker, contamination filter, audit | Gone |
| 1261 tests | ~200-300 fresh tests |
| ~72K lines Python | ~5K lines |

## Risks

1. **Two-stage generation latency**: The thinking + constrained output approach adds one extra forward pass per structured action. Mitigation: only decompose and diagnose use structured output — code/plan/integrate are free-form single-pass. Monitor latency via MLflow spans.

2. **Template compatibility with trained hypernetwork**: The hypernetwork is trained against specific template text patterns. If v2 templates produce different trajectory text for the same inputs, the checkpoint may degrade. Mitigation: templates are carried from v1 and modified incrementally in Mode 2, with MBPP scores as the regression gate.

3. **PRODUCT.md**: must be filled in before implementation begins. Determines whether any current features we're cutting are actually load-bearing.

4. **Unbounded state growth**: `trajectory: list[StepRecord]` accumulates all history. Mitigation: trajectory templates apply sliding window over StepRecord entries (keep last N steps) at render time. Full history stays in state for registry/debugging.

5. **Import path changes in carried files**: `diff_loss.py` and other carried files have internal imports. All `model_training.*` → `rune.training.*` changes must be validated by the test suite before any training runs.

## References

- [LangGraph best practices](https://www.swarnendu.de/blog/langgraph-best-practices/) — bounded cycles, TypedDict state, conditional edges
- [MLflow LangGraph tracing](https://mlflow.org/docs/latest/genai/tracing/integrations/listing/langgraph/) — `mlflow.langchain.autolog()`, `thread_id`, `run_tracer_inline=True`
- [Optuna MLflow integration](https://optuna-integration.readthedocs.io/en/stable/reference/generated/optuna_integration.MLflowCallback.html) — `MLflowCallback` with `nested=True`
- [outlines structured generation](https://github.com/dottxt-ai/outlines) — Pydantic schema constrained decoding
- [Qwen3 thinking tokens](https://qwen.readthedocs.io/en/latest/getting_started/concepts.html) — `<think>` / `</think>` token management
- [Two-stage structured generation](https://github.com/vllm-project/vllm/discussions/17638) — free-form thinking + constrained output workaround
