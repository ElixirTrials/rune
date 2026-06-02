# Rune Runner Simplification Proposal

## Problem

`rune_runner.py` is 2699 lines. The paper describes a single recursive loop:

```
state → H(state) → adapter → generate → observe → state' → repeat
```

The implementation has **five** overlapping iteration mechanisms:

1. **Per-phase retry loops** — each of the 5 phases (decompose, plan, code, integrate, repair) has its own `for evo_iter in range(iters_*)` loop with bespoke scoring rubrics, adapter naming schemes, and early-stop conditions.
2. **`_run_continuation_loop`** — handles truncated output (`finish_reason == "length"`) by concatenating fragments.
3. **`run_reasoning_loop`** — TIES/DARE adapter-compressed reasoning loop, triggered when accumulated context exceeds a budget.
4. **Per-subtask code retry** — `for attempt in range(iters_code)` inside `_code_subtask`, with two-step diagnose→repair on failure.
5. **Phase 5 diagnose→repair→re-integrate** — a repair outer loop over the integration result.

These mechanisms interact poorly. The P4/P5 branch already has three tasks that are symptoms of the mismatch:
- **T5** adds `continuation_phase` + `adapter_prefix` params to make continuation reusable outside Phase 3 — i.e., the function shouldn't be phase-specific.
- **T7** collapses the per-subtask retry to single-pass when reasoning loop fires, because both handle retries.
- **T9** (optional, unimplemented) proposes merging Phase 4+5, because separate integrate and repair loops are redundant.

Each phase also carries ~50 lines of boilerplate: adapter generation, adapter loading/unloading, registry registration, MLflow metrics, evolution sweeps. This is duplicated 5× with slight variations.

## What the paper actually needs

The paper's "parametric episodic memory" concept requires exactly:

1. **A trajectory** — accumulated state, actions, and feedback
2. **A hypernetwork call** — `H(trajectory) → LoRA adapter`
3. **A generation step** — base model + adapter → output
4. **A feedback step** — execute output, observe result, extend trajectory
5. **Repeat until done or budget exhausted**

Everything else — task decomposition, planning, code execution, integration — is **task-specific scaffolding**, not part of the memory mechanism.

### Should there be phases at all?

The paper describes ONE trajectory that grows over the full reasoning chain. The current implementation has five phase-scoped trajectories that don't carry state across phase boundaries — decompose's trajectory is discarded before plan starts. This directly contradicts the paper's claim that "the adapter reflects not just a document, but the evolving chain of attempts and fixes."

**A fully paper-aligned design** would have one growing trajectory:
```
trajectory = ""
while not done:
    trajectory += f"\n[Step {n}]: {action_description}"
    adapter = H(trajectory)
    output = generate(base_model + adapter, prompt)
    feedback = execute(output)
    trajectory += f"\n[Feedback]: {feedback}"
```

No phases. The model decides what to do next (decompose, write code, integrate) based on the trajectory's accumulated context. The hypernetwork compresses the full history into adapter weights, so context length isn't the bottleneck.

**Why we keep phases anyway (for now):** The hypernetwork's quality is not yet good enough for the model to self-direct multi-step coding. Phases provide deterministic structure that compensates for weak adapter steering. But the proposal is designed so that collapsing phases into a single loop is a future one-line change — replace the phase orchestrator with a single `run_phase("open_ended", ...)` call that uses a general-purpose trajectory template.

The key architectural decision: **extract the core loop so cleanly that phases become optional scaffolding, not structural load-bearing walls.**

## Proposed architecture

### 1. Extract a generic `AdapterReasoningStep`

One function that encapsulates the paper's core loop iteration:

```python
@dataclass
class StepResult:
    output: str
    adapter_id: str | None
    adapter_path: str | None
    feedback: StepFeedback  # stdout, stderr, exit_code, tests_passed, test_count

async def reasoning_step(
    trajectory: str,
    task_description: str,
    phase: str,
    session_id: str,
    iteration: int,
    *,
    graph: CompiledGraph,
    pool: ModelPool,
    config: PipelineConfig,
    prompt_context: dict[str, Any] | None = None,
    test_suite: str = "",
) -> StepResult:
    """One iteration of the paper's core loop.
    
    1. H(trajectory) → adapter
    2. Load adapter
    3. graph.ainvoke(state) → output
    4. Unload adapter
    5. Return output + feedback
    """
```

This replaces the ~50-line boilerplate pattern duplicated across every phase. Every phase becomes a caller that:
- Renders its trajectory template
- Calls `reasoning_step`
- Interprets the result (parse subtasks, score quality, etc.)

### 2. Unify retry into one mechanism

Replace the five overlapping retry mechanisms with a single `run_phase` loop:

```python
async def run_phase(
    phase: str,
    *,
    render_trajectory: Callable[..., str],
    score_result: Callable[[StepResult], float],
    parse_result: Callable[[str], T],
    max_iterations: int,
    early_stop: Callable[[float, T], bool],
    # ... common params
) -> PhaseResult[T]:
    """Generic phase loop: render → step → score → maybe retry."""
    best_result = None
    best_score = -1.0
    
    for iteration in range(max_iterations):
        trajectory = render_trajectory(prior_output=best_result)
        step = await reasoning_step(trajectory, ...)
        result = parse_result(step.output)
        score = score_result(step)
        
        if score > best_score:
            best_score = score
            best_result = result
        
        if early_stop(score, result):
            break
    
    return PhaseResult(result=best_result, score=best_score, iterations=iteration + 1)
```

Each phase provides its own `render_trajectory`, `score_result`, `parse_result`, and `early_stop` callables. No more 200-line inline phase blocks.

### 3. Continuation and reasoning loop as step modifiers, not separate paths

Instead of `_run_continuation_loop` and `run_reasoning_loop` being separate functions called from specific phase code:

- **Continuation** becomes part of `reasoning_step` — if `finish_reason == "length"`, the step internally handles concatenation. The caller doesn't need to know.
- **Reasoning loop escalation** becomes a `run_phase` option — if accumulated context exceeds budget, `run_phase` switches internally from simple retry to adapter-compressed reasoning. The per-phase code doesn't need to manage this.

### 4. Phase definitions become declarative

Instead of 300+ lines of inline code per phase, each phase becomes a small struct:

```python
PHASES = {
    "decompose": PhaseDefinition(
        template="decompose",
        system_prompt=DECOMPOSE_SYSTEM_PROMPT,
        score_fn=score_decompose,
        parse_fn=parse_subtask_list,
        early_stop_fn=lambda score, _: score >= 0.6,
        executes_code=False,
    ),
    "plan": PhaseDefinition(
        template="plan",
        system_prompt=PLAN_SYSTEM_PROMPT,
        score_fn=score_plan,
        parse_fn=parse_plan,
        early_stop_fn=lambda score, _: score >= 1.0,
        executes_code=False,
        parallel=True,  # runs per-subtask in parallel
    ),
    "code": PhaseDefinition(
        template="code",
        retry_template="code_retry",
        system_prompt=CODE_SYSTEM_PROMPT,
        score_fn=score_code,
        parse_fn=lambda x: x,  # code is the output
        early_stop_fn=lambda score, _: score >= 1.0,
        executes_code=True,
        parallel=True,
        dag_ordered=True,
    ),
    # ...
}
```

### 5. Pipeline orchestrator becomes thin

The top-level `run_phased_pipeline` shrinks to ~100 lines:

```python
async def run_phased_pipeline(project_prompt, *, config, pool, test_suite=""):
    # Phase 1: Decompose
    subtasks = await run_phase("decompose", ...)
    
    # Phase 2: Plan (parallel per-subtask)
    plans = await run_parallel_phase("plan", subtasks, ...)
    
    # Phase 3: Code (DAG-ordered, parallel per-layer)
    code_outputs = await run_dag_phase("code", subtasks, plans, ...)
    
    # Phase 4: Integrate (with repair on failure)
    final = await run_phase("integrate", ..., on_failure=repair_and_retry)
    
    return PipelineResult(...)
```

## What to keep

- **DAG-ordered code execution** with blackboard — genuine value for multi-subtask dependency handling.
- **Two-step diagnose→repair** — solid pattern for error recovery.
- **Evolution sweeps** — if PRODUCT.md confirms this is load-bearing (currently stubbed, can't determine).
- **MLflow tracing** — research instrumentation, but should be a decorator/context manager, not inline in every phase.
- **Adapter registry** — needed for evolution and lineage tracking.

## What to remove or consolidate

| Current | Proposed |
|---------|----------|
| 5 inline phase blocks (~300 lines each) | `run_phase()` + `PhaseDefinition` structs |
| `_run_continuation_loop` (separate function) | Built into `reasoning_step` |
| `run_reasoning_loop` (separate escalation path) | Built into `run_phase` as automatic escalation |
| Per-phase adapter naming (`phase1-decompose-v0`, `phase3-code-*-v2`, etc.) | `{session}-{phase}-{subtask}-{iteration}` universal scheme |
| Per-phase scoring rubrics (5 different inline blocks) | `score_fn` callables on `PhaseDefinition` |
| Duplicated MLflow metrics/tags in every phase | `@mlflow_phase` decorator on `run_phase` |
| `_load_adapter` + `_eager_unload` + `_cleanup_phase_adapters` | Managed by `reasoning_step` lifecycle |
| `run_project` legacy wrapper | Remove — one entry point |

## Estimated impact

- `rune_runner.py`: **2699 → ~800 lines** (70% reduction)
- New `scripts/reasoning_step.py`: ~150 lines (the core loop)
- New `scripts/phase_definitions.py`: ~200 lines (phase configs + scoring)
- Total pipeline code: ~1150 lines vs current ~2700

## Risks and open questions

1. **PRODUCT.md is entirely stubbed.** The `do-not-break:` section would tell us whether evolution sweeps, adapter registry, and MLflow are mandatory infrastructure or removable research scaffolding. This must be resolved before implementing.

2. **Evolution sweeps interleave with phase retries.** If evolution is load-bearing, `run_phase` needs to call `evolution_sweep` between iterations. If it's research instrumentation, it can be a post-phase hook.

3. **HPO integration.** `run_benchmark_hpo.py` calls `run_phased_pipeline` and inspects its return dict. The simplified return type must be backward-compatible or HPO must be updated in tandem.

4. **Reasoning loop state.** `ReasoningLoopState` (in `reasoning_loop.py`) has 27 fields including TIES/DARE merge state. Integrating this into `reasoning_step` means either exposing that complexity or making the reasoning loop a black-box escalation that `run_phase` delegates to wholesale.

## Recommended next steps

1. **Fill in PRODUCT.md** — specifically `do-not-break:` and `out-of-scope:` sections. This determines what's removable.
2. **Prototype `reasoning_step`** — extract the adapter-generate-feedback pattern into one function. Validate it works for decompose (simplest phase) and code (most complex).
3. **Migrate one phase at a time** — decompose first (no parallelism, simple scoring), then plan, code, integrate. Each migration is independently testable.
4. **Replace in-place** — migrate each phase by directly replacing its inline block with a `run_phase()` call. Each phase migration is a self-contained commit; run the test suite after each to verify.
