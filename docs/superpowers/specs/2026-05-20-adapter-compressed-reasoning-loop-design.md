# Adapter-Compressed Reasoning Loop

**Date:** 2026-05-20
**Status:** Draft
**Branch:** (to be created)

## Problem

When a pipeline phase hits `finish_reason == 'length'`, the current continuation mechanism (`code_continue` template) concatenates prior output and prompts "continue from here." Context grows linearly with each continuation. After 2-3 continuations the accumulated text approaches the model's context window limit, at which point the model either loses coherence or cannot continue at all.

The model can only reason as long as the context window allows. For genuinely complex tasks requiring extended reasoning chains, this is a hard ceiling.

## Solution

Split long reasoning into many short turns. After each turn, compress the accumulated reasoning trace into LoRA adapter weights via the hypernetwork (H()), reload the adapted model, and continue from a fresh short prompt with a sliding window of recent output. The context window is used only for the current subproblem; the long reasoning history lives in the adapter as parametric episodic memory.

This is directly analogous to Claude Code's context compaction: when the conversation fills up, compress prior context into a summary and continue from a shorter window. Here, "summary" is replaced by "adapter weights" — the compression happens in weight space rather than token space.

**Practical effect:** The agent can extend reasoning across arbitrarily many turns with roughly constant per-step cost, because each step re-encodes prior thought into the adapter instead of replaying all prior tokens.

## Design Decisions

### Escalation model, not replacement

The adapter-compressed loop does NOT replace the existing continuation mechanism. Continuation is the fast path for the common case (1-3 truncations where accumulated context fits in the window). The adapter loop is an escalation that triggers when accumulated context exceeds a configurable budget (default: 75% of model context window).

**Rationale:** Continuation has zero overhead (no hypernetwork call, no adapter reload) and is lossless (model sees exact prior output). The adapter loop trades overhead and some information loss for unbounded reasoning length. Use the cheap path when it works; escalate when it doesn't.

### Sliding window + adapter

Each turn's prompt includes:
- The last K tokens of raw output (sliding window) — exact token-level recall of recent work
- Phase-specific instructions (via Jinja2 template)
- The adapter carries everything older than the sliding window

**Rationale:** The perceiver encodes patterns and direction, not verbatim content. For code phases where the model needs exact recall of variable names, function signatures, and line-by-line logic, the sliding window provides that. The adapter provides broader context about approach, constraints discovered, and trajectory.

### Adaptive capacity control

Different phases need different adapter capacity:

- **Code phases** (code, code_repair, integrate): High capacity. The adapter must carry prior code structure. Uses multi-pass perceiver + TIES merge with boosted scaling (1.5x base).
- **Text phases** (decompose, plan, diagnose): Standard capacity. The adapter carries trajectory/decisions, not verbatim code. Single pass at base scaling.
- **Long trajectories** (any phase): When trajectory exceeds the multipass threshold, multi-pass is used regardless of phase.

**Rank control mechanisms (v1):**
1. **Scaling factor** — adjust adapter influence strength (already exists at 0.16 base). Quick adjustment, zero implementation cost.
2. **Multi-pass perceiver + TIES/DARE merge** — segment long trajectories into overlapping windows, run perceiver on each, merge adapters. Information capacity scales linearly with passes. Reuses existing `model_training.merging` infrastructure.

SVD-based rank truncation is deferred to v2 — scaling handles the "less capacity" direction, and multi-pass handles "more capacity."

### Phase-configurable turn cycles

Each phase configures whether turns include execution/testing:

| Phase | Executes per turn | Rationale |
|-------|-------------------|-----------|
| code | Yes | Needs test feedback to guide next turn |
| code_repair | Yes | Same — error feedback essential |
| integrate | Yes | Tests validate integration correctness |
| decompose | No | Text output, nothing to execute |
| plan | No | Text output |
| diagnose | No | Text output |

### Termination

The loop stops on whichever comes first:
1. `finish_reason == 'stop'` — model naturally completed
2. `turn_count >= max_turns` — configurable cap (default 20, env: `RUNE_MAX_REASONING_TURNS`)

## Architecture

### New LangGraph: Reasoning Loop

```
START → reason → [phase_executes?]
  → yes: execute → reflect → compress_to_adapter → should_continue
  → no:  compress_to_adapter → should_continue

should_continue:
  → turn_count < max_turns AND finish_reason == 'length': → reason
  → otherwise: → END
```

**Nodes:**
- `reason`: Generate with sliding window prompt + current adapter. Equivalent to `generate_node` but constructs the prompt from sliding window + phase template.
- `execute`: Run generated code in sandbox (reuses existing `execute_node`).
- `reflect`: Evaluate results (reuses existing `reflect_node`).
- `compress_to_adapter`: Encode accumulated trajectory via H(), optionally multi-pass + merge, load new adapter, update state.
- `should_continue`: Check termination conditions.

### State

```python
class ReasoningLoopState(TypedDict):
    # From RuneState
    task_description: str
    phase: str
    adapter_ids: list[str]
    session_id: str
    generated_code: str
    stdout: str
    stderr: str
    exit_code: int
    tests_passed: bool
    test_count: int
    tests_ran: bool
    finish_reason: str | None
    outcome: str | None
    prompt_context: dict[str, Any] | None

    # Reasoning loop specific
    turn_count: int
    max_turns: int
    accumulated_trajectory: str
    sliding_window: str
    sliding_window_size: int
    current_adapter_path: str | None
    scaling_factor: float
    use_multipass: bool
    multipass_window_size: int
    phase_executes: bool
    turn_history: list[dict[str, Any]]
```

### Adapter Strategy Resolution

```python
def resolve_adapter_strategy(
    phase: str,
    trajectory_tokens: int,
    multipass_threshold: int,
    base_scaling: float,
    code_scaling_boost: float = 1.5,
) -> AdapterStrategy:
    is_code_phase = phase in {'code', 'code_repair', 'integrate'}
    needs_multipass = trajectory_tokens > multipass_threshold

    if is_code_phase and needs_multipass:
        return MultiPass(
            scaling=base_scaling * code_scaling_boost,
            merge_method='ties',
        )
    elif needs_multipass:
        return MultiPass(
            scaling=base_scaling,
            merge_method='dare',
        )
    else:
        return SinglePass(scaling=base_scaling)
```

TIES for code phases (preserves high-magnitude weights important for code structure). DARE for trajectory content (random dropout works well for broader context).

### Compress-to-Adapter Pipeline

1. **Build trajectory text** — concatenate all prior turn outputs into `accumulated_trajectory`
2. **Resolve strategy** — `resolve_adapter_strategy(phase, trajectory_tokens, ...)`
3. **Generate adapter(s)**:
   - Single pass: one `run_hypernetwork()` call with full trajectory
   - Multi-pass: N calls on overlapping 512-token windows, then TIES/DARE merge via `model_training.merging`
4. **Load adapter** — unload previous turn's adapter, load new one via `_load_adapter()`
5. **Build next prompt** — sliding window (last K tokens) + phase template via `reasoning_continue.j2`

### Escalation Integration

In `rune_runner.py`, after each `run_iteration()` returns:

```python
if finish_reason == 'length':
    accumulated_tokens = len(tokenize(accumulated_output))
    if accumulated_tokens < context_budget:
        # Fast path: existing continuation
        ...
    else:
        # Escalation: adapter-compressed reasoning loop
        result = await run_reasoning_loop(
            graph=reasoning_graph,
            initial_output=accumulated_output,
            task_description=project_prompt,
            phase=current_phase,
            phase_executes=(current_phase in {'code', 'code_repair', 'integrate'}),
            adapter_config=reasoning_loop_config,
            max_turns=max_reasoning_turns,
            pool=pool,
        )
```

The `context_budget` is `model_max_context * context_budget_ratio` where the model's max context length is read from HuggingFace config at pool creation.

All token counting (context budget, sliding window, multipass thresholds) uses the base model's tokenizer via the ModelPool. No rough word-count estimates.

**Return value of `run_reasoning_loop()`:**
- Code phases (code, code_repair, integrate): returns all turns' output concatenated — the full generated code.
- Text phases (decompose, plan, diagnose): returns the last turn's output only — the final reasoning result.

## Configuration

New `ReasoningLoopConfig` added to `PipelineConfig`:

| Field | Default | Env var | Description |
|-------|---------|---------|-------------|
| `max_turns` | 20 | `RUNE_MAX_REASONING_TURNS` | Maximum adapter-compressed turns |
| `context_budget_ratio` | 0.75 | `RUNE_CONTEXT_BUDGET_RATIO` | Fraction of model context window before escalation |
| `sliding_window_tokens` | 1024 | `RUNE_SLIDING_WINDOW_TOKENS` | Default sliding window size (overridden per phase) |
| `multipass_threshold` | 1024 | `RUNE_MULTIPASS_THRESHOLD` | Trajectory tokens before triggering multi-pass |
| `multipass_window_size` | 512 | `RUNE_MULTIPASS_WINDOW_SIZE` | Per-pass token window |
| `multipass_overlap` | 128 | `RUNE_MULTIPASS_OVERLAP` | Token overlap between windows |
| `code_scaling_boost` | 1.5 | `RUNE_CODE_SCALING_BOOST` | Scaling multiplier for code phases |
| `default_merge_method` | ties | `RUNE_MERGE_METHOD` | TIES or DARE for multi-pass merge |

Phase-specific sliding window defaults:

| Phase | Sliding window tokens |
|-------|----------------------|
| decompose | 256 |
| plan | 512 |
| code | 1024 |
| code_repair | 1024 |
| integrate | 1024 |
| diagnose | 512 |

## File Changes

### New files

| File | Purpose |
|------|---------|
| `services/rune-agent/src/rune_agent/reasoning_loop.py` | `ReasoningLoopState`, `create_reasoning_loop_graph()`, `reason_node`, `compress_to_adapter_node`, `should_continue_node` |
| `services/rune-agent/src/rune_agent/adapter_strategy.py` | `AdapterStrategy`, `SinglePass`, `MultiPass`, `resolve_adapter_strategy()` |
| `services/rune-agent/tests/test_reasoning_loop.py` | Unit + integration tests for reasoning loop graph and adapter strategy |
| `libs/shared/src/shared/templates/reasoning_continue.j2` | Jinja2 template for reasoning continuation prompts |

### Modified files

| File | Change |
|------|--------|
| `services/rune-agent/src/rune_agent/graph.py` | Export `create_reasoning_loop_graph` |
| `scripts/rune_runner.py` | Add `run_reasoning_loop()`, escalation logic in phase iteration loops |
| `libs/shared/src/shared/pipeline_config.py` | Add `ReasoningLoopConfig` to `PipelineConfig` |

## Observability

MLflow tracing:
- Parent span: `reasoning_loop/{phase}` for each reasoning loop invocation
- Child spans per turn: turn number, trajectory length, adapter strategy, scaling factor, finish_reason
- Metrics: `reasoning_loop/turns`, `reasoning_loop/total_trajectory_tokens`, `reasoning_loop/escalation_trigger_tokens`
- Tags: `reasoning_loop.phase`, `reasoning_loop.strategy`

## Testing Strategy

1. **Unit tests** (no GPU):
   - `resolve_adapter_strategy()` — deterministic strategy selection for all phase/token count combinations
   - `should_continue` node — termination logic (max turns, stop signal)
   - Sliding window construction — correct token truncation

2. **Integration tests** (mock hypernetwork):
   - Reasoning loop graph with mock H() returning dummy adapter — verify turn progression, state updates, termination
   - Escalation path in `rune_runner.py` — mock context budget to be very small, verify escalation triggers

3. **End-to-end** (GPU):
   - Deliberately long task requiring >1 continuation — verify escalation triggers and loop produces coherent output
   - Compare output quality: continuation-only vs. adapter-compressed for the same long task

## Deferred to v2

- **SVD-based rank truncation** — downward rank control via singular value decomposition. Scaling handles this approximately for now.
- **Convergence detection** — stop when output between turns shows diminishing change. Requires a similarity metric.
- **Adapter caching** — cache adapters across similar trajectories to skip redundant H() calls.
- **Async multi-pass** — run perceiver passes in parallel (currently sequential).
