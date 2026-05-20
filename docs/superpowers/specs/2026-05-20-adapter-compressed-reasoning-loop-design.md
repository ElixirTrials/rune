# Adapter-Compressed Reasoning Loop

**Date:** 2026-05-20
**Status:** Draft (rev 2 — incorporates paper-grounded review)
**Branch:** feat/adapter-compressed-reasoning-loop

## Problem

When a pipeline phase hits `finish_reason == 'length'`, the current continuation mechanism (`code_continue` template) concatenates prior output and prompts "continue from here." Context grows linearly with each continuation. After 2-3 continuations the accumulated text approaches the model's context window limit, at which point the model either loses coherence or cannot continue at all.

## Solution

Split long reasoning into many short turns. After each turn, compress the accumulated reasoning trace into LoRA adapter weights via the hypernetwork (H()), reload the adapted model, and continue from a fresh short prompt with a sliding window of recent output.

This is analogous to Claude Code's context compaction: when the conversation fills up, compress prior context and continue from a shorter window. Here, "summary" is replaced by "adapter weights" — compression happens in weight space rather than token space.

### What this is and what it is not

Adapter compression is a **lossy memory mechanism**, not a guaranteed replacement for long-context token replay. The paper explicitly frames the trade-off between token-memory and weight-memory as still open. The practical bet is that for reasoning chains long enough to exhaust the context window, lossy parametric memory is better than no memory at all (which is what happens when context is simply truncated). This spec implements the mechanism and the instrumentation to measure whether that bet pays off. Several modeling decisions below are **experimental knobs** that must be validated through ablation, not treated as settled defaults.

## Design Decisions

### Escalation model, not replacement

The adapter-compressed loop does NOT replace the existing continuation mechanism. Continuation is the fast path for the common case (1-3 truncations where accumulated context fits in the window). The adapter loop is an escalation.

**Rationale:** Continuation has zero overhead and is lossless. The adapter loop trades overhead and information loss for unbounded reasoning length. Use the cheap path when it works; escalate when it doesn't.

**Escalation trigger (v1):** accumulated context exceeds a configurable budget (default: 75% of model context window). This is a simple heuristic. Other signals — semantic drift, repeated truncation patterns, exact-recall failures — may be better criteria; the implementation should log these signals from the start so that richer escalation policies can be built from data.

### Sliding window + adapter

Each turn's prompt includes:
- The last K tokens of raw output (sliding window) — exact token-level recall of recent work
- Phase-specific instructions (via Jinja2 template)
- The adapter carries everything older than the sliding window

The perceiver encodes patterns and direction, not verbatim content. The sliding window provides exact recall where needed; the adapter provides broader context.

### Fallback: expand sliding window when exact recall matters

**Not every failure should be met with more compression.** When the adapter is failing to carry enough context (detected via adapter health monitoring — see Termination), the system should be able to **expand the sliding window** rather than compress more aggressively. This means temporarily increasing K (raw tokens in prompt) at the cost of fewer remaining tokens for generation, trading generation budget for recall fidelity.

Implementation: `should_continue` node checks adapter health. On collapse detection, before halting, it tries one recovery turn with `sliding_window_size *= 2` (capped at 80% of context window). If recovery fails, halt and return best output so far.

### Structured trajectory representation

The trajectory fed to H() is NOT raw text concatenation. It is a structured object matching the paper's memory format: `(state, action, feedback)` per turn.

```python
@dataclass
class TurnRecord:
    turn: int
    state: str       # task description + sliding window at turn start
    action: str      # model's generated output this turn
    feedback: str    # execution results (stdout, stderr, test pass/fail)
                     # empty string for non-executing phases
    diagnosis: str   # reflect node's assessment (if phase_executes)
```

The trajectory text passed to H() is rendered from these records via a Jinja2 template (`trajectory_compress.j2`) that produces structured prose, not raw concatenation. Execution feedback and diagnostic reflection are first-class fields, not incidental state.

### Adaptive capacity control (experimental)

**All values below are starting hypotheses, not validated defaults.** They must be tested via ablation before being treated as production settings.

**Rank control mechanisms (v1):**
1. **Scaling factor** — adjust adapter influence strength (base: 0.16 from HPO). The paper's best finding is that useful trajectory-conditioned scaling needs to stay surprisingly low. Any boost above base is an experimental knob.
2. **Multi-pass perceiver + merge** — segment long trajectories into overlapping windows, run perceiver on each, merge adapters. **This is an ablation target, not a proven default.** The paper describes multi-adapter composition as an area for investigation, not an established technique. Multi-pass may improve coverage but also introduces merge-induced interference. The claim that "information capacity scales linearly with passes" is too strong given fixed low-rank adapters and merge interference risk. The actual relationship is empirical and must be measured.

**Strategy resolution:**

```python
def resolve_adapter_strategy(
    phase: str,
    trajectory_tokens: int,
    multipass_threshold: int,
    base_scaling: float,
    code_scaling_boost: float = 1.2,  # EXPERIMENTAL — paper HPO says 0.16 base works;
                                       # any boost is unvalidated
) -> AdapterStrategy:
    is_code_phase = phase in {'code', 'code_repair', 'integrate'}
    needs_multipass = trajectory_tokens > multipass_threshold

    if needs_multipass:
        # EXPERIMENTAL: multi-pass is an ablation target, not a proven default.
        # Default merge method is configurable; TIES vs DARE preference by phase
        # is a hypothesis to test, not an established result.
        return MultiPass(
            scaling=base_scaling * (code_scaling_boost if is_code_phase else 1.0),
            merge_method=default_merge_method,  # configurable, default 'ties'
        )
    else:
        return SinglePass(scaling=base_scaling)
```

**What changed from rev 1:**
- `code_scaling_boost` reduced from 1.5 to 1.2 as a more conservative starting point (still unvalidated — the paper's HPO found 0.16 base is optimal and that interpretation is model-family-specific)
- TIES-for-code / DARE-for-text split removed as a hardcoded decision; instead, merge method is a single configurable default. The phase-specific split is a hypothesis for the ablation plan.
- Multi-pass is gated behind a flag (`enable_multipass`, default: true) so it can be disabled in A/B tests

### Phase-configurable turn cycles

Each phase configures whether turns include execution/testing:

| Phase | Executes per turn | Rationale |
|-------|-------------------|-----------|
| code | Yes | Needs test feedback to guide next turn |
| code_repair | Yes | Error feedback essential |
| integrate | Yes | Tests validate integration correctness |
| decompose | No | Text output, nothing to execute |
| plan | No | Text output |
| diagnose | No | Text output |

### Termination and adapter health monitoring

The loop stops on whichever comes first:
1. `finish_reason == 'stop'` — model naturally completed
2. `turn_count >= max_turns` — configurable cap (default 20)
3. **Adapter collapse detected** — the paper documents that adapter influence can collapse into repetitive or near-constant behavior after several compression steps. The following health signals are computed per turn and used as halt/fallback triggers:

| Signal | Computation | Threshold | Action |
|--------|------------|-----------|--------|
| **Inter-adapter cosine similarity** | Cosine sim between flattened LoRA-A weights of turn N and turn N-1 | > 0.95 for 2 consecutive turns | Attempt recovery (expand sliding window). If recovery fails, halt. |
| **Adapter norm ratio** | L2 norm of current adapter / L2 norm of first adapter | < 0.1 (collapse) or > 10.0 (explosion) | Halt immediately |
| **Output repetition** | Fraction of output n-grams (n=4) that appear in previous turn's output | > 0.8 | Attempt recovery. If recovery fails, halt. |

These signals are logged as MLflow metrics every turn regardless of whether they trigger a halt, providing data for tuning thresholds.

**Recovery before halt:** On collapse detection, the system tries one recovery turn with an expanded sliding window (2x default, capped at 80% of context window) and reset scaling to base. If recovery produces a non-collapsed turn, continue. If not, halt and return best output so far (the turn with highest test score for executing phases, or lowest repetition for non-executing phases).

## Architecture

### New LangGraph: Reasoning Loop

```
START → reason → [phase_executes?]
  → yes: execute → reflect → compress_to_adapter → check_health → should_continue
  → no:  compress_to_adapter → check_health → should_continue

should_continue:
  → healthy AND turn_count < max_turns AND finish_reason == 'length': → reason
  → collapse detected AND not yet attempted recovery: → recover → reason
  → otherwise: → END
```

**Nodes:**
- `reason`: Generate with sliding window prompt + current adapter. Constructs prompt from sliding window + `reasoning_continue.j2` template.
- `execute`: Run generated code in sandbox (reuses existing `execute_node`).
- `reflect`: Evaluate results (reuses existing `reflect_node`).
- `compress_to_adapter`: Build structured trajectory from `TurnRecord` list, render via `trajectory_compress.j2`, encode via H(), optionally multi-pass + merge, load new adapter.
- `check_health`: Compute adapter health signals (cosine sim, norm ratio, output repetition). Update state.
- `should_continue`: Check termination conditions including health.
- `recover`: Expand sliding window, reset scaling, set recovery flag.

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
    turn_records: list[dict[str, Any]]  # list of TurnRecord dicts
    sliding_window: str
    sliding_window_size: int            # current K (may expand during recovery)
    base_sliding_window_size: int       # original K (for reset after recovery)
    current_adapter_path: str | None
    previous_adapter_weights: Any | None  # flattened LoRA-A for cosine sim
    first_adapter_norm: float | None      # baseline for norm ratio
    scaling_factor: float
    enable_multipass: bool
    multipass_window_size: int
    phase_executes: bool
    turn_history: list[dict[str, Any]]    # per-turn metadata for observability

    # Health monitoring
    adapter_cosine_sim: float             # similarity to previous turn's adapter
    adapter_norm_ratio: float             # current norm / first norm
    output_repetition: float              # n-gram overlap with previous turn
    consecutive_high_similarity: int      # counter for cosine sim threshold
    recovery_attempted: bool              # only one recovery attempt allowed
```

### Compress-to-Adapter Pipeline

1. **Build structured trajectory** — render `turn_records` list via `trajectory_compress.j2` into structured text with (state, action, feedback, diagnosis) per turn
2. **Resolve strategy** — `resolve_adapter_strategy(phase, trajectory_tokens, ...)`
3. **Generate adapter(s)**:
   - Single pass: one `run_hypernetwork()` call with rendered trajectory
   - Multi-pass (experimental, behind `enable_multipass` flag): N calls on overlapping windows, then merge via configured method. Fall back to single-pass on the full trajectory if multi-pass produces a collapsed adapter.
4. **Retain previous adapter weights** — store flattened LoRA-A for next turn's cosine similarity check
5. **Load adapter** — unload previous turn's adapter, load new one via `_load_adapter()`
6. **Build next prompt** — sliding window (last K tokens of raw output) + phase template via `reasoning_continue.j2`

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

All token counting uses the base model's tokenizer via the ModelPool.

### Return value semantics

**Code phases** (code, code_repair, integrate): The canonical output is the **last turn's output only**. Each turn is prompted to produce a complete solution (the sliding window provides context for what was written before). If the last turn fails tests but an earlier turn passed, return the best-passing turn's output. The downstream consumer (integrate phase, sandbox) receives a single coherent code block, not a concatenation of partial outputs.

**Text phases** (decompose, plan, diagnose): Last turn's output only.

**Metadata returned alongside output:** turn count, adapter IDs generated, health signals per turn, whether recovery was triggered, final adapter path.

## Configuration

New `ReasoningLoopConfig` added to `PipelineConfig`:

| Field | Default | Env var | Description | Status |
|-------|---------|---------|-------------|--------|
| `max_turns` | 20 | `RUNE_MAX_REASONING_TURNS` | Maximum adapter-compressed turns | Engineering decision |
| `context_budget_ratio` | 0.75 | `RUNE_CONTEXT_BUDGET_RATIO` | Fraction of model context window before escalation | Engineering decision; richer escalation criteria deferred |
| `sliding_window_tokens` | 1024 | `RUNE_SLIDING_WINDOW_TOKENS` | Default sliding window size | Starting value — phase-specific tuning via ablation |
| `multipass_threshold` | 1024 | `RUNE_MULTIPASS_THRESHOLD` | Trajectory tokens before triggering multi-pass | Experimental |
| `multipass_window_size` | 512 | `RUNE_MULTIPASS_WINDOW_SIZE` | Per-pass token window | Experimental |
| `multipass_overlap` | 128 | `RUNE_MULTIPASS_OVERLAP` | Token overlap between windows | Experimental |
| `enable_multipass` | true | `RUNE_ENABLE_MULTIPASS` | Enable multi-pass perceiver (disable for ablation) | Experimental |
| `code_scaling_boost` | 1.2 | `RUNE_CODE_SCALING_BOOST` | Scaling multiplier for code phases | Experimental — paper HPO found 0.16 base optimal |
| `default_merge_method` | ties | `RUNE_MERGE_METHOD` | TIES or DARE for multi-pass merge | Experimental — no evidence one is better per phase |
| `collapse_cosine_threshold` | 0.95 | `RUNE_COLLAPSE_COSINE_THRESHOLD` | Inter-adapter similarity threshold for collapse detection | Experimental |
| `collapse_norm_min` | 0.1 | `RUNE_COLLAPSE_NORM_MIN` | Min adapter norm ratio before collapse halt | Experimental |
| `collapse_norm_max` | 10.0 | `RUNE_COLLAPSE_NORM_MAX` | Max adapter norm ratio before explosion halt | Experimental |
| `collapse_repetition_threshold` | 0.8 | `RUNE_COLLAPSE_REPETITION_THRESHOLD` | N-gram overlap threshold for output repetition | Experimental |

Phase-specific sliding window defaults (all overridable via `RUNE_SLIDING_WINDOW_{PHASE}` env vars):

| Phase | Sliding window tokens | Rationale |
|-------|----------------------|-----------|
| decompose | 256 | Short text output, low locality requirement |
| plan | 512 | Architecture plans reference recent decisions |
| code | 1024 | Active code needs exact recent recall |
| code_repair | 1536 | Repair needs both the broken code and error context |
| integrate | 2048 | Integration references multiple subtask outputs |
| diagnose | 512 | Diagnostic text, moderate locality |

**Note:** These per-phase values are starting hypotheses. The phases differ materially in how much exact recent text they need. The ablation plan includes sliding window size sweeps per phase.

## File Changes

### New files

| File | Purpose |
|------|---------|
| `services/rune-agent/src/rune_agent/reasoning_loop.py` | `ReasoningLoopState`, `create_reasoning_loop_graph()`, all node functions |
| `services/rune-agent/src/rune_agent/adapter_strategy.py` | `AdapterStrategy`, `SinglePass`, `MultiPass`, `resolve_adapter_strategy()` |
| `services/rune-agent/src/rune_agent/adapter_health.py` | `compute_cosine_similarity()`, `compute_norm_ratio()`, `compute_output_repetition()`, `check_health()` |
| `services/rune-agent/tests/test_reasoning_loop.py` | Unit + integration tests |
| `services/rune-agent/tests/test_adapter_health.py` | Health monitoring unit tests |
| `libs/shared/src/shared/templates/reasoning_continue.j2` | Prompt template for reasoning continuation |
| `libs/shared/src/shared/templates/trajectory_compress.j2` | Template for rendering structured trajectory for H() input |

### Modified files

| File | Change |
|------|--------|
| `services/rune-agent/src/rune_agent/graph.py` | Export `create_reasoning_loop_graph` |
| `scripts/rune_runner.py` | Add `run_reasoning_loop()`, escalation logic in phase iteration loops |
| `libs/shared/src/shared/pipeline_config.py` | Add `ReasoningLoopConfig` to `PipelineConfig` |

## Observability

MLflow tracing per reasoning loop invocation:

**Spans:**
- Parent span: `reasoning_loop/{phase}`
- Child span per turn with all fields below

**Metrics (logged every turn, not just on failure):**

| Metric | Description | Why it matters |
|--------|-------------|---------------|
| `reasoning_loop/turn` | Current turn number | Basic progress |
| `reasoning_loop/trajectory_tokens` | Total trajectory length in tokens | Context pressure |
| `reasoning_loop/adapter_cosine_sim` | Cosine similarity to previous adapter | Collapse detection — the paper's key diagnostic |
| `reasoning_loop/adapter_norm` | L2 norm of current adapter | Collapse (→0) or explosion (→∞) detection |
| `reasoning_loop/adapter_norm_ratio` | Current norm / first adapter norm | Drift from baseline |
| `reasoning_loop/output_repetition` | N-gram overlap with previous turn | Behavioral collapse |
| `reasoning_loop/merge_count` | Number of adapters merged this turn | Multi-pass cost tracking |
| `reasoning_loop/hypernetwork_latency_ms` | Time for H() call(s) | Performance profiling |
| `reasoning_loop/adapter_load_latency_ms` | Time for adapter load/unload | Performance profiling |
| `reasoning_loop/sliding_window_tokens` | Actual sliding window size this turn | Tracks recovery expansions |
| `reasoning_loop/scaling_factor` | Actual scaling used this turn | Strategy tracking |
| `reasoning_loop/strategy` | single_pass or multi_pass | Strategy tracking |
| `reasoning_loop/recovery_triggered` | Boolean | Collapse recovery tracking |

**Tags:** `reasoning_loop.phase`, `reasoning_loop.escalation_trigger_tokens`, `reasoning_loop.final_turn_count`

## Testing Strategy

### Functional tests

1. **Unit tests** (no GPU):
   - `resolve_adapter_strategy()` — strategy selection for all phase/token count combinations
   - `should_continue` node — termination logic (max turns, stop signal, all three collapse signals)
   - `check_health` — cosine sim, norm ratio, output repetition computation
   - Sliding window construction — correct token truncation via tokenizer
   - Structured trajectory rendering — `TurnRecord` → `trajectory_compress.j2` → structured text
   - Recovery logic — sliding window expansion, scaling reset, single-attempt guard

2. **Integration tests** (mock hypernetwork):
   - Reasoning loop graph with mock H() returning dummy adapter — verify turn progression, state updates, termination
   - Escalation path in `rune_runner.py` — mock context budget to be very small, verify escalation triggers
   - Collapse detection + recovery — mock H() to return near-identical adapters, verify collapse halt
   - Return value semantics — verify code phases return best-passing turn, text phases return last turn

3. **End-to-end** (GPU):
   - Deliberately long task requiring >1 continuation — verify escalation triggers and loop produces coherent output

### Ablation plan

These ablations are required before any experimental knob becomes a hardened default. Each ablation holds all other knobs at their defaults and sweeps one variable.

| Ablation | Variable | Values | Metric | Purpose |
|----------|----------|--------|--------|---------|
| A1 | Multi-pass vs. single-pass | `enable_multipass` true/false | Pass@1, adapter cosine diversity | Does multi-pass actually help or does merge interference hurt? |
| A2 | Merge method | TIES, DARE | Pass@1, adapter norm stability | Is one merge method better? Is it phase-dependent? |
| A3 | Code scaling boost | 1.0, 1.2, 1.5, 2.0 | Pass@1, output repetition rate | What boost (if any) helps code recall without causing degeneration? |
| A4 | Sliding window size | 256, 512, 1024, 2048 per phase | Pass@1, exact-recall accuracy | How much raw context does each phase actually need? |
| A5 | Continuation vs. adapter-compressed | Same long task, both paths | Pass@1, coherence, total time | Does the adapter loop beat naive continuation for long tasks? |
| A6 | Structured vs. raw trajectory | `trajectory_compress.j2` vs. raw concat | Pass@1, adapter cosine diversity | Does structured trajectory improve adapter quality? |
| A7 | Collapse threshold sensitivity | cosine: 0.90/0.95/0.99, repetition: 0.6/0.8/0.9 | False positive rate, missed collapses | Calibrate health thresholds |

## Deferred to v2

- **SVD-based rank truncation** — downward rank control via singular value decomposition.
- **Adapter caching** — cache adapters across similar trajectories to skip redundant H() calls.
- **Async multi-pass** — run perceiver passes in parallel (currently sequential).
- **Richer escalation criteria** — semantic drift detection, exact-recall scoring, repeated truncation pattern analysis. V1 logs the signals; v2 uses them.
- **Convergence detection** — stop when output between turns shows diminishing change. Requires a similarity metric distinct from collapse detection.

## Open questions

These are acknowledged unknowns that will be resolved by the ablation plan:

1. Does multi-pass perceiver actually increase effective capacity, or does merge interference negate the coverage gain?
2. What is the actual relationship between scaling boost and code-recall fidelity?
3. Is the phase-specific merge method split (TIES for code, DARE for text) real, or is one method uniformly better?
4. At what turn count does adapter collapse typically onset for this model family?
5. Is the 75% context budget ratio the right escalation trigger, or is a richer signal needed?
