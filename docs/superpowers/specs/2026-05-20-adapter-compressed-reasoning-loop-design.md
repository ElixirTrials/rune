# Adapter-Compressed Reasoning Loop

**Date:** 2026-05-20
**Status:** Draft (rev 3 — reframed around code-state compression)
**Branch:** feat/adapter-compressed-reasoning-loop

## Problem

When a pipeline phase hits `finish_reason == 'length'`, the current continuation mechanism (`code_continue` template) concatenates prior output and prompts "continue from here." Context grows linearly with each continuation. After 2-3 continuations the accumulated text approaches the model's context window limit, at which point the model either loses coherence or cannot continue at all.

## Solution

Split long reasoning into many short turns. After each turn, compress the current **code state** into LoRA adapter weights via the hypernetwork (H()), reload the adapted model, and continue from a fresh short prompt with a sliding window of recent output.

### Framing: code-state encoding, not generic trajectory

The code continuation problem is closer to the **recall problem that Sakana addressed in doc2lora** than to generic trajectory encoding. The perceiver was designed to encode a specific document into adapter weights for faithful recall. In continuation, the "document" is the accumulated code artifact — file contents, function signatures, import structure, test state — not an abstract reasoning trace.

This reframing matters because:
- The evaluation criteria become concrete: can the model recall identifiers, preserve API signatures, maintain import structure?
- The compression input is a structured artifact, not prose
- The composition strategy for long code should follow code structure (functions, classes, modules), not arbitrary token windows

### What this is and what it is not

Adapter compression is a **lossy memory mechanism**. The paper explicitly frames the trade-off between token-memory and weight-memory as still open. The bet is that for code contexts long enough to exhaust the window, lossy parametric memory is better than no memory (truncation). This spec implements the mechanism and the instrumentation to measure whether that bet pays off. Several modeling decisions below are **experimental knobs** that must be validated through ablation.

## Design Decisions

### Escalation model, not replacement

The adapter-compressed loop does NOT replace the existing continuation mechanism. Continuation is the fast path for the common case (1-3 truncations where accumulated context fits in the window). The adapter loop is an escalation.

**Rationale:** Continuation has zero overhead and is lossless. The adapter loop trades overhead and information loss for unbounded reasoning length.

**Escalation trigger (v1):** accumulated context exceeds a configurable budget (default: 75% of model context window). This is a simple heuristic. Other signals — semantic drift, repeated truncation patterns, exact-recall failures — may be better criteria; the implementation logs these from the start so richer escalation policies can be built from data.

### Sliding window + adapter

Each turn's prompt includes:
- The last K tokens of raw output (sliding window) — exact token-level recall of recent work
- Phase-specific instructions (via Jinja2 template)
- The adapter carries everything older than the sliding window

The sliding window provides exact recall where needed; the adapter provides broader context.

### Fallback: expand sliding window when exact recall matters

**Not every failure should be met with more compression.** When the adapter is failing to carry enough context (detected via health monitoring — see Termination), the system can **expand the sliding window** rather than compress more aggressively. This means temporarily increasing K (raw tokens in prompt) at the cost of fewer remaining tokens for generation, trading generation budget for recall fidelity.

Implementation: `should_continue` node checks adapter health. On collapse detection, before halting, it tries one recovery turn with `sliding_window_size *= 2` (capped at 80% of context window). If recovery fails, halt and return best output so far.

### Artifact state as the compression object

The object fed to H() is not raw text concatenation or generic trajectory prose. It is a **structured representation of the current code state**, reflecting what the model needs to recall to continue working. This is the doc2lora analog: the "document" is the code artifact.

```python
@dataclass
class ArtifactState:
    """Structured code state for adapter compression.

    Designed for the recall problem: what does the model need to know
    about the code it has written so far to continue writing correctly?
    """
    # Current code state
    file_contents: str          # full generated code so far (canonical snapshot)
    interface_summary: str      # extracted signatures, classes, exports
                                # (via shared.blackboard.extract_interfaces)
    import_block: str           # all import statements (exact recall critical)

    # Patch history (what changed and why)
    patches: list[PatchRecord]  # ordered list of changes across turns

    # Execution feedback
    test_results: str           # last test run: pass/fail counts, failed test names
    stderr_summary: str         # extracted error summary (via _extract_error_summary)
    tests_passed: bool

    # Unresolved obligations
    todos: list[str]            # functions declared but not implemented,
                                # failing tests not yet addressed,
                                # integration points not yet connected

@dataclass
class PatchRecord:
    """A single code change between turns."""
    turn: int
    description: str            # what changed (from reflect node)
    diff_summary: str           # compact diff (added/removed/modified functions)
```

The artifact state is rendered via `artifact_compress.j2` into structured text for H(). The template emphasizes:
1. **Import block** verbatim (imports are short, exact, and critical for correct continuation)
2. **Interface summary** verbatim (function signatures are the API contract)
3. **Patch history** as structured prose (what changed and why)
4. **Test failures** as structured prose (what's broken and the error)
5. **File contents** as a code skeleton (via `_extract_code_skeleton`) — full bodies only for functions modified in the last patch

For non-code phases (decompose, plan, diagnose), a simpler `TrajectoryState` is used:

```python
@dataclass
class TrajectoryState:
    """Lightweight state for non-code phases."""
    turn: int
    output: str                 # this turn's generated text
    feedback: str               # execution feedback (empty for non-executing phases)
    diagnosis: str              # reflect node's assessment
```

### Chunk-to-rank composition for long code

When code state exceeds the perceiver's single-pass token limit, the spec uses **semantic chunking** rather than arbitrary overlapping token windows.

**Why:** Code has natural structure — functions, classes, modules. Chunking along these boundaries preserves semantic units. A function split across two windows loses coherence in both; a function in its own chunk is encoded whole.

**Implementation:**

```python
def chunk_code_state(artifact: ArtifactState, max_chunk_tokens: int) -> list[CodeChunk]:
    """Split artifact into semantic chunks for multi-pass encoding.

    Chunking order (highest priority first):
    1. Import block (always its own chunk — small, exact recall critical)
    2. Interface summary (its own chunk — the API contract)
    3. Each top-level class/function as a chunk
    4. Patch history + test failures as a chunk
    5. Remaining file contents split at class/function boundaries
    """
    ...

@dataclass
class CodeChunk:
    chunk_type: str             # 'imports', 'interfaces', 'function', 'class',
                                # 'patches', 'tests', 'body'
    name: str                   # e.g., 'MyClass', 'parse_config', 'imports'
    content: str                # the chunk text
    priority: float             # 0.0-1.0, higher = more important for recall
```

Each chunk is encoded independently via H(), producing one adapter per chunk. These are then composed. **The composition method is an ablation target:**

| Method | Description | Status |
|--------|------------|--------|
| **TIES merge** | Merge all chunk adapters via TIES | Ablation target (A1) |
| **DARE merge** | Merge all chunk adapters via DARE | Ablation target (A1) |
| **Priority-weighted stacking** | Load highest-priority chunk adapters via vLLM's multi-adapter support (if available), merge remainder | Ablation target (A1) |
| **Single-pass on full artifact** | Skip chunking, encode entire artifact text in one perceiver pass (truncated to max_length) | Baseline for ablation |

The spec **does not hardcode a composition method**. The default is single-pass (baseline). Chunk-to-rank composition is gated behind `enable_chunk_composition` (default: false) and must prove itself via ablation A1 before becoming the default.

### Adapter placement as ablation target

The current hypernetwork targets modules discovered by `d2l_probe.probe_model()`: attention projections (q/k/v/o_proj) and optionally MLP projections (gate/up/down_proj). Layer indices are all attention layers.

**For code-state encoding, the optimal placement may differ from generic trajectory encoding.** Early layers may matter more for syntactic structure (identifiers, imports), while later layers may matter more for semantic coherence (function behavior, API contracts). Attention-only vs. attention+MLP adapters may trade precision for breadth.

The implementation exposes adapter placement as configurable:

```python
@dataclass
class AdapterPlacement:
    """Controls which model layers and modules receive LoRA weights.

    All fields are ablation targets — the defaults match the current
    hypernetwork checkpoint's native configuration, but overrides can
    be tested.
    """
    target_modules: list[str] | None = None     # None = use checkpoint default
    layer_indices: list[int] | None = None      # None = use checkpoint default
    layer_selection: str = 'all'                # 'all', 'early_half', 'late_half',
                                                 # 'every_other', 'first_last_quarter'
```

**Implementation constraint:** The perceiver checkpoint was trained with specific target modules and layer counts. Changing which modules receive LoRA weights at inference time requires either:
(a) generating for all modules and zeroing out the unwanted ones (wasteful but simple), or
(b) training placement-aware checkpoints (out of scope for v1).

V1 uses approach (a): generate full adapter, then zero out weights for excluded modules/layers before loading. This is sufficient for ablation. A placement-optimized checkpoint is a v2 concern.

### Adaptive capacity control (experimental)

**All values below are starting hypotheses.**

**Rank control mechanisms (v1):**
1. **Scaling factor** — adjust adapter influence strength (base: 0.16 from HPO). Any boost above base is experimental.
2. **Chunk-to-rank composition** — for long code, decompose into semantic chunks and compose per-chunk adapters. Preferred over arbitrary token-window multi-pass when dealing with code. **This is an ablation target, not a proven default.**

**Strategy resolution:**

```python
def resolve_adapter_strategy(
    phase: str,
    artifact: ArtifactState | TrajectoryState,
    artifact_tokens: int,
    chunk_threshold: int,
    base_scaling: float,
    enable_chunk_composition: bool = False,
    code_scaling_boost: float = 1.2,  # EXPERIMENTAL — unvalidated
) -> AdapterStrategy:
    is_code_phase = phase in {'code', 'code_repair', 'integrate'}
    exceeds_single_pass = artifact_tokens > chunk_threshold

    if is_code_phase and exceeds_single_pass and enable_chunk_composition:
        # Semantic chunking for code — ablation target
        return ChunkComposition(
            scaling=base_scaling * code_scaling_boost,
            merge_method=default_merge_method,
        )
    elif exceeds_single_pass:
        # Non-code or chunk composition disabled: truncate to max_length
        # and encode what fits in a single pass
        return SinglePass(scaling=base_scaling, truncate=True)
    else:
        return SinglePass(scaling=base_scaling)
```

**What changed from rev 2:**
- Generic multi-pass (overlapping token windows) replaced with semantic chunk composition for code phases
- Non-code phases that exceed single-pass just truncate — trajectory context degrades more gracefully than code under truncation
- `enable_chunk_composition` defaults to false (single-pass baseline until ablation proves chunk composition helps)

### Phase-configurable turn cycles

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
3. **Adapter collapse detected** — the paper documents that adapter influence can collapse into repetitive or near-constant behavior after several compression steps:

| Signal | Computation | Threshold | Action |
|--------|------------|-----------|--------|
| **Inter-adapter cosine similarity** | Cosine sim between flattened LoRA-A weights of turn N and N-1 | > 0.95 for 2 consecutive turns | Attempt recovery (expand sliding window). If recovery fails, halt. |
| **Adapter norm ratio** | L2 norm of current adapter / L2 norm of first adapter | < 0.1 (collapse) or > 10.0 (explosion) | Halt immediately |
| **Output repetition** | Fraction of output 4-grams that appear in previous turn's output | > 0.8 | Attempt recovery. If recovery fails, halt. |

These signals are logged every turn regardless of whether they trigger a halt.

**Recovery before halt:** On collapse detection, the system tries one recovery turn with an expanded sliding window (2x default, capped at 80% of context window) and reset scaling to base. If recovery produces a non-collapsed turn, continue. If not, halt and return best output so far.

## Architecture

### New LangGraph: Reasoning Loop

```
START → reason → [phase_executes?]
  → yes: execute → reflect → build_artifact → compress_to_adapter → check_health → should_continue
  → no:  build_artifact → compress_to_adapter → check_health → should_continue

should_continue:
  → healthy AND turn_count < max_turns AND finish_reason == 'length': → reason
  → collapse detected AND not yet attempted recovery: → recover → reason
  → otherwise: → END
```

**Nodes:**
- `reason`: Generate with sliding window prompt + current adapter. Prompt constructed from sliding window + `reasoning_continue.j2`.
- `execute`: Run generated code in sandbox (reuses existing `execute_node`).
- `reflect`: Evaluate results (reuses existing `reflect_node`).
- `build_artifact`: Construct `ArtifactState` (code phases) or `TrajectoryState` (text phases) from current turn's output and execution results. For code phases: extract interfaces via `blackboard.extract_interfaces()`, extract imports, compute diff from previous turn, identify unresolved obligations (declared-but-empty functions, failing tests).
- `compress_to_adapter`: Render artifact via `artifact_compress.j2`, encode via H(), optionally chunk-compose, load new adapter.
- `check_health`: Compute adapter health signals. Update state.
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

    # Artifact state (code phases)
    artifact: dict[str, Any] | None       # serialized ArtifactState
    # Trajectory state (text phases)
    trajectory: dict[str, Any] | None     # serialized TrajectoryState

    # Reasoning loop control
    turn_count: int
    max_turns: int
    sliding_window: str
    sliding_window_size: int
    base_sliding_window_size: int
    current_adapter_path: str | None
    previous_adapter_weights: Any | None
    first_adapter_norm: float | None
    scaling_factor: float
    enable_chunk_composition: bool
    chunk_threshold: int
    phase_executes: bool
    turn_history: list[dict[str, Any]]

    # Adapter placement (ablation target)
    adapter_placement: dict[str, Any] | None  # serialized AdapterPlacement

    # Health monitoring
    adapter_cosine_sim: float
    adapter_norm_ratio: float
    output_repetition: float
    consecutive_high_similarity: int
    recovery_attempted: bool
```

### Compress-to-Adapter Pipeline

**Code phases:**
1. **Build ArtifactState** — extract interfaces, imports, compute diff, identify obligations
2. **Render via `artifact_compress.j2`** — structured text emphasizing imports (verbatim), interfaces (verbatim), patches (prose), test failures (structured), code skeleton
3. **Resolve strategy** — single-pass (default) or chunk composition (if enabled and artifact exceeds threshold)
4. **Generate adapter(s)**:
   - Single pass: one `run_hypernetwork()` call with rendered artifact
   - Chunk composition: semantic chunking → per-chunk H() → compose via configured method
5. **Apply placement mask** — zero out weights for excluded modules/layers (if `adapter_placement` overrides defaults)
6. **Retain previous adapter weights** — for next turn's cosine similarity check
7. **Load adapter** — unload previous, load new
8. **Build next prompt** — sliding window + `reasoning_continue.j2`

**Text phases:**
1. **Build TrajectoryState** — simpler: just output + feedback + diagnosis
2. **Render via `trajectory_compress.j2`**
3. **Single-pass H()** (truncated if too long)
4. **Load adapter, build prompt** — same as code phases

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

All token counting uses the base model's tokenizer via the ModelPool.

### Return value semantics

**Code phases:** The canonical output is the **last turn's `artifact.file_contents`** — a complete code snapshot, not a concatenation of partial outputs. Each turn produces a complete solution (the sliding window + adapter provide context for what came before). If the last turn fails tests but an earlier turn passed, return the best-passing turn's `artifact.file_contents`. The downstream consumer receives a single coherent code block.

**Text phases:** Last turn's output only.

**Metadata:** turn count, adapter IDs generated, health signals per turn, recovery triggered, final adapter path, artifact state evolution (for debugging).

## Configuration

New `ReasoningLoopConfig` added to `PipelineConfig`:

| Field | Default | Env var | Status |
|-------|---------|---------|--------|
| `max_turns` | 20 | `RUNE_MAX_REASONING_TURNS` | Engineering |
| `context_budget_ratio` | 0.75 | `RUNE_CONTEXT_BUDGET_RATIO` | Engineering |
| `sliding_window_tokens` | 1024 | `RUNE_SLIDING_WINDOW_TOKENS` | Starting value |
| `chunk_threshold` | 1024 | `RUNE_CHUNK_THRESHOLD` | Experimental |
| `enable_chunk_composition` | false | `RUNE_ENABLE_CHUNK_COMPOSITION` | Experimental — must prove via ablation |
| `code_scaling_boost` | 1.2 | `RUNE_CODE_SCALING_BOOST` | Experimental |
| `default_merge_method` | ties | `RUNE_MERGE_METHOD` | Experimental |
| `collapse_cosine_threshold` | 0.95 | `RUNE_COLLAPSE_COSINE_THRESHOLD` | Experimental |
| `collapse_norm_min` | 0.1 | `RUNE_COLLAPSE_NORM_MIN` | Experimental |
| `collapse_norm_max` | 10.0 | `RUNE_COLLAPSE_NORM_MAX` | Experimental |
| `collapse_repetition_threshold` | 0.8 | `RUNE_COLLAPSE_REPETITION_THRESHOLD` | Experimental |
| `adapter_target_modules` | null | `RUNE_ADAPTER_TARGET_MODULES` | Ablation target |
| `adapter_layer_selection` | all | `RUNE_ADAPTER_LAYER_SELECTION` | Ablation target |

Phase-specific sliding window defaults (overridable via `RUNE_SLIDING_WINDOW_{PHASE}`):

| Phase | Sliding window tokens | Rationale |
|-------|----------------------|-----------|
| decompose | 256 | Short text, low locality |
| plan | 512 | Plans reference recent decisions |
| code | 1024 | Active code needs exact recent recall |
| code_repair | 1536 | Needs broken code + error context |
| integrate | 2048 | References multiple subtask outputs |
| diagnose | 512 | Diagnostic text, moderate locality |

Per-phase values are starting hypotheses subject to ablation A3.

## File Changes

### New files

| File | Purpose |
|------|---------|
| `services/rune-agent/src/rune_agent/reasoning_loop.py` | `ReasoningLoopState`, `create_reasoning_loop_graph()`, all node functions |
| `services/rune-agent/src/rune_agent/artifact_state.py` | `ArtifactState`, `PatchRecord`, `TrajectoryState`, `CodeChunk`, `chunk_code_state()`, `build_artifact_state()` |
| `services/rune-agent/src/rune_agent/adapter_strategy.py` | `AdapterStrategy`, `SinglePass`, `ChunkComposition`, `AdapterPlacement`, `resolve_adapter_strategy()` |
| `services/rune-agent/src/rune_agent/adapter_health.py` | `compute_cosine_similarity()`, `compute_norm_ratio()`, `compute_output_repetition()`, `check_health()` |
| `services/rune-agent/tests/test_reasoning_loop.py` | Unit + integration tests |
| `services/rune-agent/tests/test_artifact_state.py` | Artifact construction and chunking tests |
| `services/rune-agent/tests/test_adapter_health.py` | Health monitoring unit tests |
| `services/rune-agent/tests/test_code_preservation.py` | Exact code-state preservation evaluation suite |
| `libs/shared/src/shared/templates/reasoning_continue.j2` | Prompt template for reasoning continuation |
| `libs/shared/src/shared/templates/artifact_compress.j2` | Template for rendering ArtifactState for H() |
| `libs/shared/src/shared/templates/trajectory_compress.j2` | Template for rendering TrajectoryState for H() |

### Modified files

| File | Change |
|------|--------|
| `services/rune-agent/src/rune_agent/graph.py` | Export `create_reasoning_loop_graph` |
| `scripts/rune_runner.py` | Add `run_reasoning_loop()`, escalation logic in phase loops |
| `libs/shared/src/shared/pipeline_config.py` | Add `ReasoningLoopConfig` to `PipelineConfig` |
| `libs/model-training/src/model_training/adapter_generator.py` | Accept optional `placement_mask` to zero out excluded modules/layers |

## Observability

MLflow tracing per reasoning loop invocation:

**Spans:**
- Parent: `reasoning_loop/{phase}`
- Child per turn

**Metrics (logged every turn):**

| Metric | Why it matters |
|--------|---------------|
| `reasoning_loop/turn` | Progress |
| `reasoning_loop/artifact_tokens` | Compression pressure |
| `reasoning_loop/adapter_cosine_sim` | Collapse detection |
| `reasoning_loop/adapter_norm` | Collapse/explosion |
| `reasoning_loop/adapter_norm_ratio` | Drift from baseline |
| `reasoning_loop/output_repetition` | Behavioral collapse |
| `reasoning_loop/chunk_count` | Composition complexity (0 = single pass) |
| `reasoning_loop/hypernetwork_latency_ms` | Performance |
| `reasoning_loop/adapter_load_latency_ms` | Performance |
| `reasoning_loop/sliding_window_tokens` | Tracks recovery expansions |
| `reasoning_loop/scaling_factor` | Strategy tracking |
| `reasoning_loop/strategy` | single_pass or chunk_composition |
| `reasoning_loop/recovery_triggered` | Collapse recovery |
| `reasoning_loop/identifier_recall` | Code preservation eval (see below) |
| `reasoning_loop/signature_consistency` | Code preservation eval |
| `reasoning_loop/import_preservation` | Code preservation eval |

## Evaluation: Exact Code-State Preservation

Standard Pass@1 is insufficient for evaluating code-state compression. The adapter must preserve specific structural properties of the code artifact. These evaluations run automatically after each turn in code phases and are logged as MLflow metrics.

| Eval | What it measures | How |
|------|-----------------|-----|
| **Identifier recall** | Can the model use variable/function names from earlier turns correctly? | Extract all identifiers from `artifact.file_contents` at turn N-1. After turn N, check what fraction appear correctly in the new output (not hallucinated variants). |
| **API signature consistency** | Do function signatures remain stable across turns? | Extract signatures from `artifact.interface_summary` at turn N-1. After turn N, diff against new signatures. Score = fraction unchanged. |
| **Import preservation** | Are imports from earlier turns preserved? | Exact string match of `artifact.import_block` across turns. Score = fraction of original imports still present. |
| **Regression reintroduction** | Does the model reintroduce bugs that were fixed in earlier turns? | Track test failures fixed in `artifact.patches`. After turn N, check if any previously-fixed tests are failing again. Score = fraction of fixes retained. |

These evals are computed in `build_artifact` node after each turn's execution. They serve dual purpose:
1. **Runtime signal** — a sudden drop in identifier recall or import preservation may indicate the adapter is losing fidelity, complementing the health monitoring signals
2. **Ablation metric** — the primary comparison axis for all adapter-related ablations

## Testing Strategy

### Functional tests

1. **Unit tests** (no GPU):
   - `resolve_adapter_strategy()` — strategy selection for all phase/artifact size combinations
   - `chunk_code_state()` — semantic chunking produces correct boundaries (functions, classes, imports as separate chunks)
   - `build_artifact_state()` — correct extraction of interfaces, imports, obligations
   - `should_continue` node — termination (max turns, stop, all collapse signals)
   - `check_health` — cosine sim, norm ratio, output repetition computation
   - Sliding window construction — correct token truncation
   - Recovery logic — window expansion, scaling reset, single-attempt guard
   - Adapter placement mask — correct zeroing of excluded modules/layers

2. **Integration tests** (mock hypernetwork):
   - Reasoning loop graph with mock H() — verify turn progression, state updates, termination
   - Escalation path — mock context budget small, verify escalation triggers
   - Collapse detection + recovery — mock H() to return near-identical adapters, verify halt
   - Return semantics — verify code phases return best-passing turn's artifact
   - Code preservation evals — verify identifier recall, signature consistency, import preservation computed correctly

3. **End-to-end** (GPU):
   - Deliberately long task requiring >1 continuation — verify escalation and coherent output

### Ablation plan

Each ablation holds all other knobs at defaults and sweeps one variable. **Primary metric for code phases is the code-state preservation eval suite (identifier recall, signature consistency, import preservation, regression reintroduction), not just Pass@1.**

| ID | Variable | Values | Metric | Purpose |
|----|----------|--------|--------|---------|
| A1 | Composition method | single-pass (baseline), TIES chunk merge, DARE chunk merge, priority-weighted stacking | Preservation evals, Pass@1 | Does chunk composition help or does merge interference hurt? |
| A2 | Code scaling boost | 1.0, 1.2, 1.5, 2.0 | Preservation evals, output repetition | What boost (if any) helps code recall without degeneration? |
| A3 | Sliding window size | 256, 512, 1024, 2048 per phase | Preservation evals, Pass@1 | How much raw context does each phase actually need? |
| A4 | Continuation vs. adapter-compressed | Same long task, both paths | Preservation evals, Pass@1, total time | Does the adapter loop beat naive continuation for long tasks? |
| A5 | Structured artifact vs. raw concat | `artifact_compress.j2` vs. raw code text | Preservation evals, adapter cosine diversity | Does structured artifact improve adapter quality? |
| A6 | Collapse threshold sensitivity | cosine: 0.90/0.95/0.99, repetition: 0.6/0.8/0.9 | False positive rate, missed collapses | Calibrate health thresholds |
| A7 | Adapter placement | all layers, early half, late half, every other, attention-only vs attention+MLP | Preservation evals, adapter norm stability | Which layers/modules matter most for code-state encoding? |

## Deferred to v2

- **SVD-based rank truncation** — downward rank control.
- **Adapter caching** — cache adapters across similar code states.
- **Async chunk composition** — encode chunks in parallel.
- **Richer escalation criteria** — semantic drift, exact-recall scoring, repeated truncation patterns. V1 logs signals; v2 uses them.
- **Convergence detection** — distinct from collapse detection.
- **Placement-optimized checkpoints** — train perceiver variants for specific module/layer subsets rather than masking at inference time.
- **Cross-turn adapter composition** — compose adapters from different turns rather than re-encoding the full artifact each turn.

## Open questions

1. Does semantic chunk composition actually increase recall fidelity compared to single-pass, or does merge interference negate the structural advantage?
2. What is the actual relationship between scaling boost and code-recall fidelity for this model family?
3. At what turn count does adapter collapse typically onset?
4. Which layers and modules matter most for code-state encoding vs. trajectory encoding?
5. Is the 75% context budget ratio the right escalation trigger?
6. Does the perceiver's doc2lora training transfer well to code artifacts specifically, or does code require qualitatively different encoding than natural language documents?
