# Engine v2: Adaptive Loop with Full-Project Diagnosis

## Problem

The current engine runs a fixed decompose(3-6) -> plan -> code -> retry -> integrate pipeline regardless of task complexity. A trivial MBPP function (2 lines) takes 25 minutes and 13 LangGraph spans because:

1. Decomposition is forced to 3-6 subtasks even for single-function tasks
2. Code retries are blind -- `fix_guidance` is always empty (diagnosis only triggers post-integration)
3. Feedback is a single global field that gets overwritten across parallel subtasks
4. Trajectory templates drop iteration history, so the hypernetwork adapter can't learn from prior attempts
5. Templates are overspecified with Python-specific examples that compete with task context
6. Budget (5 steps) is too tight for the pipeline depth, making integration unreachable with any retries

## ~~Phase 0: Adapter Retrievability Validation~~ — DONE

Validated by `tools/cont_probe.py` (2026-05-27/28). Key results:

- **Adapter carries code context**: with a 4-line prompt that mentions none of add/subtract/multiply/Calculator, the adapter-equipped model knows all methods and calls them correctly. Without adapter, model duplicates existing methods.
- **Effective scaling range**: `adapter_scaling=0.49` (effective 7.84) for normal generation, `0.75` (effective 12.0) for continuation. Scaling ≥1.5 degenerates.
- **Multi-round proven**: adapter carries accumulated code between continuation rounds. Model writes new methods (not regenerations) and stops naturally with EOS.
- **Architecture**: task spec in prompt (`task_only`), code context in adapter (`code_template` trajectory). The structural prompt (done-list) drives spec compliance but the adapter provides compressed context when prompt can't fit it.
- **Critical fix**: `add_generation_prompt=True` required for Qwen3.5 thinking suppression.

Full findings: `instructions/cont_probe_findings.md`, `instructions/adapter-as-memory-report.md`.

---

## Design

### Complexity Gate

Tasks are classified upfront into two paths. rune-gpu already implements this via `_should_skip_decompose()` with a 200-word threshold + signal detection ("write a function", "implement a function", etc.), configured via `DecomposeConfig.skip_threshold`.

**Simple path** (single-function, no decomposition needed):
```
code -> test -> if fail: diagnose -> repair -> test -> (cap at 2 repairs, then resample fresh)
```

**Complex path** (multi-component, needs decomposition):
```
decompose -> plan all -> code all -> test each
  -> integrate -> test integrated
  -> if failures: diagnose (full project state) -> repair -> test
  -> (cap at 2 repairs per subtask, then resample that subtask fresh)
```

The gate reuses rune-gpu's proven heuristic: word count threshold (default 200) + single-function signal detection. Configurable via `PipelineConfig`. Agentless (top SWE-bench, $0.34/issue) confirms that skipping decomposition for simple tasks improves both speed and accuracy.

### Key Principles

1. **Complexity-gated decomposition** -- simple tasks skip decomposition entirely. Complex tasks decompose adaptively (no forced "3-6 subtasks" minimum).

2. **Two-tier test signal** -- subtasks tested individually (delivery errors) AND after integration (integration errors). Both signals feed into diagnosis.

3. **Full-project diagnosis with structured output** -- the diagnose step sees ALL subtask results (pass/fail + stderr), the integrated code, and integration test errors. It produces **structured** output: which tests failed, error type classification (syntax/name/type/assertion), relevant code location, and targeted fix instructions. Research (FeedbackEval 2025) shows structured feedback achieves 63.6% repair success vs 57.9% for raw test output.

4. **Trajectory conditioning with bounded history** -- the trajectory template includes the task description plus the **last 1-2 iterations** of error context only. Older attempts are dropped. Research shows rounds 1-2 capture 76-95% of achievable improvement, and growing trajectory text risks diluting the task signal in the hypernetwork's perceiver.

5. **Repair with resample fallback** -- after 2 failed repairs on the same code, resample fresh (generate completely new code, not another repair). The hypernetwork adapter still carries the error history via trajectory conditioning, but the code starts clean. Research (DDI 2025, REx NeurIPS 2024) shows debugging effectiveness decays 60-80% after 2-3 attempts.

6. **Per-subtask feedback** -- change `RunState.feedback` from `Feedback | None` to `dict[str, Feedback]` keyed by subtask name. Each subtask sees its own errors.

7. **Repair can re-decompose** -- the repair step can emit multiple targeted repair subtasks, re-entering the code -> test loop.

### Execution Model

**Sequential adapter+generate, parallel sandbox.** PEFT's `set_peft_model_state_dict` is not safe for concurrent use (confirmed by PEFT issue #804). Even in single-threaded asyncio, hot-swap interleaving at `await` points would corrupt adapter state. The GPU can only run one forward pass at a time anyway, so true parallelism provides no throughput gain. Keep the current sequential loop for adapter generation + model inference. Sandbox execution remains parallel via `asyncio.gather`.

The primary speed improvement comes from **reducing the number of steps** (complexity gate, no forced decomposition) rather than parallelizing within a step.

### Template Changes

**Prompts (model instruction):** Concise, language-agnostic, general-purpose. Strip Python-specific examples, unittest references, dataclass prescriptions.

**Trajectory templates (hypernetwork conditioning):** Context matched to the situation, bounded to last 1-2 iterations of history. The `code_repair.j2` template (currently orphaned) has the right shape: diagnosis, sibling skeletons, repair history. Wire it in as the repair trajectory template.

Template mapping:

| Action | Trajectory | Prompt | Notes |
|--------|-----------|--------|-------|
| decompose | decompose.j2 | prompt_decompose_concise.j2 | Remove "3-6" constraint, use concise prompt |
| plan | plan.j2 | prompt_plan.j2 | Strip Python-specific example, language-agnostic |
| code | code.j2 | prompt_code.j2 | Strip Stack example, remove language refs |
| code_continue | code_continue.j2 | prompt_code_continue.j2 | Adapter continuation: trajectory packs accumulated code for hypernetwork, prompt is minimal (task + directive + tail). Adapter regenerated each round. |
| repair | code_repair.j2 | prompt_code_repair.j2 | Wire up orphaned templates, structured diagnosis input |
| integrate | integrate.j2 | prompt_integrate.j2 | Strip unittest reference |
| diagnose | diagnose.j2 | prompt_diagnose.j2 | Structured output: failed tests, error type, location |

Retire `code_retry.j2` and `prompt_code_retry.j2` -- `code_repair` subsumes them with richer context (diagnosis, sibling skeletons, repair history).

**Truncation limits** (aligned with rune-gpu's proven values):

| Field | Limit | Rationale |
|-------|-------|-----------|
| project/task | 300 chars | Brief context for adapter conditioning |
| subtask description | 500 chars | Enough for detailed spec |
| plan | 1200 chars | Full architecture sketch |
| existing_code | 2000 chars | ~50 lines, enough for review |
| dependency_interfaces | 800 chars | Key signatures from deps |
| error_summary | 500 chars | Full traceback tail |
| fix_guidance | 150 chars | First sentence of diagnosis (rune-gpu proven) |
| hypernetwork max_length | 512 tokens | Perceiver input cap |

### Policy Changes (`select_action`)

Current flow:
```
no subtasks -> decompose
unplanned -> plan (layer 0)
failing -> code or code_retry (serial within step)
all passing, no integrated_code -> integrate
integration failed -> diagnose
diagnosed -> integrate again
```

New flow:
```
# Simple path (1 subtask or no decomposition)
code -> test -> fail? -> diagnose (structured) -> repair
  -> fail again? -> diagnose -> repair
  -> fail third time? -> resample (fresh code, not repair)
  -> pass -> done

# Complex path
decompose -> plan all -> code all (sequential adapter, parallel sandbox)
  -> test each -> any failing? -> diagnose (full state, structured)
  -> repair failing subtasks (uses code_repair.j2)
  -> 2 repairs failed? -> resample that subtask
  -> all passing -> integrate -> test integrated
  -> integration fail? -> diagnose (with integration errors) -> repair
  -> pass -> done
```

Key differences from current:
- Complexity gate skips decomposition for simple tasks
- `code_retry` replaced by `diagnose -> repair` cycle (matches rune-gpu's two-step pattern)
- Diagnosis happens after ANY failure, not only post-integration
- Diagnosis produces structured output (not raw stderr)
- 2-repair cap with resample fallback (research-informed, not in rune-gpu)
- Repair uses `code_repair.j2` with diagnosis, sibling context, and bounded history
- **Adapter continuation**: gated sub-loop when `end_reason == LENGTH`. Each round: rebuild trajectory with accumulated code → regenerate adapter (scaled by `cont_multiplier`) → generate with minimal prompt (`task_only` pattern). Exit on EOS or budget exhaustion. Proven by cont_probe multi-round experiments.

### State Changes

```python
# Current
feedback: Feedback | None  # single global, overwritten

# New
feedback: dict[str, Feedback]  # per-subtask, keyed by name
integration_feedback: Feedback | None  # separate integration test result
```

The trajectory template handles history rendering directly from `retries`, `code_results`, and `feedback` state -- no separate `iteration_history` type needed. Templates include only the last 1-2 iterations of context.

### Budget Model

- Per-subtask: 2 repair attempts, then 1 resample (3 total code generations per subtask)
- Global: max_iterations (configurable, default 10) on the outer loop
- No step-level budget -- the loop runs until convergence or limits

## Files Changed

| File | Change |
|------|--------|
| `engine/state.py` | `feedback` -> `dict[str, Feedback]`, add `integration_feedback` |
| `engine/policy.py` | Complexity gate, diagnose-after-any-failure, repair action, resample fallback, remove code_retry |
| `engine/graph.py` | Per-subtask feedback, bounded history in context, structured diagnosis |
| `engine/parse.py` | Parse structured diagnosis output, parse repair output |
| `templates/decompose.j2` | Remove "3-6" constraint, language-agnostic |
| `templates/plan.j2` | Strip Python-specific example, language-agnostic |
| `templates/code.j2` | Strip Stack example, language-agnostic |
| `templates/code_repair.j2` | Wire up, bounded history, structured diagnosis input |
| `templates/prompt_*.j2` | Concise, language-agnostic |
| `templates/diagnose.j2` | Structured output schema: failed tests, error type, location |
| `bench/runner.py` | Update for new state schema |

## What's NOT Changing

- Hypernetwork architecture (perceiver, LoRA generation)
- Model inference (freeform + structured generation)
- Sandbox executor
- MLflow tracking
- CLI interface
- Config schema (except budget fields)

## Prior Art (rune-gpu)

Key patterns carried forward from rune-gpu's proven implementation:

- **Complexity gate**: `_should_skip_decompose()` with 200-word threshold + signal detection
- **Two-step diagnose→repair**: Separate adapter generated for diagnosis phase
- **Fix guidance cap**: 150 chars, first sentence of diagnosis text
- **Template conciseness**: Aggressive truncation (300-2000 chars per field), 512-token hypernetwork input cap
- **`code_continue` phase**: Recovery when generation hits token limit
- **Error extraction**: `_extract_error_summary()` filters for lines containing "Error", "Exception", "assert", "FAIL:", "ERROR:"
- **Trajectory recording**: Append-only per-step records with code, test results, stderr, exit code

Key patterns NOT carried forward:
- rune-gpu's `code_continue` is first-retry-always-continue behavior; v2 uses diagnose-first for repair, adapter continuation for truncation
- rune-gpu lacks resample fallback (repairs until exhausted); v2 adds research-informed 2-repair cap

## Adapter Continuation (from cont_probe findings)

When code generation hits the token limit (`end_reason == LENGTH`), a gated sub-loop extends the output:

1. Accumulated code is packed into `code_continue.j2` trajectory → hypernetwork generates new adapter
2. Adapter scaled by `adapter_scaling * cont_multiplier` (default 1.53, HPO-able `[1.0, 2.5]`)
3. Minimal prompt via `prompt_code_continue.j2`: task spec + "Write ONLY the next unimplemented part" + 4-line tail
4. Generate `cont_max_tokens` (default 128) per round
5. Repeat until EOS or budget exhausted

Each round consumes one `budget_remaining` step. No new state fields. After exit, assembled code goes through normal sandbox → diagnose/repair flow.

**Why separate templates:** `code.j2` carries PRIOR ATTEMPTS, PLAN, PRACTICES — wrong signal for continuation. The hypernetwork perceiver input needs dense code, not metadata. `prompt_code.j2` instructs "write tests FIRST" — continuation needs a minimal directive to avoid regeneration.

**Why adapter regeneration per round:** Proven by cont_probe. The adapter encodes the growing codebase — each round's adapter "remembers" what the previous rounds wrote. Keeping a stale adapter loses track of accumulated code.

## Research References

- **Agentless** (Xia et al., 2024): Top SWE-bench without decomposition or agent loops
- **FeedbackEval** (2025): Structured test feedback = 63.6% repair success
- **DDI / Debugging Decay Index** (Adnan & Kuhn, 2025): 60-80% capability loss after 2-3 repair attempts
- **REx** (Tang et al., NeurIPS 2024): Explore/exploit bandit for repair vs resample
- **Self-Debug** (Chen et al., ICLR 2024): Self-explanation improves repair by 12%
- **CYCLE** (OOPSLA 2024): Execution feedback provides 23% relative improvement
- **PEFT issue #804**: `set_peft_model_state_dict` not safe for concurrent use
- **Doc-to-LoRA** (SakanaAI, 2025): Perceiver hypernetwork conditioning on LLM activations
