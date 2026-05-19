# Pipeline Speed & Reliability Fixes

**Date:** 2026-05-19
**Branch:** `feat/training-speed-opts`
**Status:** Design approved, pending implementation

## Problem Statement

Benchmark HPO runs (May 18–19) revealed three compounding failures that cause **every trial to score 0.0**:

1. **Length stops dominate** — 93–100% of generations hit the `max_tokens` ceiling and get truncated. Thinking tokens (`<think>` blocks) consume the budget on text phases; code phases simply need more tokens than the 1024 default.
2. **Decomposition explosion** — Simple single-function tasks decompose into 16–30 subtasks because the model's chain-of-thought leaks into subtask entries (e.g., "Numbered list? Yes"). Each extra subtask triggers plan + code + retry cycles, multiplying generation count.
3. **Adapter OOM cascade** — 60+ adapters accumulate within a phase (8 subtasks × 3 retries × plan/code/diagnose). The 22GB A10G fills up, causing 277 CUDA OOM failures in a single run.

These are independent root causes that compound: decomposition explosion multiplies the generation count, length stops ensure every generation fails, and adapter accumulation kills the GPU before the pipeline can self-correct.

## Design

### 1. Multi-Turn Continuation on Length Stops

#### Current behavior

The LangGraph graph is linear: `generate → execute → reflect → [should_retry]`. When `generate` produces a truncated output (`finish_reason == "length"`), the incomplete code is executed (fails), reflected on, and counted as a failed attempt. The rune_runner layer detects truncation on **retry** (attempt > 0) and uses `code_continue` phase, but the first attempt always wastes a full cycle executing obviously-incomplete code.

#### New behavior

Add a routing decision after `generate` that checks `finish_reason`:

```
generate → [route_after_generate]
  → execute                                    (finish_reason == "stop")
  → accumulate → [back to rune_runner adapter gen] → generate  (finish_reason == "length")
```

**New `accumulate` node** in the LangGraph graph:
- Appends partial output to `accumulated_code` state field
- Increments `continuation_count` (new field, separate from `attempt_count`)
- Sets `phase = "code_continue"` for the next generate call
- Does NOT count as a retry — no `attempt_count` bump

**Adapter refresh on continuation:** The rune_runner layer (which owns hypernetwork calls) generates a **fresh adapter** from the updated trajectory that includes code-so-far. Each continuation turn gets an adapter encoding the full accumulated context. The prompt stays minimal — `code_continue.j2` already caps `existing_code` at 1200 chars. The adapter carries the deep context; the prompt just orients the model ("continue from here").

**Continuation cap:** Max 3 continuations per subtask. After 3, proceed to execute with accumulated code regardless of completion status.

#### State changes (`services/rune-agent/src/rune_agent/state.py`)

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `continuation_count` | `int` | `0` | Tracks how many continuation turns have been used |
| `accumulated_code` | `str` | `""` | Concatenated output across continuation turns |
| `finish_reason` | `str` | `"stop"` | Propagated from generate_node for routing |

#### Files changed

- `services/rune-agent/src/rune_agent/graph.py` — Add `route_after_generate`, `accumulate_node`, new edges
- `services/rune-agent/src/rune_agent/state.py` — Add state fields
- `services/rune-agent/src/rune_agent/nodes.py` — `generate_node` writes `finish_reason` to state
- `scripts/rune_runner.py` — Adapter generation on continuation; simplify existing continuation logic (lines 1189–1415) since graph handles accumulation

---

### 2. Thinking Token Budget

#### Current behavior

`TransformersProvider.generate()` passes `max_new_tokens = max_tokens` to the model. When `enable_thinking=True` (text-only phases: decompose, plan, diagnose), `<think>` blocks count against this budget. A 1024-token generation might spend 700 tokens thinking and only get 300 tokens of actual output — then hit the length cap. Tokens cluster around ~1125–1141 (just over max_tokens=1024), confirming thinking is consuming the budget.

#### New behavior

Add a `thinking_budget` parameter that gives thinking tokens their own allocation:

```python
effective_max = max_tokens + thinking_budget  # thinking doesn't starve response
```

After generation:
- Strip `<think>` blocks (as today)
- Count only non-thinking tokens for `finish_reason` determination
- `finish_reason = "length"` only if the **response** (post-strip) token count >= `max_tokens`

#### Defaults

| Phase | `enable_thinking` | `thinking_budget` | `max_tokens` | Effective `max_new_tokens` |
|-------|-------------------|-------------------|--------------|---------------------------|
| decompose, plan, diagnose | `True` | 512 | 1024 | 1536 |
| code, integrate, repair | `False` | 0 | 2048 | 2048 |

#### Files changed

- `libs/inference/src/inference/transformers_provider.py` — Add `thinking_budget` param, adjust `max_new_tokens` and `finish_reason` logic
- `libs/inference/src/inference/provider.py` — Add `thinking_budget` to `InferenceProvider` ABC
- `libs/inference/src/inference/vllm_provider.py` — Mirror the parameter
- `libs/inference/src/inference/ollama_provider.py` — Mirror the parameter
- `libs/inference/src/inference/llamacpp_provider.py` — Mirror the parameter
- `services/rune-agent/src/rune_agent/nodes.py` — Pass `thinking_budget` per phase
- `libs/shared/src/shared/pipeline_config.py` — Add `generation.thinking_budget` field (default 512)

---

### 3. Decomposition Quality (Three-Pronged)

#### 3a. Prompt improvement + few-shot examples

Update `libs/shared/src/shared/templates/decompose.j2` and `decompose_concise.j2`:

**Add explicit formatting constraints:**
```
Output ONLY a numbered list of subtasks. No preamble, no analysis, no reasoning.
Each line: "N. subtask_name — description [depends: none]"
Do NOT include your chain-of-thought as subtask entries.
```

**Add 2–3 few-shot examples:**
```
Example for "Write a function to check if a number is prime":
1. implement_is_prime — Core primality check with edge cases [depends: none]
2. add_tests — Unit tests for primes, non-primes, edge cases [depends: 1]

Example for "Write a function to merge two sorted lists":
1. implement_merge — Two-pointer merge of sorted lists [depends: none]
2. handle_edge_cases — Empty lists, single-element, duplicates [depends: 1]
3. add_tests — Comprehensive test cases [depends: 1, 2]
```

**Add negative example:**
```
BAD (do not do this):
1. Analyze the Request — ...
2. Numbered list? Yes
3. Never code? Yes
These are reasoning steps, NOT subtasks.
```

#### 3b. Task-complexity gating

Add `_should_skip_decompose()` in `scripts/rune_runner.py`:

```python
def _should_skip_decompose(project_prompt: str) -> bool:
    token_count = len(project_prompt.split())
    if token_count > 200:
        return False
    single_fn_signals = ["write a function", "implement a function",
                         "write a method", "create a function", "def "]
    return any(s in project_prompt.lower() for s in single_fn_signals)
```

When `True`, skip the entire decompose phase and use a single subtask:
```python
subtasks = [{"name": "implementation", "description": project_prompt[:200], "depends_on": []}]
```

Configurable via `PipelineConfig` field `decompose.skip_threshold` (word count, default 200).

#### 3c. Structured JSON output (larger change, phase 2)

- Add `json_schema` parameter to `TransformersProvider.generate()` for guided/constrained decoding
- Decompose phase passes `DecomposeResult.model_json_schema()` as the schema constraint
- Output parsed as JSON, validated against `DecomposeResult` directly — no regex
- Falls back to current regex parsing if JSON mode unavailable (other providers)
- Requires investigating HuggingFace `outlines` or `transformers` JSON grammar support for Qwen 3.5

**Sequencing:** 3a first (template-only), 3b second (small code change), 3c third (requires guided decoding infrastructure).

#### Files changed

- `libs/shared/src/shared/templates/decompose.j2` — Prompt + few-shot
- `libs/shared/src/shared/templates/decompose_concise.j2` — Same updates
- `scripts/rune_runner.py` — `_should_skip_decompose()`, skip logic in `run_phased_pipeline()`
- `libs/shared/src/shared/pipeline_config.py` — `decompose.skip_threshold` field
- `libs/inference/src/inference/transformers_provider.py` — `json_schema` param (phase 2)

---

### 4. Eager Adapter Unload

#### Current behavior

Adapters are loaded before generation and persist until `_cleanup_phase_adapters()` runs at phase boundaries. Within a phase (e.g., coding 8 subtasks × 3 retries), adapters accumulate. With ~60 adapters loaded on a 22GB GPU, CUDA OOM triggers and cascades to all subsequent problems.

#### New behavior

Unload each adapter immediately after `run_iteration()` returns:

```python
code_state = await run_iteration(graph, ..., adapter_id=code_adapter_id, ...)
# Immediately release GPU memory
if code_adapter_id:
    await provider.unload_adapter(code_adapter_id)
    torch.cuda.empty_cache()
```

Applied at **every** `run_iteration()` call site: decompose, plan, code, code_continue, code_retry, code_repair, integrate, diagnose phases.

**Trade-off:** If the same adapter is needed again (e.g., cycle-fix retry), it gets re-loaded. Adapter load is ~100ms vs ~120s for generation — negligible overhead. The VRAM savings prevent the OOM cascade that killed 277 problems.

`_cleanup_phase_adapters()` remains as a safety net at phase boundaries but should rarely find anything to clean.

#### Files changed

- `scripts/rune_runner.py` — Add `await provider.unload_adapter()` + `torch.cuda.empty_cache()` after every `run_iteration()` call

---

## Implementation Order

| Phase | Work | Risk | Files |
|-------|------|------|-------|
| **P1** | Eager adapter unload | Low — mechanical, no logic change | `rune_runner.py` |
| **P2** | Decompose prompts + few-shot (3a) | Low — template-only | `decompose.j2`, `decompose_concise.j2` |
| **P3** | Task-complexity gating (3b) | Low — additive, fallback-safe | `rune_runner.py`, `pipeline_config.py` |
| **P4** | Thinking token budget | Medium — touches provider interface | `transformers_provider.py`, `provider.py`, `nodes.py`, `pipeline_config.py`, other providers |
| **P5** | LangGraph continuation routing | Medium — graph topology change | `graph.py`, `state.py`, `nodes.py`, `rune_runner.py` |
| **P6** | Structured JSON decompose (3c) | Higher — new infrastructure | `transformers_provider.py`, `rune_runner.py` |

P1–P3 are independent and can be done in parallel. P4 and P5 are independent of each other. P6 depends on P4 (JSON mode shares the generate() interface changes).

## Testing

- **P1:** Run single MBPP problem, verify `nvidia-smi` shows VRAM reclaimed after each generation
- **P2–P3:** Run decompose phase on 10 simple MBPP tasks, verify subtask count ≤ 3 and no chain-of-thought leakage
- **P4:** Run plan phase, verify thinking tokens don't count against response budget; check `finish_reason` accuracy
- **P5:** Run code phase on a task that requires >1024 tokens, verify continuation produces complete code without wasting an attempt
- **P6:** Run decompose phase with JSON mode, verify output validates against `DecomposeResult`
- **Integration:** Re-run benchmark HPO on 10-problem subset, verify non-zero trial scores

## Success Criteria

- Length-stop rate drops from 93–100% to <20%
- No CUDA OOM failures in a full 88-problem HPO trial
- Simple MBPP tasks decompose into 1–3 subtasks (not 16–30)
- At least one HPO trial scores >0.0 on the 10-problem subset
