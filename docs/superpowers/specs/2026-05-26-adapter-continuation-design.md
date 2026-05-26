# Adapter-Encoded Continuation for Truncated Generation

## Problem

When structured JSON output exceeds `max_tokens`, the current `_try_completion` in `inference.py` feeds the model's own output tensor back for continuation. This works but the prompt context grows linearly with each continuation attempt -- the full thinking phase + all prior JSON tokens must be re-attended. For large outputs this hits memory limits and degrades quality as the model loses track of the goal buried far back in the sequence.

## Core Idea

Replace the growing-context continuation with an **adapter-encoded** continuation. This fires **only when the fail reason is length** (i.e., `len(json_tokens) >= max_tokens`). Other failure modes (syntax errors, sandbox failures, etc.) are handled by the engine's existing diagnose-repair loop and are not continuation candidates.

When generation truncates due to length:

1. Feed the partial output (goal + code written so far) through the hypernetwork to produce a **fresh LoRA adapter** that encodes the continuation context.
2. Hot-swap the new adapter into the model (upscaled via a fixed multiplier).
3. Resume generation with a **short prompt** (first M lines for goal context + last N lines of partial output as continuation anchor).
4. The xgrammar matcher is advanced past the partial JSON to maintain structural validity.

The adapter carries the memory, not the prompt. Prompt window stays constant regardless of how many continuations occur. This is limited only by a retry budget (`max_continuations`), not by context length.

## Design Decisions

**Single continuation path.** The adapter-encoded continuation loop lives in `wrapper.py`. A loaded hypernetwork is required; if missing, it is an error. The existing `_try_completion` in `inference.py` is removed.

**Continuation trajectory format.** The hypernetwork input for a continuation is:

```
CONTINUATION {attempt}/{max}
GOAL: {subtask description}
CODE SO FAR:
{partial_json_text}
```

This is deliberately different from the initial task trajectory. We embed *what was written*, not the original task decomposition. The hypernetwork extracts activations through the base model (which still has the prior adapter loaded), so continuation activations implicitly carry task awareness.

**Adapter scaling.** Continuation adapters are scaled by `adapter_scaling * continuation_scaling_multiplier`. The multiplier is a fixed config value determined empirically via HPO. Intuition: the model needs stronger conditioning to recover partial output from the adapter alone.

**No full prefill.** The partial JSON is NOT prefilled into the assistant turn -- that would grow the context window linearly, defeating the purpose. Instead, three mechanisms provide continuity: (1) the grammar matcher advanced past the partial JSON enforces structural validity, (2) the adapter encodes the full semantic context, (3) the short prompt provides the first M + last N lines as context anchors. The model generates new tokens constrained by grammar, and we concatenate `partial_json + new_tokens` on the caller side. The HPO script tests bounded prefill (last 256/512 tokens) as a variant to determine empirically whether the adapter alone is sufficient.

**Skip thinking on continuation.** The thinking phase (unconstrained generation until `</think>`) is skipped for continuation calls. The model has already thought; we just need it to keep writing JSON.

**Exhausted continuations feed into the engine's retry loop.** If `max_continuations` is reached and JSON is still incomplete, `wrapper.generate()` returns `GenerationResult(truncated=True)`. In `graph.py`, `step_node` checks `result.truncated` before calling `_extract_code()` to avoid a Pydantic `ValidationError` on malformed JSON. It creates a synthetic `Feedback(stderr="output truncated after N continuations", exit_code=1)` so the engine's normal diagnose-repair cycle fires. The adapter for that retry encodes the full trajectory of what was attempted across all continuations.

**`goal_summary` passed explicitly.** `wrapper.generate()` accepts an optional `goal_summary: str` parameter (populated from `subtask.description` by `step_node`). This avoids re-deriving the goal from the prompt.

## Implementation Order

The HPO script (`tools/continuation_scaling_hpo.py`) is implemented and run **first**, before any production code changes. It validates whether the adapter can actually recover partial output well enough for continuation to work. If the HPO results show the approach is not feasible (low completion rates, poor coherence), we stop — no production code is written. Only after HPO confirms viability do we proceed to implement the changes in `config.py`, `inference.py`, `adapter.py`, `wrapper.py`, and `graph.py`.

## Files Changed

### `config.py` -- 3 new fields

```python
max_continuations: int = 3
continuation_scaling_multiplier: float = 1.3
continuation_prompt_first_lines: int = 2
continuation_prompt_last_lines: int = 3
```

`continuation_prompt_first_lines`: lines from the beginning of output (goal/structure context).
`continuation_prompt_last_lines`: lines from the end of output (continuation anchor).
Defaults are starting points; actual values determined by HPO.

### `inference.py` -- 1 new optional parameter

`generate()` gains `grammar_prefix: str | None = None`. When set:
- Thinking phase is skipped entirely.
- The grammar matcher is advanced past `grammar_prefix` before constrained generation begins.
- No prefill -- the model generates fresh tokens guided by adapter + grammar + short prompt.
- The returned `GenerationResult.text` contains only the new tokens. The caller concatenates `grammar_prefix + result.text` to accumulate the full JSON.

`_try_completion` is removed. All continuation logic moves to `wrapper.py`.

### `adapter.py` -- extract `scale_lora_b()` helper

```python
def scale_lora_b(state_dict: dict[str, Any], factor: float) -> dict[str, Any]:
    return {k: v * factor if "lora_B" in k else v for k, v in state_dict.items()}
```

Currently this logic is inline in `graph.py:step_node`. Extract once, call from both `step_node` and `wrapper.py`.

### `wrapper.py` -- continuation loop (~25 lines)

`generate()` gains `goal_summary: str = ""`. After calling `inference_generate()`, if `result.truncated` and `self._hypernet is not None`:

```
accumulated = result.text
for attempt in range(max_continuations):
    continuation_trajectory = format_continuation_trajectory(
        attempt, max_continuations, goal_summary, accumulated
    )
    adapter = generate_adapter_weights(hypernet, continuation_trajectory, ...)
    scaled = scale_lora_b(adapter, adapter_scaling * continuation_scaling_multiplier)
    hotswap_adapter(model, scaled)
    head = first_n_lines(accumulated, continuation_prompt_first_lines)
    tail = last_n_lines(accumulated, continuation_prompt_last_lines)
    short_prompt = f"{head}\n...\n{tail}"
    result = inference_generate(
        model, tokenizer, short_prompt,
        grammar_prefix=accumulated,
        max_tokens=max_tokens,
        ...
    )
    accumulated += result.text
    if not result.truncated:
        break
result = result._replace(text=accumulated)
```

The short prompt has fixed size: first M lines (goal/structure) + last N lines (continuation anchor). The grammar matcher is advanced past `accumulated` so the model only generates structurally valid continuations, while the adapter carries the full semantic context. Prompt window stays constant.

### `graph.py` -- truncation guard

Before `_extract_code(a, text)`, check `result.truncated`. If true, skip code extraction and sandbox execution for that action; instead create synthetic `Feedback`:

```python
if result.truncated:
    feedback_map[name] = Feedback(
        stdout="",
        stderr=f"output truncated after continuation budget exhausted ({len(text)} chars)",
        exit_code=1,
    )
```

This prevents `CodeResult.model_validate_json()` from throwing on incomplete JSON and lets the engine's diagnose-repair loop handle it.

### `tools/continuation_scaling_hpo.py` -- new HPO script

Modeled after `tools/adapter_scaling_hpo.py`. Uses Optuna + MLflow. This script serves two purposes: (a) validate feasibility of adapter-encoded continuation, (b) discover optimal parameter values.

**Test scenarios (3 tiers of difficulty):**

1. **Fresh truncation:** A task that produces ~1.5x `max_tokens` of output. Single continuation needed. Tests basic adapter recovery.
2. **Mid-continuation:** A task that produces ~3x `max_tokens`. Requires 2+ continuations. The hypernetwork receives accumulated partial output from prior continuations -- tests whether the adapter can carry growing history.
3. **Deep continuation:** A task that produces ~5x `max_tokens`. Pushes toward `max_continuations` limit. Tests quality degradation over many continuation cycles.

Each scenario runs the full continuation loop: generate initial output, truncate, encode accumulated output into hypernetwork, hot-swap upscaled adapter, generate with short prompt + grammar, accumulate, repeat.

**Parameters swept:**

- `continuation_scaling_multiplier`: [0.8, 2.5] -- how much to upscale continuation adapter
- `continuation_prompt_first_lines`: [0, 5] -- lines from beginning of output in prompt (goal/structure context)
- `continuation_prompt_last_lines`: [1, 10] -- lines from end of output in prompt (continuation anchor)
- `prefill_mode`: categorical ["none", "bounded_256", "bounded_512"] -- whether to also prefill last N tokens into assistant turn (validates whether adapter alone is sufficient)

**Metrics:**

- JSON completion rate (did the grammar matcher reach `is_completed()`?)
- Code correctness (sandbox pass rate)
- Semantic coherence (does continuation match what came before?)
- Token efficiency (total tokens used across all continuations)

## What Does NOT Change

- `step_node` orchestration logic (adapter generation, scaling, hotswap, sandbox execution)
- `state.py` (RunState, StepRecord, Feedback, Action, Subtask)
- All templates (decompose, plan, code, repair, integrate, diagnose)
- `policy.py` (action selection, DAG layer grouping)
- Hypernetwork architecture / training

## Risks and Mitigations

**Risk: Continuation adapter doesn't recover partial output well enough.**
Mitigation: HPO script validates empirically before committing to defaults. If adapter conditioning proves insufficient, we tune scaling/prompt parameters rather than adding fallback paths.

**Risk: Hypernetwork's 2048-token input limit truncates long partial outputs.**
Mitigation: Continuation trajectory truncates code from the beginning, keeping the most recent output (which is what the model needs to continue from). Goal summary is kept short.

**Risk: Grammar matcher state diverges after prefill.**
Mitigation: `accept_string()` on the full partial JSON validates grammar consistency before continuing. If it fails, fall back to returning truncated result.
