# Adapter-Encoded Continuation for Truncated Generation

## Problem

When structured JSON output exceeds `max_tokens`, the current `_try_completion` in `inference.py` feeds the model's own output tensor back for continuation. This works but the prompt context grows linearly with each continuation attempt -- the full thinking phase + all prior JSON tokens must be re-attended. For large outputs this hits memory limits and degrades quality as the model loses track of the goal buried far back in the sequence.

## Core Idea

Replace the growing-context continuation with an **adapter-encoded** continuation. When generation truncates:

1. Feed the partial output (goal + code written so far) through the hypernetwork to produce a **fresh LoRA adapter** that encodes the continuation context.
2. Hot-swap the new adapter into the model (upscaled via a fixed multiplier).
3. Resume generation with a **short prompt** (last N lines of partial output as anchor) and **assistant-turn prefill** (the partial JSON itself, so the model continues from exactly where it left off).
4. The xgrammar matcher is advanced past the partial JSON to maintain structural validity.

The adapter carries the memory, not the prompt. Prompt window stays constant regardless of how many continuations occur. This is limited only by a retry budget (`max_continuations`), not by context length.

## Design Decisions

**Sequence-based baseline stays.** The current `_try_completion` in `inference.py` remains as the single-shot fallback. The adapter-encoded continuation loop lives in `wrapper.py` and calls `_try_completion`-style logic only if the adapter path is unavailable (no hypernetwork loaded).

**Continuation trajectory format.** The hypernetwork input for a continuation is:

```
CONTINUATION {attempt}/{max}
GOAL: {subtask description}
CODE SO FAR:
{partial_json_text}
```

This is deliberately different from the initial task trajectory. We embed *what was written*, not the original task decomposition. The hypernetwork extracts activations through the base model (which still has the prior adapter loaded), so continuation activations implicitly carry task awareness.

**Adapter scaling.** Continuation adapters are scaled by `adapter_scaling * continuation_scaling_multiplier`. The multiplier is a fixed config value determined empirically via HPO. Intuition: the model needs stronger conditioning to recover partial output from the adapter alone.

**Assistant-turn prefill.** The partial JSON is tokenized and concatenated onto the chat-encoded prompt as an assistant turn prefix. This is the real resume point -- the short prompt in the user turn provides informational context, but the prefill ensures the model continues from the exact byte where it left off.

**Skip thinking on continuation.** The thinking phase (unconstrained generation until `</think>`) is skipped for continuation calls. The model has already thought; we just need it to keep writing JSON.

**Exhausted continuations feed into the engine's retry loop.** If `max_continuations` is reached and JSON is still incomplete, `wrapper.generate()` returns `GenerationResult(truncated=True)`. In `graph.py`, `step_node` checks `result.truncated` before calling `_extract_code()` to avoid a Pydantic `ValidationError` on malformed JSON. It creates a synthetic `Feedback(stderr="output truncated after N continuations", exit_code=1)` so the engine's normal diagnose-repair cycle fires. The adapter for that retry encodes the full trajectory of what was attempted across all continuations.

**`goal_summary` passed explicitly.** `wrapper.generate()` accepts an optional `goal_summary: str` parameter (populated from `subtask.description` by `step_node`). This avoids re-deriving the goal from the prompt.

## Files Changed

### `config.py` -- 3 new fields

```python
max_continuations: int = 3
continuation_scaling_multiplier: float = 1.3
continuation_prompt_lines: int = 3
```

Defaults are starting points; actual values determined by HPO.

### `inference.py` -- 1 new optional parameter

`generate()` gains `grammar_prefix: str | None = None`. When set:
- Thinking phase is skipped entirely.
- The grammar matcher is advanced past `grammar_prefix` before constrained generation begins.
- The prefix is tokenized and appended to the encoded prompt as assistant-turn prefill.
- The returned `GenerationResult.text` contains `grammar_prefix + new_tokens` (the full accumulated JSON), so callers can pass it directly to the next continuation.

`_try_completion` is unchanged (remains as sequence-based fallback).

### `adapter.py` -- extract `scale_lora_b()` helper

```python
def scale_lora_b(state_dict: dict[str, Any], factor: float) -> dict[str, Any]:
    return {k: v * factor if "lora_B" in k else v for k, v in state_dict.items()}
```

Currently this logic is inline in `graph.py:step_node`. Extract once, call from both `step_node` and `wrapper.py`.

### `wrapper.py` -- continuation loop (~25 lines)

`generate()` gains `goal_summary: str = ""`. After calling `inference_generate()`, if `result.truncated` and `self._hypernet is not None`:

```
for attempt in range(max_continuations):
    continuation_trajectory = format_continuation_trajectory(
        attempt, max_continuations, goal_summary, result.text
    )
    adapter = generate_adapter_weights(hypernet, continuation_trajectory, ...)
    scaled = scale_lora_b(adapter, adapter_scaling * continuation_scaling_multiplier)
    hotswap_adapter(model, scaled)
    result = inference_generate(
        model, tokenizer, short_prompt,
        grammar_prefix=result.text,
        max_tokens=max_tokens,
        ...
    )
    if not result.truncated:
        break
```

The `short_prompt` is the last `continuation_prompt_lines` lines of `result.text`.

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

Modeled after `tools/adapter_scaling_hpo.py`. Uses Optuna to sweep:
- `continuation_scaling_multiplier`: [0.8, 2.5]
- `continuation_prompt_lines`: [1, 10]
- Balance between adapter conditioning vs prompt context

Test tasks: 2-3 generation tasks that reliably truncate at current `max_tokens`. Metrics: JSON completion rate, code correctness (sandbox pass), token efficiency.

## What Does NOT Change

- `step_node` orchestration logic (adapter generation, scaling, hotswap, sandbox execution)
- `state.py` (RunState, StepRecord, Feedback, Action, Subtask)
- All templates (decompose, plan, code, repair, integrate, diagnose)
- `policy.py` (action selection, DAG layer grouping)
- Hypernetwork architecture / training

## Risks and Mitigations

**Risk: Continuation adapter doesn't recover partial output well enough.**
Mitigation: Sequence-based `_try_completion` remains as baseline. The adapter path is an experiment. HPO script validates empirically before committing to defaults.

**Risk: Hypernetwork's 2048-token input limit truncates long partial outputs.**
Mitigation: Continuation trajectory truncates code from the beginning, keeping the most recent output (which is what the model needs to continue from). Goal summary is kept short.

**Risk: Grammar matcher state diverges after prefill.**
Mitigation: `accept_string()` on the full partial JSON validates grammar consistency before continuing. If it fails, fall back to returning truncated result.
