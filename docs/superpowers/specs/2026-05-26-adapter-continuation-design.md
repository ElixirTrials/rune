# Layered Continuation for Truncated Generation

## Problem

When structured JSON output exceeds `max_tokens`, the model's generation is incomplete. The current `_try_completion` in `inference.py` handles this with a single sequence-based retry, but it has no loop, no grammar advancement, and the engine crashes on truncated JSON when `CodeResult.model_validate_json()` throws.

## Core Idea

A two-layer continuation strategy that fires **only when the fail reason is length** (i.e., `len(json_tokens) >= max_tokens`). Other failure modes (syntax errors, sandbox failures, etc.) are handled by the engine's existing diagnose-repair loop and are not continuation candidates.

**Layer 1 — Sequence-based continuation.** Feed the model's own output tensor back with the xgrammar matcher advanced past the partial JSON. The model continues from its own attention context — no information loss. Repeat until the JSON is complete or the context window fills up.

**Layer 2 — Adapter-encoded continuation.** When the context window is exhausted and the JSON is still incomplete, switch strategies: encode the partial output into a fresh LoRA adapter via the hypernetwork, hot-swap it (upscaled), and continue with a short fixed-size prompt + grammar constraint. The adapter carries the memory, not the prompt. Prompt window stays constant from this point forward.

Layer 1 handles the common case cheaply. Layer 2 covers the long tail where output exceeds context capacity.

## Design Decisions

### Layer 1: Sequence-based

**Loop `_try_completion` until context fills.** The existing `_try_completion` already feeds `prior_output` back. Wrap it in a loop: after each continuation, check `matcher.is_completed()`. If not done and the total sequence length is still within the model's context window, continue. No artificial budget — the context window is the natural limit.

**Grammar advancement.** Each continuation round advances the xgrammar matcher past the accumulated partial JSON via `accept_string()`, so structural validity is maintained across rounds.

**When to escalate.** When `prior_output.shape[1] + max_tokens` would exceed the model's context window (`model.config.max_position_embeddings`), Layer 1 can't grow further. If the JSON is still incomplete, hand off to Layer 2.

### Layer 2: Adapter-encoded

**Continuation trajectory format.** The hypernetwork input for a continuation is:

```
CONTINUATION
GOAL: {subtask description}
CODE SO FAR:
{partial_json_text}
```

This embeds *what was written*, not the original task decomposition. The hypernetwork extracts activations through the base model (which still has the prior adapter loaded), so continuation activations implicitly carry task awareness.

**Adapter scaling.** Continuation adapters are scaled by `adapter_scaling * continuation_scaling_multiplier`. The multiplier is a fixed config value determined empirically via HPO. Intuition: the model needs stronger conditioning to recover partial output from the adapter alone.

**No full prefill.** The partial JSON is NOT prefilled into the assistant turn — that would grow the context window, defeating the purpose. Instead, three mechanisms provide continuity: (1) the grammar matcher advanced past the partial JSON enforces structural validity, (2) the adapter encodes the full semantic context, (3) the short prompt provides the first M + last N lines as context anchors. The model generates new tokens constrained by grammar, and we concatenate `partial_json + new_tokens` on the caller side. The HPO script tests bounded prefill (last 256/512 tokens) as a variant to determine empirically whether the adapter alone is sufficient.

**Skip thinking on continuation.** The thinking phase is skipped for continuation calls. The model has already thought; we just need it to keep writing JSON.

**`goal_summary` passed explicitly.** `wrapper.generate()` accepts an optional `goal_summary: str` parameter (populated from `subtask.description` by `step_node`). This avoids re-deriving the goal from the prompt.

### Exhausted continuations

If Layer 2 also fails to complete the JSON (context window fills again in the adapter path, or grammar rejects the accumulated output), `wrapper.generate()` returns `GenerationResult(truncated=True)`. In `graph.py`, `step_node` checks `result.truncated` before calling `_extract_code()` and creates a synthetic `Feedback(stderr="output truncated ...", exit_code=1)` so the engine's normal diagnose-repair cycle fires. The adapter for that retry encodes the full trajectory of what was attempted across all continuations.

## Implementation Order

1. **HPO script** (`tools/continuation_scaling_hpo.py`) — already written. Run first to validate that adapter-encoded continuation is feasible. If not, only Layer 1 ships.
2. **Layer 1** — harden `_try_completion` with a loop, grammar advancement, and context-window check. This is proven to work and ships regardless of HPO results.
3. **Layer 2** — implement adapter-encoded continuation in `wrapper.py`. Only proceeds if HPO confirms viability.
4. **Truncation guard in `graph.py`** — needed by both layers.

## Files Changed

### `config.py` -- new fields

```python
continuation_scaling_multiplier: float = 1.3
continuation_prompt_first_lines: int = 2
continuation_prompt_last_lines: int = 3
```

`continuation_prompt_first_lines`: lines from the beginning of output (goal/structure context).
`continuation_prompt_last_lines`: lines from the end of output (continuation anchor).
Defaults are starting points; actual values determined by HPO. Only used by Layer 2.

### `inference.py` -- Layer 1 changes

`_try_completion` gains a loop: continue feeding `prior_output` back until `matcher.is_completed()` or context window is full. Returns a flag indicating whether escalation to Layer 2 is needed.

`generate()` gains `grammar_prefix: str | None = None` for Layer 2 calls. When set:
- Thinking phase is skipped entirely.
- The grammar matcher is advanced past `grammar_prefix` before constrained generation begins.
- No prefill — the model generates fresh tokens guided by adapter + grammar + short prompt.
- The returned `GenerationResult.text` contains only the new tokens. The caller concatenates `grammar_prefix + result.text` to accumulate the full JSON.

### `adapter.py` -- extract `scale_lora_b()` helper

```python
def scale_lora_b(state_dict: dict[str, Any], factor: float) -> dict[str, Any]:
    return {k: v * factor if "lora_B" in k else v for k, v in state_dict.items()}
```

Currently this logic is inline in `graph.py:step_node`. Extract once, call from both `step_node` and `wrapper.py`.

### `wrapper.py` -- Layer 2 continuation loop

`generate()` gains `goal_summary: str = ""`. After `inference_generate()` returns with `escalate_to_adapter=True`:

```
accumulated = result.text
while result.truncated:
    continuation_trajectory = format_continuation_trajectory(
        goal_summary, accumulated
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
result = result._replace(text=accumulated)
```

The short prompt has fixed size: first M lines (goal/structure) + last N lines (continuation anchor). The grammar matcher is advanced past `accumulated` so the model only generates structurally valid continuations, while the adapter carries the full semantic context. Prompt window stays constant.

### `graph.py` -- truncation guard

Before `_extract_code(a, text)`, check `result.truncated`. If true, skip code extraction and sandbox execution for that action; instead create synthetic `Feedback`:

```python
if result.truncated:
    feedback_map[name] = Feedback(
        stdout="",
        stderr=f"output truncated after continuation exhausted ({len(text)} chars)",
        exit_code=1,
    )
```

This prevents `CodeResult.model_validate_json()` from throwing on incomplete JSON and lets the engine's diagnose-repair loop handle it.

### `tools/continuation_scaling_hpo.py` -- already written

Validates Layer 2 feasibility. Sweeps:

- `continuation_scaling_multiplier`: [0.8, 10.0] (includes Doc2LoRA-level scaling)
- `continuation_prompt_first_lines`: [0, 5]
- `continuation_prompt_last_lines`: [1, 10]
- `prompt_strategy`: categorical [head_tail, tail_only, instruction_wrapped]
- `trajectory_flavor`: categorical [minimal_goal_code, with_attempt_counter, with_structural_summary]
- `prefill_mode`: categorical [none, bounded_256, bounded_512]

Test scenarios at 3 tiers (fresh, mid-continuation, deep continuation). Triple-checks completion via grammar, schema validation, and generation-stopped signals.

## What Does NOT Change

- `step_node` orchestration logic (adapter generation, scaling, hotswap, sandbox execution)
- `state.py` (RunState, StepRecord, Feedback, Action, Subtask)
- All templates (decompose, plan, code, repair, integrate, diagnose)
- `policy.py` (action selection, DAG layer grouping)
- Hypernetwork architecture / training

## Risks and Mitigations

**Risk: Layer 2 adapter doesn't recover partial output well enough.**
Mitigation: HPO script validates empirically before building Layer 2. Layer 1 ships independently and handles the common case. If Layer 2 is not feasible, we still have a working continuation system.

**Risk: Hypernetwork's 2048-token input limit truncates long partial outputs.**
Mitigation: Continuation trajectory truncates code from the beginning, keeping the most recent output. Goal summary is kept short.

**Risk: Grammar matcher state diverges.**
Mitigation: `accept_string()` on the full partial JSON validates grammar consistency before each continuation round in both layers. If it fails, return truncated result and let the engine retry.
