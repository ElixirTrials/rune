# Prefill + Continue: Byte-Boundary Continuation with Adapter Swap

Supersedes the Layer 2 (adapter-encoded continuation) section of `2026-05-26-adapter-continuation-design.md`. Layer 1 (`_try_completion` loop for JSON-level truncation) is unchanged.

## Problem

The current continuation sub-loop in `graph.py:210-283` re-prompts the model with a separate template (`prompt_code_continue.j2`) containing only the last 4 lines of accumulated code, then merges the raw output via `extract_code` → `merge_overlap` → `dedup_code` → `chunk.strip()` concatenation. This produces three failure modes:

1. **`chunk.strip()` destroys indentation** (graph.py:270). When the model resumes inside a method body, stripping removes leading whitespace, causing `return` statements to land at column 0 ("return outside function").
2. **Text-level merging is structurally fragile.** `merge_overlap` uses exact line equality; `dedup_code` uses name-based class/function dedup. Minor reformatting (whitespace, blank lines) defeats both.
3. **No post-continuation validation.** Broken accumulated code goes straight to the sandbox, fails, and enters the slow diagnose→repair cycle for what is fundamentally a generation-budget problem.

The root cause: continuation generates in a different token context than the original output, and the text-surgery merge can't reliably reconstruct valid code from two independent generations.

## Core Idea

Replace re-prompt-and-merge with **assistant-turn resumption**: pass the full accumulated code as the assistant's partial response using `continue_final_message=True`, swap the adapter, and let the model continue from the same character boundary. No text extraction, no merging, no stripping.

**Important caveat**: `continue_final_message=True` re-tokenizes the accumulated code from its string representation. The resulting tokens are semantically equivalent but not necessarily bit-identical to the model's original output tokens (e.g., whitespace normalization, BPE merge boundaries). This is character-boundary continuation, not token-boundary KV-cache continuation — the KV cache is recomputed from scratch each round. True token-boundary continuation with KV reuse is Approach B (aLoRA).

The adapter encodes the full trajectory context (including accumulated code) via the hypernetwork, so the model has both explicit local context (the prefill) and implicit semantic context (the adapter weights). Indentation is preserved because the model continues from its own prior text.

## Design

### Continuation loop (replaces graph.py:210-283)

```
Initial generation (xgrammar JSON, adapter_0)
  → result.truncated OR compile()/tree-sitter detects SyntaxError
  ↓
accumulated_code = extract_partial_code(result.text)
cont_budget = run_config["cont_budget"]  # default 5
empty_rounds = 0

while cont_budget > 0 and empty_rounds < 2:
    1. Build continuation trajectory (code_continue template, includes accumulated_code)
    2. Generate adapter_N via hypernetwork
    3. Scale and hotswap adapter_N
    4. Build chat messages:
         system: "Output only Python code. No commentary, no explanations,
                  no markdown fences. Continue exactly from where the code left off."
         user:   task description (truncated to 200 chars)
         assistant: accumulated_code    ← partial turn
    5. tokenizer.apply_chat_template(messages,
         continue_final_message=True, enable_thinking=False, tokenize=True)
    6. model.generate(input_ids=template_ids, max_new_tokens=max_tokens,
         eos_token_id=eos, repetition_penalty=..., no_repeat_ngram_size=...)
    7. new_tokens = output_ids[len(template_ids):]
       new_chunk = tokenizer.decode(new_tokens, skip_special_tokens=True)
    8. Check degeneration: degeneration_score(new_chunk) > 0.5 → break
    9. if new_chunk.strip():
         accumulated_code += new_chunk   # direct append, no strip/merge
         empty_rounds = 0
       else:
         empty_rounds += 1
    10. cont_budget -= 1
    11. Validate syntax:
          Python: try compile(accumulated_code) — if no SyntaxError → break
          Other:  tree-sitter parse, check no ERROR nodes at EOF → break
        (If valid, exit loop early — code is done)
    12. If model emitted EOS (len(new_tokens) < max_tokens) and code still invalid:
          # Model thinks it's done but code is broken — don't loop forever
          break

raw_text = json.dumps({"code": accumulated_code})
# → continues to sandbox execution via existing flow
```

### `continue_final_message` mechanics

Qwen3 chat template with `continue_final_message=True` produces:

```
<|im_start|>system
Output only Python code....<|im_end|>
<|im_start|>user
Write a class LinkedList...<|im_end|>
<|im_start|>assistant
<think>

</think>

class Node:
    def __init__(self, data):
        self.data = data
        self.next_node = None

class LinkedList:
    def __init__(self):
        self.head = None

    def append(self, data):
        new_node = Node(data)
```

No `<|im_end|>` after the assistant content — the model continues from the exact last character. The `<think></think>` block is present but empty (thinking disabled). The model sees its own prior output and continues naturally, maintaining indentation and context.

### Syntax validation

**Primary**: tree-sitter (already in `pyproject.toml`, used in `interfaces.py`). Parse the accumulated code and check for `ERROR` or `MISSING` nodes at EOF. Tree-sitter never fails — incomplete code produces a tree with error nodes, not an exception. Language-agnostic.

**Fast path (Python only)**: `compile(accumulated_code, "<check>", "exec")`. Catches SyntaxError faster than tree-sitter for Python, which is the common case today.

```python
def validate_syntax(code: str, *, language: str = "python") -> bool:
    if language == "python":
        try:
            compile(code, "<check>", "exec")
            return True
        except SyntaxError:
            pass
    # Fall through to tree-sitter for all languages
    return _treesitter_check(code, language)
```

**Language parameter**: Hardcoded to `"python"` in this implementation. The engine does not currently track output language — all code actions produce Python. The `language` parameter exists for future extensibility but is not wired to anything yet.

### Prefill cost

Each continuation round prefills the full accumulated code. Cost is one forward pass (not autoregressive):

| Round | Prefill tokens | Estimated time (FlashAttention, A10) |
|-------|---------------|--------------------------------------|
| 1 | ~2500 (prompt + initial code) | ~200ms |
| 2 | ~3000 | ~250ms |
| 5 | ~5000 | ~400ms |

Autoregressive generation of 512 tokens takes ~10-20s on Qwen3.5-9B. Prefill is <5% of round time.

**Future optimization — sliding window**: If profiling shows prefill becomes a bottleneck (unlikely at current scale), cap the assistant prefix at the last N tokens. The adapter provides long-range context; the window provides local syntax context. This is not needed for the initial implementation.

## Files Changed

### `engine/graph.py` — continuation sub-loop rewrite

Replace lines 210-283 with the new loop. Key changes:
- Remove calls to `extract_code`, `merge_overlap`, `dedup_code`
- Use `tokenizer.apply_chat_template(continue_final_message=True)` for prompting
- Direct append (`accumulated_code += new_chunk`) instead of strip/merge
- Add post-round syntax validation with early exit

The continuation still uses a different system prompt and `thinking_budget=0` as today. The `cont_multiplier` scaling of the continuation adapter is preserved.

### `engine/continuation.py` — add `validate_syntax`, simplify

- Add `validate_syntax(code, language)` using tree-sitter + compile() fast path
- Keep `extract_partial_code` (used for initial JSON extraction)
- Keep `degeneration_score` (used for continuation quality check)
- `merge_overlap`, `dedup_code`, `extract_code` are no longer called from `graph.py` — but they are still imported by `tools/cont_probe.py`, `tools/capacity_sweep.py`, and `tests/unit/test_continuation.py`. Keep the functions for now; mark with a `# TODO: remove once probe tools are updated` comment. Do not break existing tools or tests.

### `model/inference.py` — add `generate_continuation()`

New function that handles the prefill + continue generation:

```python
async def generate_continuation(
    model, tokenizer, prefix_ids,
    *, max_tokens, temperature, top_p,
    repetition_penalty, no_repeat_ngram_size,
) -> GenerationResult:
```

Takes pre-tokenized `prefix_ids` (from `apply_chat_template`), generates `max_tokens` new tokens in raw mode (no xgrammar, no thinking). Returns `GenerationResult` with only the new text and `truncated` flag.

### `templates/` — remove prompt template only

Delete `prompt_code_continue.j2` (the model prompt). Keep `code_continue.j2` — it renders the **trajectory** that feeds the hypernetwork for adapter generation, not the model prompt. The continuation prompt is now built programmatically from chat messages in `step_node`.

## What Does NOT Change

- `_try_completion` in inference.py (Layer 1 — JSON-level truncation within a single generation)
- `policy.py` (action selection — diagnose/repair still fire if continuation fails)
- The outer engine loop / StateGraph structure
- `state.py` (RunState, StepRecord, Feedback, etc.)
- Code/repair/integrate/diagnose templates
- Hypernetwork architecture
- `adapter.py` (scaling, hotswap)
- `config.py` fields (cont_multiplier, cont_budget, no_repeat_ngram_size all still used)

## Test Strategy

### Unit tests (no GPU)

- **`validate_syntax`**: Valid Python → True, SyntaxError → False, empty string → False.
- **`generate_continuation`**: Mock `model.generate` to return a fixed token sequence. Verify only new tokens are decoded, `truncated` flag is correct, and `GenerationResult` fields are populated.
- **`step_node` continuation path**: Mock `model.generate_adapter` and `model.generate`. Verify that when `result.truncated=True`, the loop calls `generate_continuation` (not the old `extract_code`/`merge_overlap` path), appends the chunk directly, and exits early when `compile()` succeeds.

### GPU smoke test

- Run `tools/smoke_test_engine.py` with `max_tokens=512` (forces truncation). Verify:
  1. Continuation rounds fire (log messages appear).
  2. No `SyntaxError: 'return' outside function` in trajectory.
  3. Fewer total steps than the baseline (7 steps → target ≤5).
  4. Final integrated code passes sandbox execution.
- Fix the budget analysis in `smoke_test_engine.py` to track actual continuation rounds (currently always reports 0).

### Regression

- Existing `tests/unit/test_continuation.py` tests for `extract_code`, `dedup_code`, `merge_overlap` must still pass (functions are kept).
- `tests/unit/test_parse.py`, `tests/unit/test_sandbox.py` must still pass.

## Risks and Mitigations

**Risk: Prefill cost grows with accumulated code length.**
Mitigation: Prefill is a single forward pass — <5% of round time at current scale. Sliding window is a future optimization if needed.

**Risk: Model re-interprets prior code differently with new adapter.**
Mitigation: This is the desired behavior — the new adapter encodes the updated trajectory, giving the model awareness of what was written. The prior code tokens serve as explicit context anchoring.

**Risk: `continue_final_message` template quirks across model families.**
Mitigation: Validate template output for each supported model. Qwen3 is confirmed working. The fallback is raw token concatenation (bypass chat template entirely).

**Risk: Model emits EOS prematurely in continuation round.**
Mitigation: Step 12 in the loop detects this (new_tokens < max_tokens + code invalid). The loop breaks and falls through to diagnose/repair. This matches the existing behavior where continuation can't complete the code.
