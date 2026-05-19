# Design: Fix HPO decomposition over-splitting via thinking-block strip + Pydantic validation

**Date:** 2026-05-19
**Branch:** feat/training-speed-opts
**Status:** Approved

## Problem

Qwen 3.5's default chat template ends with `<think>\n`, prompting the model to generate a chain-of-thought reasoning block before its actual answer. The `<think>` (token 248068) and `</think>` (token 248069) tokens are NOT in the tokenizer's `all_special_tokens`, so `skip_special_tokens=True` does not strip them.

The reasoning block contains numbered items that match the subtask parser's regex:

```
<think>
1. Analyze the Request: Need to break down a statistics library
2. Check Constraints: 3-6 subtasks required
3. Numbered list? Yes, must use numbered format
...
</think>

1. Statistical Functions -- Implement mean, median, mode [depends: none]
2. Data Validation -- Input type checking [depends: none]
...
```

The parser (`_parse_subtask_list`) matches both the thinking items and the real subtasks, producing 16-30 subtasks instead of 3-6. This cascades:

- Too many subtasks -> too many adapter forward passes -> CUDA OOM
- CUDA OOM on first problem -> all subsequent problems fail
- 0% pass rate across all 30 HPO trials (confirmed via MLflow + logs)

This happens at all temperature levels. Even trials with temp=0.21 produced the same pattern. The root cause is structural (thinking tokens in output), not a sampling issue.

## Evidence

| Run | Date | Subtasks parsed | Unique | Thinking items in output | Pass rate |
|-----|------|-----------------|--------|--------------------------|-----------|
| benchmark-hpo-20260518-173436 | May 18 17:34 | 7 | 7 | "Analyze the Request:", "Analyze the Project Logic:", "Draft Subtasks:" | 0/7 passed |
| benchmark-hpo-20260518-194524 | May 18 19:45 | 30 | 30 | "Analyze the Request:", "Numbered list? Yes", "Never code? Yes", literal "3" | 0% (all 30 trials) |
| benchmark-hpo-20260519-065004 | May 19 06:50 | 16 | 16 | "Analyze the Request:", "Check Constraints:", "Double Check 'Never Code':" | 0% |

All 62 generations in the catastrophic run hit `finish=length` -- thinking block consumed the token budget.

## Design decision: preserve thinking

`enable_thinking=False` pre-fills `<think>\n\n</think>\n\n` in the chat template, which means the model never reasons at all. This kills a quality signal -- the chain-of-thought helps the model produce better structured output. The adapter pipeline benefits from the model reasoning before answering.

Instead: let the model think during generation, strip the thinking block from the decoded output before returning to callers.

## Solution

### 1. Provider-level thinking-block strip

**File:** `libs/inference/src/inference/transformers_provider.py`

In `generate()`, after decoding tokens to text (line 220), strip `<think>...</think>` blocks:

```python
text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
```

Edge case -- truncation mid-thinking (hit `max_tokens` inside a `<think>` block, no closing `</think>`):

```python
text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
```

The second regex only fires if a dangling `<think>` remains after the first pass. `re` is already imported at module level.

This is the single point of enforcement. All downstream consumers (rune_runner, benchmark HPO, any future caller) receive clean output.

### 2. Pydantic models for subtask validation

**File:** `libs/shared/src/shared/rune_models.py`

New models following existing patterns (`AdapterRef`, `CodingSession`):

```python
class Subtask(BaseModel):
    name: str = Field(min_length=1)
    description: str = ""
    depends_on: list[int] = []

class DecomposeResult(BaseModel):
    subtasks: list[Subtask] = Field(min_length=2, max_length=8)
```

- `min_length=2`: single-subtask results fail validation, triggering fallback
- `max_length=8`: hard cap enforced structurally

### 3. Parser validation in rune_runner

**File:** `scripts/rune_runner.py`

After `_parse_subtask_list()` builds its list via regex, validate through `DecomposeResult`:

```python
try:
    result = DecomposeResult(subtasks=[Subtask(**s) for s in subtasks])
    return [s.model_dump() for s in result.subtasks]
except ValidationError:
    return [{"name": "implementation", "description": model_output[:200].strip(), "depends_on": []}]
```

In the decompose scoring block: hard truncate to first 8 subtasks after dedup, before scoring.

A TODO marks the regex parser for future replacement with structured JSON output:

```python
# TODO: Replace regex parsing with structured JSON output from the model,
# validated directly against DecomposeResult. Requires adding JSON mode
# to TransformersProvider.generate() and updating decompose templates.
```

### 4. HPO search space narrowing

**File:** `scripts/optimization/run_benchmark_hpo.py`

```python
temperature = trial.suggest_float("temperature", 0.1, 0.4)  # was 0.7
```

## Files changed

| File | Change |
|------|--------|
| `libs/inference/src/inference/transformers_provider.py` | Strip `<think>` blocks from decoded output in `generate()` |
| `libs/shared/src/shared/rune_models.py` | Add `Subtask` and `DecomposeResult` Pydantic models |
| `scripts/rune_runner.py` | Validate parsed subtasks through `DecomposeResult`, hard cap at 8, TODO for structured output |
| `scripts/optimization/run_benchmark_hpo.py` | Temperature ceiling 0.7 -> 0.4 |

## Files NOT changed

| File | Reason |
|------|--------|
| `libs/shared/src/shared/templates/*.j2` | Templates correctly say 3-6 subtasks; problem is upstream |
| `libs/shared/src/shared/blackboard.py` | DAG logic is fine; subtask shape just needs to be valid before it gets there |
| `libs/model-training/` | Hypernetwork and adapter code not involved |

## Future work

Replace regex-based parsing in `_parse_subtask_list` with structured JSON output validated directly against the Pydantic models. This requires:

1. Adding JSON mode / constrained generation to `TransformersProvider.generate()`
2. Updating decompose trajectory and prompt templates to request JSON
3. Testing JSON output reliability with Qwen 3.5 + LoRA adapters

## Verification

- `uv run ruff check` on all changed files
- `uv run pytest -x` for regression
- Manual: confirm strip regex handles closed blocks, unclosed blocks (truncation), and no-thinking-block cases
