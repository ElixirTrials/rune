# Prefill + Continue Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the fragile re-prompt-and-merge continuation sub-loop with `continue_final_message=True` assistant-turn resumption, eliminating the `chunk.strip()` indentation bug and text-surgery merging.

**Architecture:** The continuation loop in `graph.py` stops using separate prompt templates and text merging. Instead, it builds chat messages with the accumulated code as a partial assistant turn, tokenizes via `apply_chat_template(continue_final_message=True)`, and generates new tokens that directly append to the accumulated code. A `validate_syntax()` check after each round exits early when the code compiles.

**Tech Stack:** Python 3.12, transformers tokenizer (`apply_chat_template`), tree-sitter, PEFT/LoRA

**Spec:** `docs/superpowers/specs/2026-05-28-prefill-continue-design.md`

---

## File Structure

| File | Responsibility | Change |
|------|---------------|--------|
| `src/rune/engine/continuation.py` | Continuation utilities | Add `validate_syntax()` |
| `src/rune/model/inference.py` | Model generation | Add `generate_continuation()` |
| `src/rune/model/wrapper.py` | Model layer bridge | Add `generate_continuation()` wrapper method |
| `src/rune/engine/graph.py` | Engine step_node | Rewrite continuation sub-loop (lines 210-283) |
| `src/rune/templates/prompt_code_continue.j2` | Dead prompt template | Delete |
| `tools/smoke_test_engine.py` | GPU smoke test | Fix budget tracking |
| `tests/unit/test_continuation.py` | Continuation unit tests | Add `validate_syntax` tests |
| `tests/unit/test_wrapper.py` | Wrapper unit tests | Add `generate_continuation` test |

---

### Task 0: Verify `continue_final_message` template behavior

Confirm that Qwen3's tokenizer actually produces the expected output with `continue_final_message=True` + `enable_thinking=False`. This is a throwaway verification — run it, check the output, then delete.

**Files:**
- Create: `tools/verify_chat_template.py` (throwaway)

- [ ] **Step 1: Write the verification script**

```python
"""Verify continue_final_message template behavior for Qwen3.

Run: uv run python tools/verify_chat_template.py
"""
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")

messages = [
    {"role": "system", "content": "Output only Python code."},
    {"role": "user", "content": "Write a LinkedList class."},
    {"role": "assistant", "content": "class Node:\n    def __init__(self, data):\n        self.data = data\n"},
]

result = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    continue_final_message=True,
    enable_thinking=False,
)
print("=== TEMPLATE OUTPUT ===")
print(repr(result))
print()
print("=== READABLE ===")
print(result)
print()

# Check key properties
assert "<|im_end|>" not in result.split("assistant\n")[-1], (
    "Assistant turn should NOT have <|im_end|> at end"
)
assert result.rstrip().endswith("self.data = data\n"), (
    f"Should end with the assistant content, got: ...{result[-80:]!r}"
)
print("PASS: Template produces open-ended assistant turn as expected")
```

- [ ] **Step 2: Ask user to run the script**

Run: `uv run python tools/verify_chat_template.py`

Expected output: the template ends with the assistant content (no `<|im_end|>`), and the PASS message prints. If the template includes a `<think></think>` block, note its exact format — we need to include it in the assistant prefix we construct in Task 3.

- [ ] **Step 3: Delete the verification script**

```bash
rm tools/verify_chat_template.py
```

---

### Task 1: Add `validate_syntax()` to continuation.py

Pure function with no dependencies on the model layer. Uses `compile()` fast path for Python and tree-sitter for all languages.

**Files:**
- Modify: `src/rune/engine/continuation.py`
- Modify: `tests/unit/test_continuation.py`

- [ ] **Step 1: Write failing tests for `validate_syntax`**

Add to `tests/unit/test_continuation.py`:

```python
from rune.engine.continuation import validate_syntax


class TestValidateSyntax:
    def test_valid_python(self) -> None:
        assert validate_syntax("def foo():\n    return 1\n") is True

    def test_syntax_error(self) -> None:
        assert validate_syntax("def foo(\n") is False

    def test_incomplete_class(self) -> None:
        # Truncated mid-method — SyntaxError from compile()
        code = "class Foo:\n    def bar(self):\n        x = 1\n        return"
        # "return" with no value at end of function is valid Python
        assert validate_syntax(code) is True

    def test_return_outside_function(self) -> None:
        code = "class Foo:\n    pass\nreturn 1"
        assert validate_syntax(code) is False

    def test_empty_string(self) -> None:
        assert validate_syntax("") is False

    def test_indentation_error(self) -> None:
        code = "def foo():\nreturn 1"
        assert validate_syntax(code) is False

    def test_valid_multiclass(self) -> None:
        code = (
            "class Node:\n"
            "    def __init__(self, data):\n"
            "        self.data = data\n"
            "\n"
            "class LinkedList:\n"
            "    def __init__(self):\n"
            "        self.head = None\n"
        )
        assert validate_syntax(code) is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_continuation.py::TestValidateSyntax -v`
Expected: FAIL — `ImportError: cannot import name 'validate_syntax'`

- [ ] **Step 3: Implement `validate_syntax`**

Add to `src/rune/engine/continuation.py`, after the `degeneration_score` function:

```python
def validate_syntax(code: str, *, language: str = "python") -> bool:
    """Check whether *code* is syntactically valid.

    Uses compile() as a fast path for Python, falls back to tree-sitter
    for any language. Returns False for empty input.
    """
    if not code or not code.strip():
        return False
    if language == "python":
        try:
            compile(code, "<check>", "exec")
            return True
        except SyntaxError:
            return False
    return _treesitter_check(code, language)


def _treesitter_check(code: str, language: str) -> bool:
    import tree_sitter_python as tspython  # noqa: PLC0415
    from tree_sitter import Language, Parser  # noqa: PLC0415

    lang = Language(tspython.language())
    parser = Parser(lang)
    tree = parser.parse(code.encode())
    return not tree.root_node.has_error
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_continuation.py::TestValidateSyntax -v`
Expected: all PASS

- [ ] **Step 5: Run full continuation test suite for regression**

Run: `uv run pytest tests/unit/test_continuation.py -v`
Expected: all existing tests still PASS

- [ ] **Step 6: Commit**

```bash
git add src/rune/engine/continuation.py tests/unit/test_continuation.py
git commit -m "feat: add validate_syntax() with compile() fast path and tree-sitter fallback"
```

---

### Task 2: Add `generate_continuation()` to inference.py and wrapper

New function that handles prefill + continue generation: takes chat messages with a partial assistant turn, tokenizes with `continue_final_message=True`, generates new tokens in raw mode (no xgrammar, no thinking).

**Files:**
- Modify: `src/rune/model/inference.py`
- Modify: `src/rune/model/wrapper.py`
- Modify: `tests/unit/test_wrapper.py`

- [ ] **Step 1: Write failing test for `ModelWrapper.generate_continuation`**

Add to `tests/unit/test_wrapper.py`:

```python
class TestGenerateContinuation:
    def _make_wrapper(self) -> Any:
        cfg = PipelineConfig()
        base_model = MagicMock()
        tokenizer = MagicMock()
        hypernet = MagicMock()
        hypernet.config = MagicMock()
        hypernet.config.layer_indices = [0, 1, 2]
        return ModelWrapper(base_model, tokenizer, hypernet, config=cfg)

    def test_delegates_to_inference(self) -> None:
        expected = GenerationResult(
            text="    return self.data\n", thinking="", tokens_used=10,
        )
        with patch(
            "rune.model.wrapper.inference_generate_continuation",
            new=AsyncMock(return_value=expected),
        ) as mock_gen:
            wrapper = self._make_wrapper()
            result = asyncio.run(
                wrapper.generate_continuation(
                    system_prompt="Output only Python code.",
                    user_prompt="Write a class",
                    assistant_prefix="class Node:\n    def __init__(self):\n",
                    max_tokens=512,
                )
            )
            assert result is expected
            call_kwargs = mock_gen.call_args.kwargs
            assert call_kwargs["system_prompt"] == "Output only Python code."
            assert call_kwargs["assistant_prefix"] == "class Node:\n    def __init__(self):\n"
            assert call_kwargs["max_tokens"] == 512
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_wrapper.py::TestGenerateContinuation -v`
Expected: FAIL — `AttributeError: 'ModelWrapper' object has no attribute 'generate_continuation'`

- [ ] **Step 3: Implement `generate_continuation` in inference.py**

Add after the `_try_completion` function in `src/rune/model/inference.py`:

```python
async def generate_continuation(
    model: Any,
    tokenizer: Any,
    *,
    system_prompt: str,
    user_prompt: str,
    assistant_prefix: str,
    max_tokens: int = 2048,
    temperature: float = 0.3,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    no_repeat_ngram_size: int = 0,
) -> GenerationResult:
    """Generate a continuation from a partial assistant turn.

    Builds chat messages with the assistant_prefix as an incomplete assistant
    response, tokenizes with continue_final_message=True, and generates
    max_tokens new tokens in raw mode (no xgrammar, no thinking).

    Returns GenerationResult with only the NEW text (not the prefix).
    """
    import asyncio  # noqa: PLC0415

    def _run() -> GenerationResult:
        import torch  # noqa: PLC0415

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        messages.append({"role": "assistant", "content": assistant_prefix})

        template_ids = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            continue_final_message=True,
            enable_thinking=False,
        )
        if hasattr(template_ids, "input_ids"):
            template_ids = template_ids["input_ids"]
        template_ids = template_ids.to(model.device)

        sampling: dict[str, Any] = (
            {"do_sample": True, "temperature": temperature, "top_p": top_p}
            if temperature > 0
            else {"do_sample": False}
        )

        gen_kwargs: dict[str, Any] = {
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "max_new_tokens": max_tokens,
            "repetition_penalty": repetition_penalty,
            **sampling,
        }
        if no_repeat_ngram_size > 0:
            gen_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size

        attention_mask = torch.ones_like(template_ids)
        with torch.no_grad():
            output = model.generate(
                template_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )

        new_tokens = output[0][template_ids.shape[1] :]
        new_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        truncated = len(new_tokens) >= max_tokens

        return GenerationResult(
            text=new_text,
            thinking="",
            tokens_used=template_ids.shape[1] + len(new_tokens),
            truncated=truncated,
        )

    return await asyncio.to_thread(_run)
```

- [ ] **Step 4: Add wrapper method and import in wrapper.py**

Add to the imports at the top of `src/rune/model/wrapper.py`:

```python
from rune.model.inference import generate_continuation as inference_generate_continuation
```

Add this method to `ModelWrapper`, after the existing `generate` method:

```python
    async def generate_continuation(
        self,
        system_prompt: str,
        user_prompt: str,
        assistant_prefix: str,
        max_tokens: int = 2048,
        temperature: float = 0.3,
        repetition_penalty: float = 1.1,
        top_p: float = 0.9,
        no_repeat_ngram_size: int = 0,
    ) -> GenerationResult:
        return await inference_generate_continuation(
            self._base_model,
            self._tokenizer,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            assistant_prefix=assistant_prefix,
            max_tokens=max_tokens,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_wrapper.py::TestGenerateContinuation -v`
Expected: PASS

- [ ] **Step 6: Run full wrapper test suite for regression**

Run: `uv run pytest tests/unit/test_wrapper.py -v`
Expected: all existing tests still PASS

- [ ] **Step 7: Commit**

```bash
git add src/rune/model/inference.py src/rune/model/wrapper.py tests/unit/test_wrapper.py
git commit -m "feat: add generate_continuation() for prefill+continue assistant-turn resumption"
```

---

### Task 3: Rewrite continuation sub-loop in graph.py

Replace the re-prompt-and-merge loop (lines 210-283) with the new prefill+continue loop. This is the core change.

**Files:**
- Modify: `src/rune/engine/graph.py:210-283`

- [ ] **Step 1: Update imports in graph.py**

Replace the import block at the top of `graph.py`:

Old:
```python
from rune.engine.continuation import (
    dedup_code,
    degeneration_score,
    extract_code,
    extract_partial_code,
    merge_overlap,
)
```

New:
```python
from rune.engine.continuation import (
    degeneration_score,
    extract_partial_code,
    validate_syntax,
)
```

- [ ] **Step 2: Replace the continuation sub-loop**

Replace lines 210-283 (from `if action.name in ("code", "repair") and needs_continuation:` through `raw_text = json.dumps({"code": accumulated_code})`) with:

```python
        if action.name in ("code", "repair") and needs_continuation:
            cont_multiplier = run_config.get("cont_multiplier", 1.53)
            cont_no_repeat = run_config.get("no_repeat_ngram_size", 12)
            cont_scaling = adapter_scaling * cont_multiplier
            accumulated_code = extract_partial_code(result.text)
            cont_budget = run_config.get("cont_budget", 5)
            empty_rounds = 0

            cont_sys = (
                "Output only Python code. No commentary, no explanations, "
                "no markdown fences. Continue exactly from where the code "
                "left off."
            )
            cont_user = ctx.get("task_description", "")[:200]

            while cont_budget > 0 and empty_rounds < 2:
                import torch  # noqa: PLC0415

                torch.cuda.empty_cache()

                cont_ctx = {
                    **ctx,
                    "accumulated_code": accumulated_code,
                    "resume_tail": "\n".join(accumulated_code.splitlines()[-4:]),
                }
                cont_traj = render_template("code_continue", **cont_ctx)

                cont_adapter = model.generate_adapter(cont_traj)
                cont_sd = scale_lora_b(cont_adapter.state_dict, cont_scaling)
                model.hotswap_adapter(cont_sd)
                del cont_adapter, cont_sd

                result = await model.generate_continuation(
                    system_prompt=cont_sys,
                    user_prompt=cont_user,
                    assistant_prefix=accumulated_code,
                    max_tokens=run_config.get("max_tokens", 2048),
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    top_p=top_p,
                    no_repeat_ngram_size=cont_no_repeat,
                )

                new_chunk = result.text
                degen = degeneration_score(new_chunk)
                logger.info(
                    "continuation round: +%d chars, degen=%.2f",
                    len(new_chunk),
                    degen,
                )

                if degen > 0.5:
                    logger.warning("Degeneration detected (%.2f), stopping continuation", degen)
                    break

                if new_chunk.strip():
                    accumulated_code += new_chunk
                    empty_rounds = 0
                else:
                    empty_rounds += 1

                cont_budget -= 1
                cont_budget_spent += 1

                if validate_syntax(accumulated_code):
                    logger.info("Accumulated code validates — exiting continuation")
                    break

                if not result.truncated:
                    break

            raw_text = json.dumps({"code": accumulated_code})
```

- [ ] **Step 3: Run existing unit tests for regression**

Run: `uv run pytest tests/unit/ -v -x`
Expected: all PASS. The graph module itself doesn't have unit tests that exercise `step_node` directly (it requires a full model), but continuation.py, parse.py, sandbox, and policy tests should all pass.

- [ ] **Step 4: Run linting**

Run: `uv run ruff check src/rune/engine/graph.py`
Expected: no errors (unused imports `dedup_code`, `extract_code`, `merge_overlap` are removed)

- [ ] **Step 5: Commit**

```bash
git add src/rune/engine/graph.py
git commit -m "feat: replace re-prompt-and-merge continuation with prefill+continue

Uses continue_final_message=True for character-boundary continuation.
Direct append instead of strip/merge. Syntax validation exits early
when accumulated code compiles."
```

---

### Task 4: Add step_node continuation unit test

The new continuation path in `step_node` needs a unit test that mocks the model layer and verifies the loop calls `generate_continuation` (not the old path), appends chunks directly, and exits early when syntax validates.

**Files:**
- Create: `tests/unit/test_graph_continuation.py`

- [ ] **Step 1: Write the test**

```python
"""Unit tests for the continuation sub-loop in step_node."""
from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from rune.engine.graph import step_node
from rune.engine.state import Feedback, Subtask
from rune.model.adapter import AdapterResult
from rune.model.inference import GenerationResult


def _make_state(*, code: str = "", exit_code: int = 0) -> dict[str, Any]:
    subtask = Subtask(name="_main", description="Write a LinkedList", depends_on=[])
    fb = Feedback(stdout="", stderr="", exit_code=exit_code) if code else None
    return {
        "task": "Write a class LinkedList with methods append, prepend",
        "subtasks": [subtask],
        "interfaces": {},
        "plans": {"_main": "Write a LinkedList"},
        "code_results": {"_main": code} if code else {},
        "code_passed": {"_main": exit_code == 0} if code else {},
        "retries": {},
        "integrated_code": "",
        "current_adapter": None,
        "feedback": {"_main": fb} if fb else {},
        "integration_feedback": None,
        "diagnosis": {},
        "actions": [],
        "trajectory": [],
        "step": 0,
        "budget_remaining": 5,
    }


class TestStepNodeContinuation:
    def test_continuation_uses_generate_continuation(self) -> None:
        """Verify that truncated code triggers the prefill+continue path."""
        state = _make_state()
        truncated_json = json.dumps({"code": "class Node:\n    def __init__(self):\n"})
        initial_result = GenerationResult(
            text=truncated_json, thinking="", tokens_used=100, truncated=True,
        )
        # Continuation produces the rest of the code
        cont_result = GenerationResult(
            text="        self.data = None\n\nclass LinkedList:\n    pass\n",
            thinking="", tokens_used=50, truncated=False,
        )
        model = MagicMock()
        model.generate_adapter.return_value = AdapterResult(
            adapter_id="test123", state_dict={},
        )
        model.hotswap_adapter = MagicMock()
        model.generate = AsyncMock(return_value=initial_result)
        model.generate_continuation = AsyncMock(return_value=cont_result)

        config = {
            "configurable": {
                "model": model,
                "run_config": {
                    "max_tokens": 512,
                    "cont_budget": 3,
                    "cont_multiplier": 1.5,
                    "no_repeat_ngram_size": 12,
                },
            },
        }

        with patch("rune.engine.graph.run_in_sandbox") as mock_sandbox:
            mock_sandbox.return_value = MagicMock(
                stdout="", stderr="", exit_code=0,
            )
            result = asyncio.run(step_node(state, config))

        # generate_continuation should have been called (not the old path)
        model.generate_continuation.assert_called()
        call_kwargs = model.generate_continuation.call_args.kwargs
        # The assistant_prefix should be the extracted code from initial generation
        assert "class Node:" in call_kwargs["assistant_prefix"]
        assert call_kwargs["system_prompt"].startswith("Output only Python code")

    def test_continuation_exits_early_on_valid_syntax(self) -> None:
        """Verify that the loop exits when accumulated code compiles."""
        state = _make_state()
        truncated_json = json.dumps({"code": "class Node:\n    pass\n"})
        initial_result = GenerationResult(
            text=truncated_json, thinking="", tokens_used=100, truncated=True,
        )
        # First continuation round produces valid code
        cont_result = GenerationResult(
            text="\nclass LinkedList:\n    pass\n",
            thinking="", tokens_used=30, truncated=True,  # truncated but code is valid
        )
        model = MagicMock()
        model.generate_adapter.return_value = AdapterResult(
            adapter_id="test456", state_dict={},
        )
        model.hotswap_adapter = MagicMock()
        model.generate = AsyncMock(return_value=initial_result)
        model.generate_continuation = AsyncMock(return_value=cont_result)

        config = {
            "configurable": {
                "model": model,
                "run_config": {
                    "max_tokens": 512,
                    "cont_budget": 5,
                    "cont_multiplier": 1.5,
                    "no_repeat_ngram_size": 12,
                },
            },
        }

        with patch("rune.engine.graph.run_in_sandbox") as mock_sandbox:
            mock_sandbox.return_value = MagicMock(
                stdout="", stderr="", exit_code=0,
            )
            result = asyncio.run(step_node(state, config))

        # Only 1 continuation call — exited early because code compiled
        assert model.generate_continuation.call_count == 1
```

- [ ] **Step 2: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_graph_continuation.py -v`
Expected: PASS (this test is written after the graph.py changes, so it should pass)

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_graph_continuation.py
git commit -m "test: add unit tests for prefill+continue continuation path in step_node"
```

---

### Task 5: Fix smoke test budget tracking

The budget analysis in `smoke_test_engine.py` always reports 0 continuation rounds because `cont_budget_spent` doesn't propagate outside `step_node`. Fix it to log actual continuation round counts from the engine logs.

**Files:**
- Modify: `tools/smoke_test_engine.py:150-164`

- [ ] **Step 1: Replace the broken budget analysis section**

Replace lines 150-164 in `tools/smoke_test_engine.py`:

Old:
```python
    # --- Budget analysis ---
    budget_spent = initial_state["budget_remaining"] - final_state["budget_remaining"]
    n_actions = len(final_state.get("trajectory", []))
    cont_budget = budget_spent - final_state["step"]
    print("=== BUDGET ANALYSIS ===", flush=True)
    print(f"  Actions recorded: {n_actions}", flush=True)
    print(f"  Steps (outer loop): {final_state['step']}", flush=True)
    print(f"  Budget spent: {budget_spent}", flush=True)
    print(f"  Continuation budget (budget_spent - steps): {cont_budget}", flush=True)
    if cont_budget > 0:
        print("  PASS: continuation consumed extra budget", flush=True)
    else:
        print("  INFO: no extra continuation budget consumed "
              "(truncation may not have triggered)", flush=True)
```

New:
```python
    # --- Budget analysis ---
    budget_spent = initial_state["budget_remaining"] - final_state["budget_remaining"]
    n_actions = len(final_state.get("trajectory", []))
    print("=== BUDGET ANALYSIS ===", flush=True)
    print(f"  Actions recorded: {n_actions}", flush=True)
    print(f"  Steps (outer loop): {final_state['step']}", flush=True)
    print(f"  Budget spent: {budget_spent}", flush=True)
    print(flush=True)
    print("  NOTE: Continuation rounds are internal to step_node and do not", flush=True)
    print("  consume outer budget. Check engine logs for 'continuation round'", flush=True)
    print("  messages to verify continuation fired.", flush=True)
```

- [ ] **Step 2: Commit**

```bash
git add tools/smoke_test_engine.py
git commit -m "fix: correct misleading budget analysis in smoke test

Continuation rounds are internal to step_node and don't consume
outer budget. Updated to direct users to engine logs instead."
```

---

### Task 6: Delete `prompt_code_continue.j2`

The model prompt template for continuation is no longer used — the continuation prompt is now built programmatically via `generate_continuation()`.

**Files:**
- Delete: `src/rune/templates/prompt_code_continue.j2`

- [ ] **Step 1: Verify no remaining references**

```bash
grep -rn 'prompt_code_continue' --include='*.py' --include='*.j2' .
```

Expected: no output (the graph.py reference was removed in Task 3).

- [ ] **Step 2: Delete the template**

```bash
rm src/rune/templates/prompt_code_continue.j2
```

- [ ] **Step 3: Run full test suite**

Run: `uv run pytest tests/unit/ -v`
Expected: all PASS

- [ ] **Step 4: Run lint + type check**

Run: `uv run ruff check . && uv run mypy src/`
Expected: clean

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore: delete prompt_code_continue.j2, replaced by generate_continuation()"
```

---

### Task 7: GPU smoke test

Run the full smoke test to verify the new continuation loop works end-to-end with a real model.

**Files:**
- None (verification only)

- [ ] **Step 1: Ask user to run the smoke test**

Run: `uv run python tools/smoke_test_engine.py`

- [ ] **Step 2: Verify results**

Check the output for:

1. **Continuation fired**: Look for `"continuation round: +N chars, degen=X.XX"` log messages.
2. **No indentation bugs**: No `SyntaxError: 'return' outside function` in trajectory errors.
3. **Early exit on valid syntax**: Look for `"Accumulated code validates — exiting continuation"` log message.
4. **Fewer total steps**: Target ≤5 steps (baseline was 7 with the old approach).
5. **Final code passes**: Integration step exits with `exit=0`.

If continuation doesn't fire (no log messages), check that `max_tokens=512` is low enough to trigger truncation. If indentation bugs persist, check the `apply_chat_template` output from Task 0.
