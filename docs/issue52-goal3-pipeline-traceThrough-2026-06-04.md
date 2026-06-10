# GOAL-3 engine pipeline trace-through + miswiring audit — 2026-06-04

Owner asked for a deep trace-through of the generation→extraction→parse→policy pipeline to catch
miswirings preemptively, after the v1 smoke ran pathologically slow (one task >15 min, no progress).
This documents the full path, the three confirmed miswirings (with evidence), the fixes, and the
checks that ruled other stages clean. All fixes are minimal, unit-tested, **not committed** (await
owner go); they are robustness fixes in the shared engine path and affect **all arms equally**.

## The pipeline (one `code` action, single `_main` MBPP subtask)

1. `select_action(state)` → `code(target=_main)` (policy.py). MBPP trips `_is_simple_task` ⇒ single
   `_main`, decompose/plan skipped.
2. `state_to_ctx` → ctx: `project_label = task[:200]`, `fix_guidance`, `error_summary`, etc.
3. `render_training_format_trajectory(task, existing_code, feedback)` → adapter conditioning text.
4. `render_template("prompt_code", **ctx)` → minimal prompt (note: **contains `project_label`=task**).
5. `model.generate_adapter(traj)` → LoRA; `scale_lora_b(sd, adapter_scaling)`; `hotswap_adapter`.
6. `model.generate(prompt, system, output_schema=CodeResult, …)` (inference.py):
   - Phase 1 thinking: unconstrained, `max_new_tokens=thinking_budget(1024)`, `eos=</think>`.
   - Phase 2: **xgrammar-constrained** to the `CodeResult` JSON schema (LogitsProcessor, line 178).
     ⇒ `result.text` is **valid JSON** `{"code": "…"}`. (Confirmed by a live raw dump.)
7. `extract_partial_code(result.text)` → `extract_code_from_raw(…, CodeResult, fallback_to_raw=True)`
   → `model_validate_json(raw).code` (pydantic). 
8. `run_in_sandbox(strip_self_tests(code))` → feedback (exit_code).
9. `parse_output(action, raw, feedback, state)` → `code_passed`, `code_results`, `retries`.
10. loop: fail ⇒ `diagnose` → `repair` (≤ MAX_REPAIRS=2, MAX_RETRIES=4), else done/integrate.

## Miswiring 1 — markdown fence INSIDE the JSON code value → spurious sandbox failure  [FIXED]

**Evidence (live raw dump, scale=0 arm):** `result.text == '{"code": "```py\\n# …"}'` — valid JSON,
`truncated=False`. The grammar + pydantic are working; the model wraps its code in a ```` ```py ````
fence **inside** the `code` string. `extract_partial_code` returned `` ```py\ndef … ``` `` verbatim;
the sandbox then raised `SyntaxError: invalid syntax` on line 1 (the ```` ```py ```` line).
**Impact:** logically-correct solutions fail spuriously (mbpp/106's `return tpl + tuple(lst)` is
correct — it would PASS de-fenced), polluting the "fails-attempt-1" slice and triggering needless
repair loops. This was the dominant cost + validity bug. `strip_self_tests` can't save it — it
`ast.parse`s, fails on the fence, and returns the code unchanged, so the fence must be removed at
extraction.
**Fix (`parse.py`) — maintained libraries, no fragile regex (owner directive).** `extract_code_from_raw`
is now a three-stage pipeline: **`json-repair`** robustly parses the model's JSON (repairing
truncation / prose-wrapping) → **pydantic** `model_validate` is the structural contract → **`markdown-it-py`**
(`_extract_code_block`, the CommonMark reference parser used by rich/mkdocs/jupyter) extracts the code
from the possibly-fenced `code` value. A non-pydantic path logs loudly (a raw fallback is a signal,
not silent). This **replaces** both the regex *and* the custom `src/rune/engine/json_repair.py`
(deleted — superseded by the `json-repair` package). Tests:
`test_fenced_code_inside_json_value_is_stripped`, `test_extract_code_block_variants`.

### Library research (owner asked: use known packages, don't reinvent / no fragile regex)
Researched + **empirically tested on the real captured strings** (`uv run --with`, CPU):
- **json-repair** (MIT, v0.60.1, active) — extracts JSON from prose, repairs truncation, `''` on
  non-JSON. **Adopted** (JSON layer).
- **markdown-it-py** (CommonMark ref parser; already present via rich) — handles fenced / comment-
  prefixed / **truncated (unclosed fence → EOF)** / **clean (passthrough)**; fence-only match avoids
  mis-extracting indented Python bodies. **Adopted** (code-block layer).
- **parse_llm_code** (3★, 2024) — `extract_first_code` returns **`None` on clean *and* truncated code**
  ⇒ would drop correct solutions. **Rejected** (empirically fragile for our cases).
- **llm-output-parser** — only `parse_json`/`parse_xml`, not a code-block extractor. Wrong tool.
- **instructor** — API/Ollama/vLLM clients only, no raw-HF support, no standalone parser. **Doesn't
  fit** the local xgrammar+HF path.
- **outlines** — generation-time constraint for transformers (peer of xgrammar); constrains JSON
  *structure* but **cannot prevent fences inside a free `code` string field**, so it wouldn't fix this
  bug, and xgrammar already works. **Not adopted** (orthogonal to the bug).

## Miswiring 2 — targeted diagnose + hallucinated subtask_name → diagnose livelock  [FIXED]

**Evidence:** mbpp/106 session = `code(exit=1)` then **`diagnose` ×9** (all remaining budget), never a
`repair`. The diagnose output named `subtask_name: "write_function"` though the prompt said `"_main"`.
`parse_output` wrote `diagnosis["write_function"]`; `select_action` checks `"_main" in diagnosis`
→ always False → re-`diagnose` forever; `retries` never increments (repair never runs) so the
exhaustion path never triggers → livelock until `budget_remaining<=0` (10 wasted generations/task).
The existing hallucination fallback only covered the **untargeted** (integration) diagnose
(`target_subtask is None`).
**Fix (`parse.py`):** for a **targeted** diagnose, if the target isn't among the emitted entries,
attach the combined guidance to the target and reopen it — there is exactly one target, so naming is
irrelevant. After this, `code→diagnose→repair` proceeds and terminates at MAX_REPAIRS/MAX_RETRIES.
Test: `test_targeted_diagnose_hallucinated_name_attaches_to_target`.

## Miswiring 3 — experimentation not logged to MLflow  [FIXED]

The v1 driver made MLflow optional (`--mlflow`), and the smoke ran without it ⇒ no run to inspect.
**Fix (`tools/_goal3_multiturn_probe.py`):** `run` now **always** `configure_mlflow` + `tracked_run`
+ `log_dataset(pool)` + params (arm, checkpoint, adapter_scaling, seed, pool sha256, n_tasks) +
pass@1 metrics. Under the active run, the engine step auto-logs per-turn trajectory/prompt/output
artifacts + adapter-cond/prompt token metrics (the GOAL-3 (g) hook). MLflow server is up at
`http://localhost:5000` (verified; prior issue52 experiments visible).

## Stages checked and ruled clean (preemptive)

- **xgrammar structured output:** working — raw is valid JSON; pydantic parses. Not the bug.
- **Continuation sub-loop (graph.py, cont_budget=5):** fires only on `result.truncated`; MBPP code is
  short (`truncated=False`), so it does **not** fire. The earlier "continuation explosion" hypothesis
  was **wrong** — confirmed by short per-step output lengths.
- **EOS / stop tokens:** recognized — outputs end well under `max_tokens` (97–1302 chars). Not the bug.
- **scale=0 grammar:** `output_schema` is passed regardless of adapter, so scale=0 also emits valid
  JSON ⇒ the fence/diagnose bugs hit all arms equally (fair).
- **Thinking phase (1024 tokens/action):** real cost, not a bug; faithful runner behavior. Left as-is
  (changing it is a config deviation that would confound faithfulness — owner steer).
- **`_strip_self_tests`:** removes module-level asserts/test defs; correct once code is de-fenced.

## Consequence for the experiment

Miswiring 1 (not 2) was the dominant cost: most "failures" were fence artifacts. Post-fix, attempt-1
passes skip the repair loop entirely and genuine failures run ~3 actions, so per-task wall-clock
should drop from ~15–20 min to minutes. **Re-measure steady-state wall-clock on the fixed engine
before sizing the batch** (advisor: reduce pool, never budget; keep the attempt-1-fail slice ≥40).
