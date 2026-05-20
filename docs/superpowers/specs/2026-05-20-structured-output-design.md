# Structured Output via XGrammar

**Date:** 2026-05-20
**Branch:** `feat/pipeline-speed-p4-p5`
**Status:** Design approved, pending GPU verification

## Problem Statement

Three parsing failures in the rune pipeline cause silent data loss and wasted GPU cycles:

1. **Decompose over-decomposition** — Regex parser `_parse_subtask_list` cannot enforce a subtask cap. Qwen3.5-9B produces 5–16 subtasks for single-function tasks, each triggering plan → code → retry cycles.
2. **Diagnose name mismatch** — Model invents new subtask names instead of referencing existing ones. Even with known names listed in the prompt, regex-based matching misses ~40% of diagnose outputs. Failed diagnosis aborts the repair loop.
3. **Fragile regex parsing** — `_parse_subtask_list` relies on `^\d+\.\s*(.+?)\s*(?:—|-|:)\s*(.+)$` which collides with `[depends: ...]` annotations and markdown bold, requiring ongoing patches.

## Approach

Replace regex parsing with **tokenizer-level constrained decoding** using XGrammar. The model structurally cannot produce invalid JSON — no post-hoc regex recovery needed.

- **Provider-level integration**: `TransformersProvider.generate()` gains an optional `json_schema: type[BaseModel] | None` parameter. When set, XGrammar compiles a `LogitsProcessor` from the schema's JSON schema and injects it into the generation call.
- **Phase-level dispatch**: `nodes.py` maps phase names to Pydantic schemas via a `_PHASE_SCHEMAS` dict. Only `decompose` and `diagnose` use structured output; code/plan/integrate remain free-form.
- **Existing models**: `shared.rune_models.DecomposeResult` and `Subtask` already exist with `max_length=8`. A new `DiagnoseResult` model will be added.
- **No regex fallback**: XGrammar is the only parsing path. The existing `_parse_subtask_list` regex parser is removed for decompose/diagnose phases. XGrammar supports CPU, CUDA, and Apple MPS — there is no environment where it won't run.

## Verification Gate

**XGrammar + Qwen3.5-9B must be verified before implementation.** XGrammar supports CPU, CUDA, and Apple MPS, so verification can run on a Mac (no GPU required).

A standalone verification script (`scripts/verify_xgrammar_nf4.py`) will:
1. Load Qwen3.5-9B (NF4 on CUDA if available, otherwise CPU/MPS)
2. Compile an XGrammar `LogitsProcessor` from a test JSON schema
3. Generate 5 completions, parse each as JSON
4. Verify all 5 parse successfully and match the schema

Run via: `uv run python scripts/verify_xgrammar_nf4.py`

## Design

### 1. Schemas (`libs/shared/src/shared/rune_models.py`)

`Subtask` and `DecomposeResult` already exist:

```python
class Subtask(BaseModel):
    name: str = Field(min_length=1)
    description: str = ""
    depends_on: list[str] = []

class DecomposeResult(BaseModel):
    subtasks: list[Subtask] = Field(min_length=1, max_length=8)
```

**No schema change.** The JSON schema keeps `max_length=8` — the grammar enforces structural validity, not business-logic limits. The runner applies a **post-parse truncation** to a configurable `max_subtasks` (default 4) after JSON parsing. This is a runner-layer concern, not a schema concern.

New model for diagnose:

```python
class DiagnoseItem(BaseModel):
    name: str = Field(min_length=1)
    diagnosis: str = Field(min_length=1)

class DiagnoseResult(BaseModel):
    repairs: list[DiagnoseItem] = Field(min_length=1, max_length=8)
```

**Dynamic name constraint**: For the diagnose phase, the runner knows the valid subtask names at call time. Rather than compiling a `Literal[*names]` into the schema (which would require recompilation per invocation), we keep `name: str` in the schema and validate post-parse using the existing `known_lower` substring matching from `_parse_diagnose_output`. The prompt already lists exact names and says "Use the EXACT subtask names above" — structured output ensures the model emits parseable JSON; the name matching ensures correctness.

### 2. Provider Changes (`libs/inference/src/inference/`)

**ABC** (`provider.py`): Add `json_schema: type[BaseModel] | None = None` to `generate()` signature. Default `None` preserves backward compatibility. Non-transformers providers ignore it.

**TransformersProvider** (`transformers_provider.py`):

XGrammar requires a three-step setup:
1. Create `TokenizerInfo` from the HF tokenizer via `xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=config.vocab_size)`.
2. Create a `GrammarCompiler` bound to that `TokenizerInfo`.
3. Compile the Pydantic schema via `compiler.compile_json_schema(SchemaClass)` — returns a `CompiledGrammar`.
4. Wrap the compiled grammar in `xgr.contrib.hf.LogitsProcessor(compiled_grammar)` and add to `gen_kwargs["logits_processor"]`.

The `GrammarCompiler` is created once per model (at `_load_model_if_needed` time) and cached on `self`. Compiled grammars are cached per schema class (`id(schema_class)` key) since compilation is non-trivial. XGrammar imports are deferred (INFRA-05 pattern) but XGrammar is a required dependency — import failure is a hard error.

### 3. Node Layer (`services/rune-agent/src/rune_agent/nodes.py`)

Add a phase → schema mapping:

```python
from shared.rune_models import DecomposeResult, DiagnoseResult

_PHASE_SCHEMAS: dict[str, type[BaseModel]] = {
    "decompose": DecomposeResult,
    "diagnose": DiagnoseResult,
}
```

In `generate_node`, look up the schema and pass it to the provider:

```python
schema = _PHASE_SCHEMAS.get(phase)
result = await provider.generate(..., json_schema=schema)
```

### 4. Runner Layer (`scripts/rune_runner.py`)

**Decompose parsing**: Replace `_parse_subtask_list` with `DecomposeResult.model_validate_json()`. The model output is guaranteed valid JSON by XGrammar — no regex needed.

**Diagnose parsing**: Replace `_parse_diagnose_output` regex logic with `DiagnoseResult.model_validate_json()`. Post-parse name matching against known subtasks stays (substring matching for abbreviated names).

**Subtask cap**: After parsing (JSON or regex), truncate to `max_subtasks` (default 4, configurable via `PipelineConfig`).

### 5. Template Changes

**`prompt_decompose.j2`** / **`prompt_decompose_concise.j2`**: Replace "Output ONLY a numbered list" with JSON format instructions:

```
Output ONLY valid JSON matching this schema:
{"subtasks": [{"name": "...", "description": "...", "depends_on": ["..."]}]}
```

**`prompt_diagnose.j2`**: Replace "Output ONLY a numbered list" with:

```
Output ONLY valid JSON matching this schema:
{"repairs": [{"name": "...", "diagnosis": "..."}]}
```

The templates still include the human-readable instructions (project label, known subtask names) — those guide the model's reasoning. Only the output format directive changes.

### 6. Adapter Interaction Risk

Adapters are trained on numbered-list trajectory outputs. When XGrammar forces JSON output format, the adapter's learned distribution may conflict with the grammar constraints. This could manifest as:
- Slower generation (many tokens rejected by the grammar)
- Degraded content quality (model fighting the format)

**Mitigation**: For decompose and diagnose phases, adapters carry domain context (project description, code snippets), not output format patterns. The format instruction is in the prompt, which changes from "numbered list" to "JSON". The adapter should not strongly encode output format since trajectories vary between phases.

**Monitoring**: Log `tokens_generated / wall_time` for grammar-constrained vs. unconstrained phases. If grammar-constrained phases show >3x slowdown, fall back to regex parsing for that phase.

### 7. Logging Clarity

The current HPO logs conflate two meanings of "pass": the pipeline's internal integration test (code runs without error) and MBPP's external evaluation (code matches expected output). This causes confusion when a trial scores 1.0 internally but 0.0 on MBPP validation.

**Changes:**
- `run_pipeline_on_problem` logs `"Pipeline completed (internal pass)"` or `"Pipeline completed (internal fail)"` — making clear this is the pipeline's own judgment.
- `evaluate_problem_set` logs the MBPP verdict separately: `"MBPP evaluation: PASS"` / `"MBPP evaluation: FAIL"`.
- The `ProblemVerdict` dataclass already has `passed: bool` — no schema change needed, just clearer log messages.

### 8. Acceptance Criteria

1. Decompose always produces valid `DecomposeResult` JSON (verified by `model_validate_json`)
2. Diagnose always produces valid `DiagnoseResult` JSON. Name matching against known subtasks is post-parse validation (unchanged from current substring matching) — JSON validity guarantees structure, not semantic correctness
3. No regression in HPO trial scores (0-shot, 3-problem smoke test)
4. XGrammar + NF4 verification script passes on L4 GPU
5. XGrammar is a required dependency — no regex fallback
6. Existing tests pass with no modification (except schema-related ones)
7. Pipeline completion logging clearly distinguishes "pipeline says done" from "MBPP evaluation pass" (see §7)

## Dependencies

- `xgrammar` Python package (pip-installable, Apache-2.0)
- GPU verification before implementation
- No changes to adapter training pipeline

## Non-Goals

- Structured output for code/plan/integrate phases (free-form text)
- MBPP evaluation integration into the pipeline (external evaluation only)
- Adapter retraining for JSON format (monitor first, retrain if needed)
- vLLM provider XGrammar support (vLLM has its own guided decoding; separate work)
