# Adversarial Jinja2 Template Analysis — Synthesized Report

## Scope

- **13 flat `.j2` files** in `src/rune/templates/`
- **Single renderer:** `render_template()` in `parse.py` with `PackageLoader`, `StrictUndefined`, no autoescape, no sandbox
- **Dual-channel architecture:** trajectory templates → hypernetwork; `prompt_*` templates → LLM user message; system prompts in `policy.py`; continuation prompts hardcoded in `graph.py`

---

## Verdict

The foundation is sound for a trusted, file-based agent: static templates, singleton `Environment`, no `from_string`, `StrictUndefined`. The **prompt layer is the least engineered surface in the repo** relative to its stated importance (pass@1, adapter-as-memory). The worst issues are not classic Jinja SSTI — they are **semantic drift, untested branches, and a format schizophrenia** where trajectories describe one output shape while xgrammar enforces another.

---

## Critical Findings (P0)

### 1. Format schizophrenia: trajectories lie to the hypernetwork

`decompose.j2` and `prompt_decompose_concise.j2` instruct the model to emit a **numbered list**:

```10:22:src/rune/templates/decompose.j2
Output ONLY a numbered list of subtasks. ...
Output a numbered list with dependency declarations:
1. name — one-line description [depends: none]
```

Production uses `DecomposeResult` JSON via xgrammar. The hypernetwork is conditioned on "numbered list"; the model is forced to emit `{"subtasks":[...]}`. Same pattern likely affects plan/code trajectories vs their Pydantic schemas.

**Adversarial take:** You built an adapter-as-memory system, then fed it instructions that contradict the actual output contract. Any benchmark lift attributed to prompt tuning is confounded — you may be optimizing the wrong signal.

### 2. Training data ≠ engine renders

Mining uses `_render_trajectory()` string concatenation (`Input:` / `Output:`), not Jinja. Probe tools (`cont_probe.py`, `adapter_probe.py`) duplicate template logic in Python strings. HPO on probes may not transfer to production.

**Adversarial take:** The hypernetwork learns from one distribution and runs on another. Template edits during prompt iteration may not improve (or may harm) adapter quality even when pass@1 moves.

### 3. Continuation is half-migrated

Design specifies `code_continue.j2` + `prompt_code_continue.j2`. Only the trajectory template exists. Continuation user/system prompts are hardcoded Python strings. `resume_tail` is computed in `graph.py` but never referenced in the template — docs claim it is used.

**Adversarial take:** The one path that handles truncation recovery — where output quality matters most — is the least templated and least testable.

---

## High Findings (P1)

### 4. Prompt injection, not SSTI, is the real threat

Classic SSTI/RCE is **not realistically exploitable**: no user-authored template syntax, no `|safe`, no `from_string`. The adversarial surface is **prompt/context injection**:

| Source | Fields | Risk |
|--------|--------|------|
| User/benchmark task | `project`, `task_description`, `project_label` | Direct instruction override |
| Model outputs | subtask names/descriptions, plans, code | Indirect injection in loops |
| Sandbox stderr | `error_summary`, `repair_history` | Feedback-loop poisoning |

`prompt_decompose_concise.j2` passes **unbounded** `task_description`. `integration_doc` has no cap on subtask count or description length. Model-controlled `target_subtask` is embedded inside JSON examples in `prompt_diagnose.j2`.

Industry best practice ([OWASP LLM Prompt Injection](https://cheatsheetseries.owasp.org/cheatsheets/LLM_Prompt_Injection_Prevention_Cheat_Sheet.html)): delimit user data, separate trusted system instructions from untrusted content at the API layer. Templates currently treat everything as one flat string.

### 5. Test coverage is effectively zero

Only **2 of 13 templates** have render tests (`decompose`, untargeted `prompt_diagnose`). No tests for `plan`, `code`, `repair`, `integrate`, `code_continue`, or targeted diagnose branches. `test_state.py` references deleted `prompt_decompose`.

With `StrictUndefined`, any new template variable without a matching `state_to_ctx` key **crashes mid-run** — not at CI time.

Industry standard: parametrized snapshot tests per template × minimal fixture context, branch coverage for `{% if target_subtask %}`, static analysis via `jinja2.meta.find_undeclared_variables()`.

### 6. No composition, heavy duplication

Zero `extends`, `include`, or `macro`. Repeated blocks across files:

- PROJECT headers (`code.j2`, `code_repair.j2`, `code_continue.j2`)
- Skeleton loops with **different slice caps** (400 vs 1000)
- Repair history loops with inconsistent formatting
- Diagnose JSON schema duplicated in trajectory + prompt with **different error_type enums**

**Adversarial take:** 13 files will become 26 before anyone extracts a macro. Every edit is a grep-and-hope operation.

---

## Medium Findings (P2)

### 7. Triple naming for the same task string

`state_to_ctx` exposes `project` (full task), `task_description` (full task), and `project_label` (200-char slice). Templates apply their own slices on top (`[:300]`, `[:800]`, `[:1200]`). Truncation policy lives in three places: Python ctx, template slices, and design docs — and they disagree.

### 8. Four instruction channels, no single owner

| Channel | Location |
|---------|----------|
| Trajectory | `*.j2` |
| User prompt | `prompt_*.j2` |
| System role | `policy.py` `Action.system_prompt` |
| Continuation | `graph.py` hardcoded strings |

Prompt engineers must edit 2–3 files + Python to change one action's behavior. No versioning, no A/B hooks, HPO ignores templates entirely.

### 9. Untyped context dict

`state_to_ctx` returns `dict[str, Any]`. Templates assume dataclass shapes (`subtask.name`), duplicate keys (`skeletons` = `code_outputs`), and ad-hoc structures (`code_trajectory[]` with `step`, `action`, `code`, `error`, `passed`). No `TemplateContext` TypedDict or contract validation.

### 10. Whitespace/token waste

No `trim_blocks` or `lstrip_blocks`. Conditional blocks on separate lines inject blank lines into trajectory text fed to the hypernetwork. At 2048 token cap, wasted whitespace is wasted conditioning signal.

### 11. Stale docs and dead artifacts

- `prompt_code_continue.j2` specified, never created
- `dependency_interfaces` / `interfaces` in design docs, never wired
- `resume_tail` computed, unused
- Language-agnostic goal vs Python-specific continuation system prompt

---

## What Is Actually Good

| Practice | Status |
|----------|--------|
| Singleton `Environment` | ✅ |
| `StrictUndefined` | ✅ |
| `PackageLoader` (static files only) | ✅ |
| Dual trajectory/prompt split | ✅ (right idea) |
| `Action` dataclass binds template names | ✅ |
| MLflow logs rendered text per step | ✅ (manual audit trail) |
| No `\|safe`, no `from_string` | ✅ |

---

## Prioritized Recommendations (advisory only)

| Priority | Action | Rationale |
|----------|--------|-----------|
| **P0** | Align trajectory + prompt copy with Pydantic/xgrammar schemas | Fixes adapter-as-memory lie |
| **P0** | Complete continuation (`prompt_code_continue.j2`) or document Python-only ownership | Closes half-migrated path |
| **P0** | Align mining/probes with Jinja renders (or store rendered ctx in session logs) | Fixes train/serve skew |
| **P1** | Parametrized snapshot tests for all 13 templates × branches | Catches `StrictUndefined` crashes in CI |
| **P1** | Typed `TemplateContext` + single truncation layer in `state_to_ctx` | Eliminates triple-naming and slice drift |
| **P1** | Delimit user task; cap `task_description` and `integration_doc` | Prompt injection mitigation |
| **P2** | Base layout + macros for headers, schema snippets, loops | DRY without over-abstracting |
| **P2** | `prompt_version` in `PipelineConfig` + git hash in MLflow | Reproducibility (PRODUCT.md invariant) |
| **P2** | Enable `trim_blocks=True, lstrip_blocks=True` | Token budget |
| **P2** | CI grep: no `from_string`, template names match `ACTIONS` allowlist | Future SSTI prevention |
| **Defer** | `SandboxedEnvironment`, autoescape, bytecode cache | Not warranted at current scale/trust model |

---

## Bottom Line

Jinja2 is the right tool here — better than f-strings for conditionals/loops, better than Mustache for composition needs, safer than Mako for a prompt library. The problem is not the engine choice; it is that **templates are treated as copy-paste strings rather than a versioned, tested, typed prompt product surface**. For a repo whose north star is pass@1 via hypernetwork conditioning, that is disproportionate risk: you can HPO temperature while the adapter reads "numbered list" and the grammar forces JSON.

---

## Scorecard

| Area | Grade | Biggest gap |
|------|-------|-------------|
| SSTI prevention | A- | Document static-template invariant |
| Prompt injection | C | Unbounded user content in trajectories |
| Organization | B | Flat dir, no versioning |
| Composition/DRY | D | No macros/inheritance |
| Schema alignment | F | Trajectory contradicts xgrammar |
| Train/serve alignment | D | Mining ≠ Jinja renders |
| Testing | D | 2/13 templates tested |
| Performance | B+ | Fine at agent scale |