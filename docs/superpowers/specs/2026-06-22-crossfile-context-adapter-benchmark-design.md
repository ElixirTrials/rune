# Cross-file-context-as-adapter benchmark — design (2026-06-22)

> **RESULT (executed).** Conjecture supported via an **episodic per-task** template under an
> imposed window budget W=768 (the spec did not anticipate the budget — at the base's full 262k
> window the question is vacuous): episodic 31/60 (0.517) vs floor 9/60 (0.150), McNemar p=3.0e-06,
> statistically indistinguishable from the full-context ceiling (a2_full 17/30=0.567) at ~16.7×
> shorter prompt. No adapter weights trained (frozen c3); best config `variant=use, anchor=0,
> scaling=0.91`. The earlier multi-file *dump* template was a negative (superseded). Arms were
> renamed in execution (floor/a2_clamp/a2_full/episodic). See
> [issue52-results-longcontext-2026-06-22.md](../../issue52-results-longcontext-2026-06-22.md),
> [issue52-repobench-template-hpo-findings-2026-06-22.md](../../issue52-repobench-template-hpo-findings-2026-06-22.md),
> [issue52-repobench-clamp-findings-2026-06-22.md](../../issue52-repobench-clamp-findings-2026-06-22.md).

> **Conjecture under test.** Rune's real advantage is on tasks that are *impossible for a small
> model without sufficient context length*: the hypernetwork compresses long cross-file context
> into a constant-length LoRA adapter, so rune solves cross-file completions that the base model
> fails when that context does not fit (or degrades) in its prompt. This is PRODUCT.md's **current
> bet** ("adapters … provide unbounded context at constant prompt length") and **JTBD #3**, neither
> of which LCB/HumanEval+ (short, self-contained) can ever exercise.

## 1. Why this benchmark, and what it adds

PR #57 established a *difficulty-dependent* result on a **two-point line** (LCB-v6 hard: c3 +4
superset; HumanEval+ easy: c3 −16) where difficulty is confounded with source, year/contamination,
and complexity. None of those benchmarks vary **usable context length**, which is the axis the
conjecture is actually about. This benchmark makes context length the *measured independent
variable* and asks a sharp, falsifiable question: **does putting the needed context in the adapter
(constant short prompt) match or beat putting it in the prompt, especially as that context grows?**

## 2. Benchmark: RepoBench v1.1 (Python), `cross_file_first`

- **Dataset:** `tianyang/repobench_python_v1.1` (ICLR 2024), HF-native parquet, `load_dataset`-ready
  (mirrors the repo's existing `hf.load_dataset` pattern in `tools/_he_run.py:28`,
  `src/rune/bench/mbpp.py:77`). License `cc`. **Deduplicated against Stack v2** to mitigate
  memorization/leakage — aligns with the team's contamination/pointer rigor.
- **Split:** `cross_file_first` (8.0K rows). The next line uses a cross-file API **for the first
  time**, so the task is *impossible without the cross-file snippet by construction* — exactly the
  "insufficient context length ⇒ fails" criterion, guaranteed by the dataset rather than asserted.
- **Schema → design mapping** (verified via HF dataset viewer):

  | RepoBench field | type | role in this design |
  |---|---|---|
  | `cropped_code` | str | the **in-file prompt** (context up to the cursor) |
  | `context` | List[{`identifier`,`path`,`snippet`}] | the **cross-file context** (candidate snippets) |
  | `gold_snippet_index` | int | index into `context` of the snippet that defines the needed API → the **oracle dependency** for all arms, the Phase-0 single-dependency probe, and the mutation control |
  | `next_line` | str | **ground truth** to score (EM / edit-sim / identifier recovery) |
  | `import_statement` | str | the import binding the cross-file name (compressed-template input) |
  | `level` | str | RepoBench's built-in context-length bucket (≈2k–24k tokens) → **the crossover x-axis, for free** |
  | `token_num` | int | exact context token count (continuous axis / stratification) |
  | `repo_name`,`file_path` | str | provenance / dedup keys |

- **Control split:** `in_file` (7.9K rows) — completion needs only in-file context. Injecting
  cross-file context there should **not** help → a built-in **specificity control** (§7).

## 3. Arms — only the *delivery of context* varies

All arms share the same base model (`config.yaml` / `RUNE_BASE_MODEL`), the same `cropped_code`
in-file prompt, the same decoding params, and the same scorer. Each arm is a **single constrained
completion** (RepoBench is next-line prediction — the agentic decompose→repair loop does **not**
apply; we use the surgical `generate_adapter`/`hotswap`/`generate` path, mirroring the single-shot
`base` arm in `tools/_he_run.py:_gen_base`).

| Arm | In-file prompt | Cross-file context delivered via | Prompt tokens | Path |
|---|---|---|---|---|
| **A1** base / no-context | `cropped_code` | — (none) | short | `reset_adapter()` → `generate(prompt)` |
| **A2** base / context-in-prompt | `cropped_code` | full `context` (all snippets) prepended into the **prompt** | **grows with context** | `reset_adapter()` → `generate(context⊕prompt)` |
| **A3** rune / context-in-adapter | `cropped_code` | full `context` (all snippets) encoded into the **LoRA adapter** | **short, constant (= A1)** | `generate_adapter(render_xfile_adapter(row))` → `hotswap_adapter(scale·sd)` → `generate(prompt)` |
| *A4* base / truncated context *(optional)* | `cropped_code` | first K tokens of context in prompt | capped | simulates a smaller window |

In Phase 1, **A2 and A3 receive the same full `context`** (all snippets — this is what scales with
`level`); the gold-snippet-only condition is reserved for the Phase-0 probe (§8) and the mutation
control (§7), so the length axis is honest and the A2/A3 comparison is apples-to-apples.

Concrete A3 (per task), all calls already on `ModelWrapper` (`src/rune/model/wrapper.py`):

```python
ar = model.generate_adapter(render_xfile_adapter(row, mode))      # :153  text → LoRA state_dict
model.hotswap_adapter(scale_lora_b(ar.state_dict, scaling))       # :190  scaling = template-tuning knob
gen = await model.generate(prompt=cropped_code_prompt, max_tokens=64, temperature=0.0, ...)  # :205
pred_line = first_nonblank_line(gen.text)
```

`model.count_tokens(...)` (`wrapper.py:179`, built for *exactly this thesis* — "the prompt stays
~flat while the adapter's conditioning grows") logs prompt tokens per arm, making the **constant
prompt length** claim quantitative (expect A3 ≈ A1 ≪ A2).

## 4. Independent variable & the predicted signature

Plot the primary metric vs. RepoBench `level` (and continuously vs. `token_num`), one curve per arm.
**The conjecture predicts the crossover:**

- **A1** (no context): flat-low across all lengths — confirms the task needs cross-file context.
- **A2** (context-in-prompt): high at short context, **degrades** as length → the 4B model's
  *effective* window (lost-in-the-middle), at growing prompt cost.
- **A3** (context-in-adapter): **flat-high at constant prompt length**, crossing **above A2 in the
  long bins**.

```
metric ^                         A3 (adapter, constant prompt)
       |  A2────────A2___                ___________________
       |  ╱              ╲___A2      ____╱   A3
       | ╱  A3 ≈≥ A2          ╲_____╱   ← crossover (the claim)
       |╱_______________________A2_________________________
       |  A1 ───────────────────────────────────── (floor: needs context)
       +-------------------------------------------------> context length (level / token_num)
```

**Falsification (pre-registered):**
- If **A3 ≈ A1** at all lengths → the adapter injected nothing → conjecture **not supported** (report honestly).
- If **A3 < A2** everywhere and A2 never degrades → the adapter is a worse context channel than the
  prompt → conjecture **not supported** in its strong form. Fall back to the **weak form** below.
- **Weak form (still supports the bet):** if A3 ≈ A2 in accuracy but at A1's prompt length, that is
  *parity at constant prompt length* — the JTBD#3 throughput/KV-memory win (constant prompt ⇒
  bounded latency/memory regardless of context size), even without an accuracy crossover.

## 5. Metric — context recovery, no sandbox

- **Primary — gold cross-file identifier recovery rate:** fraction of tasks whose predicted line
  contains `context[gold_snippet_index].identifier`. This *directly* measures "did the injected
  context surface the needed cross-file API," tied to the dataset's own gold label.
- **Secondary — identifier-F1:** precision/recall/F1 over the identifier multiset of `pred` vs
  `next_line`. Identifiers extracted via Python `tokenize` `NAME` tokens minus keywords/builtins
  (fallback regex `\b[A-Za-z_]\w*\b`); reuse the AST/tokenize tooling already in `src/rune/bench/lcb.py`
  and `src/rune/engine/parse.py`. (Tree-sitter is available but `tokenize` suffices for single lines.)
- **Secondary — Exact Match & Edit Similarity** on `pred` vs `next_line` (RepoBench's native
  metrics, for comparability to the leaderboard).
- **No execution / sandbox** — deterministic, cheap, and sidesteps the OOM/grading fragility that
  cost two VM crashes in PR #57.

## 6. The bridge: zero-shot, template-tuned (no weight training)

The hypernetwork is **frozen** (no distillation). The only new "knob" is how cross-file context is
**rendered into the conditioning text** passed to `generate_adapter(trajectory_text=…)`. This extends
the existing **spec-in-adapter** precedent `render_reference_adapter` (`graph.py:454`, the
`reference_a/b/c` modes) — those already demonstrate "encode out-of-prompt content into the adapter,
name it from the prompt." A new sibling `render_xfile_adapter(row, mode)` with candidate modes:

- `xfile_raw` — gold snippet (or all `context` snippets) verbatim under `## Context`.
- `xfile_structured` — per snippet `## File {path}\n{snippet}` (mirrors reference headers).
- `xfile_signature` — `import_statement` + the gold definition's signature/skeleton only (compressed).
- `xfile_traj` — wrap as a pseudo-trajectory (closest to the hypernet's training distribution).

Plus the **adapter scaling** scalar (`scale_lora_b`, default from `config.yaml`; HPO range in
`benchmarks/bench.yaml`). Template + scaling are selected **once** on the Phase-0 probe (§8) and then
**frozen** for Phase-1 — the template is not re-tuned per length bin (that would leak the IV).

## 7. Controls & validity gates (house style)

1. **A3 ≫ A1 gate** — the analog of PR #57's adapter-off logit-Δ check. If context-in-adapter does
   not beat no-context, the adapter carried nothing; nothing downstream is interpretable.
2. **Mutation / pointer-vs-content control** — port PR #57 §8: rename the gold identifier in the
   injected snippet (and its `identifier` field) to a fresh name; A3 must **track the rename**
   (emit the new name ⇒ used the injected *content*) rather than reproduce the original
   (⇒ memorized *pointer*). Reported as content/pointer/other counts, exactly as the mutated-spec control.
3. **`in_file` specificity control** — run A3 on the `in_file` split; injecting cross-file context
   should **not** lift recovery there (and ideally not regress it). Guards against "the adapter is a
   generic accuracy booster" rather than a context channel.
4. **Prompt-length ledger** — `count_tokens` per arm per task; the constant-prompt claim is a
   reported number (A3 prompt tokens ≈ A1, independent of `token_num`), not an assertion.

## 8. Staging (probe → gate → curve)

**Phase 0 — single-dependency probe + template selection (cheap, hours).**
- Subset: `cross_file_first` rows where using only `context[gold_snippet_index]` is sufficient.
- For each candidate `render_xfile_adapter` mode × a small scaling grid: measure A3 vs A1 recovery.
- **Gate:** pick the (template, scaling) maximizing A3−A1. If **no** config clears A3 ≫ A1, stop and
  report the zero-shot negative (the frozen hypernet cannot compress repo context) — a publishable,
  honest bound, and a precise pointer to "distillation needed" as separate future work.

**Phase 1 — full crossover curve (frozen template).**
- Stratified, seed-fixed sample across `level` buckets (target **N ≈ 300**: ~60/bin × ~5 bins;
  expandable). Arms A1, A2, A3 (+A4 optional). Metrics §5 + the three controls §7.
- Cost: base arms = 1 generate/task; A3 adds one hypernet forward + hotswap/task. ≈300×(3 gens +
  1 adapter) — minutes-to-~1h on the 23GB GPU. `offload_base=False` (fits; avoids the CPU-RAM OOM
  per CLAUDE.md). Seed fixed; RNG seeded `seed+i` per task as in `runner.py:369`.

## 9. Durability (reuse the PR #57 harness pattern)

Reuse the `tools/_lcb_run.py` durability shape verbatim where possible: MLflow experiments
`repobench-xfile-probe` (Phase 0) and `repobench-xfile-curve` (Phase 1); each run logs
`cfg.to_dict()`, arm, dataset id+revision sha, split, `level`, checkpoint sha256, `engine_commit`
(git HEAD), seed, decoding params, **and** per-task artifacts: `{task_id, level, token_num,
prompt_tokens, pred_line, gold_line, gold_identifier, recovered, em, es}` JSONL + aggregate metrics
(`gold_id_recovery`, `identifier_f1`, `em`, `es`) **per arm per level bin**.

## 10. Implementation surface (no engine/sandbox/training changes)

New, additive only:
- `src/rune/bench/repobench.py` — loader: `load_dataset("tianyang/repobench_python_v1.1", split=…)`
  → typed rows; level/token_num accessors; gold-snippet + mutation helpers.
- `src/rune/bench/identifier_match.py` — `gold_id_recovery`, `identifier_f1`, `exact_match`,
  `edit_similarity` (pure functions, unit-tested on CPU).
- `render_xfile_adapter(row, mode)` — beside `render_reference_adapter` in `engine/graph.py`
  (or a small `engine/xfile_adapter.py` if it keeps `graph.py` from growing).
- `tools/_repobench_clamp_run.py` — the durable clamped-window harness (plus `tools/_repobench_template_hpo.py` for episodic-template selection).
- Unit tests (CPU): scorer correctness, loader shape, each `render_xfile_adapter` mode, mutation
  correctness. Reuses `ModelWrapper`, `scale_lora_b`, `count_tokens`, MLflow helpers unchanged.

**Invariants respected:** GPU imports stay deferred (PRODUCT.md inv. 2); adapter hot-swap is the
existing safe path (inv. 3); benchmark is durable/seeded/per-task-logged (inv. 1). No base-model
mutation, no training, no new model id (CLAUDE.md hard rule).

## 11. Risks & mitigations

- **Frozen hypernet under-compresses repo code (zero-shot).** *Mitigation:* Phase-0 gate localizes
  this before any large spend; template tuning (§6) is the lever; failure is reported, not hidden.
- **4B base may not visibly degrade in A2 within RepoBench's length range** (large nominal window).
  *Mitigation:* the weak-form claim (§4) still holds via the prompt-length ledger; push the longest
  `level` bins and the `--truncated` A4 arm to force a small-window regime.
- **Next-line granularity is narrow** (one line, not a function). *Mitigation:* gold-identifier
  recovery is well-defined at line granularity; EM/ES give continuous signal; a later v2 can score
  multi-line completion if warranted (out of scope here).
- **License `cc`** permits research eval; redistribution of derived data avoided (we log metrics +
  our own per-task predictions, not the corpus).

## 12. Out of scope (non-goals for v1)

CrossCodeEval (not HF-loadable; deferred as later robustness); hypernetwork distillation/training;
multi-line/function-level completion; the agentic decompose→repair loop (RepoBench is single-line);
non-Python languages; SWE-bench-style agentic repo editing.

## 13. Success criteria (pre-registered)

- **Strong (supports conjecture):** A3 ≫ A1 (gate passed), and A3 ≥ A2 in the longest ≥2 `level`
  bins at ≈A1 prompt length, with the mutation control showing content > pointer.
- **Weak (supports the bet, bounded):** A3 ≈ A2 accuracy at ≈A1 prompt length across bins
  (constant-prompt parity), even without an accuracy crossover.
- **Negative (refutes, honestly):** A3 ≈ A1 at all lengths after template tuning → frozen-hypernet
  zero-shot cannot compress repo context; report as a bound and a pointer to distillation as future work.
