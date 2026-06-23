# Paper Evidence Map — `paper_v9.tex` ↔ measured artifacts

**Generated:** 2026-06-09 · **Paper:** `paper_v9.tex` ("Parametric Episodic Memory: A Third Axis for Language-Model Reasoning") — LaTeX source maintained outside this repo tree.
**Purpose:** for every quantitative claim, table, and gate in the paper, state **what evidence exists, where it lives, and what is still un-run** — so the manuscript can be filled honestly and reviewers' "what is actually established?" is answerable in one place.
**Companions:** [`mlflow-experiment-inventory-2026-06-09.md`](mlflow-experiment-inventory-2026-06-09.md) (system of record), [`issue52-experimentation-log.md`](issue52-experimentation-log.md) (full catalog), [`issue52-results-section-guide.md`](issue52-results-section-guide.md) (narrative framing).

> **The paper is pre-registration-style.** Its protocol (baselines, gates, statistics) is fixed; the numeric cells are placeholders to be pinned at camera-ready. This map records which placeholders we *can* fill, which need new runs, and three claims whose **current wording outruns the evidence**.

> **2026-06-22 update — two placeholders filled (durable, engine `efa7b9e`).** Full prose: [`issue52-results-longcontext-2026-06-22.md`](issue52-results-longcontext-2026-06-22.md).
> 1. **Constant-prompt / beyond-budget context delivery — the efficiency restructure §4.1 called for.** New benchmark: RepoBench v1.1 `cross_file_first` (cross-file completion) under a fixed prompt budget W=768 with the in-prompt baseline stressed past it. Frozen adapter recovers the gold cross-file symbol **31/60 (0.517 [0.393,0.638])** vs **floor 0.150**, **= full-context ceiling 0.567** at **16.7× shorter prompt**; at 32k the full-context prompt is prohibitive on 30/30 yet the adapter recovers 13/30; **McNemar 23:1, p=3.0e-6**. Control: a naïve multi-file **dump** template is a null (0.217≈floor, p=1.0) — the **episodic per-task** conditioning is the controlling variable (HPO-selected variant=use/anchor=0/scaling=0.91, held-out 4/10 vs 1/10 before the N=60 confirmation; no weight training). MLflow `issue52-repobench-clamp`, `issue52-repobench-template-hpo`. **Caveat:** metric is identifier-recovery, not pass@1; this is (v)-vs-context-channel, **not a Gate verdict**.
> 2. **Table 2 (i)/(v) now exist on HumanEval+, post grading-fix.** The earlier HE+ "−16" (c3 100 < base 116) was a **grading artifact** (harness dropped prompt imports → spurious NameError on 19 typing-signature tasks; untrusted escalation-floor discarded correct zero-shots). Corrected at `efa7b9e`: **base 134/164 (0.817)**, **c3 135/164 (0.823) — strict superset, +1, 0 regressions.** The "difficulty-dependent / hurts easy tasks" reading is **retracted** (it joins the LCB-arc as another reported-effect-dissolves-under-audit case). MLflow `issue52-humanevalplus` (`he-base-seed0`/`he-c3-seed0` @ `efa7b9e`; pre-fix runs preserved @ `db48504`/`5954c81`).

---

## 0. Three things to fix before camera-ready (ranked)

1. **`0.16×` / 200-trial / Gemma 2 2B → RESOLVED: re-anchored to Qwen (2026-06-09).** The Gemma 200-trial scaling study could not be located in any artifact reachable here (74 MLflow exps incl. run artifacts + local Optuna DB + all PR comments + both working trees); not present here ≠ never existed (may be external/pre-MLflow). The author chose to re-anchor rather than cite an unconfirmed store. `paper_v9.tex` now reports **0.627× generation-time scaling on Qwen3-4B-Instruct** (16-trial `rune-bench-hpo`). **Critical unit fix applied throughout:** `adapter_scaling` is a *multiplier* on the native `lora_alpha=45.25` (1.0× = 45.25), so 0.627× is a **mild ~0.6× attenuation**, not "Nx below" — the original "280× below" was unit-confused (dividing 45.25 by the multiplier). The structural conjecture is correspondingly **softened** (sub-unity but modest; 0.627 is outside the old 0.1–0.3× band, so B.8 reframed to "sub-unity"). Per author: the system's contribution is the **memory mechanism itself**, not the scaling magnitude — the paper now states this explicitly.
2. **Three-model muddle.** The paper names **Gemma 2 2B** (dev), **Qwen 2.5 Coder 7B** + **Qwen 3.5 9B** (production). Artifacts only support **Qwen 3.5 9B** (May `paper-table2`, sparse) and **Qwen 3-4B-Instruct** (June, the documented work). Qwen 2.5 Coder 7B and Gemma 2 2B have essentially no logged runs. Decide the canonical model line and align §4.1.
3. **Gate 1 has not been run as specified.** Gate 1 = Rune (v) vs **direct PEFT QLoRA on the same trajectories** (iii), McNemar, ≥5pp. We have never run condition (iii) against (v) on a shared held-out set. The LCB-49 result (rune 9/49 = base 9/49) is **(v) vs (i)**, not Gate 1. Do not present any current number as a Gate-1 verdict.

---

## 1. Claim-by-claim evidence status

Legend: ✅ backed (cite the home) · 🟡 partial / proxy only · ⛔ un-run / unlocatable · ✍️ wording outruns evidence.

### Abstract & §1 (Introduction)

| paper claim | status | evidence / gap |
|-------------|:------:|----------------|
| Mechanism: $H_\theta$ emits LoRA delta per step, reversible, constant per-step compute | ✅ | architectural; engine implements adapter hot-swap per step (`graph.py`) |
| scaling claim (re-anchored to ≈0.6–0.9× plateau on Qwen3-4B) | 🟡 | backed by `rune-bench-hpo` (exp 41), but single-regime (reference_a spec-in-adapter Pass@1 sweep) and a **flat tie** (0.627/0.673/0.685/0.815/0.921 all = 0.588) — report as a sub-unity plateau, not a sharp 0.627× optimum. Multiplier of native scale, not "Nx below" (see §0.1). |
| Pre-registered Gates 1–3 with fixed comparisons | ✅ (protocol) / ⛔ (results) | protocol fixed in paper; **no gate has a measured verdict** |
| SLM regime, single 24 GB GPU | ✅ | all runs on one consumer GPU; Phase-1 `gpu_peak_gb`=11.6 (exp 45) |

### §3.4 Adapter scaling / Table 1

| element | status | evidence / gap |
|---------|:------:|----------------|
| Table 1 optima (re-anchored) | 🟡 | now reports `rune-bench-hpo` (exp 41): scaling **≈0.6–0.9× plateau** (0.627× = lower edge), prompt mode **reference_a** (both searched); temp 0.3, presence 0.0, cont-mult 1.53, max-phase 4 (fixed). Objective = held-out MBPP Pass@1 (differs from the old caption's hunk-loss blend, which belongs to the distillation HPO — B.7 now distinguishes them). |
| "degenerate output at α≥1.0×" | 🟡 | directionally seen: `adapter-scaling-hpo` high-α trials score worse; degeneration root-caused to thinking-phase + presence_penalty, not α alone (`E-degen-ablation`) |
| override-vs-contextualise conjecture | ✅ (as conjecture) | explicitly labelled provisional; B.8 cross-family sweep is the falsifier — **un-run** |

### §4 Experiments / Table 2 (Pass@1, conditions i–v)

| condition | paper expects | what we have | status |
|-----------|--------------|--------------|:------:|
| (i) Base bf16 | Pass@1 on HE+/LCB union | LCB functional-49: **9/49 (18.4%)** (official harness) | 🟡 (LCB only, not HE+) |
| (ii) Trajectory-aware RAG | Pass@1 | **never built/run** | ⛔ |
| (iii) Direct PEFT QLoRA (same trajectories) | Pass@1 — Gate-1 comparator | `paper-table2` logged 0.008 (broken); no valid run | ⛔ |
| (iv) TTT-E2E | Pass@1 | `paper-table2` logged 0.000 (broken); no valid run | ⛔ |
| (v) **Rune** | Pass@1 | LCB-49 **9/49 (18.4%, de-overfit)**; MBPP (May) 0.514; MBPP absent-spec 8/24 | 🟡 |
| **(v) vs (i) headline** | — | **rune ties base, 9/49 = 9/49** on LCB functional-49 | ✅ (honest: parity, not a win) |

**Net:** Table 2 can be honestly filled only for (i) and (v) on LCB functional-49, where the result is a **tie**. (ii)/(iii)/(iv) require fresh runs. See §2 for the full LCB arc.

### §4.3 Gates

| gate | definition | status | gap |
|------|-----------|:------:|-----|
| Gate 1 (existence: v vs iii, ≥5pp, McNemar) | committed | ⛔ | (iii) never run; current LCB tie is v-vs-i |
| Gate 2 (6-benchmark robustness) | committed | ⛔ | `paper-gate2` (exp 27) has **zero metrics** |
| Gate 3 (procedural encoding, 15 fns × 8 inputs) | committed | ⛔ | never run; protocol fixed in B.6 |

### §4.4 / Figure 2 (diagnostics)

| element | status | evidence |
|---------|:------:|----------|
| Mean-pool collapses cosine diversity <0.05 | ✅ | the one in-hand ablation (motivated Perceiver); §3.2, B.10 |
| Cosine-diversity sentinel 0.1, collapse by step 4–5 without attenuation | 🟡 | pilot-observed; trace (Fig 2b) not exported |
| Fig 2a controlled-confound curve | ⛔ | designed (B.12), not measured |

### §3.5 Safety / §5 Discussion

| claim | status |
|-------|:------:|
| Reversibility, write-once lineage, promotion-by-evidence | ✅ (by construction; corpus lineage logged, exp 46 `corpus-registry`) |
| Cross-domain / non-coding extensions | ⛔ (explicitly speculative — fine) |

---

## 2. The full LCB benchmark arc (report all of it, not just the tie)

The honest value of showing the *arc* is that it documents how a reported win dissolved under audit — exactly the "all results, not just final" the project wants on record.

| stage | date | rune LCB func-49 | base | what changed |
|-------|------|:----------------:|:----:|--------------|
| pre-fix (decompose shipped nothing) | 06-09 03:00 | **0/49** | 9/49 | over-decomposition / `3768` collapse bug |
| post-engine-fix | 06-09 03:00 | **10/49** | 9/49 | `_collapse_benchmark_subtasks` fix; rune > base (+1: 3768, 3832) |
| **de-overfit (final, honest)** | 06-09 14:33 | **9/49** | 9/49 | removed task-specific answer-injection in `repair_brief.py`; `3832` no longer passes; **rune ties base** |

**Why the +1 was not real:** `src/rune/engine/repair_brief.py` (new in PR #55) had hard-coded LCB answers — `maxDifference`→task-3753 solution, any list-returning task→task-3760 "anti-diagonal" invariant, keyword suppression. The margin task `3832` passed *because* the anti-diagonal string fired in its repair prompt. Removed → task-agnostic briefs → parity. (PR #55 comment 2026-06-09 14:33Z; commit `d173ef8`.)

**Honest headline for the paper:** on LCB functional-49, the engine+adapter **match** the same base model single-shot (18.4% each), a **strict superset with zero regressions** (escalate mode's first attempt *is* the base zero-shot). The PR's defensible contributions are the **held-out-validated recall objective** (+0.105, CI excludes 0) and **general engine-correctness fixes**, *not* a pass@1 win.

### Oracle root-cause (the pass@1 ceiling) — controlled experiment

Through rune's *real* repair path (adapter-on, scaling 1.0, code not pasted in prompt):

| condition | repair fires | different code | **solved** |
|-----------|:-----------:|:--------------:|:----------:|
| real-engine oracle (public tests) | 0/11 | 0/11 | **0/11** |
| perfect oracle (hidden failing case) | 11/11 | 10/11 | **0/11** |

Two limiters, in order: **(1) in-loop oracle coverage** — shipped code passes every *public* test, so `diagnose→repair` never fires on the hidden bug (dominant, addressable); **(2) base-model capability** — even a perfect critique yields 0/11 solves on hard tasks. The earlier "byte-identical echo → capability ceiling" reading was a **synthetic-probe artifact** (code pasted at scaling 0 induced copying); the channel is demonstrably live. K=3 consensus differential oracle was **unsafe** (1 systematic FP `3817`, detection 2/11). (Exploratory probes, since removed; provenance: PR #55 comment 2026-06-09 07:11Z.)

---

## 3. Mapping the paper's framing onto our two tracks

| paper construct | which track / artifact supplies it | confidence |
|-----------------|-----------------------------------|:----------:|
| "production: Qwen 3.5 9B + DeltaCoder" | May track (`paper-table2`/`gate2`, exps 18–42) | sparse data |
| "dev: Gemma 2 2B + gemma_demo" | **no MLflow runs** — dev artifacts off-server | unbacked |
| §4.2 adapter-scaling result | re-anchored to Qwen3-4B `rune-bench-hpo` 0.627× (June) | done (2026-06-09) |
| §4.3 recall/accessibility evidence | June track (Phase-1 exp 45) — the strongest real result | ✅ |
| Algorithm 1 recursive loop | June engine (`graph.py`) | ✅ |

The cleanest honest paper would **anchor on the Qwen3-4B June track** (where the validated results live) and present Qwen3.5-9B Table-2 work as preliminary/secondary, rather than leading with Gemma 2 2B numbers that have no home.

---

## 4. Next measurements to make the paper submittable (prioritized)

Ordered by what unblocks a *central* claim.

1. **Fill Table 2 honestly (or restructure it).** Run (i) base and (v) Rune on the **HumanEval+ ∪ LCB held-out** union with $k=5$, McNemar + Wilson CIs as §4.1 specifies. Today only LCB-49 (i)/(v) exists and it's a tie. *Decision:* if the result stays a tie, restructure §4 around the recall objective + efficiency, not pass@1 dominance.
2. **Run Gate-1 comparator (iii) Direct PEFT QLoRA on the same trajectory corpus.** Without it there is no Gate-1 verdict — the paper's existence test. 200-trial budget already specified in §4.1(iii).
3. **~~Resolve `0.16×`~~ — DONE (2026-06-09).** Re-anchored §3.4/§4.2/Table 1/B.7/B.8 to the Qwen3-4B 0.627× evidence with corrected multiplier semantics. Remaining: pin the 95% CI on 0.627× from the trial-level convergence trace, and (optional) recover the Gemma log if it surfaces.
4. **Cross-family scaling sweep (B.8).** The paper's self-declared "most pressing methodological question." 50-trial α-only sweep on Qwen/Llama/Phi to test the 0.1–0.3× clustering prediction. Currently the structural conjecture rests on one data point whose store is off-environment (§0.1).
5. **Close the oracle-coverage gap for pass@1 (the real lever).** Property/metamorphic tests or broader generated inputs so `diagnose→repair` fires on hidden-failing tasks. K=3 consensus already shown unsafe. This is the only path to moving (v) above (i).
6. **Durable LCB-49 run.** Log both arms' official-harness 49-task grade to MLflow with the grade JSON as an artifact (today it's only in a PR comment). Prereq for citing the number.
7. **TTT-E2E (iv) and RAG (ii) baselines.** Needed for the secondary efficiency comparison (v vs iv) and the alternative-axis argument (v vs ii). Both currently unbuilt.
8. **Figure 2 traces.** Export the cosine-diversity-over-training trace (2b) and run the fixed-difficulty injected-history curve (2a). Both designed, neither measured.

---

*Update this map whenever a placeholder is filled or a new run lands. Every paper number should trace to an (exp, run, track) triple in the inventory.*
