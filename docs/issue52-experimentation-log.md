# Issue #52 — Experimentation Log (body-contrastive / adapter-as-memory)

**Repo:** ElixirTrials/rune · **Branch:** `issue52-bf16-body-contrastive` · **PR:** [#55](https://github.com/ElixirTrials/rune/pull/55)  
**Checkpoint of record:** c3 (`c3_t07_lp2_lg1.pt`, sha256 `53e24af243a3…`) — MLflow `issue52-phase1` run `fe72f9ddd69c`  
**Primary sources reconciled:** `instructions/scratchpad.md` (append-only, latest blocks authoritative), `instructions/reflections.md`, all `docs/issue52-*.md`, PR #55 comments, MLflow (`http://localhost:5000`, 2026-06-05).

> **2026-06-09 continuation.** This log was authored 2026-06-05; the LCB-benchmark / oracle / de-overfit arc (06-05→09) is added as **§3.7** and **§5** is updated. The complete MLflow registry (including the **May `paper-table2`/`paper-gate2` track** this June log never covered) is now indexed in [`mlflow-experiment-inventory-2026-06-09.md`](mlflow-experiment-inventory-2026-06-09.md), and every paper claim is mapped to evidence in [`paper-evidence-map-2026-06-09.md`](paper-evidence-map-2026-06-09.md). **Headline correction:** the LCB functional-49 result is **rune 9/49 = base 9/49 (a tie)**, not a win — see §3.7.

> **Status (2026-07-08):** (1) the E-phase1 heldout-24 objective estimate (+0.105, CI [+0.033,+0.182]; §1, §3.2) is superseded for the paper by the pre-registered fresh-pool re-estimate: **+0.147, n=120, sign test p=5.5e-14, CI [+0.109,+0.191]** ([`publication/c21_prep.md`](publication/c21_prep.md); MLflow exp 45 run `1769a1f8dedd43a789041536294c9825`). (2) The MLflow registry in §6.3 predates both the durable 06-19 LCB runs (`issue52-lcb-durable`, which resolved §3.7's "not yet a durable MLflow run" caveat) and the tracking-DB snapshot restore — experiments 78–86 lost their param/metric rows (S3 artifacts survive); `issue52-repobench-clamp` is re-logged as experiment id 79. (3) Current consolidated results: [`publication/handoff_realized_gates.md`](publication/handoff_realized_gates.md).

---

## 1. Executive summary

- **Positive control (D1, 2026-06-01):** Sakana Doc2LoRA on Gemma/Qwen proves the perceiver architecture can encode episodic memory; Rune #49 failed on **objective**, not architecture (`docs/issue52-findings-2026-06-01.md`).
- **Pilot 1 (`body_derangement`):** Margin opens via **91% deranged suppression**, accessibility flat — **objective misspecification**, not NULL (`docs/issue52-crossover-frozen-probe-results-2026-06-03.md`; scratchpad `2026-06-03` initial block). **Scaling audit:** +0.137 body floor is real encoding, not `alpha/r` bug.
- **Pilot 2 (`body_recall_guarded`):** Accessibility **PASS** (Δlp_matched +0.290, CI [+0.13,+0.45]); recitation clean 0/10; trained-on-test only (`docs/issue52-pilot2-recall-guarded-results-2026-06-03.md`; MLflow exp 44 run `7b82c304`).
- **Phase-1 (40-task train → 24 held-out):** **PASS** — generalizes (Δlp_matched +0.105, CI [+0.033,+0.182]); absent pass@1 **8/24** vs scale0 **0/24**, warm **3/24** (`docs/issue52-phase1-results-2026-06-04.md`; MLflow `fe72f9ddd69c`). Pilot-2 ckpt **does not** generalize (held-out flat).
- **Retention baseline (cross-domain):** **Partial** — goal/tail/avoid pass both bars on `external_codereview.val.clean` (n=24 subsample); **`diff` m-zero −0.028 flat** — not "gate-1 closed" per issue #52 DoD (`reflections.md` P1; scratchpad `2026-06-04 08:15`).
- **Goal 1 (capacity probe):** Only **k=1** survives CI scrutiny: c3 − scale0 **+0.292** [+0.083,+0.500]. **k>1 task-packing WITHDRAWN** (off-design per owner correction). **Goal 2 (40→N):** Modest monotone accessibility rise; per-doubling gains within noise at n=24; sweet spot ~80 tasks (`docs/issue52-goal1-*`, `docs/issue52-goal2-*`).
- **Goal 3 / runner:** Deep engine fixes (fence, diagnose livelock, thinking phase, presence penalty, freeform codegen, public oracle) committed. **Encoding probe:** adapter raises lp(FIX) but task-conditioning collapses margin — **not the lever** (`E-encoding-probe`). **Freeform codegen:** JSON `\n` over-escape caused phantom SyntaxError; freeform fence fixes hard pass@1 **1/4→2/4** (n=4; repair loop still unmeasured) (`E-freeform-codegen`, `da89765`).
- **Spec-in-adapter (reference modes):** adapter lifts **0.333 → 0.583** (+6/24) on easy pool — thesis works single-turn; HPO per-flavor tuning: reference_a/b1/c **0.588**, reference_b **0.471**, training_exact **0.294** (worst); **hard tasks NULL** (0.25 vs 0.25); repair-memory thesis **unmeasured** on adequate slice (`docs/issue52-goal3-conclusions-2026-06-05.md`).
- **Adversarial caveat (this doc):** Do not merge Phase-1 absent-recall, spec-in-adapter runner gains, and repair-memory claims — three **distinct regimes** (see §4). Pre-fix Goal-3 smokes are **STALE**. `prompt_mode` A/B/C/b1 experiments are **WITHDRAWN** from product (owner revert); numbers retained here only.

---

## 2. Timeline / phases

| Phase | Dates (UTC) | Focus | Outcome |
|-------|-------------|-------|---------|
| **Deliverables 1–4** | 2026-06-01 – 06-02 | Positive control, predeclared T0/E1/E2 spec, recipe MVC, crossover design | Architecture viable; qwen warm-start calibrated; frozen-probe protocol locked |
| **Pilot 1** | 2026-06-03 | `body_derangement` 30-step crossover (10 MBPP) | QUALIFIED — suppression-dominated |
| **Pilot 2** | 2026-06-03 | `body_recall_guarded` redesign | Accessibility PASS (train-on-test) |
| **Phase-1** | 2026-06-03 – 06-04 | Train-HPO 40 tasks → held-out 24 + pass@1 bench | **PASS** (generalizing); c3 baseline |
| **Infrastructure** | 2026-06-04 | Corpus recovery, `log_dataset`, `config.yaml` unification | Retention data unblocked; model-id single-sourced |
| **Retention + Goals 1–2** | 2026-06-04 | `diag_recoverability`; capacity probe; corpus 40→80→160 | Retention **partial**; G1 k=1 only; G2 modest scaling |
| **Goal 3 — runner fixes** | 2026-06-04 | Trace-through, extraction, thinking/presence, prompts, oracle | Runner corrected; commits `930cbfc`…`a1b4a78` |
| **Goal 3 — codegen + encoding** | 2026-06-04 `20:52` | Encoding probe (NULL); freeform codegen (JSON over-escape fix) | Encoding FAIL; freeform PASS mechanism (n=4) |
| **Spec-in-adapter + HPO** | 2026-06-04 – 06-05 | `prompt_mode` reference variants; flavor×scaling HPO (16 trials) | Memory +0.25 easy pool; HPO best `reference_a@0.627` |
| **Hard tasks + LCB** | 2026-06-05 | Multistep memory, LiveCodeBench probe, episodic design brainstorm | Hard NULL; LCB pipeline ready; plan JSON truncation blocker |

---

## 3. Experiment catalog

**Legend:** Verdict = PASS | FAIL | QUALIFIED | PARTIAL | STALE | OPEN | WITHDRAWN. Cross-source: ✓ aligned ≥2 sources; ⚠ partial/confounded; ✗ single source or disputed.

### 3.1 Deliverables & pre-registration

| ID | Date | Hypothesis | Spec & parameters | Procedure / tools | Results | Verdict | X-src |
|----|------|------------|-------------------|-------------------|---------|---------|-------|
| **E-D1-control** | 2026-06-01 | Doc2LoRA proves adapter-as-memory achievable | Sakana Gemma + qwen_4b_d2l warm-start; scorecard probes | Positive-control runs; MLflow `issue52-d2l-control` exp 56 | NIAH rougeL 1.0; qwen goal m−mismatch +2.235; #49 ~0 | PASS (architecture) | ✓ doc + MLflow |
| **E-T0-spec** | 2026-06-02 | Frozen go/no-go thresholds before training | T0/E1/E2 masks, qwen attribution correction | CPU scout; `docs/issue52-predeclared-spec-T0-E1-E2-2026-06-02.md` | Thresholds frozen; gemma numbers demoted | PASS (protocol) | ✓ doc |
| **E-D4-recipe** | 2026-06-02 | Recipe-4b MVC trainability | `configs/issue52_recipe_mvc_4b.yaml` | Distill smoke | Trainability demonstrated (handoff D4) | PASS | ✓ `docs/issue52-deliverable4-*` |

### 3.2 Pilot & Phase-1 (recall objective)

| ID | Date | Hypothesis | Spec & parameters | Procedure / tools | Results | Verdict | X-src |
|----|------|------------|-------------------|-------------------|---------|---------|-------|
| **E-P1-derange** | 2026-06-03 | Body derangement hinge raises matched recall | 30 steps, τ implicit hinge, 10 MBPP; warm sha `6438b46c…`; MLflow exp 43 `c401f0c0` / `d296a4e2` | `_specificity_probe`, GATE A/B | body m−mismatch +0.137→+1.026; Δlp_matched +0.075, CI spans 0; 91% suppression | QUALIFIED / objective FAIL | ✓ scratchpad + `issue52-crossover-*` |
| **E-scaling-audit** | 2026-06-03 | +0.137 is scaling artifact? | effective_scaling=45.25; same forward sig+body | Code audit + parity scripts | Sig +3.84 vs body +0.137 same pass → encoding not apply bug | PASS (no artifact) | ✓ scratchpad + reflections |
| **E-P2-guarded** | 2026-06-03 | Guarded matched-recall raises accessibility | τ=−0.7, λ_p=1, λ_g=1, 30 steps; MLflow exp 44 `7b82c304` | `_specificity_probe`, `_recitation_probe` | Δlp_matched +0.290 [+0.13,+0.45]; sig +3.84→+5.71; recite 0/10 | PASS (train-on-test) | ✓ scratchpad + pilot2 doc |
| **E-gate1-smoke** | 2026-06-03 | Accessibility → functional pass@1 | Pilot-2 ckpt; 10 tasks; real MBPP 3-test | `_pass1_probe.py` | present 6→10/10; absent 1→5/10; scale0≈0 | PASS (smoke, memorization) | ✓ scratchpad `20:35` |
| **E-pilot2-heldout** | 2026-06-03 | Pilot-2 generalizes? | 24 held-out; pilot-2 ckpt | `_specificity_probe` | m-zero +0.530→+0.514 (flat); sig generalizes | FAIL (memorization) | ✓ scratchpad `20:45` + phase1 doc |
| **E-phase1** | 2026-06-03 – 04 | 40-task train generalizes to 24 held-out | c1–c4 grid τ∈{−0.7,−0.5}, λ_p∈{1,2}, λ_g∈{1,2}; 48 steps; corpus train sha `e60f0dd8…`, heldout `cae274bf…` | `_phase1_orchestrate.py`; MLflow exp 45 | **Best c3:** m-zero +0.635 (+0.105 vs warm); absent pass@1 8/24; present 19/24; run `fe72f9ddd69c` | PASS | ✓ scratchpad `21:37` + phase1 doc + PR #55 + MLflow |
| **E-corpus-recovery** | 2026-06-04 | Retention corpus durable + logged | S3 `external_codereview.unrolled.jsonl` sha `4931fe03…`; val.clean 323 rows sha `7e3692df…` | `split_corpus.py`, `log_dataset`; MLflow `corpus-registry` `ea4f3c43` | Gate-1 data unblocked | PASS (infra) | ✓ scratchpad `07:56` + PR comment |
| **E-retention** | 2026-06-04 | c3 retains cross-domain recoverability | c3 ckpt; Qwen3-4B-Instruct bf16; n=24 subsample; scaling 45.25 | `diag_recoverability.py` | goal +0.297/ +0.539; tail +0.224/ +0.108; avoid +0.862/ +0.155; **diff −0.028**/ +0.058 | PARTIAL (3/4) | ✓ scratchpad `08:15` + reflections P1 + PR |

**Phase-1 train-HPO detail (held-out 24 accessibility, m-zero):**

| config | τ | λ_p | λ_g | m-zero | Δ vs warm (+0.530) |
|--------|---|-----|-----|--------|---------------------|
| c1 | −0.7 | 1 | 1 | +0.593 | +0.063 |
| c2 | −0.5 | 1 | 1 | +0.601 | +0.071 |
| **c3** | **−0.7** | **2** | **1** | **+0.635** | **+0.105** |
| c4 | −0.5 | 2 | 2 | +0.604 | +0.074 |

Sources: `docs/issue52-phase1-results-2026-06-04.md`, scratchpad `21:37`, MLflow exp 45.

### 3.3 Goal 1 — capacity probe (WITHDRAWN k>1)

| ID | Date | Hypothesis | Spec & parameters | Procedure / tools | Results | Verdict | X-src |
|----|------|------------|-------------------|-------------------|---------|---------|-------|
| **E-G1-capacity** | 2026-06-04 | Adapter holds multiple tasks (k) with flat prompt | k∈{1,2,4,8}; 3 arms scale0/warm/c3; heldout 24; name-cued spec-absent | `_recall_capacity_probe.py`, `_run_capacity_arms.sh` | pass@1/24: k=1 scale0 5, warm 9, **c3 12**; paired c3−scale0 k=1 **+0.292** [+0.083,+0.500]; k=8 c3−scale0 +0.042 CI spans 0; c3−warm all k CIs span 0 | **PASS k=1 only**; k>1 **WITHDRAWN** | ✓ scratchpad `09:15`+`09:25` + goal1 doc + reflections |

**Note:** Owner correction (`10:35`): hypernet is single-step; k-task packing is off-design — do not interpret k>1 decay as architectural capacity limit.

### 3.4 Goal 2 — corpus scaling

| ID | Date | Hypothesis | Spec & parameters | Procedure / tools | Results | Verdict | X-src |
|----|------|------------|-------------------|-------------------|---------|---------|-------|
| **E-G2-scale** | 2026-06-04 | More disjoint train tasks raise held-out recall | N∈{40,80,160} nested; fixed eval 24; c3 objective 48 steps; MLflow exp 47 | `build_scaling_train_corpora.py`, `_fetch_goal2_ckpts.py`, `_goal2_analysis.py` | m-zero: 0.635→0.649→0.671 monotone; pass@1 k=1: 12→15→14; beyond-40 paired CIs span 0; trained−warm m-zero significant all sizes (+0.105→+0.141); N=80 first pass@1 beat warm (+0.250 CI excl 0) | PASS (modest); diminishing returns | ✓ scratchpad `09:50`+`10:00` + goal2 doc |

### 3.5 Goal 3 — runner substrate, fixes, spec-in-adapter

| ID | Date | Hypothesis | Spec & parameters | Procedure / tools | Results | Verdict | X-src |
|----|------|------------|-------------------|-------------------|---------|---------|-------|
| **E-encoding-probe** | 2026-06-04 | Embedding/prompt changes lift hard-task repair | Episode + embed probes; toy + 3 hard tasks | `docs/issue52-goal3-encoding-probe-2026-06-04.md` | Adapter raises lp(FIX); task-conditioning collapses fix-vs-failure margin (task_only control) | FAIL (not the lever) | ✓ scratchpad `20:52` |
| **E-freeform-codegen** | 2026-06-04 | JSON over-escape → phantom SyntaxError; freeform fixes attempt-1 | code/repair/integrate `output_schema=None`; 4 hard tasks; c3; commit `da89765` | Live engine + ast/compile probe | literal `\n` 0; int_to_roman FAIL→PASS; pass@1 **1/4→2/4**; steps=3 (repair unmeasured) | PASS (mechanism); PARTIAL (n=4) | ✓ scratchpad `20:52` + §4.3 |
| **E-G3-parity** | 2026-06-04 | scale0 = true no-adapter on engine path | adapter_scaling=0 | `_goal3_multiturn_probe.py` parity | max logit delta 0.0 | PASS | ✓ scratchpad `12:55` |
| **E-G3-smokes** | 2026-06-04 | Pre-fix runner quality | verify/verify2/promptfix; 3–5 tasks | `run_benchmark` | scale0 0.80, c3 4/5, etc. | **STALE** (pre profile/thinking fix) | ✓ reflections P1 |
| **E-degen-ablation** | 2026-06-04 | Thinking phase causes single-word collapse | thinking on/off × presence 0/1.5 × base/c3 | `_degen_probe.py`, `_degen_probe_presence.py` | thinking ON 28–33% degen → OFF 0–11%; presence 1.5→0 fixes residual base 2/18 | PASS (root cause) | ✓ scratchpad `19:30`+`20:05` |
| **E-postfix-gate** | 2026-06-04 | Fixed runner beats pre-fix | model profile; thinking off; presence 0 | promptfix smoke | scale0 6/6 attempt1, c3 5/6; degen 0 | PASS (runner fix, not thesis) | ✓ scratchpad `21:10` + reflections P5 |
| **E-spec-adapter** | 2026-06-04 – 05 | Spec only in adapter, name in prompt | prompt_mode reference_a/b/c/b1; 8 MBPP; c3/warm/scale0 | Engine `prompt_mode` branch (**reverted after log**) | 1-turn c3: a 4/8, b 5/8, c 3/8, **b1 1/8**; floor 2/8; multi-turn ref_c famous names: scale0 3/4 > c3 1/4 (confounded) | PARTIAL (directional n=8); **WITHDRAWN** code | ✓ scratchpad `22:55`–`01:25` + reflections |
| **E-hpo-flavor** | 2026-06-05 | Best flavor×gen_scaling for spec-in-adapter pass@1 | 16 trials; prompt_mode×scaling[0.1,1.5]; pool 24; c3; oracle on, judge off | `rune bench --hpo`; `configs/goal3_flavor_hpo.yaml`; MLflow bench-hpo | Best **reference_a @ 0.627**; tuning **0.588** (10/17); val **0.571 (4/7)**. Per-flavor tuning: reference_a/b1/c **0.588** (tied); reference_b **0.471**; **training_exact 0.294** (worst) | PASS (HPO complete) | ✓ scratchpad `03:40`+`05:15`+`07:25` + MLflow |
| **E-true-floor** | 2026-06-05 | Apples-to-apples memory: reference_a scale0 vs c3 | Both reference_a (name-only prompt); n=24 | Manual bench arms | scale0 **0.333 (8/24)** vs c3 **0.583 (14/24)** → **+0.25** | PASS (single-turn memory) | ✓ scratchpad `04:55` + goal3-conclusions |
| **E-hard-memory** | 2026-06-05 | Adapter lifts hard multistep tasks | 8 multistep; reference_a; c3@0.627 | `issue52-goal3-hard-memory` MLflow exp 62 | Held-out pass@1: scale0 **0.25 (2/8)** = c3 **0.25 (2/8)**. c3 **public** attempt-1 **5/8** vs scale0 **1/8** (oracle reached) but held-out flat. **int_to_roman:** scale0 **7/7** held-out → c3 fails (over-perturb @0.627) | FAIL (no held-out gain) | ✓ scratchpad `06:00`+`07:25` |
| **E-lcb-probe** | 2026-06-05 | LCB v6 pipeline + generation on hard | 4 problems test6.jsonl; official `codegen_metrics` | `_lcb_run.py`, `_lcb_grade.py` | Grader validated; LCB pass@1 **0.25 (1/4)**; 3/4 empty code (plan JSON truncation). Rune-internal 0.75 run **BOGUS** (stdin empty `test_code`) | PARTIAL (pipeline PASS, gen FAIL) | ✓ scratchpad `06:40` |
| **E-oracle** | 2026-06-04 | Public doctest triggers repair | `rune/engine/oracle.py` | 4 hard tasks | decode_string repairs; commit `f08ef97` | PASS (mechanism) | ✓ scratchpad + docs |
| **E-judge** | 2026-06-04 | Model-judge flips under-tested cases | JudgeResult order fix `9d16a66` | 4 hard tasks | int_to_roman false-positive; kept OFF for HPO | FAIL (unvalidated) | ✓ scratchpad `22:40` |

**Consolidated pass@1 table (corrected runner, judge OFF):**  
Source: scratchpad `07:25`, `docs/issue52-goal3-conclusions-2026-06-05.md`.

| Config | Regime | pass@1 | n |
|--------|--------|--------|---|
| scale0 `full` | spec-in-prompt ceiling | 0.792 | 24 |
| reference_a scale0 | spec-absent floor | 0.333 | 24 |
| reference_a c3 @0.627 | spec-in-adapter | 0.583 | 24 |
| c3 full @1.0 | spec-in-prompt + adapter | 0.750 | 24 |
| hard multistep reference_a | spec-in-adapter, hard | 0.25 | 8 |
| scale0 full (val7) | spec-in-prompt, held-out 7 | 0.714 (5/7) | 7 |
| reference_a c3 HPO val (val7) | spec-in-adapter | 0.571 (4/7) | 7 |
| c3 full @1.0 (val7) | spec-in-prompt + adapter | 0.857 (6/7) | 7 |

*val7 = mbpp/115,133,118,119,106,113,135. n=7 — treat as directional only (scratchpad `07:25`).*

### 3.6 Journal / design-only blocks (no experiment entry)

**Coverage:** 70 scratchpad `### [` blocks → **26** catalog experiments (§3) + journal skips below.

Scratchpad blocks logged as planning, incorporation, or infrastructure without new measured arms: `19:31`–`19:42` reflections + two-stage RL research; `08:40`–`09:10` config refactor; `09:05` design fork; `09:30` capacity instrument smoke (n=4; full arms in **E-G1-capacity**); `09:30`–`09:35` Goal-2 checkpoint gotcha / Goal-1 doc promotion; `10:05`–`10:10` session close + advisor G2 tightening; `10:20` CI note; `10:35`–`11:05` reflection incorporation + multistep design; `12:30` Goal-3 pre-reg; `13:25` wall-clock note; `14:30` engine miswirings trace (smokes → **E-G3-smokes** STALE); `15:45`–`17:45` implementation / promptfix plans; `16:10` extraction validation; `16:45` runner diagnosis; `18:30` degen debug precursor (**E-degen-ablation**); `19:55` consolidated thoughts; `20:20` reflections; `21:40`–`22:10` spec-in-adapter design/launch; `22:58` scale0-full 0.792 launch (**STALE** invalid floor — superseded **E-true-floor**); `01:50` HPO launch; `02:25` HPO MLflow fix; `03:10` graph revision design; `05:25` benchmark calibration (Qwen MultiPL-E 76.8 ≈ scale0-full 0.792); `07:00` session summary; `07:15` episodic brainstorm; `07:35` implementation start; handoff files referenced in scratchpad but **not present on disk** in this workspace (2026-06-05).

### 3.7 LCB benchmark + engine redesign + oracle root-cause (2026-06-05 → 06-09)

This phase moved from frozen probes to the **real runner on LiveCodeBench v6 functional-49**, graded with the **official LCB harness** (`tools/_lcb_grade.py`, same grader that scores base). All on `Qwen3-4B-Instruct-2507`, c3 checkpoint, escalate mode (zero-shot base first → adapter on repair).

| ID | Date | Hypothesis | Spec & parameters | Procedure / tools | Results | Verdict | X-src |
|----|------|------------|-------------------|-------------------|---------|---------|-------|
| **E-engine-redesign** | 2026-06-08 | Engine-correctness bugs (not thesis) cap hard-task pass@1 | sig normalization, typing-name probe, ship-best-on-exhaustion, advisory `big_o` gate w/ static floor + killable subprocess, decompose-collapse, flash-attn torch-2.12 wheel | commits `d28a3ab`…`eeff35f`; 516–522 unit tests | All fixes property-based (non-task-specific); CI green | PASS (engine) | ✓ PR #55 §3 + git |
| **E-lcb49-arc** | 2026-06-09 | Does rune beat the same base single-shot on LCB func-49? | official LCB harness; escalate c3@0.627 vs base zero-shot | `_lcb_run.py`, `_lcb_grade.py` | **0/49 (pre-fix) → 10/49 (post-fix, overfit live) → 9/49 (de-overfit)**; base **9/49**. Final: **TIE, strict superset, 0 regressions** | PASS (engine ties base; NOT a win) | ✓ PR #55 comments 03:00/07:11/14:33 |
| **E-deoverfit** | 2026-06-09 | The +1 (3832) margin is robust? | audit `repair_brief.py` (new-in-PR) for answer-injection | code audit + clean rerun; commit `d173ef8` | Found hard-coded LCB answers (maxDifference→3753 soln; any list task→3760 anti-diagonal invariant; keyword suppression). 3832 passed *because* anti-diagonal fired. Removed → task-agnostic briefs → **3832 fails, rune=base** | **+1 was overfit-dependent + within noise**; corrected | ✓ PR #55 14:33 + commit |
| **E-oracle-rootcause** | 2026-06-09 | Is there a model↔oracle comms failure behind false-pass misses? | real repair path (adapter-on, scaling 1.0, code not in-prompt); 11 false-pass tasks | `_real_repair_oracle_test.py`, `_perfect_oracle_probe.py`, `_repair_trace.py` | Real oracle **0/11 fires** (public always passes); perfect oracle **11/11 fires, 10/11 change code, 0/11 solve**. Channel live (byte-echo was synthetic-probe artifact) | Two limiters: **(1) oracle coverage** (dominant, addressable); **(2) capability** | ✓ PR #55 07:11 + tools |
| **E-kconsensus** | 2026-06-09 | K=3 consensus differential oracle closes coverage gap | 3-arm consensus on hidden-bug detection | `_oracle_gate_test.py` | 1 systematic FP (`3817`), detection 2/11 | FAIL (unsafe) | ✓ handoff + tools |

**LCB functional-49 final table (official harness):**

| Config | pass@1 | breakdown |
|--------|--------|-----------|
| base, single-shot, no runner/adapter | **9/49 (18.4%)** | pass=9, runtime=3, tle=6, wrong=31 |
| rune escalate (c3@0.627), de-overfit | **9/49 (18.4%)** | strict superset of base; rune-only [], base-only [] after de-overfit |

*Note: rune escalate's first attempt **is** the base zero-shot, so it can only add tasks via escalation, never lose them — hence "strict superset." Published 35.1% is base on the **full** LCB v6 set; on this harder functional-only subset base itself is 18.4%, so 35% is not the comparable bar.*

**MLflow caveat:** exps 73/74 (`issue52-lcb-fix-rerun*`) are **6-task smokes** (1/6), *not* the 49-task grade. The authoritative 49-task number lives in the official-harness JSON + PR comment only — **not yet a durable MLflow run** (see inventory §4, action item).

---

## 4. Cross-cutting findings

### 4.1 Three evidence regimes (do not merge)

| Regime | Prompt | Adapter carries | Primary experiments | Headline (qualified) |
|--------|--------|-----------------|---------------------|----------------------|
| **Spec-absent** | Name / minimal | Task spec (body) | Phase-1, capacity k=1, reference_a | 8/24 absent recall; +0.292 capacity; runner +0.25 easy |
| **Spec-in-prompt** | Full spec (`project_label`) | Prior code/errors across repair | Goal-3 repair substrate (pre-reg) | Adapter helps attempt-1 repair when spec given — **not** same as absent-spec |
| **Recoverability** | Probe harness | goal/diff/tail/avoid tokens | `diag_recoverability` | 3/4 targets; diff flat |

### 4.2 Objective & scaling

- **Guarded matched-recall** is the validated **training objective** for continuation-body accessibility (pilot 2 → Phase-1). λ_p=2 is the generalization lever; λ_g=2 hurts slightly.
- **Probe scaling** (45.25 = `lora_alpha`) is correct for logprob probes; **generation scaling** should be tuned separately (HPO best ~0.627 on easy pool). Mixing them overstates over-perturbation claims.
- **Training-surface sensitivity:** Filling empty `## Current Code` / `## Review Feedback` (reference_b1) drove c3 **below** no-adapter floor (1/8) — on-distribution format matters more than prompt enrichment.

### 4.3 Runner bugs (fixed, commits on branch)

| Bug | Symptom | Fix commit (theme) |
|-----|---------|-------------------|
| Fence inside JSON `code` | Spurious SyntaxError, repair churn | `930cbfc` extraction pipeline |
| Diagnose phantom subtask | 10× diagnose livelock | `7927ebf` |
| Thinking phase on non-thinking Qwen3-Instruct | 28–33% single-word degen | `a1b4a78` model profile |
| presence_penalty=1.5 on code | 11% residual degen | profile default 0.0 |
| JSON codegen over-escape | 1-line phantom errors | `da89765` freeform codegen |
| In-loop oracle = bare def only | Repair never engaged | `f08ef97` public-example oracle |
| scale0-full "floor" | Spec in prompt via `project_label[:1200]` | Use reference_a for memory tests |

### 4.4 STALE / WITHDRAWN

| Item | Status | Reason |
|------|--------|--------|
| Pre-fix smokes (verify, verify2, promptfix) | STALE | Multiple engine states; reflections P1 |
| k>1 capacity probe arms | WITHDRAWN | Off-design task-packing (owner `10:35`) |
| prompt_mode A/B/C/b1 engine branch | WITHDRAWN | Owner revert after recording (`22:55`) |
| "Gate-1 closed" | **Rejected wording** | diff m-zero failed; n=24 subsample; no bootstrap |
| "Phase-1 traded capacity for peak" | **Rejected** | c3−warm CIs span zero at all k (scratchpad `09:25`) |
| scale0-full 0.792 as memory floor | **Invalid** | Spec-in-prompt ceiling (scratchpad `04:55`) |
| ref_c multi-turn scale0 3/4 > c3 1/4 | Confounded | Famous LeetCode name memorization |

### 4.5 Adversarial pass (overclaim corrections applied)

1. **"Adapter-as-memory thesis confirmed"** → Only for **spec-absent, easy MBPP**; hard tasks and spec-in-prompt repair are NULL or unmeasured.
2. **"8/24 proves product memory"** → Single-shot absent is harshest proxy; multi-turn repair thesis still OPEN; 33% is partial.
3. **"Retention gate passed"** → **Partial baseline** only; diff channel flat; full val.clean + CIs pending.
4. **HPO val 0.571 vs scale0 0.792** → Compared invalid arms until reference_a true-floor correction.
5. **training_exact HPO worst (0.294)** → Does not disprove Phase-1 probe gains — different metric/regime (runner generation vs frozen probe).
6. **"rune 10/49 > base 9/49" (06-09 morning)** → **Retracted.** The +1 (3832) was within noise *and* dependent on task-specific answer-injection in `repair_brief.py`. De-overfit → **rune 9/49 = base 9/49 (tie)**. PR value is the held-out recall objective + engine fixes, not a pass@1 win (§3.7).
7. **"oracle communication failure"** → **Ruled out.** Channel verified live through the real repair path; limiters are oracle *coverage* then *capability* (§3.7 E-oracle-rootcause).

---

## 5. Open questions & next steps

> **Paper-facing prioritized list** (Table 2, Gate-1 comparator, 0.16× provenance, cross-family sweep, oracle-coverage lever, durable LCB run) is in [`paper-evidence-map-2026-06-09.md`](paper-evidence-map-2026-06-09.md) §4. The items below are the issue-52 research backlog; they overlap but are scoped to the recall thread.

0. **Oracle-coverage mechanism (the pass@1 lever).** Real-engine oracle never fires on hidden bugs (public tests pass); perfect oracle fires but base solves 0/11. Build property/metamorphic tests or broader generated inputs so `diagnose→repair` triggers on false-pass tasks. K=3 consensus already shown unsafe (§3.7 E-kconsensus). This is the only demonstrated path to moving rune above base.
1. **Full retention panel:** Re-run `diag_recoverability` on full `val.clean` (323 rows) or pre-registered stratified 50–100 with bootstrap; close **diff** gap or scope Phase-2 objective.
2. **Repair-memory eval:** Frozen B2 slice (40–60 attempt-1-fail), 3-arm `rune run`, success-vs-turn curve + recovery gap (B1 pre-reg); report by outcome stratum (attempt-1 pass vs repair-needed).
3. **Hard-task slice:** Curate MBPP attempt-1-fail + multistep; re-HPO gen_scaling on hard only; fix plan-step spec leak (`project_label` in plan).
4. **Robustness for hard/LCB:** Freeform or json-repair for plan/decompose/diagnose; feed all public examples to oracle; episodic adapter conditioning (brainstorm approved `07:15`).
5. **Phase-2 cooperative RL:** Outcome pass@1 + KL-anchor to c3 + recall-replay; only after gates above; addresses direction-conflict (body reproduction vs base solution).
6. **CI / merge:** 83 E501 in `tools/` block CI — policy: `extend-exclude` or wrap before merge.

---

## 6. Appendix — artifact index

### 6.1 Checkpoints (sha256)

| Artifact | sha256 (prefix) | Role |
|----------|-----------------|------|
| Warm-start doc-to-lora | `6438b46c…` | Floor adapter |
| Pilot 1 step30 | `d296a4e2…` | body_derangement |
| Pilot 2 step30 | `7b82c304` (run id) | body_recall_guarded train-on-test |
| **Phase-1 c3** | **`53e24af243a3…`** | **Phase-2 retention baseline** |
| Goal-2 n80/n160 | MLflow exp 47 (`9812c7f2…`, `39a6f211…`) | Scaling arms |

### 6.2 Corpora (sha256)

| File | sha256 (prefix) | n |
|------|-----------------|---|
| `mbpp_recall_train.jsonl` | `e60f0dd8…` | 40 |
| `mbpp_recall_heldout.jsonl` | `cae274bf…` | 24 |
| `mbpp_recall_train_80.jsonl` | (built session) | 80 |
| `mbpp_recall_train_160.jsonl` | (built session) | 160 |
| `external_codereview.val.clean.jsonl` | `7e3692df…` | 323 |
| `external_codereview` source | `4931fe03…` | 7670 |
| `goal3_candidate_pool.json` | `e9e34f66…` | 144 |

### 6.3 MLflow experiments (localhost:5000)

| exp_id | name | Key runs |
|--------|------|----------|
| 43 | issue52-body-crossover | Pilot 1 |
| 44 | issue52-body-recall | `7b82c304` pilot 2 |
| 45 | issue52-phase1 | **`fe72f9ddd69c` c3** |
| 46 | corpus-registry | `ea4f3c43` |
| 47 | issue52-goal2-scaling | n80, n160 trains |
| 48–62 | issue52-goal3-* | Multiturn, specinadapter, HPO, hard-memory, truefloor |
| — | rune-bench-hpo | Flavor×scaling parent (16 trials) |

`MLFLOW_TRACKING_URI` unset in shell; server reachable at `http://localhost:5000`.

### 6.4 Key commits (branch `issue52-bf16-body-contrastive`)

| sha | Summary |
|-----|---------|
| `25dcbd2` | MLflow `log_dataset` |
| `3bf930d` | `config.yaml` single source of truth |
| `930cbfc`–`a1b4a78` | Engine fixes + model profile |
| `da89765` | Freeform codegen + graph revision start |
| `f08ef97` | Public-example oracle |
| `4643ef4` | HPO MLflow progress + prompt_mode axis |
| `9d16a66` | Judge reason-before-verdict |

### 6.5 Tools & configs (issue #52)

> **Note (2026-07-08):** most of these files were removed in the publication cleanup and remain available only in git history; of the paths below, only `tools/_specificity_probe.py` and `configs/goal3_flavor_hpo.yaml` are still in the tree.

| Path | Role |
|------|------|
| `tools/_specificity_probe.py` | Frozen E1 probe |
| `tools/_pass1_probe.py` | pass@1 absent/present |
| `tools/_recall_capacity_probe.py` | G1 capacity (RBM) |
| `tools/_phase1_orchestrate.py` | Phase-1 pipeline |
| `tools/diag_recoverability.py` | Retention scorecard |
| `tools/_goal3_multiturn_probe.py` | G3 driver (RBM) |
| `configs/issue52_body_recall_crossover_4b.yaml` | Pilot 2 / Phase-1 objective |
| `configs/goal3_flavor_hpo.yaml` | Flavor×scaling HPO |

### 6.6 Validation gaps (this log)

| Gap | Notes |
|-----|-------|
| MLflow trial-level metrics for all 16 HPO trials | Parent run + scratchpad interim; per-trial val_pass@1 not fully exported here |
| Goal-2 n80/n160 checkpoint sha256 | Fetched via MLflow; local paths ephemeral post-upload |
| Handoff files `instructions/handoff_*.md` | Cited in scratchpad; not found in workspace snapshot |
| `docs/issue52-phase1-results-2026-06-04.md` | **Cited in §1/§3.2 but absent from disk** (2026-06-09); phase-1 numbers are recoverable from §3.2 (E-phase1), MLflow exp 45 run `fe72f9ddd69c`, and scratchpad `21:37`. Restore or re-point citations. |
| Full `val.clean` retention | Only n=24 subsample scored |
| B1 repair-memory batch | Pre-registered; not completed on frozen slice post-profile |
| Judge arm | Reordered but not re-validated |
| reference_d (training-exact) | Planned; superseded by HPO `training_exact` flavor (worst 0.294) |
| E-encoding-probe / E-freeform-codegen | Added audit pass 2; scratchpad `20:52` |

---

*Document generated 2026-06-05. **Audit pass 2:** 70 scratchpad blocks → 26 catalog entries + 44 journal/skip. Maintainer: reconcile against latest scratchpad block before citing numbers externally.*
