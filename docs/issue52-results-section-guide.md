# Issue #52 — Results section guide (scientific framing)

**Purpose:** Turn the experimentation log into a **cogent Results narrative** for a paper, isolating findings of scientific interest—especially the tradeoff between **hypernetwork-generated adapters** and **in-context prompting**—without overclaiming.  
**Companion:** [`issue52-experimentation-log.md`](issue52-experimentation-log.md) (full catalog).  
**Audience:** Authors drafting §Results; reviewers asking “what is actually established?”

> **2026-06-09 update.** Two things this guide (authored 06-05) did not yet contain, now reconciled against the runner + MLflow:
> 1. **LCB functional-49 (official harness): rune 9/49 = base 9/49 — a TIE**, not a win. The morning "10/49 > 9/49" was retracted (within noise + overfit-dependent answer-injection in `repair_brief.py`, removed). See §4.6 and [`paper-evidence-map-2026-06-09.md`](paper-evidence-map-2026-06-09.md) §2.
> 2. **None of the paper's pre-registered Gates 1–3 has been run** as specified (Gate-1's PEFT-iii comparator, TTT-E2E-iv, and RAG-ii baselines do not exist; `paper-gate2` MLflow runs are empty). Do not present any current number as a gate verdict.
> Companion docs added: [`mlflow-experiment-inventory-2026-06-09.md`](mlflow-experiment-inventory-2026-06-09.md) (all 74 experiments + the May `paper-table2` track), [`paper-evidence-map-2026-06-09.md`](paper-evidence-map-2026-06-09.md) (claim→evidence). **The paper's `0.16×` scaling headline could not be located in any artifact reachable here**, so (per author, 2026-06-09) `paper_v9.tex` was **re-anchored** to the in-hand Qwen optimum `0.627×` — corrected to read as a *multiplier* of the native 45.25 scale (a mild ~0.6× attenuation), not "Nx below"; the structural conjecture was softened to match.

> **2026-06-22 update — the constant-prompt evidence now exists, and the HumanEval+ limit is retracted.** Two §4.6 gaps this guide flagged as un-measured are now measured (full prose: [`issue52-results-longcontext-2026-06-22.md`](issue52-results-longcontext-2026-06-22.md); MLflow `issue52-repobench-clamp` / `issue52-humanevalplus`, engine `efa7b9e`):
> 1. **Adapter-carried context under a token budget (the missing §4.6 eval).** On RepoBench v1.1 `cross_file_first` (cross-file completion) under a fixed prompt budget W=768 with the in-prompt baseline deliberately stressed past it, the frozen adapter recovers the required cross-file symbol at **31/60 (0.517 [0.393,0.638])** vs **floor 9/60 (0.150)** and **= the unbudgeted full-context ceiling 17/30 (0.567)** — at a **16.7× shorter prompt** (McNemar 23:1, **p=3e-6**). At 32k the full-context prompt is prohibitive on 30/30 yet the adapter recovers 13/30. This is the constant-prompt / beyond-budget result §4.6 said "requires a fixed prompt budget + window stressor … not yet measured." **Conditioning format is the controlling variable:** a naïve multi-file dump is a null (0.217≈floor, p=1.0); only the **episodic per-task surface** (name the one cross-file API) works — HPO-selected (variant=use, anchor=0, scaling=0.91), held-out validated (4/10 vs 1/10) before the N=60 confirmation. No weight training. Metric is identifier-recovery, not pass@1.
> 2. **HumanEval+ "difficulty-dependent / hurts easy tasks" is RETRACTED.** The −16 (c3 100 < base 116) was a 2-bug grading artifact (harness dropped prompt imports → spurious NameError on 19 typing-signature tasks; untrusted escalation-floor discarded correct zero-shots). Post-fix at `efa7b9e`: **base 134/164 (0.817)**, **c3 135/164 (0.823) — strict superset, +1, 0 regressions.** §4.6's E-lcb / HE+ rows and §6's "beyond window" row are updated accordingly.

---

## 1. The scientific question (one paragraph for the paper)

Modern coding agents face a structural tension: **in-context learning (ICL)** is flexible and often strong per query, but its cost scales with prompt length (KV cache, latency) [1][4]. **Parameter-efficient adaptation (PEFT)**, including LoRA, amortizes task information into a small adapter and can reduce per-query context cost once trained [4][5]. **Hypernetworks** meta-learn the mapping from a *context* (document, trajectory) to an adapter in a single forward pass—Doc-to-LoRA [1], SHINE [2]—promising “read once, query many times” with **constant prompt length**.

**Rune’s bet (issue #52):** a perceiver hypernetwork can make the generated LoRA adapter an **episodic memory substrate** for coding trajectories, so the runner can keep prompts minimal while carrying prior state in adapter conditioning across steps.

This guide states **what to report**, **how to frame it**, and **what our experiments actually support**—including where episodic-prompting revisions supersede older numbers.

---

## 2. Literature: when context wins vs when adapters win

| Dimension | In-context (prompt carries state) | Adapter / hypernetwork (weights carry state) | Key references |
|-----------|-----------------------------------|---------------------------------------------|----------------|
| **Per-query compute** | Process all context tokens every forward pass; cost ~linear in context length [4] | One hypernet pass to materialize adapter; subsequent queries avoid re-reading full context [1][2] | Liu et al. 2022 [4]; Doc-to-LoRA [1] |
| **Per-query memory (KV)** | Grows with context; dominant at long context [1] | Fixed adapter weights; KV for *query* only [1][5] | Doc-to-LoRA [1]; LoRA-as-memory [5] |
| **Information density / accuracy (short context)** | Often **stronger** when full spec fits in prompt; ICL competitive on many tasks [4] | Can underperform ICL if adapter is weak, mis-scaled, or off-distribution [2][5] | SHINE-R trails ICL on LongBench when under-trained [2] |
| **Beyond native context window** | Truncation, retrieval, or eviction required | Chunked hypernet → composed LoRA; NIAH >4× train length [1]; recurrent SHINE-R [2] | Doc-to-LoRA [1]; SHINE [2] |
| **Update latency** | Instant (edit prompt) | Hypernet: sub-second adapter gen [1]; per-doc CD: slow | Doc-to-LoRA [1] |
| **Controllability** | Prompt is explicit, inspectable | Adapter is opaque; scaling/strength is a tuning knob (our hard-task over-perturb) | Our E-hard-memory; gen vs probe scaling |
| **Coding domain** | PEFT often beats ICL on code gen benchmarks when fine-tuned [6] | Hypernet-for-code is **less established** than Doc2LoRA-for-QA | Weyssow et al. 2023 [6] |

**Consensus from prior work (not yet proven for Rune coding trajectories):**

- **Per-token / per-query efficiency:** Context wins when you *can* fit everything in the window and query once; adapters win when the **same information is queried repeatedly** or when context would exceed the window [1][4][5].
- **Long-running / multi-step:** The *architectural* argument for hypernets is strongest when history would grow unbounded in the prompt but can be **re-encoded into a fixed-size adapter each step** [1][2]. Empirical proof requires a **step-indexed** eval under prompt budget pressure—not just single-turn recall.

---

## 3. Two prompting regimes (mandatory framing in Results)

Do **not** merge numbers across regimes. Label every table/figure.

| Regime | Prompt | Adapter carries | Our primary experiments |
|--------|--------|---------------|-------------------------|
| **A — Spec-absent** | Mission name / minimal cue only | Full task spec (body), signatures | Phase-1 absent pass@1; capacity k=1; reference_a true-floor |
| **B — Spec-in-prompt** | Full spec (`project_label` or `full`) | Prior code, errors, trajectory (repair) | Goal-3 repair substrate (pre-reg); scale0-full **ceiling** 0.792 |
| **C — Recoverability probe** | Harness-defined | goal / diff / tail / avoid tokens | `diag_recoverability` (partial 3/4) |

**Episodic prompting revision:** New adapter formatting may change Regime A/B surfaces. Retain **objective-discovery** and **evaluation-validity** results; re-run **substrate** and **multistep** claims under the episodic format before citing them as final performance.

---

## 4. What to include in Results (and what to omit)

### 4.1 Include — core scientific story

| Block | Experiments | Why it belongs in Results |
|-------|-------------|---------------------------|
| **Architecture is not the bottleneck** | E-D1-control | One short paragraph: Doc2LoRA-class hypernets *can* store episodic facts [1]; Rune #49 failed on objective |
| **Objective discovery** | E-P1-derange → E-P2-guarded → E-phase1 | Central ML contribution: *which* loss shapes accessibility vs suppression |
| **Generalization + functional transfer** | E-phase1, E-pilot2-heldout (contrast) | Held-out accessibility CI + absent pass@1; memorization negative control |
| **Cross-domain retention (partial)** | E-retention | Honest transfer beyond MBPP; diff flat |
| **Runner-native spec-in-adapter (easy)** | E-true-floor, E-hpo-flavor (subset) | Direct test of “adapter replaces missing prompt tokens” on engine |
| **Limits** | E-hard-memory, E-lcb-probe (brief) | Where adapter ≈ context floor; hard multistep NULL |

### 4.2 Methods or Supplementary — not main Results

| Block | Reason |
|-------|--------|
| E-T0-spec, E-D4-recipe, E-scaling-audit, E-corpus-recovery | Protocol / infra |
| E-G3-smokes, E-postfix-gate, E-degen-ablation, E-freeform-codegen, E-oracle | Evaluation validity (cite 2–3 sentences in Results intro) |
| E-G2-scale (full tables) | Diminishing returns; one sentence + supplement |
| E-spec-adapter A/B/C/b1 (n=8), k>1 capacity (WITHDRAWN) | Off-design or withdrawn; footnote only |
| E-encoding-probe, E-judge | Negative / unvalidated; Discussion |
| Config refactor, CI lint, dataset logging | Reproducibility |

### 4.3 Re-run before citing (episodic prompting)

- E-true-floor, E-hpo-flavor (if adapter surface changes)
- B1 repair-memory batch (success-vs-turn curve)
- Any multistep “adapter carries trajectory” claim

---

## 5. Proposed Results structure (paper §4)

### §4.1 Setup and evidence regimes (½ page)

- Frozen-probe metrics (accessibility, m−mismatch) vs functional pass@1 vs recoverability scorecard.
- Three regimes table (§3 above).
- Cite positive control [1] in one sentence.

### §4.2 Training objective for body accessibility (1 page)

**Framing:** *Contrastive derangement optimizes the wrong gradient path; guarded matched-recall targets accessibility directly.*

| Result | Report | Interpretation |
|--------|--------|----------------|
| Pilot 1 | m−mismatch +0.137→+1.026; Δlp_matched +0.075 (CI spans 0); 91% suppression | Objective misspecification, not NULL |
| Pilot 2 | Δlp_matched +0.290 [+0.13,+0.45]; recitation 0/10 | Trainability of accessibility under absent conditioning |
| Pilot-2 held-out | Flat m-zero | 10-task training memorizes; motivates scale-up |

**Figure:** decomposition bars (matched vs mismatch contribution).

**Do not claim:** Product-ready memory; generalization (comes in §4.3).

### §4.3 Generalization and functional recall (1 page)

**Framing:** *Scaling disjoint training episodes yields weak but statistically supported held-out accessibility and partial absent pass@1.*

| Result | Report | Interpretation |
|--------|--------|----------------|
| Phase-1 c3 | m-zero +0.635 (+0.105 vs warm); 17/24 positive; CI [+0.033,+0.182] | Generalizing **accessibility** lever (λ_p=2) |
| pass@1 bench | absent 8/24 (c3) vs 0/24 (scale0) vs 3/24 (warm); present 19/24 stable | Proxy→functional bridge is **partial** (33%), not ceiling |
| Goal-2 (one sentence) | Monotone m-zero 0.635→0.671; beyond-40 CIs noise | Diminishing returns; ~80 tasks practical |

**Table 2:** Phase-1 HPO configs + pass@1 arms.

**Interpretation discipline:** 8/24 is evidence the channel is open, not that the agent solves MBPP from memory reliably. Aligns with Doc2LoRA’s “works on QA with limits” [1], not NIAH-perfect coding.

### §4.4 Cross-domain recoverability (½ page)

**Framing:** *Adapter trained on MBPP recall partially transfers to GitHub-review recoverability probes.*

| Target | m-zero | m-mismatch | Include? |
|--------|--------|------------|----------|
| goal | +0.297 | +0.539 | Yes |
| tail | +0.224 | +0.108 | Yes |
| avoid | +0.862 | +0.155 | Yes (n=14) |
| diff | −0.028 | +0.058 | Yes — **failure** |

**Wording:** “Retention **baseline** established (3/4); edit-local (`diff`) not lifted.” Do **not** say “gate-1 closed.”

### §4.5 Hypernetwork vs context: single-turn runner eval (1 page)

**Framing:** *When the prompt omits the spec, the adapter can substitute for missing context tokens; when the spec is in the prompt, context wins on easy tasks.*

This is the section that speaks to **scientific interest** (adapter vs context).

| Comparison | pass@1 | n | Role in argument |
|------------|--------|---|------------------|
| reference_a **scale0** (name only) | 0.333 (8/24) | 24 | **Memory floor** — no spec anywhere |
| reference_a **c3 @0.627** | 0.583 (14/24) | 24 | **Adapter carries spec** (+0.25) |
| scale0 **full** (spec in prompt) | 0.792 (19/24) | 24 | **Context ceiling** — not a fair floor |
| c3 full @1.0 | 0.750 | 24 | Adapter + context: below ceiling |

**Interpretation (supported today):**

1. **Per-query information:** For easy MBPP, putting the spec in the prompt beats carrying it in the adapter (0.792 vs 0.583). This is consistent with ICL/PEFT tradeoffs: ICL is strong when context fits [4]; adapters trade peak accuracy for repeated-query efficiency [1][5].
2. **Non-zero adapter value:** c3 beats the true floor by +6/24 — the hypernet **does** encode task information the base cannot get from the name cue alone. This is the runner-native analogue of Phase-1 absent pass@1.
3. **Training surface matters:** HPO `training_exact` worst (0.294) vs reference_a/b1/c tied (0.588) — on-distribution **format** beats naive faithfulness to distill template.
4. **Gen scaling ≠ probe scaling:** HPO best gen_scaling 0.627 vs probe 45.25 — mixing them overstates “adapter hurts generation.”

**Figure 3:** Three-bar chart: floor / adapter / ceiling (label regimes explicitly).

**Literature hook:** Doc-to-LoRA shows QA gains with reduced KV and latency [1]; our easy-pool result is **directionally consistent** but **smaller effect size** and **coding-specific** confounds (oracle, structured output) apply.

### §4.6 Limits: hard tasks, multistep, and long-horizon claims (½ page)

**Framing:** *Efficiency and constant-prompt advantages are hypothesized for long horizons; our hard/multistep evidence does not yet support them.*

| Result | Report | Interpretation |
|--------|--------|----------------|
| E-hard-memory | 0.25 (2/8) held-out; scale0 = c3 | No held-out gain on hard multistep |
| Public vs held-out | c3 public 5/8 vs scale0 1/8; held-out flat | Oracle ≪ full tests; adapter helps shallow signal only |
| int_to_roman | scale0 7/7 → c3 fails @0.627 | Over-perturbation / wrong gen scaling |
| E-lcb-probe (06-05) | 1/4; plan truncation | Early pipeline blocker — superseded by E-lcb49-arc |
| **E-lcb49-arc (06-09, official harness)** | **rune 9/49 = base 9/49 (tie)** | Engine+adapter match base single-shot; strict superset, 0 regressions. **Not** a pass@1 win |
| **E-oracle-rootcause (06-09)** | real oracle 0/11 fires; perfect oracle 11/11 fires / **0/11 solve** | pass@1 ceiling = oracle *coverage* (public tests miss hidden bugs) then base *capability* — channel verified live |
| B1 repair-memory | Not run | **Cannot claim** multistep repair substrate yet |

**Critical honesty for §Discussion:**

> We establish single-turn **spec-in-adapter** recall on easy tasks. We do **not** establish that hypernetwork adapters outperform context for **long-running coding** or **continuation beyond the context window**. That claim requires a step-indexed eval with (i) fixed prompt budget, (ii) deliberate context-window stressor on the baseline arm, and (iii) episodic re-encoding each step—architecturally aligned with Doc-to-LoRA chunking [1] and SHINE-R [2], but **not yet measured** in Rune.

### §4.7 Summary paragraph (Results closing)

Template:

> We show (i) a guarded matched-recall objective that generalizes weakly on held-out body accessibility, (ii) partial translation to absent pass@1, (iii) runner-native evidence that adapters can carry task specifications when prompts are minimal—though full in-prompt specs remain stronger on easy tasks—and (iv) partial cross-domain retention excluding edit-local tokens. Hypernetwork-generated adapters thus function as a **viable external memory channel** under spec-absent prompting, with **stability and hard-task costs** that context avoids. Claims about **dominance at long horizons** await multistep evaluation under episodic prompting.

---

## 6. Mapping “context wins per token, adapters win long-running” to evidence

| Claim | Supported by our experiments? | Evidence | Gap |
|-------|------------------------------|----------|-----|
| Context wins when spec fits in prompt (easy tasks) | **Yes (directional)** | scale0-full 0.792 vs reference_a c3 0.583 | Not a controlled per-token efficiency measurement |
| Adapter can carry spec when prompt is minimal | **Yes** | +0.25 true-floor; Phase-1 absent 8/24 | Easy MBPP; n=24 |
| Adapter reduces need for prompt tokens (architecture) | **By design** | Engine uses minimal `prompt_*`; trajectory in adapter conditioning | Token-count curves not in main log |
| Adapter delivers context **beyond a prompt budget** | **Established (directional, significant)** | RepoBench clamp W=768 (§4.x / `issue52-results-longcontext-2026-06-22.md`): adapter 0.517 = full-context ceiling 0.567 at 16.7× shorter prompt; 32k prompt-prohibitive yet adapter 0.433; McNemar p=3e-6 | identifier-recovery not pass@1; 1 regression; format-tuned |
| Adapter wins for **long-running multi-step** (pass@1) | **Not established** | hard multistep NULL; B1 repair batch not run | Need step-indexed pass@1 curves |
| Adapter wins for **repeated queries** on same episode | **Plausible, not measured** | Doc2LoRA motivation [1]; single-turn pass@1 only | Multi-query same-adapter bench |
| Hypernet beats warm doc-to-lora prior on recall | **Weak / mixed** | c3 > warm on some arms; c3−warm CIs often span 0 | Training is the lever, not architecture alone |

**Paper-safe synthesis (updated 2026-06-22):**

- **Now a Results conclusion (not just a hypothesis):** under a fixed prompt budget the adapter delivers out-of-prompt context **at parity with full in-prompt context, at ~16.7× shorter prompt**, and remains effective where the in-prompt channel is infeasible (32k prohibitive) — RepoBench clamp, McNemar p=3e-6 (§4.x). State this as an **established budget-stressed context-delivery result on coding trajectories**, directionally consistent with Doc-to-LoRA / SHINE long-context results [1][2]. Scope it precisely: the metric is **identifier-recovery**, the conditioning format is **tuned/held-out-validated**, and it is a **(v)-vs-context-channel** comparison, not a Gate verdict.
- **Still future work:** “adapters win **long-running multi-step coding** (end-to-end pass@1)” — the budget result is single-turn recovery, not a step-indexed pass@1 curve; the B1 repair-substrate batch remains un-run.
- The per-token / per-query **efficiency** tradeoff is now anchored by a direct prompt-length measurement (768 vs 12,836 mean tokens at parity), not only by ceiling-vs-floor [1,4,5].

---

## 7. Episodic prompting: what to keep when the format changes

| Finding type | Keep in Results? | Action |
|--------------|------------------|--------|
| Wrong objective (derangement) | Yes | Unchanged |
| Guarded recall generalizes (weak) | Yes | Unchanged |
| Evaluation validity (thinking, JSON escape, oracle) | Yes (brief) | Methods cross-ref |
| True-floor +0.25 | Yes, with caveat | Re-run under episodic adapter; label “pre-episodic” if not re-run |
| HPO flavor ranking | Maybe | Re-run if conditioning sections change |
| Hard NULL | Yes | Still valid as “limits under prior format” |
| Multistep repair | No numbers yet | Placeholder + pre-reg only |

**Suggested Results opener after episodic revision:**

> Results are organized into **objective and generalization findings** (independent of adapter prompt formatting) and **substrate comparisons** (reference_a true-floor and multistep evals), the latter reported under the episodic adapter format unless noted.

---

## 8. Figures and tables (minimal set)

1. **Fig 1:** Pilot 1 vs Pilot 2 accessibility decomposition.
2. **Table 1:** Phase-1 HPO (τ, λ_p, λ_g, held-out m-zero).
3. **Table 2:** pass@1 bench (scale0 / warm / c3) — Regime A.
4. **Fig 2:** True-floor vs adapter vs ceiling (Regime A vs B).
5. **Table 3:** Recoverability 4-target (partial).
6. **Fig 3 (optional):** Hard tasks — public attempt-1 vs held-out (shows oracle gap).

Supplementary: Goal-2 scaling, full HPO 16 trials, withdrawn k>1 / prompt_mode variants.

---

## 9. References (literature cited in this guide)

[1] [Doc-to-LoRA: Learning to Instantly Internalize Contexts](https://consensus.app/papers/details/9cc22e858f1f5e30a9b8063b08c95c86/) (Charakorn et al., 2026) — hypernetwork meta-learns context distillation; KV/latency; long-context NIAH.

[2] [SHINE: A Scalable In-Context Hypernetwork for Mapping Context to LoRA in a Single Pass](https://consensus.app/papers/details/1188d865aead53d3a900b5ef6b48dc2c/) (Liu et al., 2026) — single-pass LoRA; SHINE-R recurrent long context.

[3] [A survey on LoRA of large language models](https://consensus.app/papers/details/cf5ed27e5cb552d091d4b5db59da01e3/) (Mao et al., 2024) — PEFT / LoRA landscape.

[4] [Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning](https://consensus.app/papers/details/3b4b9e3082a85b5b9ca5d9ffd0559c34/) (Liu et al., 2022) — ICL vs PEFT cost and accuracy.

[5] [Understanding LoRA as Knowledge Memory: An Empirical Analysis](https://arxiv.org/html/2603.01097v2) (2026) — LoRA as memory; inference time vs ICL on repeated queries.

[6] [Exploring Parameter-Efficient Fine-Tuning Techniques for Code Generation with Large Language Models](https://consensus.app/papers/details/3006ca88cf1d54338e17a36ca5905ff7/) (Weyssow et al., 2023) — PEFT vs ICL on code generation.

**Rune primary data:** [`issue52-experimentation-log.md`](issue52-experimentation-log.md), [`issue52-phase1-results-2026-06-04.md`](issue52-phase1-results-2026-06-04.md), [`issue52-goal3-conclusions-2026-06-05.md`](issue52-goal3-conclusions-2026-06-05.md).

---

*Guide version: 2026-06-05. Update when episodic prompting experiments complete or B1 multistep batch lands.*
