# Code-side task plan — Rune repo work for the TMLR submission

**Owner:** AI Researcher · **Scope:** *only* work that happens **in this repository** (runs, builds, harness, MLflow, artifacts). Article/manuscript edits live with the paper (`drafts/paper_v13.tex`, outside this repo) and are tracked separately in `article_update_checklist.md` — **not here**. This file is what the GPU instance pulls and executes.

**Discipline:** every run carries a pre-registered gate (the outcome that supports the claim *and* the outcome that forces a reframe). Report nulls. Do not re-assert retracted results (LCB functional-49 is a tie/underpowered; Gates 1–3 unrun; 0.16× re-anchored to 0.627×). All runs are on frozen checkpoint `c3`, single RTX 4090, no retraining.

---

## C0 — Data lookups & provenance (no GPU)

| # | Task | Feeds article item | Done-when |
|---|------|--------------------|-----------|
| C0.1 | **Corpus-trajectory lookup.** Query the corpus manifest for MBPP tasks that have *gold trajectories/adapters*; cross-reference against the objective-grid selection set and the N=60 keystone set; report the disjoint count. **Rule: ≥ ~50 disjoint tasks with trajectories → C2.1 (fresh-pool re-estimate) is feasible; else it is not — signal the prose-downgrade path to the article side.** | A-OBJ | Disjoint count reported; C2.1 feasibility decided. |
| C0.2 | **SHA-256 hashes.** Compute/verify hashes for the `c3` checkpoint and each corpus split against the MLflow artifacts; emit a `hashes.txt` manifest in the repo release. (Copy-forward from paper_v8 checklist A4.) | A-REPRO (hash paste) | `hashes.txt` committed; values handed to article side. |

## C1 — Keystone campaign (the load-bearing runs; harness `tools/_repobench_clamp_run.py`, N=60, frozen c3)

| # | Task | Type | Pre-registered gate | Effort |
|---|------|------|---------------------|--------|
| C1.1 | **`a2_tail` arm** — place the identical oracle conditioning string (variant `use`, anchor 0, ~124 tok) at the prompt **tail**, adjacent to cursor, within W=768. Report vs `floor` at matched cursor-code lengths. Log to MLflow `issue52-repobench-clamp`. | run | *episodic beats a2_tail, separated CIs* → keystone strengthened (weight beats best prompt channel, info held fixed). *a2_tail ≈ episodic (overlapping CIs)* → channel not decisive under oracle conditioning; the 32k-infeasible regime becomes the headline. **This gate outcome tells the article side which keystone framing to write.** | ~1 GPU-hr |
| C1.2 | **`a2_tail_filler` control** — 124 tok neutral filler in place of the pointer, same displacement, same budget. Isolates the pointer's marginal contribution from "different tokens near cursor." | run | Report pointer effect = (a2_tail − a2_tail_filler). | +~0.5 GPU-hr |
| C1.3 | **Swap / mutation control — BUILD FIRST, then run.** Not yet in the harness (design-spec §8 prescribes "port PR #57 §8"; only `hotswap_adapter` exists today). Rename the gold identifier in `render_episodic`'s conditioning text, add a `swap` arm to the runner, then run on the keystone subset. | build+run | `s`≈floor-CI → refutes frequency/output-bias confound. between → report attributable fraction (e−s)/(e−f). `s`≈episodic-CI → keystone compromised (signal article side). | ~0.5 eng-day + ~0.5 GPU-hr |

**C1 exit:** a2_tail / filler / swap in MLflow with Wilson CIs + paired McNemar; the realized gate outcomes handed to the article side (they decide the keystone headline and the Setup oracle-conditioning wording).

## C2 — Objective de-biasing run (conditional)

| # | Task | Type | Gate | Effort |
|---|------|------|------|--------|
| C2.1 | **Fresh-pool re-estimate of +0.105** — *only if C0.1 cleared (≥~50 disjoint tasks with trajectories).* Re-estimate `c3` matched-log-prob on the disjoint pool; recompute the across-task sign test. **Do NOT generate new trajectories to chase p<0.05.** Log to MLflow `issue52-phase1`. | run (cond.) | crosses p<0.05 → hand de-biased number to article side (strip caveat). stays >0.05 (currently 0.064) → signal article side to take the prose-downgrade path. | ~1–2 GPU-hr if feasible |

## C3 — Optional / advisor-elected runs

| # | Task | Type | Trigger | Effort |
|---|------|------|---------|--------|
| C3.1 | **ReasonCACHE / prefix-KV arm (Option A).** Implement a prefix-tuning / KV-injection arm under W=768 on N=60; compare recovery. | build+run | Only if the advisor elects Option A (recommended: Option B prose this round). KV-injection harness does not exist yet. | +1–2 eng-days + ~1 GPU-hr |
| C3.2 | **Recovery-vs-budget-W sweep.** Re-run the existing set at W∈{256,512,768,1536}; the harness already logs prompt-tokens per arm per level. Produces the "advantage grows as the budget tightens" curve for a figure. | run | Accept if Phase-1 GPU budget allows; else decline explicitly. | ~1–2 GPU-hr |

---

## Critical path & GPU budget
1. **C0** (no GPU): corpus lookup gates C2.1; hashes feed the reproducibility appendix.
2. **C1** the load-bearing runs — **~2 GPU-hr** baseline (a2_tail + filler + swap). +0.5 eng-day for the swap build.
3. **C2.1** only if C0.1 clears (~1–2 GPU-hr).
4. **C3** optional: +1–2 eng-days (Option A) and/or +1–2 GPU-hr (W-sweep).

**Total GPU:** ~2 GPU-hr baseline, +1–2 hr if C2.1 runs, +1–2 hr if the W-sweep is taken. All on the single 4090, frozen c3, no retraining.

**If only one run happens: C1.1 + C1.2 (`a2_tail` + filler).** Its gate outcome decides the paper's central keystone framing.

## Handoffs to the article side (`article_update_checklist.md`)
- C1 gate outcomes → keystone headline (A-KEY) + Setup oracle-conditioning wording (A-ORACLE).
- C0.1 / C2.1 outcome → objective claim de-biased or downgraded (A-OBJ).
- C0.2 hashes → reproducibility appendix paste (A-REPRO).
- C1.3 swap result → conditioning-format / confound wording (A-FORMAT).
- C3.1 (if run) → ReasonCACHE arm result replaces the prose comparator (A-REASONCACHE).
