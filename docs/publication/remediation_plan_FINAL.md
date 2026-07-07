# Remediation plan (FINAL) — v13 paper, post-adversarial-review

**Author:** AI Researcher (record) · **Inputs:** `HANDOFF_v13_review.md` (28 findings) → `remediation_plan_v1.md` → Adversary review (`adversary_review.md`, verdict: *sound_with_revisions*, 12 critiques). This version incorporates the Adversary's critiques **and** a ground-truth code check that resolves the single biggest risk it identified.

**Governing discipline.** Every run carries a pre-registered gate naming both the supporting and the reframe outcome; nulls are reported and claims reframed where a gate fails. Nothing the repo docs retracted is re-asserted (LCB functional-49 is a tie/underpowered; Gates 1–3 unrun; 0.16× re-anchored to 0.627×).

---

## The decisive finding (resolves Adversary critique 1 / Q1)

I checked the harness source (`origin/chore/publication-cleanup:src/rune/bench/repobench.py`, `render_episodic`, and the benchmark design spec). The result changes the framing:

> **The adapter's episodic conditioning is oracle-supplied.** It is built from `row.gold_snippet_index` — the design spec's own words: *"the oracle dependency for all arms."* The eval hands the hypernetwork the gold cross-file symbol + its signature (variant `use`); the model does **not** retrieve or infer it from the trajectory.

Two consequences:
1. **The v1 fallback reframe — "the adapter internalizes retrieval; the prompt arm needs an oracle" — is false and is dropped.** Both arms are oracle-conditioned. The adapter retrieves nothing.
2. **The keystone is, honestly, a channel comparison under oracle conditioning**: holding the delivered information identical, does weight-space delivery beat prompt delivery under a token budget? That is a legitimate, well-posed question — and it is close to what §6.2 already concedes ("a channel comparison, not a Gate verdict"). The plan now makes that explicit rather than implying retrieval.

This makes the `a2_tail` arm **fair and symmetric** (critique 1 answered: both arms carry the identical oracle content; the only axis that varies is the delivery channel), and it makes the arm *more* important, because a tie would collapse the keystone to "under oracle conditioning, channel doesn't matter."

---

## Phase 0 — Decisions, corpus lookups, mechanical fixes (no GPU)

| ID | Action | Change from v1 |
|----|--------|----------------|
| **B2 venue** | **Commit to TMLR** as the default and build only that target (anonymized submission mode, keep 15pp, delete `neurips_2026.sty`, rename folder). *Adversary critique 3:* the NeurIPS branch is not a "1h build" — the 15→9pp cut is a 1–2 day restructure that would collide with Phase 3 prose. If the advisor insists on NeurIPS, that restructure is scheduled **before** Phase 3, not after. **This is the author/advisor decision I cannot make.** | v1 under-scoped NeurIPS |
| **2a-feasibility** | **Corpus lookup (author, Phase 0 — not an Adversary question, critique 6):** query the corpus manifest for MBPP tasks that have *gold trajectories/adapters*, cross-reference against the objective-grid selection set and the N=60 keystone set, report the disjoint count. Rule: **≥ ~50 disjoint tasks with trajectories → Phase 2a is feasible; else → prose downgrade path (2b) is the answer, written now.** (Raw MBPP headroom exists — 974/427 vs 224 used — but the binding requirement is trajectories, not problems.) | v1 wrongly punted this to the Adversary |
| M5.1 | SHA-256 hashes: re-verify `c3` + corpus-split hashes against MLflow artifacts, paste into Appendix A, strip `\todo`. (Copy-forward from paper_v8 checklist A4.) | — |
| M6 | Rename `functional-49` → "LiveCodeBench-v6 full post-cutoff functional set (N=63)" everywhere; introduce the 49-item subset once, explicitly. | — |
| P8 | Algorithm 1 → real `algorithm` float with `\label{alg:loop}`; replace hardcoded refs. | — |
| P9a + QA | Delete stale `figure2.pdf`; confirm `\includegraphics` resolves on clean build; **eyeball compiled Fig. 2 at print size** (axis/arm labels, Wilson whiskers, colour thread — critique 10). | added QA item |
| NOV-5 | `shi2025revisit` → "Yaorui Shi et al."; drop the unverified note. | — |
| NOV-4 | Downgrade unconfirmed `@inproceedings` tags to `@misc` (keep TC-LoRA workshop tag); fix `zhang2025ace` forward-date. | — |
| P11/C9 | "12,836" (or "~12.8k") used consistently; §6.1 → "12/63 to 16/63 (+4)". | — |

---

## Phase 1 — The keystone campaign (B1 + C7 + M4)

One campaign; all arms reuse `tools/_repobench_clamp_run.py` on frozen `c3`, N=60 held-out set.

### 1a. `a2_tail` — the honest in-prompt channel (B1, blocking)
- **What:** place the *identical oracle conditioning string* the hypernetwork receives (variant `use`, ~124 tok) at the prompt **tail**, adjacent to the cursor, within W=768. Symmetric channel test.
- **Budget accounting (Adversary critique 4):** the 124-token pointer *displaces* 124 tokens of near-cursor code from the floor's 768-token window — it is a within-budget trade, not a free addition. Report `a2_tail` and `floor` at **matched cursor-code lengths**, and include an **`a2_tail_filler` control** (124 tokens of neutral filler in place of the pointer) so the pointer's marginal contribution is isolated from the "different tokens near cursor" effect.
- **Pre-registered gate:**
  - `episodic` beats `a2_tail` with separated CIs → keystone strengthened: weight-space delivery beats the *strongest* prompt channel holding information fixed. State it as such.
  - `a2_tail ≈ episodic` (overlapping CIs) → **reframe honestly** (the correct reframe, not the retracted v1 one): *"Under oracle conditioning and a shared token budget, the delivery channel is not the deciding factor; the adapter's value is not channel superiority but [constant-prompt scaling / KV-cost / the 32k regime where the prompt cannot hold the content at all]."* The 32k stratum result (adapter 0.433 where the in-prompt arm is infeasible) survives this reframe and becomes the load-bearing claim.
- **Effort:** ~1 GPU-hour + the filler control.

### 1b. Swap / mutation control (C7) — **implement** (design-spec prescribes it; not yet in harness)
- **Status (verified against source):** the benchmark design spec §8 *prescribes* this control — "**port** PR #57 §8: rename the gold identifier in the conditioning, check recovery of the *original* symbol drops" — but it is **not yet implemented** in `tools/_repobench_clamp_run.py` or `src/rune/bench/repobench.py` (the only `swap` in the runner is `hotswap_adapter`, unrelated). So this is a small build (rename the gold identifier in `render_episodic`'s conditioning text, add a `swap` arm), not a re-run. Est. ~0.5 eng-day + ~0.5 GPU-hour. The PR #57 pattern to port is documented, so the port is mechanical.
- **Quantitative gate (Adversary critique 5 — no ambiguous middle):** let `f`=floor, `e`=episodic, `s`=swap recovery.
  - `s` within Wilson CI of `f` → cleanly refutes the frequency/output-bias confound.
  - `s` above floor-CI but below episodic-CI → partial confound; report the attributable fraction `(e − s)/(e − f)` as the share genuinely due to conditioning content.
  - `s` overlaps episodic-CI → frequency artifact dominant; the keystone is compromised and must say so.
- **Effort:** ~0.5 GPU-hour.

### 1c. ReasonCACHE / KV comparator (M4) — Option B, but *substantive* (Adversary critique 2 / Q2)
- v1's Option B ("name it as unrun") is **a dodge on its own**, because ReasonCACHE is not a neutral comparator: its abstract asserts a *theorem* that KV/prefix expressivity strictly dominates rank-`r` weight updates. Citing it as support, declining to test it, and not rebutting the theorem is three strikes on one paper.
- **Revised Option B (prose, but load-bearing):** (i) drop "we test that boundary directly"; (ii) name ReasonCACHE/prefix as a committed unrun comparator (see §6.3 table below); **and (iii) add one paragraph that engages the theorem** — either concede its premise and argue the deployment regime is not expressivity-limited (the keystone task needs a small effective rank; the LoRA rank at the operating point is adequate), *or* challenge an assumption (ReasonCACHE's bound assumes single-layer prefix injection; multi-layer LoRA composes across layers in a way the bound does not cover). Without (iii), Option B does not close M4.
- **Option A (run a prefix/KV arm)** remains the stronger move and is **recommended if the target is NeurIPS or if the advisor wants to contest for top-tier now** (est. 1–2 engineering days for the KV-injection harness + ~1 GPU-hour). For a TMLR round, revised Option B clears.

**Phase 1 gate:** `a2_tail`, `a2_tail_filler`, and swap numbers in MLflow with Wilson CIs + paired McNemar; the abstract/§5.4 rewritten to whichever outcome obtained; the oracle-conditioning nature of the conditioning stated plainly in Setup.

---

## Phase 2 — Statistical de-biasing (M1 + M5.2 + C4/C5)

### 2a. Fresh-pool re-estimate of +0.105 — *conditional on the Phase-0 corpus lookup*
- If ≥~50 disjoint tasks with trajectories exist: re-estimate `c3` matched-log-prob on that pool; recompute the sign test. Gate: crosses p<0.05 → report de-biased, strip caveat; stays >0.05 → downgrade (2b).
- **Do not manufacture a fresh pool by generating new trajectories to chase p<0.05** (Adversary Q3) — that is worse than the current selection bias.

### 2b. Prose downgrade (the fallback, written now — Adversary Q3: sufficient, not a drop)
- The +0.105 with sign test p=0.064 and selection-biased CI is a **reportable suggestive finding on a selected configuration**, not a settled result — dropping it entirely would be over-correction. Soften the abstract ("a guarded matched-recall objective shows more consistent — though not yet de-biased — held-out gains than a contrastive objective, which mainly suppresses alternatives"), report p=0.064 as-is, remove "produces generalising held-out recall."

### 2c. Abstract↔body reconciliation (M1, prose)
- **LCB (C5), in the abstract:** add that "strict superset / zero regressions" follows from the escalate control flow vs. the raw base, and that the adapter-isolated contribution (scale0→c3) is +4/−2, McNemar **p=0.69** — not significant at N=63. (Committed-but-unexecuted v10 fix.)
- **n=10 "validation" (C4):** → "checked (non-significantly; n=10 underpowered, Fisher p=0.30) on a disjoint held-out split before locking the configuration." The overfitting protection is procedural (disjoint splits), not statistical.

---

## Phase 3 — Framing & bibliography prose (no runs; **after** the venue decision, Adversary critique 12)

| ID | Action |
|----|--------|
| M2 (C3/P4) | Lead parity with the **matched within-8k** comparison (episodic 0.600 vs ceiling 0.567, same 30 problems — a clean win); report pooled 0.517 as the conservative all-strata figure with the 32k caveat; state what population 12,836 is computed over (arm, strata, N). |
| M3 (P5) | State the 32k-stratum average context length behind 26.8× (≈20.6k/768); reword as a raw prompt-length ratio against an *infeasible* arm, not parity. |
| C6 | Specify exactly what `a2_clamp`/`a2_full` prepend (full concatenated cross-file source vs. filtered snippet). |
| C10 | Ground the 32k "infeasible" claim in the paper: name the harness grading/context budget and say whether it is a base-model limit or an eval-design choice. |
| **Encoder (Q4 / M5.3)** | Narrow "architecture is not what gates this" → "Perceiver beats mean-pool" **paper-wide** (Adversary critique 7): audit every "architecture / encoder / gating / insensitive" occurrence in intro, contributions, §5, §6 — not just §3.2. Add "encoder architecture beyond mean-pool" to the §6.3 unrun-comparator list. Acceptable scope cut *only* if done paper-wide + listed as a limitation. |
| **§6.3 unrun-comparator table (Adversary critique 8)** | Add a named, durable table so "next round" is a commitment, not a disappearance: rows for `direct-PEFT (Gate-1)`, `ReasonCACHE/prefix KV, W=768, N=60`, `LSTM/vanilla-transformer encoder` — each "committed for next revision." |
| NOV-2 | Recast "third axis": token-memory (window + RAG, one substrate) vs. KV/activation memory (prefix/KV-cache) vs. weight memory — or drop the numbered-axis language. |
| NOV-3 | §5.3: stop citing Liu2022 (a PEFT-beats-ICL result) to support "ICL is strong when context fits"; cite an ICL-capability result or reword. |
| NOV-6 | ReasonCACHE "without touching parameters" → "without updating the base model's weights." |
| P10 | Re-add v10-committed cites (TMEM as the named direct-PEFT standard; Mosbach to single-family limitation) or ensure nothing implies they're cited. |
| P6 | Abstract → ~200–250 words, split problem+mechanism / results+bound; move HumanEval+ and scaffold-split to intro. |
| P7 | "X, not Y" 29 → a handful (keep "identifier recovery, not pass@1"); delete hollow "In practice" closers. |
| P9b (Adversary critique 9) | **Recovery-vs-budget-W curve:** the harness logs prompt-tokens per arm per level, but only W=768 was run — so this needs a W-sweep (e.g. W∈{256,512,768,1536}) on the existing set, ~1–2 GPU-hours. **Accept if Phase 1 GPU budget allows** (it makes "advantage grows as the budget tightens" visual, directly strengthening the keystone); else decline explicitly with this one-line reason. |
| P12 | Optional: contributions → `enumerate`, keystone item set apart. |

---

## Cross-cutting: multiple comparisons (Adversary critique 11)
Phase 1 adds up to 3 pre-registered gates (a2_tail, filler, swap) + Phase 2a = 4 at α=0.05 → family-wise error ≈0.19. **Stance:** each gate is pre-registered against a *distinct, specified* alternative (channel parity; token-displacement; frequency confound; objective de-bias), so per-comparison α is defensible — but this will be **stated explicitly** in the reproducibility appendix, and if a reviewer prefers, Holm correction is trivial to add post hoc.

---

## Critical path

1. **Phase 0** — venue commit (author) + corpus-trajectory lookup (decides 2a) + mechanical fixes. ~half day + 1 decision.
2. **Phase 1** — `a2_tail` + filler + swap (the load-bearing runs). ~2 GPU-hours (Option B); +1–2 eng-days if Option A / NeurIPS.
3. **Phase 2a** — only if the corpus lookup clears; else 2b prose.
4. **Phase 3** — prose, after the venue decision; keystone/objective wording waits on Phase 1/2 outcomes. Optional W-sweep figure if GPU budget allows.

**If only one run happens: `a2_tail` (+ its filler control).** It determines whether the keystone is "weight-space beats the best prompt channel" or "channel is immaterial under oracle conditioning; the real claim is the 32k-infeasible regime." Either way the paper ends up honest; the difference is which sentence is the headline.

**What the Adversary changed vs. v1 (net):** killed the false retrieval-reframe (replaced with the oracle-conditioning framing, grounded in source); added the token-displacement filler control and a quantitative swap gate; upgraded ReasonCACHE Option B from a dodge to a theorem-engagement; corrected the NeurIPS effort estimate and serialized it before prose; moved the 2a feasibility check to a Phase-0 corpus lookup; made the encoder narrowing paper-wide; and added the §6.3 unrun-comparator commitment table, the W-sweep figure decision, and a family-wise-error stance.
