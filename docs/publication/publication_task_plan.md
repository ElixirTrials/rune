# Publication task plan — Parametric Episodic Memory (Rune), TMLR submission

**Owner of record:** AI Researcher · **Venue:** TMLR (committed) · **Basis:** `HANDOFF_v13_review.md` (28 findings) → `remediation_plan_FINAL.md` v2 → Adversary review (`adversary_review.md`) → source-verified corrections. **Draft:** `drafts/paper_v13.tex`.

This is the execution tracker: what is left to get `paper_v13` submission-ready, who does each item, the pre-registered gate where one applies, and the order. Owner codes: **R** = AI Researcher (me), **A** = advisor/author decision I cannot make, **R+A** = I prepare, advisor signs off.

---

## Status legend
`run` = GPU experiment (frozen c3, existing/new harness) · `build` = code to write first · `prose` = text/bib only · `decision` = human call · `qa` = build/render check.

---

## Phase 0 — Decisions, corpus lookup, mechanical fixes (no GPU, unblocks everything)

| # | Task | Findings | Owner | Type | Gate / rule | Done-when |
|---|------|----------|-------|------|-------------|-----------|
| 0.1 | **TMLR build target.** Switch class out of `[preprint]`/real-names → anonymized TMLR submission mode; keep 15pp (no TMLR limit); delete staged `neurips_2026.sty`; rename folder off "NeuroIPS Submission". | B2, P1 | R | prose | — | Paper builds clean, anonymized, TMLR class. |
| 0.2 | **Corpus-trajectory lookup** — decides whether Phase 2a is feasible. Query the corpus manifest for MBPP tasks with *gold trajectories/adapters*; cross-ref against the objective-grid selection set and the N=60 keystone set; report the disjoint count. | M5.2 feasibility | R | (data) | **≥ ~50 disjoint tasks with trajectories → 2a runs; else → 2b prose path is the answer.** | Disjoint count reported; 2a/2b branch chosen. |
| 0.3 | SHA-256 hashes: re-verify c3 + corpus-split hashes vs MLflow artifacts; paste into Appendix A; strip `\todo`. (Copy-forward from paper_v8 checklist A4.) | M5.1, P3/C8 | R | prose | — | No `\todo` for hashes; hashes in Appendix A. |
| 0.4 | Rename `functional-49` → "LiveCodeBench-v6 full post-cutoff functional set (N=63)" everywhere; introduce the 49-item subset once, explicitly. | M6, P2 | R | prose | — | No term collision; subset defined once. |
| 0.5 | Algorithm 1 → real `algorithm` float + `\label{alg:loop}`; replace hardcoded refs. | P8 | R | prose | — | Float auto-numbered, `\ref` resolves. |
| 0.6 | Delete stale `figure2.pdf`; confirm `\includegraphics` resolves on clean build. | P9 | R | qa | — | Clean build, correct Fig. 2 embedded. |
| 0.7 | `shi2025revisit` → "Yaorui Shi et al."; drop unverified note. Downgrade unconfirmed `@inproceedings` → `@misc` (keep TC-LoRA workshop tag); fix `zhang2025ace` forward-date. | NOV-5, NOV-4 | R | prose | — | Bib entries corrected. |
| 0.8 | Use "12,836" (or "~12.8k") consistently; §6.1 → "12/63 to 16/63 (+4)". | P11/C9 | R | prose | — | Numbers consistent, direction correct. |

**Phase 0 exit:** all `\todo` macros gone or converted to stated limitations; paper builds clean in anonymized TMLR format; the 2a/2b branch is decided.

---

## Phase 1 — Keystone campaign (the scientific core; frozen c3, N=60)

The framing is fixed by source: the adapter's ~124-tok conditioning is **oracle-supplied** (`row.gold_snippet_index`; arm `episodic_use`, variant `use`, scaling 0.91, 31/60 = 0.517). The keystone is honestly a **channel comparison under oracle conditioning**, not a retrieval demonstration — that must be stated plainly in Setup (kills the retracted "internalizes retrieval" framing).

| # | Task | Findings | Owner | Type | Gate | Effort |
|---|------|----------|-------|------|------|--------|
| 1.1 | **`a2_tail` arm** — place the identical oracle string (variant `use`, ~124 tok) at the prompt **tail**, adjacent to cursor, within W=768. Report at matched cursor-code lengths vs floor. | **B1 (blocking)** | R | run | *episodic beats a2_tail, separated CIs* → keystone strengthened (weight beats best prompt channel, info held fixed). *a2_tail ≈ episodic* → reframe: channel is not decisive under oracle conditioning; the 32k-infeasible regime (adapter 0.433 where in-prompt is impossible) becomes the headline. | ~1 GPU-hr |
| 1.2 | **`a2_tail_filler` control** — 124 tok of neutral filler in place of the pointer, same displacement. Isolates the pointer's marginal contribution from "different tokens near cursor." | B1 (Adversary crit-4) | R | run | pointer effect = (a2_tail − a2_tail_filler); report it. | +~0.5 GPU-hr |
| 1.3 | **Swap / mutation control** — **build first** (not yet in harness; design-spec §8 prescribes "port PR #57 §8"; only `hotswap_adapter` exists today): rename gold identifier in `render_episodic` conditioning, add a `swap` arm; then run on the keystone subset. | C7 | R | build+run | `s`≈floor-CI → refutes frequency confound; between → report attributable fraction (e−s)/(e−f); `s`≈episodic-CI → keystone compromised, say so. | ~0.5 eng-day + ~0.5 GPU-hr |
| 1.4 | **ReasonCACHE / KV comparator** — **Option B (revised, prose)** for the TMLR round: (i) drop "we test that boundary directly"; (ii) name it a committed unrun comparator in the §6.3 table; (iii) **engage its rank-expressivity theorem** in one paragraph (concede premise + argue regime is not expressivity-limited, or challenge the single-layer-prefix assumption). Option A (run a prefix/KV arm) deferred to next round. | M4/NOV-1, W2 | R+A | prose | (iii) is required — without the theorem paragraph, Option B does not close M4. | prose (Option A: +1–2 eng-days if advisor elects) |

**Phase 1 exit:** a2_tail / filler / swap numbers in MLflow with Wilson CIs + paired McNemar; abstract & §5.4 rewritten to whichever gate outcome obtained; oracle-conditioning nature stated in Setup.

---

## Phase 2 — Statistical de-biasing & abstract↔body reconciliation

| # | Task | Findings | Owner | Type | Gate | Effort |
|---|------|----------|-------|------|------|--------|
| 2.1 | **Fresh-pool re-estimate of +0.105** — *only if 0.2 cleared*. Re-estimate c3 matched-log-prob on the disjoint pool; recompute across-task sign test. Do **not** generate new trajectories to chase p<0.05. | M5.2/C4 | R | run (cond.) | crosses p<0.05 → report de-biased, strip caveat; stays >0.05 (currently 0.064) → 2.2. | ~1–2 GPU-hr if feasible |
| 2.2 | **Objective prose downgrade** (fallback, write now): "a guarded matched-recall objective shows more consistent — though not yet de-biased — held-out gains…"; report sign-test p=0.064 as-is; remove "produces generalising held-out recall." | M1-obj | R | prose | — | — |
| 2.3 | **LCB reconciliation in the abstract:** state "strict superset / zero regressions" follows from the escalate control flow vs. raw base; adapter-isolated scale0→c3 is +4/−2, **McNemar p=0.69**, not significant at N=63. (Committed-but-unexecuted v10 fix.) Do **not** re-assert the retracted 10/49 win — functional-49 is a tie. | C5, M1 | R | prose | — | — |
| 2.4 | n=10 "validation" → "checked (non-significantly; n=10 underpowered, Fisher p=0.30) on a disjoint held-out split before locking the configuration." | C4 | R | prose | — | — |

---

## Phase 3 — Framing & bibliography prose (after 0.1; keystone/objective wording waits on Phase 1–2 outcomes)

| # | Task | Findings | Owner | Type |
|---|------|----------|-------|------|
| 3.1 | Lead parity with the **matched within-8k** comparison (episodic 0.600 vs ceiling 0.567, same 30 problems — clean win); report pooled 0.517 as conservative all-strata with the 32k caveat; state what population 12,836 is over (arm, strata, N). | M2/C3/P4 | R | prose |
| 3.2 | State the 32k-stratum avg context length behind 26.8× (≈20.6k/768); reword as raw prompt-length ratio vs an *infeasible* arm, not parity. | M3/P5 | R | prose |
| 3.3 | Specify what a2_clamp/a2_full prepend (full concatenated cross-file source vs filtered snippet). | C6 | R | prose |
| 3.4 | Ground the 32k "infeasible" claim: name the harness grading/context budget; say if it is a base-model limit or an eval-design choice. | C10 | R | prose |
| 3.5 | **Encoder claim narrowing, paper-wide:** "architecture is not what gates this" → "Perceiver beats mean-pool"; audit every architecture/encoder/gating/insensitive occurrence (intro, contributions, §5, §6), not just §3.2. Add to §6.3 unrun list + limitations. (Adversary Q4: acceptable scope cut only if done paper-wide.) | M5.3/NOV-* | R | prose |
| 3.6 | **§6.3 committed-unrun-comparator table:** rows for direct-PEFT (Gate-1), ReasonCACHE/prefix KV (W=768, N=60), LSTM/vanilla-transformer encoder — each "committed for next revision." Turns "next round" into a commitment, not a disappearance. | Adversary crit-8 | R | prose |
| 3.7 | Recast "third axis" (token vs KV/activation vs weight memory) or drop the numbered-axis language. | NOV-2 | R | prose |
| 3.8 | §5.3: stop citing Liu2022 (a PEFT-beats-ICL result) to support "ICL is strong when context fits"; cite an ICL-capability result or reword. | NOV-3 | R | prose |
| 3.9 | ReasonCACHE "without touching parameters" → "without updating the base model's weights." | NOV-6 | R | prose |
| 3.10 | Re-add v10-committed cites (TMEM as named direct-PEFT standard; Mosbach to single-family limitation) or ensure nothing implies they're cited. | P10 | R | prose |
| 3.11 | Abstract → ~200–250 words; split problem+mechanism / results+bound; move HumanEval+ & scaffold-split to intro. | P6 | R | prose |
| 3.12 | "X, not Y" 29 → a handful (keep "identifier recovery, not pass@1"); delete hollow "In practice" closers. | P7 | R | prose |
| 3.13 | *Optional* contributions → `enumerate`, keystone item set apart. | P12 | R | prose |

---

## Cross-cutting

- **Multiple comparisons (Adversary crit-11):** Phase 1 adds up to 3 gates + 2a = 4 at α=0.05 (family-wise ≈0.19). Stance: each gate tests a *distinct pre-registered alternative*, so per-comparison α is defensible — **state this explicitly** in the repro appendix; Holm correction is trivial to add if a reviewer prefers.
- **Optional W-sweep figure (P9b):** the harness logs prompt-tokens per arm per level, but only W=768 was run. A W∈{256,512,768,1536} sweep on the existing set (~1–2 GPU-hr) makes "advantage grows as the budget tightens" visual. **Accept if Phase-1 GPU budget allows; else decline explicitly.** | R | run (opt.)
- **Provenance to publish (committed in v10 review response):** grade JSONs + sha256s, c3 checkpoint. Confirm these are attached before submission.

---

## Critical path & GPU budget

1. **Phase 0** — ~half a day + the TMLR build (0.1) + one advisor decision on Option A vs B (1.4). The corpus lookup (0.2) gates Phase 2a.
2. **Phase 1** — the load-bearing runs. **~2 GPU-hr** for a2_tail + filler + swap (Option B). +1–2 eng-days if advisor elects Option A.
3. **Phase 2a** — only if 0.2 clears; ~1–2 GPU-hr; else 2.2 prose.
4. **Phase 3** — prose, after 0.1; keystone/objective wording waits on Phase 1–2 outcomes.

**Total GPU:** ~2 GPU-hr baseline (Phase 1 Option B), +1–2 hr if 2a runs, +1–2 hr if the W-sweep figure is taken. All on the single 4090, frozen c3 — no retraining.

**If only one run happens: `a2_tail` + its filler control (1.1 + 1.2).** It decides whether the keystone headline is "weight-space beats the best prompt channel" or "channel is immaterial under oracle conditioning; the real claim is the 32k-infeasible regime." Either way the paper is honest; only the headline sentence changes.

---

## Definition of done (submission-ready for TMLR)

- [ ] Anonymized TMLR build, clean compile, no `\todo`, Fig. 2 correct at print size.
- [ ] a2_tail (+filler) + swap runs logged; abstract & §5.4 match the realized gate outcomes; oracle-conditioning stated in Setup.
- [ ] Objective claim either de-biased (2a) or downgraded (2.2); LCB abstract reconciled to +4/−2, p=0.69; no retracted claims re-asserted.
- [ ] ReasonCACHE theorem engaged (1.4-iii); §6.3 unrun-comparator table present.
- [ ] Encoder claim narrowed paper-wide; parity led by the matched-8k comparison; 26.8× denominator stated.
- [ ] Bib corrected; multiple-comparisons stance in repro appendix; grade JSONs/sha256s/c3 checkpoint attached.

## Two items awaiting the advisor
1. **ReasonCACHE Option A vs B** (1.4) — prose engagement (TMLR-sufficient) vs. running a KV/prefix arm (+1–2 eng-days, stronger). My recommendation: **B for this round, A committed for next.**
2. **Subagent model "Fable-5"** — not resolvable to a live roster id; give me the mapping (or enable the model) and I'll bind it. No task here delegates, so this is not on the critical path.
