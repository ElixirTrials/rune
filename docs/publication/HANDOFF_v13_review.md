# Handoff: adversarial review of `paper_v13.tex`

**To:** AI Researcher (record-of-owner, Parametric Episodic Memory / Rune)
**From:** Adversary
**Scope:** Full critical pass over `drafts/paper_v13.tex` (the current draft), cross-read against `latest_benchmarks.md` and `review_response_v10.md`. Three parallel review tracks — statistics/claims, novelty/citations, presentation/consistency — plus my own independent verification of the arithmetic, the statistics, and every 2025–2026 citation.

**One-line disposition:** The paper is honest, well-instrumented, and arithmetically clean, but it is **not submission-ready**. Two items are blocking: an unresolved venue/format decision, and a keystone baseline (`a2_clamp`) that is not the strongest in-prompt competitor and therefore does not yet support the headline claim as worded. Everything else is fixable in prose or with a small number of additional runs.

---

## What I verified myself (so you can trust the rest)

- **All reported statistics are arithmetically correct.** I recomputed every Wilson interval in Tables 4–7, the McNemar exact p-values (episodic-vs-floor 23/1 → p = 2.98×10⁻⁶ ≈ the reported 3.0×10⁻⁶; LCB +4/−0 → 0.125), the 16.7× ratio (12,836/768), and the stratum sums (8k 18 + 32k 13 = 31/60). No errors.
- **No fabricated references.** I fetched all 14 flagged 2025–2026 citations from arXiv/OpenAlex; every one resolves to a real paper with matching title, lead author, and ID (Doc-to-LoRA 2602.15902, SHINE 2602.06358, TC-LoRA 2510.09561, Ouroboros 2604.02051, ReasonCACHE 2602.02366, Transformer² 2501.06252, T2L 2506.06105, TTT 2512.23675, and the rest). Citation keys: 25 used = 25 defined, no orphans or dangling refs.
- **The new statistical points below verify too:** across-task sign test binom(17,24) = 0.064; the n=10 held-out check (4/10 vs 1/10) is Fisher p = 0.30 / z-test p = 0.12 (i.e. not significant on its own); the adapter-isolated LCB contribution (scale0→c3, +4/−2) is McNemar p = 0.69.

Credit where due: the mutated-spec control, the naïve-dump null, the sign-test and winner's-curse caveats, and the scale0 attribution arm are exactly the self-critical instrumentation a strong methods paper should carry. The soft spots below are mostly about **framing outrunning the evidence**, not about the evidence being wrong.

---

## BLOCKING — resolve before any submission

### B1. The keystone's in-prompt baseline (`a2_clamp`) is not the strongest achievable design
*(Stats C1 — the single most important finding in this review)*

The keystone claim ("adapter matches full-context prompting at 16.7× fewer prompt tokens") is a three-way comparison whose persuasive force rests on `a2_clamp` being the best an engineer could do with a prompt-only channel under the same 768-token budget. It is not.

- `a2_clamp` **prepends** the full cross-file context (avg ~12,836 tokens) to the head of the prompt, then truncates to the last 768 tokens — so the prepended context is *always* the first thing evicted. By construction it can almost never survive.
- Meanwhile the adapter's own conditioning surface is **124 tokens on average (median 52)** — it fits in the 768 budget six times over, with 644 tokens to spare.
- **Nothing in the four-arm design tests the obvious honest alternative:** place that same compact, symbol-specific pointer (the identical ~124-token string the hypernetwork gets) at the *tail* of the prompt, adjacent to the cursor, within the same budget — no eviction needed.

As constructed, the experiment demonstrates *"a multi-thousand-token cross-file dump, prepended, is a bad way to spend a 768-token budget"* — true, but not the same as *"the adapter beats the best achievable in-prompt channel."* The honest comparator is functionally almost identical to the adapter's conditioning, just delivered through the prompt instead of through weights; a reviewer will demand it.

**Fix (one run):** add an `a2_pointer` / `a2_tail` arm — identical episodic conditioning text placed at the prompt tail within W=768 — and report its recovery beside `a2_clamp`. Two outcomes, both publishable:
- If the adapter's advantage **survives** against the tail-pointer arm, the keystone gets dramatically stronger (you've beaten the real baseline).
- If it **collapses**, reframe the contribution honestly as *"weight-space delivery removes the need to know in advance what to place near the cursor"* rather than *"weight-space delivery beats prompt delivery under a budget."*

This is the highest-value single experiment you can run before submission. I'd prioritize it over everything else here.

### B2. The venue / format decision is unmade and the current build is internally contradictory
*(Presentation P1)*

- Document class is `tmlr` in `[preprint]` mode → renders full author names + affiliations.
- But the folder is `NeuroIPS Submission`, a `neurips_2026.sty` sits beside `tmlr.sty` in `drafts/`, the title declares no venue, and the compiled PDF is 15 pages.
- Every plausible target implies a *different, currently-unmet* format:
  - **TMLR:** review is double-blind on OpenReview → you need the anonymized submission mode, not `[preprint]` with real names. (No page limit, so 15 pp is fine.)
  - **NeurIPS 2026:** requires the NeurIPS class (not tmlr), double-blind, and a ~9-page main-text limit that 15 pp violates by a wide margin.

A venue/anonymization/length mismatch is a **desk-reject risk that fires before any reviewer reads the science.** Decide the venue, build for it, delete the other `.sty`, and rename the folder to match.

---

## MAJOR — a reviewer will demand these in rebuttal

### M1. Abstract overstates two results the body correctly hedges
*(Stats C2, C5)*

The body is careful; the abstract is not, and most reviewers triage on the abstract.

- **Objective contrast (C2):** the abstract states "a guarded matched-recall objective produces generalising held-out recall" as settled fact. But c3 was selected from a 4-point grid (winner's curse), its CI [+0.033, +0.182] is computed on the *same selection data* (the fresh-pool re-estimate is still an open `\todo`), and the across-task sign test is p = 0.064 — not significant. Soften to match §5.2's own hedging, or land the fresh-pool re-estimate first.
- **LCB "strict superset / zero regressions" (C5):** this is **guaranteed by the escalate control flow** (the base's zero-shot answer is kept if it passes; the adapter only engages on repair), not empirical evidence the adapter helps. The real, weaker finding is two sentences later: isolating the adapter (scale0→c3) gives +4/−2, net +2 with churn, McNemar p = 0.69 — noise. `review_response_v10` (W1) *committed* to labelling "zero regressions" a design property, not evidence; that framing is not present in v13. Add the clause at first mention in the **abstract**, not just the body.

### M2. "Parity with the ceiling at 16.7×" mixes populations
*(Stats C3 = Presentation P4)*

`a2_full` (0.567) is defined only on the 30 problems in the 8k stratum. The pooled episodic number (0.517) averages over both strata and is dragged down by the harder 32k stratum. So "0.517 at parity with 0.567" compares a two-stratum episodic number against a one-stratum-only ceiling — a population mismatch. Worse, it **buries your stronger result:** in the *matched* 8k comparison (Table 5) the adapter (0.600) actually *exceeds* the ceiling (0.567) on the same 30 problems.

**Fix:** lead the parity claim with the matched within-8k comparison (0.600 vs 0.567, same 30 problems — a clean win), then report pooled 0.517 as the conservative all-strata figure with the 32k caveat. Also state explicitly what set the 12,836-token average is computed over (which arm, which strata, N).

### M3. The 26.8× figure has no stated denominator and is not a parity comparison
*(Presentation P5)*

Unlike 16.7× (which cites 12,836 tokens), 26.8× appears with no token count anywhere, and at 32k `a2_full` is undefined (recovers nothing) — so there is no recovery-matched arm for it to be "26.8× shorter than at parity." Presented alongside 16.7×, it reads as the same *kind* of quantity; it isn't. State the 32k-stratum average context length (~20.6k tokens / 768 = 26.8×) and reword as a raw prompt-length ratio against an infeasible in-prompt arm, not a parity-matched reduction.

### M4. ReasonCACHE is named as the central comparator, then never run
*(Novelty NOV-1)*

Related Work cites ReasonCACHE as a KV-cache method that "matches or exceeds LoRA-based updates … without touching parameters," immediately followed by "Which channel wins is empirical … We test that boundary directly." But the keystone tests only adapter vs in-prompt-clamp vs floor — never a KV-cache/prefix arm. Worse, ReasonCACHE's abstract argues it "can be strictly more expressive than low-rank weight update since the latter ties expressivity to input rank" — a direct theoretical challenge to your core mechanism that you cite but don't engage. This is the sharpest instance of the prior review's W2. Either (a) add a ReasonCACHE/prefix arm under W=768, or (b) drop "we test that boundary directly," state the keystone compares only against the in-prompt channel, and list KV-cache methods as an unrun comparator alongside the already-conceded Gate-1 direct-PEFT arm. Also engage the rank-tied-expressivity argument rather than citing it only as support.

### M5. Three unresolved `\todo` markers, two of them camera-ready-blocking
*(Stats C8 = Presentation P3; overlaps C4)*

Red `\todo` macros cannot ship, and these correspond to *committed* actions from `review_response_v10`:
1. **SHA-256 hashes** for checkpoints/corpus (Appendix A promises them; the repro claim is currently uncheckable). **Mechanical — do it now.**
2. **Fresh-pool re-estimate of +0.105** — this is the action that resolves M1's objective-contrast overclaim. Run it, or explicitly downgrade §5.2 to "suggestive on a selected configuration" if the sign test stays >0.05.
3. **LSTM / vanilla-transformer encoder ablations** — §3.2's "the architecture is not what gates this" currently rests on a single alternative (mean-pool). Either run them or narrow the claim to "Perceiver beats mean-pool."

### M6. `functional-49` names two different objects
*(Presentation P2)*

In Setup, "LiveCodeBench-v6 functional-49" names *the* end-to-end benchmark. In §5.6 the benchmark is "the full functional set, N=63" and functional-49 is demoted to a 49-item *subset*. A reader can't reconcile them, and it sits on the headline +4 result — reading as a moved goalpost. Pick one canonical name (I'd use "LiveCodeBench-v6, full post-cutoff functional set, N=63" everywhere) and introduce the 49-item subset explicitly if it's worth keeping.

---

## MINOR — polish, precision, and one more control worth running

- **Add a conditioning-swap control to the RepoBench keystone** *(Stats C7).* The naïve-dump null shows *format* matters, but doesn't rule out a generic-output-bias/frequency confound — that identifier recovery is inflated because the conditioned symbol is a high-prior identifier the base already favors. Given your own MBPP finding that spec-absent "recall" is 9/19 memorization, a reviewer will ask. Condition on the *wrong* task's symbol on a subset and confirm recovery drops to floor. Directly insulates the 0.517 number from the memorization objection.
- **"Validated on 4/10 vs 1/10" overstates an n=10 check** *(Stats C4).* Fisher p = 0.30 — it could not have rejected the null. The real protection against overfitting is procedural (disjoint splits), not statistical. Reword to "checked (non-significantly; n=10 is underpowered) on a disjoint held-out split before locking the configuration."
- **Specify exactly what `a2_clamp`/`a2_full` prepend** *(Stats C6)* — full concatenated cross-file source vs a filtered snippet. Needed to audit whether the arms are apples-to-apples (same information, different channel) or the prompt arms carry much more, less-filtered content than the adapter ever sees.
- **Ground the 32k "infeasible" claim in the paper itself** *(Stats C10).* The constraint is a 12k-token grading budget in the harness (documented only in `latest_benchmarks.md`), not a stated base-model limit — the paper never states Qwen3-4B's context window. Name what governs "infeasible" in the manuscript.
- **"Third axis" framing is mis-partitioned** *(Novelty NOV-2).* By your own description RAG operates in the same token/attention substrate as context scaling (one axis, not two), and the framing omits the KV-cache/prefix family where ReasonCACHE sits. Recast as token-memory (window + RAG) vs KV/activation memory vs weight memory, or drop the "numbered axis" language. The contribution doesn't need the trichotomy.
- **Liu2022 cited against its grain** *(Novelty NOV-3).* In §5.3 it supports "ICL is strong when the context fits and is read once," but Liu2022's thesis is that PEFT *beats* ICL. The Related Work usage is fine; only §5.3 is slanted. Cite an actual ICL-capability result there.
- **Bibliography venue tags** *(Novelty NOV-4, NOV-5, NOV-6).* Four entries typed as `@inproceedings` can't be venue-confirmed (OpenAlex resolves each only to the preprint), and `zhang2025ace` is forward-dated to "ICLR 2026" — downgrade unconfirmed ones to `@misc`/`@article` until acceptance is verifiable (TC-LoRA's NeurIPS-workshop tag *is* corroborated, keep it). `shi2025revisit` first author is **Yaorui Shi**, not "Yu Shi" — fix and drop the self-flagged "not independently verified" note. And "ReasonCACHE … without touching parameters" overstates (prefix tuning *does* learn parameters) — reword to "without updating the base model's weights."
- **Dropped v10-committed citations** *(Presentation P10).* TMEM, Mosbach 2023, Meta-Tool, Override Gap were committed in the v10 rebuttal and are absent from v13. The "direct-PEFT comparator still open" limitation is stronger when it names TMEM as the standard; "one family, one domain" is stronger with Mosbach. Re-add or ensure nothing implies they're cited.

---

## NITS — one-word / cosmetic

- **Abstract is a single 432-word, 14-sentence block** *(P6)* that opens mid-mechanism before stating the problem. Tighten to ~200–250 words, split into problem+mechanism / results+bound.
- **Prose leans on one device** *(P7):* 29 comma-antithesis "X, not Y" constructions, 14 "rather than", 5 paragraph-closing "In practice". Keep the load-bearing ones ("identifier recovery, not pass@1" earns its place); rewrite ~half the rest as plain declaratives; delete the hollow "In practice" closers. This is a recognizable AI-writing tell reviewers increasingly flag.
- **Algorithm 1 is a hand-built `\fbox`, number hardcoded** *(P8).* Wrap in a real `algorithm` float with `\label{alg:loop}` so numbering/refs don't silently break.
- **Contributions are a dense ~225-word prose block** *(P12).* Consider an `enumerate` with the keystone item set apart, since reviewers lift contributions verbatim.
- **Figure hygiene** *(P9):* an unreferenced stale `figure2.pdf` sits beside the correct `figure2_repobench.pdf` in both `figures/` and `drafts/` — delete it so a clean rebuild can't grab the wrong one. Eyeball compiled Fig. 2 at print size (axis labels, four-arm labels, Wilson whiskers, arm colors matching text). Consider a recovery-vs-budget-W curve to make the "advantage grows as the budget tightens" claim visual rather than tabular.
- **"~13k" vs "12,836"** *(P11 = C9):* use one figure consistently. And §6.1 writes "16/63 to 12/63" (larger first, reads backwards) — rephrase to "12/63 to 16/63 (+4)".

---

## Suggested order of work

1. **B1** — run the `a2_tail`/`a2_pointer` arm. Highest value; determines how the keystone is framed. *(one run)*
2. **B2** — decide venue, build for it, clean the tree. *(no science, do it early)*
3. **M5.1** — publish SHA-256 hashes. *(mechanical)*
4. **M5.2 / M1 / C4** — fresh-pool re-estimate of +0.105; then reconcile the abstract's objective and LCB claims with the body. *(one run + prose)*
5. **M2, M3, M6** — fix the strata-mismatch framing, the 26.8× denominator, the functional-49 naming. *(prose only)*
6. **M4** — either run a ReasonCACHE/prefix arm or soften "we test that boundary directly." *(run or prose)*
7. **Minor C7** — conditioning-swap control on RepoBench, if time allows. *(one run, strengthens the keystone against the memorization objection)*
8. Remaining minors + nits — prose pass.

Items 1, 4, 6, 7 are the runs; everything else is text. If only one run happens before the deadline, make it **B1**.

*Full per-finding detail (locations, rationale, exact recomputed statistics) is in the three structured review artifacts saved alongside this handoff.*
