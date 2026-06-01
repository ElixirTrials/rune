# Issue #52 — Deliverable 1 findings: Doc2LoRA positive control (2026-06-01)

**Status: positive control complete; cause of #49 isolated; two design conjectures tested.**
Reviewer-signed-off. All runs + provenance in MLflow `issue52-d2l-control` (exp 56, local
tracking server). Experiment code + full reproduction steps are preserved on the orphan branch
`experiment/issue52-doc2lora-positive-control` (not in the working tree — it depends on the
external `SakanaAI/doc-to-lora` repo and its pinned stack).

## One-line

The adapter-as-episodic-memory bet is achievable with the **existing perceiver architecture**.
**Rune #49 failed because its training recipe did not train queryable episode memory** — it
optimized an edit/full-revision *emission* objective that rewards generic, code-driven behavior
— **not** because of the architecture, the probe, the base model, or any ill-posedness. And the
right thing to store in the adapter is the episode's **feedback-derived facts (goal / state /
tried / failures), not the code diff.**

## Method (positive control)

Reproduced Sakana's released Doc2LoRA checkpoint (`SakanaAI/doc-to-lora`, Gemma-2-2b-it,
checkpoint-80000) on this hardware and pointed our own recoverability-scorecard metric (shared
`mean_gold_logprob` math) at it. "matched/mismatch/zero" = adapter from this episode / a
different episode / no adapter. Margins are mean gold-token logprob (nats). Provenance (versions,
commit `baa85db4`, checkpoint SHA256s, `D2L_ATTN_IMPL` state) logged per run; a local eager-attn
patch to the Sakana repo was kept **inert** (flash path) for every reported number.

## What was established

| result | number |
|---|---|
| Unmodified Sakana NIAH reproduction (rougeL.f1) | **1.0** (matches their reported near-perfect) |
| Scorecard calibration — known-good needle m−mismatch | **+7.7 nats** |
| Sakana zero-shot on Rune episodes: goal / file / diff m−mismatch | +2.30 / +1.76 / +1.01 |
| Sakana zero-shot continuation (tail) m−zero | **+2.01** (Rune #49 tail was **−0.38**) |
| Base-family control (Sakana qwen_4b_d2l, Qwen3-4B) overall m−mismatch | +1.60 (≈ Gemma +1.69) |
| Rune #49 own checkpoint (reference): goal / file / diff m−mismatch | +0.0005 / +0.011 / +0.075 |

Calibration makes #49's margins interpretable: at real recall m−mismatch ≈ +7, so #49's
+0.0005…+0.075 are ~0.01–1% of a real signal = **noise**. **The probe is not blind.**

**Ruled out as the cause of #49:** probe-blindness, adapter/perceiver capacity, ill-posed facts,
and base-model family (Qwen ≈ Gemma). **Cause:** the training recipe/objective — kept as a
*bucket* (objective, data format, query supervision, scale, batch structure, initialization are
still entangled, not separated). The same perceiver family binds Rune's own facts when trained
with a queryable-recall objective; Rune's full-revision edit-reproduction objective does not (and
#49 showed it actively hurts recall: copy m−zero −0.31, generation mode-collapse to boilerplate).

## Conjecture verdicts (experimentally tested)

**C1 — "light-finetune a recall-capable hypernet rather than train trajectory from scratch":**
*Supported in premise, sharpened in practice.* A 150-step warm-started light finetune **preserves
recall** (NIAH retention 99.6%, code 103.9% — no catastrophic forgetting). **But plain answer-CE
re-primes generic emission on the hard facet:** diff m−zero +0.72 while diff *specificity*
(m−mismatch) **−0.25** — the #49 trap. So the warm-start path is viable, but the objective inside
it must be specificity-aware, not plain CE. (Did not capture ‖Δhypernet‖ — open gap.)

**C2 — "don't embed code diffs (it primes diff emission); embed goal / tried / failures /
last-N-lines":** *Confirmed.* Against a **feedback-swap** hard negative (same code/file, different
feedback), **diff** recall collapses +1.01 → **+0.17** (code-echo), while **goal** holds +2.30 →
**+1.59** (binds the trajectory fact). Precision: the diff is a bad **memory-supervision** target,
but remains a valid **downstream action/output** target. Architectural principle: **separate
memory (recall episodic state) from policy (base emits the next edit conditioned on it)** — this
is exactly the conflation #49 fell into.

## Caveats (carried forward)

- **Recall ≠ utility.** Everything measured is recallability (logprob specificity). We have *not*
  shown that recalling state improves next-edit generation / pass@1. That is the decisive product
  link and is unmeasured.
- **`avoid` / failure facets are data-gated.** The current corpus has no failure history
  (single-turn). Recalling "what was tried / why it failed" needs real engine trajectories with
  tried-and-failed steps; until then the `avoid` scorecard facet is untestable.
- **`last-N-lines` is state, not output.** Recall it as context; never train the base to emit it
  verbatim, or the emission-prime returns through the back door.

## Forward plan

1. **memory→next-edit utility test** — embed goal + tail state; measure whether the base
   generates/ranks the correct edit better than diff-embedded or no-adapter. Closes recall≠utility.
2. **Mine real engine trajectories with failure history** (decompose→plan→code→[diagnose→repair]
   →integrate): ordered steps, queries over prior steps, failure facts. Unlocks tried/failure
   recall and the `avoid` facet.
3. **Specificity-aware specialization objective** (contrastive/preference on edit-local tokens
   with constructed, facet-paired hard negatives) when training Rune's own memory hypernet —
   plain CE is insufficient (C1).
4. **Base-model decision (free variable; two lanes):** fastest-research = warm-start Sakana's
   released checkpoint; best-product = train Sakana's recipe on a strong code base.

## Reproduction

Orphan branch `experiment/issue52-doc2lora-positive-control` carries the probe scripts,
the shared scoring core, the episode builders, the MLflow logger, and a README with exact
commands (clone Sakana, pinned env, checkpoint download, run). Nothing there is wired into Rune.
