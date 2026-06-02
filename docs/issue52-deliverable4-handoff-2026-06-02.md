# Issue #52 — Deliverable 4 handoff: experimentation phase (2026-06-02)

Self-contained handoff. Companion durable docs:
- `docs/issue52-pretraining-facts-dossier-2026-06-02.md` — facts (§0–10) + research synthesis (A–F) +
  advisor-informed interpretation. **Read this for the numbers.**
- `instructions/scratchpad.md` — chronological working log (latest entries: resume-after-crash → scaler_B
  fix → deep-dive → opinion → corrections → entropy/capacity eval). **Read the last ~10 entries. Continue to log to it**
- `instructions/reflections.md` — reviewer critiques. The 2026-06-02 entries ("Pushback on the Gate
  Reframe", "Body and Directionality Gate Controls", "Pushback on Residual Encoding") are **incorporated
  into the experiment designs below** and must keep being monitored.

Branch `feat/issue52-doc2lora-positive-control`; PR #53; last commit `1553026f`.

---

## 1. The goal (north star — not a pivot; this has been constant)
Rune is a system **unbounded by the base model's context window** that **iterates until a problem is
solved**, where **each step of the trajectory is oriented by a swappable adapter**. The hypernetwork
adapter must serve *optimally* as that **substrate**: the embedded **code + broader context** must be
**accessible to the frozen base** at each step — goal (where we're headed), what we've tried and why it
failed, and the last action / current state (esp. resuming a generation cut off mid-stream). This escapes
the O(T²) attention cost of keeping everything in the prompt.

Success is NOT "bind external code-review feedback to an edit" — that was a **proxy corpus + a policy
signal** used as a diagnostic. Success is the adapter-as-substrate making code/context **accessible and
actionable** for the next step, at long horizons, without the prompt.

Design principle (kept): **separate memory from policy** — the adapter recalls episodic trajectory state;
the frozen base emits the next step conditioned on it.

---

## 2. What we've done

### 2a. Code/correctness fixes landed this session (commit `1553026f`)
- **scaler_B clobber fixed (the big one).** The distill loop called `reinit_scaler_b_nonzero(1.0)`
  *unconditionally* after a warm-start load, overwriting the checkpoint's learned `scaler_B` (mean|·|
  0.057, structured) with uniform 1.0 — a ~17× B-side inflation at `effective_scaling=lora_alpha=45.25`
  that destroyed the adapter (smoke held-out **matched−zero = −8.8**, uniform across episodes). Added
  `scaler_b_is_collapsed()`; reinit only in the zero-init basin, else preserve. Regression tests in
  `tests/unit/test_scaler_b_init.py` (4 pass; full unit suite 317 pass).
- **OOM fixed.** `max_seq_length` 2048→**768** (the contrastive loop's designed memory regime; its own
  comment says "keeps seq=768, peak ~ single-path"). Always export
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for GPU runs.
- Config wired `val_corpus_path`/`val_steps`/`save_steps`; `_bench_entry.py` per-task dump; `checkpoints/`
  gitignored.

### 2b. Key empirical state (see dossier for full numbers)
- **Adapter apply contract validated** (anchors 1–3): `effective_scaling = lora_alpha = 45.25`; engine
  PEFT-hotswap is logit-identical to the functional contract.
- **Warm-start (qwen_4b_d2l) zero-shot recall is strong on the easy facets, weak on the hard one:**
  goal m−mismatch **+2.30**, file +1.76, diff +1.01, tail/continuation m−zero **+2.01**, hidden-task
  specificity **+1.17**, **signature +3.84 vs body +0.14**. NIAH reproduction = 1.0; calibration of a
  known-good recall ≈ **+7.7**.
- **Fixed 60-step smoke (MLflow `cbe9da363c`):** healthy mechanics (scaler_B preserved == warm-start, KL
  ~2, diff_agreement 0.364). Held-out feedback-binding **matched−swap +0.0687 (frac 0.65)** vs warm-start
  baseline **+0.0185 (frac 0.48)**; matched−zero +0.50. Per reviewer pushback: read this as
  **"insufficient and confounded," NOT "#49 proven again"** — there *is* a small held-out movement, unlike
  #49's near-zero. The "14%/86% feedback-specific vs generic" split is a **warning sign, not a rigorous
  attribution** (matched−zero and matched−swap don't additively partition one lift).
- **Feedback→edit signal on `external_codereview` is weak** across three measures: oracle in-prompt ceiling
  +0.17 (frac 0.53), in-train margin flat ~1.0, held-out +0.0687. Caveat (pushback): this is evidence
  against **this corpus + objective + metric combination**, not against the corpus in all forms — could
  also be wrong hard-negatives, unnormalized feedback, or an edit-local metric that misses the action
  signal.

### 2c. The conceptual model: four distinct levers (don't conflate them)
The single weak number "body recall +0.14" hides four independent axes:
1. **Capacity** (how much the substrate *can* hold) — lever: rank / D2L chunk count (Parametric Memory
   Law: recall loss ∝ rank; chunking concatenates per-chunk adapters along the rank dim, so effective rank
   scales with chunks).
2. **Representation** (what the *doc-Q&A-trained* compressor *chooses* to keep) — by rate-distortion, a
   QA/NIAH compressor is rewarded for discarding the verbatim middle, i.e. code-body detail. Lever:
   **fine-tune** the hypernetwork. Rank can't fix a function that doesn't encode bodies.
3. **Directionality** (does it encode the *arrow* were→heading, not just an order-agnostic *set* of facts)
   — doc-Q&A never rewarded encoding causal/temporal direction. Lever: **directional fine-tune objective /
   arch**. Strongest a-priori case for fine-tuning; never measured to date.
4. **Entropy/efficiency** (do we embed every character?) — **No.** Code is low-entropy given the base's
   prior; spend rank on the high-information **residual**, not base-predictable scaffolding. Lever:
   **per-token importance-weighted recall loss** (CaMeLS/MemFT/LLMLingua-style). Multiplies lever 1.

Honest caveat across all four: the warm-start *transfers partially* (+2.30/+1.01/+2.01), so fine-tuning is
**adaptation, not from-scratch**, and any fine-tune must pass a **retention gate** (don't erase the
NIAH/QA/tail recall prior) and a **generation-stability gate** (xgrammar pass@1 not degraded).

---

## 3. How to proceed — the experimentation phase (critique-hardened)

_All experimentation logged to MLflow and results, plans and interpretations to scratchpad.md

**Sequencing:** run the cheap diagnostics (T0, E1, E2) *first* — they decide *which* lever to invest GPU
in. Do **not** launch a long/unfiltered training run; the evidence does not yet justify it. Predeclare all
subset/scoring rules **before** looking at trained-checkpoint deltas (leakage rule). Every comparison
includes a **positive control** (a curated case the apparatus *should* pass) so a null result distinguishes
"weak signal" from "broken harness."

### Calibration ladder (use THIS, not NIAH +7.7, as the yardstick)
Reviewer pushback: edit-feedback binding ≠ needle retrieval, so don't demand +7.7. Judge movement on:
known-good **hidden-task specificity +1.17** → **feedback-swap diff collapse +0.174** → **in-context
directive ceiling +0.52** → **fixed smoke +0.0687**. Demand meaningful movement *up this ladder*, plus
retention + generation-stability thresholds, all written down before any training run.

### T0 — paired significance of the existing smoke (cheap; do first)
Both checkpoints exist (`checkpoints/issue52-recipe-4b/checkpoint.pt` = fixed smoke; warm-start =
`third_party/doc-to-lora/trained_d2l/qwen_4b_d2l/checkpoint-20000/pytorch_model.bin`). Compute the
**per-episode paired delta** (trained − baseline matched−swap) on the **exact same val rows**. Report
**paired bootstrap CI + sign test + row-level scatter** (margins are heavy-tailed; a t-test alone is
insufficient). Verify the 60 rows are byte-identical across both evals after any truncation/filter. This
tells us whether +0.0687 is a real per-episode improvement or sample-composition noise. (Tool to adapt:
`tools/_feedback_swap_eval.py` — add a per-episode dump + a second-checkpoint arm.)

### E1 — capacity vs representation (the rank-vs-fine-tune discriminator)
Oracle per-episode LoRA vs hypernet-generated adapter, **at matched rank**, scored on **code-body /
high-information tokens**. Critique-hardened:
- The oracle MUST train on the *same hidden-code facts* and be scored with the *same masks, negatives, and
  prompts* as the hypernet adapter — otherwise it's per-instance-gradient vs amortized-generation, not a
  clean result. **Report the oracle as an upper bound, not as proof the hypernet objective is wrong.**
- **Cross-over control:** if the oracle succeeds at r=8, also run a *tiny hypernet fine-tune on those exact
  facts*. If a few updates move the hypernet (warm-start didn't) → **objective mismatch (fine-tune)**. If
  it still doesn't move → **architecture/conditioning attenuation** is back on the table.
- Decision: oracle good @r8 + hypernet bad @r8 → representation wall (fine-tune). Oracle bad @r8, good
  higher → capacity (raise rank/chunks). Both bad even high rank → data/architecture.

### E2 — directionality (never measured; likely the central gap)
Does the adapter encode the *arrow*, not just the facts? Critique-hardened:
- Use **minimally-edited counterfactuals** that preserve tokens and local code and change *only* the causal
  arrow / next-action implication. Do NOT rely on time-reversal or were↔heading text swaps alone — they
  inject lexical artifacts the model rejects without understanding direction. Include a **same-bag-of-events
  control**.
- **Score on action consequences, not recall.** "What happened first?" can show order-retrieval while still
  failing the product. Decisive readout: does the adapter shift next-step code/action tokens **in the
  direction implied by the prior failure / partial trajectory**, vs the in-prompt ceiling.

### E3 — capacity sweep (sizes the lever-1 knob)
Body/code accessibility as a function of **D2L chunk count (1 vs K)** and **rank**, scored on informative
tokens. Confirms whether chunking actually buys code-body capacity.

### E4 — encoding efficiency (the entropy lever; run once fine-tuning is on the table)
Critique-hardened — **do not equate high-surprisal with useful**:
- Compare **≥3 weighting families**: (a) base-surprisal, (b) action/discriminative-token labels, (c)
  learned/proxy *utility* weights (CaMeLS-style, meta-learned against downstream next-step utility).
- **Target = downstream body/direction UTILITY, not weighted reconstruction loss.** If surprisal-weighting
  improves likelihood on rare identifiers but not next-step code choice, it's a compression trick, not a
  substrate gain.
- **Unit = utility-per-rank, not bits-per-rank.** Useful memory is counterfactual: would the encoded fact
  change the next action vs a plausible *wrong* continuation? Score matched vs hard negatives that **preserve
  surface rarity but alter the semantic constraint**.
- **"Small-token, big-effect" negative control (mandatory):** flipped inequality, missing `not`,
  int-vs-string return, inclusive/exclusive boundary, which exception to catch. A residual encoder that
  drops these (because the base finds them predictable) looks efficient but fails the product.
- **Canonicalization is dangerous, not free:** formatting/whitespace normalization is likely safe; comment
  stripping / AST abstraction / literal normalization can delete intent, contracts, and edge-case facts.
  Run canonicalized vs raw-active-code arms side by side; **never normalize identifiers** (exact-name
  recall is our +3.84 asset); predeclare which regions may be lossy.
- **Compression must be local-state aware:** active/cutoff code + current-failure facts → near-verbatim;
  distant helper code → summary + signature/contract. No single global weighting policy.

### The hard facet — failure-bearing trajectory data (run IN PARALLEL, it's the product differentiation)
"What we tried and why it failed" *is* critique binding, so the same walls (false negatives, low
feedback→action MI, single-exposure binding) will recur. The corpus must be constructed so the
**failure-reason genuinely determines the next action** (this is the one durable lesson from the
feedback→edit work — it's now a corpus-design requirement, not a discarded test). Sources: real engine
trajectories (the mining lane) + synthetic action-determining failures with provenance labels. Easy MBPP
cannot supply these (structural-repair-only finding). **Add a positive-control episode** where the swapped
critique clearly changes the correct edit and the in-context ceiling is large — if the harness can't move
on that, the apparatus is broken.

### Conditional training run (only after the gates)
If E1/E2 say fine-tune (and/or raise rank), train the substrate on real trajectories with **cross-episode
hard negatives** (derangement; the hidden-task +1.17 regime), an **importance-weighted recall objective**
(E4 winner), selecting on **body/code-content + directional utility matched-vs-mismatched-episode** (NOT
`val_diff_agreement`, which is the matched-vs-base discipline confound; NOT signature-dominated spans).
Gate on the calibration ladder + retention + generation stability, written down first. Periodic saves;
post-hoc selection.

### De-prioritized / demoted (don't burn GPU here)
- The feedback-swap **paired test (T0)** is worth doing once for rigor, but do **not** keep optimizing
  feedback→edit on the unfiltered proxy corpus.
- **T1 (directive-feedback stratification)** is demoted *as a feedback→edit test*, but its construction
  lesson is load-bearing for the failure-bearing corpus. If you do run it: freeze the subset definition
  before looking at trained deltas, apply train/val separately, and **report both filtered and full-val**.

---

## 4. How to run things (infra)
- **Env:** `uv run` for all Python. `uv sync --extra gpu` (plain `uv sync` prunes trl/bnb/flash-attn).
- **GPU/RAM rules (CLAUDE.md):** single 22 GB GPU; ~15 GB CPU RAM (tiny). Check `free -g` before loading;
  `offload_base=False`; base in 4-bit. Always run multi-minute jobs under the RAM watchdog
  `tools/run_guarded.sh <log> <script> [args]` (kills before VM OOM). Export
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. `max_seq_length=768` for training.
- **Training:** `tools/run_guarded.sh /tmp/run.log tools/_distill_entry.py --config
  configs/issue52_recipe_mvc_4b.yaml [--max-steps N]`. Metrics → `checkpoints/issue52-recipe-4b/
  distill_metrics.jsonl` + MLflow experiment `issue52-recipe` (server http://localhost:5000).
- **Feedback-swap eval:** `tools/_feedback_swap_eval.py --ckpt <path> --n 60` (held-out matched/swap/zero
  on edit-local tokens).
- **Probes/harnesses to reuse:** `tools/_specificity_probe.py` (matched/mismatch/zero, signature/body span
  split, task-in-prompt vs hidden); `third_party/doc-to-lora/rune_episode_recall.py` (positive-control
  recall harness — check it for existing continuation matched/mismatch/**ceiling** numbers before new GPU);
  `tools/_temp_sample_avoid_probe.py` (avoid-pair harvest). **Scratch tools (`tools/_*.py`) stay LOCAL** —
  they have ruff errors and would break `ruff check .` in CI; the handoff/scratchpad record their
  paths/commands/MLflow IDs for repro.
- **Corpus:** `/tmp/rune-corpus/external_codereview.{train,val.clean,test.clean}.jsonl` (proxy; val.clean =
  323 rows). Bench: `benchmarks/mbpp_*.json`.
- **Checkpoints:** warm-start `…/qwen_4b_d2l/checkpoint-20000/pytorch_model.bin`; fixed smoke
  `checkpoints/issue52-recipe-4b/checkpoint.pt`; **documented FAILED** clobbered run
  `/tmp/smoke_broken_scalerB1.pt` (do NOT use for any trend/comparison).
- **Working method (from D2 handoff):** ALWAYS write observations/plans/interpretations to
  `instructions/scratchpad.md`; **monitor `instructions/reflections.md` and respond to its critiques.**

---

## 5. Pitfalls / lessons (so they don't recur)
- **Never reinit a warm-start's learned `scaler_B`** — guarded now + regression-tested, but the bug class
  (clobbering learned warm-start params on load) is easy to reintroduce.
- **seq 2048 OOMs**; the contrastive path is built for 768.
- **Don't select on `val_diff_agreement`** (matched-vs-base = discipline/generic-boost confound).
- **Signature recall (+3.84) inflates any aggregate score** — always score on body / informative /
  action-determining tokens, never full-span, or the surface shortcut wins.
- **Aggressive gisting/canonicalization breaks fine-grained recall** — keep it conservative, local-state
  aware, identifiers never normalized.
- **High-surprisal ≠ useful**; **matched−zero and matched−swap don't additively partition a lift**; use
  paired stats with bootstrap CIs, not eyeballed means or t-tests alone.
- **Predeclare subset/scoring rules before looking at trained deltas** (leakage); always include a
  positive control so a null distinguishes weak-signal from broken-harness.
- **Every fine-tune needs a retention gate** (NIAH/QA/tail) + generation-stability gate.
