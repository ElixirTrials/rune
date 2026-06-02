# Doc2LoRA positive control — scorecard validation + Rune-episode bridge (Issue #52, Deliverable 1)

**Date:** 2026-06-01
**Issue:** [#52](https://github.com/ElixirTrials/rune/issues/52) — adapter-as-trajectory-memory: staged diagnostic, gated by the recoverability scorecard
**Status:** design approved; pending spec review → writing-plans

---

## 1. Context & goal

Issue #52 is a research epic: a 4-stage diagnostic (oracle → Doc2LoRA control →
three-initialization comparison → scale to Rune data) plus a parallel data track, all gated by
the **§4.7 recoverability scorecard** (from `docs/issue49-handoff-2026-06-01.md`): from the
adapter alone (episode NOT in the prompt), the base model must recover goal / diff / tail /
avoid, each with **m−mismatch > 0** (episode-specific) AND **m−zero > 0** (beats no-adapter).

Stages 3–4 depend on this stage's outcome, so they are out of scope here. This spec covers
**Deliverable 1 only.**

**Why this is first (a reorder of the issue's stated order).** The issue lists the oracle as
step 1. We invert: do the **Doc2LoRA reproduction first**. A Stage-1 oracle *negative* is
uninterpretable on its own — "the oracle can't pass the scorecard" could mean the target/data
is ill-posed (the signal we want) OR our probe cannot detect episode-specific recall even when
it genuinely exists. Our harness has only ever been shown to detect the *generic* effect (#49:
diff m−zero ≈ +1.17); it has **never** been validated to detect episode-*specific* recall
(m−mismatch > 0), because nothing has yet produced genuine recall. Reproducing a known-good
recall result and pointing our scorecard at it removes that ambiguity up front, making every
downstream negative interpretable.

**Goal of this deliverable:** prove our recoverability-scorecard methodology detects recall
when it genuinely exists, using Sakana's released Doc2LoRA checkpoint as the known-good
reference; then take a directional look at whether that architecture binds code-edit facts.

**Explicit scope boundary (reviewer, `reflections.md` 2026-06-01).** This deliverable is a
**probe-validation prelude, NOT a substitute for the Stage-1 oracle.** Doc2LoRA-first can kill
"our probe is blind," but it produces **no evidence about whether the Rune episode target is
well-posed** — only the oracle (direct per-episode optimization on a Rune target, no OOD gap)
answers "can a LoRA store this Rune target at all?" The reorder does not drop the oracle; it
sequences the probe validation before it. Nothing here may be read as a verdict on Rune target
well-posedness.

## 2. What we are reproducing

Sakana AI released both code and a trained checkpoint:
- **GitHub `SakanaAI/doc-to-lora`** — "Hypernetworks that update LLMs to remember factual
  information." Package: `ctx_to_lora` (same lineage as Rune's "ctx-to-lora" hypernetwork).
- **HF `SakanaAI/doc-to-lora`** — trained hypernetwork (~309M params, 8 cross-attention
  blocks), base model **Gemma-2-2b-it**, ~80k training steps; checkpoint at
  `trained_d2l/gemma_demo/checkpoint-80000/pytorch_model.bin`. Reported near-perfect
  needle-in-a-haystack (NIAH) recall at 5× the base context window.

API (from their README):
```python
state_dict = torch.load(checkpoint_path, weights_only=False)
model = ModulatedPretrainedModel.from_state_dict(state_dict, train=False)
model.internalize(doc)        # embed a document into the adapter
# forward/generate with the document REMOVED from the prompt
model.reset()                 # clear internalization
```
`internalize(matched_doc)` / `internalize(mismatch_doc)` / `reset()` (= zero) is exactly the
matched / mismatch / zero primitive the scorecard needs.

## 3. Architecture — isolation strategy

**Chosen: standalone sibling repo + its own uv environment.**

| Option | Decision | Reason |
|---|---|---|
| **(A) Standalone sibling repo + own venv** | **Chosen** | Keeps the control independent of Rune's suspect pipeline; sidesteps the `ctx_to_lora` name collision with Rune's module (separate venvs); our probe imports *their* API. |
| (B) Vendor checkpoint into Rune, run via Rune `_functional_lora` | Rejected | Defeats the independence the control exists for; adds Gemma-on-Qwen-path OOD risk. |
| (C) Their demo/eval only, no custom probe | Rejected | Fails the chosen success bar — does not validate *our* metric. |

Clone to `third_party/doc-to-lora/`, isolated `uv` venv there, `hf download` the checkpoint.
Gemma-2-2b-it (~2B) fits the 23GB GPU and 15GB CPU RAM comfortably (no 4-bit required).
Runs still go under `tools/run_guarded.sh` per CLAUDE.md (cheap insurance; capture output;
background for multi-minute jobs).

## 4. The one cross-cutting change — shared scoring core

Extract the scoring core — `_span_logprob` + span selection, **pure
`(logits, ids) → mean gold logprob`, torch-only, with zero Rune/Gemma dependencies** — into a
single module imported by **both** the Gemma control script **and**
`tools/diag_recoverability.py`.

Rationale (advisor): without a shared core, "validate our probe" silently degrades to
"validate *a* methodology." A bug in the real Qwen `_span_logprob` (off-by-one in the `t-1`
gold-token indexing; span-aggregation error) would survive untested, and we are back to the
exact "is our probe broken?" question this stage exists to kill. The core is pure tensor math,
so it runs unchanged in both venvs and does not break isolation.

## 5. Components & data flow

1. **Environment + reproduction.** Clone repo; isolated venv; download checkpoint (accept
   Gemma license if 403). **First integration step:** confirm per-token *logits* with the
   adapter active (`base_model(ids).logits` after `internalize()`, not just `.generate()`) —
   the entire scorecard depends on it and the README only shows `generate()`. Then run their
   NIAH eval (`scripts/niah/2-eval.sh`) and confirm recall ≈ their reported figure.
2. **Scorecard probe (validation).** Over a set of doc-fact episodes, for each: score the
   answer-fact span's mean gold logprob under `internalize(matched)` / `internalize(mismatch)`
   / `reset()` (zero), via the shared scoring core. Report m−mismatch and m−zero.
3. **Tiny Rune-episode bridge.** Build ~8–16 tiny Rune-style code episodes (small code
   doc/patch + QA over goal / file / pre→post diff); run the same probe on the Sakana
   checkpoint + Gemma.

## 6. Definition of Done

- **Reproduction:** NIAH recall on our hardware matches Sakana's reported figure within
  tolerance; number logged to `instructions/scratchpad.md`.
- **Probe validation:** on doc-fact episodes, m−mismatch and m−zero are **clearly positive**,
  not bare `> 0`. A tiny positive margin on a known-good control could be tokenizer/logprob
  noise or mismatch-sampling luck, so the criterion is a *statistically-clear* positive
  (reviewer): report **generation accuracy**, **per-episode margins**, **multiple mismatch
  controls per episode**, and a **standard error / bootstrap CI** on the mean margin. If the
  margin is not clearly positive on a known-good checkpoint, the *metric* is the problem and
  must be fixed before any Rune negative is trusted.
- **Calibration scale (first-class, not an afterthought):** record the *magnitude* of
  m−mismatch at known-good (~100% NIAH) recall, tied to generation accuracy. This calibrates
  whether #49's Qwen margins (+0.075 "weak", +0.0005 "noise") are nothing (if real recall
  ⇒ m−mismatch ≈ +2) or whether the logprob-margin metric is barely sensitive (if real recall
  ⇒ m−mismatch ≈ +0.1, the whole scorecard interpretation needs rethinking). Calibration is
  **effect size at known-good recall**, not just sign.
- **Bridge:** scorecard numbers on the Rune code episodes **+ explicit asymmetric
  interpretation** (see §7).
- **Record:** chronological working log appended to `instructions/scratchpad.md` — note that
  `instructions/` is a deliberate local working-notes directory and is gitignored
  (`.gitignore`), so the scratchpad is **not** committed; the auditable, committed artifact is
  the findings doc (`docs/issue52-findings-2026-06-01.md`) produced when the deliverable completes.

## 7. Interpretation rules (set before running)

- **Probe validation** PASS ⇒ scorecard trustworthy downstream. FAIL on known-good ⇒ fix the
  metric first (this stage caught a false-negative pipeline).
- **Calibration** is load-bearing: it converts the scorecard from binary pass/fail into a
  sensitivity-aware reading of the #49 margins.
- **Bridge is directional-only and asymmetric** (Sakana trained on document facts; code is
  out-of-distribution):
  - **PASS** → strong, surprising positive: this architecture binds code-edit facts zero-shot.
  - **FAIL** → nearly uninformative: likely just OOD, *not* "the code target is ill-posed."
  The clean ill-posedness test is the **oracle** (direct per-episode optimization, no OOD gap),
  not this bridge — stated so a bridge failure is not over-read later. The bridge is kept
  **small and non-gating** (reviewer): its outcome does not block or gate the oracle stage, so
  it cannot blur the cleaner oracle diagnostic.

## 8. Testing

- **TDD the deterministic pieces:** the shared scoring core (span-logprob math on toy tensors —
  the `t-1` off-by-one is exactly the bug class it guards), the episode dataset builder
  (schema/shape), and matched/mismatch pairing logic.
- Reproduction and probe runs are GPU integration runs under `tools/run_guarded.sh`, not unit
  tests.

## 8a. Operational hygiene (reviewer)

- **Secrets:** keep HF tokens entirely in the existing environment / HF credential store. Never
  write tokens into scripts or `instructions/scratchpad.md`.
- **No vendored artifacts in git:** do not commit the `third_party/doc-to-lora/` checkout,
  downloaded checkpoints, or `.venv` directories. Add them to `.gitignore` if not already
  covered.
- **GPU rule (CLAUDE.md) applies even though Gemma-2B "fits trivially":** `free -g` before any
  model load; GPU runs under `tools/run_guarded.sh`; capture/log output; background multi-minute
  jobs.

## 9. Open question for the implementation plan

Build the ~8–16 tiny Rune code episodes **from scratch** (clean, guaranteed to contain all
queried facts) vs **reformulate existing `external_codereview` rows as patch+QA** (connects to
the data track's "patches not full files", tests real-data-as-patches). Default lean:
reformulate a handful of existing rows, since it doubles as an early read on the patch
reformulation the data track will need — but each episode must actually carry the queried
goal/file/diff facts, or a "failure" is a data artifact.

## 10. Out of scope (depends on this stage's outcome)

Stage 1 oracle; Stage 3 three-initialization comparison; Stage 4 scale to real Rune trajectory
data; the parallel real-trajectory mining track. **Note for the oracle stage:** the Qwen
`_functional_lora` 4-bit application path needs its own numerical-equivalence unit test
(`base_out + delta` == patched forward) — Gemma never exercises it.
