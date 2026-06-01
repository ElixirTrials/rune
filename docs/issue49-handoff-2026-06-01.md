# Issue #49 Handoff — HyperLoRA adapter-as-memory: exactly what we did

Literal and detailed. No summary-speak. Reading time ~15 min. Everything here is
reproducible from the commands in §8.

---

## 0. State in one sentence

We trained the hypernetwork on single-turn code-review pairs and measured, seven ways,
whether the generated LoRA adapter encodes the *specific* episode it was conditioned on —
including a direct episodic test of whether you can query the episode back out (goal, edit,
recent-state-that-drives-the-next-step, and what-was-rejected). It does not: the adapter is a
generic "make an edit here" booster; the episode-specific signal moves the generated weights
by ~0.4% and none of the four recoverability targets clear the episode-specific bar (§4.7).
The training corpus is also single-turn (no trajectories, no failure history), so it cannot
supply "don't repeat what failed" memory regardless of the model.

---

## 1. The goal (concrete)

The research bet (PRODUCT.md): encode a coding *trajectory* into a LoRA adapter via a
perceiver hypernetwork ("ctx-to-lora" / HyperLoRA), so that hot-swapping that adapter into
the base model makes the model behave as if it remembers that trajectory. Success would
look like Sakana's doc-to-lora: embed a document into an adapter, then recover facts about
that document from the base model with the document no longer in the prompt.

Operationalized for code review: given a context (code + a reviewer's request), generate an
adapter such that the base model — **with the context removed from its prompt** — produces
the specific revision the reviewer asked for. The discriminating test: the adapter built
from episode A must help reproduce episode A's edit *more* than an adapter built from a
different episode B.

---

## 2. What one training example actually is

The corpus is `external_codereview` (GitHub PR review comments mined into pairs). One row =
one JSON object. Fields: `task_id`, `activation_text`, `teacher_text`, `pre_code`,
`post_code`, `quality_score`, `metadata`.

**`metadata` (real, row 0):**
```json
{"source": "external_codereview", "source_type": "external_single_turn",
 "source_task_id": "codereview_4ian/GDevelop_6970", "step_index": 0, "quality_score": 0.4}
```

**`activation_text` = the CONTEXT fed to the hypernetwork (real, row 0, code body abbreviated):**
```
## Task
Review and revise code from 4ian/GDevelop (PR #6970, file: Extensions/AnchorBehavior/anchorruntimebehavior.ts)

## Current Code
            this.owner.setHeight(bottom - top);
          }
          ... [~1830 chars of TypeScript] ...
            const width = right - left;
            this.owner.setX(
              left +
                ((this.owner.getX() - this.owner.getDrawableX()) * width) /
                  this.owner.getWidth()
            );
            ...

## Review Feedback
Should we had `if (this.owner.getX() === this.owner.getDrawableX())` to avoid extra computations 90% of the time at the cost of the `if`?
```

**`teacher_text` = `activation_text` + the answer.** The answer is the part after the
context:

**ANSWER (= `teacher_text` minus the `activation_text` prefix; real, row 0, abbreviated):**
```
## Revision
            this.owner.setHeight(bottom - top);
          }
          ... [same ~1830 chars of code] ...
            const width = right - left;
            this.owner.setX(
              this.owner.getX() === this.owner.getDrawableX()
                ? left
                : left +
                    ((this.owner.getX() - this.owner.getDrawableX()) * width) /
                      this.owner.getWidth()
            );
            ...
```

The ONLY substantive change between Current Code and the Revision is the ternary guard the
reviewer suggested. `answer == post_code` exactly (verified 400/400 rows).

**Measured over 400 rows (Qwen tokenizer):**

| quantity | median |
|---|---|
| answer tokens | 401 |
| context tokens | 472 |
| **review-feedback tokens** | **23 (4.7% of the context)** |
| **fraction of the answer that is verbatim copy of `pre_code`** | **0.89** |
| **fraction of answer tokens that are the actual edit** | **0.10** |
| quality_score | 0.40 (and p90 also 0.40 — looks capped/uniform) |

So the supervised target is ~89% a re-emission of code already present in the context, the
real change is ~10%, and the only trajectory-specific input (the feedback) is ~5% of the
context.

**`metadata.step_index` is 0 for ALL 6930 rows; `source_type` is `external_single_turn` for
all.** Rows sharing a `source_task_id` (1319 of 2420 tasks have >1, mean 2.86) are *parallel
review comments on the same PR*, not ordered steps of a trajectory. There is no multi-step
trajectory in this data — each row is one isolated (code + comment → edit) turn.

---

## 3. How an adapter is trained (exact mechanism)

Per training row, one optimizer micro-step:

1. **Generate the adapter from the context.**
   `extract_activations_with_model(activation_text)` runs the base model (4-bit, NF4,
   `disable_adapter`) with `output_hidden_states=True` and stacks the hidden states at
   `layer_indices` → features of shape `(1, num_layers, seq_len, hidden)`. The perceiver
   `hypernet.generate_weights(features, attn_mask)` consumes these and emits, per target
   module, LoRA tensors `{"A": [1, num_layers, r, d_in], "B": [1, num_layers, r, d_out]}`.
   Gradients flow back into the hypernetwork (no `no_grad`).

2. **Teacher logits (privileged context).** Forward the base model on the FULL text
   (context + answer) with `disable_adapter` and `no_grad`; keep logits over the answer
   span → `teacher_logits`. Also forward the answer ALONE (no context, no adapter) →
   `base_logits`.

3. **Diff mask.** `diff = argmax(teacher_logits) != argmax(base_logits)` over the answer
   span — the tokens where having the context changes the base model's top-1 prediction.
   Rows with zero diff tokens are skipped (`skip_zero_diff`).

4. **Student logits.** Apply the generated adapter to the base model *functionally* — a
   custom forward that adds `(x @ Aᵀ) @ B * scaling` on top of each target layer's original
   output (works with 4-bit `Linear4bit`; positional layer indexing) — and forward the base
   model on the ANSWER ALONE (context removed). The adapter is the only carrier of the
   context. → `student_logits`.

5. **Loss.** Top-K (K=50) KL divergence between `student_logits` and `teacher_logits`,
   summed over the diff-masked answer tokens. Optimizes the adapter so the student (which
   cannot see the context) matches the teacher (which can) on exactly the tokens the context
   changes. `distill_step_loss`.

6. **(Contrastive variant "B1")** additionally: build a hard negative = the SAME row with the
   review feedback swapped for another row's feedback (code identical), generate its adapter,
   and add a hinge `relu(margin − (logp_matched − logp_swapneg))` on the edit-local gold
   tokens. Gradient flows through both adapters (the fix in `9d03b454`), memory-bounded
   (detached pass fixes the active set, then matched and negative pieces back-prop
   sequentially) so sequence length stays 768.

**Config used for the runs:** base `Qwen/Qwen3.5-9B` 4-bit NF4; `lr=1.5e-4`; `max_seq_length=768`;
`grad_accum=8`; `scaler_b_init=0.1` (re-init off the collapsed warm-start); `train_scaling=0.5`;
`topk=50`; `weight_decay=0.01`; 8-bit Adam; gradient checkpointing OFF (incompatible with the
monkeypatched forward). Warm-start = the prior HPO checkpoint (`checkpoints/hypernet_hpo`).

**Runs produced:**
- **A (generic):** the loop above, no contrastive. Checkpoints at steps 200/400/600 (cut at 600).
- **B1 (contrastive):** A + step 6. Smoke run, 40 steps (steps 20/40 saved).

---

## 4. How we checked whether it worked (each probe: method → numbers → meaning)

All probes load the base once (4-bit) and evaluate on the held-out, near-duplicate-filtered
clean val split (`external_codereview.val.clean.jsonl`). "edit-local" = the difflib diff
tokens between `pre_code` and the answer (shared helper `edit_local_mask`, used by both
training and eval). "matched" = adapter from this row's context; "mismatched" = a different
row's context; "swapneg" = this row with feedback swapped (code identical); "zero" = base,
no adapter. Numbers are mean gold-token logprob unless stated.

### 4.1 Specificity gate — `tools/gate_trajectory.py`
**Method:** for each checkpoint × adapter scaling, mean edit-local gold logprob under
matched/mismatched/swapneg/zero, plus preservation (fraction of non-diff tokens kept).
**Result (A600, the most-trained checkpoint):**

| scaling | matched−mismatched | matched−swapneg | matched−zero | preservation |
|---|---|---|---|---|
| 0.5 | +0.059 | +0.005 | +1.075 | 0.81 |
| 1.0 | +0.134 | +0.010 | +0.167 | 0.45 |

**Meaning:** the adapter strongly boosts the edit over base (matched−zero +1.08). There is a
*small* difference between this row's adapter and a different row's (+0.059), but almost none
between this row's adapter and the feedback-swapped one (+0.005) → the small specificity is
driven by the surrounding **code**, not the **feedback**. Warm-start and A200/A400 are flat
(matched−mismatched ≈ 0.00–0.02); the signal only appears faintly at A600.

### 4.2 Smoke + loss magnitudes — `tools/run_corpus_distill.py --contrastive` (40 steps)
**Method:** run B1 for 40 steps, log KL vs margin per step.
**Result:** KL fell 8.12 → 3.38; **margin stayed pinned at ~1.0 the whole time** (lp_matched −
lp_swapneg ≈ 0 every step). **Meaning:** the contrastive hinge never opened — matched and
feedback-swapped adapters produce equal edit logprobs throughout, even with gradient flowing
through the negative.

### 4.3 Conditioning probe — `tools/diag_conditioning.py`
**Method:** decompose each generated adapter as `W(ctx) = W_mean + residual(ctx)` over a set
of contexts; report the context-dependent residual as a fraction of `||W_mean||`, and the
same for the extracted features (masked mean-pooled). **Result (B1 step40):**

| | feature residual / mean | weight residual / mean |
|---|---|---|
| across different rows | 0.25–0.31 | **0.006–0.011** |
| feedback-swap (only feedback changed) | 0.053 | **0.004** |

**Meaning:** the *features* differ across contexts (25–31%; 5.3% from feedback alone), but the
perceiver collapses that into a ~1% change in the generated weights — a ~25–30× attenuation of
the conditioning signal, and ~13× for feedback specifically. Cross-row weight deltas concentrate
in mid-late layers 21–24; feedback deltas are tiny everywhere. Extraction is fine; the
perceiver→weight mapping is the bottleneck.

### 4.4 Scaling gate (the Sakana up-scaling hypothesis) — `gate_trajectory.py --scalings 0.25 0.5 1.0 2.0`
**Method:** does up-scaling the adapter expose hidden specificity (Sakana up-scaled to recover
recall)? **Result (B1 step40):** matched−mismatched and matched−swapneg stay at noise (±0.003)
at *every* scale; matched−zero is non-monotonic (+0.69 → +0.80 → +0.21 → **−1.57**) and
preservation craters (0.97 → 0.46) by scaling 2.0. **Meaning:** up-scaling does not reveal
specificity; it over-fires the generic component and destabilizes. Disconfirmed as a fix.

### 4.5 Recall probe (does the adapter store the row's code?) — `tools/diag_recall.py`
**Method:** split answer tokens into COPY (verbatim ⊂ pre_code = the row's specific code body)
vs EDIT (the change), and report matched/mismatched/zero. **Result (A600):**

| scaling | slice | matched−mismatched | matched−zero |
|---|---|---|---|
| 0.5 | copy | +0.003 | **−0.314** |
| 0.5 | edit | +0.075 | +1.169 |
| 1.0 | copy | −0.009 | −1.403 |

**Meaning:** on the code body, matched ≈ mismatched (no recall of the specific code) and
matched−zero is **negative** — the adapter makes the row's own code *less* likely than base.
The adapter does not store the episode's code; it boosts edit-region tokens at the code body's
expense (net-negative on full answer).

### 4.6 Q&A / recall-out probe (the episodic-memory test) — `tools/diag_qa_recall.py`
**Method:** the genuine episodic test — recover an episode fact NOT in the training output
(the review feedback; the file path) from the adapter alone, under a neutral lead-in prompt
(`"## Review Feedback\n"` / `"file: "`), matched vs mismatched vs zero; plus free greedy
generation from base+matched-adapter.
**Result (A600, scaling 0.5):**

| fact | matched | mismatch | zero | m−mismatch | m−zero |
|---|---|---|---|---|---|
| review feedback | −3.913 | −3.913 | −4.081 | **+0.0005** | +0.168 |
| file path | −6.140 | −6.151 | −6.625 | **+0.011** | +0.484 |

Free generation from base+matched-adapter on `"## Review Feedback\n"`, three different
episodes (a GDevelop TS change; an identifier-naming comment; a side-effects question) — all
produced near-identical generic boilerplate:
```
"I have reviewed the PR and it looks good. I have a few comments: 1. I think the `@Test`
 annotation should be removed..."
"I have reviewed the PR and it looks good. ... This PR adds a new `--no-interactive` flag to the CLI."
"I have reviewed the PR and it looks good. I have a few comments: 1. ... `--no-interactive` flag..."
```
**Meaning:** the episode is **not recoverable** from the adapter. matched ≈ mismatch on both
facts (+0.0005, +0.011) — the adapter does not make THIS episode's feedback or file more
likely than a different episode's adapter. The small m−zero (+0.17, +0.48) is the adapter
putting the model into generic "code-review mode," not episode recall — confirmed by the
generations, which hallucinate an unrelated `--no-interactive` CLI flag regardless of which
episode was embedded.

---

### 4.7 Recoverability harness — goal / diff / tail / avoid — `tools/diag_recoverability.py`
**Method:** the four things an episodic-memory adapter must make recoverable, each scored as
mean gold logprob over the target span under matched / zero / mismatch:
- **goal** = the review request (`## Review Feedback\n` → feedback span)
- **diff** = the edit (edit-local tokens of the revision)
- **tail** = last 5 lines of the current code (the recent state that DRIVES THE NEXT STEP;
  scored as a continuation after the earlier code)
- **avoid** = `logp(accepted post-form) − logp(rejected pre-form)` at the first changed hunk
  (does the adapter prefer the accepted fix over the reviewer-rejected approach — "don't
  repeat the mistake")

**Result (A600, scaling 0.5):**

| target | matched | mismatch | zero | **m−mismatch** | m−zero |
|---|---|---|---|---|---|
| goal | −3.913 | −3.913 | −4.081 | **+0.0005** | +0.168 |
| diff | −2.238 | −2.313 | −3.406 | **+0.075** | +1.169 |
| tail | −1.442 | −1.449 | −1.058 | **+0.006** | **−0.384** |
| avoid (n=14) | −0.319 | −0.344 | −0.503 | **+0.026** | +0.185 |

**Meaning:** the bet needs m−mismatch > 0 (episode-specific) AND m−zero > 0 (beats no
context). None of the four clear the episode-specific bar: goal/tail/avoid are noise
(+0.0005 / +0.006 / +0.026), and diff's +0.075 is the code-driven signal from §4.1 (not
feedback). `tail` is the sharpest failure for an agent loop — m−zero is **negative**
(−0.384): the adapter makes the recent code state *less* recoverable than base, so it cannot
"drive the next step." `avoid` only had a clean rejected-vs-accepted hunk in 14/24 rows, and
even there the episode-specific margin is +0.026 — and note this corpus has only ONE rejected
form per row (the pre-edit code); it has no record of multiple tried-and-failed approaches,
so it cannot teach or test real "don't repeat mistakes" memory.

## 5. What we found

Three separate claims, kept distinct:
1. **No document-style recall.** The adapter does not let the base recover the episode's code
   (§4.5: copy matched−zero = −0.31). Sakana's premise does not transfer.
2. **It is a generic edit-booster.** Boosts edit-region tokens (+1.17 over base) while hurting
   the code body (−0.31); net-negative on full answer. Whether this helps real pass@1 is
   **unmeasured** (no generation/pass@1 was run).
3. **Faint context-conditioned edit signal, code-driven not feedback-driven.** edit
   matched−mismatched +0.075→+0.161, but matched−swapneg only +0.005. The adapter weakly knows
   *which kind* of edit from the surrounding code and essentially nothing from the review request.

**Recoverability scorecard (§4.7), the explicit episodic-memory spec — none cleared:**

| recoverable target | episode-specific (m−mismatch) | beats base (m−zero) | pass? |
|---|---|---|---|
| goal (the request) | +0.0005 | +0.17 | ✗ (not specific) |
| diff (the edit) | +0.075 | +1.17 | ~ (weak, code-driven) |
| tail (drives next step) | +0.006 | **−0.38** | ✗ (hurts continuation) |
| avoid (don't repeat fail) | +0.026 | +0.19 | ✗ (weak; no failure data) |

Two root causes:
- **Data dilution** (§2): target is 89% copy / 10% edit; feedback is 5% of input. Training-length
  independent; the structural reason the signal is tiny.
- **Conditioning attenuation** (§4.3): perceiver→weight collapses a 25–31% feature difference to
  ~1% weight difference (~13× for feedback). Measured, independent of the loss.

---

## 6. What this data is NOT

It is not agent trajectories. Every row is a single-turn GitHub review→revision pair
(`step_index=0`, `external_single_turn`). The engine's trajectory (decompose→plan→code→
[diagnose→repair]*→integrate) does not appear in training. So even a perfect hypernetwork could
not learn trajectory episodic memory from this corpus — there are no trajectories in it. This is
a data-sourcing gap, separate from §5's two causes.

---

## 7. Where we are / next steps

- **Acceptance test going forward = the §4.7 recoverability scorecard** (goal / diff / tail /
  avoid, each m−mismatch>0 AND m−zero>0). Any reformulated data/objective must move these,
  not just lower training loss. Add explicit queryable-memory supervision (train on "what was
  the request / file / change / what changed pre→post / what was rejected", with hard-negative
  controls) and track generation *diversity*/episode lexical overlap (the current adapter
  mode-collapses to identical review boilerplate across episodes — §4.6).
- The contrastive machinery is correct and verified (gradient through the negative; the term
  engages on every row). The bottleneck is not the loss.
- **Highest-leverage next experiment:** change the distillation target from full-file
  reproduction to a **compact patch / edit-program** (`pre_code → post_code` diff) conditioned on
  the feedback, then **re-run §4.3** to check whether the feedback now moves the generated
  weights. This attacks data dilution and tests whether it relieves the conditioning attenuation.
- **If feedback→weight movement still doesn't rise:** the bottleneck is architectural — perceiver/
  head capacity, factoring a shared generic adapter from a context residual, or layer placement.
- **For true trajectory memory:** source or synthesize actual multi-step engine trajectories.
  Patch supervision fixes the feedback→edit objective but does **not** by itself prove
  semi-Markov trajectory memory. A dataset for that needs: ordered transitions (real
  `step_index`), queries over *prior* steps (not just the current edit), and hard negatives
  that preserve the local code but alter an *earlier* trajectory fact — otherwise the system
  can ace patch conditioning while still failing the broader memory bet. This corpus has none
  of these (all `step_index=0`, single-turn).
- **Unmeasured:** product utility (pass@1 / edit-completion). The only proxy (logprob) is
  net-negative on full answer.

---

## 8. Reproduce

```bash
# data stats (§2)
uv run python - <<'PY'  # see scratchpad 06:05 block for the exact script
PY
# train A (generic): tools/run_corpus_distill.py --corpus .../train.jsonl --val-corpus .../val.clean.jsonl \
#   --grad-accum 8 --val-steps 150 --max-seq-length 768 --epochs 3 --lr 1.5e-4 --exp issue49-d2l-final --out /tmp/rune-ck-final
# train B1 (contrastive): add --contrastive --contrastive-weight 1.0 --contrastive-margin 1.0 --max-steps 40 --save-steps 20
# probes (run each under tools/run_guarded.sh):
#   tools/gate_trajectory.py     --ckpts <ckpt...> --scalings 0.25 0.5 1.0 2.0 --n 20
#   tools/diag_conditioning.py   --ckpt <ckpt> --n-ctx 5
#   tools/diag_recall.py         --ckpt <ckpt> --scalings 0.5 1.0
#   tools/diag_qa_recall.py      --ckpt <ckpt> --scaling 0.5
```

**Artifacts:** checkpoints in S3 (`checkpoints/hypernet_hpo` warm-start; `checkpoints/issue49-final`
A 200/400/600); gate JSON + probe logs in `/tmp/rune-ck-trajectory-safe/`; full chronological log
in `instructions/scratchpad.md`; findings summary in `docs/issue49-findings-2026-06-01.md`.
