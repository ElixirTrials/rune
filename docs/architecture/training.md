# Training Architecture

Rune's active training path is HyperLoRA context distillation. The goal is not
to fine-tune the base model; it is to train the hypernetwork that maps a coding
trajectory to LoRA adapter weights.

## D2L Context Distillation

The current loop lives in `rune.training.hypernet_distill` and follows the
privileged-context D2L setup:

1. **Teacher:** the frozen base model sees the trajectory context plus the
   answer span.
2. **Base control:** the same frozen base model sees the answer span without
   the trajectory context.
3. **Student:** the base model receives a generated adapter from the
   hypernetwork and sees the answer span without trajectory context.
4. **Loss:** top-K KL from teacher to student over answer-span positions where
   teacher top-1 differs from base top-1.

This objective asks the adapter to internalize what the teacher obtained from
context. The diff mask is therefore **teacher-vs-base disagreement**, not a
line-diff hunk mask.

## Real-Corpus Mapping

The real code-review corpus uses:

- `activation_text` as trajectory context.
- `teacher_text` as context plus `## Revision` answer block.
- `pre_code` / `post_code` for edit-local diagnostics and audits.
- `quality_score` / `metadata` for corpus analysis.

The training mapper strips `activation_text` from the front of `teacher_text` to
recover the answer span. Corpus statistics record exact-prefix and fallback
rates so template drift cannot silently change the training target.

Long contexts are handled with answer-preserving truncation: keep the full
answer whenever possible and front-truncate context. If the answer alone exceeds
the sequence cap, the answer head is used and the truncation is observable in
metrics.

## Guardrails

The loop logs metrics designed to catch failures that raw loss would hide:

- `diff_token_frac` — how much teacher-vs-base disagreement exists on the row.
- `diff_agreement` — whether the student moves toward the teacher on diff
  tokens.
- `preservation_agreement` — whether the student preserves teacher/base
  agreement-region behavior.
- `skipped` / skip fraction — whether rows produce usable gradients.
- watched parameter stats and gradient summaries for collapse diagnosis.

Early-stop guardrails abort runs that skip too much data, fail to move on diff
tokens after warmup, or damage preservation beyond the configured floor.

## Memory Mode

Training the ~4B Instruct base plus a 428M-parameter hypernetwork with optimizer
state and student-forward autograd does not fit the same budget as inference. The
default matches the engine: bf16 base (`load_in_4bit=False`) with 8-bit Adam
(`use_8bit_optim=True`, AdamW fallback) for hypernetwork parameters; a 4-bit NF4
base with bfloat16 compute is opt-in for a larger base or tighter memory. Functional LoRA is applied through the base layer's
original forward path plus an explicit LoRA delta, so the same training logic can
run over quantized base layers.

Precision matters for interpretation: training-matched 4-bit evaluation and
engine-target bf16 evaluation must be labeled separately.

## Gates and HPO

Training completion is not success. Promotion requires post-training evaluation:

- edit-relevant held-out gates,
- retrieval / shuffled / contradictory controls,
- dual-precision 4-bit and bf16 checks,
- tiny pass@1 benchmarks against base and controls.

HPO is blocked until a baseline run completes and the edit-relevant held-out
metric is wired. The HPO objective should optimize held-out edit-local behavior
under preservation and degeneration constraints, not raw training loss.
