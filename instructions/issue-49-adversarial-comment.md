# Issue #49 Comment Draft

## Adversarial review of the diagnosis and proposed cures

I agree with the main operational conclusion: do not pick a "better" checkpoint from the existing campaign. The MLflow history supports that. Experiments `20`, `23`-`26` show low loss / decent `top1_agreement` without evidence that the generated adapter retrieves trajectory content. Experiment `39` is especially damning: run `50272017` passed `gate/adapter_has_any_effect=1`, but the artifact shows generic output/style perturbation at higher scaling, not content retrieval; run `7676e1f6` had no effect at all. Experiments `40` and `42` should also be treated as post-collapse tuning artifacts: `adapter_diff`, `influence`, completion, and coherence metrics are not evidence of trajectory conditioning.

That said, I would soften "root cause confirmed" to "collapse confirmed; root cause strongly suspected." The `scaler_B ~= 0` symptom is decisive, but the causal story still needs instrumentation. In `ctx_to_lora`, `B = B_raw * scaler_B`; `scaler_B=0` blocks gradients into `B_raw`, but gradients into `scaler_B` can still exist if `B_raw` and the loss gradient are non-zero. So the failure may be weak teacher/base disagreement, but it could also be optimizer parameter omission, bf16/scale underflow, skipped/empty records, `strict=False` checkpoint loading hiding missing keys, or a harness issue. The next run should log `requires_grad`, optimizer membership, and gradient norms for `scaler_B`, `head`, and aggregator separately.

A bigger codebase caveat: the wired repo training path is not currently the two-stage hypernetwork path. `src/rune/training/orchestrator.py` still has the oracle stage as a stub, and `src/rune/training/d2l_train.py::run_distillation` projects records to `prompt`/`completion` and trains a PEFT SFT model, not the hypernet. Also, `to_sft_columns()` drops `trajectory`, and the session schema does not persist `pre_codes`/`post_codes`. So "fix the diff collator" is insufficient unless the actual hypernet KD path and corpus schema are restored first.

I would also be careful with the proposed cures:

- Initializing or removing `scaler_B` is plausible, and LoRA initialization work supports being suspicious of zero gates, but a non-zero adapter can still be non-conditioned noise. Magnitude gates like `scaler_B absmax > 0.05` should be tripwires only, never acceptance criteria.
- TIP / EDGE-OPD / KFD support token-selective distillation as a general idea, but they are not direct proof for this static hypernetwork setting. Treat them as ablation candidates, not prescriptions. The strongest local criterion is simpler: supervise and evaluate only where the trajectory-conditioned teacher differs from the base.
- `top1_agreement` should not be removed, but it must be paired with `base_teacher_top1`, `diff_token_frac`, and `diff_agreement = student==teacher on tokens where base!=teacher`.
- The `combine_lora + get_head_bias()` fix needs a shape test before being listed as mechanical. `combine_lora()` concatenates bias as extra rank; the PEFT adapter rank used by `ModelWrapper` must match whatever rank is hot-swapped, or this will fail or silently misapply weights.
- Activation extraction should be fixed before trusting any multi-step adapter experiment: `generate_adapter_weights()` extracts activations through the PEFT-wrapped model, so after a hot-swap the next adapter can be conditioned through the previous adapter unless extraction runs under disabled adapters.

Recommended next steps before full retraining:

1. Add a tiny synthetic overfit gate: train on a handful of unguessable facts, then require `real > zero` and `real > contradictory` on held-out recall prompts. This should pass before any MBPP/HumanEval run.
2. Log collapse diagnostics every N steps: `scaler_B` stats, generated `Delta W` norm, real-vs-zero logit KL, real-vs-contradictory adapter cosine, per-component gradient norms, skipped records, and side-channel coverage.
3. Rebuild the corpus schema so each record has explicit conditioning text, target output span, and pre/post diff metadata. Verify no held-out test leakage into repair or training prompts.
4. Restore the actual two-stage training path before tuning it: oracle QLoRA, hypernet KD against the oracle, and a real success gate. The current wired `run_distillation()` path is plain PEFT SFT and does not train the hypernet.
5. Only after the synthetic retrieval gate passes, run benchmark gates against base-only, zero-adapter, shuffled-trajectory, and contradictory-trajectory controls.
6. Do not relaunch HPO until the objective includes content retrieval or pass@1 lift over base. The previous HPO optimized sensitivity/noise proxies from a collapsed adapter.

Research-wise, Doc-to-LoRA's NIAH setup is the right validation shape: the adapter must recall hidden facts without seeing them in the prompt. LoRA-init / ALLoRA papers justify testing non-zero or reparameterized gates. Selective KD papers justify masking toward informative tokens. None of those replace the local controls above.
