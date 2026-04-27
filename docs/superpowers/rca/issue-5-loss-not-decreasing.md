# RCA: Issue #5 — Loss Not Decreasing

**Context:** HPO Trial 1 on Qwen/Qwen3.5-9B, `diff_aware_loss=True`, alpha=16, lr=3.1e-5, dropout=0.05, warmup=0.05, grad_accum=32, scheduler=constant.

---

## Hypothesis 1 (Highest Confidence): Double-Wrapped Adapter Cripples Trial 2+ Gradient Flow

**Evidence: smoking gun in the logs**

The training log shows:

```
[W] Trial 1 failed … OutOfMemoryError (during eval)
UserWarning: Already found a `peft_config` attribute in the model. This will lead to having multiple adapters in the model.
LoRA dropout override: adapter=default new_p=0.000 updated_layers=0
```

The `updated_layers=0` on dropout override is the tell: `_override_lora_dropout` walked the module tree and found zero `lora_dropout` ModuleDicts under the name `"default"`. That can only happen if the adapter name visible at override time (`active_adapter`) doesn't match the actual adapter name, or the adapter isn't attached.

The causal chain:

1. Trial 1 OOMs inside `_forward_hunk_metrics` at `run_training_hpo.py:545` — during the *eval* pass, after training already completed.
2. `_evaluate_adapter_on_heldout` raises `OutOfMemoryError` before reaching the `adapter_model.unload()` cleanup at line 595. That cleanup sits *after* the forward call, not in a `finally` block — so it's skipped entirely.
3. `RUNE_PERSIST_BASE_MODEL=1` is set by the HPO harness (see `run_training_hpo.py:499`). The cached base model now has Trial 1's eval `PeftModel` still wrapped around it.
4. Trial 2's `_get_or_load_base` returns the cache hit — a base that is already a `PeftModel`. `PeftModel.from_pretrained` is called again on it, stacking a second adapter. PEFT emits the warning.
5. With two stacked adapters, the `active_adapter` is the newly loaded one (`"default"`), but the module tree has two sets of LoRA layers. The `_override_lora_dropout` walk at `trainer.py:328` looks for `lora_dropout` ModuleDicts keyed by `"default"` — it finds the inner adapter's keys but the outer wrapper intercepts them differently, yielding `updated_layers=0`.
6. The model being trained in Trial 2 is the outer PeftModel wrapping an inner PeftModel. Gradients may flow through the outer LoRA layers, but the inner ones are entangled. The effective loss signal is split across two adapters, both of which are partially frozen, producing no coherent descent.

**Files/lines to instrument:**
- `run_training_hpo.py:590-598` — move `adapter_model.unload()` into a `try/finally` block so cleanup runs even on OOM.
- `trainer.py:88-93` — `_release_trial_state` already has an `unload()` call, but this is the *training* cleanup; the *eval* cleanup at `run_training_hpo.py:595` is separate and unguarded.
- `trainer.py:391` — add an assertion or guard: if the model returned from `_get_or_load_base` already has a `peft_config` attribute, it was not properly cleaned. Fail fast rather than double-wrapping.

**Fix direction:** Wrap `adapter_model.unload()` in `_evaluate_adapter_on_heldout` in `try/finally`. Add a pre-flight check in `_setup_lora_adapter`: if `hasattr(model, 'peft_config')` when entering a warm-start load, call `model.unload()` first (or raise). Cross-link: Issue #3 (PEFT multi-adapter).

---

## Hypothesis 2 (High Confidence): `diff_aware_loss` Path Skips `_attach_assistant_masks`, Leaving Labels All -100

**Evidence: code path analysis**

In `trainer.py:877-878`:

```python
if not diff_aware_loss:
    dataset = _attach_assistant_masks(dataset, tokenizer)
```

The diff-aware path explicitly skips `_attach_assistant_masks`. This is intentional per the comment at `trainer.py:596-601`: the diff collator is supposed to own masking. But this only works if `DiffWeightedDataCollator` correctly identifies assistant tokens.

The collator delegates masking to `DataCollatorForLanguageModeling(completion_only_loss=True)` (`diff_loss.py:636-641`). That inner collator needs either:
- `assistant_masks` in the batch (produced by `_attach_assistant_masks`), or
- `assistant_only_loss=True` in `SFTConfig` to trigger TRL's own template-based masking.

Both are explicitly disabled. `SFTConfig` has `assistant_only_loss=False` (`trainer.py:601`) and the dataset has no `assistant_masks` column on the diff path. The inner `DataCollatorForLanguageModeling` thus produces `labels` where all tokens are `-100` (masked), because `completion_only_loss=True` without any masking signal produces all-masked output.

Consequence: `_compute_weighted_loss` in `diff_loss.py:476-489` computes:

```python
denom = (shift_weights * label_mask).sum()  # → 0.0 for all-masked batch
weighted_loss = weighted.sum() / denom.clamp(min=1e-8)  # → 0/1e-8 ≈ 0
```

Loss is clamped to effectively zero every step. Gradients are ~zero. No learning. The `logger.debug` at line 484 fires silently (DEBUG level, not WARNING) so operators see nothing unusual.

**Verification:** Log `batch["labels"].eq(-100).all()` inside `DiffAwareSFTTrainer.compute_loss` for a single batch. If True, this hypothesis is confirmed.

**Files/lines to instrument:**
- `diff_loss.py:483-488` — promote the all-masked-batch log to `WARNING` and include a count of how many batches triggered it.
- `trainer.py:596-602` — the `assistant_only_loss=False` comment acknowledges the tension but the resolution (diff collator owns masking) is only valid if the collator actually receives `assistant_masks`. It does not on the diff path.
- `diff_loss.py:636-641` — `completion_only_loss=True` on the inner collator is appropriate only when `assistant_masks` is present in the batch.

**Fix direction:** On the diff path, still run `_attach_assistant_masks` and include `assistant_masks` in the dataset before passing to `DiffWeightedDataCollator`. The outer collator applies hunk weights on top of the label mask; it does not replace it. Alternatively, pass `completion_only_loss=False` to the inner collator and let `DiffWeightedDataCollator` handle full masking via the `label == IGNORE_INDEX` check in `compute_hunk_loss_weights`.

---

## Hypothesis 3 (Medium Confidence): Effective Training Steps Too Few to Show Descent

**Evidence: arithmetic**

Parameters: `grad_accum=32`, `per_device_train_batch_size=1` (hardcoded at `trainer.py:608`), `epochs=1`, `subsample=500`, `heldout_fraction=0.1`.

With `encoding_mode=multi_turn`, 500 pairs clustered by `source_task_id` collapse to far fewer conversations. The `_build_sft_config` dataset_size calculation uses `len(dataset)` which is the number of *conversations* post-clustering, not pairs. If there are 50 unique tasks, that is 50 conversations.

```
steps_per_epoch = ceil(50 / 32) = 2
total_steps = 1 * 2 = 2
warmup_steps = int(2 * 0.051) = 0
```

Two gradient updates total. With `lr_scheduler=constant` and `warmup_steps=0`, all two updates happen at full LR `3.1e-5`. That is too few steps for observable loss movement, especially on a held-out evaluation set. Even at 450 pairs it is 15 total steps.

The user "noticed loss did not go down" — if they are looking at the MLflow training loss curve, 2-15 data points all near the initial loss value will appear flat. This is not a pathology but an observation window problem.

However, this alone would not explain complete loss flatline if the model were actually receiving a non-zero gradient signal. It amplifies whichever gradient-blockage issue is primary.

**Verification:** Check `trainer_state.json` in the trial output dir for actual `num_training_steps` and confirm `log_history` has `loss` entries at each step.

**Files/lines to instrument:**
- `trainer.py:612-615` — log the computed `warmup_steps` and `total_steps` before constructing `SFTConfig`.
- `run_training_hpo.py:687-692` — the log line already shows `train_pairs` count; add the post-clustering `dataset_size` (number of conversations) so the total-steps estimation is visible at trial start.

**Fix direction:** For HPO proxy runs, switch to `encoding_mode=single_turn` or increase subsample to `>1000`. Consider logging `total_steps` as an MLflow param so it surfaces in the UI.

---

## Cross-Issue Links

| Issue | Connection |
|-------|-----------|
| Issue #1 | Vocabulary-not-modified warning means `AutoTokenizer.from_pretrained` is loading the base tokenizer without the chat template special tokens configured. `compute_assistant_masks` relies on `<\|im_start\|>` / `<\|im_end\|>` marker IDs; if these resolve to `unk_token_id`, the function raises `ValueError` — but that would surface loudly. More likely it works but the tokenizer lacks any custom vocab, making Issue #2's diff collator rely solely on the base vocab, which is fine for Qwen3.5 (special tokens are native). Low independent impact on loss flatline. |
| Issue #3 | The double-adapter stacking (Hypothesis 1) is the runtime manifestation of Issue #3. The PEFT `Already found a peft_config` warning is the same symptom. |
| Issue #4 | MLflow run-hierarchy: the HPO script opens an outer `mlflow.start_run` at `run_training_hpo.py:731`, then `train_and_register` calls `setup_mlflow` which may open a *nested* child run. TRL's `MLflowCallback` logs per-step training loss to whichever run is active at training time. If the trainer's inner run is the active one during training, step-level `loss` lands in the child run. The user may be looking at the outer HPO run (which only has `eval/*` metrics), seeing no training loss. This is an observation artifact that would make a real loss decrease invisible. |

---

## Recommended Investigation Order

1. **Confirm Hypothesis 2 first** (cheapest): add one `logger.warning` inside `DiffAwareSFTTrainer.compute_loss` at `diff_loss.py:538` to count all-masked batches. Run a single trial with 10 pairs. If every batch logs the warning, Hypothesis 2 is confirmed and is the primary cause.
2. **Confirm Hypothesis 1**: add a guard in `_setup_lora_adapter` (`trainer.py:388`) to assert the base model has no `peft_config` attribute before warm-starting. Run two consecutive trials with `RUNE_PERSIST_BASE_MODEL=1` and observe whether the warning fires on Trial 2.
3. **Confirm Hypothesis 3**: log `total_steps` and `n_conversations` (post-clustering dataset size) as MLflow params at the start of each trial. Compare against the visible loss curve resolution.
