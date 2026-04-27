# RCA: Issue #3 — PEFT Multi-Adapter Warning During HPO

**Symptom**: `tuners_utils.py:302 UserWarning: Already found a peft_config attribute in the model. This will lead to having multiple adapters in the model.` fires on every trial ≥ 1.

---

## 1. PEFT Trigger Condition

`BaseTuner.__init__` in `.venv/lib/python3.12/site-packages/peft/tuners/tuners_utils.py:282` guards on `hasattr(self, "peft_config")`. At construction time, `self` is the new `BaseTuner` wrapper, not the raw model. However, at line 301 it immediately stamps:

```python
self.model.peft_config = self.peft_config   # tuners_utils.py:301
```

This writes `peft_config` directly onto the *inner* `AutoModelForCausalLM` instance. On the next construction (`BaseTuner.__init__`) PEFT does `hasattr(self, "peft_config")` — this is the new tuner wrapper, whose `self.model` is the same cached base. Because `self.model.peft_config` now exists from the previous trial, the check at line 282 (`hasattr(self, "peft_config")`) evaluates via attribute inheritance/delegation on `PeftModel` subclass machinery and fires the warning at line 285.

---

## 2. Where the Duplicate Wrapping Happens

The cached-base path starts in `trainer.py:_get_or_load_base()` (lines 108–138), gated on `RUNE_PERSIST_BASE_MODEL=1`. When the cache is hot, the same `AutoModelForCausalLM` object is returned for every trial.

Trial N proceeds to `trainer.py:_setup_lora_adapter()` (lines 366–422). With a warm-start, line 391 calls:

```python
model = PeftModel.from_pretrained(model, str(warm_start))   # trainer.py:391
```

At end-of-trial, `_release_trial_state()` (lines 68–99) calls `peft_wrapper.unload()`. PEFT's `unload()` at `tuners_utils.py:649` calls `_unload_and_optionally_merge(merge=False)` (lines 580–609), which walks the module tree and replaces `LoraLayer` wrappers with their `base_layer` counterparts. **It never executes `delattr(self.model, 'peft_config')` or sets it to `None`**. After `unload()` returns, the `AutoModelForCausalLM` still carries `.peft_config = {"default": <LoraConfig>}`.

Trial N+1 then calls `PeftModel.from_pretrained(same_base_model, ...)` again. `BaseTuner.__init__` constructs, sees `peft_config` on the model at line 282, emits the warning, and appends the new config into the existing dict instead of replacing it. The model now holds two adapter registrations under the same key name `"default"`.

---

## 3. Lifecycle Confirmation

The `unload()` contract is documented as "not an in-place operation — assign the returned model" (`tuners_utils.py:653`). The caller in `_release_trial_state` (trainer.py:87–93) never assigns the return value:

```python
peft_wrapper.unload()   # return value discarded — trainer.py:90
```

So the `unload()` call does strip the LoRA linear layers from the module tree (the VRAM for adapter weights is freed) but the `peft_config` attribute survives on the base model. Trial N+1's `PeftModel.from_pretrained` sees a model that looks half-unwrapped: no LoRA layers, but `.peft_config` still present.

---

## 4. Issue #4: Alpha/Dropout Overrides Fire After the Duplicate Wrap

The log sequence is:

```
tuners_utils.py:302  UserWarning: Already found a peft_config ...   ← wrap N+1
model_training.trainer: LoRA alpha override ...                       ← trainer.py:302
model_training.trainer: LoRA dropout override ...                     ← trainer.py:337
```

Both `_override_lora_alpha` and `_override_lora_dropout` are called inside `_setup_lora_adapter()` (trainer.py:401–404) **after** `PeftModel.from_pretrained` at line 391 returns. The override log lines therefore confirm they operate on the already-doubled model — which explains why `updated_layers=248` is non-zero (both adapter copies are walked and updated) while `updated_layers=0` for dropout suggests the dropout path only matched one of the two stacked adapters.

---

## 5. Causal Chain to Issue #2 (OOM)

Each trial that observes the warning leaves one additional orphaned `LoraConfig` entry in `peft_config` and, critically, PEFT's `inject_adapter` (tuners_utils.py:298) has already re-injected LoRA layers for the new adapter name on top of whatever remnant state existed. After `unload()` stripped layer wrappers but left `peft_config`, `PeftModel.from_pretrained` calls `inject_adapter` again: it re-adds `lora_A`, `lora_B` parameter matrices for the new trial's adapter. These tensors are the dominant VRAM consumers (~200 MB for Qwen3.5-9B at rank 64 across 248 layers). Because `unload()` returned a value that was discarded, the base model is not truly clean: the layer-level wrappers were stripped but `inject_adapter` on the next wrap re-allocates fresh ones. The cumulative effect across trials is not compound (layers are stripped each time), but the transition state where the old adapter's tensors are still reachable via the `peft_config` registry while the new adapter's tensors are being injected creates a peak that is ~2× the per-trial adapter VRAM. On the 22 GB L4 at 21.49 GB already in use (Issue #2 traceback: `training_issues.md:89`), any extra 848 MiB allocation during the eval forward pass hits the ceiling.

The OOM fires during `_evaluate_adapter_on_heldout` at `run_training_hpo.py:512`:

```python
adapter_model = PeftModel.from_pretrained(base_model, adapter_path)   # run_training_hpo.py:512
```

This is a **second** PEFT wrap in the same trial — first the training wrap (via `train_and_register` → `_setup_lora_adapter`), then a second wrap here for eval. Trial 0's training `unload()` is called inside `_release_trial_state` (trainer.py:971) but the `peft_config` residue means the eval wrap at line 512 doubles up again, compounding peak VRAM.

---

## 6. Possible Link to Issue #5 (Loss Not Decreasing)

When two adapters share the name `"default"` in `peft_config`, PEFT's `active_adapter` (tuners_utils.py:295) is set to the *most recently injected* one. However, `_override_lora_alpha` and `_override_lora_dropout` walk all modules and update both adapter slots, so scaling is consistent. The more dangerous case is parameter identity: `SFTTrainer` collects trainable parameters via `model.parameters()` at construction. If the second `PeftModel.from_pretrained` re-uses the same parameter objects from the previous trial's LoRA matrices (because `unload()` only replaced module references, not tensors), the optimizer may receive stale gradient state from trial N-1. The optimizer is freshly constructed each trial so this is unlikely to carry gradient history, but the starting point of `lora_A` and `lora_B` at trial N+1 would be the *trained* weights from trial N rather than the fresh DeltaCoder initialization, making every trial after the first a continuation of the previous trial's warm-start — effectively doing multi-epoch training on cumulative data rather than independent HPO trials.

---

## 7. Issue #5 vs. round2_train.py Contamination

`round2_train.py` uses `apply_functional_lora` from `model_training.d2l_lora` (lines 57–59), a weight-patching context manager that never calls `get_peft_model` or `PeftModel.from_pretrained`. It operates by direct tensor mutation and restoration, not PEFT structural wrapping, so it cannot introduce `peft_config` attributes. There is no import of `model_training.trainer` in `round2_train.py`. The HPO and round-2 paths are cleanly separated at the import level.

---

## 8. Recommended Fix Directions (No Implementation)

1. **Strip `peft_config` after `unload()`**: After `peft_wrapper.unload()` in `_release_trial_state` (trainer.py:90), call `delattr(base_model, 'peft_config')` if the attribute exists. Also do the same after `adapter_model.unload()` in `_evaluate_adapter_on_heldout` (run_training_hpo.py:595). This removes the residue that triggers the warning.

2. **Capture `unload()` return value**: `unload()` is documented as returning the restored base model. The return value should be captured and (when in cached mode) used to update `_BASE_MODEL_CACHE[model_id]`, replacing the stale reference.

3. **Use `delete_adapter` instead of `unload`**: `BaseTuner.delete_adapter` (tuners_utils.py:476–490) removes the adapter entry from `peft_config` and its LoRA layers. Combined with `unload`, this would leave the base truly clean.

4. **Separate base-model instance per trial in eval**: `_evaluate_adapter_on_heldout` should not reuse the same cached base as training within the same trial. A per-eval `from_pretrained` (with cache opt-in only for base weight tensors, not object identity) would eliminate the double-wrap peak.
