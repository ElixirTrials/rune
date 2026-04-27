# RCA: Issue #2 — CUDA OOM During Heldout Eval (Trial 1)

**Symptom**: `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 848.00 MiB. GPU 0 has a total capacity of 22.03 GiB of which 545.12 MiB is free. ... 21.49 GiB memory in use.` at `run_training_hpo.py:545` inside `_forward_hunk_metrics`, during eval for Trial 1.

---

## Root Causes (Ranked by Likelihood)

### Cause 1 — Double PEFT Wrap Within the Same Trial (PRIMARY)

**Likelihood: High. Directly produces the ~2× adapter VRAM peak that closes the gap to OOM.**

Each trial follows this sequence:

1. `train_and_register` → `_setup_lora_adapter` → `PeftModel.from_pretrained(cached_base, warm_start)` — wraps the cached base with LoRA for training. (`trainer.py:391`)
2. `_release_trial_state(persist_base=True)` calls `peft_wrapper.unload()` to strip LoRA layers and free adapter VRAM. But `unload()` **returns** the restored base model and that return value is discarded: `trainer.py:90`. PEFT's `unload()` strips module-level `LoraLayer` wrappers but leaves `.peft_config = {"default": <LoraConfig>}` stamped on the base object. The base is not truly clean.
3. `_evaluate_adapter_on_heldout` immediately calls `PeftModel.from_pretrained(base_model, adapter_path)` at `run_training_hpo.py:512` — a **second** PEFT wrap in the same trial, on the same base object that already carries a `peft_config` residue from step 2. `BaseTuner.inject_adapter` re-allocates fresh `lora_A`/`lora_B` parameter tensors (the Issue #3 "Already found a peft_config" warning fires here).
4. At the moment of the OOM allocation in step 3, the base model has: the first wrap's cleaned-up module tree + the `peft_config` residue from training + the second wrap's freshly injected LoRA tensors. The peak VRAM at this point is base + eval-adapter + residual bookkeeping from the training-adapter path.

**File:line evidence**: `trainer.py:90` (return value of `unload()` discarded), `trainer.py:971` (`_release_trial_state` call immediately before `_evaluate_adapter_on_heldout`), `run_training_hpo.py:512` (second `PeftModel.from_pretrained`), `run_training_hpo.py:758` (eval called with no intervening full GPU flush).

---

### Cause 2 — No Sequence-Length Cap in Eval Tokenizer Call (HIGH CONTRIBUTION)

**Likelihood: High. Produces the actual spike that exceeds the remaining headroom.**

`_forward_hunk_metrics` tokenizes each `teacher_text` with no truncation:

```python
enc = tokenizer(teach, return_offsets_mapping=True, return_tensors="pt")   # run_training_hpo.py:540
```

No `max_length=`, no `truncation=True`. Mined GitHub pairs routinely contain full file contents. At Qwen3.5-9B with eager attention (not flash-attn), the attention matrix for a single sequence of length L requires `L² × 4 bytes` (float32 softmax in `eager_attention_forward`). The OOM traceback confirms the crash fires at exactly the softmax: `modeling_qwen3_5.py:611 → attn_weights = nn.functional.softmax(...)`. At L=2048 that is 16 MB; at L=4096 it is 64 MB; a single long pair at L=8192 is 256 MB. The 848 MiB allocation request is consistent with an attention matrix for a multi-thousand-token sequence. With only 545 MiB free this is the proximate kill shot.

**File:line evidence**: `run_training_hpo.py:540` (no truncation args), `training_issues.md:83` (OOM traceback at `eager_attention_forward`).

---

### Cause 3 — Optimizer State and Gradient Buffers Still Resident at Eval Entry (MODERATE)

**Likelihood: Moderate. Explains why 21.49 GB is occupied before eval starts.**

`_release_trial_state` deletes `trainer` and `dataset` and calls `gc.collect()` + `torch.cuda.empty_cache()` (`trainer.py:95–105`). However:

- `del trainer` removes the Python reference but Python GC with cyclic references (the trainer holds a ref to model which holds a ref back via parameter groups) is not guaranteed to run synchronously before `torch.cuda.empty_cache()`.
- `paged_adamw_8bit` (`SFTConfig` at `trainer.py:608`) spills optimizer state to host RAM under pressure, but the optimizer's CUDA-resident moment buffers (the 8-bit quantization tables, not the moments themselves) stay on GPU until the optimizer object is GC'd.
- With `RUNE_PERSIST_BASE_MODEL=1`, `del model` is skipped in `_release_trial_state` (`trainer.py:96–97`) — only `del trainer, dataset` runs. The base model (Qwen3.5-9B NF4 at ~5–6 GB) is intentionally kept alive, but any VRAM the training loop left as gradient buffers on LoRA parameters is only freed when the `trainer` reference drops and GC runs, which may not happen before the eval `PeftModel.from_pretrained` allocates.

**File:line evidence**: `trainer.py:86–105` (`_release_trial_state` body), `trainer.py:608` (`paged_adamw_8bit`), `trainer.py:604` (gradient checkpointing enabled — activation buffers freed per step, not per trial).

---

### Cause 4 — `expandable_segments:True` Not Set at Eval Time (LOW / CONTRIBUTING)

**Likelihood: Low as root cause; real as contributing factor.**

`train_qlora` sets `os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")` at `trainer.py:817`. This runs before the first `torch` import inside training. However `_evaluate_adapter_on_heldout` is called **after** `train_and_register` returns, by which point PyTorch is already imported and the allocator is initialized. `setdefault` only writes if the key is absent, so the flag is present in the environment for the eval call — but the allocator reads `PYTORCH_CUDA_ALLOC_CONF` once at import time, not per-call. If `torch` was imported before `train_qlora` was ever called (e.g. by another import in the HPO process), the flag never took effect. `run_hpo.sh` does not set `PYTORCH_CUDA_ALLOC_CONF` before launching Python (`run_hpo.sh:118`), so the flag depends entirely on `train_qlora` being called before any other `torch` import path — fragile.

**File:line evidence**: `trainer.py:817`, `run_hpo.sh:118` (no pre-set of `PYTORCH_CUDA_ALLOC_CONF`).

---

## Memory Math Estimate

| Component | VRAM |
|-----------|------|
| Qwen3.5-9B NF4 base (double-quant) | ~5.5 GB |
| BF16 activations + KV cache during training (grad checkpointing active) | ~4–6 GB |
| LoRA params + gradients (rank 64, 248 layers, BF16) | ~0.5 GB |
| `paged_adamw_8bit` CUDA-resident buffers | ~0.3–0.5 GB |
| Second PEFT wrap (`PeftModel.from_pretrained` in eval) — fresh lora_A/B tensors | ~0.5 GB |
| Attention matrix spike at eval (single long sequence, eager attn, float32 softmax) | **0.85 GB** (the 848 MiB allocation) |
| **Total at OOM moment** | **~21.5 GB** |

The trace confirms 21.49 GB in use and 545 MiB free — matching the model where training residuals + double-wrap adapter tensors fill to capacity and a single long-sequence attention matrix is the straw that breaks it.

---

## Connection to Issue #3

Issue #3 (the "Already found a peft_config" warning) is the leak signature that directly enables Cause 1. The `peft_config` residue left by the discarded `unload()` return value (`trainer.py:90`) causes `PeftModel.from_pretrained` at `run_training_hpo.py:512` to call `inject_adapter` on a base that still looks PEFT-wrapped. Without the Issue #3 residue, Cause 1 would not produce a second adapter allocation — the eval wrap would operate on a truly clean base.

---

## Connection to Issue #5 (Loss Not Decreasing)

Issue #3 RCA (section 6) identifies that with two `peft_config` entries sharing the key `"default"`, each trial after the first starts from the **trained weights of trial N-1** rather than fresh DeltaCoder initialization. The optimizer is new each trial, but the LoRA weight tensors carry forward — making trials continuations of each other rather than independent HPO samples. This confounds the search: the fitness landscape is not IID across trials, so Optuna's acquisition function operates on a corrupted signal. See `issue-3-peft-multi-adapter.md` section 6 for full analysis.

---

## Fix Directions (No Implementation)

1. **Wrap eval forward in `torch.inference_mode()`**: `_forward_hunk_metrics` already uses `torch.no_grad()` at `run_training_hpo.py:524`, but `inference_mode` also disables version tracking and is lower overhead. More importantly, explicitly verify no gradient computation escapes via the `disable_adapter()` context path (`run_training_hpo.py:523`).

2. **Free training model before eval**: In `_release_trial_state`, after `peft_wrapper.unload()`, call `delattr(base_model, 'peft_config')` (if present), then run `gc.collect()` + `torch.cuda.empty_cache()` **before returning to the HPO caller**. Add a second `gc.collect()` + `empty_cache()` in `_run_single_trial` between `train_and_register` and `_evaluate_adapter_on_heldout` (`run_training_hpo.py:751–758`).

3. **Cap eval sequence length**: In `_evaluate_adapter_on_heldout`, pass `truncation=True, max_length=2048` to the tokenizer call at `run_training_hpo.py:540`. Hunk character ranges must be clipped to the truncated boundary or pairs that exceed `max_length` should be skipped.

4. **Set `expandable_segments` unconditionally in `run_hpo.sh`**: Add `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to `run_hpo.sh` before the `uv run` invocation at line 118 so the flag is in place before `torch` is first imported.

5. **Capture `unload()` return value and clear `peft_config`**: In both `_release_trial_state` (trainer.py:87–93) and `_evaluate_adapter_on_heldout` (run_training_hpo.py:594–598), assign the return value of `unload()` and call `delattr(returned_base, 'peft_config')` to prevent the double-wrap residue that drives Cause 1 and Issue #3.
