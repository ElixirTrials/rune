# RCA: Issue #1 — "Could not find a config file in Qwen/Qwen3.5-9B" Warning

## Exact Source

**File:** `.venv/lib/python3.12/site-packages/peft/utils/save_and_load.py:296` (peft 0.18.1)
**Function:** `get_peft_model_state_dict`

```python
# save_and_load.py:289-298
local_config_exists = os.path.exists(os.path.join(model_id, "config.json"))
exists = local_config_exists or check_file_exists_on_hf_hub(model_id, "config.json")
if exists is None:
    warnings.warn(
        f"Could not find a config file in {model_id} - will assume that the vocabulary was not modified."
    )
```

`check_file_exists_on_hf_hub` (same file, imported from `peft/utils/other.py:1384`) returns `None` — not `True` or `False` — when `HF_HUB_OFFLINE=1` is set:

```python
# other.py:1383-1386
if str_to_bool(os.environ.get("HF_HUB_OFFLINE", "0")):
    # user set offline mode, cannot check
    return exists  # returns None
```

## Root Cause

`run_hpo.sh` exports `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` before launching the study. This is intentional (avoids per-trial Hub HEAD requests, added in commit `b9252b5`). However, PEFT's `save_pretrained` path — called via `trainer.save_model(output_dir)` at `trainer.py:960` — runs `get_peft_model_state_dict` to decide whether to save embedding layers. That function calls `check_file_exists_on_hf_hub("Qwen/Qwen3.5-9B", "config.json")`, which immediately returns `None` in offline mode because it has no cached `config.json` to look at locally either (it only checks `os.path.join(model_id, "config.json")`, i.e., a literal directory path `Qwen/Qwen3.5-9B`, not the HF cache). When `exists is None`, the warning fires and PEFT assumes the vocabulary was not modified, setting `save_embedding_layers=False`.

The warning fires once per trial at adapter save time, just before (or after) the "Reusing cached NF4 base model" log line that appears on subsequent trials from `trainer.py:124`.

## Severity: Cosmetic — No Real Bug (for this model)

The assumption PEFT falls back to — vocabulary not modified, skip saving embedding layers — is **correct** for this setup. The LoRA training in `_setup_lora_adapter` targets `q_proj`/`v_proj` only (`trainer.py:408`), never resizes the embedding table, and uses a fresh `AutoTokenizer.from_pretrained` with no `add_special_tokens` calls that would change vocab size. So PEFT's fallback decision matches ground truth. The saved adapter is complete and correct.

There is one secondary concern worth noting: `Qwen/Qwen3.5-9B` on HuggingFace Hub resolves to an **image-text-to-text VLM** (`AutoModelForImageTextToText`, architecture `qwen3_5`), not a pure text causal LM. The local transformers (≥5.5) does include `Qwen3_5ForCausalLM` (see `modeling_qwen3_5.py:1690`), and `AutoModelForCausalLM.from_pretrained` resolves via the `auto_map` in the model's `config.json` — the OOM traceback confirms the model does load and forward-pass through `modeling_qwen3_5.py`. So the model ID resolves correctly for causal LM use; the VLM designation on the Hub card is because the same base architecture underlies both the instruction-tuned VLM and the base text model. This is not a root cause of the warning.

## Connection to Other Issues

- **Issue #2 (OOM):** Unrelated to this warning. The OOM occurs during the held-out eval forward pass at `run_training_hpo.py:545`, where the full attention matrix materializes. The warning fires at save time, after training completes successfully.
- **Issue #3 (multiple peft_config):** Also unrelated — that fires when `PeftModel.from_pretrained` sees a model already wrapped with PEFT (the cache-reuse path). The config-warning fires independently at save time.
- **Issue #5 (loss not going down):** No connection. The warning causes `save_embedding_layers=False`, which is the correct value. If embedding layers were incorrectly excluded when they had been resized, it could corrupt the adapter — but they are not targeted or resized here.

## Recommended Fix Direction

Pass `save_embedding_layers=False` explicitly to `trainer.save_model` (or to the underlying `PeftModel.save_pretrained`) so PEFT skips the `check_file_exists_on_hf_hub` probe entirely. In TRL's `SFTTrainer`, this means overriding `save_model` or passing `save_embedding_layers=False` via `SFTConfig`. Alternatively, populate a local `config.json` stub in the HF cache before setting `HF_HUB_OFFLINE=1` so `os.path.exists(os.path.join(model_id, "config.json"))` returns `True` — but this is brittle. The explicit `save_embedding_layers=False` is the minimal, correct fix.
