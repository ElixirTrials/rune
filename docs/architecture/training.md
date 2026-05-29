# Training Pipeline

`run_training_pipeline` (`orchestrator.py`) chains three stages — oracle → adapter fine-tuning → success gate — over a corpus of per-bin JSONL shards. The orchestrated pipeline does **not run end-to-end today**: Stage 1 raises `NotImplementedError` first on both the single-run and HPO paths, so execution aborts before the (implemented) Stage 2 or the (placeholder) Stage 3 is ever reached. Each stage is described below with its current status.

## Corpus (mining → shards)

`mining/miner.py` feeds the pipeline. `mine_corpus` scans session directories (`scan_sessions` finds every `session.jsonl`), and `extract_trajectories` emits one record per unique `(action, target)` pair — keyed this way so each subtask yields its own trajectory rather than collapsing all of a session's steps into one. Records carry `task_id = "<benchmark>/<problem_id>"`, the rendered `trajectory` text, and `metadata` (`phase`, `target`, `benchmark`, `problem_id`). Records are bucketed into `{action}_{benchmark}` JSONL shards under `output_dir`; `mine_corpus` returns per-bin record counts.

## Stage 1 — oracle (per-bin QLoRA): NOT IMPLEMENTED

`_run_oracle_training` raises `NotImplementedError`. It is a stub on both code paths — the single-run pipeline and every HPO trial. No oracle adapters are trained.

`oracle_cache.py` ([API](../api/training/oracle_cache.md)) is the scaffolding for the intended design: an `OracleAdapterCache` (LRU) that resolves a bin key to a registered `oracle_<bin_key>` PEFT adapter and reshapes its flat safetensors state dict into a per-layer functional-LoRA dict, for use as a teacher in a later distillation pass. `_bin_key_for_record` reconstructs the `{phase}_{benchmark}` bin key from a record's `metadata` (pooling all `diagnose` steps into `diagnose_pooled`). None of this is wired into the stub Stage 1 or the Stage 2 trainer — nothing in the active training path imports it.

## Stage 2 — adapter fine-tuning (diff-weighted LoRA SFT)

This stage is implemented — `run_distillation` ([d2l_train.py](../api/training/d2l_train.md)) is a complete standalone entrypoint — but the orchestrator never invokes it, because Stage 1 raises first; it is reachable only by calling `run_distillation` directly. Despite the "distillation"/"hypernetwork" naming carried in the function names and docstrings, the mechanism is plain PEFT-LoRA supervised fine-tuning with a diff-aware per-token loss. There is **no perceiver hypernetwork, no teacher model, and no KL term** anywhere in this path; the perceiver lives in `src/rune/model/` and is untouched here.

`run_distillation` loads the JSONL corpus at `config.corpus_path` into a HF `Dataset`, instantiates the base `AutoModelForCausalLM`, a `LoraConfig` (`r=lora_rank`, `lora_alpha`, `CAUSAL_LM`), and a `trl.SFTConfig` (sequence cap via `max_length`, MLflow reporting), then builds a `DiffAwareSFTTrainer` and runs `trainer.train()`.

### Diff-aware loss ([diff_loss.py](../api/training/diff_loss.md))

`build_diff_aware_sft_trainer` wraps trl's `DataCollatorForLanguageModeling` (completion-only masking, `truncation_mode="keep_end"` so the trailing assistant turn survives truncation) in a `DiffWeightedDataCollator`, then constructs a `DiffAwareSFTTrainer` (subclass of trl `SFTTrainer`). The weighting chain:

1. **Line diff.** `_compute_hunk_ranges(before, after)` runs `difflib.SequenceMatcher` over lines; `insert`/`replace` opcodes contribute half-open character ranges in the *after* text (`equal`/`delete` contribute nothing).
2. **Token alignment.** Per assistant span, the post-revision body is re-tokenized with `return_offsets_mapping=True`; `_apply_span_weights` (the inline hunk-intersection at `diff_loss.py:567-578`) assigns each token `changed_weight=1.0` if its char offset intersects a hunk range (`ts < h_end and te > h_start`), else `unchanged_weight=0.3`. Masked (`IGNORE_INDEX`) and special `(0,0)`-offset tokens get `0.0`. (The standalone `compute_hunk_loss_weights` helper encodes the same rule but is unused — nothing in `src/` or `tests/` calls it.) An optional per-record `quality_score` (validated to `(0, 10]`; out-of-range or NaN values are reset to the `1.0` default, with a logged warning) multiplies the row's weights.
3. **Fallbacks.** Missing `pre_codes`/`post_codes` side-channels, no tokenizer, or a failed span match collapse to identity weights (`1.0` for labeled tokens) so gradient signal is preserved; per-span failure counters surface as `train/diff_*` metrics.

`_compute_weighted_loss` applies the causal shift and returns the weighted mean per-token cross-entropy `L = Σ(CE·w·mask) / Σ(w·mask)`. When all weights equal `1.0` this reduces exactly to standard mean CE (the identity invariant). `compute_loss` also accumulates no-extra-forward-pass diagnostics (changed/context CE split, token accuracy, entropy, effective-token count, all-masked-batch fraction), flushed via `log` at the trainer's logging cadence.

## Stage 3 — success gate: runs on placeholder scores

`_run_success_gate` → `evaluate_gate` ([gate.py](../api/training/gate.md)) compares baseline vs. new benchmark scores over the **union** of their keys. A benchmark passes as an improvement when `delta ≥ 0.02` and counts as a regression when `delta < -0.01` (the band `[-0.01, 0.02)` is neither). Union asymmetry: a baseline benchmark absent from the new scores is a regression (`-base_score`); a new benchmark with no baseline is skipped. The gate passes only when `len(improvements) ≥ 4` **and** `len(regressions) == 0`; exit code is `0` on pass, `1` on fail.

As wired, the gate is **never reached**, because Stage 1 raises first. The code that would invoke it (`run_training_pipeline`) feeds `baseline_scores={}` and `new_scores={}` with a logged warning, pending bench-runner wiring (TODO #46-followup). So even once Stage 1 lands, the gate stays non-meaningful — empty scores make it deterministic — until the bench runner supplies real scores.

## HPO

`hpo=True` routes to `_run_hpo`, an Optuna study (`direction="minimize"`) over `learning_rate`, `warmup_ratio`, `lora_rank`, and `neftune_alpha`, with per-trial `MLflowCallback` logging. Each trial's objective calls Stage 1 first, so it raises `NotImplementedError` before Stage 2 or any gate evaluation runs. (`lora_rank`, `learning_rate`, and `warmup_ratio` are threaded into `run_distillation`'s `SFTConfig`; `neftune_alpha` is read into the config but never passed to `SFTConfig`, so it is currently a dead knob.) Once Stage 1 lands, the gate's empty placeholder scores would still leave every trial undifferentiated until the bench runner supplies real scores.
