# Model Training

Fine-tuning, hypernetwork, and adapter merging for Rune.

## Components

### Hypernetwork (`hypernetwork.py`)

`DocToLoraHypernetwork` — Perceiver-based model that generates rank-8 LoRA adapter weights from token IDs in a single forward pass. Cross-attends over token embeddings with learned latents to produce PEFT-compatible `lora_A` and `lora_B` matrices for all target modules.

Key functions:
- `DocToLoraHypernetwork(input_dim)` — Constructor (default vocab size 32000)
- `save_hypernetwork_adapter(weights, path, base_model_id)` — Save adapter to disk

### D2L Training Pipeline

End-to-end pipeline for training the hypernetwork on coding trajectory → adapter pairs:

| Module | Purpose |
|--------|---------|
| `d2l_train.py` | Main training loop |
| `d2l_data.py` | Dataset preparation and loading; `normalize_mined_pairs` (per-step pair extraction, task_id leakage guard); `pairs_to_chat_messages` SFT converter returning `(conversations, pre_post_records)` tuple; `task_description` propagated through `_make_pair_record` (unblocks `MIN_RETENTION_RATIO=0.80` gate) |
| `d2l_config.py` | Training configuration |
| `d2l_lora.py` | LoRA adapter utilities for D2L |
| `d2l_prep.py` | Data preprocessing |
| `d2l_mining.py` | Trajectory mining from coding sessions |
| `d2l_probe.py` | Probing trained hypernetwork quality |
| `d2l_diff.py` | RTK-style diff compression |
| `sakana_d2l.py` | Sakana AI Doc-to-LoRA integration |
| `diff_loss.py` | `DiffAwareSFTTrainer` + `DiffWeightedDataCollator`; hunk-weighted token loss; identity-under-uniform-weights (regression-guarded); fallback emits identity weights when side-channels or tokenizer are missing |
| `kill_switch.py` | `KillSwitchConfig`, `KillSwitchState`, `maybe_run_kill_switch`, `build_benchmark_evaluate_fn`; wired into `train_d2l_qwen3` via `kill_switch_evaluate_fn` kwarg; default off; triggers on ≥5% Pass@1 regression on HumanEval (20–30 held-out tasks, k=5) |
| `training_common.py` | `mlflow_log_params` shared helper |
| `round2_config.py` | `Round2TrainConfig` (Pydantic, inherits `D2LTrainConfig`) |
| `oracle_cache.py` | `_bin_key_for_record`, `lookup_oracle_path`, `audit_oracle_coverage`, `_load_oracle_as_lora_dict`, `OracleAdapterCache` (LRU, max 4 loaded, stores `LoraDict` tensor dicts) |
| `round2_train.py` | `_apply_functional_lora`, `_teacher_forward_with_oracle`, `_compute_kl_ce_loss`, `_training_step_round2`, `train_d2l_qwen3_round2`, `register_round2_adapter` |
| `round2_gate.py` | `evaluate_round2_gate` — strict success gate |

### Round-2 Distillation

Trains the Sakana HyperLoRA hypernetwork using **per-bin oracle adapters as teacher signals** instead of the bare base model.

#### Oracle structure

- 25 bins: `<phase>_<benchmark>` for 4 phases × 6 benchmarks, plus `diagnose_pooled`.
- Oracle adapter IDs: `oracle_<bin_key>` — set by `libs/corpus-producer/src/corpus_producer/trainer_bridge.py`.
- Required benchmarks: `humaneval`, `mbpp`, `apps`, `bigcodebench`, `ds_1000`, `livecodebench`.

#### Functional-LoRA teacher

Oracle is applied to the base model via `apply_functional_lora` context manager — identical mechanism used for the student pass. Base model is **never structurally mutated** (no `PeftModel` wrappers, no `LoraLayer` replacements), eliminating PEFT hook-leakage risk between passes.

`OracleAdapterCache` stores `LoraDict` tensor dicts (`{module: {"A": Tensor[L,r,in], "B": Tensor[L,r,out]}}`); LRU cap of 4 loaded oracles.

#### Round-2 adapter identity

- ID: `round2_<uuid[:8]>`
- `task_type="round2_hypernet"`, `generation=2`
- `parent_ids=json.dumps(sorted(oracle_ids))` for lineage tracking

#### `Round2TrainConfig` (beyond inherited `D2LTrainConfig`)

| Field | Default | Purpose |
|-------|---------|---------|
| `oracle_registry_url: str` | *(required)* | SQLAlchemy URL for the `AdapterRegistry` holding the 25 oracle records |
| `max_loaded_oracles: int` | `4` | LRU cap for cached oracle LoRA dicts |
| `min_oracle_coverage: float` | `0.8` | Minimum fraction of training records whose bin has a registered oracle; below → abort at startup |
| `oracle_fallback: Literal["skip", "base_model"]` | `"skip"` | `"skip"` drops records with no oracle (preserves oracle-only signal); `"base_model"` is an ablation mode |
| `checkpoint_dir: str` | `"./checkpoints/round2"` | Does not clobber round-1 checkpoints |
| `experiment_name: str` | `"d2l-qwen3-round2"` | MLflow separation from round-1 |

#### Startup guards

- Coverage < `min_oracle_coverage` → `RuntimeError` before any model load. `dry_run` surfaces the same gate.
- `_training_step_round2` returns `(None, {})` when an oracle is missing and `oracle_fallback == "skip"`; `steps_completed` only advances on successful optimizer steps.

#### Strict success gate (`round2_gate.py`)

- ≥ 4/6 benchmarks improved ≥ 2.0% Pass@1 **and** no regression > 1.0% on any benchmark.
- Verdict JSON keys: `passed`, `deltas`, `improved_count`, `max_regression`, `reasons`, `round2_adapter_id`, `scores`.

#### CLIs

```bash
# Train round-2 hypernet with oracle teachers
uv run scripts/train_round2.py \
    --sakana-checkpoint-path /path/to/sakana.bin \
    --oracle-registry-url sqlite:///~/.rune/adapters.db \
    --dataset-path data/phase_corpus/all_bins_concat.jsonl \
    --num-steps 1000

# Apply the strict gate (exit 0 = PASS, exit 1 = FAIL)
uv run scripts/evaluate_round2.py \
    --round2-adapter-id round2_<hex8> \
    --base-model Qwen/Qwen3.5-9B \
    --oracle-registry-url sqlite:///~/.rune/adapters.db \
    --baseline-report round1_scores.json \
    --output-report round2_verdict.json
```

### Merging (`merging.py`)

Adapter combination strategies for evolutionary merging:

- `ties_merge(state_dicts, density)` — TIES-Merging: trim-elect-sign-disjoint merge
- `dare_merge(state_dicts, density)` — DARE-Merging: drop-and-rescale merge

Both accept lists of adapter state dicts and return a single merged state dict.

### Model Registry (`model_configs.py`)

`ModelConfig` registry providing pre-configured model settings with DeltaCoder warm-start support. Includes configurations for models like Qwen3.5-9B with warm-start adapter from `danielcherubini/Qwen3.5-DeltaCoder-9B`.

### Trainer (`trainer.py`)

QLoRA fine-tuning utilities:
- `train_and_register()` — Fine-tune a LoRA adapter and register it in the adapter registry
- `train_qlora()` — Lower-level SFT pipeline; accepts a mined-pair JSONL via
  `dataset_path=` or a recorded trajectory via `session_id=` (mutually exclusive).
  Optional `override_lora_alpha` / `override_lora_dropout` retune a warm-started
  adapter without discarding the saved safetensor shapes.
- `diff_aware_loss=True` wraps the SFT collator with `DiffWeightedDataCollator`
  and swaps in `DiffAwareSFTTrainer` (see `diff_loss.py`) so per-token loss is
  biased toward the revision delta vs. carried-over context.

MLflow tracking is enabled by default (`report_to="mlflow"`,
experiment `rune-qlora`). Tracking URI falls back to
`sqlite:///./mlflow.db` (the filesystem `./mlruns` backend was deprecated
by MLflow in February 2026); override via `MLFLOW_TRACKING_URI` or the
`mlflow_tracking_uri` kwarg. Set `RUNE_DISABLE_MLFLOW=1` to skip MLflow
for CPU CI.

### CLI wrapper

One-command fine-tuning via `scripts/train.sh`:

```bash
bash scripts/train.sh --dataset data/pairs/repo.jsonl --adapter-id my-adapter
bash scripts/train.sh --session-id sess-001 --adapter-id from-trajectory
bash scripts/train.sh --dataset data/pairs/repo.jsonl --adapter-id smoke --dry-run
```

The shell wrapper forwards to `model_training.trainer_cli.main`; all flags
map 1:1 to `train_and_register` kwargs. `--dry-run` resolves args to JSON
without importing torch — useful for CI validation.

### Training-hyperparameter HPO

`scripts/optimization/run_training_hpo.py` tunes the DeltaCoder warm-start
fine-tune's training hyperparameters (LR, alpha, dropout, warmup, grad-accum,
scheduler, diff-aware-loss flag). Uses Optuna with Hyperband pruning.

Fitness metrics: `hunk_loss`, `hunk_accuracy`, `adapter_improvement`, `hunk_entropy`.
Task-level heldout split with two strategies (`step_index` | `random`), no pair-level leakage.
Heldout evaluator uses 4-bit NF4 `BitsAndBytesConfig` + `device_map="auto"` + `torch_dtype=torch.bfloat16`
+ `attention_mask` threading (required for 9B models; CPU eval is infeasible).

### Training Data Mining

- `d2l_mining.py` — Trajectory mining from coding sessions for hypernetwork training data
- `scripts/mine_github.py` — Mines GitHub PRs, issues, and commits for hypernetwork training data

### Other Modules

- `peft_utils.py` — PEFT configuration helpers
- `trajectory.py` — Trajectory formatting for training data
- `config.py` — Training configuration models

## GPU Import Pattern

All GPU imports (torch, safetensors, transformers, peft) are deferred inside function/method bodies per INFRA-05 pattern. Every module in this package is importable in CPU-only CI environments.
