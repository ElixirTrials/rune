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
| `d2l_data.py` | Dataset preparation and loading |
| `d2l_config.py` | Training configuration |
| `d2l_lora.py` | LoRA adapter utilities for D2L |
| `d2l_prep.py` | Data preprocessing |
| `d2l_mining.py` | Trajectory mining from coding sessions |
| `d2l_probe.py` | Probing trained hypernetwork quality |
| `sakana_d2l.py` | Sakana AI Doc-to-LoRA integration |

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
experiment `rune-qlora`). Tracking URI falls back to `./mlruns`; override
via `MLFLOW_TRACKING_URI` or the `mlflow_tracking_uri` kwarg. Set
`RUNE_DISABLE_MLFLOW=1` to skip MLflow for CPU CI.

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
scheduler, diff-aware-loss flag). Uses Optuna with Hyperband pruning. See
`docs/plans/training_upgrade.md` for search-space rationale and L4 budget.

### Training Data Mining

- `d2l_mining.py` — Trajectory mining from coding sessions for hypernetwork training data
- `scripts/mine_github.py` — Mines GitHub PRs, issues, and commits for hypernetwork training data

### Other Modules

- `peft_utils.py` — PEFT configuration helpers
- `trajectory.py` — Trajectory formatting for training data
- `config.py` — Training configuration models

## GPU Import Pattern

All GPU imports (torch, safetensors, transformers, peft) are deferred inside function/method bodies per INFRA-05 pattern. Every module in this package is importable in CPU-only CI environments.
