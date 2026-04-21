# Training Infrastructure Upgrade

Comprehensive overhaul of Rune's QLoRA fine-tuning pipeline to support
optimal DeltaCoder warm-start training on mined GitHub trajectories,
with MLflow tracking, a one-command CLI wrapper, diff-aware loss
weighting, and a training-hyperparameter HPO study.

## Motivation

The existing `trainer.py` fine-tuning path had three gaps relative to the
goal (fine-tune the pre-warmed DeltaCoder adapter on mined trajectories
optimally, so the resulting LoRA can train a Doc-to-LoRA hypernetwork):

1. **No MLflow** — experiments were untracked. Only `d2l_train.py`
   (the KL-divergence distillation path) had MLflow wiring.
2. **Only trajectory JSONs were accepted** — mined pairs from
   `scripts/mine_github.py --batch` use an activation/teacher split
   that `format_for_sft` can't consume.
3. **HPO optimized inference, not training** — the existing Optuna study
   in `scripts/optimization/run_optimization.py` tunes scaling, prompt
   style, and sampling temperature, not learning rate / rank / alpha.

The upgrade closes all three gaps and adds a diff-aware loss variant so
HPO can A/B the collator against vanilla SFT.

## Usage

### One-command training

```bash
# Mined-pair training (DeltaCoder warm-start on Qwen3.5-9B by default)
bash scripts/train.sh \
    --dataset data/pairs/owner_repo.jsonl \
    --adapter-id my-adapter \
    --epochs 3 \
    --lr 2e-4

# Dry-run — resolves args to JSON without touching the GPU
bash scripts/train.sh --dataset data/pairs/owner_repo.jsonl \
    --adapter-id smoke --dry-run

# Enable the diff-aware loss collator
bash scripts/train.sh --dataset data/pairs/owner_repo.jsonl \
    --adapter-id diff-test --diff-aware-loss

# Skip warm-start (fresh LoRA init from probe cache + defaults)
bash scripts/train.sh --dataset data/pairs/owner_repo.jsonl \
    --adapter-id fresh --warm-start off

# Self-generated trajectory instead of mined pairs
bash scripts/train.sh --session-id sess-001 --adapter-id from-trajectory
```

### MLflow UI

Local runs write to `./mlruns`:

```bash
mlflow ui --backend-store-uri ./mlruns
```

Set `MLFLOW_TRACKING_URI=http://localhost:5000` (matching
`infra/docker-compose.yml`) to log to the compose-hosted MLflow server.

Set `RUNE_DISABLE_MLFLOW=1` to skip MLflow entirely (CI-friendly).

### Training hyperparameter HPO

```bash
# Print the study plan without running trials
uv run python scripts/optimization/run_training_hpo.py \
    --dataset data/pairs/owner_repo.jsonl --print-only

# Overnight study on one L4 (~8–14 GPU-hours with Hyperband pruning)
uv run python scripts/optimization/run_training_hpo.py \
    --dataset data/pairs/owner_repo.jsonl \
    --study-name rune-training-v1 \
    --n-trials 10 \
    --subsample 500

# Smoke — 2 trials × 4 records for CI-adjacent validation
uv run python scripts/optimization/run_training_hpo.py \
    --dataset data/pairs/owner_repo.jsonl --smoke
```

Study state persists to `./optuna_training.db` (resumable). Per-trial
adapters are written under `./hpo_artifacts/<study>/trial_NNN/`; after
the study completes, a retention pass keeps the top-K by fitness
(default 3) and removes the rest.

### Search space

| Parameter | Space | Rationale |
|---|---|---|
| `lr` | log-uniform 1e-5 … 5e-4 | Centered on repo default 2e-4; Thinking Machines "LoRA Without Regret" finds optimal LoRA LR ≈ 10× FullFT LR and is approximately rank-invariant. |
| `alpha_override` | {16, 32, 64, 128} | Applied post-load via module-tree walk; DeltaCoder's saved alpha is the baseline. |
| `lora_dropout` | {0.0, 0.05, 0.1} | Applied post-load; research calls short-run LoRA dropout an unreliable regularizer, so small grid. |
| `warmup_ratio` | uniform 0 … 0.1 | |
| `grad_accum` | {8, 16, 32} | LoRA penalizes large effective batches more than FullFT. |
| `lr_scheduler` | {constant, cosine} | |
| `diff_aware_loss` | {False, True} | A/B flag so HPO adjudicates whether the custom collator wins. |

**Not searched.** `rank` and `target_modules` are locked by the saved
safetensor shapes — changing them would discard the DeltaCoder warm-start.
Alpha and dropout ARE tunable because they are runtime quantities (PEFT
computes `scaling = alpha / r` at forward time; `lora_dropout` is an
`nn.Module`).

### Fitness

```
fitness = 0.6 * (1 - normalize(eval_loss)) + 0.4 * pass_at_1_humaneval_smoke
```

`normalize(eval_loss)` is min-max across the study's completed trials
(with `0.5` fallback for the first 3 trials). Weights live in a
`FitnessConfig` dataclass so they can be swept later without code
changes. Defense: pure loss overrates trials that overfit a small
subsample; pure pass@1 on a 20-task smoke tier has too much variance
to rank trials reliably.

## Trajectory encoding

Mined-pair records have the structure produced by `normalize_mined_pairs`
in `d2l_data.py`:

- `activation_text`: `"## Task ... ## Current Code ... ## Review Feedback ..."`
- `teacher_text`: `activation_text + "## Revision ..."` (or
  `"## Implementation ..."` for the initial commit pair).

`pairs_to_chat_messages(pairs, mode="multi_turn")` clusters pairs by
`metadata.source_task_id` (falling back to `task_id`) and emits one
multi-turn conversation per task, ordered by `metadata.step_index`:

```
[system, user=task+current+review_1, assistant=revision_1,
         user=review_2,               assistant=revision_2, ...]
```

This preserves the attempt → review → correction procedural structure
that the DeltaCoder warm-start already encodes (its DPO alignment on
self-correction pairs maps directly to this pattern). `single_turn`
mode is a flat fallback — one conversation per pair — used when
clustering information is absent.

## Diff-aware loss

Default SFT with `assistant_only_loss=True` weights every assistant
token equally. For trajectory pairs where the assistant reproduces
substantial context (imports, unchanged lines) before emitting the
revision delta, uniform weighting dilutes the procedural signal we
care about.

`DiffWeightedDataCollator` wraps the default SFT collator and produces
a `loss_weights` tensor per batch:

- Tokens masked by `assistant_only_loss` (labels == -100): weight 0.
- Assistant tokens whose token id appears in the masked context span
  (presumed carried-over context): `unchanged_weight` (default 0.3).
- Assistant tokens whose id is NOT in the masked context span
  (presumed new delta): `changed_weight` (default 1.0).

`DiffAwareSFTTrainer.compute_loss` multiplies the per-token CE loss by
`loss_weights` before averaging. When `changed_weight == unchanged_weight`,
the subclass is loss-equivalent to vanilla SFT (regression guard tested
in `test_diff_loss.py`).

Enable with `--diff-aware-loss` on `train.sh`, or let HPO adjudicate.

## Files changed / added

**Core logic** (modified or new)

- `libs/model-training/src/model_training/trainer.py` — MLflow helpers,
  `pairs_to_chat_messages` branch, `_construct_sft_trainer` dispatcher,
  `_override_lora_alpha` / `_override_lora_dropout` module-tree walks.
- `libs/model-training/src/model_training/d2l_data.py` —
  `pairs_to_chat_messages` and supporting helpers.
- `libs/model-training/src/model_training/diff_loss.py` —
  `compute_diff_loss_weights`, `DiffWeightedDataCollator`,
  `build_diff_aware_sft_trainer`.
- `libs/model-training/src/model_training/trainer_cli.py` — argparse
  entrypoint called by `train.sh`.
- `scripts/train.sh` — shell wrapper with `--dry-run`.
- `scripts/optimization/run_training_hpo.py` — Optuna HPO study.

**Tests** (all new, all CPU-safe)

- `libs/model-training/tests/test_trainer_mlflow.py` — 5 cases.
- `libs/model-training/tests/test_pairs_to_chat.py` — 8 cases.
- `libs/model-training/tests/test_trainer_cli.py` — 7 cases (including
  a `bash scripts/train.sh --dry-run` subprocess smoke).
- `libs/model-training/tests/test_diff_loss.py` — 10 cases (pure-fn
  math + trainer_cli flag pass-through).
- `libs/model-training/tests/test_lora_overrides.py` — 8 cases against
  a fake PEFT model (no torch required).
- `scripts/optimization/tests/test_training_hpo.py` — 10 cases for the
  HPO CLI, fitness blend math, subsample helper, and kwargs mapping.

## Known constraints

- **Flash-attn + packing loss collapse** on qwen3.5-9b is respected:
  `attn_implementation="eager"` is inherited from the registry, and
  `SFTConfig(packing=False)` is the default (we never set `packing=True`).
- **L4 throughput** means full-dataset × 3 epochs ≈ 8–12 h. HPO uses
  subsampled proxy trials to keep overnight budget realistic; the
  winning config is intended to drive a final full-data run separately.
- **Warm-start locks rank and target_modules**: HPO does not search
  either. Alpha and dropout ARE tuned — see helper docstrings.
- **adapter_registry import failure** in
  `test_trainer.py::test_train_and_register_creates_adapter_dir` is a
  pre-existing environment issue (workspace packages not installed
  into the venv by default `uv sync`). Unchanged by this PR.

## Future work

- Thread `warmup_ratio` through `SFTConfig` (currently hard-coded at
  0.03 in `trainer.py`). The HPO samples it and logs it via MLflow but
  the trainer doesn't consume it yet.
- Wire `evaluation.metrics.run_humaneval_subset` through `_pass_at_1_humaneval`
  once the eval loader accepts an adapter directory. Today the fitness
  falls back entirely on the loss term for pass@1 ≡ 0.
- Add exhausted-trajectory training (DPO-style negative pairing).
  Explicitly deferred in this PR.
