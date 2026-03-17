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

### Trainer (`trainer.py`)

QLoRA fine-tuning utilities:
- `train_and_register()` — Fine-tune a LoRA adapter and register it in the adapter registry

### Other Modules

- `peft_utils.py` — PEFT configuration helpers
- `trajectory.py` — Trajectory formatting for training data
- `config.py` — Training configuration models

## GPU Import Pattern

All GPU imports (torch, safetensors, transformers, peft) are deferred inside function/method bodies per INFRA-05 pattern. Every module in this package is importable in CPU-only CI environments.
