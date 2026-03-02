# Model Training

## Purpose
This component handles fine-tuning (LoRA), distillation, or training of custom models.

## Structure
- `src/model_training/datasets/`: PyTorch/HuggingFace dataset definitions.
- `src/model_training/trainers/`: Training loops or Trainer configurations.
- `notebooks/`: Exploratory training notebooks.

## Best Practices
- **Experiment Tracking**: Use MLflow (configured in root) to track loss and metrics.
- **Artifacts**: Save checkpoints to a cloud bucket or local `models/` directory (gitignored).
- **Resources**: If training requires GPU, ensure your environment (Docker/local) passes through GPU resources.
