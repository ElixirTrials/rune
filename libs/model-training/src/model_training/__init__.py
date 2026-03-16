"""Training pipelines for LoRA fine-tuning and trajectory management."""

from model_training.d2l_train import D2LTrainConfig, train_d2l_qwen3
from model_training.trajectory import format_for_sft, load_trajectory, record_trajectory

__all__ = [
    "D2LTrainConfig",
    "format_for_sft",
    "load_trajectory",
    "record_trajectory",
    "train_d2l_qwen3",
]
# GPU-dependent modules (peft_utils, config, trainer, hypernetwork, merging) must be
# imported from their submodules directly to avoid top-level GPU import:
#   from model_training.peft_utils import build_qlora_config
#   from model_training.config import get_training_config
#   from model_training.trainer import train_qlora
#   from model_training.hypernetwork import DocToLoraHypernetwork  # deferred GPU import
#   from model_training.merging import ties_merge, dare_merge  # deferred GPU import
#
# Distillation config/data modules (transformers, peft, ctx_to_lora deferred):
#   from model_training.d2l_config import (
#       get_d2l_qwen3_config, build_qwen3_hypernet_config,
#   )
#   from model_training.d2l_data import (
#       format_for_distillation, generate_needle_dataset,
#       save_jsonl, load_jsonl, split_by_task_id,
#   )
#
# Architecture probe and activation extraction (torch/transformers deferred):
#   from model_training.d2l_probe import (
#       probe_model, extract_activations_with_model,
#       load_probe_cache, save_probe_cache,
#   )
#
# Functional LoRA injection (torch/ctx_to_lora deferred):
#   from model_training.d2l_lora import apply_functional_lora
