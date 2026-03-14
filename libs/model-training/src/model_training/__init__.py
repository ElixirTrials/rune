"""Training pipelines for LoRA fine-tuning and trajectory management."""

from model_training.trajectory import format_for_sft, load_trajectory, record_trajectory

__all__ = ["format_for_sft", "load_trajectory", "record_trajectory"]
# GPU-dependent modules (peft_utils, config, trainer, hypernetwork, merging) must be
# imported from their submodules directly to avoid top-level GPU import:
#   from model_training.peft_utils import build_qlora_config
#   from model_training.config import get_training_config
#   from model_training.trainer import train_qlora
#   from model_training.hypernetwork import DocToLoraHypernetwork  # deferred GPU import
#   from model_training.merging import ties_merge, dare_merge  # deferred GPU import
