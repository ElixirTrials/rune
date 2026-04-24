"""Trajectory encoder pretraining subpackage.

Pretrains a coding-aware sentence encoder (option a: InfoNCE fine-tuning of
all-mpnet-base-v2) on augmented mined coding pairs from data/pairs/.

Strict: task_description field required; pairs without it are dropped.

Public API:
    from model_training.encoder_pretrain.augment import augment_corpus
    from model_training.encoder_pretrain.train_encoder import run_training
    from model_training.encoder_pretrain.eval_encoder import run_retrieval_eval
"""
