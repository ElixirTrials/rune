"""Unit tests for D2LTrainConfig."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from rune.config import DEFAULT_MODEL_ID
from rune.training.d2l_train import D2LTrainConfig, load_train_config


class TestD2LTrainConfig:
    def test_defaults(self) -> None:
        cfg = D2LTrainConfig()
        assert cfg.model_id == DEFAULT_MODEL_ID
        assert cfg.learning_rate == pytest.approx(2e-5)
        assert cfg.warmup_ratio == pytest.approx(0.1)
        assert cfg.num_epochs == 3
        assert cfg.batch_size == 4
        assert cfg.gradient_accumulation_steps == 4
        assert cfg.lora_rank == 8
        assert cfg.lora_alpha == 16
        assert cfg.neftune_alpha == pytest.approx(0.0)
        assert cfg.max_seq_length == 2048
        assert cfg.checkpoint_dir == "./checkpoints"
        assert cfg.experiment_name == "d2l-qwen3-round1"
        assert cfg.logging_steps == 10
        assert cfg.save_steps == 500
        assert cfg.eval_steps == 500
        assert cfg.fp16 is True
        assert cfg.checkpoint_path == ""
        assert cfg.corpus_path == ""

    def test_override(self) -> None:
        cfg = D2LTrainConfig(learning_rate=1e-4, lora_rank=16)
        assert cfg.learning_rate == pytest.approx(1e-4)
        assert cfg.lora_rank == 16

    def test_model_copy(self) -> None:
        cfg = D2LTrainConfig()
        updated = cfg.model_copy(update={"lora_rank": 32, "neftune_alpha": 5.0})
        assert updated.lora_rank == 32
        assert updated.neftune_alpha == pytest.approx(5.0)
        assert cfg.lora_rank == 8  # original unchanged

    def test_no_gpu_imports_at_module_level(self) -> None:
        # Confirms the module imported without error even when GPU libs are stubbed.
        assert D2LTrainConfig is not None


class TestLoadTrainConfig:
    def test_reads_training_section_and_inherits_model_id(
        self, tmp_path: Path
    ) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            'model_id: "Org/Top-Level"\n'
            "temperature: 0.9\n"  # inference-only key must be ignored here
            "training:\n  learning_rate: 9.0e-4\n  num_epochs: 7\n"
        )
        cfg = load_train_config(cfg_file)
        assert cfg.model_id == "Org/Top-Level"  # inherited from top level
        assert cfg.learning_rate == pytest.approx(9e-4)
        assert cfg.num_epochs == 7

    def test_training_section_can_override_model_id(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            'model_id: "Org/Top-Level"\ntraining:\n  model_id: "Org/Train-Only"\n'
        )
        assert load_train_config(cfg_file).model_id == "Org/Train-Only"

    def test_env_overrides_everything(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text('model_id: "Org/Top-Level"\ntraining: {}\n')
        os.environ["RUNE_BASE_MODEL"] = "Org/Env"
        try:
            assert load_train_config(cfg_file).model_id == "Org/Env"
        finally:
            del os.environ["RUNE_BASE_MODEL"]

    def test_missing_file_yields_defaults(self, tmp_path: Path) -> None:
        cfg = load_train_config(tmp_path / "nope.yaml")
        assert cfg.model_id == DEFAULT_MODEL_ID
        assert cfg.num_epochs == 3
