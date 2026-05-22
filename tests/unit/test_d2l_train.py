"""Unit tests for D2LTrainConfig and run_distillation."""

from __future__ import annotations

import pathlib
import sys
from unittest.mock import MagicMock, patch

import pytest

from rune.training.d2l_train import D2LTrainConfig, run_distillation

# Stub out optional heavy dependencies so tests run without GPU/datasets.
_STUB_MODULES = ["datasets", "trl"]
for _mod in _STUB_MODULES:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()  # type: ignore[assignment]


class TestD2LTrainConfig:
    def test_defaults(self) -> None:
        cfg = D2LTrainConfig()
        assert cfg.model_id == "Qwen/Qwen3.5-9B"
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


def _make_transformers_mock(tokenizer: MagicMock, model: MagicMock) -> MagicMock:
    m = MagicMock()
    m.AutoTokenizer.from_pretrained.return_value = tokenizer
    m.AutoModelForCausalLM.from_pretrained.return_value = model
    m.TrainingArguments = MagicMock(return_value=MagicMock())
    return m


class TestRunDistillation:
    def test_calls_trainer_and_saves(self, tmp_path: object) -> None:
        corpus = pathlib.Path(str(tmp_path)) / "corpus.jsonl"
        corpus.write_text('{"messages": []}\n')

        cfg = D2LTrainConfig(
            corpus_path=str(corpus),
            checkpoint_dir=str(pathlib.Path(str(tmp_path)) / "ckpt"),  # type: ignore[arg-type]
        )

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 0
        mock_model = MagicMock()
        mock_trainer = MagicMock()

        mock_torch = MagicMock()
        mock_torch.float16 = "float16"
        mock_torch.float32 = "float32"
        mock_transformers = _make_transformers_mock(mock_tokenizer, mock_model)
        mock_datasets = MagicMock()
        mock_datasets.Dataset.from_list.return_value = MagicMock()
        mock_peft = MagicMock()
        mock_diff_loss = MagicMock()
        mock_diff_loss.build_diff_aware_sft_trainer.return_value = mock_trainer

        with patch.dict(
            sys.modules,
            {
                "torch": mock_torch,
                "transformers": mock_transformers,
                "peft": mock_peft,
                "datasets": mock_datasets,
                "rune.training.diff_loss": mock_diff_loss,
            },
        ):
            run_distillation(cfg)

        mock_trainer.train.assert_called_once()
        mock_trainer.save_model.assert_called_once()

    def test_resumes_from_checkpoint(self, tmp_path: object) -> None:
        corpus = pathlib.Path(str(tmp_path)) / "corpus.jsonl"
        corpus.write_text("")

        cfg = D2LTrainConfig(
            corpus_path=str(corpus),
            checkpoint_path="/some/checkpoint",
            checkpoint_dir=str(tmp_path),
        )

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_model = MagicMock()
        mock_trainer = MagicMock()

        mock_torch = MagicMock()
        mock_torch.float16 = "float16"
        mock_torch.float32 = "float32"
        mock_transformers = _make_transformers_mock(mock_tokenizer, mock_model)
        mock_datasets = MagicMock()
        mock_datasets.Dataset.from_list.return_value = MagicMock()
        mock_peft = MagicMock()
        mock_diff_loss = MagicMock()
        mock_diff_loss.build_diff_aware_sft_trainer.return_value = mock_trainer

        with patch.dict(
            sys.modules,
            {
                "torch": mock_torch,
                "transformers": mock_transformers,
                "peft": mock_peft,
                "datasets": mock_datasets,
                "rune.training.diff_loss": mock_diff_loss,
            },
        ):
            run_distillation(cfg)

        mock_trainer.train.assert_called_once_with(
            resume_from_checkpoint="/some/checkpoint"
        )
