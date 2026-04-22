"""Tests for the encoder pretraining CLI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_dry_run_prints_json_without_torch(tmp_path: Path) -> None:
    """--dry-run must print valid JSON config without importing torch."""
    aug_dir = tmp_path / "pairs_augmented"
    aug_dir.mkdir()
    out_dir = tmp_path / "encoder_out"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "model_training.encoder_pretrain.cli",
            "--augmented-dir",
            str(aug_dir),
            "--output-dir",
            str(out_dir),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    config = json.loads(result.stdout)
    assert config["augmented_dir"] == str(aug_dir)
    assert config["output_dir"] == str(out_dir)
    assert config["base_encoder"] == "sentence-transformers/all-mpnet-base-v2"
    assert "temperature" in config
    assert "batch_size" in config
    assert "epochs" in config


def test_dry_run_respects_overrides(tmp_path: Path) -> None:
    """--dry-run prints overridden hyperparameter values."""
    aug_dir = tmp_path / "pairs_augmented"
    aug_dir.mkdir()
    out_dir = tmp_path / "encoder_out"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "model_training.encoder_pretrain.cli",
            "--augmented-dir",
            str(aug_dir),
            "--output-dir",
            str(out_dir),
            "--batch-size",
            "32",
            "--epochs",
            "3",
            "--temperature",
            "0.1",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    config = json.loads(result.stdout)
    assert config["batch_size"] == 32
    assert config["epochs"] == 3
    assert abs(config["temperature"] - 0.1) < 1e-6


def test_help_exits_cleanly() -> None:
    """--help exits with code 0 and prints usage."""
    result = subprocess.run(
        [sys.executable, "-m", "model_training.encoder_pretrain.cli", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "augmented-dir" in result.stdout
    assert "dry-run" in result.stdout


def test_missing_required_arg_exits_nonzero() -> None:
    """Missing --augmented-dir or --output-dir exits with code 2."""
    result = subprocess.run(
        [sys.executable, "-m", "model_training.encoder_pretrain.cli"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
