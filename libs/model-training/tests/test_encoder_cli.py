"""Tests for the encoder pretraining CLI."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_PATHS = [
    str(_REPO_ROOT / "libs" / "model-training" / "src"),
    str(_REPO_ROOT / "libs" / "shared" / "src"),
    str(_REPO_ROOT / "libs" / "adapter-registry" / "src"),
    str(_REPO_ROOT / "libs" / "inference" / "src"),
]


def _subprocess_env() -> dict[str, str]:
    """Return env dict with PYTHONPATH covering the workspace source tree.

    Subprocess `python -m model_training.*` calls rely on this because
    the runner's .venv may not have the workspace installed in
    site-packages (only via pytest's pythonpath config).
    """
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    extra = os.pathsep.join(_SRC_PATHS)
    env["PYTHONPATH"] = f"{extra}{os.pathsep}{existing}" if existing else extra
    return env


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
        env=_subprocess_env(),
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
        env=_subprocess_env(),
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
        env=_subprocess_env(),
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
        env=_subprocess_env(),
    )
    assert result.returncode == 2
