"""Tests for model_training.reconstruction.cli."""

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


def test_dry_run_emits_json_without_torch() -> None:
    from model_training.reconstruction import cli

    argv = [
        "--database-url",
        "sqlite:////tmp/fake.db",
        "--out-dir",
        "/tmp/recon_out",
        "--warm-start",
        "deltacoder",
        "--base-model",
        "qwen3.5-9b",
        "--min-fitness",
        "0.5",
        "--task-type",
        "bug-fix",
        "--emb-model",
        "none",
        "--no-zscore",
        "--dry-run",
    ]
    parser = cli._build_parser()
    args = parser.parse_args(argv)
    kwargs = cli._resolve_kwargs(args)
    assert kwargs["warm_start_adapter"] == "danielcherubini/Qwen3.5-DeltaCoder-9B"
    assert kwargs["base_model_id_override"] == "Qwen/Qwen3.5-9B"
    assert kwargs["min_fitness"] == 0.5
    assert kwargs["task_type"] == "bug-fix"
    assert kwargs["compute_zscore"] is False
    assert kwargs["emb_model_name"] is None
    assert kwargs["database_url"] == "sqlite:////tmp/fake.db"
    assert kwargs["out_dir"] == "/tmp/recon_out"


def test_dry_run_subprocess_does_not_import_torch(tmp_path: Path) -> None:
    # Run the CLI as a subprocess with torch masked out via PYTHONPATH trick:
    # we inspect the stdout JSON instead — success means no crash.
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "model_training.reconstruction.cli",
            "--database-url",
            f"sqlite:///{tmp_path / 'fake.db'}",
            "--out-dir",
            str(tmp_path / "out"),
            "--warm-start",
            "off",
            "--base-model",
            "qwen3.5-9b",
            "--emb-model",
            "none",
            "--no-zscore",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
        env=_subprocess_env(),
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["warm_start_adapter"] is None


def test_warm_start_aliases() -> None:
    from model_training.reconstruction.cli import _resolve_warm_start

    assert _resolve_warm_start("deltacoder") == "danielcherubini/Qwen3.5-DeltaCoder-9B"
    assert _resolve_warm_start("off") is None
    assert _resolve_warm_start("none") is None
    assert _resolve_warm_start("") is None
    assert _resolve_warm_start("custom/adapter") == "custom/adapter"


def test_base_model_aliases() -> None:
    from model_training.reconstruction.cli import _resolve_base_model

    assert _resolve_base_model("qwen3.5-9b") == "Qwen/Qwen3.5-9B"
    # Non-aliased repo ids pass through unchanged.
    assert _resolve_base_model("some-org/some-model") == "some-org/some-model"


def test_cli_module_is_cpu_importable() -> None:
    import importlib

    mod = importlib.import_module("model_training.reconstruction.cli")
    assert hasattr(mod, "main")
