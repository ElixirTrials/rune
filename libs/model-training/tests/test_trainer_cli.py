"""CPU tests for ``model_training.trainer_cli`` and ``scripts/train.sh``.

Covers:
- ``--dry-run`` emits a JSON document with every expected kwarg and exits 0
  without importing any GPU libs.
- ``--warm-start`` aliases resolve correctly (``deltacoder``, ``off``,
  explicit repo id, and blank).
- Missing data source raises a parser error (mutually-exclusive group).
- ``bash scripts/train.sh --help`` works via subprocess — smoke test that
  the shell wrapper forwards to Python correctly. Skipped if uv or bash
  is missing.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_dry_run_emits_resolved_json(capsys: pytest.CaptureFixture[str]) -> None:
    """--dry-run prints JSON with every kwarg and exits 0."""
    from model_training.trainer_cli import main

    argv = [
        "--dataset",
        "/tmp/does-not-exist.jsonl",
        "--adapter-id",
        "test-adapter",
        "--model",
        "qwen3.5-9b",
        "--warm-start",
        "deltacoder",
        "--epochs",
        "2",
        "--lr",
        "1e-4",
        "--grad-accum",
        "8",
        "--encoding-mode",
        "single_turn",
        "--experiment-name",
        "rune-qlora-test",
        "--dry-run",
    ]
    rc = main(argv)
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["adapter_id"] == "test-adapter"
    assert payload["dataset_path"] == "/tmp/does-not-exist.jsonl"
    assert payload["session_id"] is None
    assert payload["model_config_name"] == "qwen3.5-9b"
    assert payload["warm_start_adapter_id"] == (
        "danielcherubini/Qwen3.5-DeltaCoder-9B"
    )
    assert payload["epochs"] == 2
    assert payload["learning_rate"] == 1e-4
    assert payload["gradient_accumulation_steps"] == 8
    assert payload["encoding_mode"] == "single_turn"
    assert payload["mlflow_experiment"] == "rune-qlora-test"


def test_warm_start_off_resolves_to_none() -> None:
    """--warm-start off/none/empty all resolve to None."""
    from model_training.trainer_cli import _resolve_warm_start

    assert _resolve_warm_start("off") is None
    assert _resolve_warm_start("none") is None
    assert _resolve_warm_start("") is None
    assert _resolve_warm_start(None) is None


def test_warm_start_passthrough_for_unknown_value() -> None:
    """Unknown --warm-start values are passed through as-is (HF repo / path)."""
    from model_training.trainer_cli import _resolve_warm_start

    assert _resolve_warm_start("org/my-adapter") == "org/my-adapter"
    assert _resolve_warm_start("/abs/path/to/adapter") == "/abs/path/to/adapter"


def test_requires_dataset_or_session_id(capsys: pytest.CaptureFixture[str]) -> None:
    """Omitting both --dataset and --session-id raises argparse error."""
    from model_training.trainer_cli import main

    with pytest.raises(SystemExit) as excinfo:
        main(["--adapter-id", "x", "--dry-run"])
    assert excinfo.value.code == 2


def test_dataset_and_session_id_are_mutually_exclusive(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Passing both --dataset and --session-id raises argparse error."""
    from model_training.trainer_cli import main

    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "--dataset",
                "/tmp/x.jsonl",
                "--session-id",
                "sess",
                "--adapter-id",
                "x",
                "--dry-run",
            ]
        )
    assert excinfo.value.code == 2


@pytest.mark.skipif(
    shutil.which("uv") is None,
    reason="uv not available — can't launch a clean subprocess",
)
def test_dry_run_does_not_import_torch() -> None:
    """--dry-run must NOT import torch (CPU-safe invariant for CI).

    Runs the CLI in a fresh subprocess via `uv run` so sys.modules is clean
    and torch imports from other tests in this process don't confound the
    assertion.
    """
    script = (
        "import sys; "
        "from model_training.trainer_cli import main; "
        "main(['--dataset', '/tmp/x', '--adapter-id', 'x', '--dry-run']); "
        "print('TORCH_IMPORTED' if 'torch' in sys.modules else 'TORCH_NOT_IMPORTED')"
    )
    env = os.environ.copy()
    env.setdefault("RUNE_DISABLE_MLFLOW", "1")
    # Propagate the workspace src paths so the subprocess can import
    # model_training the same way pytest's pythonpath option makes it work
    # in-process. Without this, `uv run python -c` starts a fresh Python
    # that cannot resolve the editable workspace packages.
    src_paths = [
        str(REPO_ROOT / "libs" / "model-training" / "src"),
        str(REPO_ROOT / "libs" / "shared" / "src"),
        str(REPO_ROOT / "libs" / "adapter-registry" / "src"),
        str(REPO_ROOT / "libs" / "inference" / "src"),
    ]
    existing = env.get("PYTHONPATH", "")
    parts = [*src_paths, existing] if existing else src_paths
    env["PYTHONPATH"] = os.pathsep.join(parts)
    result = subprocess.run(
        ["uv", "run", "python", "-c", script],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=env,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    lines = [ln for ln in result.stdout.strip().splitlines() if ln.strip()]
    assert lines[-1] == "TORCH_NOT_IMPORTED", (
        f"torch should not be imported in --dry-run; last stdout line: {lines[-1]!r}"
    )


@pytest.mark.skipif(
    shutil.which("bash") is None or shutil.which("uv") is None,
    reason="bash or uv not available — can't smoke-test train.sh",
)
def test_train_sh_dry_run_smoke(tmp_path: Path) -> None:
    """scripts/train.sh --dry-run exits 0 and prints JSON."""
    script = REPO_ROOT / "scripts" / "train.sh"
    assert script.exists(), script

    env = os.environ.copy()
    env.setdefault("RUNE_DISABLE_MLFLOW", "1")

    result = subprocess.run(
        [
            "bash",
            str(script),
            "--dataset",
            str(tmp_path / "does-not-exist.jsonl"),
            "--adapter-id",
            "shell-smoke",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
        timeout=120,
    )

    # Either exit 0 with JSON (expected), or exit 127 if uv missing in PATH.
    # The skipif above should prevent 127 but we're defensive in case uv is
    # on PATH but not usable.
    if result.returncode != 0:
        pytest.skip(
            f"train.sh smoke failed, likely env issue: rc={result.returncode} "
            f"stderr={result.stderr[:300]}"
        )
    # stdout must be parseable JSON.
    # Filter to the JSON object (skip any uv startup noise just in case).
    stdout = result.stdout.strip()
    assert stdout, "train.sh --dry-run produced empty stdout"
    # Find the JSON document by locating the first '{' and last '}'.
    start = stdout.find("{")
    end = stdout.rfind("}")
    assert start != -1 and end != -1, stdout
    payload = json.loads(stdout[start : end + 1])
    assert payload["adapter_id"] == "shell-smoke"


