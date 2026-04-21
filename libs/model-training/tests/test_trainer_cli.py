"""CPU tests for model_training.trainer_cli — dry-run flag coverage.

All tests use --dry-run so no GPU libraries are loaded.
"""

from __future__ import annotations

import json

import pytest
from model_training.trainer_cli import main


def _dry_run(extra_args: list[str]) -> dict:
    """Run trainer_cli in --dry-run mode and return parsed JSON output."""
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    base = ["--dataset", "/tmp/x.jsonl", "--adapter-id", "test-adapter"]
    rc = None
    with redirect_stdout(buf):
        rc = main(base + extra_args + ["--dry-run"])
    assert rc == 0
    return json.loads(buf.getvalue())


def test_dry_run_includes_warmup_ratio() -> None:
    """--warmup-ratio value appears in dry-run JSON; None when omitted."""
    payload = _dry_run(["--warmup-ratio", "0.05"])
    assert payload["warmup_ratio"] == pytest.approx(0.05)

    payload_none = _dry_run([])
    assert payload_none["warmup_ratio"] is None


def test_dry_run_includes_override_lora_alpha_and_dropout() -> None:
    """--override-lora-alpha and --override-lora-dropout appear in dry-run JSON."""
    payload = _dry_run(
        ["--override-lora-alpha", "64", "--override-lora-dropout", "0.05"]
    )
    assert payload["override_lora_alpha"] == 64
    assert payload["override_lora_dropout"] == pytest.approx(0.05)

    payload_none = _dry_run([])
    assert payload_none["override_lora_alpha"] is None
    assert payload_none["override_lora_dropout"] is None


def test_dry_run_includes_neftune_noise_alpha() -> None:
    """--neftune-noise-alpha value appears in dry-run JSON; None when omitted."""
    payload = _dry_run(["--neftune-noise-alpha", "5.0"])
    assert payload["neftune_noise_alpha"] == pytest.approx(5.0)

    payload_none = _dry_run([])
    assert payload_none["neftune_noise_alpha"] is None
