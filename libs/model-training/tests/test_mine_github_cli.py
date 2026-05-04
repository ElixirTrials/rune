"""Smoke test for the trajectory mining CLI argument parsing."""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
sys.path.insert(0, str(_SCRIPTS))

import mine_github  # noqa: E402  type: ignore[import-untyped]


def test_cli_rejects_missing_token(monkeypatch, capsys, tmp_path) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setattr(
        sys, "argv", ["mine_github.py", "--repo", "owner/repo", "-o", str(tmp_path / "out.jsonl")]
    )
    try:
        mine_github.main()
    except SystemExit as e:
        assert e.code == 1


def test_cli_no_mode_flag(monkeypatch, tmp_path) -> None:
    """The --mode flag is gone (issue mining removed)."""
    monkeypatch.setenv("GITHUB_TOKEN", "x")
    monkeypatch.setattr(
        sys, "argv", ["mine_github.py", "--repo", "o/r", "--mode", "issues", "-o", str(tmp_path / "out.jsonl")]
    )
    try:
        mine_github.main()
    except SystemExit as e:
        assert e.code == 2
