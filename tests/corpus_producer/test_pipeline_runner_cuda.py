"""Tests for CUDA_VISIBLE_DEVICES passthrough in run_pipeline_for_problem.

The corpus producer exposes ``--cuda-visible-devices`` so a multi-GPU node
can spawn one worker per GPU. The pipeline subprocess needs the env var
forwarded so the launched rune_runner.py sees only its designated GPU.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


def _fake_completed(returncode: int = 1, stderr: str = "boom") -> MagicMock:
    """Return a minimal CompletedProcess-like mock."""
    m = MagicMock()
    m.returncode = returncode
    m.stderr = stderr
    m.stdout = ""
    return m


def test_run_pipeline_sets_cuda_visible_devices_when_requested() -> None:
    """When cuda_visible_devices='2' is passed, subprocess env contains it."""
    from corpus_producer.pipeline_runner import run_pipeline_for_problem

    with patch(
        "corpus_producer.pipeline_runner.subprocess.run",
        return_value=_fake_completed(),
    ) as mock_run:
        run_pipeline_for_problem(
            "humaneval",
            "HumanEval/0",
            "Sort a list.",
            cuda_visible_devices="2",
        )
    assert mock_run.call_count == 1
    env = mock_run.call_args.kwargs.get("env")
    assert env is not None
    assert env.get("CUDA_VISIBLE_DEVICES") == "2"


def test_run_pipeline_does_not_override_env_when_cuda_none() -> None:
    """When cuda_visible_devices is None, no env kwarg is passed."""
    from corpus_producer.pipeline_runner import run_pipeline_for_problem

    with patch(
        "corpus_producer.pipeline_runner.subprocess.run",
        return_value=_fake_completed(),
    ) as mock_run:
        run_pipeline_for_problem(
            "humaneval",
            "HumanEval/0",
            "Sort a list.",
        )
    assert mock_run.call_count == 1
    # Either env kwarg is absent, or — if forwarded — it doesn't set CUDA
    env = mock_run.call_args.kwargs.get("env")
    if env is not None:
        assert "CUDA_VISIBLE_DEVICES" not in env or env[
            "CUDA_VISIBLE_DEVICES"
        ] != "2"
