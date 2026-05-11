"""Tests for run_all_conditions: incremental result flushing and HPO adapter fetch."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.paper.run_all_conditions import (
    assemble_table2,
    fetch_best_hpo_adapter,
    flush_partial_results,
)

# ── flush_partial_results ────────────────────────────────────────────


class TestFlushPartialResults:
    def test_writes_json_after_one_condition(self, tmp_path: Path) -> None:
        out = tmp_path / "results" / "table2.json"
        all_results = {"i": {"humaneval": 0.35, "mbpp": 0.42}}
        flush_partial_results(all_results, out)

        assert out.exists()
        data = json.loads(out.read_text())
        assert "i" in data["conditions"]
        assert data["conditions"]["i"]["scores"]["humaneval"] == 0.35

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        out = tmp_path / "deep" / "nested" / "dir" / "table2.json"
        flush_partial_results({"i": {"humaneval": 0.5}}, out)
        assert out.exists()

    def test_accumulates_across_conditions(self, tmp_path: Path) -> None:
        out = tmp_path / "table2.json"

        all_results: dict[str, dict[str, float]] = {}

        all_results["i"] = {"humaneval": 0.30}
        flush_partial_results(all_results, out)
        data = json.loads(out.read_text())
        assert len(data["conditions"]) == 1

        all_results["iii"] = {"humaneval": 0.45}
        flush_partial_results(all_results, out)
        data = json.loads(out.read_text())
        assert len(data["conditions"]) == 2
        assert data["conditions"]["iii"]["scores"]["humaneval"] == 0.45

    def test_includes_metadata_when_provided(self, tmp_path: Path) -> None:
        out = tmp_path / "table2.json"
        meta = {"model": "Qwen/Qwen3.5-9B", "warm_start_adapter": "deltacoder"}
        flush_partial_results({"i": {"humaneval": 0.5}}, out, metadata=meta)

        data = json.loads(out.read_text())
        assert data["metadata"]["model"] == "Qwen/Qwen3.5-9B"

    def test_overwrites_previous_file(self, tmp_path: Path) -> None:
        out = tmp_path / "table2.json"
        flush_partial_results({"i": {"humaneval": 0.30}}, out)
        flush_partial_results({"i": {"humaneval": 0.35}}, out)

        data = json.loads(out.read_text())
        assert data["conditions"]["i"]["scores"]["humaneval"] == 0.35

    def test_computes_deltas_vs_substrate(self, tmp_path: Path) -> None:
        out = tmp_path / "table2.json"
        all_results = {
            "i": {"humaneval": 0.30, "mbpp": 0.40},
            "iii": {"humaneval": 0.45, "mbpp": 0.50},
        }
        flush_partial_results(all_results, out)

        data = json.loads(out.read_text())
        delta = data["conditions"]["iii"]["delta_vs_substrate"]["humaneval"]
        assert abs(delta - 0.15) < 1e-9


# ── assemble_table2 ─────────────────────────────────────────────────


class TestAssembleTable2:
    def test_empty_results(self) -> None:
        table = assemble_table2({})
        assert table["conditions"] == {}

    def test_single_condition_no_delta(self) -> None:
        table = assemble_table2({"i": {"humaneval": 0.30}})
        assert table["conditions"]["i"]["delta_vs_substrate"]["humaneval"] == 0.0

    def test_delta_computed_against_condition_i(self) -> None:
        results = {
            "i": {"humaneval": 0.30},
            "v": {"humaneval": 0.50},
        }
        table = assemble_table2(results)
        assert (
            abs(table["conditions"]["v"]["delta_vs_substrate"]["humaneval"] - 0.20)
            < 1e-9
        )

    def test_label_assignment(self) -> None:
        table = assemble_table2({"v": {"humaneval": 0.5}})
        assert table["conditions"]["v"]["label"] == "Rune (ours)"


# ── fetch_best_hpo_adapter ──────────────────────────────────────────


class TestFetchBestHpoAdapter:
    def test_returns_immediately_if_adapter_exists(self, tmp_path: Path) -> None:
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_config.json").write_text("{}")

        result = fetch_best_hpo_adapter(adapter_dir)
        assert result == adapter_dir

    def test_s3_download_success(self, tmp_path: Path) -> None:
        adapter_dir = tmp_path / "adapter"

        def fake_s3_run(cmd, capture_output=False, **kwargs):
            adapter_dir.mkdir(parents=True, exist_ok=True)
            (adapter_dir / "adapter_config.json").write_text('{"type": "lora"}')
            return MagicMock(returncode=0)

        with patch(
            "scripts.paper.run_all_conditions.subprocess.run", side_effect=fake_s3_run
        ):
            result = fetch_best_hpo_adapter(adapter_dir)

        assert result == adapter_dir
        assert (adapter_dir / "adapter_config.json").exists()

    def test_falls_back_to_mlflow_on_s3_failure(self, tmp_path: Path) -> None:
        adapter_dir = tmp_path / "adapter"
        call_log: list[list[str]] = []

        def fake_run(cmd, capture_output=False, env=None, **kwargs):
            call_log.append(list(cmd))
            if "s3" in cmd[0] or "aws" in cmd[0]:
                return MagicMock(returncode=1)
            # MLflow fallback
            adapter_dir.mkdir(parents=True, exist_ok=True)
            (adapter_dir / "adapter_config.json").write_text('{"type": "lora"}')
            return MagicMock(returncode=0)

        with patch(
            "scripts.paper.run_all_conditions.subprocess.run", side_effect=fake_run
        ):
            result = fetch_best_hpo_adapter(adapter_dir)

        assert result == adapter_dir
        assert len(call_log) == 2
        assert "aws" in call_log[0][0]
        assert "mlflow" in call_log[1]

    def test_raises_when_both_fail(self, tmp_path: Path) -> None:
        adapter_dir = tmp_path / "adapter"

        def fake_run(cmd, **kwargs):
            return MagicMock(returncode=1)

        with patch(
            "scripts.paper.run_all_conditions.subprocess.run", side_effect=fake_run
        ):
            with pytest.raises(FileNotFoundError, match="Could not fetch HPO adapter"):
                fetch_best_hpo_adapter(adapter_dir)

    def test_uses_custom_run_id(self, tmp_path: Path) -> None:
        adapter_dir = tmp_path / "adapter"
        custom_id = "abc123"
        cmd_log: list[list[str]] = []

        def fake_run(cmd, capture_output=False, env=None, **kwargs):
            cmd_log.append(list(cmd))
            if "aws" in cmd[0]:
                return MagicMock(returncode=1)
            adapter_dir.mkdir(parents=True, exist_ok=True)
            (adapter_dir / "adapter_config.json").write_text("{}")
            return MagicMock(returncode=0)

        with patch(
            "scripts.paper.run_all_conditions.subprocess.run", side_effect=fake_run
        ):
            fetch_best_hpo_adapter(
                adapter_dir,
                run_id=custom_id,
                s3_prefix=f"s3://bucket/{custom_id}",
            )

        mlflow_cmd = cmd_log[1]
        assert custom_id in mlflow_cmd

    def test_passes_mlflow_tracking_uri(self, tmp_path: Path) -> None:
        adapter_dir = tmp_path / "adapter"
        env_log: list[dict | None] = []

        def fake_run(cmd, capture_output=False, env=None, **kwargs):
            env_log.append(env)
            if "aws" in cmd[0]:
                return MagicMock(returncode=1)
            adapter_dir.mkdir(parents=True, exist_ok=True)
            (adapter_dir / "adapter_config.json").write_text("{}")
            return MagicMock(returncode=0)

        with patch(
            "scripts.paper.run_all_conditions.subprocess.run", side_effect=fake_run
        ):
            fetch_best_hpo_adapter(
                adapter_dir,
                mlflow_tracking_uri="http://mlflow:5000",
            )

        mlflow_env = env_log[1]
        assert mlflow_env is not None
        assert mlflow_env["MLFLOW_TRACKING_URI"] == "http://mlflow:5000"
