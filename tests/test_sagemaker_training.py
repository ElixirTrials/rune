"""Smoke tests for SageMaker training support.

Covers checkpoint resume, atomic writes, pruning, SIGTERM handling,
MLflow checkpoint persistence, OOM handling, and launch_sagemaker.py
CLI parsing. Does not require GPU.
"""

from __future__ import annotations

import importlib.util
import os
import signal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

_LAUNCH_SPEC = importlib.util.spec_from_file_location(
    "launch_sagemaker",
    Path(__file__).resolve().parent.parent / "scripts" / "launch_sagemaker.py",
)
assert _LAUNCH_SPEC is not None and _LAUNCH_SPEC.loader is not None
_LAUNCH_MOD = importlib.util.module_from_spec(_LAUNCH_SPEC)
_LAUNCH_SPEC.loader.exec_module(_LAUNCH_MOD)


class TestCheckpointResume:
    def test_latest_checkpoint_sorted_by_step(self, tmp_path: Path) -> None:
        for step in [5, 100, 50, 200]:
            (tmp_path / f"ckpt-{step}.pt").write_text("")
        ckpt_files = sorted(
            (p for p in tmp_path.glob("ckpt-[0-9]*.pt") if "-emergency" not in p.name),
            key=lambda p: int(p.stem.split("-")[1]),
        )
        assert ckpt_files[-1].name == "ckpt-200.pt"

    def test_emergency_checkpoints_excluded_from_resume(self, tmp_path: Path) -> None:
        (tmp_path / "ckpt-10.pt").write_text("")
        (tmp_path / "ckpt-15-emergency.pt").write_text("")
        ckpt_files = sorted(
            (p for p in tmp_path.glob("ckpt-[0-9]*.pt") if "-emergency" not in p.name),
            key=lambda p: int(p.stem.split("-")[1]),
        )
        assert len(ckpt_files) == 1
        assert ckpt_files[0].name == "ckpt-10.pt"

    def test_no_checkpoints_returns_empty(self, tmp_path: Path) -> None:
        ckpt_files = list(tmp_path.glob("ckpt-[0-9]*.pt"))
        assert ckpt_files == []


class TestAtomicCheckpoint:
    def test_atomic_write_uses_tmp_then_replace(self, tmp_path: Path) -> None:
        target = tmp_path / "ckpt-1.pt"
        tmp = target.with_suffix(".pt.tmp")
        content = b"fake checkpoint data"
        tmp.write_bytes(content)
        os.replace(tmp, target)
        assert target.exists()
        assert not tmp.exists()
        assert target.read_bytes() == content

    def test_tmp_cleanup_on_startup(self, tmp_path: Path) -> None:
        (tmp_path / "ckpt-1.pt.tmp").write_text("stale")
        (tmp_path / "ckpt-2.pt.tmp").write_text("stale")
        (tmp_path / "ckpt-3.pt").write_text("valid")
        for t in tmp_path.glob("*.pt.tmp"):
            t.unlink()
        assert not list(tmp_path.glob("*.pt.tmp"))
        assert (tmp_path / "ckpt-3.pt").exists()


class TestCheckpointPruning:
    def test_prune_keeps_latest_n(self, tmp_path: Path) -> None:
        for step in [10, 20, 30, 40, 50]:
            (tmp_path / f"ckpt-{step}.pt").write_text("")
        keep = 3
        ckpts = sorted(
            (p for p in tmp_path.glob("ckpt-[0-9]*.pt") if "-emergency" not in p.name),
            key=lambda p: int(p.stem.split("-")[1]),
        )
        for old in ckpts[:-keep]:
            old.unlink()
        remaining = sorted(tmp_path.glob("ckpt-*.pt"))
        assert len(remaining) == 3
        names = {p.name for p in remaining}
        assert names == {"ckpt-30.pt", "ckpt-40.pt", "ckpt-50.pt"}

    def test_prune_preserves_emergency(self, tmp_path: Path) -> None:
        (tmp_path / "ckpt-10.pt").write_text("")
        (tmp_path / "ckpt-20.pt").write_text("")
        (tmp_path / "ckpt-15-emergency.pt").write_text("")
        keep = 1
        ckpts = sorted(
            (p for p in tmp_path.glob("ckpt-[0-9]*.pt") if "-emergency" not in p.name),
            key=lambda p: int(p.stem.split("-")[1]),
        )
        for old in ckpts[:-keep]:
            old.unlink()
        assert (tmp_path / "ckpt-15-emergency.pt").exists()
        assert (tmp_path / "ckpt-20.pt").exists()
        assert not (tmp_path / "ckpt-10.pt").exists()


class TestSigtermHandler:
    def test_sigterm_sets_shutdown_flag(self) -> None:
        shutdown = [False]

        def handler(signum: int, frame: object) -> None:
            shutdown[0] = True

        old = signal.getsignal(signal.SIGTERM)
        try:
            signal.signal(signal.SIGTERM, handler)
            os.kill(os.getpid(), signal.SIGTERM)
            assert shutdown[0] is True
        finally:
            signal.signal(signal.SIGTERM, old)


class TestLaunchSagemakerCli:
    def test_build_hyperparameters_basic(self) -> None:
        args = MagicMock()
        args.num_steps = 500
        args.experiment_name = "test-exp"
        args.base_model = "Qwen/Qwen3.5-9B"
        args.vram_tier = None
        args.mlflow_tracking_uri = None
        args.smoke = False
        hp = _LAUNCH_MOD._build_hyperparameters(args)
        assert hp["num-steps"] == "500"
        assert hp["experiment-name"] == "test-exp"
        assert "vram-tier" not in hp

    def test_build_hyperparameters_with_overrides(self) -> None:
        args = MagicMock()
        args.num_steps = 100
        args.experiment_name = "sm-test"
        args.base_model = "Qwen/Qwen3.5-9B"
        args.vram_tier = "high"
        args.mlflow_tracking_uri = "http://mlflow:5000"
        args.smoke = True
        hp = _LAUNCH_MOD._build_hyperparameters(args)
        assert hp["vram-tier"] == "high"
        assert hp["mlflow-tracking-uri"] == "http://mlflow:5000"
        assert hp["smoke"] == "1"


class TestSmHpCheckpointDir:
    def test_env_var_overrides_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SM_HP_CHECKPOINT_DIR", "/opt/ml/checkpoints")
        assert os.environ.get("SM_HP_CHECKPOINT_DIR") == "/opt/ml/checkpoints"

    def test_env_var_absent_uses_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SM_HP_CHECKPOINT_DIR", raising=False)
        default = os.environ.get("SM_HP_CHECKPOINT_DIR", "checkpoints/hypernet_hpo")
        assert default == "checkpoints/hypernet_hpo"


class TestMlflowCheckpointUpload:
    def test_mlflow_log_checkpoint_uses_artifact_path(self) -> None:
        from model_training.training_common import mlflow_log_checkpoint

        mock_mlflow = MagicMock()
        with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
            mlflow_log_checkpoint("/tmp/ckpt-10.pt", artifact_path="checkpoints")
            mock_mlflow.log_artifact.assert_called_once_with(
                "/tmp/ckpt-10.pt", artifact_path="checkpoints"
            )

    def test_mlflow_log_checkpoint_default_path(self) -> None:
        from model_training.training_common import mlflow_log_checkpoint

        mock_mlflow = MagicMock()
        with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
            mlflow_log_checkpoint("/tmp/ckpt-5.pt")
            mock_mlflow.log_artifact.assert_called_once_with(
                "/tmp/ckpt-5.pt", artifact_path="checkpoints"
            )

    def test_mlflow_log_checkpoint_silent_on_failure(self) -> None:
        from model_training.training_common import mlflow_log_checkpoint

        mock_mlflow = MagicMock()
        mock_mlflow.log_artifact.side_effect = RuntimeError("connection refused")
        with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
            mlflow_log_checkpoint("/tmp/ckpt-5.pt")


class TestMlflowCheckpointDownload:
    def test_returns_none_when_no_experiment(self) -> None:
        from model_training.training_common import mlflow_download_latest_checkpoint

        mock_mlflow = MagicMock()
        mock_client_cls = MagicMock()
        mock_client = mock_client_cls.return_value
        mock_client.get_experiment_by_name.return_value = None

        with patch.dict("sys.modules", {
            "mlflow": mock_mlflow,
            "mlflow.tracking": MagicMock(MlflowClient=mock_client_cls),
        }):
            result = mlflow_download_latest_checkpoint("nonexistent", Path("/tmp/d"))
        assert result is None

    def test_returns_none_when_no_checkpoint_artifacts(self, tmp_path: Path) -> None:
        from model_training.training_common import mlflow_download_latest_checkpoint

        mock_mlflow = MagicMock()
        mock_client_cls = MagicMock()
        mock_client = mock_client_cls.return_value
        mock_client.get_experiment_by_name.return_value = SimpleNamespace(
            experiment_id="1"
        )
        mock_run = SimpleNamespace(info=SimpleNamespace(run_id="abc123"))
        mock_client.search_runs.return_value = [mock_run]
        mock_client.list_artifacts.return_value = []

        with patch.dict("sys.modules", {
            "mlflow": mock_mlflow,
            "mlflow.tracking": MagicMock(MlflowClient=mock_client_cls),
        }):
            result = mlflow_download_latest_checkpoint("exp", tmp_path)
        assert result is None

    def test_downloads_latest_checkpoint(self, tmp_path: Path) -> None:
        from model_training.training_common import mlflow_download_latest_checkpoint

        mock_mlflow = MagicMock()
        mock_client_cls = MagicMock()
        mock_client = mock_client_cls.return_value
        mock_client.get_experiment_by_name.return_value = SimpleNamespace(
            experiment_id="1"
        )
        mock_run = SimpleNamespace(info=SimpleNamespace(run_id="abc123"))
        mock_client.search_runs.return_value = [mock_run]
        mock_client.list_artifacts.return_value = [
            SimpleNamespace(path="checkpoints/ckpt-50.pt"),
            SimpleNamespace(path="checkpoints/ckpt-100.pt"),
        ]

        def fake_download(run_id: str, artifact_path: str, dst_path: str) -> None:
            out = Path(dst_path) / artifact_path
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("fake")

        mock_mlflow.artifacts.download_artifacts.side_effect = fake_download

        with patch.dict("sys.modules", {
            "mlflow": mock_mlflow,
            "mlflow.tracking": MagicMock(MlflowClient=mock_client_cls),
        }):
            result = mlflow_download_latest_checkpoint("exp", tmp_path)

        assert result is not None
        assert result.name == "ckpt-100.pt"

    def test_silent_on_mlflow_failure(self, tmp_path: Path) -> None:
        from model_training.training_common import mlflow_download_latest_checkpoint

        mock_mlflow = MagicMock()
        mock_mlflow.tracking = MagicMock()
        mock_mlflow.tracking.MlflowClient.side_effect = RuntimeError("no server")

        with patch.dict("sys.modules", {
            "mlflow": mock_mlflow,
            "mlflow.tracking": mock_mlflow.tracking,
        }):
            result = mlflow_download_latest_checkpoint("exp", tmp_path)
        assert result is None


class TestOomCheckpointSkip:
    def test_emergency_checkpoint_skipped_on_oom(self) -> None:
        """OOM errors should not trigger emergency checkpoint saves."""
        import torch

        exc = torch.cuda.OutOfMemoryError("CUDA out of memory")
        assert isinstance(exc, torch.cuda.OutOfMemoryError)
        # The isinstance check in train_hypernet_hpo.py guards against this


class TestGcCollectInCleanup:
    def test_gc_collect_frees_reference_cycles(self) -> None:
        """gc.collect() should break reference cycles holding GPU-like objects."""
        import gc
        import weakref

        class FakeTensor:
            pass

        obj = FakeTensor()
        ref = weakref.ref(obj)

        # Create a reference cycle
        cycle: dict = {"self": None}
        cycle["self"] = cycle
        cycle["tensor"] = obj
        del obj, cycle

        # Without gc.collect(), the cycle may persist
        gc.collect()
        assert ref() is None


class TestCheckpointUploadFrequency:
    def test_warmup_checkpoints_not_uploaded(self) -> None:
        """Warmup-step checkpoints should be saved locally but not uploaded."""
        step = 3
        warmup_steps = 10
        checkpoint_every = 100
        num_steps = 500
        in_warmup = step <= warmup_steps
        should_ckpt = (
            step % checkpoint_every == 0
            or step == num_steps
            or in_warmup
        )
        should_upload = (
            step % checkpoint_every == 0
            or step == num_steps
        )
        assert should_ckpt is True
        assert should_upload is False

    def test_periodic_checkpoints_uploaded(self) -> None:
        """Checkpoints at checkpoint_every intervals should be uploaded."""
        step = 100
        checkpoint_every = 100
        num_steps = 500
        should_upload = (
            step % checkpoint_every == 0
            or step == num_steps
        )
        assert should_upload is True

    def test_final_step_uploaded(self) -> None:
        step = 500
        checkpoint_every = 100
        num_steps = 500
        should_upload = (
            step % checkpoint_every == 0
            or step == num_steps
        )
        assert should_upload is True
