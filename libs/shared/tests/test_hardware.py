"""Tests for shared.hardware — HardwareProbe and HardwareBudget."""

import sys
from types import ModuleType
from unittest.mock import MagicMock

from shared.hardware import GPUInfo, HardwareBudget, HardwareProbe


def test_probe_returns_correct_cpu_ram() -> None:
    """HardwareProbe.detect() returns positive cpu_count and ram_total_mb."""
    probe = HardwareProbe.detect()
    assert probe.cpu_count > 0
    assert probe.ram_total_mb > 0


def test_probe_gpu_fallback_no_pynvml(monkeypatch) -> None:
    """When pynvml is unavailable, gpus list is empty."""
    # Ensure pynvml import fails
    original = sys.modules.get("pynvml")
    monkeypatch.setitem(sys.modules, "pynvml", None)
    try:
        probe = HardwareProbe.detect()
        assert probe.gpus == []
    finally:
        if original is not None:
            sys.modules["pynvml"] = original
        elif "pynvml" in sys.modules:
            del sys.modules["pynvml"]


def test_probe_with_mock_gpu(monkeypatch) -> None:
    """When pynvml is available, GPU info is detected."""
    fake_pynvml = ModuleType("pynvml")
    fake_pynvml.nvmlInit = MagicMock()  # type: ignore[attr-defined]
    fake_pynvml.nvmlShutdown = MagicMock()  # type: ignore[attr-defined]
    fake_pynvml.nvmlDeviceGetCount = MagicMock(return_value=1)  # type: ignore[attr-defined]

    mock_handle = MagicMock()
    fake_pynvml.nvmlDeviceGetHandleByIndex = MagicMock(return_value=mock_handle)  # type: ignore[attr-defined]
    fake_pynvml.nvmlDeviceGetName = MagicMock(return_value="NVIDIA RTX 4090")  # type: ignore[attr-defined]

    mock_mem = MagicMock()
    mock_mem.total = 24 * 1024 * 1024 * 1024  # 24 GB
    fake_pynvml.nvmlDeviceGetMemoryInfo = MagicMock(return_value=mock_mem)  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "pynvml", fake_pynvml)
    try:
        probe = HardwareProbe.detect()
        assert len(probe.gpus) == 1
        assert probe.gpus[0].name == "NVIDIA RTX 4090"
        assert probe.gpus[0].vram_mb == 24 * 1024
    finally:
        del sys.modules["pynvml"]


def test_compute_budget_no_gpu() -> None:
    """No GPUs: training_slots=0, single_gpu_mode=True."""
    probe = HardwareProbe(cpu_count=8, ram_total_mb=16384, gpus=[])
    budget = probe.compute_budget()
    assert budget.training_slots == 0
    assert budget.single_gpu_mode is True
    assert budget.max_agents >= 1
    assert budget.vram_per_gpu_mb == 0


def test_compute_budget_single_gpu() -> None:
    """Single 24GB GPU: training_slots=1, single_gpu_mode=True."""
    gpu = GPUInfo(name="RTX 4090", vram_mb=24576)
    probe = HardwareProbe(cpu_count=16, ram_total_mb=65536, gpus=[gpu])
    budget = probe.compute_budget(base_model_vram_mb=8000)
    assert budget.training_slots == 1
    assert budget.single_gpu_mode is True
    assert budget.vram_per_gpu_mb == 24576
    assert budget.max_concurrent_loras >= 1


def test_compute_budget_multi_gpu() -> None:
    """Two GPUs: single_gpu_mode=False."""
    gpus = [
        GPUInfo(name="RTX 4090 #0", vram_mb=24576),
        GPUInfo(name="RTX 4090 #1", vram_mb=24576),
    ]
    probe = HardwareProbe(cpu_count=16, ram_total_mb=65536, gpus=gpus)
    budget = probe.compute_budget(base_model_vram_mb=8000)
    assert budget.single_gpu_mode is False
    assert budget.training_slots == 1
    assert isinstance(budget, HardwareBudget)
