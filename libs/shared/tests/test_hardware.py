"""Tests for shared.hardware — HardwareProbe and HardwareBudget."""

import sys
from types import ModuleType
from unittest.mock import MagicMock

import torch
from shared.hardware import GPUInfo, HardwareBudget, HardwareProbe, resolve_model_dtype


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


# ---------------------------------------------------------------------------
# resolve_model_dtype tests
# ---------------------------------------------------------------------------

# Bytes-per-param for reference: fp32=4, bf16=2, fp16=2
_2B_PARAMS = 2_600_000_000  # gemma-2-2b
_300M_PARAMS = 300_000_000  # HyperLoRA perceiver (approx)


def test_resolve_dtype_override_respected() -> None:
    """Manual override always wins regardless of VRAM."""
    dt = resolve_model_dtype(
        param_count=_2B_PARAMS,
        device="cuda",
        available_vram_bytes=4 * 1024**3,  # 4 GB — way too small for fp32
        dtype_override="float32",
    )
    assert dt == torch.float32


def test_resolve_dtype_override_bfloat16() -> None:
    """Override 'bfloat16' returns torch.bfloat16."""
    dt = resolve_model_dtype(
        param_count=_2B_PARAMS,
        device="cuda",
        available_vram_bytes=100 * 1024**3,
        dtype_override="bfloat16",
    )
    assert dt == torch.bfloat16


def test_resolve_dtype_override_float16() -> None:
    """Override 'float16' returns torch.float16."""
    dt = resolve_model_dtype(
        param_count=_2B_PARAMS,
        device="cuda",
        available_vram_bytes=100 * 1024**3,
        dtype_override="float16",
    )
    assert dt == torch.float16


def test_resolve_dtype_cpu_always_float32() -> None:
    """CPU device always returns float32 (no VRAM constraint)."""
    dt = resolve_model_dtype(
        param_count=_2B_PARAMS,
        device="cpu",
        available_vram_bytes=0,
    )
    assert dt == torch.float32


def test_resolve_dtype_plenty_of_vram_returns_float32() -> None:
    """With 40GB free and a 2B model, fp32 fits — use it."""
    dt = resolve_model_dtype(
        param_count=_2B_PARAMS,
        device="cuda",
        available_vram_bytes=40 * 1024**3,  # 40 GB
    )
    assert dt == torch.float32


def test_resolve_dtype_tight_vram_returns_bfloat16() -> None:
    """With 8GB free and a 2B model, fp32 doesn't fit but bf16 does."""
    dt = resolve_model_dtype(
        param_count=_2B_PARAMS,
        device="cuda",
        available_vram_bytes=8 * 1024**3,  # 8 GB
    )
    assert dt == torch.bfloat16


def test_resolve_dtype_very_tight_vram_still_bfloat16() -> None:
    """Even with very tight VRAM, bf16 is the floor (we don't go lower)."""
    dt = resolve_model_dtype(
        param_count=_2B_PARAMS,
        device="cuda",
        available_vram_bytes=2 * 1024**3,  # 2 GB — even bf16 won't fit
    )
    assert dt == torch.bfloat16


def test_resolve_dtype_small_model_fits_fp32() -> None:
    """A small 300M model in fp32 needs ~1.2GB — fits in 8GB easily."""
    dt = resolve_model_dtype(
        param_count=_300M_PARAMS,
        device="cuda",
        available_vram_bytes=8 * 1024**3,
    )
    assert dt == torch.float32


def test_resolve_dtype_overhead_bytes_counted() -> None:
    """Overhead bytes reduce effective VRAM, can push fp32 to bf16."""
    # 2B params * 4 bytes = 10GB. With 12GB VRAM and 4GB overhead, only 8GB left.
    dt = resolve_model_dtype(
        param_count=_2B_PARAMS,
        device="cuda",
        available_vram_bytes=12 * 1024**3,
        overhead_bytes=4 * 1024**3,
    )
    assert dt == torch.bfloat16


def test_resolve_dtype_env_var_override(monkeypatch) -> None:
    """RUNE_DTYPE_OVERRIDE env var takes precedence over auto-detection."""
    monkeypatch.setenv("RUNE_DTYPE_OVERRIDE", "float16")
    dt = resolve_model_dtype(
        param_count=_300M_PARAMS,
        device="cuda",
        available_vram_bytes=100 * 1024**3,
    )
    assert dt == torch.float16
