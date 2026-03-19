"""Hardware detection and resource budgeting for swarm execution.

Provides HardwareProbe for detecting CPU, RAM, and GPU resources, and
HardwareBudget for computing concurrency limits based on available hardware.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import psutil

logger = logging.getLogger(__name__)

# Named constants for VRAM budgeting
VRAM_PER_LORA_MB: int = 100  # Approximate VRAM consumed by one LoRA adapter
MIN_TRAINING_VRAM_MB: int = 4096  # Minimum VRAM required to run a training job


@dataclass(frozen=True)
class GPUInfo:
    """Information about a single GPU device.

    Attributes:
        name: Human-readable GPU name (e.g. 'NVIDIA RTX 4090').
        vram_mb: Total VRAM in megabytes.
    """

    name: str
    vram_mb: int


@dataclass(frozen=True)
class HardwareBudget:
    """Concurrency limits derived from detected hardware.

    Attributes:
        max_agents: Maximum concurrent swarm agents.
        max_concurrent_loras: Maximum LoRA adapters loaded at once.
        training_slots: Number of concurrent training jobs (0 if no GPU).
        vram_per_gpu_mb: VRAM available per GPU (0 if no GPU).
        single_gpu_mode: True when only 0 or 1 GPUs are available.
    """

    max_agents: int
    max_concurrent_loras: int
    training_slots: int
    vram_per_gpu_mb: int
    single_gpu_mode: bool


@dataclass(frozen=True)
class HardwareProbe:
    """Detected hardware resources on the current machine.

    Attributes:
        cpu_count: Number of logical CPU cores.
        ram_total_mb: Total system RAM in megabytes.
        gpus: List of detected GPU devices.
    """

    cpu_count: int
    ram_total_mb: int
    gpus: list[GPUInfo] = field(default_factory=list)

    @classmethod
    def detect(cls) -> HardwareProbe:
        """Detect hardware resources on the current machine.

        Uses psutil for CPU/RAM detection (always available). Attempts to
        import pynvml for GPU detection; falls back to empty GPU list if
        pynvml is not installed or NVML initialization fails.

        Returns:
            A HardwareProbe with detected resource information.
        """
        cpu_count = psutil.cpu_count(logical=True) or 1
        ram_total_mb = psutil.virtual_memory().total // (1024 * 1024)

        gpus: list[GPUInfo] = []
        try:
            import pynvml

            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode("utf-8")
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                vram_mb = mem_info.total // (1024 * 1024)
                gpus.append(GPUInfo(name=name, vram_mb=vram_mb))
            pynvml.nvmlShutdown()
        except ImportError:
            pass  # pynvml not installed; CPU-only mode
        except Exception as e:
            logger.debug("GPU detection failed (will run CPU-only): %s", e)

        return cls(cpu_count=cpu_count, ram_total_mb=ram_total_mb, gpus=gpus)

    def compute_budget(self, base_model_vram_mb: int = 0) -> HardwareBudget:
        """Compute concurrency limits based on detected hardware.

        Args:
            base_model_vram_mb: VRAM consumed by the base inference model.
                Used to calculate how much VRAM is left for LoRA adapters
                and training.

        Returns:
            A HardwareBudget with computed concurrency limits.
        """
        num_gpus = len(self.gpus)
        single_gpu_mode = num_gpus <= 1

        if num_gpus == 0:
            return HardwareBudget(
                max_agents=max(1, self.cpu_count // 2),
                max_concurrent_loras=0,
                training_slots=0,
                vram_per_gpu_mb=0,
                single_gpu_mode=True,
            )

        vram_per_gpu = self.gpus[0].vram_mb
        free_vram = max(0, vram_per_gpu - base_model_vram_mb)

        max_loras = max(1, free_vram // VRAM_PER_LORA_MB)
        training_slots = 1 if free_vram >= MIN_TRAINING_VRAM_MB else 0

        # Agents limited by CPU cores and available VRAM
        max_agents = max(1, min(self.cpu_count // 2, num_gpus * 4))

        return HardwareBudget(
            max_agents=max_agents,
            max_concurrent_loras=max_loras,
            training_slots=training_slots,
            vram_per_gpu_mb=vram_per_gpu,
            single_gpu_mode=single_gpu_mode,
        )


# Bytes per parameter for each dtype
_BYTES_PER_PARAM = {
    "float32": 4,
    "bfloat16": 2,
    "float16": 2,
}

# VRAM safety margin — reserve 30% for KV cache, activations during forward
# pass, and CUDA memory fragmentation.
_VRAM_MARGIN = 0.70


def resolve_model_dtype(
    param_count: int,
    device: str = "cuda",
    available_vram_bytes: int = 0,
    overhead_bytes: int = 0,
    dtype_override: str | None = None,
) -> object:
    """Pick the highest precision dtype that fits in available VRAM.

    Resolution order:
      1. ``dtype_override`` parameter (explicit caller override)
      2. ``RUNE_DTYPE_OVERRIDE`` env var (global user override)
      3. Auto-detect: pick fp32 if it fits, otherwise bf16

    On CPU, always returns float32 (no VRAM constraint).

    Args:
        param_count: Number of model parameters.
        device: Target device (``"cuda"``, ``"cpu"``, ``"mps"``).
        available_vram_bytes: Total VRAM available on the device in bytes.
            If 0 and device is ``"cuda"``, queries the GPU automatically.
        overhead_bytes: Additional VRAM already in use or reserved by other
            models (e.g. an inference model already loaded).
        dtype_override: Manual override. One of ``"float32"``,
            ``"bfloat16"``, ``"float16"``.

    Returns:
        A ``torch.dtype`` suitable for loading the model.
    """
    import os  # noqa: PLC0415

    import torch  # noqa: PLC0415

    dtype_map = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }

    # 1. Check explicit override
    override = dtype_override or os.environ.get("RUNE_DTYPE_OVERRIDE")
    if override and override in dtype_map:
        return dtype_map[override]

    # 2. CPU — always fp32, no VRAM constraint
    if device == "cpu":
        return torch.float32

    # 3. Auto-detect VRAM if not provided
    if available_vram_bytes <= 0 and device == "cuda" and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        allocated = torch.cuda.memory_allocated(0)
        available_vram_bytes = props.total_memory - allocated

    usable_vram = int((available_vram_bytes - overhead_bytes) * _VRAM_MARGIN)
    fp32_bytes = param_count * _BYTES_PER_PARAM["float32"]

    if fp32_bytes <= usable_vram:
        return torch.float32

    return torch.bfloat16


def get_best_device() -> str:
    """Return the best available compute device: ``cuda`` > ``mps`` > ``cpu``.

    Imports torch inside the function body per INFRA-05 pattern so this module
    is importable in CPU-only CI without torch installed.

    Returns:
        ``"cuda"`` if an NVIDIA GPU is available,
        ``"mps"`` if an Apple Silicon GPU is available,
        ``"cpu"`` otherwise.

    Example:
        >>> device = get_best_device()
        >>> device in ("cuda", "mps", "cpu")
        True
    """
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"
