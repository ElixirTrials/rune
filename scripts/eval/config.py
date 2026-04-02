"""Evaluation framework configuration.

Pluggable model/adapter/backend config, tier definitions, and generation
parameters for coding benchmark evaluation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum


class Backend(str, Enum):
    """Inference backend for completion generation."""

    VLLM = "vllm"
    TRANSFORMERS = "transformers"


class Tier(int, Enum):
    """Evaluation tier controlling scope and speed."""

    SMOKE = 1  # ~5 min, 20 HumanEval problems
    MINI = 2  # ~30 min, full HumanEval + MBPP sample
    FULL = 3  # hours, all benchmarks


class BenchmarkName(str, Enum):
    """Supported benchmark identifiers."""

    HUMANEVAL = "humaneval"
    MBPP = "mbpp"
    BIGCODEBENCH_COMPLETE = "bigcodebench-complete"
    BIGCODEBENCH_INSTRUCT = "bigcodebench-instruct"


@dataclass(frozen=True)
class GenerationParams:
    """Generation parameters for a specific pass@k setting."""

    n_samples: int
    temperature: float
    top_p: float
    max_tokens: int = 1024


PASS_AT_1 = GenerationParams(n_samples=1, temperature=0.0, top_p=1.0)
PASS_AT_10 = GenerationParams(n_samples=10, temperature=0.8, top_p=0.95)


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration for a single benchmark run."""

    name: BenchmarkName
    n_problems: int | None  # None = use all
    pass_k: list[GenerationParams] = field(default_factory=lambda: [PASS_AT_1])


# Tier definitions: which benchmarks and how many problems per tier.
TIER_CONFIGS: dict[Tier, list[BenchmarkConfig]] = {
    Tier.SMOKE: [
        BenchmarkConfig(
            name=BenchmarkName.HUMANEVAL,
            n_problems=20,
            pass_k=[PASS_AT_1],
        ),
    ],
    Tier.MINI: [
        BenchmarkConfig(
            name=BenchmarkName.HUMANEVAL,
            n_problems=None,
            pass_k=[PASS_AT_1],
        ),
        BenchmarkConfig(
            name=BenchmarkName.MBPP,
            n_problems=100,
            pass_k=[PASS_AT_1],
        ),
    ],
    Tier.FULL: [
        BenchmarkConfig(
            name=BenchmarkName.HUMANEVAL,
            n_problems=None,
            pass_k=[PASS_AT_1, PASS_AT_10],
        ),
        BenchmarkConfig(
            name=BenchmarkName.MBPP,
            n_problems=None,
            pass_k=[PASS_AT_1, PASS_AT_10],
        ),
        BenchmarkConfig(
            name=BenchmarkName.BIGCODEBENCH_COMPLETE,
            n_problems=None,
            pass_k=[PASS_AT_1],
        ),
        BenchmarkConfig(
            name=BenchmarkName.BIGCODEBENCH_INSTRUCT,
            n_problems=None,
            pass_k=[PASS_AT_1],
        ),
    ],
}


@dataclass(frozen=True)
class EvalConfig:
    """Top-level evaluation configuration."""

    model_id: str = field(
        default_factory=lambda: os.environ.get("EVAL_MODEL_ID", "google/gemma-2-2b-it")
    )
    adapter_path: str | None = field(
        default_factory=lambda: os.environ.get("RUNE_LORA_PATH")
    )
    backend: Backend = field(
        default_factory=lambda: Backend(os.environ.get("EVAL_BACKEND", "transformers"))
    )
    vllm_base_url: str = field(
        default_factory=lambda: os.environ.get(
            "VLLM_BASE_URL", "http://localhost:8100/v1"
        )
    )
    tier: Tier = Tier.SMOKE
    output_dir: str = "evaluation_results"
    seed: int = 42


# Published scores for leaderboard comparison (pass@1).
# Sources: EvalPlus leaderboard, BigCodeBench leaderboard (as of 2026-04).
PUBLISHED_SCORES: dict[str, dict[str, float]] = {
    "GPT-4o": {
        "humaneval+": 87.2,
        "mbpp+": 72.7,
        "bigcodebench-complete": 60.0,
        "bigcodebench-instruct": 48.0,
    },
    "Claude 3.5 Sonnet": {
        "humaneval+": 86.6,
        "mbpp+": 73.0,
    },
    "DeepSeek-Coder-V2": {
        "humaneval+": 82.3,
        "mbpp+": 70.1,
    },
}
