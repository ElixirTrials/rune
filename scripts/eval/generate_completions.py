"""Generate code completions for benchmark evaluation.

Loads benchmark prompts from EvalPlus (HumanEval+, MBPP+) and generates
completions using either a local transformers model or a remote vLLM server.
Supports pluggable base models and optional LoRA adapters.

Usage:
    uv run scripts/eval/generate_completions.py --tier 1 --mode base
    uv run scripts/eval/generate_completions.py --tier 2 --mode lora --adapter-path ./adapter
    uv run scripts/eval/generate_completions.py --benchmark humaneval --backend vllm
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

from config import (
    TIER_CONFIGS,
    Backend,
    BenchmarkConfig,
    BenchmarkName,
    EvalConfig,
    GenerationParams,
    Tier,
)


def load_humaneval_problems(
    n_problems: int | None = None, seed: int = 42
) -> list[dict[str, Any]]:
    """Load HumanEval+ problems via evalplus."""
    from evalplus.data import get_human_eval_plus

    problems = get_human_eval_plus()
    task_list = [
        {"task_id": tid, "prompt": data["prompt"]} for tid, data in problems.items()
    ]
    if n_problems is not None and n_problems < len(task_list):
        rng = random.Random(seed)
        task_list = rng.sample(task_list, n_problems)
    return task_list


def load_mbpp_problems(
    n_problems: int | None = None, seed: int = 42
) -> list[dict[str, Any]]:
    """Load MBPP+ problems via evalplus."""
    from evalplus.data import get_mbpp_plus

    problems = get_mbpp_plus()
    task_list = [
        {"task_id": tid, "prompt": data["prompt"]} for tid, data in problems.items()
    ]
    if n_problems is not None and n_problems < len(task_list):
        rng = random.Random(seed)
        task_list = rng.sample(task_list, n_problems)
    return task_list


def load_problems(benchmark: BenchmarkConfig, seed: int = 42) -> list[dict[str, Any]]:
    """Load problems for a given benchmark config."""
    if benchmark.name == BenchmarkName.HUMANEVAL:
        return load_humaneval_problems(benchmark.n_problems, seed)
    elif benchmark.name == BenchmarkName.MBPP:
        return load_mbpp_problems(benchmark.n_problems, seed)
    elif benchmark.name in (
        BenchmarkName.BIGCODEBENCH_COMPLETE,
        BenchmarkName.BIGCODEBENCH_INSTRUCT,
    ):
        print(
            f"BigCodeBench requires separate installation (pip install bigcodebench). "
            f"Skipping {benchmark.name.value}."
        )
        return []
    else:
        raise ValueError(f"Unknown benchmark: {benchmark.name}")


class TransformersBackend:
    """Generate completions using HuggingFace transformers + optional PEFT."""

    def __init__(self, model_id: str, adapter_path: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading model: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        if adapter_path:
            from peft import PeftModel

            print(f"Loading LoRA adapter: {adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, adapter_path)  # type: ignore[assignment]

        self.model.eval()

    def generate(self, prompt: str, params: GenerationParams) -> list[str]:
        """Generate n_samples completions for a single prompt."""
        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        completions = []
        for _ in range(params.n_samples):
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=params.max_tokens,
                    temperature=params.temperature if params.temperature > 0 else None,
                    top_p=params.top_p if params.temperature > 0 else None,
                    do_sample=params.temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
            completion = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            completions.append(completion)
        return completions


class VLLMBackend:
    """Generate completions via vLLM's OpenAI-compatible API."""

    def __init__(
        self,
        base_url: str,
        model_id: str,
        adapter_path: str | None = None,
    ):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key="not-needed")
        # When LoRA is loaded in vLLM, the model name is the adapter name
        self.model_name = "rune-adapter" if adapter_path else model_id

    def generate(self, prompt: str, params: GenerationParams) -> list[str]:
        """Generate n_samples completions for a single prompt."""
        response = self.client.completions.create(
            model=self.model_name,
            prompt=prompt,
            n=params.n_samples,
            max_tokens=params.max_tokens,
            temperature=params.temperature,
            top_p=params.top_p,
        )
        return [choice.text for choice in response.choices]


def create_backend(
    config: EvalConfig,
) -> TransformersBackend | VLLMBackend:
    """Create the appropriate inference backend."""
    adapter = config.adapter_path
    if config.backend == Backend.TRANSFORMERS:
        return TransformersBackend(config.model_id, adapter)
    elif config.backend == Backend.VLLM:
        return VLLMBackend(config.vllm_base_url, config.model_id, adapter)
    else:
        raise ValueError(f"Unknown backend: {config.backend}")


def generate_for_benchmark(
    backend: TransformersBackend | VLLMBackend,
    benchmark: BenchmarkConfig,
    config: EvalConfig,
) -> dict[str, list[dict[str, Any]]]:
    """Generate completions for all problems in a benchmark.

    Returns a dict keyed by pass@k label (e.g. "pass@1", "pass@10")
    with lists of {task_id, completion} dicts in EvalPlus format.
    """
    problems = load_problems(benchmark, config.seed)
    if not problems:
        return {}

    results: dict[str, list[dict[str, Any]]] = {}

    for gen_params in benchmark.pass_k:
        k_label = f"pass@{gen_params.n_samples}"
        samples: list[dict[str, Any]] = []

        total = len(problems)
        for i, problem in enumerate(problems, 1):
            task_id = problem["task_id"]
            prompt = problem["prompt"]

            print(
                f"  [{i}/{total}] {task_id} "
                f"(n={gen_params.n_samples}, t={gen_params.temperature})"
            )

            completions = backend.generate(prompt, gen_params)
            for completion in completions:
                samples.append({"task_id": task_id, "completion": completion})

        results[k_label] = samples

    return results


def write_samples(samples: list[dict[str, Any]], output_path: Path) -> None:
    """Write samples to JSONL file in EvalPlus format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    print(f"  Wrote {len(samples)} samples to {output_path}")


def run(config: EvalConfig) -> dict[str, Path]:
    """Run completion generation for the configured tier.

    Returns a dict mapping "{benchmark}_{k_label}" to the samples file path.
    """
    mode = "lora" if config.adapter_path else "base"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_dir = Path(config.output_dir) / timestamp / mode

    benchmarks = TIER_CONFIGS[config.tier]
    backend = create_backend(config)

    output_files: dict[str, Path] = {}

    for benchmark in benchmarks:
        print(f"\n=== {benchmark.name.value} ===")
        results = generate_for_benchmark(backend, benchmark, config)

        for k_label, samples in results.items():
            key = f"{benchmark.name.value}_{k_label}"
            output_path = base_dir / f"{key}_samples.jsonl"
            write_samples(samples, output_path)
            output_files[key] = output_path

    return output_files


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate code completions for benchmark evaluation"
    )
    parser.add_argument(
        "--tier",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help="Evaluation tier (1=smoke, 2=mini, 3=full)",
    )
    parser.add_argument(
        "--mode",
        choices=["base", "lora"],
        default="base",
        help="Run mode: base model or with LoRA adapter",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="HuggingFace model ID (default: EVAL_MODEL_ID env or gemma-2-2b-it)",
    )
    parser.add_argument(
        "--adapter-path",
        default=None,
        help="Path to PEFT adapter directory (default: RUNE_LORA_PATH env)",
    )
    parser.add_argument(
        "--backend",
        choices=["transformers", "vllm"],
        default=None,
        help="Inference backend (default: EVAL_BACKEND env or transformers)",
    )
    parser.add_argument(
        "--benchmark",
        choices=["humaneval", "mbpp", "bigcodebench-complete", "bigcodebench-instruct"],
        default=None,
        help="Run a single benchmark instead of the full tier",
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation_results",
        help="Output directory for results",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()

    config = EvalConfig(
        model_id=args.model or EvalConfig().model_id,
        adapter_path=args.adapter_path if args.mode == "lora" else None,
        backend=Backend(args.backend) if args.backend else EvalConfig().backend,
        tier=Tier(args.tier),
        output_dir=args.output_dir,
    )

    if args.mode == "lora" and not config.adapter_path:
        print("Error: --adapter-path or RUNE_LORA_PATH required for lora mode")
        sys.exit(1)

    print(f"Model: {config.model_id}")
    print(f"Backend: {config.backend.value}")
    print(f"Mode: {args.mode}")
    print(f"Tier: {config.tier.value}")
    if config.adapter_path:
        print(f"Adapter: {config.adapter_path}")

    output_files = run(config)

    print("\n=== Done ===")
    print(f"Generated {len(output_files)} sample files:")
    for key, path in output_files.items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
