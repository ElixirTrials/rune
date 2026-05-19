"""Gate 3: Procedural-encoding strength (§4.3).

Evaluates 15 OOD functions × 8 held-out inputs via exact-match output
comparison. Compares substrate baseline vs Rune (substrate + hypernetwork
adapter). Applies paired McNemar + Bonferroni.

Supports two modes for the Rune condition:
  --rune-adapter-dir  Use pre-generated adapters (two-phase GPU sharing).
  --hypernet-checkpoint  Live hypernetwork (requires exclusive GPU).

Usage:
    uv run python scripts/paper/run_gate3.py \
        --rune-adapter-dir evaluation_results/paper/rune_adapters_ood \
        --output evaluation_results/gate3.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bootstrap import setup_path  # type: ignore[import-not-found]

setup_path()

from statistical_tests import (  # type: ignore[import-not-found]
    bonferroni_correct,
    mcnemar_test,
    wilson_score_ci,
)

logger = logging.getLogger(__name__)

DEFAULT_BASE_MODEL = "Qwen/Qwen3.5-9B"
DEFAULT_WARM_START = "danielcherubini/Qwen3.5-DeltaCoder-9B"


def _generate_completions(
    tasks: list[dict[str, str]],
    model: str,
    provider: Any,
    adapter_generator: Any | None = None,
) -> dict[str, str]:
    """Generate completions for all OOD tasks.

    Args:
        tasks: OOD task dicts with "task_id" and "prompt" fields.
        model: Base model ID.
        provider: InferenceProvider instance.
        adapter_generator: Optional callable(prompt) -> adapter_path for Rune.

    Returns:
        Dict mapping task_id -> generated completion.
    """
    loop = asyncio.new_event_loop()
    completions: dict[str, str] = {}
    try:
        for task in tasks:
            task_id = task["task_id"]
            prompt = task["prompt"]

            adapter_id_to_unload: str | None = None
            try:
                if adapter_generator is not None:
                    adapter_path = adapter_generator(prompt)
                    if adapter_path is not None:
                        adapter_id_to_unload = f"hypernet_{task_id}"
                        loop.run_until_complete(
                            provider.load_adapter(adapter_id_to_unload, adapter_path)
                        )

                gen_adapter_id = adapter_id_to_unload
                result = loop.run_until_complete(
                    provider.generate(
                        prompt=prompt,
                        model=model,
                        adapter_id=gen_adapter_id,
                        max_tokens=512,
                    )
                )
                completions[task_id] = result.text
            finally:
                if adapter_id_to_unload is not None:
                    try:
                        loop.run_until_complete(
                            provider.unload_adapter(adapter_id_to_unload)
                        )
                    except RuntimeError:
                        logger.warning(
                            "Failed to unload adapter %s", adapter_id_to_unload, exc_info=True
                        )
    finally:
        loop.close()

    return completions


def _pregenerate_ood_adapters(
    tasks: list[dict[str, str]],
    model: str,
    hypernet_checkpoint: str,
    device: str,
    output_dir: str,
) -> None:
    """Pre-generate adapters for OOD tasks (GPU-exclusive phase)."""
    import ctx_to_lora.modeling.hypernet as _hypernet_mod
    import torch
    from ctx_to_lora.modeling.lora_merger import combine_lora as _combine_lora
    from model_training.adapter_generator import _save_adapter
    from model_training.d2l_probe import extract_activations_with_model
    from model_training.hypernetwork import load_hypernetwork
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    def _device_safe_forward(
        self, features, attn_mask=None, position_ids=None, n_ctx_chunks=None
    ):
        dev = "cuda" if features.is_cuda else "cpu"
        with torch.autocast(device_type=dev, dtype=torch.bfloat16):
            if self.aggregator.layer_to_layer and self.iterative_mode:
                bs, n_layers = features.shape[0:2]
                lora_emb = torch.empty(
                    (bs, n_layers, self.num_modules, self.r, self.config.latent_size),
                    device=features.device,
                )
                for i in range(n_layers):
                    lora_emb[:, i], _ = self.aggregator(
                        features[:, i], attn_mask, position_ids
                    )
            else:
                lora_emb, _ = self.aggregator(features, attn_mask, position_ids)
        flat_loras = None
        if self.target_modules:
            lora_emb = self.layers(lora_emb)
            norm = torch.norm(lora_emb, dim=-1, keepdim=True)
            norm_lora_emb = lora_emb / norm
            flat_loras = self.head(norm_lora_emb)
        return flat_loras, None

    _hypernet_mod.HyperLoRA.forward = _device_safe_forward

    hypernet, hc = load_hypernetwork(hypernet_checkpoint, device=device)
    layer_indices = list(hc.layer_indices)

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model)
    base_model = AutoModelForCausalLM.from_pretrained(
        model, quantization_config=bnb_cfg, device_map="auto"
    )
    base_model.eval()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, str] = {}

    for task in tasks:
        task_id = task["task_id"]
        prompt = task["prompt"]
        task_dir = str(out / task_id)
        Path(task_dir).mkdir(parents=True, exist_ok=True)

        features, attn_mask = extract_activations_with_model(
            text=prompt,
            model=base_model,
            tokenizer=tokenizer,
            layer_indices=layer_indices,
        )
        with torch.no_grad():
            lora_dict, _ = hypernet.generate_weights(features, attn_mask, None)

        n_chunks = torch.ones(1, dtype=torch.int32)
        lora_bias = hypernet.get_head_bias() if hypernet.config.use_bias else None
        lora_dict = _combine_lora(lora_dict, n_chunks, lora_bias=lora_bias)

        _save_adapter(
            lora_dict=lora_dict,
            output_dir=task_dir,
            base_model_name=model,
            hc=hc,
            scaling_factor=0.16,
        )
        manifest[task_id] = task_dir
        logger.info("  OOD adapter: %s", task_id)

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info("Wrote %d OOD adapters to %s", len(manifest), output_dir)

    del hypernet, base_model
    import gc

    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate 3: OOD procedural encoding")
    parser.add_argument("--model", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument(
        "--warm-start-adapter",
        default=DEFAULT_WARM_START,
        help="Warm-start LoRA for substrate (DeltaCoder)",
    )
    parser.add_argument(
        "--hypernet-checkpoint",
        type=str,
        default=None,
        help="Path to trained hypernetwork checkpoint (live mode)",
    )
    parser.add_argument(
        "--rune-adapter-dir",
        type=str,
        default=None,
        help="Path to pre-generated OOD adapter dir (with manifest.json)",
    )
    parser.add_argument(
        "--pregenerate",
        action="store_true",
        help="Pre-generate OOD adapters (GPU phase) and exit",
    )
    parser.add_argument("--n-inputs", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output", type=Path, default=Path("evaluation_results/gate3.json")
    )
    args = parser.parse_args()

    ood_data_path = Path("libs/evaluation/src/evaluation/data/ood_tasks.json")
    with ood_data_path.open() as f:
        all_tasks: list[dict[str, str]] = json.load(f)

    if args.pregenerate:
        if not args.hypernet_checkpoint:
            parser.error("--pregenerate requires --hypernet-checkpoint")
        adapter_out = args.rune_adapter_dir or str(
            args.output.parent / "rune_adapters_ood"
        )
        _pregenerate_ood_adapters(
            all_tasks,
            args.model,
            args.hypernet_checkpoint,
            args.device,
            adapter_out,
        )
        print(f"OOD adapters written to {adapter_out}")
        return

    if not args.rune_adapter_dir and not args.hypernet_checkpoint:
        parser.error("Either --rune-adapter-dir or --hypernet-checkpoint is required")

    import subprocess

    from evaluation.ood_benchmark import run_ood_benchmark
    from inference.factory import get_provider
    from model_training.training_common import (
        mlflow_log_artifact,
        mlflow_log_params,
        setup_mlflow,
    )

    mlflow_ok = setup_mlflow("paper-gate3", tracking_uri=None)
    if mlflow_ok:
        import mlflow

        mlflow.start_run(run_name="gate3")
        mlflow_log_params(
            {
                "model": args.model,
                "warm_start_adapter": args.warm_start_adapter,
                "hypernet_checkpoint": args.hypernet_checkpoint or "pregenerated",
                "rune_adapter_dir": args.rune_adapter_dir or "",
                "n_inputs": args.n_inputs,
                "git_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    text=True,
                ).strip(),
            }
        )

    print(f"Gate 3: {len(all_tasks)} OOD tasks × {args.n_inputs} inputs")
    print(f"Model: {args.model}")
    print(f"Substrate: {args.warm_start_adapter}")

    provider = get_provider()

    print("\n=== Substrate baseline ===")
    substrate_completions = _generate_completions(
        all_tasks,
        args.model,
        provider,
    )
    substrate_result = run_ood_benchmark(
        adapter_id=None,
        completions=substrate_completions,
        benchmark_name="ood_python_substrate",
    )
    print(
        f"  Pass rate: {substrate_result['ood_pass_rate']:.2%} "
        f"({substrate_result['pass_count']}/{substrate_result['total']})"
    )

    print("\n=== Rune (substrate + hypernetwork adapter) ===")

    if args.rune_adapter_dir:
        manifest_path = Path(args.rune_adapter_dir) / "manifest.json"
        task_to_adapter = json.loads(manifest_path.read_text())
        prompt_to_adapter: dict[str, str] = {}
        for task in all_tasks:
            if task["task_id"] in task_to_adapter:
                prompt_to_adapter[task["prompt"]] = task_to_adapter[task["task_id"]]

        def adapter_generator(prompt: str) -> str | None:
            return prompt_to_adapter.get(prompt)
    else:
        import tempfile

        from rune_runner import run_hypernetwork  # type: ignore[import-not-found]

        tmp_dir = tempfile.mkdtemp(prefix="rune_gate3_")

        def adapter_generator(prompt: str) -> str | None:
            return run_hypernetwork(
                trajectory_text=prompt,
                output_dir=tmp_dir,
                base_model_id=args.model,
                checkpoint_path=args.hypernet_checkpoint,
                device=args.device,
            )

    rune_completions = _generate_completions(
        all_tasks,
        args.model,
        provider,
        adapter_generator=adapter_generator,
    )
    rune_result = run_ood_benchmark(
        adapter_id="rune_hypernet",
        completions=rune_completions,
        benchmark_name="ood_python_rune",
    )
    print(
        f"  Pass rate: {rune_result['ood_pass_rate']:.2%} "
        f"({rune_result['pass_count']}/{rune_result['total']})"
    )

    substrate_by_task = {
        r["task_id"]: r["passed"] for r in substrate_result["task_results"]
    }
    rune_by_task = {r["task_id"]: r["passed"] for r in rune_result["task_results"]}

    common_ids = sorted(set(substrate_by_task) & set(rune_by_task))
    paired = [(substrate_by_task[tid], rune_by_task[tid]) for tid in common_ids]
    mcnemar = mcnemar_test(paired)
    bonferroni = bonferroni_correct([mcnemar["p_value"]])

    n_sub = sum(1 for tid in common_ids if substrate_by_task[tid])
    n_rune = sum(1 for tid in common_ids if rune_by_task[tid])
    ci_substrate = wilson_score_ci(len(common_ids), n_sub)
    ci_rune = wilson_score_ci(len(common_ids), n_rune)

    print("\n=== Statistical Tests ===")
    print(f"  McNemar chi2={mcnemar['chi2']:.3f}, p={mcnemar['p_value']:.4f}")
    print(f"  Substrate CI: [{ci_substrate[0]:.3f}, {ci_substrate[1]:.3f}]")
    print(f"  Rune CI: [{ci_rune[0]:.3f}, {ci_rune[1]:.3f}]")

    report: dict[str, Any] = {
        "n_tasks": len(all_tasks),
        "n_inputs_per_task": args.n_inputs,
        "model": args.model,
        "warm_start_adapter": args.warm_start_adapter,
        "hypernet_checkpoint": args.hypernet_checkpoint or "pregenerated",
        "rune_adapter_dir": args.rune_adapter_dir,
        "substrate": {
            "pass_rate": substrate_result["ood_pass_rate"],
            "pass_count": substrate_result["pass_count"],
            "total": substrate_result["total"],
            "ci_95": {"lower": ci_substrate[0], "upper": ci_substrate[1]},
            "task_results": substrate_result["task_results"],
        },
        "rune": {
            "pass_rate": rune_result["ood_pass_rate"],
            "pass_count": rune_result["pass_count"],
            "total": rune_result["total"],
            "ci_95": {"lower": ci_rune[0], "upper": ci_rune[1]},
            "task_results": rune_result["task_results"],
        },
        "mcnemar": mcnemar,
        "bonferroni": bonferroni,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    delta = rune_result["ood_pass_rate"] - substrate_result["ood_pass_rate"]
    print(f"\nDelta (Rune - Substrate): {delta:+.2%}")
    print(f"Output: {args.output}")

    if mlflow_ok:
        mlflow.log_metric("substrate_ood_pass_rate", substrate_result["ood_pass_rate"])
        mlflow.log_metric("rune_ood_pass_rate", rune_result["ood_pass_rate"])
        mlflow.log_metric("ood_delta", delta)
        mlflow.log_metric("mcnemar_p_value", mcnemar["p_value"])
        mlflow_log_artifact(str(args.output))
        mlflow.end_run()


if __name__ == "__main__":
    main()
