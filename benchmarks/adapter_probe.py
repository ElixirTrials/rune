# benchmarks/adapter_probe.py
"""Adapter retrievability probe: verify hypernetwork adapters influence generation.

Run: uv run python benchmarks/adapter_probe.py [--config benchmarks/bench.yaml]

All generation params and sweep values come from bench.yaml.
Results logged to MLflow experiment 'adapter-probe'.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

_DEFAULT_CONFIG = Path("benchmarks/bench.yaml")


@dataclass
class ProbeResult:
    condition: str
    trajectory: str
    prompt: str
    output: str
    adapter_scaling: float
    tokens_used: int = 0
    elapsed_s: float = 0.0
    tags: dict[str, str] = field(default_factory=dict)


def _run_condition(
    model: Any,
    condition: str,
    trajectory: str,
    prompt: str,
    system_prompt: str,
    scaling: float,
    gen_kwargs: dict[str, object],
    tags: dict[str, str] | None = None,
) -> ProbeResult:
    import asyncio  # noqa: PLC0415

    t0 = time.monotonic()
    adapter = model.generate_adapter(trajectory)
    scaled_sd = {
        k: v * scaling if "lora_B" in k else v
        for k, v in adapter.state_dict.items()
    }
    model.hotswap_adapter(scaled_sd)
    out = asyncio.run(
        model.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            **gen_kwargs,
        )
    )
    elapsed = time.monotonic() - t0
    return ProbeResult(
        condition=condition,
        trajectory=trajectory,
        prompt=prompt,
        output=out.text,
        adapter_scaling=scaling,
        tokens_used=out.tokens_used,
        elapsed_s=round(elapsed, 3),
        tags=tags or {},
    )


def run_probe(cfg: Any) -> list[ProbeResult]:
    from rune.model.wrapper import ModelWrapper  # noqa: PLC0415

    model = ModelWrapper.from_config(cfg)
    default_scaling = cfg.adapter_scaling
    bench = cfg.bench
    scaling_sweep: list[float] = bench["probe_scaling_sweep"]
    gen_kwargs = {
        "max_tokens": bench["probe_max_tokens"],
        "temperature": bench["gen_temperature"],
    }

    task_prompt = (
        "Write a Python function add(a, b) that returns a + b."
    )
    task_trajectory = (
        f"ROLE: coder\nTASK: {task_prompt}\n"
        "PLAN: Implement add function."
    )
    enriched_trajectory = (
        f"ROLE: coder\nTASK: {task_prompt}\n"
        "PLAN: Implement add(a, b) -> int returning a + b.\n"
        "INTERFACE: def add(a: int, b: int) -> int\n"
        "ERROR: NameError: name 'ad' is not defined\n"
        "FIX: Correct the typo in the function name."
    )
    contradictory_trajectory = (
        "ROLE: coder\n"
        "TASK: Sort a list of integers in ascending order.\n"
        "PLAN: Implement bubble sort with early termination.\n"
        "INTERFACE: def sort_list(items: list[int]) -> list[int]"
    )

    system_prompt = "You are a code generator."
    results: list[ProbeResult] = []

    logger.info("Condition 1/5: Baseline (adapter_scaling=0)")
    results.append(_run_condition(
        model, "baseline", task_trajectory, task_prompt,
        system_prompt, 0.0, gen_kwargs,
        tags={"phase": "baseline"},
    ))

    logger.info(
        "Condition 2/5: Task trajectory (scaling=%.3f)",
        default_scaling,
    )
    results.append(_run_condition(
        model, "task_trajectory", task_trajectory, task_prompt,
        system_prompt, default_scaling, gen_kwargs,
        tags={"phase": "task_trajectory"},
    ))

    logger.info(
        "Condition 3/5: Enriched trajectory (scaling=%.3f)",
        default_scaling,
    )
    results.append(_run_condition(
        model, "enriched_trajectory", enriched_trajectory,
        task_prompt, system_prompt, default_scaling, gen_kwargs,
        tags={"phase": "enriched_trajectory"},
    ))

    logger.info(
        "Condition 4/5: Contradictory trajectory (scaling=%.3f)",
        default_scaling,
    )
    results.append(_run_condition(
        model, "contradictory", contradictory_trajectory,
        task_prompt, system_prompt, default_scaling, gen_kwargs,
        tags={"phase": "contradictory"},
    ))

    for scale in scaling_sweep:
        logger.info(
            "Condition 5/5: Scaling sweep (adapter_scaling=%.3f)",
            scale,
        )
        results.append(_run_condition(
            model, f"scaling_{scale}", task_trajectory, task_prompt,
            system_prompt, scale, gen_kwargs,
            tags={
                "phase": "scaling_sweep",
                "sweep_value": str(scale),
            },
        ))

    return results


def log_to_mlflow(results: list[ProbeResult], cfg: Any) -> None:
    import mlflow  # noqa: PLC0415, I001
    from rune.tracking import configure_mlflow, tracked_run  # noqa: PLC0415

    configure_mlflow("adapter-probe")

    bench = cfg.bench
    gen_kwargs = {
        "max_tokens": bench["probe_max_tokens"],
        "temperature": bench["gen_temperature"],
    }

    params = {
        **cfg.to_dict(),
        "n_conditions": len(results),
        "generation_max_tokens": gen_kwargs["max_tokens"],
        "generation_temperature": gen_kwargs["temperature"],
        "scaling_sweep_values": json.dumps(
            bench["probe_scaling_sweep"],
        ),
    }

    with tracked_run("adapter-probe", params=params) as run:
        mlflow.set_tags({
            "experiment_type": "adapter_retrievability_probe",
            "phase": "0",
        })

        baseline_output = results[0].output

        for r in results:
            differs = r.output != baseline_output
            mlflow.log_text(
                json.dumps(asdict(r), indent=2),
                f"probe/{r.condition}.json",
            )
            mlflow.log_metrics({
                f"differs_from_baseline/{r.condition}": int(differs),
                f"tokens_used/{r.condition}": r.tokens_used,
                f"elapsed_s/{r.condition}": r.elapsed_s,
                f"output_len/{r.condition}": len(r.output),
            })
            logger.info(
                "%s (scale=%.3f): differs=%s, tokens=%d,"
                " elapsed=%.1fs, len=%d",
                r.condition, r.adapter_scaling, differs,
                r.tokens_used, r.elapsed_s, len(r.output),
            )

        task_differs = results[1].output != baseline_output
        enriched_differs = (
            results[2].output != results[1].output
        )
        contradictory_differs = (
            results[3].output != results[1].output
        )
        mlflow.log_metrics({
            "gate/adapter_has_any_effect": int(task_differs),
            "gate/enriched_trajectory_differs": int(
                enriched_differs,
            ),
            "gate/contradictory_shows_contamination": int(
                contradictory_differs,
            ),
        })

        header = (
            "| Condition | Scaling | Differs "
            "| Tokens | Time (s) | Output (first 120 chars) |"
        )
        separator = "|---|---|---|---|---|---|"
        rows = [header, separator]
        for r in results:
            differs = r.output != baseline_output
            preview = (
                r.output[:120].replace("\n", " ").replace("|", "\\|")
            )
            rows.append(
                f"| {r.condition} | {r.adapter_scaling} "
                f"| {differs} "
                f"| {r.tokens_used} | {r.elapsed_s} "
                f"| {preview} |"
            )
        mlflow.log_text(
            "\n".join(rows), "probe/comparison_table.md",
        )

        all_outputs = {r.condition: r.output for r in results}
        mlflow.log_text(
            json.dumps(all_outputs, indent=2),
            "probe/all_outputs.json",
        )

        logger.info("=== GATE RESULTS ===")
        logger.info(
            "Adapter has any effect: %s", task_differs,
        )
        logger.info(
            "Enriched trajectory differs: %s", enriched_differs,
        )
        logger.info(
            "Contradictory shows contamination: %s",
            contradictory_differs,
        )
        logger.info("MLflow run ID: %s", run.info.run_id)


def main() -> None:
    from rune.config import load_config  # noqa: PLC0415

    parser = argparse.ArgumentParser(
        description="Adapter retrievability probe",
    )
    parser.add_argument(
        "--config", type=Path, default=_DEFAULT_CONFIG,
        help=f"Config YAML path (default: {_DEFAULT_CONFIG})",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if not cfg.checkpoint_path:
        parser.error("Config must set checkpoint_path")

    logger.info(
        "Config: model_id=%s, checkpoint=%s, adapter_scaling=%.3f",
        cfg.model_id, cfg.checkpoint_path, cfg.adapter_scaling,
    )

    results = run_probe(cfg)

    print("\n=== OUTPUTS ===")
    for r in results:
        print(
            f"\n--- {r.condition} (scale={r.adapter_scaling}, "
            f"{r.tokens_used} tokens, {r.elapsed_s}s) ---"
        )
        print(r.output[:200])

    log_to_mlflow(results, cfg)


if __name__ == "__main__":
    main()
