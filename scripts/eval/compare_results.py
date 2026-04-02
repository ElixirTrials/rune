"""Compare evaluation results between base and LoRA models.

Computes deltas, per-task flip analysis, and leaderboard positioning.
Generates a markdown report.

Usage:
    uv run scripts/eval/compare_results.py evaluation_results/20260402_120000/results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from config import PUBLISHED_SCORES


def extract_pass_at_1(eval_result: dict[str, Any]) -> float | None:
    """Extract pass@1 score from an EvalPlus result dict."""
    # EvalPlus result format varies; try common keys
    if "error" in eval_result:
        return None

    # Try direct pass@1 key
    if "pass@1" in eval_result:
        return float(eval_result["pass@1"])

    # Try nested eval format
    if "eval" in eval_result:
        eval_data = eval_result["eval"]
        if isinstance(eval_data, dict):
            # Count passing tasks
            passed = sum(
                1
                for v in eval_data.values()
                if isinstance(v, list)
                and len(v) > 0
                and v[0].get("base_status") == "pass"
            )
            total = len(eval_data)
            if total > 0:
                return (passed / total) * 100.0

    return None


def compute_deltas(
    base_results: dict[str, Any],
    lora_results: dict[str, Any],
) -> list[dict[str, Any]]:
    """Compute pass@1 deltas between base and LoRA for each benchmark."""
    deltas = []

    for key in base_results:
        base_score = extract_pass_at_1(base_results[key])
        lora_score = extract_pass_at_1(lora_results.get(key, {}))

        if base_score is not None and lora_score is not None:
            delta = lora_score - base_score
            deltas.append(
                {
                    "benchmark": key,
                    "base_pass@1": base_score,
                    "lora_pass@1": lora_score,
                    "delta": delta,
                    "delta_pct": ((delta / base_score * 100) if base_score > 0 else 0),
                }
            )

    return deltas


def format_leaderboard(
    model_name: str,
    lora_results: dict[str, Any],
) -> list[dict[str, Any]]:
    """Position the model against published leaderboard scores."""
    rows = []

    # Collect our scores
    our_scores: dict[str, float] = {}
    for key, result in lora_results.items():
        score = extract_pass_at_1(result)
        if score is not None:
            # Normalize key to match PUBLISHED_SCORES format
            benchmark = key.rsplit("_pass@", 1)[0]
            if benchmark == "humaneval":
                benchmark = "humaneval+"
            elif benchmark == "mbpp":
                benchmark = "mbpp+"
            our_scores[benchmark] = score

    # Build comparison table for each benchmark we have scores for
    for benchmark, our_score in our_scores.items():
        entries: list[tuple[str, float]] = [(model_name, our_score)]

        for other_model, scores in PUBLISHED_SCORES.items():
            if benchmark in scores:
                entries.append((other_model, scores[benchmark]))

        entries.sort(key=lambda x: x[1], reverse=True)
        rank = next(i + 1 for i, (name, _) in enumerate(entries) if name == model_name)

        rows.append(
            {
                "benchmark": benchmark,
                "score": our_score,
                "rank": rank,
                "total": len(entries),
                "entries": entries,
            }
        )

    return rows


def generate_report(
    all_results: dict[str, Any],
    output_path: Path,
) -> None:
    """Generate a markdown comparison report."""
    config = all_results.get("config", {})
    base_results = all_results.get("base", {})
    lora_results = all_results.get("lora", {})

    lines = [
        "# Coding Benchmark Evaluation Report",
        "",
        "## Configuration",
        "",
        f"- **Model:** {config.get('model_id', 'unknown')}",
        f"- **Adapter:** {config.get('adapter_path', 'N/A')}",
        f"- **Backend:** {config.get('backend', 'unknown')}",
        f"- **Tier:** {config.get('tier', 'unknown')}",
        "",
    ]

    # Delta table
    if lora_results:
        deltas = compute_deltas(base_results, lora_results)
        if deltas:
            lines.extend(
                [
                    "## Base vs LoRA Comparison",
                    "",
                    "| Benchmark | Base pass@1 | LoRA pass@1 | Delta | Rel. Change |",
                    "|-----------|------------|------------|-------|-------------|",
                ]
            )
            for d in deltas:
                sign = "+" if d["delta"] >= 0 else ""
                lines.append(
                    f"| {d['benchmark']} | {d['base_pass@1']:.1f}% "
                    f"| {d['lora_pass@1']:.1f}% | {sign}{d['delta']:.1f}% "
                    f"| {sign}{d['delta_pct']:.1f}% |"
                )
            lines.append("")

    # Leaderboard positioning
    results_to_rank = lora_results if lora_results else base_results
    model_label = (
        f"{config.get('model_id', 'model')} + LoRA"
        if lora_results
        else config.get("model_id", "model")
    )

    leaderboard = format_leaderboard(model_label, results_to_rank)
    if leaderboard:
        lines.extend(
            [
                "## Leaderboard Positioning",
                "",
            ]
        )
        for entry in leaderboard:
            lines.append(f"### {entry['benchmark']}")
            lines.append("")
            lines.append(f"**Rank: {entry['rank']}/{entry['total']}**")
            lines.append("")
            lines.append("| Model | Score |")
            lines.append("|-------|-------|")
            for name, score in entry["entries"]:
                marker = " **<--**" if name == model_label else ""
                lines.append(f"| {name} | {score:.1f}%{marker} |")
            lines.append("")

    report = "\n".join(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)


def main() -> None:
    """Entry point for standalone comparison."""
    parser = argparse.ArgumentParser(
        description="Compare base vs LoRA evaluation results"
    )
    parser.add_argument(
        "results_file",
        help="Path to results.json from run_benchmarks.py",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for report (default: alongside results.json)",
    )
    args = parser.parse_args()

    results_path = Path(args.results_file)
    with open(results_path) as f:
        all_results = json.load(f)

    output_path = (
        Path(args.output) if args.output else results_path.parent / "report.md"
    )

    generate_report(all_results, output_path)
    print(f"Report written to {output_path}")


if __name__ == "__main__":
    main()
