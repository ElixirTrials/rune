"""Statistical tests for paper Table 2 and Gates 1–3.

Provides:
- Paired McNemar test with continuity correction
- 95% Wilson-score confidence intervals
- Bonferroni correction for multiple comparisons

Usage:
    uv run python scripts/paper/statistical_tests.py \
        --results-a evaluation_results/condition_v.json \
        --results-b evaluation_results/condition_i.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def mcnemar_test(
    paired_results: list[tuple[bool, bool]],
    continuity: bool = True,
) -> dict[str, Any]:
    """Paired McNemar test with optional continuity correction.

    Args:
        paired_results: List of (model_a_correct, model_b_correct) per problem.
        continuity: Apply Edwards' continuity correction (default True).

    Returns:
        Dict with chi2 statistic, p_value, n_discordant_ab, n_discordant_ba.
    """
    from scipy.stats import chi2 as chi2_dist

    n_ab = sum(1 for a, b in paired_results if a and not b)
    n_ba = sum(1 for a, b in paired_results if not a and b)

    n_discordant = n_ab + n_ba

    if n_discordant == 0:
        return {
            "chi2": 0.0,
            "p_value": 1.0,
            "n_discordant_ab": n_ab,
            "n_discordant_ba": n_ba,
        }

    if continuity:
        chi2 = (abs(n_ab - n_ba) - 1) ** 2 / (n_ab + n_ba)
    else:
        chi2 = (n_ab - n_ba) ** 2 / (n_ab + n_ba)

    p_value = 1.0 - chi2_dist.cdf(chi2, df=1)

    return {
        "chi2": chi2,
        "p_value": p_value,
        "n_discordant_ab": n_ab,
        "n_discordant_ba": n_ba,
    }


def wilson_score_ci(
    n_total: int,
    n_success: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion.

    Args:
        n_total: Total number of trials.
        n_success: Number of successes.
        confidence: Confidence level (default 0.95).

    Returns:
        (lower, upper) bounds of the CI.
    """
    from scipy.stats import norm

    if n_total == 0:
        return (0.0, 1.0)

    z = norm.ppf(1 - (1 - confidence) / 2)
    p_hat = n_success / n_total

    denominator = 1 + z**2 / n_total
    center = (p_hat + z**2 / (2 * n_total)) / denominator
    spread = z * math.sqrt(p_hat * (1 - p_hat) / n_total + z**2 / (4 * n_total**2)) / denominator

    lower = max(0.0, center - spread)
    upper = min(1.0, center + spread)
    return (lower, upper)


def bonferroni_correct(
    p_values: list[float],
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Apply Bonferroni correction to a list of p-values.

    Args:
        p_values: Raw p-values from multiple comparisons.
        alpha: Family-wise error rate.

    Returns:
        Dict with effective_alpha, adjusted_p_values, and significance list.
    """
    m = len(p_values)
    effective_alpha = alpha / m if m > 0 else alpha
    significant = [p < effective_alpha for p in p_values]

    return {
        "effective_alpha": effective_alpha,
        "n_comparisons": m,
        "significant": significant,
        "adjusted_p_values": [min(p * m, 1.0) for p in p_values],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="McNemar + Wilson CI for paper")
    parser.add_argument("--results-a", type=Path, required=True)
    parser.add_argument("--results-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    with args.results_a.open() as f:
        results_a = json.load(f)
    with args.results_b.open() as f:
        results_b = json.load(f)

    verdicts_a = {v["problem_id"]: v["passed"] for v in results_a["verdicts"]}
    verdicts_b = {v["problem_id"]: v["passed"] for v in results_b["verdicts"]}

    common_ids = sorted(set(verdicts_a) & set(verdicts_b))
    paired = [(verdicts_a[pid], verdicts_b[pid]) for pid in common_ids]

    mcnemar_result = mcnemar_test(paired)
    n_a = sum(1 for pid in common_ids if verdicts_a[pid])
    n_b = sum(1 for pid in common_ids if verdicts_b[pid])
    ci_a = wilson_score_ci(len(common_ids), n_a)
    ci_b = wilson_score_ci(len(common_ids), n_b)

    report = {
        "mcnemar": mcnemar_result,
        "ci_a": {"lower": ci_a[0], "upper": ci_a[1], "pass_rate": n_a / len(common_ids)},
        "ci_b": {"lower": ci_b[0], "upper": ci_b[1], "pass_rate": n_b / len(common_ids)},
        "n_problems": len(common_ids),
    }

    output = json.dumps(report, indent=2)
    print(output)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)


if __name__ == "__main__":
    main()
