"""Exact McNemar significance for paired pass/fail benchmark outcomes (issue #52).

For paired binary results, significance is governed by the discordant pairs, not
by N. With a strict superset (no regressions, ``base_only == 0``) the one-sided
exact McNemar p collapses to ``0.5 ** gains`` — so significance is reachable at a
fixed N. No scipy dependency: the exact binomial tail uses stdlib ``math.comb``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb


def mcnemar_exact(base_only: int, c3_only: int) -> tuple[float, float]:
    """Exact McNemar p-values from the discordant counts.

    ``base_only`` = regressions (base passed, c3 failed); ``c3_only`` = gains.
    Returns ``(p_one_sided, p_two_sided)`` where the one-sided alternative is
    "c3 better" (more gains than regressions). Under H0 each discordant is a gain
    with probability 0.5.
    """
    b, c = base_only, c3_only
    n = b + c
    if n == 0:
        return 1.0, 1.0
    half_n = 0.5**n
    p_ge_c = sum(comb(n, i) for i in range(c, n + 1)) * half_n  # P(X >= c)
    p_le_c = sum(comb(n, i) for i in range(0, c + 1)) * half_n  # P(X <= c)
    p_one_sided = p_ge_c
    p_two_sided = min(1.0, 2.0 * min(p_ge_c, p_le_c))
    return p_one_sided, p_two_sided


@dataclass(frozen=True)
class PairedResult:
    n: int
    both_pass: int
    both_fail: int
    base_only: int  # regressions
    c3_only: int  # gains
    strict_superset: bool
    p_one_sided: float
    p_two_sided: float


def paired_compare(base: dict[str, bool], c3: dict[str, bool]) -> PairedResult:
    """Compare two per-task pass/fail maps on their shared task ids."""
    keys = set(base) & set(c3)
    both_pass = sum(1 for k in keys if base[k] and c3[k])
    both_fail = sum(1 for k in keys if not base[k] and not c3[k])
    base_only = sum(1 for k in keys if base[k] and not c3[k])
    c3_only = sum(1 for k in keys if not base[k] and c3[k])
    p_one, p_two = mcnemar_exact(base_only, c3_only)
    return PairedResult(
        n=len(keys),
        both_pass=both_pass,
        both_fail=both_fail,
        base_only=base_only,
        c3_only=c3_only,
        strict_superset=(base_only == 0),
        p_one_sided=p_one,
        p_two_sided=p_two,
    )


def format_report(r: PairedResult, alpha: float = 0.05) -> str:
    """One-line headline + a transparency line (exact counts and both p-values)."""
    verdict = "SIGNIFICANT" if r.p_one_sided <= alpha else "n.s."
    regr_str = f"{r.base_only} regression(s)" if r.base_only else "strict superset"
    return (
        f"base+{r.c3_only} / -{r.base_only} (n={r.n}, {regr_str}); "
        f"McNemar one-sided p={r.p_one_sided:.4f} [{verdict} @ alpha={alpha}]\n"
        f"  transparency: both_pass={r.both_pass} both_fail={r.both_fail} "
        f"base_only={r.base_only} c3_only={r.c3_only} two-sided p={r.p_two_sided:.4f}"
    )
