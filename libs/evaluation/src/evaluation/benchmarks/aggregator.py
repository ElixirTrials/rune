"""Pass@1 aggregation from a list of PassVerdict instances.

Pure function; no I/O, no GPU dependencies. Suitable for direct import
in CPU-only environments and in test fixtures.
"""

from __future__ import annotations

from evaluation.benchmarks.protocol import PassVerdict


def pass_at_1_from_verdicts(verdicts: list[PassVerdict]) -> float:
    """Compute Pass@1 from a list of per-problem verdicts.

    Pass@1 is the fraction of problems where the model's single generation
    passed all tests. Timed-out problems count as failures.

    Args:
        verdicts: List of PassVerdict instances. May be empty.

    Returns:
        Float in [0.0, 1.0]. Returns 0.0 for an empty list.

    Example:
        >>> v = [PassVerdict("p1", True, "", None, False),
        ...      PassVerdict("p2", False, "", "err", False)]
        >>> pass_at_1_from_verdicts(v)
        0.5
    """
    if not verdicts:
        return 0.0
    n_passed = sum(1 for v in verdicts if v.passed)
    return n_passed / len(verdicts)
