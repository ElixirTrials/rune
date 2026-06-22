from __future__ import annotations

from rune.bench.significance import mcnemar_exact, paired_compare


def test_mcnemar_strict_superset_thresholds() -> None:
    # b=0: one-sided p = 0.5**c, two-sided = 2*0.5**c (capped at 1).
    assert mcnemar_exact(0, 4) == (0.0625, 0.125)
    one5, two5 = mcnemar_exact(0, 5)
    assert round(one5, 5) == 0.03125 and round(two5, 5) == 0.0625
    one6, _ = mcnemar_exact(0, 6)
    assert round(one6, 5) == 0.01562


def test_mcnemar_no_discordants_is_one() -> None:
    assert mcnemar_exact(0, 0) == (1.0, 1.0)


def test_mcnemar_with_regressions() -> None:
    # b=1, c=2: n_d=3; P(X>=2)=4/8=0.5 one-sided; two-sided=min(1,2*0.5)=1.0
    one, two = mcnemar_exact(1, 2)
    assert round(one, 5) == 0.5 and two == 1.0


def test_paired_compare_counts_and_superset() -> None:
    base = {"a": True, "b": True, "c": False, "d": False}
    c3 = {"a": True, "b": False, "c": True, "d": True}  # b=1 (lost b), c=2 (gained c,d)
    r = paired_compare(base, c3)
    assert (r.n, r.both_pass, r.both_fail, r.base_only, r.c3_only) == (4, 1, 0, 1, 2)
    assert r.strict_superset is False

    base2 = {"a": True, "b": False, "c": False}
    c3_2 = {"a": True, "b": True, "c": True}  # b=0, c=2
    r2 = paired_compare(base2, c3_2)
    assert r2.strict_superset is True and r2.c3_only == 2
