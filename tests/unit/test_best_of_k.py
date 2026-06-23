from __future__ import annotations

import asyncio

from rune.engine.graph import oracle_gated_best_of_k


def _run(coro):
    return asyncio.run(coro)


def test_returns_first_oracle_passing_candidate() -> None:
    seq = iter(["c0", "c1_PASS", "c2"])
    calls = {"gen": 0, "eval": 0}

    async def gen():
        calls["gen"] += 1
        return next(seq)

    async def evaluate(r):
        calls["eval"] += 1
        return (r.endswith("PASS"), 1)

    out = _run(oracle_gated_best_of_k(8, gen, evaluate))
    assert out == "c1_PASS"
    assert calls["gen"] == 2  # stopped at the first passing candidate


def test_falls_back_to_best_quality_when_none_pass() -> None:
    results = iter([("a", 1), ("b", 3), ("c", 2)])

    async def gen():
        return next(results)

    async def evaluate(r):
        return (False, r[1])  # never passes; quality is the second element

    out = _run(oracle_gated_best_of_k(3, gen, evaluate))
    assert out == ("b", 3)  # highest quality
