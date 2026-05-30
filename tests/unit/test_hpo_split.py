"""Unit tests for the HPO tuning/validation split."""

from __future__ import annotations

import random

from rune.bench.hpo import split_tasks
from rune.bench.runner import BenchTask


def _tasks(n: int) -> list[BenchTask]:
    return [
        BenchTask(task_id=f"mbpp/{i}", description="", test_code="") for i in range(n)
    ]


class TestSplitTasks:
    def test_fraction_and_disjoint(self) -> None:
        tasks = _tasks(100)
        tun, val = split_tasks(tasks, seed=42, tuning_fraction=0.70)
        assert len(tun) == 70
        assert len(val) == 30
        ids_t = {t.task_id for t in tun}
        ids_v = {t.task_id for t in val}
        assert ids_t.isdisjoint(ids_v)  # held-out, never seen during tuning
        assert ids_t | ids_v == {t.task_id for t in tasks}

    def test_deterministic_and_order_independent(self) -> None:
        tasks = _tasks(50)
        shuffled = list(tasks)
        random.Random(7).shuffle(shuffled)
        a = [t.task_id for t in split_tasks(tasks, seed=42)[0]]
        b = [t.task_id for t in split_tasks(shuffled, seed=42)[0]]
        assert a == b  # split depends only on (ids, seed), not input order

    def test_seed_changes_split(self) -> None:
        tasks = _tasks(50)
        a = {t.task_id for t in split_tasks(tasks, seed=1)[0]}
        b = {t.task_id for t in split_tasks(tasks, seed=2)[0]}
        assert a != b
