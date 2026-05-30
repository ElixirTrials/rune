"""Generate benchmark tasks from the MBPP dataset.

Loads ``google-research-datasets/mbpp`` (``sanitized`` config, ``test`` split)
and converts each row into a :class:`~rune.bench.runner.BenchTask`. The prompt
follows the standard MBPP eval protocol (description + a doctest hint from the
first assertion) so the model can infer the expected function name/signature.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from rune.bench.runner import BenchTask

logger = logging.getLogger(__name__)

# Expected entry-point function name = the symbol the first assertion calls.
_FUNC_NAME_RE = re.compile(r"assert\s+(\w+)\s*\(")


def _row_to_task(row: dict[str, Any]) -> BenchTask:
    """Convert one raw MBPP row into a BenchTask (pure; no I/O)."""
    test_list = row.get("test_list") or []
    test_code = (
        test_list
        if isinstance(test_list, str)
        else "\n".join(str(t) for t in test_list)
    )

    # Fold imports/setup into test_code: BenchTask has no separate setup field
    # and the asserts often need them.
    imports = row.get("test_imports") or []
    setup_lines = (
        imports if isinstance(imports, str) else "\n".join(str(i) for i in imports)
    )
    setup = "\n".join(
        s for s in (setup_lines, str(row.get("test_setup_code", "") or "")) if s.strip()
    )
    full_test = f"{setup}\n{test_code}".strip() if setup else test_code

    description = str(row.get("text") or row.get("prompt") or "")
    hint = test_code.split("\n", 1)[0] if test_code else ""
    prompt = f'"""\n{description}\n\n>>> {hint}\n"""\n'

    match = _FUNC_NAME_RE.search(test_code)
    entry_point = match.group(1) if match else "solution"

    return BenchTask(
        task_id=f"mbpp/{row.get('task_id', '')}",
        description=prompt,
        test_code=full_test,
        entry_point=entry_point,
    )


def load_mbpp_tasks(
    *,
    ids: set[str] | None = None,
    limit: int | None = None,
) -> list[BenchTask]:
    """Build BenchTasks from MBPP (sanitized/test).

    Args:
        ids: If given, keep only tasks whose ``task_id`` (e.g. ``"mbpp/42"``)
            is in this set.
        limit: If given, keep at most this many tasks (after id-filtering),
            in ascending problem-number order.

    Returns:
        BenchTasks in ascending problem-number order.
    """
    import datasets as hf_datasets  # noqa: PLC0415

    rows = list(
        hf_datasets.load_dataset(
            "google-research-datasets/mbpp", "sanitized", split="test"
        )
    )
    tasks = [_row_to_task(r) for r in rows]
    if ids is not None:
        tasks = [t for t in tasks if t.task_id in ids]
    tasks.sort(key=lambda t: int(t.task_id.split("/")[-1] or 0))
    if limit is not None:
        tasks = tasks[:limit]
    logger.info("load_mbpp_tasks: %d tasks", len(tasks))
    return tasks
