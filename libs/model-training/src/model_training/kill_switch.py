"""Training kill-switch: halt hypernetwork training on Pass@1 regression.

At a configurable step cadence, the training loop re-runs a small benchmark
(default: HumanEval @ 10 problems) and compares the current Pass@1 to a
baseline captured on the first evaluation (step 0 semantic). When
``current < baseline - delta`` (default delta = 0.05 = 5 pts absolute), the
kill-switch fires and the loop halts.

All heavy imports (``evaluation.benchmarks``) are deferred into the closure
returned by :func:`build_benchmark_evaluate_fn` so this module stays
importable without the evaluation package on the path.

Usage (in a training loop):

    from model_training.kill_switch import (
        KillSwitchConfig,
        KillSwitchState,
        maybe_run_kill_switch,
    )

    ks_config = KillSwitchConfig(enabled=True, step_cadence=100)
    ks_state = KillSwitchState()
    for step in range(1, num_steps + 1):
        ...  # optimizer step, etc.
        if maybe_run_kill_switch(
            step=step,
            config=ks_config,
            state=ks_state,
            evaluate_fn=my_eval_fn,
        ):
            logger.error("Kill-switch triggered; halting training")
            break
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)

__all__ = [
    "KillSwitchConfig",
    "KillSwitchState",
    "build_benchmark_evaluate_fn",
    "evaluate_and_check",
    "maybe_run_kill_switch",
    "regression_detected",
    "should_evaluate",
    "update_and_check",
]


@dataclass
class KillSwitchConfig:
    """Kill-switch configuration.

    Attributes:
        enabled: Master toggle. When False, :func:`maybe_run_kill_switch` is a
            no-op regardless of other fields.
        step_cadence: Evaluate Pass@1 every ``step_cadence`` training steps.
        benchmark_id: Benchmark to evaluate against (default: HumanEval).
        max_samples: Cap on problems evaluated per call (kept small to amortise
            the per-step cost).
        delta: Absolute Pass@1 regression threshold. When current pass_at_1 is
            strictly less than ``baseline - delta`` the switch fires.
    """

    enabled: bool = False
    step_cadence: int = 100
    benchmark_id: str = "humaneval"
    max_samples: int = 10
    delta: float = 0.05


@dataclass
class KillSwitchState:
    """Per-run kill-switch state, mutated in place by :func:`update_and_check`.

    Attributes:
        baseline: Pass@1 captured on the first evaluation. ``None`` until then.
        triggered: True once a regression has fired the switch.
        last_pass_at_1: Pass@1 from the most recent evaluation.
        evaluations: Count of evaluations performed so far.
    """

    baseline: float | None = None
    triggered: bool = False
    last_pass_at_1: float | None = None
    evaluations: int = 0


def should_evaluate(step: int, cadence: int) -> bool:
    """Return True when ``step`` is a positive multiple of ``cadence``.

    Args:
        step: Current training step (1-based).
        cadence: Evaluation cadence. Non-positive values disable evaluation.
    """
    if cadence <= 0 or step <= 0:
        return False
    return step % cadence == 0


def regression_detected(current: float, baseline: float, delta: float) -> bool:
    """Return True when ``current < baseline - delta`` (strict).

    Boundary ``current == baseline - delta`` is NOT treated as a regression.
    """
    return current < baseline - delta


def update_and_check(
    state: KillSwitchState, current: float, delta: float
) -> bool:
    """Update ``state`` with a new Pass@1 reading; return True to halt.

    The first call captures ``current`` as the baseline and never halts.
    Subsequent calls compare to the baseline via :func:`regression_detected`.

    Args:
        state: Mutable kill-switch state.
        current: Pass@1 from the latest evaluation.
        delta: Absolute regression threshold.

    Returns:
        True when training should halt; False otherwise.
    """
    state.last_pass_at_1 = current
    state.evaluations += 1
    if state.baseline is None:
        state.baseline = current
        logger.info(
            "Kill-switch baseline captured: pass_at_1=%.3f", current
        )
        return False
    if regression_detected(current, state.baseline, delta):
        state.triggered = True
        logger.error(
            "Kill-switch TRIGGERED: pass_at_1=%.3f < baseline=%.3f - delta=%.3f",
            current,
            state.baseline,
            delta,
        )
        return True
    return False


def evaluate_and_check(
    evaluate_fn: Callable[[], float],
    state: KillSwitchState,
    delta: float,
) -> tuple[float, bool]:
    """Run ``evaluate_fn``, update state, return (pass_at_1, should_halt)."""
    current = evaluate_fn()
    return current, update_and_check(state, current, delta)


def maybe_run_kill_switch(
    *,
    step: int,
    config: KillSwitchConfig,
    state: KillSwitchState,
    evaluate_fn: Callable[[], float],
) -> bool:
    """Step-level hook for a training loop.

    When the kill-switch is disabled or ``step`` is not a cadence multiple,
    ``evaluate_fn`` is NOT invoked and the function returns False. Otherwise
    it runs the evaluator and updates state.

    Args:
        step: Current (1-based) training step.
        config: Kill-switch configuration.
        state: Mutable kill-switch state.
        evaluate_fn: Zero-arg callable returning the current Pass@1.

    Returns:
        True when training should halt.
    """
    if not config.enabled:
        return False
    if not should_evaluate(step, config.step_cadence):
        return False
    _, halt = evaluate_and_check(evaluate_fn, state, config.delta)
    return halt


def build_benchmark_evaluate_fn(
    *,
    base_model: str,
    benchmark_id: str,
    max_samples: int,
    provider: Any,
    registry: Any,
    adapter_ids: list[str] | None = None,
) -> Callable[[], float]:
    """Return a zero-arg closure that calls ``run_benchmark`` once.

    The evaluation import is deferred into the closure body, so calling
    :func:`build_benchmark_evaluate_fn` does not load the evaluation package.

    Args:
        base_model: HF repo id or local path of the base model.
        benchmark_id: Benchmark registry key (e.g. ``"humaneval"``).
        max_samples: Per-benchmark problem cap.
        provider: :class:`inference.provider.InferenceProvider` instance.
        registry: :class:`adapter_registry.registry.AdapterRegistry` (or
            compatible duck-type with ``retrieve_by_id``).
        adapter_ids: Ordered adapter stack; empty list = base only.

    Returns:
        A callable that returns ``BenchmarkResult.pass_at_1`` when invoked.
    """
    ids = list(adapter_ids) if adapter_ids else []

    def _evaluate() -> float:
        # evaluation.benchmarks.run_benchmark is lazily exposed via
        # module __getattr__, which mypy sees as "object" — cast to Any
        # so the call is typed correctly without a heavy import.
        from typing import Any as _Any  # noqa: PLC0415

        from evaluation.benchmarks import run_benchmark as _rb  # noqa: PLC0415
        from evaluation.benchmarks.adapter_stack import (  # noqa: PLC0415
            load_adapter_stack,
        )

        run_benchmark: _Any = _rb
        stack = load_adapter_stack(
            base_model=base_model,
            adapter_ids=ids,
            provider=provider,
            registry=registry,
        )
        result = run_benchmark(
            adapter_stack=stack,
            benchmark_id=benchmark_id,
            max_samples=max_samples,
        )
        return float(result.pass_at_1)

    return _evaluate
