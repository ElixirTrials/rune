"""Unit tests for the training kill-switch.

Pure-Python tests — no GPU, no model loading. The benchmark evaluation is
always mocked via the injected ``evaluate_fn`` so tests stay fast.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


class TestShouldEvaluate:
    """``should_evaluate(step, cadence)`` picks out cadence-multiple steps."""

    def test_step_zero_never_evaluates(self) -> None:
        from model_training.kill_switch import should_evaluate

        assert should_evaluate(0, 10) is False

    def test_cadence_multiple_evaluates(self) -> None:
        from model_training.kill_switch import should_evaluate

        assert should_evaluate(10, 10) is True
        assert should_evaluate(20, 10) is True
        assert should_evaluate(100, 10) is True

    def test_non_multiple_does_not_evaluate(self) -> None:
        from model_training.kill_switch import should_evaluate

        assert should_evaluate(5, 10) is False
        assert should_evaluate(11, 10) is False

    def test_cadence_zero_disables(self) -> None:
        from model_training.kill_switch import should_evaluate

        assert should_evaluate(10, 0) is False
        assert should_evaluate(10, -5) is False


class TestRegressionDetected:
    """``regression_detected(current, baseline, delta)``."""

    def test_no_regression_when_equal(self) -> None:
        from model_training.kill_switch import regression_detected

        assert regression_detected(0.50, 0.50, 0.03) is False

    def test_small_drop_within_delta(self) -> None:
        from model_training.kill_switch import regression_detected

        # 0.48 >= 0.50 - 0.03 = 0.47
        assert regression_detected(0.48, 0.50, 0.03) is False

    def test_drop_exceeding_delta(self) -> None:
        from model_training.kill_switch import regression_detected

        # 0.45 < 0.50 - 0.03 = 0.47
        assert regression_detected(0.45, 0.50, 0.03) is True

    def test_improvement_not_regression(self) -> None:
        from model_training.kill_switch import regression_detected

        assert regression_detected(0.55, 0.50, 0.03) is False

    def test_boundary_exactly_at_threshold(self) -> None:
        from model_training.kill_switch import regression_detected

        # current == baseline - delta → NOT a regression (strict less-than)
        assert regression_detected(0.47, 0.50, 0.03) is False


class TestUpdateAndCheck:
    """``update_and_check`` captures the baseline on first call and compares after."""

    def test_first_call_captures_baseline(self) -> None:
        from model_training.kill_switch import KillSwitchState, update_and_check

        state = KillSwitchState()
        halt = update_and_check(state, 0.5, 0.03)
        assert halt is False
        assert state.baseline == 0.5
        assert state.last_pass_at_1 == 0.5
        assert state.triggered is False

    def test_regression_triggers_halt(self) -> None:
        from model_training.kill_switch import KillSwitchState, update_and_check

        state = KillSwitchState()
        update_and_check(state, 0.50, 0.03)  # baseline
        halt = update_and_check(state, 0.40, 0.03)
        assert halt is True
        assert state.triggered is True
        assert state.last_pass_at_1 == 0.40

    def test_stable_no_halt(self) -> None:
        from model_training.kill_switch import KillSwitchState, update_and_check

        state = KillSwitchState()
        update_and_check(state, 0.50, 0.03)
        halt = update_and_check(state, 0.48, 0.03)
        assert halt is False
        assert state.triggered is False

    def test_evaluations_counter_increments(self) -> None:
        from model_training.kill_switch import KillSwitchState, update_and_check

        state = KillSwitchState()
        assert state.evaluations == 0
        update_and_check(state, 0.5, 0.03)
        assert state.evaluations == 1
        update_and_check(state, 0.4, 0.03)
        assert state.evaluations == 2


class TestEvaluateAndCheck:
    """``evaluate_and_check`` composes the evaluator + state update."""

    def test_first_call_no_halt(self) -> None:
        from model_training.kill_switch import KillSwitchState, evaluate_and_check

        fn = MagicMock(return_value=0.5)
        state = KillSwitchState()
        pass_at_1, halt = evaluate_and_check(fn, state, 0.03)
        assert pass_at_1 == 0.5
        assert halt is False
        assert fn.call_count == 1
        assert state.baseline == 0.5

    def test_regressing_sequence(self) -> None:
        from model_training.kill_switch import KillSwitchState, evaluate_and_check

        # baseline 0.60; delta 0.10 → regression when current < 0.50
        fn = MagicMock(side_effect=[0.60, 0.55, 0.40])
        state = KillSwitchState()
        _, h1 = evaluate_and_check(fn, state, 0.10)
        _, h2 = evaluate_and_check(fn, state, 0.10)
        _, h3 = evaluate_and_check(fn, state, 0.10)
        assert h1 is False
        assert h2 is False  # 0.55 >= 0.50
        assert h3 is True   # 0.40 < 0.50


class TestMaybeRunKillSwitch:
    """``maybe_run_kill_switch`` — full step-level hook for the training loop."""

    def test_disabled_config_never_runs_fn(self) -> None:
        from model_training.kill_switch import (
            KillSwitchConfig,
            KillSwitchState,
            maybe_run_kill_switch,
        )

        fn = MagicMock(return_value=0.5)
        config = KillSwitchConfig(enabled=False)
        state = KillSwitchState()
        halt = maybe_run_kill_switch(
            step=100, config=config, state=state, evaluate_fn=fn
        )
        assert halt is False
        assert fn.call_count == 0

    def test_non_cadence_step_skips_fn(self) -> None:
        from model_training.kill_switch import (
            KillSwitchConfig,
            KillSwitchState,
            maybe_run_kill_switch,
        )

        fn = MagicMock(return_value=0.5)
        config = KillSwitchConfig(enabled=True, step_cadence=100)
        state = KillSwitchState()
        halt = maybe_run_kill_switch(
            step=5, config=config, state=state, evaluate_fn=fn
        )
        assert halt is False
        assert fn.call_count == 0

    def test_cadence_step_halts_on_regression(self) -> None:
        from model_training.kill_switch import (
            KillSwitchConfig,
            KillSwitchState,
            maybe_run_kill_switch,
        )

        fn = MagicMock(side_effect=[0.6, 0.40])
        config = KillSwitchConfig(enabled=True, step_cadence=10, delta=0.05)
        state = KillSwitchState()
        h1 = maybe_run_kill_switch(
            step=10, config=config, state=state, evaluate_fn=fn
        )
        h2 = maybe_run_kill_switch(
            step=20, config=config, state=state, evaluate_fn=fn
        )
        assert h1 is False
        assert h2 is True
        assert state.triggered is True

    def test_loop_simulation_halts_on_decreasing_pass_at_1(self) -> None:
        """Mocked run_benchmark returns decreasing Pass@1; the loop stops."""
        from model_training.kill_switch import (
            KillSwitchConfig,
            KillSwitchState,
            maybe_run_kill_switch,
        )

        # Decreasing Pass@1: 0.6 → 0.55 → 0.40 → ...
        fn = MagicMock(side_effect=[0.6, 0.55, 0.40, 0.35, 0.30])
        config = KillSwitchConfig(enabled=True, step_cadence=10, delta=0.05)
        state = KillSwitchState()
        stopped_at: int | None = None
        for step in range(1, 101):
            halt = maybe_run_kill_switch(
                step=step, config=config, state=state, evaluate_fn=fn
            )
            if halt:
                stopped_at = step
                break
        assert stopped_at == 30, f"Expected halt at step 30, got {stopped_at}"
        assert state.triggered is True
        assert state.baseline == 0.6
        assert state.last_pass_at_1 == 0.40
        # Called at steps 10, 20, 30 only
        assert fn.call_count == 3

    def test_loop_does_not_halt_when_stable(self) -> None:
        from model_training.kill_switch import (
            KillSwitchConfig,
            KillSwitchState,
            maybe_run_kill_switch,
        )

        # Stable Pass@1 around baseline → never halt
        fn = MagicMock(return_value=0.60)
        config = KillSwitchConfig(enabled=True, step_cadence=10, delta=0.05)
        state = KillSwitchState()
        for step in range(1, 101):
            halt = maybe_run_kill_switch(
                step=step, config=config, state=state, evaluate_fn=fn
            )
            assert halt is False
        assert state.triggered is False
        assert state.baseline == 0.60
        assert fn.call_count == 10  # steps 10, 20, ..., 100


class TestBuildBenchmarkEvaluateFn:
    """``build_benchmark_evaluate_fn`` wraps ``run_benchmark`` into a closure."""

    def test_build_closure_calls_run_benchmark(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The returned closure invokes run_benchmark and returns pass_at_1."""
        from model_training.kill_switch import build_benchmark_evaluate_fn

        fake_result = MagicMock(pass_at_1=0.73)
        mock_run_benchmark = MagicMock(return_value=fake_result)
        mock_load_stack = MagicMock(return_value="stack-stub")

        # Patch the evaluation imports so the closure can be exercised
        # without the real benchmark adapters.
        monkeypatch.setitem(
            __import__("sys").modules,
            "evaluation.benchmarks",
            MagicMock(run_benchmark=mock_run_benchmark),
        )
        monkeypatch.setitem(
            __import__("sys").modules,
            "evaluation.benchmarks.adapter_stack",
            MagicMock(load_adapter_stack=mock_load_stack),
        )

        fn = build_benchmark_evaluate_fn(
            base_model="Qwen/Qwen3.5-9B",
            benchmark_id="humaneval",
            max_samples=5,
            provider="provider-stub",
            registry="registry-stub",
            adapter_ids=["adapter-1"],
        )
        pass_at_1 = fn()
        assert pass_at_1 == 0.73
        assert mock_run_benchmark.call_count == 1
        # run_benchmark got the stack from load_adapter_stack
        kwargs = mock_run_benchmark.call_args.kwargs
        assert kwargs["adapter_stack"] == "stack-stub"
        assert kwargs["benchmark_id"] == "humaneval"
        assert kwargs["max_samples"] == 5

    def test_build_closure_defers_heavy_import_until_invoked(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Constructing the closure must not trigger evaluation imports."""
        import sys

        # Capture current module set; make sure constructing the closure
        # doesn't inject evaluation.benchmarks.* into sys.modules.
        for mod in ("evaluation.benchmarks", "evaluation.benchmarks.adapter_stack"):
            monkeypatch.delitem(sys.modules, mod, raising=False)

        from model_training.kill_switch import build_benchmark_evaluate_fn

        build_benchmark_evaluate_fn(
            base_model="m",
            benchmark_id="humaneval",
            max_samples=1,
            provider=None,
            registry=None,
        )
        # Closure hasn't been called — imports should not have happened yet.
        assert "evaluation.benchmarks" not in sys.modules
