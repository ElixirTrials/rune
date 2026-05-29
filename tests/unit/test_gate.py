from rune.training.gate import evaluate_gate


class TestEvaluateGate:
    def test_passes_with_sufficient_improvements(self) -> None:
        baseline = {"a": 10.0, "b": 10.0, "c": 10.0, "d": 10.0, "e": 10.0, "f": 10.0}
        new = {"a": 13.0, "b": 14.0, "c": 12.5, "d": 15.0, "e": 10.5, "f": 10.0}
        result = evaluate_gate(baseline, new)
        assert result.passed is True
        assert result.passing_benchmarks >= 4

    def test_fails_with_regression(self) -> None:
        baseline = {"a": 10.0, "b": 10.0, "c": 10.0, "d": 10.0}
        new = {"a": 13.0, "b": 14.0, "c": 12.5, "d": 7.0}  # d regresses
        result = evaluate_gate(baseline, new)
        assert result.passed is False
        assert "d" in result.regressions

    def test_fails_with_too_few_improvements(self) -> None:
        baseline = {"a": 10.0, "b": 10.0, "c": 10.0, "d": 10.0}
        new = {"a": 13.0, "b": 10.5, "c": 10.0, "d": 10.0}  # only 1 improves enough
        result = evaluate_gate(baseline, new)
        assert result.passed is False
