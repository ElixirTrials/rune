"""CPU unit tests for ``scripts/optimization/run_benchmark_hpo.py``.

GPU paths (run_phased_pipeline) are monkeypatched; pure helpers are
tested directly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = REPO_ROOT / "scripts" / "optimization"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_benchmark_hpo import (  # noqa: E402
    _build_parser,
    load_failed_ids,
    split_problems,
    subsample_problems,
)


def _problem(pid: str):
    """Build a minimal Problem stand-in with a problem_id."""
    from evaluation.benchmarks.protocol import Problem

    return Problem(problem_id=pid, prompt="p", test_code="assert True")


class TestSplitProblems:
    def test_split_is_seventy_thirty(self) -> None:
        problems = [_problem(f"mbpp/{i}") for i in range(100)]
        tuning, validation = split_problems(problems, seed=42)
        assert len(tuning) == 70
        assert len(validation) == 30

    def test_split_is_deterministic(self) -> None:
        problems = [_problem(f"mbpp/{i}") for i in range(50)]
        a_tune, a_val = split_problems(problems, seed=42)
        b_tune, b_val = split_problems(problems, seed=42)
        assert [p.problem_id for p in a_tune] == [p.problem_id for p in b_tune]
        assert [p.problem_id for p in a_val] == [p.problem_id for p in b_val]

    def test_split_is_disjoint_and_complete(self) -> None:
        problems = [_problem(f"mbpp/{i}") for i in range(127)]
        tuning, validation = split_problems(problems, seed=42)
        ids = {p.problem_id for p in tuning} | {p.problem_id for p in validation}
        assert ids == {p.problem_id for p in problems}
        assert len(tuning) + len(validation) == 127

    def test_custom_fraction(self) -> None:
        problems = [_problem(f"mbpp/{i}") for i in range(10)]
        tuning, validation = split_problems(problems, seed=1, tuning_fraction=0.5)
        assert len(tuning) == 5
        assert len(validation) == 5


class TestLoadFailedIds:
    def test_loads_json_list(self, tmp_path: Path) -> None:
        path = tmp_path / "failed.json"
        path.write_text(json.dumps(["mbpp/1", "mbpp/2", "mbpp/3"]))
        assert load_failed_ids(path) == {"mbpp/1", "mbpp/2", "mbpp/3"}

    def test_missing_file_raises_clear_error(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Failed-ID file not found"):
            load_failed_ids(tmp_path / "nope.json")


class TestSubsampleProblems:
    def test_returns_requested_count(self) -> None:
        problems = [_problem(f"mbpp/{i}") for i in range(200)]
        subset = subsample_problems(problems, n=127, seed=42)
        assert len(subset) == 127

    def test_is_deterministic(self) -> None:
        problems = [_problem(f"mbpp/{i}") for i in range(200)]
        a = subsample_problems(problems, n=50, seed=42)
        b = subsample_problems(problems, n=50, seed=42)
        assert [p.problem_id for p in a] == [p.problem_id for p in b]

    def test_n_larger_than_pool_returns_all(self) -> None:
        problems = [_problem(f"mbpp/{i}") for i in range(10)]
        subset = subsample_problems(problems, n=99, seed=42)
        assert len(subset) == 10


class TestExtractPhaseMetrics:
    def test_full_result_all_phases(self) -> None:
        from run_benchmark_hpo import extract_phase_metrics

        result = {
            "phases": {
                "decompose": {"subtasks": [{"name": "a"}], "best_score": 1.0},
                "plan": {"plans": {"a": "plan text"}},
                "code": {"iterations": 3},
                "integrate": {"tests_passed": True},
                "repair": {"iterations": 1},
            },
            "evolution": {"sweeps": {"phase1": {}, "final": {}}},
            "adapters": [{"id": "x"}, {"id": "y"}, {"id": "z"}],
        }
        m = extract_phase_metrics(result)
        assert m["phase_decompose_ok"] == 1.0
        assert m["phase_plan_ok"] == 1.0
        assert m["phase_code_attempts"] == 3.0
        assert m["phase_integrate_ok"] == 1.0
        assert m["evolution_sweeps"] == 2.0
        assert m["adapters_generated"] == 3.0

    def test_empty_result_defaults_to_zero(self) -> None:
        from run_benchmark_hpo import extract_phase_metrics

        m = extract_phase_metrics({})
        assert m["phase_decompose_ok"] == 0.0
        assert m["phase_plan_ok"] == 0.0
        assert m["phase_code_attempts"] == 0.0
        assert m["phase_integrate_ok"] == 0.0
        assert m["evolution_sweeps"] == 0.0
        assert m["adapters_generated"] == 0.0

    def test_failed_integrate_is_zero(self) -> None:
        from run_benchmark_hpo import extract_phase_metrics

        m = extract_phase_metrics({"phases": {"integrate": {"tests_passed": False}}})
        assert m["phase_integrate_ok"] == 0.0


class TestProblemVerdict:
    def test_fields_default_phase_metrics_to_empty_dict(self) -> None:
        from run_benchmark_hpo import ProblemVerdict

        v = ProblemVerdict(
            problem_id="mbpp/1",
            passed=True,
            code_attempts=2,
            diagnose_fired=False,
            n_subtasks=1,
            wall_time_s=1.5,
            accumulated_code_len=42,
            error="",
        )
        assert v.phase_metrics == {}
        assert v.passed is True


class TestTrialConfig:
    def test_write_trial_pipeline_config_round_trips_scaling(
        self, tmp_path: Path
    ) -> None:
        from run_benchmark_hpo import write_trial_pipeline_config
        from shared.pipeline_config import load_config

        cfg_path = write_trial_pipeline_config(0.37, tmp_path)
        assert cfg_path.exists()
        loaded = load_config(cfg_path)
        assert loaded.adapter.scaling == pytest.approx(0.37)

    def test_apply_trial_env_sets_all_vars(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import os

        from run_benchmark_hpo import apply_trial_env
        from shared.pipeline_config import load_config

        for var in (
            "RUNE_PIPELINE_CONFIG",
            "RUNE_TEMPERATURE",
            "RUNE_REPETITION_PENALTY",
            "RUNE_MAX_PHASE_ITERATIONS",
            "RUNE_MAX_TOKENS",
        ):
            monkeypatch.delenv(var, raising=False)

        apply_trial_env(
            scaling_factor=0.2,
            temperature=0.45,
            repetition_penalty=1.07,
            max_phase_iterations=4,
            config_dir=tmp_path,
            max_tokens=2048,
        )
        assert os.environ["RUNE_TEMPERATURE"] == "0.45"
        assert os.environ["RUNE_REPETITION_PENALTY"] == "1.07"
        assert os.environ["RUNE_MAX_PHASE_ITERATIONS"] == "4"
        assert os.environ["RUNE_MAX_TOKENS"] == "2048"
        # RUNE_PIPELINE_CONFIG must point at a config carrying the scaling.
        loaded = load_config(Path(os.environ["RUNE_PIPELINE_CONFIG"]))
        assert loaded.adapter.scaling == pytest.approx(0.2)
        assert loaded.generation.max_tokens == 2048


class TestScorePipelineResult:
    def test_passing_code_yields_passed_verdict(self) -> None:
        from evaluation.benchmarks.protocol import Problem
        from run_benchmark_hpo import score_pipeline_result

        problem = Problem(
            problem_id="mbpp/add",
            prompt='"""Add two numbers."""',
            test_code="assert add(1, 2) == 3\nassert add(0, 0) == 0",
        )
        result = {
            "accumulated_code": "def add(a, b):\n    return a + b\n",
            "phases": {
                "code": {"iterations": 2},
                "integrate": {"tests_passed": True},
            },
            "subtasks": ["add"],
            "evolution": {"sweeps": {"final": {}}},
            "adapters": [{"id": "a"}],
        }
        verdict = score_pipeline_result(problem, result, wall_time_s=3.0)
        assert verdict.passed is True
        assert verdict.problem_id == "mbpp/add"
        assert verdict.code_attempts == 2
        assert verdict.diagnose_fired is False
        assert verdict.n_subtasks == 1
        assert verdict.wall_time_s == pytest.approx(3.0)
        assert verdict.accumulated_code_len == len(result["accumulated_code"])
        assert verdict.error == ""
        assert verdict.generation == result["accumulated_code"]
        assert verdict.phase_metrics["phase_integrate_ok"] == 1.0

    def test_failing_code_yields_failed_verdict_with_error(self) -> None:
        from evaluation.benchmarks.protocol import Problem
        from run_benchmark_hpo import score_pipeline_result

        problem = Problem(
            problem_id="mbpp/bad",
            prompt='"""Add."""',
            test_code="assert add(1, 2) == 3",
        )
        result = {
            "accumulated_code": "def add(a, b):\n    return a - b\n",
            "phases": {"code": {"iterations": 1}, "repair": {"iterations": 1}},
        }
        verdict = score_pipeline_result(problem, result, wall_time_s=1.0)
        assert verdict.passed is False
        assert verdict.error != ""
        assert verdict.diagnose_fired is True


class _FakeMlflow:
    """Captures mlflow log calls for assertions (no tracking server needed)."""

    def __init__(self) -> None:
        self.metrics: dict[str, float] = {}
        self.params: dict[str, object] = {}
        self.artifacts: list[tuple[str, str | None]] = []

    def log_metrics(self, d: dict[str, float], step: int | None = None) -> None:
        self.metrics.update(d)

    def log_metric(self, k: str, v: float, step: int | None = None) -> None:
        self.metrics[k] = v

    def log_param(self, k: str, v: object) -> None:
        self.params[k] = v

    def log_params(self, d: dict[str, object]) -> None:
        self.params.update(d)

    def log_artifact(self, path: str, artifact_path: str | None = None) -> None:
        self.artifacts.append((path, artifact_path))


def _verdict(pid: str, passed: bool, attempts: int, diagnosed: bool):
    from run_benchmark_hpo import ProblemVerdict

    return ProblemVerdict(
        problem_id=pid,
        passed=passed,
        code_attempts=attempts,
        diagnose_fired=diagnosed,
        n_subtasks=1,
        wall_time_s=1.0,
        accumulated_code_len=10,
        error="" if passed else "boom",
        phase_metrics={"phase_decompose_ok": 1.0},
    )


class TestVerdictsJsonl:
    def test_write_verdicts_jsonl_one_record_per_line(
        self, tmp_path: Path
    ) -> None:
        from run_benchmark_hpo import write_verdicts_jsonl

        verdicts = [
            _verdict("mbpp/1", True, 2, False),
            _verdict("mbpp/2", False, 1, True),
        ]
        path = tmp_path / "v.jsonl"
        write_verdicts_jsonl(verdicts, path)
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 2
        rec = json.loads(lines[0])
        assert rec["problem_id"] == "mbpp/1"
        assert rec["passed"] is True
        assert "generation" in rec
        assert "phase_metrics" in rec

    def test_log_verdicts_artifact_logs_then_cleans_up(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import run_benchmark_hpo as mod

        captured: dict[str, object] = {}

        def _capture(path: str, artifact_path: str | None = None) -> None:
            captured["existed_during_call"] = Path(path).exists()
            captured["temp_path"] = path
            captured["artifact_path"] = artifact_path

        fake = _FakeMlflow()
        fake.log_artifact = _capture  # type: ignore[method-assign]
        monkeypatch.setattr(mod, "_mlflow", lambda: fake)

        mod.log_verdicts_artifact(
            [_verdict("mbpp/1", True, 1, False)], "trial-001"
        )
        # The artifact must exist at log time and be gone afterwards.
        assert captured["existed_during_call"] is True
        assert captured["artifact_path"] == "verdicts"
        assert not Path(str(captured["temp_path"])).exists()

    def test_log_verdicts_artifact_cleans_up_on_exception(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import run_benchmark_hpo as mod

        captured_dirs: list[Path] = []

        def _raise(path: str, artifact_path: str | None = None) -> None:
            captured_dirs.append(Path(path).parent)
            raise RuntimeError("upload failed")

        fake = _FakeMlflow()
        fake.log_artifact = _raise  # type: ignore[method-assign]
        monkeypatch.setattr(mod, "_mlflow", lambda: fake)

        with pytest.raises(RuntimeError, match="upload failed"):
            mod.log_verdicts_artifact(
                [_verdict("mbpp/1", True, 1, False)], "trial-001"
            )
        assert not captured_dirs[0].exists()


class TestLogTrialMetrics:
    def test_aggregate_metrics(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import run_benchmark_hpo as mod

        fake = _FakeMlflow()
        monkeypatch.setattr(mod, "_mlflow", lambda: fake)
        verdicts = [
            _verdict("mbpp/1", True, 2, False),
            _verdict("mbpp/2", True, 4, True),
            _verdict("mbpp/3", False, 3, False),
            _verdict("mbpp/4", False, 1, True),
        ]
        mod.log_trial_metrics(verdicts, wall_time_s=12.0)
        assert fake.metrics["pass_rate"] == pytest.approx(0.5)
        assert fake.metrics["n_passed"] == 2
        assert fake.metrics["n_problems"] == 4
        assert fake.metrics["wall_time_s"] == pytest.approx(12.0)
        assert fake.metrics["mean_attempts_used"] == pytest.approx(2.5)
        assert fake.metrics["diagnose_fire_rate"] == pytest.approx(0.5)

    def test_per_problem_metrics_and_error_param(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import run_benchmark_hpo as mod

        fake = _FakeMlflow()
        monkeypatch.setattr(mod, "_mlflow", lambda: fake)
        mod.log_trial_metrics([_verdict("mbpp/9", False, 1, False)], wall_time_s=1.0)
        assert fake.metrics["problem/mbpp/9/passed"] == 0.0
        assert fake.metrics["problem/mbpp/9/code_attempts"] == 1
        assert fake.metrics["problem/mbpp/9/phase_decompose_ok"] == 1.0
        assert fake.params["problem/mbpp/9/error"] == "boom"


class TestMakeObjective:
    def test_objective_returns_pass_rate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import contextlib

        import optuna
        import run_benchmark_hpo as mod

        fake = _FakeMlflow()
        monkeypatch.setattr(mod, "_mlflow", lambda: fake)
        monkeypatch.setattr(
            mod, "_mlflow_run", lambda **kw: contextlib.nullcontext()
        )
        monkeypatch.setattr(
            mod, "apply_trial_env", lambda **kw: None
        )
        monkeypatch.setattr(
            mod,
            "evaluate_problem_set",
            lambda problems, *a, **k: [
                _verdict(p.problem_id, True, 1, False) for p in problems
            ],
        )
        problems = [_problem(f"mbpp/{i}") for i in range(8)]
        objective = mod.make_objective(
            problems,
            hypernet_checkpoint="ckpt",
            base_model="m",
            device="cpu",
            problems_per_trial=4,
            seed=42,
            work_dir=tmp_path,
        )
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=1)
        assert study.best_value == pytest.approx(1.0)


def _study_with_two_trials():
    """Build an in-memory Optuna study with two completed trials."""
    import optuna

    study = optuna.create_study(direction="maximize")

    def obj(trial: "optuna.Trial") -> float:
        trial.suggest_float("scaling_factor", 0.02, 0.5, log=True)
        trial.suggest_float("temperature", 0.1, 0.7)
        trial.suggest_float("repetition_penalty", 1.0, 1.3)
        trial.suggest_categorical("max_tokens", [1024, 2048, 4096])
        trial.suggest_int("max_phase_iterations", 2, 6)
        return 0.5 + 0.1 * trial.number

    study.optimize(obj, n_trials=2)
    return study


class TestArtifactWriters:
    def test_save_best_params_writes_json_and_config(self, tmp_path: Path) -> None:
        from run_benchmark_hpo import save_best_params
        from shared.pipeline_config import load_config

        study = _study_with_two_trials()
        config_path = tmp_path / "pipeline_config.json"
        params_path = save_best_params(
            study, out_dir=tmp_path, config_path=config_path
        )
        assert params_path == tmp_path / "best_params.json"
        best = json.loads(params_path.read_text())
        assert set(best) == {
            "scaling_factor",
            "temperature",
            "repetition_penalty",
            "max_tokens",
            "max_phase_iterations",
        }
        loaded = load_config(config_path)
        assert loaded.adapter.scaling == pytest.approx(best["scaling_factor"])
        assert loaded.generation.temperature == pytest.approx(best["temperature"])
        assert loaded.generation.max_tokens == best["max_tokens"]

    def test_write_validation_results(self, tmp_path: Path) -> None:
        from run_benchmark_hpo import write_validation_results

        verdicts = [
            _verdict("mbpp/1", True, 2, False),
            _verdict("mbpp/2", False, 1, True),
        ]
        path = tmp_path / "validation_results.json"
        write_validation_results(verdicts, path)
        data = json.loads(path.read_text())
        assert data["pass_rate"] == pytest.approx(0.5)
        assert data["problems"]["mbpp/1"]["passed"] is True
        assert data["problems"]["mbpp/2"]["passed"] is False
        assert data["problems"]["mbpp/2"]["error"] == "boom"

    def test_write_trial_summary_csv(self, tmp_path: Path) -> None:
        from run_benchmark_hpo import write_trial_summary

        study = _study_with_two_trials()
        path = tmp_path / "trial_summary.csv"
        write_trial_summary(study, path)
        lines = path.read_text().strip().splitlines()
        assert lines[0].startswith("trial_number,state,pass_rate")
        assert len(lines) == 3  # header + 2 trials


class TestBuildParser:
    def test_required_hypernet_checkpoint(self) -> None:
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_defaults(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["--hypernet-checkpoint", "ckpt.pt"])
        assert args.failed_ids == Path("evaluation_results/paper/mbpp_failed_ids.json")
        assert args.n_problems == 127
        assert args.n_trials == 30
        assert args.problems_per_trial == 8
        assert args.seed == 42
        assert args.base_model == "Qwen/Qwen3.5-9B"
        assert args.device == "cuda"
        assert args.study_name.startswith("mbpp-hpo-")
        assert args.db is None
        assert args.smoke is False

    def test_smoke_flag(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["--hypernet-checkpoint", "ckpt.pt", "--smoke"])
        assert args.smoke is True

    def test_overrides(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(
            [
                "--hypernet-checkpoint", "ckpt.pt",
                "--failed-ids", "/tmp/f.json",
                "--n-problems", "50",
                "--n-trials", "5",
                "--problems-per-trial", "3",
                "--device", "cpu",
            ]
        )
        assert args.failed_ids == Path("/tmp/f.json")
        assert args.n_problems == 50
        assert args.n_trials == 5
        assert args.problems_per_trial == 3
        assert args.device == "cpu"
