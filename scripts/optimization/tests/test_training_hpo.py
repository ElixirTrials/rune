"""CPU tests for ``scripts/optimization/run_training_hpo.py``.

Focus: the argparse surface, the fitness blend function, the subsample
helper, and the trial-param → train_qlora-kwargs translation. Running
actual Optuna trials requires a real GPU + dataset, so those paths are
deferred to manual / GPU-gated smoke runs documented in the plan.
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

from run_training_hpo import (  # noqa: E402
    FitnessConfig,
    HPORunArgs,
    _build_parser,
    _build_trial_kwargs,
    _compute_fitness,
    _subsample_dataset,
    main,
)


def test_parser_defaults_are_sensible() -> None:
    parser = _build_parser()
    args = parser.parse_args(["--dataset", "/tmp/x.jsonl"])
    assert args.n_trials == 10
    assert args.model_config_name == "qwen3.5-9b"
    assert args.warm_start == "deltacoder"
    assert args.subsample == 500
    assert args.eval_tier == "smoke"
    assert args.keep_top_k == 3
    assert args.loss_weight == pytest.approx(0.6)
    assert args.pass_at_1_weight == pytest.approx(0.4)


def test_print_only_mode_prints_plan_and_exits(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """--print-only must NOT import optuna or run any trials."""
    rc = main(
        [
            "--dataset",
            str(tmp_path / "x.jsonl"),
            "--output-root",
            str(tmp_path / "hpo"),
            "--print-only",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out.strip())
    assert payload["study_name"] == "rune-training-v1"
    assert payload["n_trials"] == 10
    assert "fitness" in payload
    # optuna should not be imported in --print-only mode.
    assert "optuna" not in sys.modules


def test_smoke_mode_shrinks_budget() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        ["--dataset", "/tmp/x.jsonl", "--n-trials", "50", "--smoke"]
    )
    assert args.smoke is True
    # _run_single_trial uses subsample=4 when --smoke is set; we can't run
    # the study here, but we can spot-check that HPORunArgs honors the flag
    # by replicating the construction logic from main().
    run_args = HPORunArgs(
        dataset=args.dataset,
        adapter_id_prefix=args.study_name,
        model_config_name=args.model_config_name,
        warm_start=args.warm_start,
        subsample=args.subsample if not args.smoke else 4,
        eval_tier=args.eval_tier,
        output_root=Path(args.output_root),
        experiment_name=args.experiment_name,
        keep_top_k=args.keep_top_k,
    )
    assert run_args.subsample == 4


def test_subsample_dataset_writes_first_n_lines(tmp_path: Path) -> None:
    src = tmp_path / "src.jsonl"
    lines = [json.dumps({"task_id": f"t{i}"}) for i in range(20)]
    src.write_text("\n".join(lines) + "\n", encoding="utf-8")
    dest = tmp_path / "sub.jsonl"
    written = _subsample_dataset(src, n=5, dest=dest)
    assert written == 5
    got = dest.read_text(encoding="utf-8").strip().splitlines()
    assert len(got) == 5
    parsed = [json.loads(x) for x in got]
    assert [p["task_id"] for p in parsed] == ["t0", "t1", "t2", "t3", "t4"]


def test_subsample_dataset_skips_blank_lines(tmp_path: Path) -> None:
    src = tmp_path / "src.jsonl"
    src.write_text(
        json.dumps({"a": 1}) + "\n\n" + json.dumps({"a": 2}) + "\n",
        encoding="utf-8",
    )
    dest = tmp_path / "sub.jsonl"
    written = _subsample_dataset(src, n=10, dest=dest)
    assert written == 2  # blank line skipped


def test_fitness_falls_back_to_midpoint_when_few_priors() -> None:
    cfg = FitnessConfig()
    # With fewer than 3 priors, loss contributes the midpoint baseline.
    fit = _compute_fitness(0.5, 0.0, prior_losses=[], cfg=cfg)
    # 0.6 * (1 - 0.5) + 0.4 * 0 = 0.3
    assert fit == pytest.approx(0.3)


def test_fitness_rewards_low_loss_and_high_pass_at_1() -> None:
    cfg = FitnessConfig(loss_weight=0.6, pass_at_1_weight=0.4)
    priors = [1.0, 2.0, 3.0]  # normalize maps 1.0→0.0, 3.0→1.0
    low = _compute_fitness(1.0, 1.0, prior_losses=priors, cfg=cfg)
    high = _compute_fitness(3.0, 0.0, prior_losses=priors, cfg=cfg)
    # low loss + perfect pass@1 = 0.6*1 + 0.4*1 = 1.0
    assert low == pytest.approx(1.0)
    # high loss + zero pass@1 = 0.6*0 + 0.4*0 = 0.0
    assert high == pytest.approx(0.0)


def test_fitness_clamps_out_of_range_loss() -> None:
    cfg = FitnessConfig()
    priors = [1.0, 2.0, 3.0]
    # loss below min → loss_norm clamps to 0 → reward is max from loss term.
    fit = _compute_fitness(0.5, 0.0, prior_losses=priors, cfg=cfg)
    assert fit == pytest.approx(0.6 * 1.0)


def test_fitness_handles_inf_loss() -> None:
    """A crashed trial with inf loss still produces a finite fitness."""
    cfg = FitnessConfig()
    priors = [1.0, 2.0, 3.0]
    fit = _compute_fitness(float("inf"), 0.0, prior_losses=priors, cfg=cfg)
    # With inf loss we fall back to midpoint baseline (0.5), so:
    # 0.6 * (1 - 0.5) + 0.4 * 0 = 0.3
    assert fit == pytest.approx(0.3)


def test_build_trial_kwargs_maps_sampled_params() -> None:
    run_args = HPORunArgs(
        dataset="/tmp/x.jsonl",
        adapter_id_prefix="test",
        model_config_name="qwen3.5-9b",
        warm_start="deltacoder",
        subsample=100,
        eval_tier="smoke",
        output_root=Path("/tmp/hpo"),
        experiment_name="rune-qlora-hpo",
        keep_top_k=3,
    )
    sampled = {
        "lr": 3e-4,
        "alpha_override": 64,
        "lora_dropout": 0.05,
        "warmup_ratio": 0.05,
        "grad_accum": 16,
        "lr_scheduler": "cosine",
        "diff_aware_loss": True,
        "neftune_noise_alpha": 5.0,
    }
    kwargs = _build_trial_kwargs(
        run_args=run_args,
        sampled=sampled,
        adapter_id="test-t001",
        trial_dataset_path="/tmp/trial.jsonl",
    )
    assert kwargs["learning_rate"] == 3e-4
    assert kwargs["override_lora_alpha"] == 64
    assert kwargs["override_lora_dropout"] == 0.05
    assert kwargs["gradient_accumulation_steps"] == 16
    assert kwargs["lr_scheduler_type"] == "cosine"
    assert kwargs["diff_aware_loss"] is True
    assert kwargs["dataset_path"] == "/tmp/trial.jsonl"
    assert kwargs["session_id"] is None
    assert kwargs["model_config_name"] == "qwen3.5-9b"
    # warm_start alias resolution happened through trainer_cli helper.
    assert kwargs["warm_start_adapter_id"] == (
        "danielcherubini/Qwen3.5-DeltaCoder-9B"
    )
    # warmup_ratio now forwarded directly — NOT under extra_hpo_params.
    assert kwargs["warmup_ratio"] == pytest.approx(0.05)
    assert "extra_hpo_params" not in kwargs
    # neftune_noise_alpha forwarded directly.
    assert kwargs["neftune_noise_alpha"] == pytest.approx(5.0)


def test_neftune_search_dim_includes_none() -> None:
    """_suggest_trial_params samples neftune_noise_alpha from {None, 5.0, 10.0}."""
    from run_training_hpo import _suggest_trial_params  # noqa: PLC0415

    recorded: dict[str, list] = {}

    class _FakeTrial:
        number = 0

        def suggest_float(self, name: str, *args: object, **kwargs: object) -> float:
            return 0.0

        def suggest_categorical(self, name: str, choices: list) -> object:
            recorded[name] = list(choices)
            return choices[0]

    _suggest_trial_params(_FakeTrial())
    assert "neftune_noise_alpha" in recorded
    choices = recorded["neftune_noise_alpha"]
    assert None in choices
    assert 5.0 in choices
    assert 10.0 in choices
