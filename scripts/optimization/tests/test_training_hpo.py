"""CPU tests for ``scripts/optimization/run_training_hpo.py``.

Focus: argparse surface, fitness blend, subsample helper, trial-param →
train_qlora-kwargs translation, heldout stratification, and the
adapter-improvement evaluator shape (GPU paths monkeypatched).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = REPO_ROOT / "scripts" / "optimization"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_training_hpo import (  # noqa: E402
    FitnessConfig,
    HPORunArgs,
    _all_masked_batch_frac_from_trainer_state,
    _build_parser,
    _build_trial_kwargs,
    _compute_fitness,
    _evaluate_adapter_on_heldout,
    _rebalanced_fitness_config,
    _stratify_heldout_split,
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
    assert args.keep_top_k == 3
    assert args.hunk_loss_weight == pytest.approx(0.5)
    assert args.hunk_accuracy_weight == pytest.approx(0.3)
    assert args.adapter_improvement_weight == pytest.approx(0.2)
    assert args.adapter_improvement_eval is True
    assert args.heldout_fraction == pytest.approx(0.1)
    assert args.heldout_strategy == "step_index"


def test_adapter_improvement_flag_default_on() -> None:
    parser = _build_parser()
    args = parser.parse_args(["--dataset", "/tmp/x.jsonl"])
    assert args.adapter_improvement_eval is True


def test_adapter_improvement_flag_off_rebalances_weights() -> None:
    cfg = FitnessConfig()  # defaults 0.5 / 0.3 / 0.2
    rebalanced = _rebalanced_fitness_config(cfg)
    assert rebalanced.adapter_improvement_weight == 0.0
    assert rebalanced.hunk_loss_weight == pytest.approx(0.625)
    assert rebalanced.hunk_accuracy_weight == pytest.approx(0.375)
    # Summing the two remaining weights gives 1.0 exactly.
    assert (
        rebalanced.hunk_loss_weight + rebalanced.hunk_accuracy_weight
        == pytest.approx(1.0)
    )


def test_rebalance_falls_back_when_weights_zero() -> None:
    cfg = FitnessConfig(
        hunk_loss_weight=0.0, hunk_accuracy_weight=0.0, adapter_improvement_weight=1.0
    )
    out = _rebalanced_fitness_config(cfg)
    assert out == FitnessConfig(
        hunk_loss_weight=0.6, hunk_accuracy_weight=0.4, adapter_improvement_weight=0.0
    )


def test_print_only_mode_prints_plan_and_exits(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """--print-only must NOT import optuna / torch and must print the new formula."""
    # Drop any optuna left in sys.modules by an earlier test so the assertion
    # below tests the --print-only codepath itself, not global import state.
    optuna_mod = sys.modules.pop("optuna", None)
    try:
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
        assert "fitness_formula" in payload
        assert "hunk_loss" in payload["fitness_formula"]
        assert "adapter_improvement" in payload["fitness_formula"]
        assert set(payload["fitness"].keys()) == {
            "hunk_loss_weight",
            "hunk_accuracy_weight",
            "adapter_improvement_weight",
        }
        assert payload["heldout"]["fraction"] == pytest.approx(0.1)
        assert payload["heldout"]["strategy"] == "step_index"
        assert payload["heldout"]["adapter_improvement_eval"] is True
        # --print-only must not load optuna (the heavy HPO-side dependency).
        assert "optuna" not in sys.modules
    finally:
        if optuna_mod is not None:
            sys.modules["optuna"] = optuna_mod


def test_print_only_no_adapter_improvement_rebalances(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    rc = main(
        [
            "--dataset",
            str(tmp_path / "x.jsonl"),
            "--output-root",
            str(tmp_path / "hpo"),
            "--no-adapter-improvement-eval",
            "--print-only",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["fitness"]["adapter_improvement_weight"] == pytest.approx(0.0)
    assert payload["heldout"]["adapter_improvement_eval"] is False


def test_smoke_mode_shrinks_budget() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        ["--dataset", "/tmp/x.jsonl", "--n-trials", "50", "--smoke"]
    )
    assert args.smoke is True


def test_subsample_dataset_writes_first_n_lines(tmp_path: Path) -> None:
    src = tmp_path / "src.jsonl"
    lines = [json.dumps({"task_id": f"t{i}"}) for i in range(20)]
    src.write_text("\n".join(lines) + "\n", encoding="utf-8")
    dest = tmp_path / "sub.jsonl"
    written = _subsample_dataset(src, n=5, dest=dest)
    assert written == 5


def test_fitness_blend_new_formula() -> None:
    cfg = FitnessConfig(
        hunk_loss_weight=0.5,
        hunk_accuracy_weight=0.3,
        adapter_improvement_weight=0.2,
    )
    priors = [1.0, 2.0, 3.0]  # normalize maps 1.0→0, 3.0→1
    # Best case: lowest loss (norm 0), perfect accuracy, full improvement.
    best = _compute_fitness(1.0, 1.0, 1.0, prior_losses=priors, cfg=cfg)
    assert best == pytest.approx(0.5 * 1.0 + 0.3 * 1.0 + 0.2 * 1.0)
    # Worst loss + zeros elsewhere.
    worst = _compute_fitness(3.0, 0.0, 0.0, prior_losses=priors, cfg=cfg)
    assert worst == pytest.approx(0.0)


def test_fitness_floors_negative_adapter_improvement() -> None:
    cfg = FitnessConfig()
    priors = [1.0, 2.0, 3.0]
    # Negative adapter improvement (regression) earns 0 credit, not a penalty.
    fit = _compute_fitness(1.0, 0.0, -0.5, prior_losses=priors, cfg=cfg)
    # 0.5 * 1 + 0.3 * 0 + 0.2 * max(0, -0.5) = 0.5
    assert fit == pytest.approx(0.5)


def test_fitness_falls_back_to_midpoint_when_few_priors() -> None:
    cfg = FitnessConfig()
    fit = _compute_fitness(0.5, 0.0, 0.0, prior_losses=[], cfg=cfg)
    # 0.5 * (1 - 0.5) + 0.3 * 0 + 0.2 * 0 = 0.25
    assert fit == pytest.approx(0.25)


def test_build_trial_kwargs_maps_sampled_params() -> None:
    run_args = HPORunArgs(
        dataset="/tmp/x.jsonl",
        adapter_id_prefix="test",
        model_config_name="qwen3.5-9b",
        warm_start="deltacoder",
        subsample=100,
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
    assert kwargs["warmup_ratio"] == pytest.approx(0.05)
    assert "extra_hpo_params" not in kwargs
    assert kwargs["neftune_noise_alpha"] == pytest.approx(5.0)


def test_neftune_search_dim_includes_none() -> None:
    from run_training_hpo import _suggest_trial_params  # noqa: PLC0415

    recorded: dict[str, list[Any]] = {}

    class _FakeTrial:
        number = 0

        def suggest_float(self, name: str, *args: object, **kwargs: object) -> float:
            return 0.0

        def suggest_categorical(self, name: str, choices: list[Any]) -> Any:
            recorded[name] = list(choices)
            return choices[0]

    _suggest_trial_params(_FakeTrial())
    assert None in recorded["neftune_noise_alpha"]
    assert 5.0 in recorded["neftune_noise_alpha"]
    assert 10.0 in recorded["neftune_noise_alpha"]


# ---------------------------------------------------------------------------
# Heldout split stratification
# ---------------------------------------------------------------------------


def _mkpair(task_id: str, step: int) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "activation_text": f"## Task\n{task_id} step {step}",
        "teacher_text": f"## Task\n{task_id}\n\n## Revision\npass",
        "metadata": {"source_task_id": task_id, "step_index": step},
    }


def test_heldout_split_step_index_no_task_leakage() -> None:
    pairs = []
    for t in range(10):
        for step in range(3):
            pairs.append(_mkpair(f"t{t}", step))

    train, heldout = _stratify_heldout_split(
        pairs, fraction=0.3, strategy="step_index", seed=0
    )
    train_tids = {(p.get("metadata") or {}).get("source_task_id") for p in train}
    heldout_tids = {(p.get("metadata") or {}).get("source_task_id") for p in heldout}
    # Heldout pairs' tasks MUST have their earlier steps in train (step_index
    # strategy keeps pre-terminal pairs in train, only terminal moves to heldout).
    assert heldout_tids <= train_tids  # subset or equal
    # Each held-out task contributed exactly its max-step pair.
    for hp in heldout:
        tid = (hp.get("metadata") or {}).get("source_task_id")
        hp_step = (hp.get("metadata") or {}).get("step_index")
        assert hp_step == 2  # the terminal revision
        # No training pair for this task has step_index 2.
        for tr in train:
            tr_meta = tr.get("metadata") or {}
            if tr_meta.get("source_task_id") == tid:
                assert tr_meta.get("step_index") != 2


def test_heldout_split_random_no_task_overlap() -> None:
    pairs = []
    for t in range(10):
        for step in range(3):
            pairs.append(_mkpair(f"t{t}", step))

    train, heldout = _stratify_heldout_split(
        pairs, fraction=0.3, strategy="random", seed=0
    )
    train_tids = {(p.get("metadata") or {}).get("source_task_id") for p in train}
    heldout_tids = {(p.get("metadata") or {}).get("source_task_id") for p in heldout}
    # Zero overlap: random mode moves entire tasks.
    assert train_tids.isdisjoint(heldout_tids)


def test_heldout_split_fraction_zero_returns_all_train() -> None:
    """fraction=0 must return (pairs, []) unchanged — no heldout."""
    pairs = [_mkpair(f"t{i}", 0) for i in range(5)]
    train, heldout = _stratify_heldout_split(
        pairs, fraction=0, strategy="random", seed=0
    )
    assert heldout == []
    assert len(train) == len(pairs)


def test_heldout_split_single_task_raises() -> None:
    """Two pairs sharing one task_id (N_tasks=1) with fraction>0 raises ValueError."""
    pairs = [_mkpair("only", 0), _mkpair("only", 1)]
    with pytest.raises(ValueError, match="N_tasks=1"):
        _stratify_heldout_split(pairs, fraction=0.5, strategy="random", seed=0)


def test_heldout_split_two_tasks_half_fraction() -> None:
    """2 tasks, fraction=0.5 → 1 heldout, 1 train (clamp not needed)."""
    pairs = [_mkpair("a", 0), _mkpair("b", 0)]
    train, heldout = _stratify_heldout_split(
        pairs, fraction=0.5, strategy="random", seed=0
    )
    train_tids = {(p.get("metadata") or {}).get("source_task_id") for p in train}
    heldout_tids = {(p.get("metadata") or {}).get("source_task_id") for p in heldout}
    assert len(heldout_tids) == 1
    assert len(train_tids) == 1
    assert train_tids.isdisjoint(heldout_tids)


def test_heldout_split_full_fraction_raises() -> None:
    """n_tasks=2, fraction=1.0 must raise ValueError (would hold out all tasks)."""
    pairs = [_mkpair("a", 0), _mkpair("b", 0)]
    with pytest.raises(ValueError, match="would leave train set empty"):
        _stratify_heldout_split(pairs, fraction=1.0, strategy="random", seed=0)


def test_heldout_split_clamp_fires_for_high_fraction() -> None:
    """fraction=0.9 with 2 tasks raises ValueError (ceil(0.9*2)=2 >= n_tasks)."""
    pairs = [_mkpair("a", 0), _mkpair("b", 0)]
    with pytest.raises(ValueError, match="would leave train set empty"):
        _stratify_heldout_split(pairs, fraction=0.9, strategy="random", seed=0)


def test_heldout_split_five_tasks_high_fraction_no_clamp() -> None:
    """5 tasks, fraction=0.6 → ceil(3)=3 heldout, 2 train. Clamp not needed."""
    pairs = [_mkpair(f"t{i}", 0) for i in range(5)]
    train, heldout = _stratify_heldout_split(
        pairs, fraction=0.6, strategy="random", seed=0
    )
    heldout_tids = {(p.get("metadata") or {}).get("source_task_id") for p in heldout}
    train_tids = {(p.get("metadata") or {}).get("source_task_id") for p in train}
    assert len(heldout_tids) == 3
    assert len(train_tids) == 2
    assert train_tids.isdisjoint(heldout_tids)


# ---------------------------------------------------------------------------
# Evaluator shape (CPU path: empty pairs, no torch imports needed)
# ---------------------------------------------------------------------------


def test_evaluate_adapter_on_heldout_empty_returns_zeros() -> None:
    out = _evaluate_adapter_on_heldout(
        "/nonexistent/adapter",
        [],
        base_model_id="Qwen/Qwen3.5-9B",
        compute_adapter_delta=True,
    )
    assert out == {
        "hunk_loss": 0.0,
        "hunk_accuracy": 0.0,
        "adapter_improvement": 0.0,
        "hunk_entropy": 0.0,
    }


def test_tokenize_for_eval_passes_max_length_and_truncation() -> None:
    """_tokenize_for_eval must forward truncation=True and max_length=2048 so
    long mined pairs do not OOM the heldout forward (RCA-2 Cause 2).
    """
    import importlib

    captured: dict[str, object] = {}

    class _FakeTok:
        def __call__(self, text: str, **kwargs: object) -> dict[str, object]:
            captured.update(kwargs)
            return {
                "input_ids": [[1]],
                "attention_mask": [[1]],
                "offset_mapping": [[(0, 0)]],
            }

    hpo = importlib.import_module("run_training_hpo")
    fn = getattr(hpo, "_tokenize_for_eval", None)
    assert fn is not None, "_tokenize_for_eval helper missing"
    fn(_FakeTok(), "hello")
    assert captured.get("truncation") is True
    assert captured.get("max_length") == 2048
    assert captured.get("return_offsets_mapping") is True
    assert captured.get("return_tensors") == "pt"


def test_evaluate_adapter_unload_runs_on_oom(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the heldout forward pass raises (e.g. OOM), the adapter unload
    must still run so the cached base re-enters the next trial clean
    (regression: RCA-5 H1).
    """
    import peft as peft_mod
    import torch
    import transformers

    unload_calls: list[str] = []

    class _FakeAdapterModel:
        device = "cpu"
        peft_config = {"default": object()}

        def eval(self) -> "_FakeAdapterModel":
            return self

        def disable_adapter(self) -> object:
            class _NullCtx:
                def __enter__(self) -> object:
                    return self

                def __exit__(self, *exc: object) -> None:
                    return None

            return _NullCtx()

        def __call__(self, *a: object, **k: object) -> None:
            raise RuntimeError("simulated OOM")

        def unload(self) -> object:
            unload_calls.append("unload")
            return None

    class _FakeTok:
        def __call__(self, text: str, **kwargs: object) -> dict[str, object]:
            return {
                "input_ids": torch.tensor([[1, 2]]),
                "attention_mask": torch.tensor([[1, 1]]),
                "offset_mapping": torch.tensor([[(0, 0), (0, 1)]]),
            }

    class _FakePeftModel:
        @staticmethod
        def from_pretrained(base: object, path: str) -> "_FakeAdapterModel":
            return _FakeAdapterModel()

    # Resolve lazy-loaded classes first (transformers uses _LazyModule).
    _real_bnb = transformers.BitsAndBytesConfig

    # Patch the class methods directly on the already-imported objects so the
    # deferred `from transformers import ...` inside _evaluate_adapter_on_heldout
    # picks up our fakes (the deferred import resolves the same class object).
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        classmethod(lambda cls, *a, **k: _FakeTok()),
    )
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        classmethod(lambda cls, *a, **k: object()),
    )
    # Patch __init__ on the real class so the deferred `from transformers import
    # BitsAndBytesConfig` still gets the same class but with no-op init.
    monkeypatch.setattr(_real_bnb, "__init__", lambda self, **kwargs: None)
    monkeypatch.setattr(peft_mod, "PeftModel", _FakePeftModel)

    pairs = [{"activation_text": "a", "teacher_text": "post = 1"}]
    with pytest.raises(RuntimeError, match="simulated OOM"):
        _evaluate_adapter_on_heldout(
            "y",
            pairs,
            base_model_id="x",
            compute_adapter_delta=False,
        )
    assert unload_calls == ["unload"], (
        "adapter_model.unload() not called on OOM — residue leaks to next trial"
    )


def test_flush_gpu_runs_after_train_and_register_in_trial_body() -> None:
    """The trial body must call _flush_gpu_between_phases AFTER
    train_and_register and BEFORE _evaluate_adapter_on_heldout (RCA-2 Cause 3).

    Source-level contract check: this avoids reconstructing a full HPO trial
    fixture in CPU CI. If _run_single_trial is nested inside _objective
    (closure), substitute hpo._objective in the getsource call below and
    adjust the surrounding scope.
    """
    import inspect

    import run_training_hpo as hpo

    # Pick the enclosing function — whichever one actually defines the trial
    # body. Module-level _run_single_trial first, fall back to _objective.
    target = getattr(hpo, "_run_single_trial", None) or getattr(hpo, "_objective", None)
    assert target is not None, (
        "neither _run_single_trial nor _objective is module-level; adjust test"
    )

    src = inspect.getsource(target)
    train_idx = src.find("train_and_register(")
    flush_idx = src.find("_flush_gpu_between_phases(")
    eval_idx = src.find("_evaluate_adapter_on_heldout(")
    assert train_idx >= 0, "train_and_register call site not found"
    assert eval_idx >= 0, "_evaluate_adapter_on_heldout call site not found"
    assert flush_idx >= 0, "_flush_gpu_between_phases call site not found"
    assert train_idx < flush_idx < eval_idx, (
        "Wrong ordering: flush must run between train_and_register and "
        "_evaluate_adapter_on_heldout (RCA-2 Cause 3)"
    )


def test_flush_gpu_helper_invokes_gc_collect(monkeypatch: pytest.MonkeyPatch) -> None:
    """_flush_gpu_between_phases must run gc.collect (twice for promoted gens).
    torch path is best-effort and not asserted (CPU CI may not have torch).
    """
    import gc as gc_module

    import run_training_hpo as hpo

    collect_calls: list[None] = []
    monkeypatch.setattr(
        gc_module, "collect", lambda *a, **k: collect_calls.append(None)
    )

    hpo._flush_gpu_between_phases()
    assert len(collect_calls) >= 2, "expected at least two gc.collect passes"


def test_all_masked_batch_frac_averages_log_history(tmp_path: Path) -> None:
    """The HPO helper must average ``train/all_masked_batch_frac`` entries.

    Trials whose adapter looks fine on hunk metrics but trained against
    largely empty gradients should be discounted; the per-trial mean is
    the signal that flips the operator's confidence.
    """
    state = {
        "log_history": [
            {"loss": 1.0, "train/all_masked_batch_frac": 0.0},
            {"loss": 0.9, "train/all_masked_batch_frac": 0.4},
            {"loss": 0.8},  # entries without the metric must not break
        ]
    }
    (tmp_path / "trainer_state.json").write_text(json.dumps(state))
    assert _all_masked_batch_frac_from_trainer_state(str(tmp_path)) == pytest.approx(
        0.2
    )


def test_all_masked_batch_frac_returns_zero_when_state_absent(
    tmp_path: Path,
) -> None:
    """Missing trainer_state.json → 0.0 (vanilla SFTTrainer trials, etc.)."""
    assert _all_masked_batch_frac_from_trainer_state(str(tmp_path)) == 0.0


def test_upload_adapter_uses_mlflow_and_removes_local(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Successful upload removes the local adapter dir; failure preserves it.

    The MLflow server in this stack is started with ``--serve-artifacts``
    and an S3 ``--default-artifact-root`` (see infra/docker-compose.yml),
    so ``mlflow.log_artifacts`` from the client streams into S3 with no
    additional configuration. We assert the call goes to MLflow exactly
    once per trial and that the local copy is cleaned up only after
    success — both halves of the "don't store twice / don't fill up our
    HD" contract.
    """
    import run_training_hpo as hpo

    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"\x00" * 16)

    calls: list[tuple[str, str]] = []

    class _FakeMlflow:
        @staticmethod
        def log_artifacts(local_dir: str, *, artifact_path: str = "") -> None:
            calls.append((local_dir, artifact_path))

        @staticmethod
        def active_run() -> None:
            return None

    monkeypatch.setitem(sys.modules, "mlflow", _FakeMlflow)

    result = hpo._upload_adapter_and_cleanup(
        str(adapter_dir), upload=True, cleanup=True
    )
    assert result is True
    assert calls == [(str(adapter_dir), "adapter")]
    assert not adapter_dir.exists(), "local copy must be removed after upload"


def test_upload_adapter_keeps_local_when_upload_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A failed upload must NOT delete the local adapter — that would lose data."""
    import run_training_hpo as hpo

    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"\x00" * 16)

    class _FakeMlflow:
        @staticmethod
        def log_artifacts(local_dir: str, *, artifact_path: str = "") -> None:
            raise RuntimeError("S3 unreachable")

        @staticmethod
        def active_run() -> None:
            return None

    monkeypatch.setitem(sys.modules, "mlflow", _FakeMlflow)

    result = hpo._upload_adapter_and_cleanup(
        str(adapter_dir), upload=True, cleanup=True
    )
    assert result is False
    assert adapter_dir.exists()
    assert (adapter_dir / "adapter_model.safetensors").exists()


def test_running_topk_admits_first_entries_until_full() -> None:
    """The first ``k`` offers fill the slate without displacing anyone."""
    import run_training_hpo as hpo

    top_k = hpo._RunningTopK(k=3)
    assert top_k.offer(0.5, trial_number=0, run_id="r0") == (True, None)
    assert top_k.offer(0.7, trial_number=1, run_id="r1") == (True, None)
    assert top_k.offer(0.6, trial_number=2, run_id="r2") == (True, None)


def test_running_topk_rejects_below_worst_when_full() -> None:
    """Once full, an offer below the worst-of-top-K is rejected verbatim."""
    import run_training_hpo as hpo

    top_k = hpo._RunningTopK(k=3)
    for f, n in [(0.5, 0), (0.7, 1), (0.6, 2)]:
        top_k.offer(f, trial_number=n, run_id=f"r{n}")
    # 0.4 < worst (0.5): rejected, nobody displaced.
    assert top_k.offer(0.4, trial_number=3, run_id="r3") == (False, None)


def test_running_topk_displaces_worst_when_full_and_better_offered() -> None:
    """A higher-fitness offer displaces the lowest entry and returns its run id."""
    import run_training_hpo as hpo

    top_k = hpo._RunningTopK(k=3)
    for f, n in [(0.5, 0), (0.7, 1), (0.6, 2)]:
        top_k.offer(f, trial_number=n, run_id=f"r{n}")
    # 0.55 > worst (0.5) → displaces trial 0 (run r0).
    accepted, displaced = top_k.offer(0.55, trial_number=3, run_id="r3")
    assert accepted is True
    assert displaced == "r0"


def test_running_topk_zero_k_never_admits() -> None:
    """``--keep-top-k 0`` must short-circuit: no uploads, no displacements."""
    import run_training_hpo as hpo

    top_k = hpo._RunningTopK(k=0)
    assert top_k.offer(99.0, trial_number=0, run_id="r0") == (False, None)


def test_running_topk_rejects_negative_k() -> None:
    """Negative k is a programming error — surface it loudly."""
    import run_training_hpo as hpo

    with pytest.raises(ValueError, match="k must be >= 0"):
        hpo._RunningTopK(k=-1)


def test_gate_uploads_when_trial_enters_top_k(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """First trial fills an empty slot → upload happens, no displacement."""
    import run_training_hpo as hpo

    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"\x00" * 16)

    log_calls: list[tuple[str, str]] = []
    delete_calls: list[str] = []

    class _FakeMlflow:
        @staticmethod
        def log_artifacts(local_dir: str, *, artifact_path: str = "") -> None:
            log_calls.append((local_dir, artifact_path))

        @staticmethod
        def set_tag(key: str, value: str) -> None:  # for displaced tag (no-op here)
            pass

        @staticmethod
        def active_run() -> None:
            return None

    monkeypatch.setitem(sys.modules, "mlflow", _FakeMlflow)

    class _FakeMlflowTracking:
        class MlflowClient:
            def delete_run(self, run_id: str) -> None:  # noqa: D401
                delete_calls.append(run_id)

    monkeypatch.setitem(sys.modules, "mlflow.tracking", _FakeMlflowTracking)

    run_args = HPORunArgs(
        dataset="/tmp/x",
        adapter_id_prefix="test",
        model_config_name="qwen3.5-9b",
        warm_start=None,
        subsample=10,
        output_root=tmp_path,
        experiment_name="exp",
        keep_top_k=3,
        upload_adapters_to_mlflow=True,
        cleanup_local_adapters=True,
    )
    top_k = hpo._RunningTopK(k=3)
    status = hpo._gate_and_upload_adapter(
        str(adapter_dir),
        run_id="r0",
        trial_number=0,
        fitness=0.5,
        run_args=run_args,
        top_k=top_k,
    )
    assert status == "true"
    assert log_calls == [(str(adapter_dir), "adapter")]
    assert delete_calls == []


def test_gate_skips_upload_when_trial_below_top_k(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A trial below the worst-of-top-K never touches MLflow.

    This is the bound that keeps S3 at ≤ K adapters per study.
    """
    import run_training_hpo as hpo

    log_calls: list[tuple[str, str]] = []

    class _ExplodingMlflow:
        @staticmethod
        def log_artifacts(*_a: Any, **_kw: Any) -> None:
            log_calls.append(("WROTE", "WROTE"))
            raise AssertionError(
                "mlflow.log_artifacts must not run for non-top-K trials"
            )

        @staticmethod
        def set_tag(key: str, value: str) -> None:
            pass

    monkeypatch.setitem(sys.modules, "mlflow", _ExplodingMlflow)

    run_args = HPORunArgs(
        dataset="/tmp/x",
        adapter_id_prefix="test",
        model_config_name="qwen3.5-9b",
        warm_start=None,
        subsample=10,
        output_root=tmp_path,
        experiment_name="exp",
        keep_top_k=1,
        upload_adapters_to_mlflow=True,
        cleanup_local_adapters=True,
    )
    top_k = hpo._RunningTopK(k=1)
    # Seed the slot with a very high fitness.
    top_k.offer(fitness=0.95, trial_number=0, run_id="r0")

    adapter_dir = tmp_path / "adapter_t1"
    adapter_dir.mkdir()
    status = hpo._gate_and_upload_adapter(
        str(adapter_dir),
        run_id="r1",
        trial_number=1,
        fitness=0.10,
        run_args=run_args,
        top_k=top_k,
    )
    assert status == "skipped_not_top_k"
    assert log_calls == []
    # Local copy was cleaned up because cleanup_local_adapters=True.
    assert not adapter_dir.exists()


def test_gate_displaces_and_deletes_old_run_when_full_and_improved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An improving trial displaces the worst slot AND deletes its MLflow run."""
    import run_training_hpo as hpo

    log_calls: list[tuple[str, str]] = []
    delete_calls: list[str] = []
    set_tag_calls: list[tuple[str, str]] = []

    class _FakeMlflow:
        @staticmethod
        def log_artifacts(local_dir: str, *, artifact_path: str = "") -> None:
            log_calls.append((local_dir, artifact_path))

        @staticmethod
        def set_tag(key: str, value: str) -> None:
            set_tag_calls.append((key, value))

        @staticmethod
        def active_run() -> None:
            return None

    class _FakeMlflowTracking:
        class MlflowClient:
            def delete_run(self, run_id: str) -> None:
                delete_calls.append(run_id)

    monkeypatch.setitem(sys.modules, "mlflow", _FakeMlflow)
    monkeypatch.setitem(sys.modules, "mlflow.tracking", _FakeMlflowTracking)

    run_args = HPORunArgs(
        dataset="/tmp/x",
        adapter_id_prefix="test",
        model_config_name="qwen3.5-9b",
        warm_start=None,
        subsample=10,
        output_root=tmp_path,
        experiment_name="exp",
        keep_top_k=1,
        upload_adapters_to_mlflow=True,
        cleanup_local_adapters=False,  # don't try to rm the seeded fake dir
    )
    top_k = hpo._RunningTopK(k=1)
    top_k.offer(fitness=0.20, trial_number=0, run_id="r0")  # fills the slot

    adapter_dir = tmp_path / "adapter_t1"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"\x00" * 16)

    status = hpo._gate_and_upload_adapter(
        str(adapter_dir),
        run_id="r1",
        trial_number=1,
        fitness=0.80,
        run_args=run_args,
        top_k=top_k,
    )
    assert status == "true"
    assert log_calls == [(str(adapter_dir), "adapter")]
    assert delete_calls == ["r0"]
    assert ("hpo.displaced_run_id", "r0") in set_tag_calls


def test_gate_disabled_short_circuits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``--no-upload-adapters-to-mlflow`` returns ``disabled`` and never uploads."""
    import run_training_hpo as hpo

    class _ExplodingMlflow:
        @staticmethod
        def log_artifacts(*_a: Any, **_kw: Any) -> None:
            raise AssertionError("mlflow must not be called when upload disabled")

        @staticmethod
        def set_tag(*_a: Any, **_kw: Any) -> None:
            raise AssertionError("mlflow must not be called when upload disabled")

    monkeypatch.setitem(sys.modules, "mlflow", _ExplodingMlflow)

    run_args = HPORunArgs(
        dataset="/tmp/x",
        adapter_id_prefix="test",
        model_config_name="qwen3.5-9b",
        warm_start=None,
        subsample=10,
        output_root=tmp_path,
        experiment_name="exp",
        keep_top_k=3,
        upload_adapters_to_mlflow=False,
        cleanup_local_adapters=True,  # must be ignored on the disabled path
    )
    top_k = hpo._RunningTopK(k=3)
    adapter_dir = tmp_path / "adapter_t0"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"\x00" * 16)

    status = hpo._gate_and_upload_adapter(
        str(adapter_dir),
        run_id="r0",
        trial_number=0,
        fitness=0.99,
        run_args=run_args,
        top_k=top_k,
    )
    assert status == "disabled"
    # Local copy preserved because deleting it without an upload would
    # be data loss.
    assert adapter_dir.exists()


def test_upload_adapter_noop_when_upload_disabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``upload=False`` short-circuits BEFORE any cleanup attempt.

    Operators using ``--no-upload-adapters-to-mlflow`` keep adapters
    local-only; deleting the local copy in that path would lose the
    adapter outright.
    """
    import run_training_hpo as hpo

    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"\x00" * 16)

    # Sentinel that fails loudly if mlflow is touched.
    class _ExplodingMlflow:
        @staticmethod
        def log_artifacts(*_a: Any, **_kw: Any) -> None:
            raise AssertionError("mlflow must not be called when upload=False")

    monkeypatch.setitem(sys.modules, "mlflow", _ExplodingMlflow)

    result = hpo._upload_adapter_and_cleanup(
        str(adapter_dir), upload=False, cleanup=True
    )
    assert result is False
    assert adapter_dir.exists()
