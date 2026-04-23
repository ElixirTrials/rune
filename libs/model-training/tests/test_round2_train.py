"""CPU-only unit tests for round2_train module."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from model_training.round2_train import (
    _teacher_forward_with_oracle,
    _training_step_round2,
    register_round2_adapter,
    train_d2l_qwen3_round2,
)


class _StubLogits:
    """Stand-in for the ``.logits`` attribute of an HF model output."""

    def __init__(self, marker: str) -> None:
        self.marker = marker


class _StubOutput:
    def __init__(self, marker: str) -> None:
        self.logits = _StubLogits(marker)


class _FakeCtxMgr:
    """Context manager that tracks enter/exit counts via outer closure."""

    def __init__(self, enter_log: list, exit_log: list, name: str) -> None:
        self._enter_log = enter_log
        self._exit_log = exit_log
        self._name = name

    def __enter__(self) -> None:
        self._enter_log.append(self._name)

    def __exit__(self, *exc: object) -> None:
        self._exit_log.append(self._name)


def test_teacher_forward_applies_oracle_lora_dict_via_functional_lora(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """oracle_lora_dict is not None → apply_functional_lora wraps the base forward."""
    from model_training import round2_train

    enter_log: list = []
    exit_log: list = []
    applied_dicts: list = []

    def _fake_apply(base: object, lora_dict: object, hc: object) -> _FakeCtxMgr:
        applied_dicts.append(lora_dict)
        return _FakeCtxMgr(enter_log, exit_log, "functional_lora")

    monkeypatch.setattr(round2_train, "_apply_functional_lora", _fake_apply)

    base = MagicMock(name="base")
    base.return_value = _StubOutput("from_base_with_oracle_patch")
    oracle_dict = {"q_proj": {"A": MagicMock(), "B": MagicMock()}}
    hc = MagicMock(name="hc")
    inputs = {"input_ids": MagicMock(), "attention_mask": MagicMock()}

    logits = _teacher_forward_with_oracle(
        base_model=base,
        oracle_lora_dict=oracle_dict,
        hc=hc,
        inputs=inputs,
    )

    assert logits.marker == "from_base_with_oracle_patch"
    assert applied_dicts == [oracle_dict]
    assert enter_log == ["functional_lora"]
    assert exit_log == ["functional_lora"]
    base.assert_called_once_with(**inputs, output_hidden_states=False)


def test_teacher_forward_bypasses_functional_lora_when_oracle_is_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """oracle_lora_dict=None → bare base model forward, functional_lora NOT called."""
    from model_training import round2_train

    def _must_not_call(*a: object, **kw: object) -> None:
        raise AssertionError(
            "apply_functional_lora must not be called when oracle is None"
        )

    monkeypatch.setattr(round2_train, "_apply_functional_lora", _must_not_call)

    base = MagicMock(name="base")
    base.return_value = _StubOutput("from_bare_base")
    hc = MagicMock(name="hc")
    inputs = {"input_ids": MagicMock(), "attention_mask": MagicMock()}

    logits = _teacher_forward_with_oracle(
        base_model=base,
        oracle_lora_dict=None,
        hc=hc,
        inputs=inputs,
    )

    assert logits.marker == "from_bare_base"
    base.assert_called_once_with(**inputs, output_hidden_states=False)


def _make_record() -> dict[str, object]:
    return {
        "task_id": "humaneval/HE-0/decompose",
        "activation_text": "## Task\nwrite X",
        "teacher_text": "## Task\nwrite X\n\n## Implementation\nreturn 0",
        "metadata": {
            "phase": "decompose",
            "benchmark": "humaneval",
            "problem_id": "HE-0",
        },
    }


def test_training_step_round2_routes_to_oracle_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_training_step_round2 asks cache.get(bin_key) and plumbs lora_dict through."""
    from model_training import round2_train

    cache = MagicMock()
    oracle_lora_dict = {"q_proj": {"A": MagicMock(), "B": MagicMock()}}
    cache.get.return_value = oracle_lora_dict

    fake_features = MagicMock(name="features")
    fake_mask = MagicMock(name="mask")
    monkeypatch.setattr(
        round2_train,
        "_extract_activations_with_model",
        lambda **_kw: (fake_features, fake_mask),
    )

    # apply_functional_lora is invoked TWICE per step: once for the teacher
    # pass (oracle lora_dict) and once for the student pass (hypernet lora_dict).
    applied: list = []

    class _FakeCtxMgr2:
        def __enter__(self) -> None: return None
        def __exit__(self, *exc: object) -> None: return None

    def _fake_apply(base: object, lora: object, hc: object) -> _FakeCtxMgr2:
        applied.append(lora)
        return _FakeCtxMgr2()

    monkeypatch.setattr(round2_train, "_apply_functional_lora", _fake_apply)

    fake_loss = MagicMock(name="loss_tensor")
    fake_metrics = {"total_loss": 0.42, "kl_loss": 0.2, "ce_loss": 0.22}
    monkeypatch.setattr(
        round2_train,
        "_compute_kl_ce_loss",
        lambda s, t, start, cfg: (fake_loss, fake_metrics),
    )

    class _NoGrad(_FakeCtxMgr2): ...
    monkeypatch.setattr(round2_train, "_torch_no_grad", lambda: _NoGrad())

    hypernet = MagicMock()
    hypernet_lora_dict = MagicMock(name="hypernet_lora_dict")
    hypernet.generate_weights.return_value = (hypernet_lora_dict, None)

    tokenizer = MagicMock()
    tokenizer.return_value = {"input_ids": MagicMock(), "attention_mask": MagicMock()}

    base = MagicMock(name="base")
    base.return_value.logits = MagicMock(name="student_logits")
    base.parameters.return_value = iter([MagicMock(device="cpu")])

    config = MagicMock(max_length=512, oracle_fallback="skip")
    hc = MagicMock(layer_indices=[0, 1])

    loss, metrics = _training_step_round2(
        record=_make_record(),
        base_model=base,
        tokenizer=tokenizer,
        hypernet=hypernet,
        hc=hc,
        config=config,
        oracle_cache=cache,
    )

    cache.get.assert_called_once_with("decompose_humaneval")
    # Teacher pass applied the oracle; student pass applied the hypernet output.
    assert applied == [oracle_lora_dict, hypernet_lora_dict]
    assert loss is fake_loss
    assert metrics["total_loss"] == pytest.approx(0.42)


def test_training_step_round2_skip_fallback_returns_sentinel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """oracle_fallback='skip' + missing oracle → returns (None, {}) so caller skips."""
    cache = MagicMock()
    cache.get.return_value = None   # missing oracle

    config = MagicMock(oracle_fallback="skip", max_length=512)

    result = _training_step_round2(
        record=_make_record(),
        base_model=MagicMock(),
        tokenizer=MagicMock(),
        hypernet=MagicMock(),
        hc=MagicMock(),
        config=config,
        oracle_cache=cache,
    )
    assert result == (None, {})


def test_training_step_round2_base_model_fallback_uses_bare_teacher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """oracle_fallback='base_model' + missing oracle → teacher runs bare base model.

    apply_functional_lora is invoked ONCE (student pass only); the teacher
    pass bypasses it because oracle_lora_dict is None.
    """
    from model_training import round2_train

    cache = MagicMock()
    cache.get.return_value = None   # no oracle for this bin

    fake_features = MagicMock(name="features")
    fake_mask = MagicMock(name="mask")
    monkeypatch.setattr(
        round2_train,
        "_extract_activations_with_model",
        lambda **_kw: (fake_features, fake_mask),
    )

    applied: list = []

    class _FakeCtxMgr3:
        def __enter__(self) -> None: return None
        def __exit__(self, *exc: object) -> None: return None

    def _fake_apply(base: object, lora: object, hc: object) -> _FakeCtxMgr3:
        applied.append(lora)
        return _FakeCtxMgr3()

    monkeypatch.setattr(round2_train, "_apply_functional_lora", _fake_apply)
    monkeypatch.setattr(
        round2_train,
        "_compute_kl_ce_loss",
        lambda s, t, start, cfg: (MagicMock(), {"total_loss": 0.1}),
    )

    class _NoGrad(_FakeCtxMgr3): ...
    monkeypatch.setattr(round2_train, "_torch_no_grad", lambda: _NoGrad())

    hypernet = MagicMock()
    hypernet_lora_dict = MagicMock(name="hypernet_lora_dict")
    hypernet.generate_weights.return_value = (hypernet_lora_dict, None)

    tokenizer = MagicMock()
    tokenizer.return_value = {"input_ids": MagicMock(), "attention_mask": MagicMock()}

    base = MagicMock(name="base")
    base.return_value.logits = MagicMock(name="logits")
    base.parameters.return_value = iter([MagicMock(device="cpu")])

    config = MagicMock(max_length=512, oracle_fallback="base_model")
    hc = MagicMock(layer_indices=[0, 1])

    loss, metrics = _training_step_round2(
        record=_make_record(),
        base_model=base,
        tokenizer=tokenizer,
        hypernet=hypernet,
        hc=hc,
        config=config,
        oracle_cache=cache,
    )

    # Only the student-pass LoRA was applied; teacher ran bare base model.
    assert applied == [hypernet_lora_dict]
    assert loss is not None


def test_train_round2_aborts_when_coverage_below_threshold(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """coverage < min_oracle_coverage → RuntimeError before any training."""
    from model_training import round2_train
    from model_training.round2_config import Round2TrainConfig

    # Stub dataset loader to return 4 records, none with a registered oracle.
    records = [
        {"metadata": {"phase": "decompose", "benchmark": "humaneval"},
         "activation_text": "a", "teacher_text": "at", "task_id": "x/0/decompose"},
    ] * 4
    monkeypatch.setattr(round2_train, "_load_records", lambda path: records)

    # Stub registry: nothing registered.
    from adapter_registry.exceptions import AdapterNotFoundError

    stub_registry = MagicMock()
    stub_registry.retrieve_by_id.side_effect = AdapterNotFoundError("missing")
    monkeypatch.setattr(
        round2_train, "_open_registry", lambda url: stub_registry
    )

    # Stub model+hypernet setup so the pre-audit path does not load models.
    monkeypatch.setattr(round2_train, "_setup_training", lambda cfg: MagicMock())

    cfg = Round2TrainConfig(
        sakana_checkpoint_path=str(tmp_path / "fake.bin"),
        oracle_registry_url="sqlite:///tmp.db",
        dataset_path=str(tmp_path / "x.jsonl"),
        num_steps=2,
        min_oracle_coverage=0.8,
        dry_run=True,
    )
    with pytest.raises(RuntimeError, match="coverage"):
        train_d2l_qwen3_round2(cfg)


def test_train_round2_dry_run_reports_full_coverage(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """dry_run=True + full coverage → returns a report dict without training."""
    from model_training import round2_train
    from model_training.round2_config import Round2TrainConfig

    records = [
        {"metadata": {"phase": "plan", "benchmark": "humaneval"},
         "activation_text": "a", "teacher_text": "at", "task_id": "humaneval/0/plan"},
    ]
    monkeypatch.setattr(round2_train, "_load_records", lambda path: records)

    stub_registry = MagicMock()
    fake_rec = MagicMock()
    fake_rec.is_archived = False
    fake_rec.file_path = "/a/oracle_plan_humaneval"
    stub_registry.retrieve_by_id.return_value = fake_rec
    monkeypatch.setattr(round2_train, "_open_registry", lambda url: stub_registry)

    monkeypatch.setattr(round2_train, "_setup_training", lambda cfg: MagicMock())

    cfg = Round2TrainConfig(
        sakana_checkpoint_path=str(tmp_path / "fake.bin"),
        oracle_registry_url="sqlite:///tmp.db",
        dataset_path=str(tmp_path / "x.jsonl"),
        num_steps=1,
        dry_run=True,
    )
    report = train_d2l_qwen3_round2(cfg)

    assert report["dry_run"] is True
    assert report["coverage_ratio"] == pytest.approx(1.0)
    assert report["bin_counts"] == {"plan_humaneval": 1}
    assert report["num_records"] == 1


def test_training_step_round2_falls_back_to_cpu_when_no_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """base_model with no parameters → tensors moved to CPU via torch.device('cpu').

    Matches round-1's device-placement contract.
    """
    from model_training import round2_train

    cache = MagicMock()
    oracle_lora_dict = {"q_proj": {"A": MagicMock(), "B": MagicMock()}}
    cache.get.return_value = oracle_lora_dict

    fake_features = MagicMock(name="features")
    fake_mask = MagicMock(name="mask")
    monkeypatch.setattr(
        round2_train,
        "_extract_activations_with_model",
        lambda **_kw: (fake_features, fake_mask),
    )

    class _PassthroughCtxMgr:
        def __enter__(self) -> None: return None
        def __exit__(self, *exc: object) -> None: return None

    monkeypatch.setattr(
        round2_train,
        "_apply_functional_lora",
        lambda *a, **kw: _PassthroughCtxMgr(),
    )
    monkeypatch.setattr(round2_train, "_torch_no_grad", lambda: _PassthroughCtxMgr())
    monkeypatch.setattr(
        round2_train,
        "_compute_kl_ce_loss",
        lambda s, t, start, cfg: (MagicMock(), {"total_loss": 0.0}),
    )

    hypernet = MagicMock()
    hypernet.generate_weights.return_value = (MagicMock(), None)

    # Tensor stand-in that records what device it was moved to.
    moved_to: list = []

    class _FakeTensor:
        def to(self, device: object) -> "_FakeTensor":
            moved_to.append(device)
            return self

    # First tokenizer call (answer_start) must return a real list for len().
    # Second call (teacher_inputs) returns _FakeTensor so we can track .to(device).
    _call_count = [0]

    def _tokenizer_side_effect(*args: object, **kwargs: object) -> dict:
        _call_count[0] += 1
        if _call_count[0] == 1:
            # answer_start path: needs len(result["input_ids"])
            return {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}
        return {"input_ids": _FakeTensor(), "attention_mask": _FakeTensor()}

    tokenizer = MagicMock(side_effect=_tokenizer_side_effect)

    base = MagicMock(name="base")
    base.return_value.logits = MagicMock()
    # StopIteration: no parameters at all.
    base.parameters.return_value = iter([])

    config = MagicMock(max_length=512, oracle_fallback="skip")
    hc = MagicMock(layer_indices=[0, 1])

    record = {
        "task_id": "humaneval/HE-0/decompose",
        "activation_text": "a",
        "teacher_text": "at",
        "metadata": {"phase": "decompose", "benchmark": "humaneval"},
    }
    loss, _metrics = _training_step_round2(
        record=record,
        base_model=base,
        tokenizer=tokenizer,
        hypernet=hypernet,
        hc=hc,
        config=config,
        oracle_cache=cache,
    )

    # StopIteration branch hit torch.device('cpu').
    assert loss is not None
    assert len(moved_to) == 2
    for dev in moved_to:
        # torch.device('cpu') has type attribute 'cpu'
        assert str(dev) == "cpu"


def test_cli_build_config_parses_flags(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI argv → Round2TrainConfig via build_config()."""
    import importlib.util

    script = (
        __import__("pathlib").Path(__file__).resolve().parents[3]
        / "scripts"
        / "train_round2.py"
    )
    spec = importlib.util.spec_from_file_location("train_round2", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    argv = [
        "--sakana-checkpoint-path", "/tmp/fake.bin",
        "--oracle-registry-url", "sqlite:///tmp.db",
        "--dataset-path", "/tmp/x.jsonl",
        "--num-steps", "5",
        "--dry-run",
    ]
    cfg = module.build_config(argv)
    assert cfg.num_steps == 5
    assert cfg.dry_run is True
    assert cfg.oracle_registry_url == "sqlite:///tmp.db"
    # Default propagation
    assert cfg.oracle_fallback == "skip"
    assert cfg.lora_r == 8


def test_register_round2_adapter_writes_lineage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Register round-2 adapter: parent_ids = sorted oracle_ids, generation=2."""
    import json

    registry = MagicMock()
    bin_counts = {"plan_mbpp": 3, "decompose_humaneval": 5, "diagnose_pooled": 2}
    adapter_id = register_round2_adapter(
        registry=registry,
        bin_counts=bin_counts,
        adapter_file_path="/adapters/round2_run_42",
        base_model_id="Qwen/Qwen3.5-9B",
        rank=8,
    )
    assert adapter_id.startswith("round2_")
    (stored_record,) = [c.args[0] for c in registry.store.call_args_list]
    assert stored_record.source == "distillation"
    assert stored_record.generation == 2
    parent_ids = json.loads(stored_record.parent_ids)
    assert parent_ids == sorted(
        ["oracle_plan_mbpp", "oracle_decompose_humaneval", "oracle_diagnose_pooled"]
    )
