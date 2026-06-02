import torch

from rune.training.hypernet_distill import (
    DistillConfig,
    _artifact_uploaded,
    _contrastive_logprob_readout,
    _deranged_partner_context,
    _shuffled,
)


def test_distill_config_defaults() -> None:
    cfg = DistillConfig(corpus_path="/tmp/x.jsonl", checkpoint_dir="/tmp/ck")
    assert cfg.l1_reg_coef == 0.0  # L1 sink disabled (#49 §A)
    assert cfg.scaler_b_init == 1.0
    assert cfg.topk == 50
    assert cfg.shuffle is True
    assert cfg.grad_accum_steps == 8
    assert cfg.skip_zero_diff is True


def test_shuffled_deterministic_and_epoch_varying() -> None:
    items = list(range(50))
    a = _shuffled(items, seed=0, epoch=0)
    b = _shuffled(items, seed=0, epoch=0)
    c = _shuffled(items, seed=0, epoch=1)
    assert a == b  # same seed+epoch -> identical
    assert a != c  # different epoch -> different order
    assert sorted(a) == items  # permutation, no loss
    assert items == list(range(50))  # input not mutated


def test_deranged_partner_context_never_returns_current_episode() -> None:
    records = [
        {"task_id": "a", "context": "ctx-a"},
        {"task_id": "b", "context": "ctx-b"},
        {"task_id": "c", "context": "ctx-c"},
    ]

    assert _deranged_partner_context(records, records[1], seed_index=1) == "ctx-c"


def test_deranged_partner_context_falls_back_to_context_identity() -> None:
    records = [
        {"context": "ctx-a"},
        {"context": "ctx-b"},
    ]

    assert _deranged_partner_context(records, records[0], seed_index=0) == "ctx-b"


def test_contrastive_logprob_readout_reports_raw_body_metrics() -> None:
    # Logits predict gold ids [1, 2] from two positions; the mask keeps only the
    # second BODY token so the expected values are easy to inspect.
    matched = torch.tensor([[0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
    mismatch = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    zero = torch.tensor([[0.0, 0.5, 0.0], [0.0, 0.0, 0.0]])
    gold = torch.tensor([1, 2])
    mask = torch.tensor([False, True])

    readout = _contrastive_logprob_readout(
        matched_logits=matched,
        mismatch_logits=mismatch,
        zero_logits=zero,
        gold=gold,
        mask=mask,
        margin=0.25,
    )

    assert readout["contrastive_tokens"] == 1.0
    assert readout["lp_matched"] > readout["lp_mismatch"] > readout["lp_zero"]
    assert readout["hinge_active_frac"] == 0.0


def test_artifact_uploaded_requires_matching_file_size() -> None:
    class Artifact:
        def __init__(self, path: str, file_size: int) -> None:
            self.path = path
            self.file_size = file_size

    class RunInfo:
        run_id = "run-1"

    class Run:
        info = RunInfo()

    class Artifacts:
        @staticmethod
        def list_artifacts(run_id: str, artifact_path: str) -> list[Artifact]:
            assert run_id == "run-1"
            assert artifact_path == "checkpoints"
            return [Artifact("checkpoints/checkpoint.pt", 123)]

    class Mlflow:
        artifacts = Artifacts()

        @staticmethod
        def active_run() -> Run:
            return Run()

    assert _artifact_uploaded(Mlflow(), "checkpoints/checkpoint.pt", 123)
    assert not _artifact_uploaded(Mlflow(), "checkpoints/checkpoint.pt", 456)
