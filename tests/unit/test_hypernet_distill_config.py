from rune.training.hypernet_distill import DistillConfig, _shuffled


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
