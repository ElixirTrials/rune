from rune.training.hypernet_distill import DistillConfig


def test_distill_config_defaults() -> None:
    cfg = DistillConfig(corpus_path="/tmp/x.jsonl", checkpoint_dir="/tmp/ck")
    assert cfg.l1_reg_coef == 0.0  # L1 sink disabled (#49 §A)
    assert cfg.scaler_b_init == 1.0
    assert cfg.topk == 50
