from rune.model.hypernetwork import audit_checkpoint_keys


def test_audit_checkpoint_keys_flags_dropped_scaler_b() -> None:
    model_keys = {"scaler_A.q_proj", "scaler_B.q_proj", "bias_A.q_proj", "head.weight"}
    ckpt_keys = {"scaler_A.q_proj", "head.weight"}  # scaler_B + bias_A missing
    missing = audit_checkpoint_keys(model_keys, ckpt_keys)
    assert "scaler_B.q_proj" in missing
    assert "bias_A.q_proj" in missing


def test_audit_checkpoint_keys_empty_when_all_present() -> None:
    keys = {"scaler_A.q_proj", "scaler_B.q_proj"}
    assert audit_checkpoint_keys(keys, keys) == set()
