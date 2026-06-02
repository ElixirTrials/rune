import torch

from rune.model.hypernetwork import reinit_scaler_b_nonzero, scaler_b_is_collapsed


class _FakeHypernet:
    def __init__(self, init: torch.Tensor | None = None) -> None:
        t = torch.zeros((1, 2, 4, 1)) if init is None else init
        self.scaler_B = torch.nn.ParameterDict({"q_proj": torch.nn.Parameter(t)})


def test_reinit_scaler_b_sets_ones() -> None:
    h = _FakeHypernet()
    reinit_scaler_b_nonzero(h, value=1.0)
    assert float(h.scaler_B["q_proj"].abs().min()) == 1.0
    assert h.scaler_B["q_proj"].requires_grad


def test_scaler_b_is_collapsed_detects_zero_init() -> None:
    # ctx_to_lora zero-init = the collapse basin → should be re-initialized.
    assert scaler_b_is_collapsed(_FakeHypernet()) is True


def test_scaler_b_is_collapsed_preserves_learned_warm_start() -> None:
    # A trained warm-start carries a learned, structured scaler_B (mean|·|~0.057).
    # Regression guard for the clobber bug: it must NOT be flagged as collapsed,
    # so the unconditional reinit-to-1.0 (which inflated the B-side ~17x and broke
    # the adapter) cannot recur. See hypernet_distill warm-start init.
    learned = torch.full((1, 2, 4, 1), 0.057)
    assert scaler_b_is_collapsed(_FakeHypernet(learned)) is False


def test_learned_scaler_b_survives_init_unchanged() -> None:
    learned = torch.full((1, 2, 4, 1), 0.057)
    h = _FakeHypernet(learned)
    before = h.scaler_B["q_proj"].detach().clone()
    if scaler_b_is_collapsed(h):  # the guard used at warm-start load
        reinit_scaler_b_nonzero(h, value=1.0)
    assert torch.equal(h.scaler_B["q_proj"], before)
