import torch

from rune.model.hypernetwork import reinit_scaler_b_nonzero


class _FakeHypernet:
    def __init__(self) -> None:
        self.scaler_B = torch.nn.ParameterDict(
            {"q_proj": torch.nn.Parameter(torch.zeros((1, 2, 4, 1)))}
        )


def test_reinit_scaler_b_sets_ones() -> None:
    h = _FakeHypernet()
    reinit_scaler_b_nonzero(h, value=1.0)
    assert float(h.scaler_B["q_proj"].abs().min()) == 1.0
    assert h.scaler_B["q_proj"].requires_grad
