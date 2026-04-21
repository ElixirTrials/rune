"""CPU tests verifying new kwargs are exposed on train_qlora / train_and_register.

No GPU libraries are loaded — these tests only inspect function signatures.
"""

from __future__ import annotations

import inspect


def test_warmup_ratio_kwarg_exposed() -> None:
    """train_qlora and train_and_register both accept warmup_ratio with default None."""
    from model_training.trainer import train_and_register, train_qlora

    sig_qlora = inspect.signature(train_qlora)
    assert "warmup_ratio" in sig_qlora.parameters
    assert sig_qlora.parameters["warmup_ratio"].default is None

    sig_register = inspect.signature(train_and_register)
    assert "warmup_ratio" in sig_register.parameters
    assert sig_register.parameters["warmup_ratio"].default is None


def test_neftune_noise_alpha_kwarg_exposed() -> None:
    """train_qlora and train_and_register accept neftune_noise_alpha=None."""
    from model_training.trainer import train_and_register, train_qlora

    sig_qlora = inspect.signature(train_qlora)
    assert "neftune_noise_alpha" in sig_qlora.parameters
    assert sig_qlora.parameters["neftune_noise_alpha"].default is None

    sig_register = inspect.signature(train_and_register)
    assert "neftune_noise_alpha" in sig_register.parameters
    assert sig_register.parameters["neftune_noise_alpha"].default is None
