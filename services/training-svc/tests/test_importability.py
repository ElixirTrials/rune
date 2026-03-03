"""CPU-only importability smoke test for training-svc."""

import training_svc
from training_svc.routers.training import router


def test_training_svc_is_importable() -> None:
    """training_svc can be imported without a GPU present."""
    assert training_svc is not None


def test_training_router_is_importable() -> None:
    """training router can be imported without a GPU present."""
    assert router is not None
