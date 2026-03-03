"""CPU-only importability smoke test for evolution-svc."""

import evolution_svc
from evolution_svc.routers.evolution import router


def test_evolution_svc_is_importable() -> None:
    """evolution_svc can be imported without a GPU present."""
    assert evolution_svc is not None


def test_evolution_router_is_importable() -> None:
    """evolution router can be imported without a GPU present."""
    assert router is not None
