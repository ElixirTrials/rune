"""CPU-only importability smoke test for rune-agent."""

from rune_agent import RuneState, create_graph, get_graph


def test_rune_agent_is_importable() -> None:
    """rune_agent can be imported without a GPU present."""
    assert RuneState is not None
    assert create_graph is not None
    assert get_graph is not None
