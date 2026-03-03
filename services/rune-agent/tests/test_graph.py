"""TDD tests for rune-agent graph functions.

should_retry and create_graph are REAL implementations; all tests pass.
Tests cover both retry and terminal branches of should_retry, plus
successful graph compilation via create_graph.
"""

from rune_agent.graph import create_graph, should_retry


def test_should_retry_returns_generate_on_retry():
    """Test should_retry returns 'generate' when attempts remain and tests failed."""
    state = {"tests_passed": False, "attempt_count": 0, "max_attempts": 3}
    assert should_retry(state) == "generate"


def test_should_retry_returns_save_trajectory_on_exhausted():
    """Test should_retry returns 'save_trajectory' when attempts exhausted."""
    state = {"tests_passed": False, "attempt_count": 3, "max_attempts": 3}
    assert should_retry(state) == "save_trajectory"


def test_should_retry_returns_save_trajectory_on_success():
    """Test should_retry returns 'save_trajectory' when tests passed."""
    state = {"tests_passed": True, "attempt_count": 1, "max_attempts": 3}
    assert should_retry(state) == "save_trajectory"


def test_create_graph_compiles():
    """Test create_graph returns a compiled graph without error."""
    graph = create_graph()
    assert graph is not None
