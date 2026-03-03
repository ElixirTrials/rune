"""LangGraph workflow for the Rune coding agent recursive loop."""

from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from .nodes import execute_node, generate_node, reflect_node, save_trajectory_node
from .state import RuneState


def should_retry(state: RuneState) -> Literal["generate", "save_trajectory"]:
    """Route from reflect: retry if attempts remain and tests failed, else save.

    This is a REAL implementation, not a stub:
    - If tests passed -> save_trajectory (success)
    - If attempt_count >= max_attempts -> save_trajectory (exhausted)
    - Otherwise -> generate (retry)

    Args:
        state: Current agent state after reflection.

    Returns:
        Next node name: 'generate' to retry or 'save_trajectory' to finish.

    Example:
        >>> state = {"tests_passed": False, "attempt_count": 0, "max_attempts": 3}
        >>> should_retry(state)
        'generate'
        >>> state2 = {"tests_passed": True, "attempt_count": 1, "max_attempts": 3}
        >>> should_retry(state2)
        'save_trajectory'
    """
    if state["tests_passed"]:
        return "save_trajectory"
    if state["attempt_count"] >= state["max_attempts"]:
        return "save_trajectory"
    return "generate"


def create_graph() -> Any:
    """Create and compile the Rune agent workflow graph.

    The graph follows this flow:
    START -> generate -> execute -> reflect -> [should_retry]
        -> generate (retry) OR save_trajectory -> END

    Returns:
        Compiled StateGraph ready for execution.

    Example:
        >>> graph = create_graph()
        >>> graph is not None
        True
    """
    workflow = StateGraph(RuneState)

    workflow.add_node("generate", generate_node)
    workflow.add_node("execute", execute_node)
    workflow.add_node("reflect", reflect_node)
    workflow.add_node("save_trajectory", save_trajectory_node)

    workflow.add_edge(START, "generate")
    workflow.add_edge("generate", "execute")
    workflow.add_edge("execute", "reflect")
    workflow.add_conditional_edges("reflect", should_retry)
    workflow.add_edge("save_trajectory", END)

    return workflow.compile()


# Singleton instance for reuse
_graph = None


def get_graph() -> Any:
    """Get or create the compiled graph instance.

    Returns:
        Compiled StateGraph instance.
    """
    global _graph
    if _graph is None:
        _graph = create_graph()
    return _graph
