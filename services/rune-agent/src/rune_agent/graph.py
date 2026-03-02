"""LangGraph workflow definition for the guest interaction agent."""

from typing import Any

from langgraph.graph import END, START, StateGraph

from .nodes import extraction_node, validation_node
from .state import AgentState


def create_graph() -> Any:
    """Create and compile the agent workflow graph.

    The graph follows this flow:
    START -> extract -> validate -> END

    Returns:
        Compiled StateGraph ready for execution.
    """
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("extract", extraction_node)
    workflow.add_node("validate", validation_node)

    # Define edges
    workflow.add_edge(START, "extract")
    workflow.add_edge("extract", "validate")
    workflow.add_edge("validate", END)

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
