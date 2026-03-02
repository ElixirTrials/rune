"""Agent state definition for the secondary agent."""

from typing import Annotated, Any, Dict, List

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """State for the secondary agent.

    Attributes:
        messages: Conversation messages with automatic accumulation.
        context: Additional context data for the agent.
        results: Results from agent processing.
    """

    messages: Annotated[List[Any], add_messages]
    context: Dict[str, Any]
    results: Dict[str, Any]
