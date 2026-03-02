"""Agent state definition for the guest interaction agent."""

from typing import Annotated, Any, Dict, List

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """State for the guest interaction agent.

    Attributes:
        messages: Conversation messages with automatic accumulation.
        context: Additional context data for the agent.
        extracted_data: Data extracted by the agent during processing.
    """

    messages: Annotated[List[Any], add_messages]
    context: Dict[str, Any]
    extracted_data: Dict[str, Any]
