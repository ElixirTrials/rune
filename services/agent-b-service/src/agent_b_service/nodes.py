"""Node functions for the secondary agent."""

import logging
from typing import Any, Dict

from .state import AgentState

logger = logging.getLogger(__name__)


async def process_node(state: AgentState) -> Dict[str, Any]:
    """Process the input data.

    This is a placeholder node that should be customized for your use case.

    Args:
        state: Current agent state containing messages and context.

    Returns:
        State updates with processing results.
    """
    logger.info("Running process node")
    context = state.get("context", {})

    # Placeholder processing logic
    results = {
        "processed": True,
        "context_size": len(context),
    }

    return {"results": results}


async def finalize_node(state: AgentState) -> Dict[str, Any]:
    """Finalize the agent output.

    Args:
        state: Current agent state.

    Returns:
        Final state updates.
    """
    logger.info("Running finalize node")
    results = state.get("results", {})

    return {
        "results": {
            **results,
            "finalized": True,
        }
    }
