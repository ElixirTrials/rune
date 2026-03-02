"""Node functions for the guest interaction agent."""

import logging
from typing import Any, Dict

from .state import AgentState

logger = logging.getLogger(__name__)


async def extraction_node(state: AgentState) -> Dict[str, Any]:
    """Extract relevant information from the input context.

    This is a placeholder node that should be customized for your use case.

    Args:
        state: Current agent state containing messages and context.

    Returns:
        State updates with extracted data.
    """
    logger.info("Running extraction node")
    context = state.get("context", {})

    # Placeholder extraction logic
    # In a real implementation, this would call the inference component
    extracted = {
        "processed": True,
        "input_keys": list(context.keys()),
    }

    return {"extracted_data": extracted}


async def validation_node(state: AgentState) -> Dict[str, Any]:
    """Validate extracted data before proceeding.

    Args:
        state: Current agent state.

    Returns:
        State updates after validation.
    """
    logger.info("Running validation node")
    extracted = state.get("extracted_data", {})

    # Placeholder validation logic
    is_valid = extracted.get("processed", False)

    return {
        "extracted_data": {
            **extracted,
            "validated": is_valid,
        }
    }
