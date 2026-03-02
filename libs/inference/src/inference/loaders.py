"""Model loaders for the inference component.

This module provides standard ways to initialize LLMs and other models,
ensuring consistent configuration across all services.
"""

import logging
from typing import Any

from shared.lazy_cache import lazy_singleton

logger = logging.getLogger(__name__)


@lazy_singleton
def get_llm() -> Any:
    """Get the default LLM for the repository.

    This is a stub implementation. In a real application, this would
    initialize ChatVertexAI, ChatOpenAI, or a local model via Transformers.
    """
    try:
        from langchain_google_vertexai import (
            ChatVertexAI,  # type: ignore[import-not-found]
        )

        return ChatVertexAI(model_name="gemini-1.5-pro", temperature=0)
    except ImportError:
        logger.warning("langchain-google-vertexai not installed, returning placeholder")
        # Return a mock or raise error in production
        raise ImportError(
            "Please install langchain-google-vertexai or configure a model."
        )


def get_vertex_model(model_name: str = "gemini-1.5-pro") -> Any:
    """Loader specifically for Vertex AI models as referenced in README.md."""
    from langchain_google_vertexai import ChatVertexAI  # type: ignore[import-not-found]

    return ChatVertexAI(model_name=model_name, temperature=0)
