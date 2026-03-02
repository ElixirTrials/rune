from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Awaitable, Callable, List, Mapping, TypeVar

from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel, ValidationError
from shared.lazy_cache import lazy_singleton
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)
TModel = TypeVar("TModel", bound=BaseModel)


class AgentError(Exception):
    """Base exception for agent failures."""

    pass


class AgentTimeoutError(AgentError):
    """Agent exceeded max steps or time limit."""

    pass


def render_prompts(
    *,
    prompts_dir: Path,
    system_template: str,
    user_template: str,
    prompt_vars: Mapping[str, Any],
) -> tuple[str, str]:
    """Render Jinja2 templates for system and user prompts."""
    jinja_env = Environment(loader=FileSystemLoader(str(prompts_dir)))
    system_tpl = jinja_env.get_template(system_template)
    user_tpl = jinja_env.get_template(user_template)
    return system_tpl.render(**prompt_vars), user_tpl.render(**prompt_vars)


def create_structured_extractor(
    *,
    model_loader: Callable[[], Any],
    prompts_dir: Path,
    response_schema: type[TModel],
    system_template: str,
    user_template: str,
    max_retries: int = 3,
) -> Callable[[Mapping[str, Any]], Awaitable[TModel]]:
    """Create a structured extraction function with retry logic.

    Args:
        model_loader: Callable returning a LangChain-compatible model.
        prompts_dir: Directory containing Jinja2 templates.
        response_schema: Pydantic model for structured output.
        system_template: Filename of system prompt template.
        user_template: Filename of user prompt template.
        max_retries: Number of retry attempts on failure.

    Returns:
        Async function that takes prompt variables and returns validated output.
    """

    @lazy_singleton
    def _get_structured_model() -> Any:
        model = model_loader()
        return model.with_structured_output(response_schema)

    @retry(
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=1, min=2, max=30),
    )
    async def invoke(prompt_vars: Mapping[str, Any]) -> TModel:
        system_prompt, user_prompt = render_prompts(
            prompts_dir=prompts_dir,
            system_template=system_template,
            user_template=user_template,
            prompt_vars=prompt_vars,
        )
        structured_model = _get_structured_model()

        try:
            result = await structured_model.ainvoke(
                [("system", system_prompt), ("user", user_prompt)]
            )
        except Exception as e:
            logger.error(f"Model invocation failed: {e}")
            raise AgentError(f"Model invocation failed: {e}") from e

        # Validate and return
        if isinstance(result, response_schema):
            return result
        if isinstance(result, dict):
            try:
                return response_schema(**result)
            except ValidationError as e:
                logger.warning(f"Validation failed, retrying: {e}")
                raise AgentError(f"Output validation failed: {e}") from e

        raise AgentError(f"Unexpected output type: {type(result)}")

    return invoke


def create_react_agent_with_tools(
    *,
    model_loader: Callable[[], Any],
    prompts_dir: Path,
    tools: List[Any],
    response_schema: type[TModel],
    system_template: str,
    user_template: str,
    max_steps: int = 10,
) -> Callable[[Mapping[str, Any]], Awaitable[TModel]]:
    """Create a ReAct agent with tools and cycle prevention.

    Includes hard stop after max_steps to prevent infinite loops.
    """
    from langgraph.prebuilt import create_react_agent

    @lazy_singleton
    def _get_agent():
        model = model_loader()
        return create_react_agent(model=model, tools=tools)

    async def invoke(prompt_vars: Mapping[str, Any]) -> TModel:
        system_prompt, user_prompt = render_prompts(
            prompts_dir=prompts_dir,
            system_template=system_template,
            user_template=user_template,
            prompt_vars=prompt_vars,
        )

        agent = _get_agent()
        config = {"recursion_limit": max_steps}

        try:
            result = await agent.ainvoke(
                {"messages": [("system", system_prompt), ("user", user_prompt)]},
                config=config,
            )
        except Exception as e:
            if "recursion" in str(e).lower():
                raise AgentTimeoutError(f"Agent exceeded {max_steps} steps") from e
            raise AgentError(str(e)) from e

        # Extract structured output from final message
        messages = result.get("messages", [])
        if not messages:
            raise AgentError("Agent returned no messages")

        final_message = messages[-1]
        content = getattr(final_message, "content", str(final_message))

        # Parse the content as structured output
        try:
            if isinstance(content, dict):
                return response_schema(**content)
            elif isinstance(content, str):
                parsed = json.loads(content)
                return response_schema(**parsed)
            elif isinstance(content, response_schema):
                return content
            else:
                raise AgentError(f"Cannot parse content of type {type(content)}")
        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(f"Failed to parse agent output: {e}")
            raise AgentError(f"Failed to parse agent output: {e}") from e

    return invoke
