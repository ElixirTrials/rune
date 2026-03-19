"""Jinja2 template loader for phase trajectory and prompt rendering."""

from jinja2 import Environment, PackageLoader, Undefined

_env = Environment(
    loader=PackageLoader("shared", "templates"),
    keep_trailing_newline=False,
    trim_blocks=True,
    lstrip_blocks=True,
    undefined=Undefined,
)


def render_trajectory(phase: str, **kwargs: object) -> str:
    """Render a phase trajectory template with task-specific variables."""
    template = _env.get_template(f"{phase}.j2")
    return template.render(**kwargs)


def render_prompt(phase: str, **kwargs: object) -> str:
    """Render a phase prompt template with task-specific variables."""
    template = _env.get_template(f"prompt_{phase}.j2")
    return template.render(**kwargs)
