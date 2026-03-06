"""Pytest configuration for libs/shared.

Note: Shared factory fixtures (make_adapter_ref, make_coding_session, etc.)
are defined in the root conftest.py. They are only auto-discovered when
running pytest from the repo root. When running tests in isolation
(e.g., ``uv run pytest libs/shared/tests/``), pytest rootdir isolation
from this component's pyproject.toml prevents root conftest discovery.
Use ``uv run pytest libs/shared/tests/ -c pyproject.toml`` to pick up
root fixtures, or duplicate needed fixtures here.
"""
