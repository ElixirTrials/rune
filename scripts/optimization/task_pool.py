"""Diverse task pool for cross-task optimization.

Each task defines a project description, a specific subtask to generate code
for, and domain keywords for scoring relevance.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalTask:
    """A single evaluation task for optimization trials."""

    name: str
    project: str
    subtask: str
    plan: str
    domain_keywords: tuple[str, ...]


TASK_POOL: tuple[EvalTask, ...] = (
    EvalTask(
        name="event_store",
        project=(
            "Build an event-sourced bank ledger in Python. "
            "LedgerEvent dataclass, EventStore backed by sqlite3, "
            "Ledger class with credit/debit/transfer, Decimal amounts."
        ),
        subtask="EventStore Implementation",
        plan="sqlite3-backed event store with append_event and replay_events.",
        domain_keywords=("event", "store", "sqlite", "account", "ledger"),
    ),
    EvalTask(
        name="regex_nfa",
        project=(
            "Build a regular expression engine in Python from scratch. "
            "Lexer, Parser, NFA via Thompson's construction, "
            "RegexEngine with match/search/findall."
        ),
        subtask="NFA Construction",
        plan="Thompson's algorithm: State/Fragment classes, "
        "convert parsed AST to NFA with epsilon transitions.",
        domain_keywords=("state", "nfa", "epsilon", "transition", "fragment"),
    ),
    EvalTask(
        name="cli_manager",
        project=(
            "Build a Python CLI task manager with SQLite storage, "
            "CRUD operations, status tracking, priority levels, "
            "argparse subcommands."
        ),
        subtask="Storage Layer",
        plan="SQLite-backed TaskRepository with create, read, update, "
        "delete, list_by_status methods.",
        domain_keywords=("task", "sqlite", "status", "priority", "repository"),
    ),
    EvalTask(
        name="http_client",
        project=(
            "Build a Python HTTP client wrapper with retry logic, "
            "exponential backoff, auth token refresh, response caching, "
            "and request/response logging."
        ),
        subtask="Retry Logic",
        plan="RetryPolicy class with max_retries, backoff_factor, "
        "retry_on status codes. Wraps requests.Session.",
        domain_keywords=("retry", "backoff", "request", "response", "status"),
    ),
    EvalTask(
        name="data_pipeline",
        project=(
            "Build a data pipeline in Python that reads CSV files, "
            "validates schema, transforms columns, computes aggregates, "
            "and outputs results as JSON."
        ),
        subtask="Schema Validation",
        plan="SchemaValidator class that checks column names, types, "
        "required fields, and value constraints against a schema dict.",
        domain_keywords=("schema", "column", "validate", "field", "constraint"),
    ),
)


def get_tasks(names: list[str] | None = None) -> list[EvalTask]:
    """Return tasks, optionally filtered by name."""
    if names is None:
        return list(TASK_POOL)
    name_set = set(names)
    return [t for t in TASK_POOL if t.name in name_set]
