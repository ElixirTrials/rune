"""Event envelope and kind definitions."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, TypedDict
from uuid import uuid4


class EventKind(str, Enum):
    """Supported event kinds for Rune service events.

    Defines the three base event types. Rune-specific kinds
    (e.g. TRAINING_STARTED) will be added in service phases.

    Example:
        >>> EventKind.CREATED.value
        'created'
        >>> EventKind("updated") == EventKind.UPDATED
        True
    """

    CREATED = "created"
    UPDATED = "updated"
    DELETED = "deleted"


class EventEnvelope(TypedDict, total=True):
    """Base event envelope for Pub/Sub or internal events.

    All events share this structure. The payload field carries
    event-specific data as a JSON-serializable dict.

    Example:
        >>> envelope: EventEnvelope = {
        ...     "id": "evt-001",
        ...     "kind": "created",
        ...     "payload": {"name": "adapter-1"},
        ...     "timestamp": "2026-01-01T00:00:00+00:00",
        ... }
    """

    id: str
    kind: Literal["created", "updated", "deleted"]
    payload: Any
    timestamp: str


def create_event(
    kind: EventKind,
    payload: Any,
    event_id: str | None = None,
) -> EventEnvelope:
    """Create an event envelope.

    Args:
        kind: Event kind (created, updated, deleted).
        payload: Event payload (must be JSON-serializable).
        event_id: Optional ID; if omitted, a UUID is generated.

    Returns:
        Event envelope dict.

    Raises:
        ValueError: If kind is not an EventKind member or payload is None.

    Example:
        >>> from events_py import EventKind, create_event
        >>> ev = create_event(EventKind.CREATED, {"name": "foo"})
        >>> ev["kind"]
        'created'
    """
    if not isinstance(kind, EventKind):
        raise ValueError(f"kind must be an EventKind member, got {kind!r}")
    if payload is None:
        raise ValueError("payload must not be None")
    return {
        "id": event_id or str(uuid4()),
        "kind": kind.value,
        "payload": payload,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
