"""Event envelope and kind definitions."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, TypedDict
from uuid import uuid4


class EventKind(str, Enum):
    """Supported event kinds."""

    CREATED = "created"
    UPDATED = "updated"
    DELETED = "deleted"


class EventEnvelope(TypedDict, total=True):
    """Base event envelope for Pub/Sub or internal events."""

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

    Example:
        >>> from events_py import EventKind, create_event
        >>> ev = create_event(EventKind.CREATED, {"name": "foo"})
        >>> ev["kind"]
        'created'
    """
    return {
        "id": event_id or str(uuid4()),
        "kind": kind.value,
        "payload": payload,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
