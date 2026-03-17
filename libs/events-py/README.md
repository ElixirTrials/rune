# events-py

Event envelope shapes and helpers for Python service communication.

## Event Types

`EventKind` enum: `CREATED`, `UPDATED`, `DELETED`

## Event Envelope

```python
class EventEnvelope(TypedDict):
    id: str           # UUID
    kind: str         # "created" | "updated" | "deleted"
    payload: Any      # JSON-serializable event data
    timestamp: str    # ISO 8601 UTC
```

## Creating Events

```python
from events_py import EventKind, create_event

event = create_event(EventKind.CREATED, {"adapter_id": "abc-123"})
# => {"id": "uuid...", "kind": "created", "payload": {...}, "timestamp": "..."}
```

## Consumers

Used by: training-svc, evolution-svc, rune-agent, swarm orchestrator (scripts/swarm.py).
