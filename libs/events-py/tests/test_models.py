"""Tests for events_py.models."""

import re

from events_py.models import EventKind, create_event


def test_create_event_has_required_fields() -> None:
    """create_event returns an envelope with id, kind, payload, timestamp."""
    ev = create_event(EventKind.CREATED, {"foo": "bar"})
    assert "id" in ev
    assert ev["kind"] == "created"
    assert ev["payload"] == {"foo": "bar"}
    assert "timestamp" in ev
    assert re.match(r"^\d{4}-\d{2}-\d{2}T", ev["timestamp"])


def test_create_event_with_explicit_id() -> None:
    """create_event uses provided event_id when given."""
    ev = create_event(EventKind.UPDATED, {}, event_id="my-id")
    assert ev["id"] == "my-id"
    assert ev["kind"] == "updated"
