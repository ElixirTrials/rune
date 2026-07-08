"""Structural stop for the continuation sub-loop (issue #52 §4 lever 4)."""

from __future__ import annotations

from rune.engine.continuation import continuation_should_abort

_SALVAGEABLE = "def maxDistance(s, k):\n    total = 0\n    return total\n"
_HEADLESS = "def maxDistance(s, k):\n    total = 0\n"  # no return, still salvageable
_PROSE_CHUNK = (
    "\n\nGiven the ambiguity, and since the user input is empty, "
    "I will output:\n\n0\n"
)
_CODE_CHUNK = "\n    for c in s:\n        total += 1\n"


class TestContinuationShouldAbort:
    def test_aborts_on_prose_chunk_over_salvageable_entry(self) -> None:
        assert continuation_should_abort(_PROSE_CHUNK, _SALVAGEABLE, "maxDistance")

    def test_aborts_even_when_entry_is_headless(self) -> None:
        # 3754's real shape: a complete-enough def with no return; the salvage
        # extractor still recovers it, so keep it and stop pumping prose.
        assert continuation_should_abort(_PROSE_CHUNK, _HEADLESS, "maxDistance")

    def test_no_abort_when_chunk_is_code(self) -> None:
        assert not continuation_should_abort(_CODE_CHUNK, _SALVAGEABLE, "maxDistance")

    def test_no_abort_without_salvageable_entry(self) -> None:
        # Only prose accumulated so far; nothing to keep, so let continuation run.
        assert not continuation_should_abort(
            _PROSE_CHUNK, "Let me think about this problem.", "maxDistance"
        )

    def test_no_abort_without_entry_point(self) -> None:
        assert not continuation_should_abort(_PROSE_CHUNK, _SALVAGEABLE, "")

    def test_recovers_from_prose_tailed_blob(self) -> None:
        # Accumulated blob is a valid def followed by an unparseable prose tail;
        # the salvage extractor recovers the def, so a further prose chunk aborts.
        blob = _SALVAGEABLE + "\nAnd therefore the answer is: definitely 0!!!\n"
        assert continuation_should_abort(_PROSE_CHUNK, blob, "maxDistance")
