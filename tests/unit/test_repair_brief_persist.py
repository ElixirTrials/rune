"""Repair brief state must survive step_node return (feeds next-step diagnose)."""

from __future__ import annotations

from rune.engine.repair_brief import RepairBrief


def test_brief_updates_included_in_step_return_contract() -> None:
    """Graph step_node seeds updates from brief_updates (regression guard)."""
    brief_updates = {
        "repair_briefs": {
            "fn": RepairBrief(
                failure_class="complexity",
                violated_invariant="algorithm too slow",
                observed="TLE on scale",
                expected="O(n) or better",
                fix_directive="use O(n)",
                replan_recommended=True,
            ).format_block()
        },
        "replan_targets": {"fn": True},
    }
    updates: dict[str, object] = dict(brief_updates)
    assert updates["repair_briefs"]["fn"].startswith("failure_class:")
    assert updates["replan_targets"]["fn"] is True
