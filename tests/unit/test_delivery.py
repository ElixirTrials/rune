"""Delivery contract and cross-task isolation fixes."""

from __future__ import annotations

from rune.engine.delivery import format_delivery_contract
from rune.engine.graph import _bare_signature_stub, render_episode_adapter, state_to_ctx
from rune.engine.parse import render_template
from rune.engine.repair_brief import _enrich_assertion_invariant
from rune.engine.state import Action, Subtask, make_initial_state

_SIG_3748 = "class Solution:\n    def sortMatrix(self, grid: List[List[int]]) -> List[List[int]]:\n        "
_PUBLIC_3748 = (
    "assert sortMatrix(*[[[1, 7, 3], [9, 8, 2], [4, 5, 6]]]) == "
    "[[8, 2, 3], [9, 6, 7], [4, 5, 1]]"
)


def test_bare_signature_preserves_grid_param_and_return_type() -> None:
    stub = _bare_signature_stub("sortMatrix", _SIG_3748, "")
    assert "grid" in stub
    assert "matrix" not in stub
    assert "List" in stub


def test_delivery_contract_names_exact_params() -> None:
    stub = _bare_signature_stub("sortMatrix", _SIG_3748, "")
    block = format_delivery_contract(
        entry_point="sortMatrix",
        bare_signature=stub,
        public_checks=_PUBLIC_3748,
    )
    assert "sortMatrix" in block
    assert "grid" in block
    assert "Public grader call shape" in block


def test_episodic_code_prompt_includes_delivery_contract() -> None:
    stub = "def sortMatrix(grid: List[List[int]]):"
    contract = format_delivery_contract(
        entry_point="sortMatrix",
        bare_signature=stub,
        public_checks=_PUBLIC_3748,
    )
    prompt = render_template(
        "prompt_episodic_code",
        subtask_name="sortMatrix",
        delivery_contract=contract,
    )
    assert "grid" in prompt
    assert "Deliverable function name" in prompt


def test_adapter_includes_required_deliverable_for_entry_subtask() -> None:
    state = make_initial_state(
        "sort matrix task", 12, "sortMatrix", _SIG_3748, _PUBLIC_3748
    )
    state["subtasks"] = [
        Subtask("sortMatrix", "Sort anti-diagonals", [], _PUBLIC_3748, "sortMatrix")
    ]
    adp = render_episode_adapter("code", "sortMatrix", state)
    assert "## Required deliverable" in adp
    assert "grid" in adp


def test_enrich_invariant_not_applied_to_even_numbers_task() -> None:
    inv = _enrich_assertion_invariant(
        "generic",
        entry_point="totalNumbers",
        plan="Count three-digit even numbers from digits",
    )
    assert inv == "generic"


def test_enrich_invariant_applied_to_max_difference() -> None:
    inv = _enrich_assertion_invariant(
        "generic",
        entry_point="maxDifference",
        plan="odd and even character frequencies",
    )
    assert "odd-frequency" in inv.lower()


def test_state_to_ctx_sets_delivery_contract() -> None:
    state = make_initial_state("task", 12, "sortMatrix", _SIG_3748, _PUBLIC_3748)
    state["subtasks"] = [Subtask("sortMatrix", "d", [], _PUBLIC_3748, "sortMatrix")]
    action = Action(
        "repair",
        "code_repair",
        "prompt_episodic_repair",
        "",
        None,
        True,
        "sortMatrix",
    )
    ctx = state_to_ctx(state, action)
    assert "grid" in ctx["delivery_contract"]
