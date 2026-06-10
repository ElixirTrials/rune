from rune.engine.policy import select_action
from rune.engine.state import Feedback, Subtask


def _make_state(**overrides: object) -> dict:
    base: dict = {
        "task": "test",
        "subtasks": [],
        "plans": {},
        "code_results": {},
        "code_passed": {},
        "retries": {},
        "integrated_code": "",
        "current_adapter": None,
        "feedback": {},
        "integration_feedback": None,
        "diagnosis": {},
        "repair_briefs": {},
        "actions": [],
        "trajectory": [],
        "step": 0,
        "budget_remaining": 20,
    }
    base.update(overrides)
    return base


def test_nonempty_brief_no_diagnosis_skips_to_repair() -> None:
    subtasks = [Subtask("a", "do a", [])]
    fb = Feedback(stdout="", stderr="NameError", exit_code=1)
    state = _make_state(
        subtasks=subtasks,
        plans={"a": "plan"},
        code_results={"a": "bad code"},
        code_passed={"a": False},
        feedback={"a": fb},
        repair_briefs={"a": "failure_class: import\nmissing module foo"},
    )
    actions = select_action(state)
    assert len(actions) == 1
    assert actions[0].name == "repair"
    assert actions[0].target_subtask == "a"


def test_empty_brief_no_diagnosis_still_diagnoses() -> None:
    subtasks = [Subtask("a", "do a", [])]
    fb = Feedback(stdout="", stderr="NameError", exit_code=1)
    state = _make_state(
        subtasks=subtasks,
        plans={"a": "plan"},
        code_results={"a": "bad code"},
        code_passed={"a": False},
        feedback={"a": fb},
        repair_briefs={"a": "   "},
    )
    actions = select_action(state)
    assert len(actions) == 1
    assert actions[0].name == "diagnose"
    assert actions[0].target_subtask == "a"


def test_absent_brief_no_diagnosis_still_diagnoses() -> None:
    subtasks = [Subtask("a", "do a", [])]
    fb = Feedback(stdout="", stderr="NameError", exit_code=1)
    state = _make_state(
        subtasks=subtasks,
        plans={"a": "plan"},
        code_results={"a": "bad code"},
        code_passed={"a": False},
        feedback={"a": fb},
    )
    actions = select_action(state)
    assert len(actions) == 1
    assert actions[0].name == "diagnose"
    assert actions[0].target_subtask == "a"
