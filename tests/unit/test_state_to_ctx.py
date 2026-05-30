from rune.engine.graph import _INTEGRATION_DOC_LINE_CAP, _PROJECT_CAP, state_to_ctx
from rune.engine.state import Subtask, make_initial_state


def test_integration_doc_caps_line_length() -> None:
    state = make_initial_state("t", 5)
    state["subtasks"] = [Subtask(name="a", description="x" * 5000, depends_on=[])]
    ctx = state_to_ctx(state)
    line = ctx["integration_doc"].splitlines()[0]
    assert len(line) <= _INTEGRATION_DOC_LINE_CAP + len("- a: ")


def test_project_is_pre_sliced() -> None:
    state = make_initial_state("z" * 5000, 5)
    ctx = state_to_ctx(state)
    assert len(ctx["project"]) == _PROJECT_CAP
