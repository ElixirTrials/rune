from rune.engine.policy import build_execution_layers, select_action
from rune.engine.state import Subtask


def _make_state(**overrides: object) -> dict:
    base: dict = {
        "task": "test",
        "subtasks": [],
        "interfaces": {},
        "plans": {},
        "code_results": {},
        "code_passed": {},
        "retries": {},
        "integrated_code": "",
        "current_adapter": None,
        "feedback": None,
        "diagnosis": None,
        "actions": [],
        "trajectory": [],
        "step": 0,
        "budget_remaining": 20,
    }
    base.update(overrides)
    return base


class TestSelectAction:
    def test_empty_subtasks_returns_decompose(self) -> None:
        actions = select_action(_make_state())
        assert len(actions) == 1
        assert actions[0].name == "decompose"

    def test_unplanned_subtasks_returns_plan(self) -> None:
        subtasks = [Subtask("a", "do a", []), Subtask("b", "do b", [])]
        actions = select_action(_make_state(subtasks=subtasks))
        assert all(a.name == "plan" for a in actions)
        assert len(actions) == 2  # both are independent

    def test_uncoded_subtask_returns_code(self) -> None:
        subtasks = [Subtask("a", "do a", [])]
        actions = select_action(_make_state(subtasks=subtasks, plans={"a": "plan a"}))
        assert len(actions) == 1
        assert actions[0].name == "code"

    def test_failed_code_returns_code_retry(self) -> None:
        subtasks = [Subtask("a", "do a", [])]
        state = _make_state(
            subtasks=subtasks,
            plans={"a": "plan"},
            code_results={"a": "bad code"},
            code_passed={"a": False},
            retries={"a": 1},
        )
        actions = select_action(state)
        assert actions[0].name == "code_retry"

    def test_max_retries_returns_empty(self) -> None:
        subtasks = [Subtask("a", "do a", [])]
        state = _make_state(
            subtasks=subtasks,
            plans={"a": "plan"},
            code_results={"a": "bad"},
            code_passed={"a": False},
            retries={"a": 3},
        )
        actions = select_action(state)
        assert actions == []

    def test_all_passing_returns_integrate(self) -> None:
        subtasks = [Subtask("a", "do a", [])]
        state = _make_state(
            subtasks=subtasks,
            plans={"a": "plan"},
            code_results={"a": "good"},
            code_passed={"a": True},
        )
        actions = select_action(state)
        assert actions[0].name == "integrate"

    def test_done_returns_empty(self) -> None:
        subtasks = [Subtask("a", "do a", [])]
        state = _make_state(
            subtasks=subtasks,
            plans={"a": "plan"},
            code_results={"a": "good"},
            code_passed={"a": True},
            integrated_code="final code",
        )
        actions = select_action(state)
        assert actions == []


class TestBuildExecutionLayers:
    def test_no_deps_single_layer(self) -> None:
        subtasks = [Subtask("a", "", []), Subtask("b", "", [])]
        layers = build_execution_layers(subtasks)
        assert len(layers) == 1
        assert set(layers[0]) == {"a", "b"}

    def test_chain_dependency(self) -> None:
        subtasks = [
            Subtask("a", "", []),
            Subtask("b", "", ["a"]),
            Subtask("c", "", ["b"]),
        ]
        layers = build_execution_layers(subtasks)
        assert len(layers) == 3
        assert layers[0] == ["a"]
        assert layers[1] == ["b"]
        assert layers[2] == ["c"]

    def test_diamond_dependency(self) -> None:
        subtasks = [
            Subtask("a", "", []),
            Subtask("b", "", ["a"]),
            Subtask("c", "", ["a"]),
            Subtask("d", "", ["b", "c"]),
        ]
        layers = build_execution_layers(subtasks)
        assert layers[0] == ["a"]
        assert set(layers[1]) == {"b", "c"}
        assert layers[2] == ["d"]
