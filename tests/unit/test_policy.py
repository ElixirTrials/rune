from rune.engine.policy import (
    build_execution_layers,
    select_action,
)
from rune.engine.state import Feedback, Subtask


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
        "feedback": {},
        "integration_feedback": None,
        "diagnosis": {},
        "actions": [],
        "trajectory": [],
        "step": 0,
        "budget_remaining": 20,
    }
    base.update(overrides)
    return base


class TestSelectAction:
    def test_empty_subtasks_complex_returns_decompose(self) -> None:
        state = _make_state(task="Build a web API with endpoints and database")
        actions = select_action(state)
        assert len(actions) == 1
        assert actions[0].name == "decompose"

    def test_empty_subtasks_simple_returns_decompose(self) -> None:
        state = _make_state(task="Write a function that adds two numbers")
        actions = select_action(state)
        assert len(actions) == 1
        assert actions[0].name == "decompose"

    def test_unplanned_subtasks_returns_plan(self) -> None:
        subtasks = [Subtask("a", "do a", []), Subtask("b", "do b", [])]
        actions = select_action(_make_state(subtasks=subtasks))
        assert all(a.name == "plan" for a in actions)
        assert len(actions) == 2

    def test_uncoded_subtask_returns_code(self) -> None:
        subtasks = [Subtask("a", "do a", [])]
        actions = select_action(_make_state(subtasks=subtasks, plans={"a": "plan a"}))
        assert len(actions) == 1
        assert actions[0].name == "code"

    def test_failed_no_diagnosis_returns_diagnose(self) -> None:
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

    def test_failed_with_diagnosis_returns_repair(self) -> None:
        subtasks = [Subtask("a", "do a", [])]
        fb = Feedback(stdout="", stderr="NameError", exit_code=1)
        state = _make_state(
            subtasks=subtasks,
            plans={"a": "plan"},
            code_results={"a": "bad code"},
            code_passed={"a": False},
            feedback={"a": fb},
            diagnosis={"a": "Fix the import"},
        )
        actions = select_action(state)
        assert len(actions) == 1
        assert actions[0].name == "repair"
        assert actions[0].target_subtask == "a"

    def test_two_repairs_failed_returns_code_resample(self) -> None:
        subtasks = [Subtask("a", "do a", [])]
        fb = Feedback(stdout="", stderr="err", exit_code=1)
        state = _make_state(
            subtasks=subtasks,
            plans={"a": "plan"},
            code_results={"a": "bad"},
            code_passed={"a": False},
            retries={"a": 2},
            feedback={"a": fb},
        )
        actions = select_action(state)
        assert len(actions) == 1
        assert actions[0].name == "code"
        assert actions[0].target_subtask == "a"

    def test_max_retries_exhausted_skips_subtask(self) -> None:
        subtasks = [Subtask("a", "do a", [])]
        fb = Feedback(stdout="", stderr="err", exit_code=1)
        state = _make_state(
            subtasks=subtasks,
            plans={"a": "plan"},
            code_results={"a": "bad"},
            code_passed={"a": False},
            retries={"a": 4},
            feedback={"a": fb},
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

    def test_integration_failure_returns_diagnose(self) -> None:
        subtasks = [Subtask("a", "do a", [])]
        fb = Feedback(stdout="", stderr="ImportError", exit_code=1)
        state = _make_state(
            subtasks=subtasks,
            plans={"a": "plan"},
            code_results={"a": "good"},
            code_passed={"a": True},
            integration_feedback=fb,
        )
        actions = select_action(state)
        assert actions[0].name == "diagnose"

    def test_integration_failure_with_diagnosis_returns_integrate(self) -> None:
        subtasks = [Subtask("a", "do a", [])]
        fb = Feedback(stdout="", stderr="ImportError", exit_code=1)
        state = _make_state(
            subtasks=subtasks,
            plans={"a": "plan"},
            code_results={"a": "good"},
            code_passed={"a": True},
            integration_feedback=fb,
            diagnosis={"a": "Fix the import"},
        )
        actions = select_action(state)
        assert actions[0].name == "integrate"


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

    def test_phantom_dependency_excluded(self) -> None:
        subtasks = [
            Subtask("a", "do a", []),
            Subtask("b", "do b", ["phantom"]),
        ]
        layers = build_execution_layers(subtasks)
        all_names = [name for layer in layers for name in layer]
        assert "phantom" not in all_names
        assert set(all_names) == {"a", "b"}


class TestSelectActionPhantomDep:
    def test_plan_action_never_targets_phantom(self) -> None:
        subtasks = [
            Subtask("a", "do a", []),
            Subtask("b", "do b", ["nonexistent"]),
        ]
        actions = select_action(_make_state(subtasks=subtasks))
        for a in actions:
            assert a.target_subtask in {"a", "b"}
