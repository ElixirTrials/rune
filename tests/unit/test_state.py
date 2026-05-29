from rune.engine.state import Action, Feedback, RunState, Subtask


class TestSubtask:
    def test_create_subtask(self) -> None:
        s = Subtask(name="parse_input", description="Parse user input", depends_on=[])
        assert s.name == "parse_input"
        assert s.depends_on == []

    def test_subtask_with_dependencies(self) -> None:
        s = Subtask(
            name="validate",
            description="Validate parsed input",
            depends_on=["parse_input"],
        )
        assert s.depends_on == ["parse_input"]


class TestAction:
    def test_create_action(self) -> None:
        a = Action(
            name="decompose",
            trajectory_template="decompose",
            prompt_template="prompt_decompose_concise",
            system_prompt="You are a decomposer.",
            output_schema=None,
            executes_code=False,
            target_subtask=None,
        )
        assert a.name == "decompose"
        assert a.executes_code is False

    def test_action_with_target(self) -> None:
        a = Action(
            name="code",
            trajectory_template="code",
            prompt_template="prompt_code",
            system_prompt="You are a coder.",
            output_schema=None,
            executes_code=True,
            target_subtask="parse_input",
        )
        assert a.target_subtask == "parse_input"


class TestFeedback:
    def test_passing_feedback(self) -> None:
        f = Feedback(stdout="ok", stderr="", exit_code=0)
        assert f.exit_code == 0

    def test_failing_feedback(self) -> None:
        f = Feedback(stdout="", stderr="NameError", exit_code=1)
        assert f.exit_code == 1


class TestRunState:
    def test_empty_initial_state(self) -> None:
        state: RunState = {
            "task": "build a calculator",
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
            "actions": [],
            "trajectory": [],
            "step": 0,
            "budget_remaining": 20,
        }
        assert state["task"] == "build a calculator"
        assert state["budget_remaining"] == 20
        assert state["actions"] == []
        assert state["feedback"] == {}
        assert state["integration_feedback"] is None
        assert state["diagnosis"] == {}

    def test_per_subtask_feedback(self) -> None:
        fb_a = Feedback(stdout="ok", stderr="", exit_code=0)
        fb_b = Feedback(stdout="", stderr="err", exit_code=1)
        state: RunState = {
            "task": "test",
            "subtasks": [],
            "plans": {},
            "code_results": {},
            "code_passed": {},
            "retries": {},
            "integrated_code": "",
            "current_adapter": None,
            "feedback": {"subtask_a": fb_a, "subtask_b": fb_b},
            "integration_feedback": None,
            "diagnosis": {},
            "actions": [],
            "trajectory": [],
            "step": 0,
            "budget_remaining": 20,
        }
        assert state["feedback"]["subtask_a"].exit_code == 0
        assert state["feedback"]["subtask_b"].exit_code == 1

    def test_per_subtask_diagnosis(self) -> None:
        state: RunState = {
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
            "diagnosis": {
                "subtask_a": "Fix the import",
                "subtask_b": "Add return statement",
            },
            "actions": [],
            "trajectory": [],
            "step": 0,
            "budget_remaining": 20,
        }
        assert state["diagnosis"]["subtask_a"] == "Fix the import"
        assert len(state["diagnosis"]) == 2
