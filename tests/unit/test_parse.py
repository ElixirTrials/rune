import json

from rune.engine.parse import (
    DecomposeResult,
    DiagnoseResult,
    PlanResult,
    parse_output,
    render_template,
)
from rune.engine.state import Action, Feedback, Subtask


class TestRenderTemplate:
    def test_renders_jinja2(self) -> None:
        text = render_template("decompose", project="build a calculator", subtasks=[])
        assert "calculator" in text

    def test_decompose_describes_json_schema(self) -> None:
        text = render_template("decompose", project="build a calculator", subtasks=[])
        assert "subtasks" in text
        assert "depends_on" in text
        assert "numbered list" not in text.lower()

    def test_decompose_prompt_describes_json_schema(self) -> None:
        text = render_template(
            "prompt_decompose_concise", task_description="build a calculator"
        )
        assert "subtasks" in text
        assert "depends_on" in text
        assert "numbered list" not in text.lower()

    def test_integration_diagnose_prompt_lists_subtasks(self) -> None:
        # The model must be shown the real subtask names (and the integration
        # error) so its diagnosis maps back to actual subtasks for repair.
        text = render_template(
            "prompt_diagnose",
            target_subtask="",
            project_label="build X",
            integration_error="ImportError: foo",
            integration_doc="- _main: implement everything",
        )
        assert "_main" in text
        assert "ImportError: foo" in text


class TestDecomposeResult:
    def test_parse_valid_json(self) -> None:
        raw = '{"subtasks": [{"name": "parse", "description": "Parse input", "depends_on": []}]}'
        result = DecomposeResult.model_validate_json(raw)
        assert len(result.subtasks) == 1
        assert result.subtasks[0].name == "parse"


class TestPlanResult:
    def test_parse_plan(self) -> None:
        raw = '{"plan": "Step 1: Define types. Step 2: Implement logic."}'
        result = PlanResult.model_validate_json(raw)
        assert "Define types" in result.plan

    def test_plan_no_thinking_tokens(self) -> None:
        raw = '{"plan": "Clean plan without thinking tokens."}'
        result = PlanResult.model_validate_json(raw)
        assert "<think>" not in result.plan


class TestDiagnoseResult:
    def test_parse_structured_json(self) -> None:
        raw = '{"entries": [{"subtask_name": "solver", "error_type": "name", "location": "line 5", "fix_guidance": "Add missing import for math module"}]}'
        result = DiagnoseResult.model_validate_json(raw)
        assert len(result.entries) == 1
        assert result.entries[0].subtask_name == "solver"
        assert result.entries[0].error_type == "name"
        assert "import" in result.entries[0].fix_guidance

    def test_multiple_entries(self) -> None:
        raw = '{"entries": [{"subtask_name": "a", "error_type": "syntax", "location": "line 1", "fix_guidance": "fix syntax"}, {"subtask_name": "b", "error_type": "assertion", "location": "line 10", "fix_guidance": "fix assertion"}]}'
        result = DiagnoseResult.model_validate_json(raw)
        assert len(result.entries) == 2


class TestParseOutput:
    def test_decompose_action(self) -> None:
        action = Action(
            "decompose",
            "decompose",
            "prompt_decompose_concise",
            "",
            DecomposeResult,
            False,
            None,
        )
        raw = '{"subtasks": [{"name": "a", "description": "do a", "depends_on": []}]}'
        state_stub: dict = {
            "plans": {},
            "code_results": {},
            "code_passed": {},
            "retries": {},
            "subtasks": [],
            "feedback": {},
            "diagnosis": {},
        }
        updates = parse_output(action, raw, None, state_stub)
        assert len(updates["subtasks"]) == 1

    def test_plan_action(self) -> None:
        action = Action("plan", "plan", "prompt_plan", "", PlanResult, False, "task_a")
        raw = json.dumps({"plan": "Architecture: define types, then implement."})
        state_stub: dict = {"plans": {}}
        updates = parse_output(action, raw, None, state_stub)
        assert "Architecture" in updates["plans"]["task_a"]

    def test_code_action_passing(self) -> None:
        action = Action("code", "code", "prompt_code", "", None, True, "task_a")
        fb = Feedback(stdout="ok", stderr="", exit_code=0)
        state_stub: dict = {
            "code_results": {},
            "code_passed": {},
            "retries": {},
            "feedback": {},
            "diagnosis": {},
        }
        raw = "print(1)"
        updates = parse_output(action, raw, fb, state_stub)
        assert updates["code_passed"]["task_a"] is True
        assert "print(1)" in updates["code_results"]["task_a"]

    def test_single_subtask_code_pass_sets_integrated_code(self) -> None:
        action = Action("code", "code", "prompt_code", "", None, True, "_main")
        fb = Feedback(stdout="ok", stderr="", exit_code=0)
        state_stub: dict = {
            "code_results": {},
            "code_passed": {},
            "retries": {},
            "feedback": {},
            "diagnosis": {},
            "subtasks": [Subtask("_main", "task body", [])],
        }
        raw = "def solution(): return 42"
        updates = parse_output(action, raw, fb, state_stub)
        assert updates["integrated_code"] == "def solution(): return 42"

    def test_code_increments_retries(self) -> None:
        action = Action("code", "code", "prompt_code", "", None, True, "task_a")
        fb = Feedback(stdout="ok", stderr="", exit_code=0)
        state_stub: dict = {
            "code_results": {"task_a": "old code"},
            "code_passed": {"task_a": False},
            "retries": {"task_a": 2},
            "feedback": {},
            "diagnosis": {"task_a": "old diagnosis"},
        }
        raw = "print('new')"
        updates = parse_output(action, raw, fb, state_stub)
        assert updates["retries"]["task_a"] == 3
        assert "task_a" not in updates["diagnosis"]

    def test_repair_increments_retries(self) -> None:
        action = Action(
            "repair",
            "code_repair",
            "prompt_code_repair",
            "",
            None,
            True,
            "task_a",
        )
        fb = Feedback(stdout="", stderr="err", exit_code=1)
        state_stub: dict = {
            "code_results": {"task_a": "old"},
            "code_passed": {"task_a": False},
            "retries": {"task_a": 0},
            "feedback": {},
            "diagnosis": {"task_a": "fix the bug"},
        }
        raw = "pass"
        updates = parse_output(action, raw, fb, state_stub)
        assert updates["retries"]["task_a"] == 1
        assert "task_a" not in updates["diagnosis"]

    def test_diagnose_action_structured(self) -> None:
        action = Action(
            "diagnose",
            "diagnose",
            "prompt_diagnose",
            "",
            DiagnoseResult,
            False,
            None,
        )
        raw = '{"entries": [{"subtask_name": "solver", "error_type": "name", "location": "line 5", "fix_guidance": "Add missing import"}]}'
        state_stub: dict = {"diagnosis": {}}
        updates = parse_output(action, raw, None, state_stub)
        assert "solver" in updates["diagnosis"]
        assert updates["diagnosis"]["solver"] == "Add missing import"

    def test_diagnose_caps_fix_guidance_at_150(self) -> None:
        action = Action(
            "diagnose",
            "diagnose",
            "prompt_diagnose",
            "",
            DiagnoseResult,
            False,
            None,
        )
        long_guidance = "x" * 300
        raw = f'{{"entries": [{{"subtask_name": "a", "error_type": "name", "location": "line 1", "fix_guidance": "{long_guidance}"}}]}}'
        updates = parse_output(action, raw, None, {"diagnosis": {}})
        assert len(updates["diagnosis"]["a"]) == 150

    def test_targeted_diagnose_hallucinated_name_attaches_to_target(self) -> None:
        # The model invents subtask_name "write_function" for the real target
        # "_main". Without the fallback, the diagnosis lands on the phantom key,
        # select_action never routes "_main" to repair, and the engine livelocks
        # on diagnose until the budget is spent. The fix attaches guidance to the
        # target and reopens it regardless of the emitted name.
        action = Action(
            "diagnose",
            "diagnose",
            "prompt_diagnose",
            "",
            DiagnoseResult,
            False,
            "_main",
        )
        raw = '{"entries": [{"subtask_name": "write_function", "error_type": "syntax", "location": "line 1", "fix_guidance": "close the code block"}]}'
        state_stub: dict = {"diagnosis": {}, "code_passed": {"_main": False}}
        updates = parse_output(action, raw, None, state_stub)
        assert "_main" in updates["diagnosis"]
        assert "close the code block" in updates["diagnosis"]["_main"]

    def test_integrate_action(self) -> None:
        action = Action(
            "integrate",
            "integrate",
            "prompt_integrate",
            "",
            None,
            True,
            None,
        )
        fb = Feedback(stdout="ok", stderr="", exit_code=0)
        state_stub: dict = {"feedback": {}, "diagnosis": {}}
        raw = "print('all')"
        updates = parse_output(action, raw, fb, state_stub)
        assert updates["integrated_code"] != ""
        assert updates["integration_feedback"].exit_code == 0

    def test_integrate_freeform_fenced_extracts(self) -> None:
        action = Action(
            "integrate",
            "integrate",
            "prompt_integrate",
            "",
            None,
            True,
            None,
        )
        fb = Feedback(stdout="ok", stderr="", exit_code=0)
        state_stub: dict = {"feedback": {}, "diagnosis": {}}
        raw = "```python\nimport os\ndef main():\n    print('hello')\n```"
        updates = parse_output(action, raw, fb, state_stub)
        assert (
            updates["integrated_code"]
            == "import os\ndef main():\n    print('hello')"
        )
        assert updates["integration_feedback"].exit_code == 0

    def test_code_freeform_unterminated_fence_extracts(self) -> None:
        action = Action("code", "code", "prompt_code", "", None, True, "task_a")
        fb = Feedback(stdout="", stderr="SyntaxError", exit_code=1)
        state_stub: dict = {
            "code_results": {},
            "code_passed": {},
            "retries": {},
            "feedback": {},
            "diagnosis": {},
        }
        # truncated freeform output: opening fence, no close -> body to EOF
        raw = "```py\nx = 1\ny = 2\nz = x +"
        updates = parse_output(action, raw, fb, state_stub)
        assert updates["code_results"]["task_a"] == "x = 1\ny = 2\nz = x +"
        assert updates["code_passed"]["task_a"] is False

    def test_decompose_drops_phantom_and_self_deps(self) -> None:
        action = Action(
            "decompose",
            "decompose",
            "prompt_decompose_concise",
            "",
            DecomposeResult,
            False,
            None,
        )
        raw = json.dumps(
            {
                "subtasks": [
                    {"name": "a", "description": "da", "depends_on": ["a", "ghost"]},
                    {"name": "b", "description": "db", "depends_on": ["a"]},
                ]
            }
        )
        state_stub: dict = {"subtasks": []}
        updates = parse_output(action, raw, None, state_stub)
        deps = {s.name: list(s.depends_on) for s in updates["subtasks"]}
        assert deps["a"] == []  # self-ref and phantom 'ghost' dropped
        assert deps["b"] == ["a"]  # real dependency kept

    def test_decompose_malformed_json_returns_empty(self) -> None:
        action = Action(
            "decompose",
            "decompose",
            "prompt_decompose_concise",
            "",
            DecomposeResult,
            False,
            None,
        )
        updates = parse_output(action, '{"subtasks": [trunc', None, {"subtasks": []})
        assert updates == {}  # graceful: no crash, engine re-decomposes

    def test_first_code_attempt_not_counted_as_retry(self) -> None:
        action = Action("code", "code", "prompt_code", "", None, True, "task_a")
        fb = Feedback(stdout="ok", stderr="", exit_code=0)
        state_stub: dict = {
            "code_results": {},  # no prior code → first attempt
            "code_passed": {},
            "retries": {},
            "feedback": {},
            "diagnosis": {},
        }
        updates = parse_output(action, "print(1)", fb, state_stub)
        assert updates["retries"].get("task_a", 0) == 0

    def test_diagnose_reopens_diagnosed_subtasks(self) -> None:
        action = Action(
            "diagnose",
            "diagnose",
            "prompt_diagnose",
            "",
            DiagnoseResult,
            False,
            None,
        )
        raw = (
            '{"entries": [{"subtask_name": "a", "error_type": "integration", '
            '"location": "x", "fix_guidance": "fix a"}]}'
        )
        state_stub: dict = {"diagnosis": {}, "code_passed": {"a": True, "b": True}}
        updates = parse_output(action, raw, None, state_stub)
        assert updates["code_passed"]["a"] is False  # reopened → repairable
        assert updates["code_passed"]["b"] is True  # untouched
        assert updates["diagnosis"]["a"] == "fix a"

    def test_untargeted_diagnose_reopens_all_on_name_mismatch(self) -> None:
        # Integration-failure diagnose where the model names a subtask that does
        # not exist must still reopen the real subtasks (deterministic fallback)
        # so they route to repair instead of livelocking integrate<->diagnose.
        action = Action(
            "diagnose",
            "diagnose",
            "prompt_diagnose",
            "",
            DiagnoseResult,
            False,
            None,  # target_subtask=None → untargeted
        )
        raw = (
            '{"entries": [{"subtask_name": "does_not_exist", '
            '"error_type": "integration", "location": "x", '
            '"fix_guidance": "wire the pieces together"}]}'
        )
        state_stub: dict = {"diagnosis": {}, "code_passed": {"_main": True}}
        updates = parse_output(action, raw, None, state_stub)
        assert updates["code_passed"]["_main"] is False  # reopened despite mismatch
        assert updates["diagnosis"]["_main"]  # has guidance → routes to repair

    def test_integrate_failure_stores_integration_feedback(self) -> None:
        action = Action(
            "integrate",
            "integrate",
            "prompt_integrate",
            "",
            None,
            True,
            None,
        )
        fb = Feedback(stdout="", stderr="ImportError", exit_code=1)
        state_stub: dict = {"feedback": {}, "diagnosis": {}}
        raw = json.dumps({"code": "broken"})
        updates = parse_output(action, raw, fb, state_stub)
        assert updates["integrated_code"] == ""
        assert updates["integration_feedback"].exit_code == 1
