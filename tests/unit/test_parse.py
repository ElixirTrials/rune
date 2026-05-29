import json

from rune.engine.parse import (
    CodeResult,
    DecomposeResult,
    DiagnoseResult,
    IntegrateResult,
    PlanResult,
    parse_output,
    render_template,
)
from rune.engine.state import Action, Feedback


class TestRenderTemplate:
    def test_renders_jinja2(self) -> None:
        text = render_template("decompose", project="build a calculator", subtasks=[])
        assert "calculator" in text


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


class TestCodeResult:
    def test_parse_code(self) -> None:
        raw = json.dumps({"code": "def add(a, b):\n    return a + b"})
        result = CodeResult.model_validate_json(raw)
        assert "def add" in result.code

    def test_code_with_special_chars(self) -> None:
        raw = json.dumps({"code": 'x = "hello\\nworld"'})
        result = CodeResult.model_validate_json(raw)
        assert "hello" in result.code


class TestIntegrateResult:
    def test_parse_integrate(self) -> None:
        raw = json.dumps({"code": "# Full integrated module\ndef main(): pass"})
        result = IntegrateResult.model_validate_json(raw)
        assert "def main" in result.code


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
        action = Action(
            "plan", "plan", "prompt_plan", "", PlanResult, False, "task_a"
        )
        raw = json.dumps({"plan": "Architecture: define types, then implement."})
        state_stub: dict = {"plans": {}}
        updates = parse_output(action, raw, None, state_stub)
        assert "Architecture" in updates["plans"]["task_a"]

    def test_code_action_passing(self) -> None:
        action = Action("code", "code", "prompt_code", "", CodeResult, True, "task_a")
        fb = Feedback(stdout="ok", stderr="", exit_code=0)
        state_stub: dict = {
            "code_results": {},
            "code_passed": {},
            "retries": {},
            "feedback": {},
            "diagnosis": {},
        }
        raw = json.dumps({"code": "print(1)"})
        updates = parse_output(action, raw, fb, state_stub)
        assert updates["code_passed"]["task_a"] is True
        assert "print(1)" in updates["code_results"]["task_a"]

    def test_code_increments_retries(self) -> None:
        action = Action("code", "code", "prompt_code", "", CodeResult, True, "task_a")
        fb = Feedback(stdout="ok", stderr="", exit_code=0)
        state_stub: dict = {
            "code_results": {"task_a": "old code"},
            "code_passed": {"task_a": False},
            "retries": {"task_a": 2},
            "feedback": {},
            "diagnosis": {"task_a": "old diagnosis"},
        }
        raw = json.dumps({"code": "print('new')"})
        updates = parse_output(action, raw, fb, state_stub)
        assert updates["retries"]["task_a"] == 3
        assert "task_a" not in updates["diagnosis"]

    def test_repair_increments_retries(self) -> None:
        action = Action(
            "repair", "code_repair", "prompt_code_repair", "", CodeResult, True, "task_a"
        )
        fb = Feedback(stdout="", stderr="err", exit_code=1)
        state_stub: dict = {
            "code_results": {"task_a": "old"},
            "code_passed": {"task_a": False},
            "retries": {"task_a": 0},
            "feedback": {},
            "diagnosis": {"task_a": "fix the bug"},
        }
        raw = json.dumps({"code": "pass"})
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

    def test_integrate_action(self) -> None:
        action = Action(
            "integrate", "integrate", "prompt_integrate", "", IntegrateResult, True, None
        )
        fb = Feedback(stdout="ok", stderr="", exit_code=0)
        state_stub: dict = {"feedback": {}, "diagnosis": {}}
        raw = json.dumps({"code": "print('all')"})
        updates = parse_output(action, raw, fb, state_stub)
        assert updates["integrated_code"] != ""
        assert updates["integration_feedback"].exit_code == 0

    def test_integrate_truncated_json_uses_fallback(self) -> None:
        action = Action(
            "integrate", "integrate", "prompt_integrate", "", IntegrateResult, True, None
        )
        fb = Feedback(stdout="", stderr="SyntaxError", exit_code=1)
        state_stub: dict = {"feedback": {}, "diagnosis": {}}
        raw = '{"code": "import os\\ndef main():\\n    print(\\"hello'
        updates = parse_output(action, raw, fb, state_stub)
        assert "import os" in updates["integrated_code"] or updates["integrated_code"] == ""
        assert updates["integration_feedback"].exit_code == 1

    def test_code_truncated_json_uses_fallback(self) -> None:
        action = Action("code", "code", "prompt_code", "", CodeResult, True, "task_a")
        fb = Feedback(stdout="", stderr="SyntaxError", exit_code=1)
        state_stub: dict = {
            "code_results": {},
            "code_passed": {},
            "retries": {},
            "feedback": {},
            "diagnosis": {},
        }
        raw = '{"code": "x = 1\\ny = 2\\nz = x +'
        updates = parse_output(action, raw, fb, state_stub)
        assert "x = 1" in updates["code_results"]["task_a"]
        assert updates["code_passed"]["task_a"] is False

    def test_integrate_failure_stores_integration_feedback(self) -> None:
        action = Action(
            "integrate", "integrate", "prompt_integrate", "", IntegrateResult, True, None
        )
        fb = Feedback(stdout="", stderr="ImportError", exit_code=1)
        state_stub: dict = {"feedback": {}, "diagnosis": {}}
        raw = json.dumps({"code": "broken"})
        updates = parse_output(action, raw, fb, state_stub)
        assert updates["integrated_code"] == ""
        assert updates["integration_feedback"].exit_code == 1
