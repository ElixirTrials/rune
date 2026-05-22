from rune.engine.parse import (
    DecomposeResult,
    DiagnoseResult,
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


class TestDiagnoseResult:
    def test_parse_valid_json(self) -> None:
        raw = '{"fix_guidance": "Add missing import for os module"}'
        result = DiagnoseResult.model_validate_json(raw)
        assert "import" in result.fix_guidance


class TestParseOutput:
    def test_decompose_action(self) -> None:
        action = Action(
            "decompose",
            "decompose",
            "prompt_decompose",
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
        }
        updates = parse_output(action, raw, None, state_stub)
        assert len(updates["subtasks"]) == 1

    def test_code_action_passing(self) -> None:
        action = Action("code", "code", "prompt_code", "", None, True, "task_a")
        fb = Feedback(stdout="ok", stderr="", exit_code=0)
        state_stub: dict = {"code_results": {}, "code_passed": {}, "retries": {}}
        updates = parse_output(action, "```python\nprint(1)\n```", fb, state_stub)
        assert updates["code_passed"]["task_a"] is True
        assert "print(1)" in updates["code_results"]["task_a"]

    def test_code_retry_increments_retries(self) -> None:
        action = Action(
            "code_retry", "code_retry", "prompt_code_retry", "", None, True, "task_a"
        )
        fb = Feedback(stdout="", stderr="err", exit_code=1)
        state_stub: dict = {
            "code_results": {},
            "code_passed": {},
            "retries": {"task_a": 1},
        }
        updates = parse_output(action, "```python\npass\n```", fb, state_stub)
        assert updates["retries"]["task_a"] == 2

    def test_diagnose_action(self) -> None:
        action = Action(
            "diagnose", "diagnose", "prompt_diagnose", "", DiagnoseResult, False, None
        )
        raw = '{"fix_guidance": "fix the bug"}'
        updates = parse_output(action, raw, None, {})
        assert updates["diagnosis"] == "fix the bug"
