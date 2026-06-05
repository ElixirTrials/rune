"""Robust structured parsing for decompose/plan/diagnose + episodic decompose schema.

The JSON-structured decompose/plan/diagnose steps truncated on long hard-task output
-> pydantic validation failed -> re-plan/re-decompose loop -> empty code (3/4 empty on
the LiveCodeBench probe). These tests pin: (1) json_repair recovers truncated/garbled
JSON; (2) the episodic decompose schema (overall_goal, acceptance_check, builds);
(3) the <=3 cap; (4) graceful degrade to a single whole-task subtask instead of an
empty re-decompose loop.
"""

from __future__ import annotations

from rune.engine.oracle import build_subtask_probe, split_acceptance_checks
from rune.engine.parse import (
    DecomposeResult,
    SubtaskSchema,
    _loads_structured,
    parse_output,
)
from rune.engine.state import Action


def _decompose_action() -> Action:
    return Action(
        "decompose",
        "decompose",
        "prompt_decompose_concise",
        "",
        DecomposeResult,
        False,
        None,
    )


def _state(task: str = "Write add(a,b) returning a+b.") -> dict:
    return {"subtasks": [], "task": task, "entry_point": "add"}


class TestLoadsStructured:
    def test_valid_json(self) -> None:
        raw = '{"subtasks": [{"name": "a", "description": "do a", "depends_on": []}]}'
        res = _loads_structured(raw, DecomposeResult)
        assert res is not None
        assert res.subtasks[0].name == "a"

    def test_truncated_json_recovered(self) -> None:
        # truncated mid-object (no closing braces) -> json_repair recovers it
        raw = '{"subtasks": [{"name": "core", "description": "the whole thing"'
        res = _loads_structured(raw, DecomposeResult)
        assert res is not None
        assert res.subtasks and res.subtasks[0].name == "core"

    def test_unrecoverable_returns_none(self) -> None:
        assert _loads_structured("total garbage no json here", DecomposeResult) is None


class TestEpisodicSchema:
    def test_subtask_new_fields_default(self) -> None:
        s = SubtaskSchema(name="a", description="d")
        assert s.acceptance_check == ""
        assert s.builds == ""

    def test_subtask_accepts_episodic_fields(self) -> None:
        s = SubtaskSchema(
            name="tok",
            description="tokenizer",
            acceptance_check="assert tokenize('1+2') == ['1','+','2']",
            builds="calculate",
        )
        assert s.acceptance_check.startswith("assert")
        assert s.builds == "calculate"

    def test_subtask_normalizes_list_of_checks(self) -> None:
        s = SubtaskSchema(
            name="solve_p3",
            description="roman",
            acceptance_check=[
                "assert solve_p3(9) == 'IX'",
                "assert solve_p3(3) == 'III'",
            ],
        )
        lines = s.acceptance_check.splitlines()
        assert len(lines) == 2
        assert "solve_p3(9)" in lines[0]
        assert "solve_p3(3)" in lines[1]

    def test_decompose_overall_goal_default(self) -> None:
        d = DecomposeResult(subtasks=[SubtaskSchema(name="a", description="d")])
        assert d.overall_goal == ""


class TestDecomposeBoundAndDegrade:
    def test_caps_to_three_subtasks(self) -> None:
        raw = (
            '{"subtasks": ['
            '{"name":"a","description":"da"},{"name":"b","description":"db"},'
            '{"name":"c","description":"dc"},{"name":"d","description":"dd"},'
            '{"name":"e","description":"de"}]}'
        )
        updates = parse_output(_decompose_action(), raw, None, _state())
        assert len(updates["subtasks"]) <= 3

    def test_garbled_degrades_to_single_whole_task(self) -> None:
        # unparseable decompose must NOT return {} (which re-decompose-loops);
        # it degrades to ONE subtask = the whole task.
        updates = parse_output(_decompose_action(), "??? not json ???", None, _state())
        assert len(updates.get("subtasks", [])) == 1


class TestDecomposeNameConsistency:
    def test_single_subtask_name_forced_to_entry_point(self) -> None:
        # The lone subtask of a single-function task MUST be named for the
        # entry_point, so the prompt/check/held-out test all call the same name
        # (the model otherwise names it after the descriptive subtask -> NameError).
        raw = (
            '{"overall_goal":"roman","subtasks":[{"name":'
            '"Convert integer to Roman numeral","description":"d",'
            '"acceptance_check":"assert solve_p3(9)==\'IX\'","builds":"solve_p3"}]}'
        )
        st = {"subtasks": [], "task": "spec", "entry_point": "solve_p3"}
        updates = parse_output(_decompose_action(), raw, None, st)
        assert len(updates["subtasks"]) == 1
        assert updates["subtasks"][0].name == "solve_p3"
        assert updates["subtasks"][0].builds == "solve_p3"


class TestDecomposeNormalization:
    def test_duplicate_names_deduped_to_single(self) -> None:
        # model emits the entry_point name 3x -> collapse to ONE subtask
        raw = (
            '{"overall_goal":"g","subtasks":['
            '{"name":"decode_string","description":"d","builds":"decode_string"},'
            '{"name":"decode_string","description":"d","builds":"decode_string"},'
            '{"name":"decode_string","description":"d","builds":"decode_string"}]}'
        )
        st = {"subtasks": [], "task": "t", "entry_point": "decode_string"}
        updates = parse_output(_decompose_action(), raw, None, st)
        assert len(updates["subtasks"]) == 1
        assert updates["subtasks"][0].name == "decode_string"

    def test_entry_point_among_helpers_collapses_to_entry_point(self) -> None:
        # entry_point listed AS a subtask alongside helpers -> it IS the whole task
        raw = (
            '{"overall_goal":"g","subtasks":['
            '{"name":"calc","description":"whole","builds":"calc"},'
            '{"name":"tokenize","description":"helper","builds":"calc"}]}'
        )
        st = {"subtasks": [], "task": "t", "entry_point": "calc"}
        updates = parse_output(_decompose_action(), raw, None, st)
        assert [s.name for s in updates["subtasks"]] == ["calc"]


class TestNameFromCheck:
    def test_helper_name_derived_from_its_check(self) -> None:
        # multi-subtask: a helper's name comes from the function its check calls,
        # NOT the descriptive phrase the model wrote in `name`.
        raw = (
            '{"overall_goal":"g","subtasks":['
            '{"name":"Split the expression into tokens","description":"d",'
            '"acceptance_check":"assert tokenize(\'2+3\')==[\'2\',\'+\',\'3\']","builds":"calc"},'
            '{"name":"Evaluate the token list","description":"d",'
            '"acceptance_check":"assert evaluate([\'2\',\'+\',\'3\'])==5","builds":"calc"}]}'
        )
        st = {"subtasks": [], "task": "t", "entry_point": "calc"}
        updates = parse_output(_decompose_action(), raw, None, st)
        names = sorted(s.name for s in updates["subtasks"])
        assert names == ["evaluate", "tokenize"]  # identifiers from the checks

    def test_descriptive_name_with_entrypoint_check_becomes_entrypoint(self) -> None:
        raw = (
            '{"overall_goal":"g","subtasks":[{"name":"Convert to Roman",'
            '"description":"d","acceptance_check":"assert solve_p3(9)==\'IX\'",'
            '"builds":"solve_p3"}]}'
        )
        st = {"subtasks": [], "task": "t", "entry_point": "solve_p3"}
        updates = parse_output(_decompose_action(), raw, None, st)
        assert updates["subtasks"][0].name == "solve_p3"

    def test_decompose_accepts_list_acceptance_check(self) -> None:
        raw = (
            '{"overall_goal":"g","subtasks":[{"name":"solve_p3","description":"d",'
            '"acceptance_check":["assert solve_p3(9)==\'IX\'",'
            '"assert solve_p3(3)==\'III\'"],"builds":"solve_p3"}]}'
        )
        st = {"subtasks": [], "task": "t", "entry_point": "solve_p3"}
        updates = parse_output(_decompose_action(), raw, None, st)
        check = updates["subtasks"][0].acceptance_check
        assert "solve_p3(9)" in check and "solve_p3(3)" in check

    def test_check_rewritten_to_call_the_forced_name(self) -> None:
        # The latent bug: forcing name->entry_point but leaving the check calling
        # the OLD name makes the in-loop probe run a call to an undefined function
        # -> spurious NameError. The check must be rewritten to call the final name.
        raw = (
            '{"overall_goal":"g","subtasks":[{"name":"foo","description":"d",'
            '"acceptance_check":"assert foo(9)==\'IX\'","builds":"foo"}]}'
        )
        st = {"subtasks": [], "task": "t", "entry_point": "maxDistance"}
        updates = parse_output(_decompose_action(), raw, None, st)
        sub = updates["subtasks"][0]
        assert sub.name == "maxDistance"
        assert "maxDistance(9)" in sub.acceptance_check  # check now calls the name
        assert "foo(" not in sub.acceptance_check  # old name gone

    def test_overescaped_newline_check_splits_and_fires(self) -> None:
        # The model over-escapes a multi-assert check as ONE JSON string with
        # literal backslash-n between asserts. The AST parser decodes the escapes
        # and recovers both asserts, so the oracle FIRES instead of silently
        # no-opping to a code-only false pass (issue #52).
        s = SubtaskSchema(
            name="f",
            description="d",
            acceptance_check="assert f(1) == 1\\nassert f(2) == 2",  # literal \n
        )
        assert len(split_acceptance_checks(s.acceptance_check)) == 2
        _probe, fired = build_subtask_probe("def f(x):\n    return x", s.acceptance_check)
        assert fired  # checks actually run -- NOT a code-only fallback false-pass

    def test_separator_inside_string_literal_is_not_split(self) -> None:
        # AST-based splitting (not textual) keeps a ';' / newline inside a string
        # literal as part of the one assert -- where regex/text splitting breaks.
        checks = split_acceptance_checks("assert parse('a;b\\nc') == ['a', 'b', 'c']")
        assert len(checks) == 1

    def test_check_rewrite_preserves_builtins(self) -> None:
        # only the function-under-test is renamed; builtins in the check stay.
        raw = (
            '{"overall_goal":"g","subtasks":[{"name":"old","description":"d",'
            '"acceptance_check":"assert old([1,2]) == sorted([2,1])","builds":"old"}]}'
        )
        st = {"subtasks": [], "task": "t", "entry_point": "mySort"}
        updates = parse_output(_decompose_action(), raw, None, st)
        check = updates["subtasks"][0].acceptance_check
        assert "mySort([1, 2])" in check
        assert "sorted([2, 1])" in check  # builtin untouched
