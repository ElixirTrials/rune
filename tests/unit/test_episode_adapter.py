"""Episodic adapter conditioning: the right context per step (not the full spec)."""

from __future__ import annotations

from rune.engine.graph import render_episode_adapter
from rune.engine.parse import render_template
from rune.engine.state import Feedback, Subtask


class TestEpisodicPromptReferences:
    def test_repair_prompt_cues_recall_not_reading(self) -> None:
        # (1): the adapter episode is in the WEIGHTS, not the model's context.
        # The prompt must cue RECALL of learned knowledge about the specific
        # function, never tell the model to "read" a generic section.
        p = render_template("prompt_episodic_repair", subtask_name="decode_string")
        low = p.lower()
        assert "decode_string" in p  # specific function, not generic
        assert "recall" in low  # recall-of-learned-knowledge framing
        assert "## review feedback" not in low  # no generic header references
        assert "## current code" not in low
        assert "in your context" not in low  # adapter is NOT in the context
        assert "read `##" not in low

    def test_code_prompt_cues_recall(self) -> None:
        p = render_template("prompt_episodic_code", subtask_name="decode_string")
        assert "decode_string" in p
        assert "recall" in p.lower()
        assert "in your context" not in p.lower()

    def test_repair_adapter_uses_specific_named_headers(self) -> None:
        # The adapter conditioning must use distinctive, function-named headers
        # (a sharp recall anchor) rather than generic "## Review Feedback".
        st = _state()
        st["diagnosis"] = {"tokenize": "drop the delimiter"}
        adp = render_episode_adapter("code", "tokenize", st)
        assert "## Mission `tokenize`" in adp
        assert "`tokenize` — what you learned was wrong with it" in adp
        assert "## Review Feedback" not in adp  # generic header gone
        assert "## Current Code" not in adp


def _state() -> dict:
    return {
        "task": '"""Implement calculate(expr) -> int. >>> assert calculate("2+3")==5"""',
        "entry_point": "calculate",
        "overall_goal": "Evaluate an arithmetic expression string to an int.",
        "subtasks": [
            Subtask(
                "tokenize",
                "Split the expression into tokens",
                [],
                "assert tokenize('2+3')==['2','+','3']",
                "calculate",
            ),
            Subtask(
                "evaluate",
                "Evaluate the token list",
                ["tokenize"],
                "assert evaluate(['2','+','3'])==5",
                "calculate",
            ),
        ],
        "code_results": {"tokenize": "def tokenize(s): return list(s)"},
        "feedback": {
            "tokenize": Feedback(
                stdout="", stderr="AssertionError: bad split", exit_code=1
            )
        },
        "diagnosis": {},
        "integration_feedback": None,
    }


class TestEpisodeAdapter:
    def test_code_step_is_focused_on_the_subgoal(self) -> None:
        adp = render_episode_adapter("code", "tokenize", _state())
        # focused: the current sub-goal + acceptance, the overall goal, the local state
        assert "tokenize" in adp
        assert "Split the expression into tokens" in adp
        assert "assert tokenize('2+3')" in adp  # the sub-goal's acceptance check
        assert "Evaluate an arithmetic expression" in adp  # condensed overall goal
        # NOT the full original spec / the OTHER subtask
        assert ">>> assert calculate" not in adp
        assert "Evaluate the token list" not in adp

    def test_code_step_carries_local_code_and_error(self) -> None:
        adp = render_episode_adapter("code", "tokenize", _state())
        assert "def tokenize(s): return list(s)" in adp  # ## Current Code
        assert "AssertionError: bad split" in adp  # ## Review Feedback

    def test_integration_step_carries_all_subtasks(self) -> None:
        st = _state()
        st["code_results"] = {
            "tokenize": "def tokenize(s): ...",
            "evaluate": "def evaluate(t): ...",
        }
        adp = render_episode_adapter("integrate", None, st)
        assert "calculate" in adp  # the entry_point to integrate into
        assert "def tokenize(s): ..." in adp  # ALL subtasks' code
        assert "def evaluate(t): ..." in adp

    def test_decompose_step_sees_the_full_spec(self) -> None:
        adp = render_episode_adapter("decompose", None, _state())
        assert ">>> assert calculate" in adp  # decompose needs the full task

    def test_entry_subtask_first_attempt_seeds_bare_signature(self) -> None:
        # R2: the adapter must carry the real call contract (params minus self)
        # on the FIRST code attempt, or the model invents parameter names.
        st = _state()
        st["entry_point"] = "maxDistance"
        st["signature"] = (
            "class Solution:\n    def maxDistance(self, s: str, k: int) -> int:\n        "
        )
        st["subtasks"] = [
            Subtask("maxDistance", "Max distance", [], "assert maxDistance('NS', 1)==2", "maxDistance")
        ]
        st["code_results"] = {}
        st["feedback"] = {}
        adp = render_episode_adapter("code", "maxDistance", st)
        assert "def maxDistance(s: str, k: int)" in adp  # real params, no self
        assert "self" not in adp

    def test_repair_adapter_colocates_error_and_diagnosis(self) -> None:
        # (1): the repair episode must carry BOTH what went wrong (traceback) AND
        # the diagnosis together in ## Review Feedback (within-distribution).
        st = _state()
        st["diagnosis"] = {"tokenize": "the split keeps the delimiter; drop it"}
        adp = render_episode_adapter("code", "tokenize", st)
        assert "AssertionError: bad split" in adp  # the error
        assert "Diagnosis: the split keeps the delimiter" in adp  # the diagnosis

    def test_existing_code_overrides_signature_seed(self) -> None:
        # once code exists, condition on it (not the bare stub).
        st = _state()
        st["entry_point"] = "maxDistance"
        st["signature"] = "class Solution:\n    def maxDistance(self, s, k):\n        "
        st["subtasks"] = [
            Subtask("maxDistance", "Max distance", [], "assert maxDistance('NS', 1)==2", "maxDistance")
        ]
        st["code_results"] = {"maxDistance": "def maxDistance(s, k): return 0"}
        st["feedback"] = {}
        adp = render_episode_adapter("code", "maxDistance", st)
        assert "return 0" in adp
