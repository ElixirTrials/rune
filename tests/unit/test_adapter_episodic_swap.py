"""Episodic adapter invariant: fresh hypernet LoRA every step, never static reuse."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import torch

from rune.engine.graph import render_episode_adapter, step_node
from rune.engine.state import Action, Feedback, Subtask, make_initial_state
from rune.model.adapter import AdapterResult, apply_episodic_adapter, scale_lora_b
from rune.model.inference import GenerationResult


def _adapter_result(adapter_id: str = "a1") -> AdapterResult:
    return AdapterResult(
        adapter_id=adapter_id,
        state_dict={
            "base.lora_A.weight": torch.ones(2, 2),
            "base.lora_B.weight": torch.ones(2, 2),
        },
    )


class TestApplyEpisodicAdapter:
    def test_resets_before_generate_and_hotswap(self) -> None:
        model = MagicMock()
        model.generate_adapter.return_value = _adapter_result("fresh")
        model.reset_adapter = MagicMock()
        model.hotswap_adapter = MagicMock()

        aid = apply_episodic_adapter(model, "episode A", scaling=0.627)

        assert aid == "fresh"
        model.reset_adapter.assert_called_once()
        model.generate_adapter.assert_called_once_with("episode A")
        model.hotswap_adapter.assert_called_once()
        swapped = model.hotswap_adapter.call_args[0][0]
        assert swapped["base.lora_B.weight"].eq(torch.ones(2, 2) * 0.627).all()

    def test_different_trajectories_produce_separate_generate_calls(self) -> None:
        model = MagicMock()
        model.generate_adapter.side_effect = [
            _adapter_result("ep1"),
            _adapter_result("ep2"),
        ]
        apply_episodic_adapter(model, "step-1 ctx", scaling=1.0)
        apply_episodic_adapter(model, "step-2 ctx", scaling=1.0)
        assert model.generate_adapter.call_args_list[0][0][0] == "step-1 ctx"
        assert model.generate_adapter.call_args_list[1][0][0] == "step-2 ctx"


class TestStepNodeAdapterSwap:
    def _model(self, *, text: str = "def f(): pass") -> MagicMock:
        model = MagicMock()
        model.generate_adapter.side_effect = lambda _t: _adapter_result()
        model.reset_adapter = MagicMock()
        model.hotswap_adapter = MagicMock()
        model.generate = AsyncMock(
            return_value=GenerationResult(
                text=text, thinking="", tokens_used=1, truncated=False
            )
        )
        model.count_tokens = MagicMock(return_value=1)
        return model

    def test_one_adapter_swap_per_action_before_generate(self) -> None:
        state = make_initial_state("task", 5, "fn", "", "")
        state["subtasks"] = [Subtask("fn", "do fn", [], "assert fn() == 1", "fn")]
        state["plans"] = {"fn": "plan"}
        state["actions"] = []
        model = self._model()
        config = {
            "configurable": {
                "model": model,
                "run_config": {"prompt_mode": "episodic", "model_judge": False},
            }
        }
        with patch("rune.engine.graph.run_in_sandbox") as sandbox:
            sandbox.return_value = MagicMock(stdout="", stderr="", exit_code=0)
            asyncio.run(step_node(state, config))

        assert model.reset_adapter.call_count == model.generate_adapter.call_count
        assert model.generate_adapter.call_count == model.generate.call_count == 1
        assert model.hotswap_adapter.call_count == 1

    def test_repair_step_uses_new_trajectory_not_prior_code_step(self) -> None:
        base = make_initial_state(
            "freq task",
            8,
            "maxDifference",
            "class Solution:\n    def maxDifference(self, s: str) -> int:\n        ",
            "assert maxDifference('a') == 1",
        )
        base["subtasks"] = [
            Subtask(
                "maxDifference",
                "odd/even freq",
                [],
                "assert maxDifference('a') == 1",
                "maxDifference",
            )
        ]
        base["plans"] = {"maxDifference": "plan"}
        base["code_results"] = {"maxDifference": "def maxDifference(s): return 0"}
        base["feedback"] = {
            "maxDifference": Feedback(
                stdout="", stderr="AssertionError: bad", exit_code=1
            )
        }
        base["diagnosis"] = {"maxDifference": "fix parity"}
        base["actions"] = []
        trajectories: list[str] = []
        model = self._model(text="def maxDifference(s): return 1")

        def _capture(traj: str) -> AdapterResult:
            trajectories.append(traj)
            return _adapter_result(f"a{len(trajectories)}")

        model.generate_adapter.side_effect = _capture

        repair = Action(
            "repair",
            "code_repair",
            "prompt_episodic_repair",
            "",
            None,
            True,
            "maxDifference",
        )
        config = {
            "configurable": {
                "model": model,
                "run_config": {"prompt_mode": "episodic", "model_judge": False},
            }
        }
        with (
            patch("rune.engine.graph.select_action", return_value=[repair]),
            patch("rune.engine.graph.run_in_sandbox") as sandbox,
        ):
            sandbox.return_value = MagicMock(stdout="", stderr="", exit_code=1)
            asyncio.run(step_node(base, config))

        assert len(trajectories) == 1
        assert "AssertionError: bad" in trajectories[0]
        assert "maxDifference" in trajectories[0]

    def test_continuation_regenerates_adapter_each_round(self) -> None:
        state = make_initial_state("task", 5, "", "", "")
        state["subtasks"] = [Subtask("_main", "d", [], "", "_main")]
        state["plans"] = {"_main": "p"}
        state["actions"] = []
        model = self._model(text="def f():\n    pass")
        model.generate = AsyncMock(
            return_value=GenerationResult(
                text="def f():\n    # trunc",
                thinking="",
                tokens_used=1,
                truncated=True,
            )
        )
        model.generate_continuation = AsyncMock(
            return_value=GenerationResult(
                text="\n    return 1\n",
                thinking="",
                tokens_used=1,
                truncated=False,
            )
        )
        config = {
            "configurable": {
                "model": model,
                "run_config": {
                    "prompt_mode": "full",
                    "model_judge": False,
                    "cont_budget": 2,
                },
            }
        }
        with patch("rune.engine.graph.run_in_sandbox") as sandbox:
            sandbox.return_value = MagicMock(stdout="", stderr="", exit_code=0)
            asyncio.run(step_node(state, config))

        # initial action + one continuation round
        assert model.generate_adapter.call_count == 2
        assert model.reset_adapter.call_count == 2

    def test_model_judge_does_not_regenerate_adapter(self) -> None:
        """Judge intentionally reuses the code-step adapter; not an extra episode."""
        state = make_initial_state("task", 5, "fn", "", "assert fn() == 1")
        state["subtasks"] = [Subtask("fn", "d", [], "assert fn() == 1", "fn")]
        state["plans"] = {"fn": "p"}
        state["actions"] = []
        model = self._model(text="def fn(): return 1")
        code_action = Action("code", "code", "prompt_code", "", None, True, "fn")
        judge_result = GenerationResult(
            text='{"reason":"","failing_input":"","correct":true}',
            thinking="",
            tokens_used=1,
            truncated=False,
        )
        model.generate = AsyncMock(
            side_effect=[
                GenerationResult(
                    text="def fn(): return 1",
                    thinking="",
                    tokens_used=1,
                    truncated=False,
                ),
                judge_result,
            ]
        )
        config = {
            "configurable": {
                "model": model,
                "run_config": {"prompt_mode": "episodic", "model_judge": True},
            }
        }
        with (
            patch("rune.engine.graph.select_action", return_value=[code_action]),
            patch("rune.engine.graph.run_in_sandbox") as sandbox,
        ):
            sandbox.return_value = MagicMock(stdout="", stderr="", exit_code=0)
            asyncio.run(step_node(state, config))

        assert model.generate_adapter.call_count == 1
        assert model.generate.call_count == 2


class TestEpisodeAdapterTrajectory:
    def test_trajectory_changes_when_feedback_arrives(self) -> None:
        st = make_initial_state("task", 5, "fn", "", "")
        st["subtasks"] = [Subtask("fn", "goal", [], "assert fn() == 1", "fn")]
        before = render_episode_adapter("code", "fn", st)
        st["feedback"] = {
            "fn": Feedback(stdout="", stderr="AssertionError: nope", exit_code=1)
        }
        after = render_episode_adapter("code", "fn", st)
        assert before != after
        assert "AssertionError: nope" in after


class TestScaleLoraB:
    def test_zero_scaling_zeros_only_lora_b(self) -> None:
        sd = {
            "layer.lora_A.weight": torch.tensor([[2.0]]),
            "layer.lora_B.weight": torch.tensor([[3.0]]),
        }
        out = scale_lora_b(sd, 0.0)
        assert out["layer.lora_A.weight"].item() == 2.0
        assert out["layer.lora_B.weight"].item() == 0.0
