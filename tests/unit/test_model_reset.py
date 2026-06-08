"""Benchmark runs must not bleed adapters across tasks."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from rune.model.wrapper import ModelWrapper


def test_reset_adapter_zeroes_lora_b() -> None:
    wrapper = ModelWrapper.__new__(ModelWrapper)
    wrapper._base_model = MagicMock()
    fake_sd = {"base.lora_B.weight": object(), "base.lora_A.weight": object()}

    with (
        patch("peft.get_peft_model_state_dict", return_value=fake_sd),
        patch("rune.model.wrapper.hotswap_adapter_fn") as swap,
        patch("rune.model.wrapper.scale_lora_b", side_effect=lambda sd, f: {k: f for k in sd}),
    ):
        wrapper.reset_adapter()
        swap.assert_called_once()
        applied = swap.call_args[0][1]
        assert all(v == 0.0 for v in applied.values())
