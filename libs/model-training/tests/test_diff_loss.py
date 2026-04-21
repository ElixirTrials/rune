"""CPU tests for diff-aware loss weighting.

Tests the pure function ``compute_diff_loss_weights`` and the identity
invariant (uniform weight ⇒ vanilla SFT loss) that the collator is
expected to preserve. Trainer-level identity is deferred to a GPU-gated
integration test (outside this file) because ``SFTTrainer`` requires a
real tokenizer + model at construction time.
"""

from __future__ import annotations

import pytest
from model_training.diff_loss import (
    IGNORE_INDEX,
    compute_diff_loss_weights,
)


def test_empty_input_returns_empty_weights() -> None:
    assert compute_diff_loss_weights([], []) == []


def test_masked_positions_receive_zero() -> None:
    """Positions with labels == IGNORE_INDEX get weight 0 regardless of token id."""
    input_ids = [1, 2, 3, 4]
    labels = [IGNORE_INDEX, IGNORE_INDEX, 5, 6]
    weights = compute_diff_loss_weights(input_ids, labels)
    assert weights[0] == 0.0
    assert weights[1] == 0.0
    assert weights[2] > 0.0
    assert weights[3] > 0.0


def test_tokens_in_context_get_unchanged_weight() -> None:
    """Assistant tokens matching a context token id receive the unchanged weight."""
    # Context span: ids [10, 20] are masked.
    # Assistant span: id 10 repeats in context (unchanged),
    #                 id 99 is new (changed).
    input_ids = [10, 20, 10, 99]
    labels = [IGNORE_INDEX, IGNORE_INDEX, 10, 99]
    weights = compute_diff_loss_weights(
        input_ids, labels, changed_weight=2.0, unchanged_weight=0.5
    )
    assert weights == [0.0, 0.0, 0.5, 2.0]


def test_identity_under_uniform_weights() -> None:
    """When changed == unchanged, every non-masked position gets that weight.

    This is the regression guard for the trainer subclass: uniform weights
    must produce gradients proportional to vanilla CE (up to a constant).
    """
    input_ids = [1, 2, 3, 4, 5]
    labels = [IGNORE_INDEX, 2, 3, IGNORE_INDEX, 5]
    weights = compute_diff_loss_weights(
        input_ids, labels, changed_weight=1.0, unchanged_weight=1.0
    )
    assert weights == [0.0, 1.0, 1.0, 0.0, 1.0]


def test_length_mismatch_raises() -> None:
    """Defensive: mismatched input_ids / labels lengths must raise."""
    with pytest.raises(ValueError, match="equal length"):
        compute_diff_loss_weights([1, 2, 3], [IGNORE_INDEX, 2])


def test_all_masked_returns_all_zeros() -> None:
    """If every position is masked (no assistant turn), weights are all zero."""
    weights = compute_diff_loss_weights([1, 2, 3], [IGNORE_INDEX] * 3)
    assert weights == [0.0, 0.0, 0.0]


def test_no_context_tokens_every_assistant_token_is_changed() -> None:
    """With an empty context span, every non-masked token is 'changed'."""
    input_ids = [1, 2, 3]
    labels = [1, 2, 3]  # nothing masked → context set empty
    weights = compute_diff_loss_weights(
        input_ids, labels, changed_weight=1.0, unchanged_weight=0.25
    )
    assert weights == [1.0, 1.0, 1.0]


def test_module_is_cpu_importable() -> None:
    """diff_loss must import without torch / trl (deferred import)."""
    import importlib
    import sys

    # Fresh import; if any top-level import pulls torch, this would fail
    # in a no-torch CI. We can't simulate 'no torch' here cleanly since
    # it's already installed, but we CAN assert that importing the
    # module does not bring torch into sys.modules if torch wasn't
    # already imported.
    already_had_torch = "torch" in sys.modules
    if not already_had_torch:
        # sys.modules is unlikely to be fresh in a full test run, so we
        # only assert the invariant when torch isn't already loaded.
        importlib.reload(
            importlib.import_module("model_training.diff_loss")
        )
        assert "torch" not in sys.modules, (
            "diff_loss top-level import must not pull in torch"
        )


def test_trainer_cli_passes_diff_flag_through() -> None:
    """The --diff-aware-loss flag threads through _resolve_kwargs."""
    from model_training.trainer_cli import _build_parser, _resolve_kwargs

    parser = _build_parser()
    args = parser.parse_args(
        [
            "--dataset",
            "/tmp/x",
            "--adapter-id",
            "x",
            "--diff-aware-loss",
            "--diff-changed-weight",
            "2.5",
            "--diff-unchanged-weight",
            "0.1",
            "--dry-run",
        ]
    )
    kwargs = _resolve_kwargs(args)
    assert kwargs["diff_aware_loss"] is True
    assert kwargs["diff_changed_weight"] == 2.5
    assert kwargs["diff_unchanged_weight"] == 0.1


def test_trainer_cli_diff_aware_off_by_default() -> None:
    """Omitting --diff-aware-loss keeps it False."""
    from model_training.trainer_cli import _build_parser, _resolve_kwargs

    parser = _build_parser()
    args = parser.parse_args(
        ["--dataset", "/tmp/x", "--adapter-id", "x", "--dry-run"]
    )
    kwargs = _resolve_kwargs(args)
    assert kwargs["diff_aware_loss"] is False
    assert kwargs["diff_changed_weight"] == 1.0
    assert kwargs["diff_unchanged_weight"] == 0.3
