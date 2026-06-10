"""Contrastive / specificity primitives (issue #49)."""

import torch

from rune.training.contrastive import (
    contrastive_margin_loss,
    edit_local_mask,
    extract_review_feedback,
    has_feedback,
    make_hard_negative,
)

_CTX = (
    "## Task\nReview and revise.\n"
    "## Current Code\ndef f(): return any(x)\n"
    "## Review Feedback\nUse all() not any() so every element is checked."
)


def test_extract_review_feedback() -> None:
    assert (
        extract_review_feedback(_CTX)
        == "Use all() not any() so every element is checked."
    )
    assert extract_review_feedback("## Task\nx") == ""
    assert has_feedback(_CTX) and not has_feedback("## Task\nx")
    # new recall format (issue #52): feedback under the function-named header
    recall = (
        "## Mission `f`\ng\n\n## `f` — your last attempt\ndef f(): ...\n\n"
        "## `f` — what you learned was wrong with it\nf(1) -> 0, want 1"
    )
    assert extract_review_feedback(recall) == "f(1) -> 0, want 1"
    assert has_feedback(recall)


def test_make_hard_negative_swaps_feedback_keeps_scaffold() -> None:
    neg = make_hard_negative(_CTX, other_feedback="Rename the variable for clarity.")
    # scaffold preserved (Task + Current Code + header), only feedback content changed
    assert "## Task" in neg and "## Current Code" in neg and "## Review Feedback" in neg
    assert "Rename the variable for clarity." in neg
    assert "Use all() not any()" not in neg  # the discriminative fact is gone


def test_make_hard_negative_placeholder_when_no_other() -> None:
    neg = make_hard_negative(_CTX)
    assert "## Review Feedback" in neg
    assert "Use all() not any()" not in neg
    assert len(neg) > 0


def test_make_hard_negative_noop_without_header() -> None:
    assert make_hard_negative("## Task\nx") == "## Task\nx"


class _FakeTok:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [hash(t) % 997 for t in text.split()]}


def test_edit_local_mask_marks_changed_tokens() -> None:
    tok = _FakeTok()
    pre = "a b c d"
    ans_ids = tok("a b X d")["input_ids"]  # position 2 changed
    mask = edit_local_mask(tok, pre, ans_ids)
    assert mask[2] is True and mask[0] is False


def test_edit_local_mask_wholespan_without_pre() -> None:
    assert edit_local_mask(_FakeTok(), "", [1, 2, 3]) == [True, True, True]


def test_contrastive_margin_loss_zero_when_matched_beats_neg() -> None:
    lp_m = torch.tensor([-1.0, -0.5, -2.0])
    lp_n = torch.tensor([-3.0, -2.5, -4.0])  # matched beats neg by 2.0 each
    assert float(contrastive_margin_loss(lp_m, lp_n, margin=0.5)) == 0.0


def test_contrastive_margin_loss_positive_when_neg_competitive() -> None:
    lp_m = torch.tensor([-2.0, -2.0])
    lp_n = torch.tensor([-2.0, -1.0])  # neg as good / better -> penalty
    loss = contrastive_margin_loss(lp_m, lp_n, margin=0.5)
    # pos0: relu(0.5-0)=0.5 ; pos1: relu(0.5-(-1.0))=1.5 ; mean=1.0
    assert abs(float(loss) - 1.0) < 1e-5


def test_contrastive_margin_loss_empty() -> None:
    assert (
        float(contrastive_margin_loss(torch.tensor([]), torch.tensor([]), 0.5)) == 0.0
    )
