"""Contrastive / specificity-forcing primitives for D2L training (issue #49).

The diff-masked KL objective alone can be satisfied by a GENERIC "make code-review
edits more likely" adapter (measured: matched ~= mismatched edit-local logprob).
To force adapter-as-MEMORY, add a contrastive term: for the same target edit, the
MATCHED adapter (row's real context) must beat a HARD-NEGATIVE adapter on the
edit-local gold tokens by a margin.

Hard negative (reviewer): keep the prompt scaffold/distribution and change only the
CONTENT of the ``## Review Feedback`` section — either swap in ANOTHER row's
feedback (distribution-matched, realistic, wrong content) or a neutral placeholder.
Dropping the whole section is too weak — the model could win by detecting
"has feedback vs not" instead of binding the specific trajectory fact.

Pure (CPU-testable) helpers; the GPU loop wiring lives in hypernet_distill.
"""
from __future__ import annotations

import difflib
from typing import Any

_FEEDBACK_HEADER = "## Review Feedback"
_NEUTRAL_FEEDBACK = "Please review the code and apply the necessary corrections."


def extract_review_feedback(activation_text: str) -> str:
    """The Review Feedback body (everything after the header), stripped. '' if none."""
    idx = activation_text.find(_FEEDBACK_HEADER)
    if idx == -1:
        return ""
    return activation_text[idx + len(_FEEDBACK_HEADER):].strip()


def has_feedback(activation_text: str) -> bool:
    return len(extract_review_feedback(activation_text)) > 0


def make_hard_negative(activation_text: str, other_feedback: str | None = None) -> str:
    """Hard-negative context: same Task/Current Code, feedback CONTENT replaced.

    Replaces only the ``## Review Feedback`` body — with ``other_feedback`` (a
    different row's real feedback; distribution-matched, wrong content) when given,
    else a neutral placeholder. The scaffold (Task, Current Code, header) is
    preserved so the model cannot win by "has feedback vs not"; it must use the
    SPECIFIC feedback content. If there is no feedback header, returns text unchanged.
    """
    idx = activation_text.find(_FEEDBACK_HEADER)
    if idx == -1:
        return activation_text
    head_end = idx + len(_FEEDBACK_HEADER)
    replacement = (other_feedback or "").strip() or _NEUTRAL_FEEDBACK
    return activation_text[:head_end] + "\n" + replacement


def edit_local_mask(tok: Any, pre_code: str, ans_ids: list[int]) -> list[bool]:
    """Canonical edit-local token mask: answer positions inside insert/replace blocks
    of difflib(pre_code_tokens, ans_ids). SHARED by contrastive training and the
    specificity gate so the contrast and eval use the identical span (reviewer).
    Whole-span (all True) when pre_code is empty.
    """
    if not pre_code:
        return [True] * len(ans_ids)
    pre = tok(pre_code, add_special_tokens=False)["input_ids"]
    mask = [False] * len(ans_ids)
    sm = difflib.SequenceMatcher(a=pre, b=ans_ids, autojunk=False)
    for op, _i1, _i2, j1, j2 in sm.get_opcodes():
        if op in ("insert", "replace"):
            for j in range(j1, j2):
                mask[j] = True
    return mask


def contrastive_margin_loss(lp_matched: Any, lp_neg: Any, margin: float) -> Any:
    """Hinge: the matched adapter must beat the hard-negative on each gold token.

    Args:
        lp_matched: [N] gold-token logprobs under the MATCHED adapter (edit-local).
        lp_neg:     [N] gold-token logprobs under the HARD-NEGATIVE adapter.
        margin: required logprob advantage of matched over negative.

    Returns mean ``relu(margin - (lp_matched - lp_neg))`` — 0 when matched beats
    negative by >= margin everywhere; positive when the negative is as good/better
    (adapter not using the trajectory). 0 for an empty span.
    """
    import torch  # noqa: PLC0415

    if lp_matched.numel() == 0:
        return lp_matched.sum() * 0.0
    return torch.clamp(margin - (lp_matched - lp_neg), min=0.0).mean()
