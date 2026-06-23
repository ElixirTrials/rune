"""Scoring for RepoBench-style next-line completion (issue #52 long-context probe).

Pure functions (CPU, no GPU imports) so they unit-test fast and stay importable in
CPU-only CI. The metric of record for the cross-file-context-as-adapter probe is
gold cross-file identifier recovery; exact-match and edit-similarity mirror
RepoBench's native metrics for comparability to the leaderboard.
"""

from __future__ import annotations

import difflib
import keyword
import re
from collections import Counter

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_KEYWORDS = frozenset(keyword.kwlist) | frozenset(keyword.softkwlist)


def extract_identifiers(line: str) -> list[str]:
    """Identifier tokens in ``line`` (Python keywords removed, order preserved)."""
    return [t for t in _IDENT_RE.findall(line) if t not in _KEYWORDS]


def gold_id_recovery(pred_line: str, gold_identifier: str) -> bool:
    """True iff ``gold_identifier`` appears as a whole token in ``pred_line``.

    Whole-token (not substring) so ``tool`` does not match ``tooltip``.
    """
    if not gold_identifier:
        return False
    return gold_identifier in set(extract_identifiers(pred_line))


def identifier_f1(pred_line: str, gold_line: str) -> float:
    """Multiset F1 over identifier tokens of prediction vs gold line."""
    pred = Counter(extract_identifiers(pred_line))
    gold = Counter(extract_identifiers(gold_line))
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    overlap = sum((pred & gold).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(pred.values())
    recall = overlap / sum(gold.values())
    return 2 * precision * recall / (precision + recall)


def exact_match(pred_line: str, gold_line: str) -> bool:
    """RepoBench-style exact match: compare stripped lines."""
    return pred_line.strip() == gold_line.strip()


def edit_similarity(pred_line: str, gold_line: str) -> float:
    """Edit similarity in [0, 1] (difflib ratio; proxy for RepoBench's fuzz.ratio)."""
    return difflib.SequenceMatcher(None, pred_line.strip(), gold_line.strip()).ratio()
