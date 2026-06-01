"""Unit tests for the episode serialization-contract fingerprint."""

from __future__ import annotations

import string

from rune.training.serialization_contract import episode_serialization_fingerprint

_TRAIN = "TRAIN: {goal}\n{facts}\n{code}"
_INFER = "INFER: {goal}\n{code}"
_EPISODE = "goal=sort a list; facts=index error; code=def f(x): return sorted(x)"


def test_fingerprint_is_stable() -> None:
    a = episode_serialization_fingerprint(_TRAIN, _INFER, _EPISODE)
    b = episode_serialization_fingerprint(_TRAIN, _INFER, _EPISODE)
    assert a == b


def test_fingerprint_is_64_char_hex() -> None:
    h = episode_serialization_fingerprint(_TRAIN, _INFER, _EPISODE)
    assert len(h) == 64
    assert all(c in string.hexdigits for c in h)


def test_fingerprint_sensitive_to_infer_template() -> None:
    base = episode_serialization_fingerprint(_TRAIN, _INFER, _EPISODE)
    changed = episode_serialization_fingerprint(_TRAIN, _INFER + " EXTRA", _EPISODE)
    assert base != changed
