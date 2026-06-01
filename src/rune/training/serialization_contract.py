"""Episode serialization contract fingerprint.

A template mismatch between the train-time and inference-time episode
serialization (and the episode shape they assume) would masquerade as a recipe
failure (#52 risk: "Template mismatch ... logged as a first-class contract
artifact"). This module produces a single stable fingerprint over the three
parts so the train and inference serializations can be pinned and compared.
"""

from __future__ import annotations

import hashlib


def episode_serialization_fingerprint(
    train_template: str,
    infer_template: str,
    sample_episode: str,
) -> str:
    """Return a sha256 hex fingerprint over the serialization contract parts.

    The three parts are joined by a NUL byte before hashing. NUL is not a
    legal character in the templates or rendered episode text, so the join is
    unambiguous (it cannot collide across a boundary, e.g. ``("ab", "c")`` vs
    ``("a", "bc")``).

    Args:
        train_template: The train-time adapter-context serialization template.
        infer_template: The inference-time serialization template.
        sample_episode: A rendered sample episode used to pin the episode shape.

    Returns:
        A 64-character lowercase sha256 hex digest.
    """
    joined = "\x00".join([train_template, infer_template, sample_episode])
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()
