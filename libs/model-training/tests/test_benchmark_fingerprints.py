"""Tests for benchmark fingerprint normalization."""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
sys.path.insert(0, str(_SCRIPTS))

import build_benchmark_fingerprints as bf  # noqa: E402


def test_fingerprint_normalizes_whitespace() -> None:
    a = bf.fingerprint("def  foo(x):\n  return  x")
    b = bf.fingerprint("def foo(x):\n    return x")
    assert a == b


def test_fingerprint_normalizes_quotes() -> None:
    a = bf.fingerprint("print('hello')")
    b = bf.fingerprint('print("hello")')
    assert a == b


def test_fingerprint_distinguishes_different_functions() -> None:
    assert bf.fingerprint("def foo(): pass") != bf.fingerprint("def bar(): pass")
