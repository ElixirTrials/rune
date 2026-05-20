"""Tests for code preservation evaluations."""

import pytest

from rune_agent.code_preservation import (
    compute_identifier_recall,
    compute_import_preservation,
    compute_regression_reintroduction,
    compute_signature_consistency,
)


def test_identifier_recall_perfect():
    prev_code = "def compute_mean(data_vals):\n    running_total = accumulate(data_vals)\n    return running_total"
    curr_output = "data_vals = [1, 2]\nrunning_total = compute_mean(data_vals)\naccumulate(running_total)"
    recall = compute_identifier_recall(prev_code, curr_output)
    assert recall == pytest.approx(1.0)


def test_identifier_recall_partial():
    prev_code = "def compute_mean(values):\n    total = sum(values)\n    return total / len(values)"
    curr_output = "result = compute_mean([1, 2])"
    recall = compute_identifier_recall(prev_code, curr_output)
    assert 0.0 < recall < 1.0


def test_identifier_recall_empty():
    assert compute_identifier_recall("", "anything") == 1.0
    assert compute_identifier_recall("def foo(): pass", "") == 0.0


def test_signature_consistency_perfect():
    interfaces = "def foo(a, b)\ndef bar(x)"
    score = compute_signature_consistency(interfaces, interfaces)
    assert score == 1.0


def test_signature_consistency_partial():
    prev = "def foo(a, b)\ndef bar(x)"
    curr = "def foo(a, b)\ndef baz(y)"
    score = compute_signature_consistency(prev, curr)
    assert score == pytest.approx(0.5)


def test_import_preservation_perfect():
    imports = "import os\nfrom pathlib import Path"
    code = "import os\nfrom pathlib import Path\ndef main(): pass"
    score = compute_import_preservation(imports, code)
    assert score == 1.0


def test_import_preservation_lost():
    imports = "import os\nimport sys"
    code = "import os\ndef main(): pass"
    score = compute_import_preservation(imports, code)
    assert score == pytest.approx(0.5)


def test_regression_reintroduction_none():
    score = compute_regression_reintroduction(["test_foo"], ["test_bar"])
    assert score == 1.0


def test_regression_reintroduction_one():
    score = compute_regression_reintroduction(["test_foo", "test_bar"], ["test_foo"])
    assert score == pytest.approx(0.5)


def test_regression_reintroduction_empty():
    assert compute_regression_reintroduction([], ["test_foo"]) == 1.0
