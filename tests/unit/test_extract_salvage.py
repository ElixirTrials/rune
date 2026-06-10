"""Tests for trailing-garbage salvage in extract_entry_function."""

from __future__ import annotations

import ast

from rune.bench.lcb import extract_entry_function


class TestSalvageEntryFunction:
    def test_recovers_function_before_trailing_garbage(self) -> None:
        blob = (
            "def minCosts(cost):\n"
            "    return [cost[0]]\n"
            "\n"
            "min_from (n-1) downto 0:\n"
            "# ramble pseudo-code that is not valid python\n"
        )
        out = extract_entry_function(blob, "minCosts")
        ast.parse(out)  # must not raise
        assert "minCosts" in out

    def test_recovers_method_from_solution_class(self) -> None:
        blob = (
            "class Solution:\n"
            "    def minCosts(self, cost):\n"
            "        return cost[0]\n"
            "\n"
            "then we loop downto 0:\n"
            "garbage !! ??\n"
        )
        out = extract_entry_function(blob, "minCosts")
        ast.parse(out)
        assert "minCosts" in out
        assert "self" not in out  # bare top-level form

    def test_valid_input_unchanged_behavior(self) -> None:
        code = "def helper():\n    return 1\n\ndef solve(x):\n    return x + 1\n"
        out = extract_entry_function(code, "solve")
        assert out == "def solve(x):\n    return x + 1"

    def test_unsalvageable_blob_returns_raw_text(self) -> None:
        blob = "this is not code at all\n!! ?? downto 0\nminCosts but no def\n"
        out = extract_entry_function(blob, "minCosts")
        assert out == blob.strip()

    def test_empty_and_missing_entry(self) -> None:
        assert extract_entry_function("", "minCosts") == ""
        assert extract_entry_function("def f(): pass", "") == "def f(): pass"
