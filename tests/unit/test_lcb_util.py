"""Unit tests for LiveCodeBench submission normalization."""

from __future__ import annotations

import json

from rune.bench.lcb import (
    build_public_assert_checks,
    extract_entry_function,
    normalize_lcb_submission,
)


class TestBuildPublicAssertChecks:
    def test_bare_fn_calls_no_solution(self) -> None:
        row = {
            "metadata": json.dumps({"func_name": "maxDifference"}),
            "public_test_cases": json.dumps(
                [{"input": '"ab"', "output": "1"}],
            ),
        }
        out = build_public_assert_checks(row)
        assert "Solution" not in out
        assert "maxDifference(" in out
        assert "assert" in out


class TestExtractEntryFunction:
    def test_returns_matching_top_level_function(self) -> None:
        code = "def helper():\n    return 1\n\ndef solve(x):\n    return x + 1\n"
        assert (
            extract_entry_function(code, "solve") == "def solve(x):\n    return x + 1"
        )

    def test_last_definition_wins(self) -> None:
        code = "def solve():\n    return 0\n\ndef solve():\n    return 1\n"
        assert "return 1" in extract_entry_function(code, "solve")

    def test_converts_solution_method_to_bare(self) -> None:
        code = (
            "class Solution:\n"
            "    def solve(self, x: int) -> int:\n"
            "        return x + 1\n"
        )
        bare = extract_entry_function(code, "solve")
        assert bare.startswith("def solve(x: int) -> int:")
        assert "return x + 1" in bare

    def test_empty_entry_point_returns_original(self) -> None:
        code = "def solve():\n    return 1\n"
        assert extract_entry_function(code, "").strip() == code.strip()


class TestNormalizeLcbSubmission:
    def test_strips_concatenated_helpers(self) -> None:
        code = (
            "def helper():\n"
            "    pass\n\n"
            "def maxDistance(s, k):\n"
            "    return abs(s.count('N') - s.count('S'))\n"
        )
        out = normalize_lcb_submission(code, "maxDistance")
        assert "def helper" not in out
        assert "def maxDistance(s, k):" in out
