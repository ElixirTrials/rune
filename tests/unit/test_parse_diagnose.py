"""Unit tests for _parse_diagnose_output JSON parsing."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from bootstrap import setup_path  # type: ignore[import-not-found]

setup_path()

from scripts.rune_runner import _parse_diagnose_output  # noqa: E402


def test_diagnose_exact_match() -> None:
    output = json.dumps({"repairs": [
        {"name": "parse_input", "diagnosis": "Fix off-by-one error"},
    ]})
    result = _parse_diagnose_output(output, ["parse_input", "write_output"])
    assert len(result) == 1
    assert result[0]["name"] == "parse_input"
    assert result[0]["diagnosis"] == "Fix off-by-one error"


def test_diagnose_substring_match() -> None:
    output = json.dumps({"repairs": [
        {"name": "parse", "diagnosis": "Fix the bug"},
    ]})
    result = _parse_diagnose_output(output, ["parse_input", "write_output"])
    assert len(result) == 1
    assert result[0]["name"] == "parse_input"


def test_diagnose_no_match() -> None:
    output = json.dumps({"repairs": [
        {"name": "unknown_task", "diagnosis": "Fix something"},
    ]})
    result = _parse_diagnose_output(output, ["parse_input", "write_output"])
    assert len(result) == 0


def test_diagnose_invalid_json_returns_empty() -> None:
    result = _parse_diagnose_output("not json", ["parse_input"])
    assert result == []


def test_diagnose_multiple_repairs() -> None:
    output = json.dumps({"repairs": [
        {"name": "parse_input", "diagnosis": "Fix error"},
        {"name": "write_output", "diagnosis": "Handle edge case"},
    ]})
    result = _parse_diagnose_output(output, ["parse_input", "write_output"])
    assert len(result) == 2
