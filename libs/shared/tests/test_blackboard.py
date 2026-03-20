"""Tests for blackboard module."""

from __future__ import annotations

from shared.blackboard import (
    Blackboard,
    SubtaskArtifact,
    build_execution_layers,
    extract_interfaces,
    parse_dependencies,
)


class TestExtractInterfaces:
    def test_extracts_class_and_def(self) -> None:
        code = "import os\n\nclass Foo:\n    x = 1\n\ndef bar():\n    pass\n"
        result = extract_interfaces(code)
        assert "import os" in result
        assert "class Foo:" in result
        assert "def bar():" in result
        assert "x = 1" not in result

    def test_extracts_decorators(self) -> None:
        code = "@dataclass\nclass Event:\n    name: str\n"
        result = extract_interfaces(code)
        assert "@dataclass" in result
        assert "class Event:" in result

    def test_empty_code(self) -> None:
        assert extract_interfaces("") == ""

    def test_max_lines(self) -> None:
        code = "\n".join(f"def func_{i}(): pass" for i in range(100))
        result = extract_interfaces(code, max_lines=5)
        assert len(result.splitlines()) == 5


class TestBlackboard:
    def test_publish_and_get(self) -> None:
        bb = Blackboard()
        bb.publish(SubtaskArtifact(
            name="A", code="x=1", interfaces="class A:", tests_passed=True
        ))
        assert bb.get("A") is not None
        assert bb.get("A").interfaces == "class A:"  # type: ignore[union-attr]
        assert bb.get("B") is None

    def test_dependency_interfaces(self) -> None:
        bb = Blackboard()
        bb.publish(SubtaskArtifact(
            name="Model", code="", interfaces="class Model:\n    id: int",
            tests_passed=True,
        ))
        bb.publish(SubtaskArtifact(
            name="Store", code="", interfaces="class Store:\n    def save():",
            tests_passed=True,
        ))
        subtask = {"name": "API", "depends_on": ["Model", "Store"]}
        result = bb.get_dependency_interfaces(subtask)
        assert "class Model:" in result
        assert "class Store:" in result

    def test_missing_dependency(self) -> None:
        bb = Blackboard()
        subtask = {"name": "API", "depends_on": ["Missing"]}
        assert bb.get_dependency_interfaces(subtask) == ""

    def test_all_interfaces(self) -> None:
        bb = Blackboard()
        bb.publish(SubtaskArtifact(
            name="A", code="", interfaces="class A:", tests_passed=True
        ))
        bb.publish(SubtaskArtifact(
            name="B", code="", interfaces="class B:", tests_passed=True
        ))
        result = bb.all_interfaces()
        assert "class A:" in result
        assert "class B:" in result


class TestParseDependencies:
    def test_no_depends(self) -> None:
        assert parse_dependencies("1. Foo — bar", ["Foo", "Bar"]) == []

    def test_depends_none(self) -> None:
        assert parse_dependencies(
            "1. Foo — bar [depends: none]", ["Foo"]
        ) == []

    def test_depends_indices(self) -> None:
        names = ["Data Model", "EventStore", "Ledger"]
        result = parse_dependencies(
            "3. Ledger — impl [depends: 1, 2]", names
        )
        assert result == ["Data Model", "EventStore"]

    def test_depends_name_match(self) -> None:
        names = ["Data Model", "EventStore"]
        result = parse_dependencies(
            "2. EventStore — impl [depends: Data Model]", names
        )
        assert result == ["Data Model"]


class TestBuildExecutionLayers:
    def test_no_deps_single_layer(self) -> None:
        subtasks = [
            {"name": "A"}, {"name": "B"}, {"name": "C"},
        ]
        layers = build_execution_layers(subtasks)
        assert len(layers) == 1
        assert len(layers[0]) == 3

    def test_linear_chain(self) -> None:
        subtasks = [
            {"name": "A"},
            {"name": "B", "depends_on": ["A"]},
            {"name": "C", "depends_on": ["B"]},
        ]
        layers = build_execution_layers(subtasks)
        assert len(layers) == 3
        assert layers[0][0]["name"] == "A"
        assert layers[1][0]["name"] == "B"
        assert layers[2][0]["name"] == "C"

    def test_diamond(self) -> None:
        subtasks = [
            {"name": "Base"},
            {"name": "Left", "depends_on": ["Base"]},
            {"name": "Right", "depends_on": ["Base"]},
            {"name": "Top", "depends_on": ["Left", "Right"]},
        ]
        layers = build_execution_layers(subtasks)
        assert len(layers) == 3
        assert layers[0][0]["name"] == "Base"
        assert len(layers[1]) == 2  # Left and Right in parallel
        assert layers[2][0]["name"] == "Top"

    def test_unknown_deps_ignored(self) -> None:
        subtasks = [
            {"name": "A", "depends_on": ["NonExistent"]},
        ]
        layers = build_execution_layers(subtasks)
        assert len(layers) == 1
        assert layers[0][0]["name"] == "A"
