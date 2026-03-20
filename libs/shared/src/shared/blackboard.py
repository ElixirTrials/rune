"""Typed blackboard for inter-subtask context passing.

Subtasks publish their code and extracted interface signatures after
completion. Dependent subtasks read predecessor interfaces from the
blackboard, which then flow through adapter trajectories.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class SubtaskArtifact:
    """A completed subtask's publishable artifacts."""

    name: str
    code: str
    interfaces: str
    tests_passed: bool
    dependencies: list[str] = field(default_factory=list)


class Blackboard:
    """Shared artifact store for inter-subtask communication."""

    def __init__(self) -> None:
        """Initialize empty blackboard."""
        self._artifacts: dict[str, SubtaskArtifact] = {}

    def publish(self, artifact: SubtaskArtifact) -> None:
        """Publish a subtask's artifacts."""
        self._artifacts[artifact.name] = artifact

    def get(self, name: str) -> SubtaskArtifact | None:
        """Retrieve a subtask's artifacts by name."""
        return self._artifacts.get(name)

    def get_dependency_interfaces(self, subtask: dict[str, object]) -> str:
        """Get concatenated interfaces for a subtask's declared dependencies."""
        raw_deps = subtask.get("depends_on", [])
        deps = raw_deps if isinstance(raw_deps, list) else []
        if not deps:
            return ""
        parts: list[str] = []
        for dep_name in deps:
            artifact = self._artifacts.get(str(dep_name))
            if artifact and artifact.interfaces:
                parts.append(f"--- {artifact.name} ---\n{artifact.interfaces}")
        return "\n".join(parts)

    def all_interfaces(self) -> str:
        """Get all published interfaces."""
        parts: list[str] = []
        for artifact in self._artifacts.values():
            if artifact.interfaces:
                parts.append(f"--- {artifact.name} ---\n{artifact.interfaces}")
        return "\n".join(parts)


def extract_interfaces(code: str, max_lines: int = 60) -> str:
    """Extract class/function signatures, imports, and decorators from code.

    Returns a compact structural summary suitable for adapter trajectories.
    """
    if not code:
        return ""
    result: list[str] = []
    for line in code.splitlines():
        stripped = line.strip()
        if stripped.startswith((
            "import ", "from ",
            "class ", "def ",
            "@",
        )):
            result.append(line)
        elif stripped.startswith(('"""', "'''")):
            result.append(line)
    return "\n".join(result[:max_lines])


def build_execution_layers(
    subtasks: list[dict[str, object]],
) -> list[list[dict[str, object]]]:
    """Group subtasks into dependency layers for ordered execution.

    Layer 0 contains subtasks with no dependencies. Layer N contains
    subtasks whose dependencies are all in layers 0..N-1.

    Args:
        subtasks: List of subtask dicts with optional ``depends_on`` key.

    Returns:
        List of layers, each a list of subtask dicts.
    """
    from graphlib import TopologicalSorter

    name_to_subtask = {str(st["name"]): st for st in subtasks}
    known_names = set(name_to_subtask.keys())

    # Build graph: {node: set_of_dependencies}
    graph: dict[str, set[str]] = {}
    for st in subtasks:
        name = str(st["name"])
        raw_deps = st.get("depends_on", [])
        deps = raw_deps if isinstance(raw_deps, list) else []
        # Filter to known subtask names only
        valid_deps = {str(d) for d in deps if str(d) in known_names}
        graph[name] = valid_deps

    sorter = TopologicalSorter(graph)
    sorter.prepare()

    layers: list[list[dict[str, object]]] = []
    while sorter.is_active():
        ready = list(sorter.get_ready())
        layer = [name_to_subtask[n] for n in ready if n in name_to_subtask]
        layers.append(layer)
        for n in ready:
            sorter.done(n)

    return layers


_DEPENDS_RE = re.compile(r"\[depends?:\s*([^\]]*)\]", re.IGNORECASE)


def parse_dependencies(
    line: str,
    all_subtask_names: list[str],
) -> list[str]:
    """Extract dependency names from a decompose output line.

    Supports formats:
      - ``[depends: none]`` → empty list
      - ``[depends: 1, 3]`` → resolved to subtask names by index
      - ``[depends: Data Model]`` → matched by name substring

    Args:
        line: A single line from decompose output.
        all_subtask_names: Ordered list of all subtask names for index resolution.

    Returns:
        List of dependency subtask names.
    """
    match = _DEPENDS_RE.search(line)
    if not match:
        return []

    raw = match.group(1).strip()
    if not raw or raw.lower() == "none":
        return []

    deps: list[str] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue

        # Try numeric index (1-based)
        try:
            idx = int(part) - 1
            if 0 <= idx < len(all_subtask_names):
                deps.append(all_subtask_names[idx])
                continue
        except ValueError:
            pass

        # Try name match
        part_lower = part.lower()
        for name in all_subtask_names:
            if part_lower in name.lower() or name.lower() in part_lower:
                deps.append(name)
                break

    return deps
