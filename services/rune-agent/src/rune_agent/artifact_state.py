"""Structured state representations for adapter compression.

ArtifactState represents the current code artifact (code phases).
TrajectoryState represents text-phase output (decompose, plan, diagnose).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PatchRecord:
    """A single code change between turns."""

    turn: int
    description: str
    diff_summary: str


@dataclass
class ArtifactState:
    """Structured code state for adapter compression."""

    file_contents: str
    interface_summary: str
    import_block: str
    patches: list[PatchRecord]
    test_results: str
    stderr_summary: str
    tests_passed: bool
    todos: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "file_contents": self.file_contents,
            "interface_summary": self.interface_summary,
            "import_block": self.import_block,
            "patches": [
                {"turn": p.turn, "description": p.description, "diff_summary": p.diff_summary}
                for p in self.patches
            ],
            "test_results": self.test_results,
            "stderr_summary": self.stderr_summary,
            "tests_passed": self.tests_passed,
            "todos": self.todos,
        }

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> ArtifactState:
        patches = [PatchRecord(**p) for p in d.get("patches", [])]  # type: ignore[arg-type]
        return cls(
            file_contents=str(d.get("file_contents", "")),
            interface_summary=str(d.get("interface_summary", "")),
            import_block=str(d.get("import_block", "")),
            patches=patches,
            test_results=str(d.get("test_results", "")),
            stderr_summary=str(d.get("stderr_summary", "")),
            tests_passed=bool(d.get("tests_passed", False)),
            todos=list(d.get("todos", [])),  # type: ignore[arg-type]
        )


@dataclass
class TrajectoryState:
    """Lightweight state for non-code phases."""

    turn: int
    output: str
    feedback: str
    diagnosis: str

    def to_dict(self) -> dict[str, object]:
        return {
            "turn": self.turn,
            "output": self.output,
            "feedback": self.feedback,
            "diagnosis": self.diagnosis,
        }

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> TrajectoryState:
        return cls(
            turn=int(d.get("turn", 0)),  # type: ignore[arg-type]
            output=str(d.get("output", "")),
            feedback=str(d.get("feedback", "")),
            diagnosis=str(d.get("diagnosis", "")),
        )


@dataclass
class CodeChunk:
    """A semantic chunk of code for multi-pass encoding."""

    chunk_type: str
    name: str
    content: str
    priority: float
