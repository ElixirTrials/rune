"""Structured state representations for adapter compression.

ArtifactState represents the current code artifact (code phases).
TrajectoryState represents text-phase output (decompose, plan, diagnose).
"""

from __future__ import annotations

import ast as _ast
import re
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
        """Serialize to plain dict."""
        return {
            "file_contents": self.file_contents,
            "interface_summary": self.interface_summary,
            "import_block": self.import_block,
            "patches": [
                {
                    "turn": p.turn,
                    "description": p.description,
                    "diff_summary": p.diff_summary,
                }
                for p in self.patches
            ],
            "test_results": self.test_results,
            "stderr_summary": self.stderr_summary,
            "tests_passed": self.tests_passed,
            "todos": self.todos,
        }

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> ArtifactState:
        """Deserialize from plain dict."""
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
        """Serialize to plain dict."""
        return {
            "turn": self.turn,
            "output": self.output,
            "feedback": self.feedback,
            "diagnosis": self.diagnosis,
        }

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> TrajectoryState:
        """Deserialize from plain dict."""
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


def _extract_imports(code: str) -> str:
    """Extract all import lines from code."""
    lines = []
    for line in code.splitlines():
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")):
            lines.append(line)
    return "\n".join(lines)


def _extract_interfaces(code: str) -> str:
    """Extract function/class signatures from code."""
    from shared.blackboard import extract_interfaces
    return extract_interfaces(code)


def _compute_diff_summary(current: str, previous: str) -> str:
    """Compute a compact diff summary between code versions."""
    current_funcs = set(re.findall(r"^(?:def|class)\s+(\w+)", current, re.MULTILINE))
    previous_funcs = set(re.findall(r"^(?:def|class)\s+(\w+)", previous, re.MULTILINE))

    added = current_funcs - previous_funcs
    removed = previous_funcs - current_funcs
    parts = []
    if added:
        parts.append("+" + ", ".join(sorted(added)))
    if removed:
        parts.append("-" + ", ".join(sorted(removed)))
    if not parts:
        parts.append("modified existing")
    return "; ".join(parts)


def _extract_error_summary(stderr: str) -> str:
    """Extract the most actionable error info from raw stderr."""
    if not stderr:
        return ""
    lines = stderr.strip().splitlines()
    error_lines = [
        ln for ln in lines
        if "Error" in ln or "Exception" in ln or "assert" in ln.lower()
    ]
    final_error = error_lines[-1].strip() if error_lines else lines[-1].strip()
    failed_tests = [
        ln.strip() for ln in lines
        if ln.strip().startswith("FAIL:") or ln.strip().startswith("ERROR:")
    ]
    parts = []
    if failed_tests:
        parts.append("Failed: " + "; ".join(failed_tests[:5]))
    parts.append(final_error)
    return "\n".join(parts)


def _find_unresolved_todos(code: str, tests_passed: bool, stderr: str) -> list[str]:
    """Identify unresolved obligations in the code."""
    todos: list[str] = []
    for match in re.finditer(r"^(def|class)\s+(\w+).*:\s*$", code, re.MULTILINE):
        name = match.group(2)
        end = match.end()
        body_start = code[end:end + 50].strip()
        if body_start.startswith("pass") or body_start == "...":
            todos.append(f"stub: {match.group(1)} {name}")
    if not tests_passed and stderr:
        for line in stderr.splitlines():
            stripped = line.strip()
            if stripped.startswith(("FAIL:", "ERROR:")):
                todos.append(f"failing: {stripped[:80]}")
    return todos


def build_artifact_state(
    generated_code: str,
    stdout: str,
    stderr: str,
    tests_passed: bool,
    turn: int,
    previous_artifact: ArtifactState | None,
) -> ArtifactState:
    """Build an ArtifactState from the current turn's output."""
    import_block = _extract_imports(generated_code)
    interface_summary = _extract_interfaces(generated_code)

    if previous_artifact is not None:
        diff_summary = _compute_diff_summary(
            generated_code, previous_artifact.file_contents
        )
        description = f"turn {turn}: {diff_summary[:100]}"
        new_patch = PatchRecord(
            turn=turn,
            description=description,
            diff_summary=diff_summary,
        )
        patches = list(previous_artifact.patches) + [new_patch]
    else:
        funcs = set(
            re.findall(
                r"^(?:def|class)\s+(\w+)",
                generated_code,
                re.MULTILINE,
            )
        )
        diff_summary = (
            "+" + ", ".join(sorted(funcs)) if funcs else "initial"
        )
        new_patch = PatchRecord(
            turn=turn,
            description="initial",
            diff_summary=diff_summary,
        )
        patches = [new_patch]

    from shared.sandbox import count_test_results
    passed, total = count_test_results(stdout, stderr)
    test_results = f"{passed}/{total} passed" if total > 0 else "no tests"

    stderr_summary = _extract_error_summary(stderr)
    todos = _find_unresolved_todos(generated_code, tests_passed, stderr)

    return ArtifactState(
        file_contents=generated_code,
        interface_summary=interface_summary,
        import_block=import_block,
        patches=patches,
        test_results=test_results,
        stderr_summary=stderr_summary,
        tests_passed=tests_passed,
        todos=todos,
    )


def _split_top_level_blocks(code: str) -> list[tuple[str, str, str]]:
    """Split code into top-level blocks: (type, name, source)."""
    try:
        tree = _ast.parse(code)
    except SyntaxError:
        return [("body", "unparseable", code)]

    blocks: list[tuple[str, str, str]] = []
    lines = code.splitlines(keepends=True)

    for node in _ast.iter_child_nodes(tree):
        if isinstance(node, _ast.ClassDef):
            start = node.lineno - 1
            end = node.end_lineno or start + 1
            source = "".join(lines[start:end])
            blocks.append(("class", node.name, source))
        elif isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
            start = node.lineno - 1
            end = node.end_lineno or start + 1
            source = "".join(lines[start:end])
            blocks.append(("function", node.name, source))

    return blocks


def chunk_code_state(
    artifact: ArtifactState,
    max_chunk_tokens: int,
) -> list[CodeChunk]:
    """Split artifact into semantic chunks for multi-pass encoding."""
    chunks: list[CodeChunk] = []

    if artifact.import_block.strip():
        chunks.append(CodeChunk(
            chunk_type="imports",
            name="imports",
            content=artifact.import_block,
            priority=1.0,
        ))

    if artifact.interface_summary.strip():
        chunks.append(CodeChunk(
            chunk_type="interfaces",
            name="interfaces",
            content=artifact.interface_summary,
            priority=0.95,
        ))

    blocks = _split_top_level_blocks(artifact.file_contents)
    for block_type, name, source in blocks:
        chunks.append(CodeChunk(
            chunk_type=block_type,
            name=name,
            content=source,
            priority=0.7 if block_type == "class" else 0.6,
        ))

    if artifact.patches or artifact.test_results:
        patch_text = "\n".join(
            f"Turn {p.turn}: {p.description} ({p.diff_summary})"
            for p in artifact.patches
        )
        if artifact.test_results:
            patch_text += f"\nTests: {artifact.test_results}"
        if artifact.stderr_summary:
            patch_text += f"\nErrors: {artifact.stderr_summary}"
        chunks.append(CodeChunk(
            chunk_type="patches",
            name="history",
            content=patch_text,
            priority=0.5,
        ))

    return chunks
