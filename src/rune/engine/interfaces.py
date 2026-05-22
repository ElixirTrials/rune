from __future__ import annotations

import tree_sitter_python as tspython
from tree_sitter import Language, Parser

PY_LANGUAGE = Language(tspython.language())
_parser = Parser(PY_LANGUAGE)


def extract_interfaces(code: str) -> str:
    """Extract function and class signatures from Python source code.

    Args:
        code: Python source code string.

    Returns:
        Newline-joined string of top-level function signatures and class
        definitions (with method signatures).
    """
    if not code.strip():
        return ""

    tree = _parser.parse(code.encode())
    lines: list[str] = []

    for node in tree.root_node.children:
        if node.type == "function_definition":
            first_line = code[node.start_byte : node.end_byte].split("\n", maxsplit=1)[
                0
            ]
            lines.append(first_line)
        elif node.type == "class_definition":
            class_code = code[node.start_byte : node.end_byte]
            class_lines = class_code.split("\n")
            lines.append(class_lines[0])
            for child in node.children:
                if child.type == "block":
                    for stmt in child.children:
                        if stmt.type == "function_definition":
                            method_line = code[stmt.start_byte : stmt.end_byte].split(
                                "\n", maxsplit=1
                            )[0]
                            lines.append(method_line)

    return "\n".join(lines)
