"""Grader delivery contract: exact function name, signature, and public call shape."""

from __future__ import annotations


def format_delivery_contract(
    *,
    entry_point: str,
    bare_signature: str,
    public_checks: str = "",
) -> str:
    """Human-readable deliverable the official grader expects.

    Args:
        entry_point: Required top-level function name.
        bare_signature: ``def name(params):`` stub from the starter signature.
        public_checks: In-loop public assert lines for this task only.

    Returns:
        Multi-line contract block for prompts and adapter conditioning.
    """
    lines = [
        f"Deliverable function name: `{entry_point}` (exactly — no aliases)",
        f"Required signature: {bare_signature}",
        "Emit a bare top-level function with these parameter names (not a class).",
    ]
    for raw in (public_checks or "").splitlines():
        line = raw.strip()
        if line.startswith("assert "):
            lines.append(f"Public grader call shape: {line[:240]}")
            break
    return "\n".join(lines)
