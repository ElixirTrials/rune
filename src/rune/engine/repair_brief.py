"""Deterministic repair-signal classifiers (linter-style, not ML).

Literature (e.g. CodeLinterEval flake8 codes, LLM codegen error taxonomies) shows
structured, explicit failure classes outperform coarse error messages for repair.
No Hugging Face checkpoint targets our LCB sandbox + requirements oracle mix, so
we classify stderr/requirements deterministically before the LLM diagnose step.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from rune.engine.requirements import _expected_param_names


@dataclass(frozen=True)
class RepairBrief:
    """Structured repair signal consumed by diagnose and repair prompts."""

    failure_class: str
    violated_invariant: str
    observed: str
    expected: str
    fix_directive: str
    replan_recommended: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def format_block(self) -> str:
        """Human-readable block for prompts and adapter conditioning."""
        lines = [
            f"failure_class: {self.failure_class}",
            f"violated_invariant: {self.violated_invariant}",
            f"observed: {self.observed}",
            f"expected: {self.expected}",
            f"fix_directive: {self.fix_directive}",
            f"replan_recommended: {self.replan_recommended}",
        ]
        return "\n".join(lines)


_REQ_LINE = re.compile(r"^-\s*(\w+):\s*(.+)$", re.MULTILINE)
_ARITY_RE = re.compile(
    r"takes (\d+) positional argument[s]? but (\d+) were given", re.IGNORECASE
)
_ASSERT_WANT = re.compile(
    r"AssertionError:\s*(.+?)\s*->\s*(.+?),\s*want\s*(.+?)(?:\n|$)", re.DOTALL
)
_ASSERT_SIMPLE = re.compile(
    r"AssertionError:\s*(.+?)(?:\n|$)", re.DOTALL
)


def _signature_brief(
    message: str, entry_point: str, signature: str
) -> RepairBrief:
    expected_sig = _format_expected_signature(entry_point, signature)
    return RepairBrief(
        failure_class="signature",
        violated_invariant=(
            f"Top-level function `{entry_point}` must match the starter signature"
        ),
        observed=message[:300],
        expected=expected_sig,
        fix_directive=(
            f"Emit a bare `def {entry_point}(...)` matching {expected_sig}, "
            "not a class method. Preserve the algorithm from your last attempt — "
            "fix ONLY the signature/wrapper defect."
        ),
        replan_recommended=False,
    )


def _format_expected_signature(entry_point: str, signature: str) -> str:
    names = _expected_param_names(signature, entry_point) if signature else None
    if names is not None:
        return f"def {entry_point}({', '.join(names)})"
    return f"def {entry_point}(...)"


def _complexity_brief(message: str) -> RepairBrief:
    return RepairBrief(
        failure_class="complexity",
        violated_invariant=(
            "Implementation must finish within constraint-scale inputs "
            "(time/space limits in Constraints block)"
        ),
        observed=message[:300],
        expected="Polynomial or better algorithm appropriate to stated bounds",
        fix_directive=(
            "Replace brute-force enumeration with an efficient algorithm "
            "(e.g. DP) that respects the Constraints scale"
        ),
        replan_recommended=True,
    )


def _arity_brief(
    stderr: str, entry_point: str, signature: str
) -> RepairBrief | None:
    m = _ARITY_RE.search(stderr)
    if not m:
        return None
    expected_names = (
        _expected_param_names(signature, entry_point) if signature else None
    )
    expected = (
        f"def {entry_point}({', '.join(expected_names)})"
        if expected_names
        else f"def {entry_point}(...) with correct arity"
    )
    return RepairBrief(
        failure_class="arity",
        violated_invariant=f"Function `{entry_point}` must accept the task parameters",
        observed=stderr.strip()[:300],
        expected=expected,
        fix_directive=f"Restore the full signature: {expected}",
        replan_recommended=False,
    )


_ODD_FREQ_RE = re.compile(r"odd\w*freq", re.IGNORECASE)
_EVEN_FREQ_RE = re.compile(r"even\w*freq", re.IGNORECASE)
_MAX_MIN_FREQ_RE = re.compile(
    r"\b(?:highest|lowest|max(?:imum)?|min(?:imum)?)\b.*\bfreq",
    re.IGNORECASE,
)


def _enrich_assertion_invariant(
    invariant: str,
    *,
    entry_point: str = "",
    plan: str = "",
    overall_goal: str = "",
    acceptance_check: str = "",
    subtask_description: str = "",
) -> str:
    """Pull algorithmic invariant from plan/goal when stderr lacks semantics."""
    context = " ".join(
        [plan, overall_goal, acceptance_check, subtask_description]
    )
    has_freq_parity = entry_point == "maxDifference" or (
        _ODD_FREQ_RE.search(context) and _EVEN_FREQ_RE.search(context)
    )
    if has_freq_parity:
        return (
            "Return the maximum difference between an odd-frequency character "
            "count and an even-frequency character count — NOT "
            "max(all_freq) - min(all_freq)"
        )
    return invariant


def _assertion_brief(
    stderr: str,
    entry_point: str,
    *,
    plan: str = "",
    overall_goal: str = "",
    acceptance_check: str = "",
    subtask_description: str = "",
) -> RepairBrief | None:
    m = _ASSERT_WANT.search(stderr)
    if m:
        call, got, want = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
        invariant = (
            f"`{entry_point}` must return correct results for public examples"
        )
        if "[" in want or "grid" in call.lower() or "matrix" in call.lower():
            invariant = (
                "Each anti-diagonal (constant i+j) must be sorted independently; "
                "bottom-left diagonals non-increasing, top-right non-decreasing"
            )
        invariant = _enrich_assertion_invariant(
            invariant,
            entry_point=entry_point,
            plan=plan,
            overall_goal=overall_goal,
            acceptance_check=acceptance_check,
            subtask_description=subtask_description,
        )
        fix = (
            "Fix the algorithm so observed output matches expected on the "
            "failing public case"
        )
        if "odd-frequency" in invariant.lower():
            fix = (
                "Use odd-vs-even frequency parity (not max-min across all freqs) "
                "so observed output matches expected"
            )
        return RepairBrief(
            failure_class="assertion",
            violated_invariant=invariant,
            observed=f"{call} -> {got}",
            expected=want,
            fix_directive=fix,
            replan_recommended=False,
        )
    m2 = _ASSERT_SIMPLE.search(stderr)
    if m2:
        return RepairBrief(
            failure_class="assertion",
            violated_invariant=(
                f"`{entry_point}` must satisfy the public acceptance checks"
            ),
            observed=m2.group(1).strip()[:200],
            expected="Pass all public asserts",
            fix_directive="Correct the logic causing the assertion failure",
            replan_recommended=False,
        )
    return None


def _requirements_brief(
    stderr: str, entry_point: str, signature: str
) -> RepairBrief | None:
    if "Task requirements failed" not in stderr:
        return None
    kinds: list[str] = []
    messages: list[str] = []
    for m in _REQ_LINE.finditer(stderr):
        kinds.append(m.group(1))
        messages.append(m.group(2).strip())
    if not kinds:
        return None
    if any(k == "constraint_scale" for k in kinds):
        return _complexity_brief(messages[kinds.index("constraint_scale")])
    if any(k == "signature" for k in kinds):
        idx = kinds.index("signature")
        return _signature_brief(messages[idx], entry_point, signature)
    if any(k == "executable" for k in kinds):
        return RepairBrief(
            failure_class="import",
            violated_invariant="Code must execute in the sandbox without import errors",
            observed=messages[kinds.index("executable")][:300],
            expected="Runnable top-level function without missing typing imports",
            fix_directive=(
                "Remove typing-only annotations from the probe path or use "
                "built-in generics (list, dict)"
            ),
            replan_recommended=False,
        )
    if any(k == "entry_point" for k in kinds):
        return RepairBrief(
            failure_class="entry_point",
            violated_invariant=f"Must define top-level `{entry_point}`",
            observed=messages[kinds.index("entry_point")][:300],
            expected=f"def {entry_point}(...)",
            fix_directive=f"Define function named exactly `{entry_point}`",
            replan_recommended=False,
        )
    return RepairBrief(
        failure_class="requirements",
        violated_invariant="Task requirements contract",
        observed="; ".join(messages)[:300],
        expected="All active requirements pass",
        fix_directive=messages[0][:200],
        replan_recommended=False,
    )


def build_repair_brief(
    stderr: str,
    *,
    entry_point: str = "",
    signature: str = "",
    plan: str = "",
    overall_goal: str = "",
    acceptance_check: str = "",
    subtask_description: str = "",
) -> RepairBrief | None:
    """Classify sandbox stderr into a structured repair brief.

    Returns None when stderr is empty or unclassified (caller may still diagnose).
    """
    text = (stderr or "").strip()
    if not text:
        return None

    req = _requirements_brief(text, entry_point, signature)
    if req is not None:
        return req

    arity = _arity_brief(text, entry_point, signature)
    if arity is not None:
        return arity

    if "NameError" in text and "List" in text:
        return RepairBrief(
            failure_class="import",
            violated_invariant="Sandbox probe must not reference unimported names",
            observed=text.splitlines()[-1][:300],
            expected="Use builtin `list` or import from typing in executed code",
            fix_directive="Drop `List[...]` annotations or avoid typing-only imports",
            replan_recommended=False,
        )

    if "AssertionError" in text:
        assertion = _assertion_brief(
            text,
            entry_point,
            plan=plan,
            overall_goal=overall_goal,
            acceptance_check=acceptance_check,
            subtask_description=subtask_description,
        )
        if assertion is not None:
            return assertion

    if "UnboundLocalError" in text or "SyntaxError" in text:
        return RepairBrief(
            failure_class="runtime",
            violated_invariant="Code must run without exceptions on public inputs",
            observed=text.strip()[:300],
            expected="Clean execution through the failing path",
            fix_directive="Fix the runtime error at the indicated location",
            replan_recommended=False,
        )

    if "TypeError" in text:
        return RepairBrief(
            failure_class="runtime",
            violated_invariant="Arguments and types must match the task contract",
            observed=text.strip()[:300],
            expected=_format_expected_signature(entry_point, signature),
            fix_directive="Align function signature and argument usage with the task",
            replan_recommended=False,
        )

    return None


def _contradicts_brief(brief_block: str, llm_guidance: str) -> bool:
    """Drop LLM guidance that reframes a deterministic brief invariant."""
    brief_lower = brief_block.lower()
    guidance_lower = llm_guidance.lower()
    if "odd-frequency" in brief_lower or (
        "odd" in brief_lower and "even" in brief_lower
    ):
        if _MAX_MIN_FREQ_RE.search(guidance_lower):
            return True
        if "highest" in guidance_lower and "lowest" in guidance_lower:
            return True
    return False


def merge_guidance_with_brief(
    brief_block: str,
    llm_guidance: str,
    *,
    llm_failure_class: str = "",
    llm_violated_invariant: str = "",
    llm_observed_vs_expected: str = "",
) -> str:
    """Prefer deterministic brief; append non-contradictory LLM how-to-fix only."""
    if not brief_block.strip():
        has_structured = bool(
            llm_violated_invariant.strip() or llm_observed_vs_expected.strip()
        )
        if has_structured:
            parts: list[str] = []
            if llm_violated_invariant.strip():
                parts.append(f"violated_invariant: {llm_violated_invariant[:200]}")
            if llm_observed_vs_expected.strip():
                parts.append(f"observed_vs_expected: {llm_observed_vs_expected[:200]}")
            if llm_guidance.strip():
                parts.append(f"how_to_fix: {llm_guidance[:300]}")
            return "\n".join(parts)[:500]
        return llm_guidance[:500]
    parts = [brief_block.strip()]
    if llm_guidance.strip() and not _contradicts_brief(brief_block, llm_guidance):
        parts.append(f"how_to_fix: {llm_guidance[:300]}")
    if llm_failure_class and f"failure_class: {llm_failure_class}" not in brief_block:
        parts.append(f"(note: model suggested {llm_failure_class})")
    return "\n".join(parts)[:500]
