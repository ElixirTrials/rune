"""Pluggable task-requirement oracle for benchmark runs.

Each ``TaskRequirement`` activates only when the task itself supplies evidence
(starter signature, public_checks, Constraints block, etc.). No problem-topic
keyword rules. Failed requirements produce explicit repair deficiencies.

Add new requirement types by implementing ``TaskRequirement`` and appending to
``TASK_REQUIREMENTS``.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Protocol

from rune.engine.complexity import ComplexityProbeConfig, check_constraint_scale
from rune.engine.oracle import defines_entry_point, parse_public_call_arglists
from rune.sandbox.executor import run_in_sandbox


@dataclass(frozen=True)
class RequirementContext:
    """Task fields used to decide which requirements apply."""

    entry_point: str
    signature: str
    spec: str
    public_checks: str
    complexity_probe: ComplexityProbeConfig | None = None

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> RequirementContext:
        from rune.config import PipelineConfig  # noqa: PLC0415

        defaults = PipelineConfig()
        return cls(
            entry_point=str(state.get("entry_point", "") or ""),
            signature=str(state.get("signature", "") or ""),
            spec=str(state.get("task", "") or ""),
            public_checks=str(state.get("public_checks", "") or ""),
            complexity_probe=ComplexityProbeConfig(
                min_n=int(
                    state.get("complexity_probe_min_n", defaults.complexity_probe_min_n)
                ),
                max_n=int(
                    state.get("complexity_probe_max_n", defaults.complexity_probe_max_n)
                ),
                n_repeats=int(
                    state.get(
                        "complexity_probe_n_repeats",
                        defaults.complexity_probe_n_repeats,
                    )
                ),
                per_run_timeout_s=float(
                    state.get(
                        "complexity_probe_per_run_timeout_s",
                        defaults.complexity_probe_per_run_timeout_s,
                    )
                ),
            ),
        )

    @cached_property
    def public_calls(self) -> list[list[Any]]:
        if not self.public_checks.strip() or not self.entry_point:
            return []
        return parse_public_call_arglists(self.public_checks, self.entry_point)


@dataclass(frozen=True)
class RequirementOutcome:
    """Result of one requirement check."""

    kind: str
    required: bool
    ok: bool
    message: str = ""


class TaskRequirement(Protocol):
    """One check derived from structured task evidence."""

    kind: str

    def applies(self, ctx: RequirementContext) -> bool:
        """True when this task states the requirement."""

    def check(self, code: str, ctx: RequirementContext) -> RequirementOutcome:
        """Evaluate *code* when ``applies`` is true."""


def _entry_function(code: str, entry_point: str) -> ast.FunctionDef | None:
    if not code.strip() or not entry_point:
        return None
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    funcs = [
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == entry_point
    ]
    if funcs:
        return funcs[-1]
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Solution":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == entry_point:
                    return item
    return None


def _expected_param_names(signature: str, entry_point: str) -> list[str] | None:
    src = signature.strip()
    if not src:
        return None
    parse_src = f"{src} pass" if src.endswith(":") else src
    try:
        tree = ast.parse(parse_src)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
            not entry_point or node.name == entry_point
        ):
            return [a.arg for a in node.args.args if a.arg not in ("self", "cls")]
    return None


class EntryPointRequirement:
    """Task names an entry_point — implementation must define it."""

    kind = "entry_point"

    def applies(self, ctx: RequirementContext) -> bool:
        return bool(ctx.public_checks.strip() and ctx.entry_point)

    def check(self, code: str, ctx: RequirementContext) -> RequirementOutcome:
        ok = defines_entry_point(code, ctx.entry_point)
        return RequirementOutcome(
            kind=self.kind,
            required=True,
            ok=ok,
            message="" if ok else f"entry_point: must define `{ctx.entry_point}`",
        )


class ExecutableRequirement:
    """Shipped code must load in the same sandbox used for public checks."""

    kind = "executable"

    def applies(self, ctx: RequirementContext) -> bool:
        return bool(ctx.public_checks.strip())

    def check(self, code: str, ctx: RequirementContext) -> RequirementOutcome:
        if not code.strip():
            return RequirementOutcome(
                kind=self.kind, required=True, ok=False, message="empty code"
            )
        from rune.engine.oracle import with_probe_imports  # noqa: PLC0415

        result = run_in_sandbox(with_probe_imports(code.strip() + "\n"), timeout=5)
        if result.exit_code == 0:
            return RequirementOutcome(kind=self.kind, required=True, ok=True)
        err = (result.stderr or result.stdout or "load failed").strip()
        last = err.splitlines()[-1] if err else "load failed"
        return RequirementOutcome(
            kind=self.kind,
            required=True,
            ok=False,
            message=f"executable: code failed to load in sandbox ({last})",
        )


class SignatureRequirement:
    """Starter signature defines the required API when present."""

    kind = "signature"

    def applies(self, ctx: RequirementContext) -> bool:
        return bool(
            ctx.public_checks.strip()
            and ctx.entry_point
            and _expected_param_names(ctx.signature, ctx.entry_point) is not None
        )

    def check(self, code: str, ctx: RequirementContext) -> RequirementOutcome:
        expected = _expected_param_names(ctx.signature, ctx.entry_point)
        assert expected is not None
        fn = _entry_function(code, ctx.entry_point)
        if fn is None:
            return RequirementOutcome(
                kind=self.kind,
                required=True,
                ok=False,
                message=(
                    f"signature: define top-level `{ctx.entry_point}` "
                    f"matching the starter"
                ),
            )
        actual = [a.arg for a in fn.args.args]
        if actual == expected:
            return RequirementOutcome(kind=self.kind, required=True, ok=True)
        exp = ", ".join(expected)
        got = ", ".join(actual) or "(none)"
        return RequirementOutcome(
            kind=self.kind,
            required=True,
            ok=False,
            message=(
                f"signature: expected def {ctx.entry_point}({exp}) but got "
                f"def {ctx.entry_point}({got})"
            ),
        )


class PublicContractRequirement:
    """Public assert call shapes must match the implementation."""

    kind = "contract"

    def applies(self, ctx: RequirementContext) -> bool:
        return bool(ctx.public_checks.strip() and ctx.entry_point and ctx.public_calls)

    def check(self, code: str, ctx: RequirementContext) -> RequirementOutcome:
        fn = _entry_function(code, ctx.entry_point)
        if fn is None:
            return RequirementOutcome(kind=self.kind, required=True, ok=True)
        actual_arity = len(fn.args.args)
        for args in ctx.public_calls:
            if len(args) != actual_arity:
                return RequirementOutcome(
                    kind=self.kind,
                    required=True,
                    ok=False,
                    message=(
                        f"contract: public checks call {ctx.entry_point} with "
                        f"{len(args)} argument(s) {args!r} but implementation has "
                        f"{actual_arity} parameter(s)"
                    ),
                )
        g: dict[str, Any] = {}
        try:
            exec(compile(code, "<requirements>", "exec"), g)
        except Exception as exc:
            return RequirementOutcome(
                kind=self.kind,
                required=True,
                ok=False,
                message=f"contract: could not load implementation: {exc}",
            )
        fn_obj = g.get(ctx.entry_point)
        if not callable(fn_obj):
            return RequirementOutcome(
                kind=self.kind,
                required=True,
                ok=False,
                message=f"contract: `{ctx.entry_point}` is not callable",
            )
        for args in ctx.public_calls:
            try:
                fn_obj(*args)
            except TypeError as exc:
                return RequirementOutcome(
                    kind=self.kind,
                    required=True,
                    ok=False,
                    message=(
                        f"contract: public checks call {ctx.entry_point}{args!r} "
                        f"but implementation rejected it ({exc})"
                    ),
                )
            except Exception:
                # Runtime probe errors are OK; only TypeError means bad contract.
                pass
        return RequirementOutcome(kind=self.kind, required=True, ok=True)


class ConstraintScaleRequirement:
    """Constraints allow inputs much larger than public examples — probe scale."""

    kind = "constraint_scale"

    def applies(self, ctx: RequirementContext) -> bool:
        return bool(ctx.public_checks.strip() and ctx.entry_point and ctx.spec.strip())

    def check(self, code: str, ctx: RequirementContext) -> RequirementOutcome:
        outcome = check_constraint_scale(
            code,
            entry_point=ctx.entry_point,
            spec=ctx.spec,
            public_checks=ctx.public_checks,
            signature=ctx.signature,
            probe_config=ctx.complexity_probe,
        )
        return RequirementOutcome(
            kind=self.kind,
            required=outcome.required,
            ok=outcome.ok,
            message=outcome.message,
        )


TASK_REQUIREMENTS: list[TaskRequirement] = [
    EntryPointRequirement(),
    ExecutableRequirement(),
    SignatureRequirement(),
    PublicContractRequirement(),
    ConstraintScaleRequirement(),
]


def _normalize_entry_code(code: str, entry_point: str) -> str:
    """Reduce to the gradeable entry form (``class Solution`` -> bare ``def``).

    The in-loop probe (`_normalize_probe_code`) and the LCB grader
    (`normalize_lcb_submission`) both grade the bare entry function; only this
    oracle used to see the raw class form, so the canonical
    ``class Solution: def m(self, ...)`` shape failed signature/contract on the
    spurious ``self`` parameter and flipped a passing solution to failing
    (issue #52, q3753). Normalizing here keeps all three stages consistent while
    still catching a genuinely wrong bare signature.
    """
    if not entry_point or not code.strip():
        return code
    from rune.bench.lcb import extract_entry_function  # noqa: PLC0415

    normalized = extract_entry_function(code, entry_point)
    return normalized if normalized.strip() else code


def is_constraint_scale_only_failure(stderr: str) -> bool:
    """True when stderr is only an advisory constraint-scale timeout."""
    text = (stderr or "").strip()
    if "constraint_scale:" not in text:
        return False
    if "AssertionError" in text:
        return False
    if "Task requirements failed" not in text:
        return False
    kinds = [m.group(1) for m in re.finditer(r"^-\s*(\w+):", text, re.MULTILINE)]
    return kinds == ["constraint_scale"]


def evaluate_task_requirements(
    code: str,
    ctx: RequirementContext,
    *,
    requirements: Sequence[TaskRequirement] = TASK_REQUIREMENTS,
    skip_kinds: frozenset[str] = frozenset(),
) -> tuple[bool, tuple[str, ...]]:
    """Run all requirements that apply to this task; return (ok, deficiencies)."""
    if not ctx.public_checks.strip():
        return True, ()
    code = _normalize_entry_code(code, ctx.entry_point)
    deficiencies: list[str] = []
    for req in requirements:
        if req.kind in skip_kinds:
            continue
        if not req.applies(ctx):
            continue
        outcome = req.check(code, ctx)
        if outcome.required and not outcome.ok and outcome.message:
            deficiencies.append(outcome.message)
            break
    return not deficiencies, tuple(deficiencies)


def evaluate_state_requirements(
    state: Mapping[str, Any],
    code: str,
    *,
    skip_kinds: frozenset[str] = frozenset(),
) -> tuple[bool, tuple[str, ...]]:
    """Evaluate requirements using fields from RunState."""
    return evaluate_task_requirements(
        code,
        RequirementContext.from_state(state),
        skip_kinds=skip_kinds,
    )


def format_requirements_feedback(deficiencies: tuple[str, ...]) -> str:
    """Repair-facing message listing each failed requirement."""
    lines = ["Task requirements failed — fix exactly:"]
    lines.extend(f"- {d}" for d in deficiencies)
    return "\n".join(lines)
