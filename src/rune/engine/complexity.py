"""Constraint-scale probes for the task requirements oracle.

Activated only when the task ``Constraints:`` block allows inputs much larger
than the public examples. Used by ``ConstraintScaleRequirement`` — not a
standalone gate.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Any

from rune.engine.oracle import parse_public_call_arglists
from rune.sandbox.executor import run_in_sandbox

_SCALE_RATIO = 8
_RANGE_SPAN_THRESHOLD = 100_000
_PROBE_TIMEOUT_S = 2
_MAX_LIST_STRESS = 40
_MAX_STRING_STRESS = 5_000
_MAX_RANGE_STRESS = 2_500_000


@dataclass(frozen=True)
class TaskConstraints:
    """Upper bounds parsed from the task ``Constraints:`` section."""

    length_max: dict[str, int]
    range_upper: dict[str, int]


@dataclass(frozen=True)
class ScaleProbeOutcome:
    """Outcome of a constraint-scale probe."""

    required: bool
    ok: bool
    message: str = ""


def _parse_int_bound(raw: str) -> int:
    text = raw.strip().replace(" ", "")
    if "^" in text:
        base, exp = text.split("^", 1)
        return int(int(base) ** int(exp))
    return int(text)


def parse_task_constraints(spec: str) -> TaskConstraints | None:
    """Parse the ``Constraints:`` block from a benchmark task description."""
    lower = spec.lower()
    marker = "constraints:"
    idx = lower.find(marker)
    if idx < 0:
        return None
    block = spec[idx + len(marker) :]
    stop = re.search(r"\n\s*\n\s*[A-Z]", block)
    if stop:
        block = block[: stop.start()]
    length_max: dict[str, int] = {}
    range_upper: dict[str, int] = {}
    for line in block.splitlines():
        text = line.strip()
        if not text:
            continue
        m = re.match(
            r"(\d+)\s*<=\s*(\w+)\.length\s*<=\s*(.+)$",
            text,
            re.IGNORECASE,
        )
        if m:
            name = m.group(2).lower()
            bound = _parse_int_bound(m.group(3))
            length_max[name] = max(length_max.get(name, 0), bound)
            continue
        m = re.match(
            r"1\s*<=\s*(\w+)\s*<=\s*(\w+)\s*<\s*(.+)$",
            text,
            re.IGNORECASE,
        )
        if m:
            hi = m.group(2).lower()
            range_upper[hi] = max(range_upper.get(hi, 0), _parse_int_bound(m.group(3)))
    if not length_max and not range_upper:
        return None
    return TaskConstraints(length_max=length_max, range_upper=range_upper)


def _param_names(signature: str, entry_point: str) -> list[str]:
    src = signature.strip()
    if not src:
        return []
    parse_src = f"{src} pass" if src.endswith(":") else src
    try:
        tree = ast.parse(parse_src)
    except SyntaxError:
        return []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
            not entry_point or node.name == entry_point
        ):
            return [
                a.arg for a in node.args.args if a.arg not in ("self", "cls")
            ]
    return []


def _public_metrics(calls: list[list[Any]]) -> tuple[int, int, int, int]:
    max_list = 0
    max_str = 0
    max_int = 0
    max_span = 0
    for args in calls:
        ints: list[int] = []
        for val in args:
            if isinstance(val, list):
                max_list = max(max_list, len(val))
            elif isinstance(val, str):
                max_str = max(max_str, len(val))
            elif isinstance(val, int) and not isinstance(val, bool):
                max_int = max(max_int, val)
                ints.append(val)
        if len(ints) >= 2:
            max_span = max(max_span, max(ints) - min(ints))
    return max_list, max_str, max_int, max_span


def constraint_scale_required(
    public_checks: str,
    entry_point: str,
    spec: str,
    *,
    signature: str = "",
) -> bool:
    """True when parsed constraints imply inputs beyond public-example scale."""
    constraints = parse_task_constraints(spec)
    if constraints is None:
        return False
    calls = parse_public_call_arglists(public_checks, entry_point)
    if not calls:
        return False
    pub_list, pub_str, _, pub_span = _public_metrics(calls)
    for bound in constraints.length_max.values():
        if pub_list > 0 and bound / pub_list >= _SCALE_RATIO:
            return True
        if pub_str > 0 and bound / pub_str >= _SCALE_RATIO:
            return True
    for bound in constraints.range_upper.values():
        if bound >= _RANGE_SPAN_THRESHOLD and pub_span < _RANGE_SPAN_THRESHOLD:
            return True
    names = _param_names(signature, entry_point)
    if len(names) >= 2 and constraints.range_upper:
        lo_name, hi_name = names[-2].lower(), names[-1].lower()
        if hi_name in constraints.range_upper:
            hi_bound = constraints.range_upper[hi_name]
            if hi_bound >= _RANGE_SPAN_THRESHOLD and pub_span < _RANGE_SPAN_THRESHOLD:
                return True
        if lo_name in constraints.range_upper or hi_name in constraints.range_upper:
            return True
    return False


def _stress_list(val: list[Any], target_len: int) -> list[Any]:
    if not val:
        return [0] * target_len
    out = list(val)
    while len(out) < target_len:
        out.extend(val)
    return out[:target_len]


def _stress_value(
    val: Any,
    *,
    param_name: str,
    constraints: TaskConstraints,
    public_list_max: int,
    public_str_max: int,
) -> Any:
    name = param_name.lower()
    if isinstance(val, list) and constraints.length_max:
        bound = constraints.length_max.get(name)
        if bound is None:
            bound = max(constraints.length_max.values(), default=0)
        if bound and public_list_max > 0 and bound / public_list_max >= _SCALE_RATIO:
            target = min(bound, _MAX_LIST_STRESS)
            return _stress_list(val, max(target, public_list_max * _SCALE_RATIO))
    if isinstance(val, str) and constraints.length_max:
        bound = constraints.length_max.get(name)
        if bound is None:
            bound = max(constraints.length_max.values(), default=0)
        if bound and public_str_max > 0 and bound / public_str_max >= _SCALE_RATIO:
            target = min(bound, _MAX_STRING_STRESS)
            repeat = max(1, target // max(len(val), 1))
            return (val * repeat)[:target]
    return val


def _stress_range_args(
    args: list[Any],
    param_names: list[str],
    constraints: TaskConstraints,
) -> list[Any] | None:
    if len(args) < 2 or not constraints.range_upper:
        return None
    hi_name = param_names[-1].lower() if len(param_names) >= 1 else ""
    hi_bound = constraints.range_upper.get(hi_name) or max(
        constraints.range_upper.values(), default=0
    )
    if hi_bound < _RANGE_SPAN_THRESHOLD:
        return None
    if not all(isinstance(a, int) and not isinstance(a, bool) for a in args[-2:]):
        return None
    stressed = list(args)
    span = min(hi_bound - 1, _MAX_RANGE_STRESS)
    stressed[-2] = 1
    stressed[-1] = 1 + span
    return stressed


def build_constraint_scale_probe(
    code: str,
    *,
    entry_point: str,
    spec: str,
    public_checks: str,
    signature: str = "",
) -> str | None:
    """Stress-test script, or None when constraints do not require a probe."""
    if not constraint_scale_required(
        public_checks, entry_point, spec, signature=signature
    ):
        return None
    constraints = parse_task_constraints(spec)
    if constraints is None:
        return None
    calls = parse_public_call_arglists(public_checks, entry_point)
    if not calls:
        return None
    pub_list, pub_str, _, _ = _public_metrics(calls)
    names = _param_names(signature, entry_point)
    stressed_calls: list[list[Any]] = []
    for args in calls:
        range_stress = _stress_range_args(args, names, constraints)
        if range_stress is not None:
            stressed_calls.append(range_stress)
            continue
        stressed = [
            _stress_value(
                v,
                param_name=names[i] if i < len(names) else "",
                constraints=constraints,
                public_list_max=pub_list,
                public_str_max=pub_str,
            )
            for i, v in enumerate(args)
        ]
        stressed_calls.append(stressed)
    lines = [code.strip(), ""]
    for i, args in enumerate(stressed_calls):
        lines.append(f"_constraint_scale_{i} = {entry_point}(*{args!r})")
    from rune.engine.oracle import with_probe_imports  # noqa: PLC0415

    return with_probe_imports("\n".join(lines))


def check_constraint_scale(
    code: str,
    *,
    entry_point: str,
    spec: str,
    public_checks: str,
    signature: str = "",
    timeout_s: int = _PROBE_TIMEOUT_S,
) -> ScaleProbeOutcome:
    """Run constraint-scale probe when the task Constraints block requires it."""
    required = constraint_scale_required(
        public_checks, entry_point, spec, signature=signature
    )
    if not required:
        return ScaleProbeOutcome(required=False, ok=True)
    probe = build_constraint_scale_probe(
        code,
        entry_point=entry_point,
        spec=spec,
        public_checks=public_checks,
        signature=signature,
    )
    if probe is None:
        return ScaleProbeOutcome(required=False, ok=True)
    result = run_in_sandbox(probe, timeout=timeout_s)
    if result.exit_code == -1:
        return ScaleProbeOutcome(
            required=True,
            ok=False,
            message=(
                f"constraint_scale: timed out ({timeout_s}s) on constraint-scale "
                f"input (Constraints allow much larger inputs than public "
                f"examples — need a faster algorithm)"
            ),
        )
    return ScaleProbeOutcome(required=True, ok=True)


# Back-compat aliases for tests/tools written against the earlier API.
ComplexityResult = ScaleProbeOutcome
complexity_probe_required = constraint_scale_required
build_complexity_probe = build_constraint_scale_probe


def check_constraint_complexity(
    code: str,
    *,
    entry_point: str,
    spec: str,
    public_checks: str,
    signature: str = "",
    timeout_s: int = _PROBE_TIMEOUT_S,
) -> ScaleProbeOutcome:
    """Back-compat wrapper around :func:`check_constraint_scale`."""
    return check_constraint_scale(
        code,
        entry_point=entry_point,
        spec=spec,
        public_checks=public_checks,
        signature=signature,
        timeout_s=timeout_s,
    )


def format_complexity_feedback(message: str) -> str:
    """Back-compat; prefer :func:`requirements.format_requirements_feedback`."""
    return f"Task requirements failed — fix exactly:\n- {message}"
