"""Constraint-scale complexity probes for the task requirements oracle.

When a task ``Constraints:`` block allows inputs much larger than the public
examples, the engine first tries empirical ``big_o`` fitting (bounded wall clock).
If that exceeds the budget, a dedicated complexity-assessment adapter judges
asymptotic class from static code structure.
"""

from __future__ import annotations

import ast
import concurrent.futures
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, cast

from rune.engine.oracle import parse_public_call_arglists

_SCALE_RATIO = 8
_RANGE_SPAN_THRESHOLD = 100_000
_MAX_LIST_STRESS = 40
_MAX_STRING_STRESS = 5_000
_MAX_RANGE_STRESS = 2_500_000

_COMPLEXITY_RANK: dict[str, int] = {
    "Constant": 0,
    "Logarithmic": 1,
    "Linear": 2,
    "Linearithmic": 3,
    "Quadratic": 4,
    "Cubic": 5,
    "Exponential": 6,
    "Factorial": 7,
}

_COMPLEXITY_NOTATION: dict[str, str] = {
    "Constant": "O(1)",
    "Logarithmic": "O(log n)",
    "Linear": "O(n)",
    "Linearithmic": "O(n log n)",
    "Quadratic": "O(n²)",
    "Cubic": "O(n³)",
    "Exponential": "O(2^n)",
    "Factorial": "O(n!)",
}

_RANK_TO_NOTATION: dict[int, str] = {
    rank: _COMPLEXITY_NOTATION[name] for name, rank in _COMPLEXITY_RANK.items()
}


def _notation_for_rank(rank: int) -> str:
    """Big-O notation for a complexity rank (the highest known class at/below it)."""
    if rank in _RANK_TO_NOTATION:
        return _RANK_TO_NOTATION[rank]
    best = min(_RANK_TO_NOTATION)
    for r in sorted(_RANK_TO_NOTATION):
        if r <= rank:
            best = r
    return _RANK_TO_NOTATION[best]


COMPLEXITY_ANALYSIS_RUBRIC = """\
1. Nested loops over the same input dimension multiply: depth d → up to O(n^d).
2. `for i in range(l, r+1)` or scanning a range of width W → O(W); if W scales with
   a constraint bound, treat as O(n) in that dimension.
3. `itertools.combinations` / `permutations` / recursive subset enumeration →
   typically O(2^n) or O(n!).
4. Sorting costs O(n log n); a loop with a sort inside → O(n log n) or worse.
5. Standard DP over n states with O(1) transitions → O(n); DP with n² states → O(n²).
6. Binary search on a monotone predicate → O(log n) per query.
7. Compare your assessed class to the required bound — do not confuse correctness
   on small public inputs with feasibility at constraint scale.
"""


@dataclass(frozen=True)
class ComplexityProbeConfig:
    """Settings for empirical complexity measurement."""

    min_n: int = 8
    max_n: int = 400
    n_repeats: int = 3
    per_run_timeout_s: float = 5.0

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> ComplexityProbeConfig:
        from rune.config import PipelineConfig  # noqa: PLC0415

        defaults = PipelineConfig()
        return cls(
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
        )


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


def _parse_int_bound(raw: str) -> int | None:
    """Parse a numeric constraint bound, e.g. ``10^5``, ``5*10^4``, ``2 * 10^9``.

    Returns ``None`` for any form we cannot parse rather than raising, so an
    unforeseen notation no-ops instead of killing the whole benchmark run.
    """
    text = raw.strip().replace(" ", "").rstrip(".,;")
    try:
        product = 1
        for factor in text.split("*"):
            if "^" in factor:
                base, exp = factor.split("^", 1)
                product *= int(base) ** int(exp)
            else:
                product *= int(factor)
        return product
    except (ValueError, TypeError):
        return None


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
            if bound is not None:
                length_max[name] = max(length_max.get(name, 0), bound)
            continue
        m = re.match(
            r"1\s*<=\s*(\w+)\s*<=\s*(\w+)\s*<\s*(.+)$",
            text,
            re.IGNORECASE,
        )
        if m:
            hi = m.group(2).lower()
            bound = _parse_int_bound(m.group(3))
            if bound is not None:
                range_upper[hi] = max(range_upper.get(hi, 0), bound)
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
            return [a.arg for a in node.args.args if a.arg not in ("self", "cls")]
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


def extract_constraints_block(spec: str) -> str:
    """Return the raw ``Constraints:`` section from a task description."""
    lower = spec.lower()
    marker = "constraints:"
    idx = lower.find(marker)
    if idx < 0:
        return "(no Constraints section)"
    block = spec[idx:].strip()
    stop = re.search(r"\n\s*\n\s*[A-Z]", block[len(marker) :])
    if stop:
        block = block[: len(marker) + stop.start()]
    return block


def static_complexity_signals(code: str) -> list[str]:
    """Deterministic pre-scan hints for the adapter complexity judge."""
    signals: list[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ["(could not parse code for static scan)"]

    class _LoopVisitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.max_depth = 0
            self._depth = 0

        def visit_For(self, node: ast.For) -> None:
            self._depth += 1
            self.max_depth = max(self.max_depth, self._depth)
            self.generic_visit(node)
            self._depth -= 1

        def visit_While(self, node: ast.While) -> None:
            self._depth += 1
            self.max_depth = max(self.max_depth, self._depth)
            self.generic_visit(node)
            self._depth -= 1

    visitor = _LoopVisitor()
    visitor.visit(tree)
    if visitor.max_depth >= 2:
        signals.append(f"nested loops: depth {visitor.max_depth} (multiplicative cost)")

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "itertools":
            names = [alias.name for alias in node.names]
            if any(n in ("combinations", "permutations", "product") for n in names):
                signals.append(f"itertools combinatorial import: {', '.join(names)}")
        if isinstance(node, ast.Call):
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id == "itertools"
                and func.attr in ("combinations", "permutations", "product")
            ):
                signals.append(f"itertools.{func.attr}() call")
            if isinstance(func, ast.Name) and func.id == "sorted":
                signals.append("sorted() call (typically O(n log n))")
            if (
                isinstance(func, ast.Name)
                and func.id == "range"
                and len(node.args) >= 2
            ):
                signals.append("range(l, r) style iteration over a span")

    if not signals:
        signals.append("(no strong combinatorial or deep-loop signals detected)")
    return signals


def static_complexity_floor_rank(code: str) -> int:
    """Minimum asymptotic rank implied by static structure (not small-n timing)."""
    joined = " ".join(static_complexity_signals(code)).lower()
    if "combinations" in joined or "permutations" in joined or "product" in joined:
        return _COMPLEXITY_RANK["Exponential"]
    if "nested loops" in joined:
        depth_match = re.search(r"depth (\d+)", joined)
        if depth_match:
            depth = int(depth_match.group(1))
            if depth >= 3:
                return _COMPLEXITY_RANK["Cubic"]
            if depth >= 2:
                return _COMPLEXITY_RANK["Quadratic"]
    if "range(l, r)" in joined:
        return _COMPLEXITY_RANK["Linear"]
    return 0


def build_complexity_assessment_task(
    spec: str,
    entry_point: str,
    *,
    signature: str = "",
) -> str:
    """Adapter ``## Task`` text for the time-complexity assessment trajectory."""
    constraints = extract_constraints_block(spec)
    sig = signature.strip()
    sig_line = f"\nStarter signature:\n{sig}" if sig else ""
    return (
        f"Assess TIME complexity for `{entry_point}` against the task Constraints.\n"
        "Public examples already pass — determine asymptotic feasibility at scale.\n\n"
        f"{constraints}"
        f"{sig_line}\n\n"
        f"Analysis rubric:\n{COMPLEXITY_ANALYSIS_RUBRIC}"
    )


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


def constraint_max_n(constraints: TaskConstraints) -> int:
    return max(
        max(constraints.length_max.values(), default=0),
        max(constraints.range_upper.values(), default=0),
    )


def allowed_complexity_for_max_n(max_n: int) -> tuple[str, int]:
    """Return ``(notation, rank)`` of the slowest feasible class for bound *max_n*.

    Uses standard competitive-programming feasibility (operations budget ~10^8).
    """
    if max_n <= 10:
        label = _COMPLEXITY_NOTATION["Factorial"]
        return label, _COMPLEXITY_RANK["Factorial"]
    if max_n <= 20:
        label = _COMPLEXITY_NOTATION["Exponential"]
        return label, _COMPLEXITY_RANK["Exponential"]
    if max_n <= 500:
        label = _COMPLEXITY_NOTATION["Cubic"]
        return label, _COMPLEXITY_RANK["Cubic"]
    if max_n <= 5_000:
        label = _COMPLEXITY_NOTATION["Quadratic"]
        return label, _COMPLEXITY_RANK["Quadratic"]
    if max_n <= 100_000:
        label = _COMPLEXITY_NOTATION["Linearithmic"]
        return label, _COMPLEXITY_RANK["Linearithmic"]
    if max_n <= 1_000_000:
        label = _COMPLEXITY_NOTATION["Linear"]
        return label, _COMPLEXITY_RANK["Linear"]
    label = _COMPLEXITY_NOTATION["Logarithmic"]
    return label, _COMPLEXITY_RANK["Logarithmic"]


def _probe_floor_rank(label: str) -> int:
    """Minimum asymptotic rank implied by how this probe scales *n*.

    Empirical fits at small ``n`` can under-report (e.g. a linear range scan
    looking ``O(log n)`` before the loop term dominates). Floors keep the gate
    tied to the scaled dimension, not a single wall-clock sample.
    """
    if label.startswith(("range_span", "list_len", "str_len")):
        return _COMPLEXITY_RANK["Linear"]
    return 0


def measured_complexity_rank(measured: Any) -> int:
    """Map a ``big_o`` best-fit class to a comparable rank."""
    name = type(measured).__name__
    if name == "Polynomial":
        coef = getattr(measured, "coef", None)
        if coef is not None and len(coef) >= 2:
            exponent = float(coef[1])
            if exponent >= 2.5:
                return _COMPLEXITY_RANK["Cubic"]
            if exponent >= 1.5:
                return _COMPLEXITY_RANK["Quadratic"]
            if exponent > 1.05:
                return _COMPLEXITY_RANK["Linearithmic"]
            if exponent > 0.5:
                return _COMPLEXITY_RANK["Linear"]
        return _COMPLEXITY_RANK["Linear"]
    return _COMPLEXITY_RANK.get(name, _COMPLEXITY_RANK["Factorial"])


def format_measured_complexity(measured: Any) -> str:
    """Human-readable notation for a ``big_o`` best-fit class."""
    name = type(measured).__name__
    if name == "Polynomial":
        coef = getattr(measured, "coef", None)
        if coef is not None and len(coef) >= 2:
            exponent = float(coef[1])
            if abs(exponent - round(exponent)) < 0.15:
                exp_int = int(round(exponent))
                if exp_int == 1:
                    return "O(n)"
                return f"O(n^{exp_int})"
            return f"O(n^{exponent:.2f})"
        return "O(n^p)"
    return _COMPLEXITY_NOTATION.get(name, name)


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
    n: int,
) -> Any:
    name = param_name.lower()
    if isinstance(val, list) and constraints.length_max:
        bound = constraints.length_max.get(name)
        if bound is None:
            bound = max(constraints.length_max.values(), default=0)
        if bound and public_list_max > 0 and bound / public_list_max >= _SCALE_RATIO:
            target = min(
                bound,
                _MAX_LIST_STRESS,
                max(n, public_list_max * _SCALE_RATIO),
            )
            return _stress_list(val, target)
    if isinstance(val, str) and constraints.length_max:
        bound = constraints.length_max.get(name)
        if bound is None:
            bound = max(constraints.length_max.values(), default=0)
        if bound and public_str_max > 0 and bound / public_str_max >= _SCALE_RATIO:
            target = min(
                bound,
                _MAX_STRING_STRESS,
                max(n, public_str_max * _SCALE_RATIO),
            )
            repeat = max(1, target // max(len(val), 1))
            return (val * repeat)[:target]
    return val


def _build_scaling_probes(
    fn: Callable[..., Any],
    *,
    entry_point: str,
    spec: str,
    public_checks: str,
    signature: str,
) -> list[tuple[str, Callable[[int], Any], int]]:
    """Return ``(label, probe, effective_max_n)`` triples for ``big_o`` measurement."""
    constraints = parse_task_constraints(spec)
    if constraints is None:
        return []
    calls = parse_public_call_arglists(public_checks, entry_point)
    if not calls:
        return []
    pub_list, pub_str, _, pub_span = _public_metrics(calls)
    names = _param_names(signature, entry_point)
    probes: list[tuple[str, Callable[[int], Any], int]] = []

    for args in calls:
        if (
            len(args) >= 2
            and constraints.range_upper
            and all(isinstance(a, int) and not isinstance(a, bool) for a in args[-2:])
        ):
            hi_name = names[-1].lower() if names else ""
            hi_bound = constraints.range_upper.get(hi_name) or max(
                constraints.range_upper.values(), default=0
            )
            if hi_bound >= _RANGE_SPAN_THRESHOLD and pub_span < _RANGE_SPAN_THRESHOLD:
                cap = min(hi_bound - 1, _MAX_RANGE_STRESS)

                def _range_probe(n: int, _fn: Callable[..., Any] = fn) -> Any:
                    span = max(1, n)
                    return _fn(1, 1 + span)

                probes.append(("range_span", _range_probe, cap))
                continue

        scaled_template = list(args)
        for i, val in enumerate(args):
            if isinstance(val, list) and constraints.length_max:
                name = names[i].lower() if i < len(names) else ""
                bound = constraints.length_max.get(name) or max(
                    constraints.length_max.values(), default=0
                )
                if bound and pub_list > 0 and bound / pub_list >= _SCALE_RATIO:
                    cap = min(bound, _MAX_LIST_STRESS)

                    def _list_probe(
                        n: int,
                        _fn: Callable[..., Any] = fn,
                        _template: list[Any] = scaled_template,
                        _idx: int = i,
                        _val: list[Any] = val,
                        _names: list[str] = names,
                        _constraints: TaskConstraints = constraints,
                        _pub_list: int = pub_list,
                        _pub_str: int = pub_str,
                    ) -> Any:
                        stressed = list(_template)
                        stressed[_idx] = _stress_value(
                            _val,
                            param_name=_names[_idx] if _idx < len(_names) else "",
                            constraints=_constraints,
                            public_list_max=_pub_list,
                            public_str_max=_pub_str,
                            n=n,
                        )
                        return _fn(*stressed)

                    probes.append((f"list_len:{name or 'arg'}", _list_probe, cap))
                    break
            if isinstance(val, str) and constraints.length_max:
                name = names[i].lower() if i < len(names) else ""
                bound = constraints.length_max.get(name) or max(
                    constraints.length_max.values(), default=0
                )
                if bound and pub_str > 0 and bound / pub_str >= _SCALE_RATIO:
                    cap = min(bound, _MAX_STRING_STRESS)

                    def _str_probe(
                        n: int,
                        _fn: Callable[..., Any] = fn,
                        _template: list[Any] = scaled_template,
                        _idx: int = i,
                        _val: str = val,
                        _names: list[str] = names,
                        _constraints: TaskConstraints = constraints,
                        _pub_list: int = pub_list,
                        _pub_str: int = pub_str,
                    ) -> Any:
                        stressed = list(_template)
                        stressed[_idx] = _stress_value(
                            _val,
                            param_name=_names[_idx] if _idx < len(_names) else "",
                            constraints=_constraints,
                            public_list_max=_pub_list,
                            public_str_max=_pub_str,
                            n=n,
                        )
                        return _fn(*stressed)

                    probes.append((f"str_len:{name or 'arg'}", _str_probe, cap))
                    break

    return probes


def _load_entry_callable(code: str, entry_point: str) -> Callable[..., Any]:
    from rune.bench.lcb import extract_entry_function  # noqa: PLC0415
    from rune.engine.oracle import with_probe_imports  # noqa: PLC0415

    normalized = extract_entry_function(code, entry_point)
    if not normalized.strip():
        msg = "entry_point: could not extract implementation for complexity probe"
        raise ValueError(msg)
    namespace: dict[str, Any] = {}
    exec(compile(with_probe_imports(normalized), "<complexity>", "exec"), namespace)
    fn = namespace.get(entry_point)
    if not callable(fn):
        msg = f"entry_point: `{entry_point}` is not callable after load"
        raise TypeError(msg)
    return cast("Callable[..., Any]", fn)


def _timed_probe(
    probe: Callable[[int], Any],
    n: int,
    *,
    timeout_s: float,
) -> Any:
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(probe, n)
        return fut.result(timeout=timeout_s)


def _measure_complexity(
    probe: Callable[[int], Any],
    *,
    config: ComplexityProbeConfig,
    effective_max_n: int,
) -> Any:
    import big_o  # noqa: PLC0415

    max_n = max(config.min_n + 1, min(config.max_n, effective_max_n))
    min_n = min(config.min_n, max_n - 1)

    def _wrapped(n: int) -> Any:
        return _timed_probe(probe, n, timeout_s=config.per_run_timeout_s)

    best, _others = big_o.big_o(
        _wrapped,
        big_o.datagen.n_,
        min_n=min_n,
        max_n=max_n,
        n_repeats=config.n_repeats,
    )
    return best


def build_constraint_scale_probe(
    code: str,
    *,
    entry_point: str,
    spec: str,
    public_checks: str,
    signature: str = "",
) -> str | None:
    """Back-compat: retained for tools that expect a sandbox script (unused by gate)."""
    if not constraint_scale_required(
        public_checks, entry_point, spec, signature=signature
    ):
        return None
    return code.strip() or None


def check_constraint_scale(
    code: str,
    *,
    entry_point: str,
    spec: str,
    public_checks: str,
    signature: str = "",
    probe_config: ComplexityProbeConfig | None = None,
    # Back-compat alias; mapped into probe_config when provided alone.
    timeout_s: int | None = None,
) -> ScaleProbeOutcome:
    """Measure asymptotic complexity when Constraints require scale beyond publics."""
    required = constraint_scale_required(
        public_checks, entry_point, spec, signature=signature
    )
    if not required:
        return ScaleProbeOutcome(required=False, ok=True)

    constraints = parse_task_constraints(spec)
    if constraints is None:
        return ScaleProbeOutcome(required=False, ok=True)

    max_n = constraint_max_n(constraints)
    allowed_label, allowed_rank = allowed_complexity_for_max_n(max_n)
    static_rank = static_complexity_floor_rank(code)
    if static_rank > allowed_rank:
        # Static structure alone proves infeasibility — skip the empirical big_o
        # probe (never run combinatorial/deep-nested code at constraint scale).
        return ScaleProbeOutcome(
            required=True,
            ok=False,
            message=(
                f"constraint_scale: static analysis indicates "
                f"{_notation_for_rank(static_rank)}; Constraints allow n≤{max_n} "
                f"— need {allowed_label} or better"
            ),
        )

    if probe_config is None:
        from rune.config import PipelineConfig  # noqa: PLC0415

        defaults = PipelineConfig()
        per_run = (
            float(timeout_s)
            if timeout_s is not None
            else defaults.complexity_probe_per_run_timeout_s
        )
        probe_config = ComplexityProbeConfig(
            min_n=defaults.complexity_probe_min_n,
            max_n=defaults.complexity_probe_max_n,
            n_repeats=defaults.complexity_probe_n_repeats,
            per_run_timeout_s=per_run,
        )

    try:
        fn = _load_entry_callable(code, entry_point)
    except (SyntaxError, TypeError, ValueError) as exc:
        return ScaleProbeOutcome(
            required=True,
            ok=False,
            message=f"constraint_scale: could not load implementation ({exc})",
        )

    probes = _build_scaling_probes(
        fn,
        entry_point=entry_point,
        spec=spec,
        public_checks=public_checks,
        signature=signature,
    )
    if not probes:
        return ScaleProbeOutcome(required=False, ok=True)

    worst_rank = static_rank
    worst_label = ""
    worst_measured: Any = None
    if worst_rank >= _COMPLEXITY_RANK["Exponential"]:
        worst_label = _COMPLEXITY_NOTATION["Exponential"]
    elif worst_rank >= _COMPLEXITY_RANK["Quadratic"]:
        worst_label = _COMPLEXITY_NOTATION["Quadratic"]
    elif worst_rank >= _COMPLEXITY_RANK["Linear"]:
        worst_label = _COMPLEXITY_NOTATION["Linear"]

    for label, probe, cap in probes:
        try:
            measured = _measure_complexity(
                probe, config=probe_config, effective_max_n=cap
            )
        except concurrent.futures.TimeoutError:
            return ScaleProbeOutcome(
                required=True,
                ok=False,
                message=(
                    "constraint_scale: complexity measurement timed out during "
                    f"sampling (per-run safety cap {probe_config.per_run_timeout_s}s) "
                    f"— likely exponential or worse for n≤{max_n}"
                ),
            )
        except Exception as exc:
            return ScaleProbeOutcome(
                required=True,
                ok=False,
                message=f"constraint_scale: could not measure complexity ({exc})",
            )
        rank = max(
            measured_complexity_rank(measured),
            _probe_floor_rank(label),
            static_complexity_floor_rank(code),
        )
        if rank > worst_rank:
            worst_rank = rank
            worst_label = format_measured_complexity(measured)
            if rank > measured_complexity_rank(measured):
                worst_label = f">= {_COMPLEXITY_NOTATION.get('Linear', 'O(n)')}"
            worst_measured = measured

    if worst_rank > allowed_rank:
        source = (
            f"({type(worst_measured).__name__})"
            if worst_measured is not None
            else "(static analysis)"
        )
        return ScaleProbeOutcome(
            required=True,
            ok=False,
            message=(
                f"constraint_scale: measured {worst_label} "
                f"{source}; Constraints allow n≤{max_n} "
                f"— need {allowed_label} or better"
            ),
        )
    return ScaleProbeOutcome(required=True, ok=True)


def _complexity_probe_worker(
    q: Any,
    code: str,
    entry_point: str,
    spec: str,
    public_checks: str,
    signature: str,
    probe_config_dict: dict[str, Any],
) -> None:
    """Subprocess body: run the empirical probe and return the outcome via *q*."""
    try:
        cfg = ComplexityProbeConfig(**probe_config_dict)
        outcome = check_constraint_scale(
            code,
            entry_point=entry_point,
            spec=spec,
            public_checks=public_checks,
            signature=signature,
            probe_config=cfg,
        )
        q.put(("ok", outcome))
    except Exception as exc:  # noqa: BLE001 - surfaced to the parent as a string
        q.put(("err", repr(exc)))


def check_constraint_scale_guarded(
    code: str,
    *,
    entry_point: str,
    spec: str,
    public_checks: str,
    signature: str = "",
    probe_config: ComplexityProbeConfig | None = None,
    wall_timeout_s: float = 15.0,
) -> ScaleProbeOutcome | None:
    """Run :func:`check_constraint_scale` in a hard-killable subprocess.

    A Python thread cannot be killed, so an empirical ``big_o`` probe on a slow
    implementation (e.g. exponential recursion the static floor can't see) would
    run to completion and stall the whole benchmark via GIL contention. Isolating
    it in a ``spawn``-ed process lets us hard-kill on the wall budget. Returns the
    outcome, or ``None`` when the budget was exceeded (caller escalates to the
    adapter judge). ``spawn`` (never ``fork``) avoids copying a CUDA-initialised
    parent; the child imports only the torch-free complexity module.
    """
    import multiprocessing as mp  # noqa: PLC0415

    if probe_config is None:
        probe_config = ComplexityProbeConfig()
    cfg_dict = {
        "min_n": probe_config.min_n,
        "max_n": probe_config.max_n,
        "n_repeats": probe_config.n_repeats,
        "per_run_timeout_s": probe_config.per_run_timeout_s,
    }
    ctx = mp.get_context("spawn")
    q: Any = ctx.Queue()
    proc = ctx.Process(
        target=_complexity_probe_worker,
        args=(q, code, entry_point, spec, public_checks, signature, cfg_dict),
    )
    proc.start()
    proc.join(wall_timeout_s)
    if proc.is_alive():
        proc.kill()
        proc.join()
        return None
    try:
        status, payload = q.get(timeout=5.0)
    except Exception:  # noqa: BLE001 - empty/closed queue => treat as overran
        return None
    if status == "ok":
        return payload  # type: ignore[no-any-return]
    return ScaleProbeOutcome(
        required=True,
        ok=False,
        message=f"constraint_scale: could not measure complexity ({payload})",
    )


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
    probe_config: ComplexityProbeConfig | None = None,
    timeout_s: int | None = None,
) -> ScaleProbeOutcome:
    """Back-compat wrapper around :func:`check_constraint_scale`."""
    return check_constraint_scale(
        code,
        entry_point=entry_point,
        spec=spec,
        public_checks=public_checks,
        signature=signature,
        probe_config=probe_config,
        timeout_s=timeout_s,
    )


def format_complexity_feedback(message: str) -> str:
    """Back-compat; prefer :func:`requirements.format_requirements_feedback`."""
    return f"Task requirements failed — fix exactly:\n- {message}"
