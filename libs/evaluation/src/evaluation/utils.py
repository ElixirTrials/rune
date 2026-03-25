"""Shared utilities for evaluation benchmarks.

Organised into five sections:

1. **Subprocess Execution** — safe_subprocess_run (used by HumanEval / OOD runners)
2. **Template Rendering** — Jinja2 rendering + system-prompt resolution
3. **Answer Extraction & Scoring** — boxed extraction, numeric matching, scorer registry
4. **Dataset Loading** — integer filtering, path resolution, Arrow dataset loading
5. **Results & Output** — slug helpers, error rows, summary computation, file I/O

Tool-calling helpers (MATH_TOOLS schema, XML parser, near-int hint) live in
``shared.mathbox`` alongside the ``MathSandbox`` they serve.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ── 1. Subprocess Execution ───────────────────────────────────────────────────


def safe_subprocess_run(
    script_path: Path,
    cwd: str,
    timeout: int = 30,
) -> bool:
    """Run a Python script in a subprocess and return whether it passed.

    Handles ``subprocess.TimeoutExpired`` by returning ``False``.
    Used by both HumanEval and OOD benchmark runners to avoid duplicated
    try/except + subprocess.run boilerplate.

    Args:
        script_path: Path to the Python script to execute.
        cwd: Working directory for the subprocess.
        timeout: Maximum execution time in seconds before the process is killed.

    Returns:
        ``True`` if the process exits with return code 0, ``False`` otherwise
        (including timeout).

    Example:
        >>> from pathlib import Path
        >>> passed = safe_subprocess_run(Path("/tmp/test.py"), cwd="/tmp")
        >>> isinstance(passed, bool)
        True
    """
    try:
        proc = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
        return proc.returncode == 0
    except subprocess.TimeoutExpired:
        return False


# ── 2. Template Rendering ─────────────────────────────────────────────────────


def render_template(
    template_name: str,
    template_vars: dict[str, Any] | None = None,
    templates_dir: Path | str | None = None,
) -> str:
    """Render a Jinja2 template from the shared templates directory.

    Args:
        template_name: Filename of the template, e.g. ``"math_prompt_v2.j2"``.
        template_vars: Variables forwarded to the template context.
        templates_dir: Override the template search directory.  Defaults to
            ``libs/shared/src/shared/templates/``.

    Returns:
        Rendered string with leading/trailing newlines preserved.

    Example::

        prompt = render_template("math_prompt_v2.j2", {"answer_range": "0 to 99999"})
    """
    from jinja2 import Environment, FileSystemLoader  # deferred

    from evaluation.config import _TEMPLATES_DIR

    tdir = Path(templates_dir) if templates_dir else _TEMPLATES_DIR
    env = Environment(loader=FileSystemLoader(str(tdir)), keep_trailing_newline=True)
    return env.get_template(template_name).render(**(template_vars or {}))


def _resolve_system_prompt(
    ds_cfg: Any,
    run_cfg: Any,
) -> str:
    """Resolve the effective system prompt for a dataset.

    Priority (highest → lowest):

    1. Dataset-level ``system_prompt_template`` (rendered with merged vars)
    2. Run-level ``system_prompt_template`` (rendered with run-level vars)
    3. Dataset-level inline ``system_prompt``
    4. Run-level inline ``system_prompt``
    5. Empty string

    Args:
        ds_cfg: ``DatasetConfig`` instance.
        run_cfg: ``BenchmarkRunConfig`` instance.

    Returns:
        Rendered or inline system prompt string.
    """
    tdir = run_cfg.templates_dir  # may be None → uses default

    if ds_cfg.system_prompt_template:
        merged_vars = {**run_cfg.template_vars, **ds_cfg.template_vars}
        return render_template(ds_cfg.system_prompt_template, merged_vars, tdir)

    if run_cfg.system_prompt_template:
        return render_template(run_cfg.system_prompt_template, run_cfg.template_vars, tdir)

    return ds_cfg.system_prompt or run_cfg.system_prompt or ""


# ── 3. Answer Extraction & Scoring ────────────────────────────────────────────

_NEAR_INT_TOL = 1e-4


def _extract_boxed(text: str) -> str | None:
    """Return the content of the last ``\\boxed{…}`` expression in *text*."""
    results: list[str] = []
    i = 0
    while i < len(text):
        idx = text.find(r"\boxed{", i)
        if idx == -1:
            break
        depth = 0
        start = idx + len(r"\boxed{")
        j = start
        while j < len(text):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                if depth == 0:
                    results.append(text[start:j])
                    break
                depth -= 1
            j += 1
        i = idx + 1
    return results[-1] if results else None


def _to_number(s: str) -> float | None:
    s = s.strip().replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def _split_thinking(raw: str) -> tuple[str, str]:
    """Split ``<think>…</think>`` prefix from the rest of the response."""
    if "</think>" in raw:
        idx = raw.index("</think>")
        return raw[:idx].strip(), raw[idx + len("</think>"):].strip()
    return "", raw.strip()


def _clean_response(text: str) -> str:
    return re.sub(r"<\|im_end\|>|<\|endoftext\|>", "", text).strip()


def _answers_match(pred: str, gt: str) -> bool:
    pred, gt = pred.strip(), gt.strip()
    if pred == gt:
        return True
    pn, gn = _to_number(pred), _to_number(gt)
    if pn is not None and gn is not None:
        if pn == gn:
            return True
        if abs(pn - gn) < _NEAR_INT_TOL and abs(pn - round(pn)) < _NEAR_INT_TOL:
            return int(round(pn)) == int(round(gn))
    return False


def _extract_answer(text: str) -> str | None:
    """Extract the best answer from a raw model output."""
    thinking, body = _split_thinking(text)
    response = _clean_response(body)
    return (
        _extract_boxed(response)
        or _extract_boxed(thinking)
        or _extract_boxed(text)
    )


def score_boxed_extract(prediction: str | None, ground_truth: str) -> bool:
    """Score by extracting ``\\boxed{}`` from the prediction then matching."""
    if prediction is None:
        return False
    candidate = _extract_boxed(prediction) or prediction.strip()
    return _answers_match(candidate, ground_truth)


def score_exact_match(prediction: str | None, ground_truth: str) -> bool:
    """Case-insensitive exact match after stripping whitespace."""
    if prediction is None:
        return False
    return prediction.strip().lower() == ground_truth.strip().lower()


def score_numeric(prediction: str | None, ground_truth: str) -> bool:
    """Numeric equality with near-integer tolerance."""
    if prediction is None:
        return False
    return _answers_match(prediction, ground_truth)


SCORERS: dict[str, Callable[[str | None, str], bool]] = {
    "boxed_extract": score_boxed_extract,
    "exact_match": score_exact_match,
    "numeric": score_numeric,
}


# ── 4. Dataset Loading ────────────────────────────────────────────────────────


def _integer_in_range(value: object, lo: int = 0, hi: int = 99999) -> int | None:
    """Parse *value* as a non-negative integer within ``[lo, hi]``.

    Accepts plain ``int``, float-valued ints (e.g. ``42.0``), and digit-only
    strings (commas stripped).  Returns ``None`` for LaTeX expressions,
    multi-part answers, fractions, or values outside the range.

    Args:
        value: Raw answer value from the dataset row.
        lo: Inclusive lower bound (default 0).
        hi: Inclusive upper bound (default 99999).

    Returns:
        Parsed integer, or ``None`` if the value is not a valid integer answer.

    Examples::

        >>> _integer_in_range(42)
        42
        >>> _integer_in_range("42.0")
        42
        >>> _integer_in_range("\\\\frac{1}{2}")   # LaTeX → None
        >>> _integer_in_range(100000)              # out of range → None
    """
    if isinstance(value, int):
        return value if lo <= value <= hi else None
    if isinstance(value, float) and value == int(value):
        v = int(value)
        return v if lo <= v <= hi else None
    if isinstance(value, str):
        s = value.strip().replace(",", "")
        if s.lstrip("-").isdigit():
            v = int(s)
            return v if lo <= v <= hi else None
    return None


def _resolve_data_path(ds_cfg: Any) -> Path:
    from evaluation.config import DATASET_REGISTRY, _DATA_ROOT

    if ds_cfg.data_path:
        return Path(ds_cfg.data_path)
    rel = DATASET_REGISTRY.get(ds_cfg.name)
    if rel is None:
        raise KeyError(
            f"Dataset '{ds_cfg.name}' not in DATASET_REGISTRY "
            f"(known: {sorted(DATASET_REGISTRY)}).  "
            "Set DatasetConfig.data_path to use a custom path."
        )
    return _DATA_ROOT / rel


def load_problems(ds_cfg: Any) -> list[dict[str, Any]]:
    """Load and normalise problems from a local Arrow dataset.

    Returns a uniform list of dicts::

        {
            "id":           str,   # stable problem identifier
            "prompt":       str,   # question / input text
            "ground_truth": str,   # expected answer
            "_raw":         dict,  # original row (all columns preserved)
        }

    Rows missing either the ``prompt_key`` or ``answer_key`` column are
    skipped with a DEBUG-level log entry.
    """
    from datasets import load_from_disk  # type: ignore[import-untyped]

    path = _resolve_data_path(ds_cfg)
    logger.info("Loading dataset '%s'  path=%s", ds_cfg.name, path)
    ds = load_from_disk(str(path))

    lo, hi = ds_cfg.integer_range
    problems: list[dict[str, Any]] = []
    skipped_missing = 0
    skipped_non_int = 0

    for i, item in enumerate(ds):
        if ds_cfg.n_samples is not None and len(problems) >= ds_cfg.n_samples:
            break

        prompt = item.get(ds_cfg.prompt_key)
        answer = item.get(ds_cfg.answer_key)
        if prompt is None or answer is None:
            logger.debug(
                "Row %d missing key '%s'/'%s', skipping",
                i,
                ds_cfg.prompt_key,
                ds_cfg.answer_key,
            )
            skipped_missing += 1
            continue

        if ds_cfg.filter_integer_answers:
            gt_int = _integer_in_range(answer, lo, hi)
            if gt_int is None:
                skipped_non_int += 1
                continue
            answer = str(gt_int)

        prob_id = (
            str(item[ds_cfg.id_key])
            if ds_cfg.id_key and ds_cfg.id_key in item
            else f"{ds_cfg.name}_{i}"
        )
        problems.append(
            {
                "id": prob_id,
                "prompt": str(prompt),
                "ground_truth": str(answer),
                "_raw": dict(item),
            }
        )

    if skipped_missing:
        logger.debug(
            "Skipped %d rows with missing keys in '%s'", skipped_missing, ds_cfg.name
        )
    if skipped_non_int:
        logger.info(
            "Skipped %d rows with non-integer / out-of-range answers in '%s' "
            "(filter_integer_answers=True, range=[%d, %d])",
            skipped_non_int,
            ds_cfg.name,
            lo,
            hi,
        )
    logger.info("Loaded %d problems from '%s'", len(problems), ds_cfg.name)
    return problems


# ── 5. Results & Output ───────────────────────────────────────────────────────


def _model_slug(model_id: str) -> str:
    """Derive a filesystem-safe slug from a model ID or HF cache path."""
    p = Path(model_id)
    for part in reversed(p.parts):
        if part.startswith("models--"):
            return re.sub(r"[^\w\-.]", "_", part[len("models--"):])[:64]
    slug = p.name or model_id.split("/")[-1] or model_id
    return re.sub(r"[^\w\-.]", "_", slug)[:64]


def _dataset_slug(name: str) -> str:
    return re.sub(r"[^\w\-.]", "_", name)[:64]


def _error_result(prob: dict[str, Any], error_reason: str = "") -> dict[str, Any]:
    return {
        "problem_id": prob["id"],
        "prompt": prob["prompt"],
        "ground_truth": prob["ground_truth"],
        "raw_output": None,
        "model_answer": None,
        "correct": False,
        "n_tokens": 0,
        "elapsed_s": 0.0,
        "tok_per_sec": 0.0,
        "error": True,
        "error_reason": error_reason,
    }


def compute_summary(
    results: list[dict[str, Any]],
    dataset_name: str,
    model_id: str,
    run_id: str,
    elapsed_s: float,
    n_samples: int = 1,
) -> dict[str, Any]:
    """Compute aggregate metrics from per-problem results.

    Returns:
        Dict with accuracy, throughput, token statistics, timing, and a
        breakdown of errors by reason when failures occurred.
    """
    total = len(results)
    n_correct = sum(1 for r in results if r.get("correct"))
    errors = [r for r in results if r.get("error")]
    n_error = len(errors)
    total_tokens = sum(r.get("n_tokens", 0) for r in results)
    valid_tps = [r["tok_per_sec"] for r in results if r.get("tok_per_sec", 0) > 0]
    avg_tps = sum(valid_tps) / len(valid_tps) if valid_tps else 0.0

    summary: dict[str, Any] = {
        "run_id": run_id,
        "dataset": dataset_name,
        "model": model_id,
        "n_samples": n_samples,
        "total": total,
        "correct": n_correct,
        "errors": n_error,
        "accuracy": n_correct / total if total > 0 else 0.0,
        "total_tokens": total_tokens,
        "avg_tok_per_sec": avg_tps,
        "wall_time_s": elapsed_s,
        "avg_time_per_problem_s": elapsed_s / total if total > 0 else 0.0,
    }

    if errors:
        unique_reasons = list(
            {r.get("error_reason", "unknown")[:200] for r in errors}
        )
        summary["error_problem_ids"] = [r["problem_id"] for r in errors]
        summary["error_reasons_sample"] = unique_reasons[:5]

    return summary


def save_results(
    results: list[dict[str, Any]],
    summary: dict[str, Any],
    run_cfg: Any,
    dataset_name: str,
) -> Path:
    """Write results, summary, and config JSON to the output directory.

    Directory layout::

        output_dir / run_id / dataset_name / model_name /
            ├── results.json
            ├── summary.json
            └── config.json

    Returns:
        Path to the leaf output directory.
    """
    out_dir = (
        run_cfg.output_dir
        / run_cfg.run_id
        / _dataset_slug(dataset_name)
        / _model_slug(run_cfg.model.model_id)
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "config.json").write_text(
        json.dumps(asdict(run_cfg), indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info(
        "Saved '%s': %d/%d correct (%.1f%%)  errors=%d  →  %s",
        dataset_name,
        summary["correct"],
        summary["total"],
        summary["accuracy"] * 100,
        summary["errors"],
        out_dir,
    )
    return out_dir
