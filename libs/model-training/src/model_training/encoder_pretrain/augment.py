"""Pair augmentation: associate task descriptions with mined coding pairs.

Loads normalized mined pairs (data/pairs/*.jsonl), extracts pre_code /
post_code from the activation_text / teacher_text fields, and associates a
task_description via strict field access only:

    strict: ``task_description`` field required; pairs without it are dropped.

Each output row records ``task_desc_source`` (always ``"explicit_field"``) so
downstream consumers can audit which source was used.

All GPU imports are omitted (INFRA-05); this module is CPU-safe.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from model_training.d2l_data import load_jsonl, save_jsonl

logger = logging.getLogger(__name__)

MIN_RETENTION_RATIO = 0.80


def _extract_pre_code(activation_text: str) -> str:
    """Extract the ``## Current Code`` section from activation_text.

    Returns the body after the ``## Current Code`` header if present;
    otherwise returns the full activation_text (which may be the initial
    task-only activation with no prior code).

    Args:
        activation_text: Formatted activation from normalize_mined_pairs.

    Returns:
        Pre-code string (may be empty).
    """
    marker = "## Current Code"
    if marker in activation_text:
        return activation_text.split(marker, 1)[1].lstrip("\n")
    return ""


def _extract_post_code(activation_text: str, teacher_text: str) -> str:
    r"""Extract the revision/implementation section from teacher_text.

    teacher_text == activation_text + trailing ``## Revision\n...`` or
    ``## Implementation\n...``. Return the trailing section.

    Args:
        activation_text: Activation portion of the pair.
        teacher_text: Full teacher (activation + revision).

    Returns:
        Post-code string (trailing section after activation).
    """
    if teacher_text.startswith(activation_text):
        return teacher_text[len(activation_text) :].lstrip("\n")
    return teacher_text  # corrupt record; return full teacher as fallback


def _select_task_desc(pair: dict[str, Any]) -> str | None:
    """Return the pair's explicit task_description, or None to signal DROP.

    Task descriptions must come from the upstream mining pipeline's
    association with a GitHub issue or PR. No fallbacks.

    Args:
        pair: Mined pair dict as produced by normalize_mined_pairs.

    Returns:
        Stripped, non-empty task_description string, or None if the pair
        should be dropped.
    """
    value = pair.get("task_description")
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped if stripped else None


def _build_row(pair: dict[str, Any], desc: str) -> dict[str, Any]:
    """Build an augmented row dict from a mined pair and its task description.

    Args:
        pair: Mined pair dict as produced by normalize_mined_pairs.
        desc: The resolved task description string.

    Returns:
        Augmented row dict with fields: task_id, pre_code, post_code,
        task_desc, task_desc_source, encoder_input, metadata.
    """
    activation = pair.get("activation_text", "")
    teacher = pair.get("teacher_text", "")
    pre_code: str = pair.get("pre_code") or _extract_pre_code(activation)
    post_code: str = pair.get("post_code") or _extract_post_code(activation, teacher)

    encoder_input = f"{desc}\n\n{pre_code}".strip()

    meta: dict[str, Any] = dict(pair.get("metadata") or {})
    meta.setdefault("source_repo", "")

    return {
        "task_id": pair.get("task_id", ""),
        "pre_code": pre_code,
        "post_code": post_code,
        "task_desc": desc,
        "task_desc_source": "explicit_field",
        "encoder_input": encoder_input,
        "metadata": meta,
    }


def augment_pairs_with_task_desc(
    pairs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Augment a list of mined pairs with task descriptions.

    For each pair, reads the explicit ``task_description`` field. Pairs
    where ``task_description`` is missing, None, or empty-after-strip are
    dropped with a WARNING log. Kept pairs are returned as augmented row
    dicts with ``task_desc_source`` set to ``"explicit_field"``.

    Args:
        pairs: List of mined pair dicts from normalize_mined_pairs.

    Returns:
        List of augmented row dicts with fields: task_id, pre_code,
        post_code, task_desc, task_desc_source, encoder_input, metadata.
    """
    if not pairs:
        return []

    dropped = 0
    kept: list[dict[str, Any]] = []
    for pair in pairs:
        desc = _select_task_desc(pair)
        if desc is None:
            dropped += 1
            continue
        kept.append(_build_row(pair, desc))

    if dropped:
        logger.warning(
            "augment: dropped %d/%d pairs with missing task_description",
            dropped,
            len(pairs),
        )
    logger.info("augment: kept %d augmented pairs", len(kept))
    return kept


def augment_corpus(
    pairs_dir: Path,
    output_dir: Path,
    glob: str = "*.jsonl",
) -> None:
    """Augment all mined-pair JSONL files in pairs_dir and write to output_dir.

    Reads each ``<repo>.jsonl`` from ``pairs_dir``, calls
    ``augment_pairs_with_task_desc``, and writes the resulting rows to
    ``output_dir/<repo>.jsonl``. Existing output files are overwritten.

    Enforces a minimum retention ratio: if fewer than
    ``MIN_RETENTION_RATIO`` (0.80) of all input pairs survive augmentation,
    raises RuntimeError directing the operator to re-run the mining pipeline
    with issue/PR task_description population.

    Args:
        pairs_dir: Directory containing mined pair JSONL files (e.g. data/pairs/).
        output_dir: Destination directory for augmented JSONL (e.g.
            data/pairs_augmented/).
        glob: Glob pattern for JSONL files within pairs_dir.

    Raises:
        FileNotFoundError: If pairs_dir does not exist.
        RuntimeError: If retention ratio < MIN_RETENTION_RATIO.
    """
    if not pairs_dir.exists():
        raise FileNotFoundError(
            f"pairs_dir does not exist: {pairs_dir}. "
            "Run scripts/mine_github.py first to populate data/pairs/."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_files = sorted(pairs_dir.glob(glob))

    if not jsonl_files:
        logger.warning(
            "augment_corpus: no JSONL files found in %s matching %s",
            pairs_dir,
            glob,
        )
        return

    # First pass: count totals for retention gate
    total = 0
    kept_total = 0
    per_file: list[tuple[Path, list[dict[str, Any]]]] = []

    for src_path in jsonl_files:
        logger.info("Augmenting %s ...", src_path.name)
        pairs = load_jsonl(src_path)
        rows = augment_pairs_with_task_desc(pairs)
        total += len(pairs)
        kept_total += len(rows)
        per_file.append((src_path, rows))

    if total > 0 and (kept_total / total) < MIN_RETENTION_RATIO:
        raise RuntimeError(
            f"augment_corpus: only {kept_total}/{total} "
            f"({100 * kept_total / total:.1f}%) mined pairs have a "
            f"task_description. Re-run the mining pipeline with "
            f"issue/PR task_description population before continuing."
        )

    for src_path, rows in per_file:
        dest = output_dir / src_path.name
        save_jsonl(rows, dest)
        logger.info("  -> wrote %d rows to %s", len(rows), dest)

    logger.info(
        "augment_corpus complete: %d total rows across %d files",
        kept_total,
        len(jsonl_files),
    )
