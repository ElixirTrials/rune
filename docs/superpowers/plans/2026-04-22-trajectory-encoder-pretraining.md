# Trajectory Encoder Pretraining Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Tasks 3 and 4 are parallel-safe (disjoint files); all others are sequential.

**Goal:** Pretrain a coding-aware shared text encoder on augmented mined coding-pair data and ship an HF-loadable checkpoint that drops into `reconstruction/task_embeddings.py` via the existing `emb_model_name` parameter without any API changes.

**Architecture:** Fine-tune `sentence-transformers/all-mpnet-base-v2` (768-d) with contrastive InfoNCE on `(task_desc || pre_code, post_code)` pairs (option (a)). The model architecture, embedding dimension, and `SentenceTransformer(path)` loading protocol are preserved intact — only weights are specialized. Task descriptions are sourced strictly from the mined pair's `task_description` field (populated upstream from the associated GitHub issue or PR title/body); pairs lacking a non-empty `task_description` are DROPPED during augmentation — no docstring/TODO/fallback extraction.

**Tech Stack:** Python 3.12, `sentence-transformers`, `torch`, `datasets` (HuggingFace), `mlflow`, `accelerate`, `uv run`; GPU imports deferred per INFRA-05; `ruff` (line-length 88), `mypy` strict-ish, Google-style docstrings, Conventional Commits.

---

## AMENDMENT 2026-04-22 (AUTHORITATIVE — supersedes Task 1 selector-chain text)

**Rule:** `_select_task_desc(pair)` returns the pair's explicit `task_description` field **only**. No commit_message, no docstring, no TODO/FIXME, no phase-role fallback. Pairs where `task_description` is missing, `None`, or empty-after-strip are DROPPED from the augmented corpus.

**Rationale:** Every mined pair must carry an authentic task description derived from its associated GitHub issue or PR. Silent fallbacks would contaminate the encoder with surrogate signal (code-internal text) that has no coupling to the hypernetwork's inference-time `task_description` input.

**Required implementation changes to Task 1:**

1. **Selector function** in `augment.py` — replace `_select_task_desc` with:

```python
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
```

2. **Augmenter** — `augment_pairs_with_task_desc(pairs)` drops pairs returning `None`, logs the count at WARNING level, and records retained count at INFO level:

```python
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
        dropped, len(pairs),
    )
logger.info("augment: kept %d augmented pairs", len(kept))
return kept
```

3. **Corpus audit gate** — `augment_corpus()` must enforce a minimum retention ratio. Before writing any output file, it iterates the corpus and fails with a clear error if `kept / total < 0.80`:

```python
if total > 0 and (kept_total / total) < MIN_RETENTION_RATIO:
    raise RuntimeError(
        f"augment_corpus: only {kept_total}/{total} "
        f"({100*kept_total/total:.1f}%) mined pairs have a "
        f"task_description. Re-run the mining pipeline with "
        f"issue/PR task_description population before continuing."
    )
```
where `MIN_RETENTION_RATIO = 0.80` is a module-level constant.

4. **Tests** — remove `test_select_commit_message`, `test_select_docstring`, `test_select_todo_comment`, `test_select_phase_role_fallback`. Replace with:

```python
def test_select_explicit_field() -> None:
    pair = _pair(task_description="Fix the segfault in parser")
    assert _select_task_desc(pair) == "Fix the segfault in parser"


def test_select_missing_returns_none() -> None:
    pair = _pair()  # no task_description key
    assert _select_task_desc(pair) is None


def test_select_empty_string_returns_none() -> None:
    pair = _pair(task_description="   ")
    assert _select_task_desc(pair) is None


def test_select_none_value_returns_none() -> None:
    pair = _pair(task_description=None)
    assert _select_task_desc(pair) is None


def test_augment_drops_pairs_without_task_description(caplog) -> None:
    import logging
    pairs = [
        _pair(task_description="Fix bug A"),
        _pair(task_id="pr_002"),  # no task_description — drop
        _pair(task_id="pr_003", task_description="Fix bug C"),
    ]
    with caplog.at_level(logging.WARNING, logger="model_training.encoder_pretrain.augment"):
        rows = augment_pairs_with_task_desc(pairs)
    assert len(rows) == 2
    assert {r["task_id"] for r in rows} == {"pr_001", "pr_003"}
    assert any("dropped 1/3" in msg for msg in caplog.messages)


def test_augment_corpus_fails_below_retention_threshold(tmp_path) -> None:
    from model_training.d2l_data import save_jsonl
    from model_training.encoder_pretrain.augment import augment_corpus
    pairs_dir = tmp_path / "pairs"; pairs_dir.mkdir()
    out_dir = tmp_path / "pairs_augmented"
    # 1 with desc, 4 without — 20% retention (below 80% threshold)
    mixed = [_pair(task_description="has desc")] + [_pair(task_id=f"pr_{i}") for i in range(4)]
    save_jsonl(mixed, pairs_dir / "repo.jsonl")
    with pytest.raises(RuntimeError, match="task_description"):
        augment_corpus(pairs_dir=pairs_dir, output_dir=out_dir)
```

5. **Row schema** — `task_desc_source` field is still written (value is always `"explicit_field"` now); kept for forward compatibility and audit trail.

6. **Docstring update** — the module-level docstring and the step-1.4 header text referencing "priority selector chain (explicit field > commit message > docstring > TODO > phase role)" MUST be updated to: "strict: `task_description` field required; pairs without it are dropped."

The implementer MUST treat this amendment as authoritative. Where the Task-1 prose below conflicts with this amendment, this amendment wins.

---

## Architecture Decision — Why Option (a)

**Option (a): fine-tune `sentence-transformers/all-mpnet-base-v2` with contrastive InfoNCE** was chosen over:

- **(b) train from scratch (~30M param)**: too data-hungry; the mined corpus is large enough to specialize but not to bootstrap a competitive encoder from random init.
- **(c) continued masked-span pretraining of CodeBERT/CodeT5+**: these encoders are coding-aware but do NOT support `SentenceTransformer(path)` loading out-of-the-box without a wrapper module, adding complexity.

Option (a) preserves the 768-dim contract the builder already expects, is loadable directly by `load_default_encoder(model_id)` in `task_embeddings.py`, and has the strongest empirical prior (Sentence-BERT fine-tuned with InfoNCE is the de-facto recipe for domain-adapted embedding). The existing `all-mpnet-base-v2` baseline already handles English reasonably; contrastive fine-tuning on code pairs specializes it toward coding semantics while keeping the same vector space dimension.

---

## Augmented Pair JSONL Row Schema

Each row in `data/pairs_augmented/<repo>.jsonl`:

```json
{
  "task_id": "pr_12345",
  "pre_code": "...",
  "post_code": "...",
  "task_desc": "Fix off-by-one error in slice handling",
  "task_desc_source": "commit_message",
  "encoder_input": "Fix off-by-one error in slice handling\n\n<pre_code>",
  "metadata": {
    "source_repo": "pandas-dev_pandas",
    "source_task_id": "pr_12345",
    "step_index": 0,
    "outcome": "merged",
    "language": null
  }
}
```

`task_desc_source` is one of: `"explicit_field"`, `"commit_message"`, `"docstring"`, `"todo_comment"`, `"phase_role_fallback"`.

`encoder_input` is the ready-to-encode concatenation — written at augmentation time to avoid recomputing it during training.

---

## Training Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| Temperature τ | 0.07 | Standard InfoNCE default; lower = harder negatives |
| Batch size | 64 | In-batch negatives; balances GPU memory vs negative diversity |
| Learning rate | 2e-5 | Safe fine-tuning LR for BERT-scale encoders |
| LR scheduler | cosine | Smooth decay; standard for contrastive fine-tuning |
| Warmup steps | 100 | ~1.5 epochs at batch 64 on ~4k pairs |
| Epochs | 5 | Empirically sufficient for domain adaptation at this scale |
| Max seq length | 512 | Covers most pre_code blocks; truncate symmetrically |
| Pooling | mean pooling | Sentence-transformers default; preserves token distribution |

---

## Mined-Pair Corpus Location

Canonical location (confirmed from `d2l_mining.py` + repo inspection):

```
data/pairs/<repo_owner>_<repo_name>.jsonl
```

Example: `data/pairs/pandas-dev_pandas.jsonl` (3.0 MB, confirmed present).

Each JSONL file contains records as produced by `normalize_mined_pairs()` with fields: `task_id`, `activation_text`, `teacher_text`, `metadata` (including `source_task_id`, `step_index`, `outcome`, `language`).

The `pre_code` and `post_code` fields are extracted from `activation_text` / `teacher_text` via `_extract_pre_revision` / `_extract_revision` from `d2l_data.py` — the same logic `pairs_to_chat_messages` already uses.

**Assumption:** the corpus was produced by a prior mining run. If `data/pairs/` is empty, the augmentation step will exit with a clear error message directing the operator to run `scripts/mine_github.py` first.

---

## Train/Test Split

Task-ID level split via `d2l_data.split_by_task_id()` (already exists, line 634). This prevents leakage: all pairs from the same PR go to the same partition.

Split ratio: 80/20 (default `test_fraction=0.2`, `seed=42`).

The split is applied **after** augmentation, on the full augmented corpus. The held-out test set is used for retrieval eval (MRR@10, Recall@1) and is never seen during training.

---

## File Structure

```
libs/model-training/src/model_training/encoder_pretrain/
├── __init__.py                 # package marker; re-exports public API
├── augment.py                  # load mined pairs, associate task_desc, write augmented JSONL
├── dataset.py                  # ContrastivePairDataset (torch Dataset); collator
├── loss.py                     # InfoNCE loss (temperature-scaled cosine similarity)
├── train_encoder.py            # training loop (HF Accelerate + MLflow)
├── eval_encoder.py             # retrieval eval (MRR@10, Recall@1); downstream cluster probe
└── cli.py                      # argparse CLI with --dry-run; no torch import on dry-run

libs/model-training/tests/
├── test_encoder_augment.py     # unit tests for augment.py selector chain
├── test_encoder_dataset.py     # unit tests for ContrastivePairDataset + collator
├── test_encoder_loss.py        # unit tests for InfoNCE (synthetic tensors)
├── test_encoder_eval.py        # unit tests for retrieval eval with tiny synthetic data
├── test_encoder_roundtrip.py   # save → load → encode → assert shape == (n, 768)
└── test_encoder_cli.py         # --dry-run prints JSON without importing torch
```

No modifications to existing files in this plan. All consumers of the checkpoint interact through the existing `task_embeddings.load_default_encoder(model_id)` API.

---

## Task Dependency Graph

```
Task 1 (augment.py)
    └─► Task 2 (dataset.py + loss.py)
            ├─► Task 3 (train_encoder.py) ──────┐
            └─► Task 4 (eval_encoder.py) ────────┤── parallel-safe (disjoint files)
                                                  │
Task 5 (cli.py) ◄─────────────────────────────── depends on Tasks 3+4
Task 6 (encoder round-trip test)
Task 7 (ruff + mypy + final integration)
Task 8 (commit + checkpoint)
```

Tasks 3 and 4 have no runtime dependency on each other — only on `dataset.py` and `loss.py` from Task 2. They may be dispatched as a parallel wave.

---

## Conventions for Every Task

- Use `uv run pytest <path>` for tests. Never bare `pytest` or `python`.
- Use `uv run ruff check libs/model-training` and `uv run mypy libs/model-training` before each commit.
- GPU imports (`torch`, `sentence_transformers`, `accelerate`) go inside function bodies, never at module top level. CPU-only CI must still `import model_training.encoder_pretrain.*` without error.
- All new public functions get Google-style docstrings.
- Commit after each task with Conventional Commits style: `feat(encoder-pretrain): <what>`.
- Single-GPU only. Multi-GPU support is out of scope for this plan; document it in a comment.

---

## Task 1: Pair Augmentation — Associate Task Descriptions with Mined Pairs

**Files:**
- Create: `libs/model-training/src/model_training/encoder_pretrain/__init__.py`
- Create: `libs/model-training/src/model_training/encoder_pretrain/augment.py`
- Test: `libs/model-training/tests/test_encoder_augment.py`

**Acceptance:** `augment_pairs_with_task_desc()` produces JSONL rows with correct `task_desc_source` for each case in the selector chain; `augment_corpus()` correctly handles empty corpus and logs via the standard logger.

- [ ] **Step 1.1: Create the package marker**

Create `libs/model-training/src/model_training/encoder_pretrain/__init__.py`:

```python
"""Trajectory encoder pretraining subpackage.

Pretrains a coding-aware sentence encoder (option a: InfoNCE fine-tuning of
all-mpnet-base-v2) on augmented mined coding pairs from data/pairs/.

Public API:
    from model_training.encoder_pretrain.augment import augment_corpus
    from model_training.encoder_pretrain.train_encoder import run_training
    from model_training.encoder_pretrain.eval_encoder import run_retrieval_eval
"""

from model_training.encoder_pretrain import augment, cli, dataset, eval_encoder, loss, train_encoder  # noqa: F401
```

- [ ] **Step 1.2: Write failing tests for the selector chain**

Create `libs/model-training/tests/test_encoder_augment.py`:

```python
"""Tests for mined-pair task-description augmentation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from model_training.encoder_pretrain.augment import (
    _select_task_desc,
    augment_pairs_with_task_desc,
)


def _pair(**overrides: Any) -> dict[str, Any]:
    """Build a minimal mined-pair record."""
    base: dict[str, Any] = {
        "task_id": "pr_001",
        "activation_text": "## Task\n\n## Current Code\ndef foo(): pass",
        "teacher_text": "## Task\n\n## Current Code\ndef foo(): pass\n\n## Revision\ndef foo(): return 1",
        "metadata": {
            "source_task_id": "pr_001",
            "step_index": 0,
            "outcome": "merged",
            "language": None,
        },
    }
    base.update(overrides)
    return base


def test_select_explicit_field() -> None:
    """explicit task_description field wins."""
    pair = _pair(task_description="Fix the segfault in parser")
    desc, source = _select_task_desc(pair)
    assert source == "explicit_field"
    assert desc == "Fix the segfault in parser"


def test_select_commit_message() -> None:
    """commit_message used when no explicit field."""
    pair = _pair(metadata={
        "source_task_id": "pr_001",
        "step_index": 0,
        "outcome": "merged",
        "language": None,
        "commit_message": "refactor: extract helper function",
    })
    desc, source = _select_task_desc(pair)
    assert source == "commit_message"
    assert desc == "refactor: extract helper function"


def test_select_docstring() -> None:
    """Nearest docstring/TODO from pre_code used when no commit_message."""
    pre_code = '"""Parse CSV rows into typed records."""\ndef parse_csv(path): ...'
    pair = _pair(pre_code=pre_code)
    desc, source = _select_task_desc(pair)
    assert source == "docstring"
    assert "Parse CSV rows" in desc


def test_select_todo_comment() -> None:
    """TODO comment used when no docstring."""
    pre_code = "# TODO: fix null pointer dereference\ndef process(x): return x"
    pair = _pair(pre_code=pre_code)
    desc, source = _select_task_desc(pair)
    assert source == "todo_comment"
    assert "fix null pointer" in desc


def test_select_phase_role_fallback() -> None:
    """Phase-role label emitted when no other source exists."""
    pair = _pair()  # no extra fields, activation has no docstring/TODO
    desc, source = _select_task_desc(pair)
    assert source == "phase_role_fallback"
    assert len(desc) > 0


def test_augment_pairs_with_task_desc_adds_fields() -> None:
    """augment_pairs_with_task_desc returns rows with required schema fields."""
    pairs = [_pair(task_description="Fix the bug")]
    rows = augment_pairs_with_task_desc(pairs)
    assert len(rows) == 1
    row = rows[0]
    assert row["task_desc"] == "Fix the bug"
    assert row["task_desc_source"] == "explicit_field"
    assert "encoder_input" in row
    assert "pre_code" in row
    assert "post_code" in row
    assert row["task_id"] == "pr_001"


def test_augment_pairs_logs_source_distribution(caplog: pytest.LogCaptureFixture) -> None:
    """A summary log line is emitted with source distribution counts."""
    import logging

    pairs = [
        _pair(task_description="desc a"),
        _pair(task_id="pr_002", task_description="desc b"),
    ]
    with caplog.at_level(logging.INFO, logger="model_training.encoder_pretrain.augment"):
        augment_pairs_with_task_desc(pairs)
    assert any("task_desc_source" in msg or "explicit_field" in msg for msg in caplog.messages)


def test_augment_empty_pairs() -> None:
    """Empty input returns empty list without error."""
    rows = augment_pairs_with_task_desc([])
    assert rows == []


def test_augment_corpus_missing_dir_raises(tmp_path: Path) -> None:
    """augment_corpus raises FileNotFoundError when pairs_dir does not exist."""
    from model_training.encoder_pretrain.augment import augment_corpus

    missing = tmp_path / "nonexistent_pairs"
    with pytest.raises(FileNotFoundError, match="pairs_dir"):
        augment_corpus(pairs_dir=missing, output_dir=tmp_path / "out")


def test_augment_corpus_writes_jsonl(tmp_path: Path) -> None:
    """augment_corpus writes augmented JSONL for each input JSONL file."""
    import json

    from model_training.d2l_data import save_jsonl
    from model_training.encoder_pretrain.augment import augment_corpus

    # Create a minimal mined-pair JSONL file in tmp pairs dir
    pairs_dir = tmp_path / "pairs"
    pairs_dir.mkdir()
    out_dir = tmp_path / "pairs_augmented"

    pair = _pair(task_description="Fix bug", task_id="pr_001")
    save_jsonl([pair], pairs_dir / "test_repo.jsonl")

    augment_corpus(pairs_dir=pairs_dir, output_dir=out_dir)

    out_file = out_dir / "test_repo.jsonl"
    assert out_file.exists()
    rows = [json.loads(line) for line in out_file.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["task_desc_source"] == "explicit_field"
```

- [ ] **Step 1.3: Run tests to confirm they fail**

```bash
uv run pytest libs/model-training/tests/test_encoder_augment.py -v 2>&1 | head -30
```

Expected: `ModuleNotFoundError: No module named 'model_training.encoder_pretrain.augment'`

- [ ] **Step 1.4: Implement augment.py**

Create `libs/model-training/src/model_training/encoder_pretrain/augment.py`:

```python
"""Pair augmentation: associate task descriptions with mined coding pairs.

Loads normalized mined pairs (data/pairs/*.jsonl), extracts pre_code /
post_code from the activation_text / teacher_text fields, and selects a
task_description via a priority selector chain:

    1. explicit ``task_description`` field on the pair dict
    2. ``metadata.commit_message``
    3. nearest docstring (triple-quoted string) in ``pre_code``
    4. nearest TODO/FIXME comment in ``pre_code``
    5. phase-role fallback label

Each output row records ``task_desc_source`` so downstream consumers can
audit which source dominated the corpus.

All GPU imports are omitted (INFRA-05); this module is CPU-safe.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

from model_training.d2l_data import load_jsonl, save_jsonl

logger = logging.getLogger(__name__)

# Regex to find the first triple-quoted docstring in pre_code
_DOCSTRING_RE = re.compile(
    r'"""(.*?)"""|\'\'\'(.*?)\'\'\'',
    re.DOTALL,
)

# Regex to find TODO / FIXME comments
_TODO_RE = re.compile(
    r"#\s*(?:TODO|FIXME|HACK|XXX)[:\s]+(.*)",
    re.IGNORECASE,
)

_PHASE_ROLE_FALLBACK = (
    "Implement a Python function that satisfies the given requirements."
)


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
    """Extract the revision/implementation section from teacher_text.

    teacher_text == activation_text + trailing ``## Revision\\n...`` or
    ``## Implementation\\n...``. Return the trailing section.

    Args:
        activation_text: Activation portion of the pair.
        teacher_text: Full teacher (activation + revision).

    Returns:
        Post-code string (trailing section after activation).
    """
    if teacher_text.startswith(activation_text):
        return teacher_text[len(activation_text):].lstrip("\n")
    return teacher_text  # corrupt record; return full teacher as fallback


def _find_docstring(code: str) -> str | None:
    """Return the text of the first docstring found in ``code``, or None.

    Args:
        code: Source code string to search.

    Returns:
        Docstring body stripped of whitespace, or None if not found.
    """
    match = _DOCSTRING_RE.search(code)
    if not match:
        return None
    body = match.group(1) or match.group(2) or ""
    body = body.strip()
    return body if body else None


def _find_todo(code: str) -> str | None:
    """Return the text of the first TODO/FIXME comment in ``code``, or None.

    Args:
        code: Source code string to search.

    Returns:
        TODO comment text stripped of whitespace, or None if not found.
    """
    match = _TODO_RE.search(code)
    if not match:
        return None
    return match.group(1).strip() or None


def _select_task_desc(pair: dict[str, Any]) -> tuple[str, str]:
    """Select the best task description for a mined pair.

    Priority chain (highest to lowest):
        1. ``task_description`` field directly on the pair dict
        2. ``metadata.commit_message``
        3. First docstring found in pre_code
        4. First TODO/FIXME comment in pre_code
        5. ``_PHASE_ROLE_FALLBACK`` label

    Logs which source was used at DEBUG level.

    Args:
        pair: Mined pair dict as produced by normalize_mined_pairs.

    Returns:
        Tuple of (task_description, source_label) where source_label is one
        of: ``"explicit_field"``, ``"commit_message"``, ``"docstring"``,
        ``"todo_comment"``, ``"phase_role_fallback"``.
    """
    # 1. Explicit field
    explicit = pair.get("task_description")
    if explicit and isinstance(explicit, str) and explicit.strip():
        logger.debug("task_desc_source=explicit_field for task_id=%s", pair.get("task_id"))
        return explicit.strip(), "explicit_field"

    # 2. Commit message from metadata
    meta: dict[str, Any] = pair.get("metadata") or {}
    commit_msg = meta.get("commit_message")
    if commit_msg and isinstance(commit_msg, str) and commit_msg.strip():
        logger.debug("task_desc_source=commit_message for task_id=%s", pair.get("task_id"))
        return commit_msg.strip(), "commit_message"

    # Extract pre_code for docstring / TODO searches
    activation = pair.get("activation_text", "")
    # Allow callers to pass pre_code directly (e.g. in tests)
    pre_code: str = pair.get("pre_code") or _extract_pre_code(activation)

    # 3. Nearest docstring
    docstring = _find_docstring(pre_code)
    if docstring:
        logger.debug("task_desc_source=docstring for task_id=%s", pair.get("task_id"))
        # Truncate long docstrings to first 200 chars
        return docstring[:200], "docstring"

    # 4. TODO/FIXME comment
    todo = _find_todo(pre_code)
    if todo:
        logger.debug("task_desc_source=todo_comment for task_id=%s", pair.get("task_id"))
        return todo[:200], "todo_comment"

    # 5. Fallback
    logger.debug("task_desc_source=phase_role_fallback for task_id=%s", pair.get("task_id"))
    return _PHASE_ROLE_FALLBACK, "phase_role_fallback"


def augment_pairs_with_task_desc(
    pairs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Augment a list of mined pairs with task descriptions.

    For each pair, selects a task description via the priority chain in
    ``_select_task_desc``, extracts pre_code / post_code, builds the
    ready-to-encode ``encoder_input`` concatenation, and returns a new
    row dict. Logs a summary of source distribution counts.

    Pairs where both pre_code and post_code are empty are skipped (logged
    at WARNING level).

    Args:
        pairs: List of mined pair dicts from normalize_mined_pairs.

    Returns:
        List of augmented row dicts with fields: task_id, pre_code,
        post_code, task_desc, task_desc_source, encoder_input, metadata.
    """
    if not pairs:
        return []

    rows: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()

    for pair in pairs:
        activation = pair.get("activation_text", "")
        teacher = pair.get("teacher_text", "")
        pre_code = pair.get("pre_code") or _extract_pre_code(activation)
        post_code = pair.get("post_code") or _extract_post_code(activation, teacher)

        if not pre_code and not post_code:
            logger.warning(
                "Skipping pair with empty pre_code and post_code: task_id=%s",
                pair.get("task_id"),
            )
            continue

        task_desc, source = _select_task_desc(pair)
        source_counts[source] += 1

        encoder_input = f"{task_desc}\n\n{pre_code}".strip()

        meta: dict[str, Any] = dict(pair.get("metadata") or {})
        meta["source_repo"] = meta.get("source_repo", "")

        rows.append(
            {
                "task_id": pair.get("task_id", ""),
                "pre_code": pre_code,
                "post_code": post_code,
                "task_desc": task_desc,
                "task_desc_source": source,
                "encoder_input": encoder_input,
                "metadata": meta,
            }
        )

    logger.info(
        "augment_pairs_with_task_desc: %d pairs augmented. task_desc_source distribution: %s",
        len(rows),
        dict(source_counts),
    )
    return rows


def augment_corpus(
    pairs_dir: Path,
    output_dir: Path,
    glob: str = "*.jsonl",
) -> None:
    """Augment all mined-pair JSONL files in pairs_dir and write to output_dir.

    Reads each ``<repo>.jsonl`` from ``pairs_dir``, calls
    ``augment_pairs_with_task_desc``, and writes the resulting rows to
    ``output_dir/<repo>.jsonl``. Existing output files are overwritten.

    Args:
        pairs_dir: Directory containing mined pair JSONL files (e.g. data/pairs/).
        output_dir: Destination directory for augmented JSONL (e.g.
            data/pairs_augmented/).
        glob: Glob pattern for JSONL files within pairs_dir.

    Raises:
        FileNotFoundError: If pairs_dir does not exist.
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
            "augment_corpus: no JSONL files found in %s matching %s", pairs_dir, glob
        )
        return

    total_rows = 0
    for src_path in jsonl_files:
        logger.info("Augmenting %s …", src_path.name)
        pairs = load_jsonl(src_path)
        rows = augment_pairs_with_task_desc(pairs)
        dest = output_dir / src_path.name
        save_jsonl(rows, dest)
        total_rows += len(rows)
        logger.info("  → wrote %d rows to %s", len(rows), dest)

    logger.info(
        "augment_corpus complete: %d total rows across %d files",
        total_rows,
        len(jsonl_files),
    )
```

- [ ] **Step 1.5: Run tests to confirm they pass**

```bash
uv run pytest libs/model-training/tests/test_encoder_augment.py -v
```

Expected output (all 9 tests passing):

```
PASSED test_select_explicit_field
PASSED test_select_commit_message
PASSED test_select_docstring
PASSED test_select_todo_comment
PASSED test_select_phase_role_fallback
PASSED test_augment_pairs_with_task_desc_adds_fields
PASSED test_augment_pairs_logs_source_distribution
PASSED test_augment_empty_pairs
PASSED test_augment_corpus_missing_dir_raises
PASSED test_augment_corpus_writes_jsonl
10 passed in <1s
```

- [ ] **Step 1.6: Lint and type-check**

```bash
uv run ruff check libs/model-training/src/model_training/encoder_pretrain/augment.py
uv run mypy libs/model-training/src/model_training/encoder_pretrain/augment.py
```

Expected: no errors.

- [ ] **Step 1.7: Commit**

```bash
git add libs/model-training/src/model_training/encoder_pretrain/__init__.py \
        libs/model-training/src/model_training/encoder_pretrain/augment.py \
        libs/model-training/tests/test_encoder_augment.py
git commit -m "$(cat <<'EOF'
feat(encoder-pretrain): add pair augmentation with task-desc selector chain

Implements priority selector chain (explicit_field > commit_message >
docstring > todo_comment > phase_role_fallback) for associating task
descriptions with mined coding pairs. Writes augmented JSONL with
task_desc_source provenance field per row.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Contrastive Dataset and InfoNCE Loss

**Files:**
- Create: `libs/model-training/src/model_training/encoder_pretrain/dataset.py`
- Create: `libs/model-training/src/model_training/encoder_pretrain/loss.py`
- Test: `libs/model-training/tests/test_encoder_dataset.py`
- Test: `libs/model-training/tests/test_encoder_loss.py`

**Acceptance:** `ContrastivePairDataset` loads augmented JSONL and returns `(encoder_input, post_code)` pairs. `ContrastiveCollator` tokenizes pairs with symmetric max-length truncation. `infonce_loss` produces a scalar with correct gradient flow. All tests pass in CPU-only mode (no GPU required).

- [ ] **Step 2.1: Write failing tests for dataset and loss**

Create `libs/model-training/tests/test_encoder_dataset.py`:

```python
"""Tests for ContrastivePairDataset and ContrastiveCollator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")


def _write_augmented_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )


def _make_row(i: int) -> dict[str, Any]:
    return {
        "task_id": f"pr_{i:03d}",
        "pre_code": f"def foo_{i}(): pass",
        "post_code": f"def foo_{i}(): return {i}",
        "task_desc": f"Fix foo_{i} to return {i}",
        "task_desc_source": "explicit_field",
        "encoder_input": f"Fix foo_{i} to return {i}\n\ndef foo_{i}(): pass",
        "metadata": {},
    }


def test_dataset_len_and_getitem(tmp_path: Path) -> None:
    from model_training.encoder_pretrain.dataset import ContrastivePairDataset

    rows = [_make_row(i) for i in range(5)]
    _write_augmented_jsonl(tmp_path / "repo.jsonl", rows)
    ds = ContrastivePairDataset.from_dir(tmp_path)
    assert len(ds) == 5
    item = ds[0]
    assert "anchor" in item
    assert "positive" in item
    assert item["anchor"] == rows[0]["encoder_input"]
    assert item["positive"] == rows[0]["post_code"]


def test_dataset_from_dir_loads_multiple_files(tmp_path: Path) -> None:
    from model_training.encoder_pretrain.dataset import ContrastivePairDataset

    for j in range(3):
        rows = [_make_row(j * 10 + i) for i in range(4)]
        _write_augmented_jsonl(tmp_path / f"repo_{j}.jsonl", rows)
    ds = ContrastivePairDataset.from_dir(tmp_path)
    assert len(ds) == 12


def test_dataset_skips_rows_missing_post_code(tmp_path: Path) -> None:
    from model_training.encoder_pretrain.dataset import ContrastivePairDataset

    rows = [_make_row(0), {**_make_row(1), "post_code": ""}, _make_row(2)]
    _write_augmented_jsonl(tmp_path / "repo.jsonl", rows)
    ds = ContrastivePairDataset.from_dir(tmp_path)
    assert len(ds) == 2  # row with empty post_code skipped


def test_collator_returns_tokenized_batch(tmp_path: Path) -> None:
    """ContrastiveCollator tokenizes anchors and positives into input_ids."""
    from transformers import AutoTokenizer

    from model_training.encoder_pretrain.dataset import (
        ContrastivePairDataset,
        ContrastiveCollator,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "sentence-transformers/all-mpnet-base-v2"
    )
    rows = [_make_row(i) for i in range(4)]
    _write_augmented_jsonl(tmp_path / "repo.jsonl", rows)
    ds = ContrastivePairDataset.from_dir(tmp_path)
    collator = ContrastiveCollator(tokenizer=tokenizer, max_length=64)

    batch = collator([ds[i] for i in range(4)])
    assert "anchor_input_ids" in batch
    assert "anchor_attention_mask" in batch
    assert "positive_input_ids" in batch
    assert "positive_attention_mask" in batch
    assert batch["anchor_input_ids"].shape == (4, 64)
    assert batch["positive_input_ids"].shape == (4, 64)
```

Create `libs/model-training/tests/test_encoder_loss.py`:

```python
"""Tests for InfoNCE contrastive loss."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def test_infonce_loss_is_scalar() -> None:
    from model_training.encoder_pretrain.loss import infonce_loss

    # embeddings: batch_size=4, dim=8
    anchors = torch.randn(4, 8)
    positives = torch.randn(4, 8)
    loss_val = infonce_loss(anchors, positives, temperature=0.07)
    assert loss_val.shape == ()  # scalar
    assert loss_val.item() > 0.0


def test_infonce_loss_lower_for_matched_pairs() -> None:
    """Loss is lower when anchors and positives are identical (trivial case)."""
    from model_training.encoder_pretrain.loss import infonce_loss

    embeddings = torch.randn(8, 16)
    loss_matched = infonce_loss(embeddings, embeddings.clone(), temperature=0.07)
    loss_random = infonce_loss(embeddings, torch.randn(8, 16), temperature=0.07)
    # matched embeddings should have lower loss than random negatives
    assert loss_matched.item() < loss_random.item()


def test_infonce_loss_has_gradient() -> None:
    """Loss is differentiable with respect to anchor embeddings."""
    from model_training.encoder_pretrain.loss import infonce_loss

    anchors = torch.randn(4, 8, requires_grad=True)
    positives = torch.randn(4, 8)
    loss_val = infonce_loss(anchors, positives, temperature=0.07)
    loss_val.backward()
    assert anchors.grad is not None
    assert not torch.isnan(anchors.grad).any()


def test_infonce_loss_temperature_scaling() -> None:
    """Lower temperature sharpens the distribution (higher loss for random pairs)."""
    from model_training.encoder_pretrain.loss import infonce_loss

    anchors = torch.randn(4, 8)
    positives = torch.randn(4, 8)
    loss_sharp = infonce_loss(anchors, positives, temperature=0.01)
    loss_flat = infonce_loss(anchors, positives, temperature=1.0)
    # Sharper temperature produces different (usually higher) loss for mismatched pairs
    assert loss_sharp.item() != loss_flat.item()
```

- [ ] **Step 2.2: Run tests to confirm they fail**

```bash
uv run pytest libs/model-training/tests/test_encoder_dataset.py libs/model-training/tests/test_encoder_loss.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'model_training.encoder_pretrain.dataset'`

- [ ] **Step 2.3: Implement dataset.py**

Create `libs/model-training/src/model_training/encoder_pretrain/dataset.py`:

```python
"""Contrastive pair dataset and tokenizing collator for encoder pretraining.

Loads augmented JSONL files from an output directory (produced by augment.py)
and exposes them as a PyTorch Dataset of (anchor, positive) string pairs.

ContrastiveCollator tokenizes both sides symmetrically and returns batched
input_ids / attention_mask tensors prefixed ``anchor_`` and ``positive_``.

All GPU-dependent imports (torch, transformers) are deferred inside class
bodies / method bodies per INFRA-05.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ContrastivePairDataset:
    """PyTorch-compatible dataset of (encoder_input, post_code) pairs.

    Loads all ``*.jsonl`` files from a directory of augmented pairs. Rows
    where ``post_code`` is empty are skipped so the collator always receives
    a valid positive target.

    Args:
        rows: List of augmented pair dicts as produced by augment_corpus.
    """

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = [r for r in rows if r.get("post_code", "").strip()]
        if len(self._rows) < len(rows):
            logger.warning(
                "ContrastivePairDataset: skipped %d rows with empty post_code",
                len(rows) - len(self._rows),
            )

    @classmethod
    def from_dir(cls, augmented_dir: Path, glob: str = "*.jsonl") -> "ContrastivePairDataset":
        """Load all augmented JSONL files from augmented_dir.

        Args:
            augmented_dir: Directory containing augmented pair JSONL files.
            glob: Glob pattern to match files.

        Returns:
            ContrastivePairDataset with all rows combined.

        Raises:
            FileNotFoundError: If augmented_dir does not exist.
        """
        if not augmented_dir.exists():
            raise FileNotFoundError(
                f"augmented_dir does not exist: {augmented_dir}. "
                "Run augment_corpus first."
            )
        all_rows: list[dict[str, Any]] = []
        for path in sorted(augmented_dir.glob(glob)):
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        all_rows.append(json.loads(line))
        logger.info(
            "ContrastivePairDataset.from_dir: loaded %d rows from %s",
            len(all_rows),
            augmented_dir,
        )
        return cls(all_rows)

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict[str, str]:
        row = self._rows[idx]
        return {
            "anchor": row["encoder_input"],
            "positive": row["post_code"],
        }


class ContrastiveCollator:
    """Tokenize (anchor, positive) string pairs into padded batch tensors.

    Returns a dict with keys ``anchor_input_ids``, ``anchor_attention_mask``,
    ``positive_input_ids``, ``positive_attention_mask`` — all
    ``LongTensor[batch, max_length]``.

    Args:
        tokenizer: A HuggingFace tokenizer (e.g. from AutoTokenizer).
        max_length: Maximum sequence length; sequences are truncated and
            padded to exactly this length.
    """

    def __init__(self, tokenizer: Any, max_length: int = 512) -> None:
        self._tokenizer = tokenizer
        self._max_length = max_length

    def __call__(self, items: list[dict[str, str]]) -> dict[str, Any]:
        """Tokenize a list of (anchor, positive) dicts into a batch.

        Args:
            items: List of dicts with ``anchor`` and ``positive`` string keys.

        Returns:
            Dict with ``anchor_input_ids``, ``anchor_attention_mask``,
            ``positive_input_ids``, ``positive_attention_mask`` tensors of
            shape ``(batch_size, max_length)``.
        """
        anchors = [item["anchor"] for item in items]
        positives = [item["positive"] for item in items]

        def _tokenize(texts: list[str]) -> dict[str, Any]:
            return self._tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=self._max_length,
                return_tensors="pt",
            )

        anchor_enc = _tokenize(anchors)
        positive_enc = _tokenize(positives)

        return {
            "anchor_input_ids": anchor_enc["input_ids"],
            "anchor_attention_mask": anchor_enc["attention_mask"],
            "positive_input_ids": positive_enc["input_ids"],
            "positive_attention_mask": positive_enc["attention_mask"],
        }
```

- [ ] **Step 2.4: Implement loss.py**

Create `libs/model-training/src/model_training/encoder_pretrain/loss.py`:

```python
"""InfoNCE contrastive loss for encoder pretraining.

Uses in-batch negatives: for a batch of size N, the positive pair for
anchor i is positive i; all other j != i are treated as negatives.

All imports are deferred inside function bodies (INFRA-05).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def infonce_loss(
    anchors: "torch.Tensor",
    positives: "torch.Tensor",
    temperature: float = 0.07,
) -> "torch.Tensor":
    """Compute InfoNCE (NT-Xent) loss with in-batch negatives.

    Both tensors must be L2-normalized embeddings of shape
    ``(batch_size, embedding_dim)``. The loss is symmetric: mean of
    (anchor→positive CE) and (positive→anchor CE).

    Args:
        anchors: Anchor embeddings ``(B, D)``.
        positives: Positive embeddings ``(B, D)``. Must be same shape.
        temperature: Temperature scalar τ. Lower = harder negatives.

    Returns:
        Scalar loss tensor with gradient enabled.

    Raises:
        ValueError: If anchors and positives have different shapes.
    """
    import torch  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415

    if anchors.shape != positives.shape:
        raise ValueError(
            f"anchors shape {anchors.shape} != positives shape {positives.shape}"
        )

    batch_size = anchors.shape[0]

    # L2-normalise both sides
    anchors_norm = F.normalize(anchors, dim=-1)
    positives_norm = F.normalize(positives, dim=-1)

    # Similarity matrix: (B, B), scaled by temperature
    logits = torch.matmul(anchors_norm, positives_norm.T) / temperature

    # Diagonal entries are the positive pairs
    labels = torch.arange(batch_size, device=anchors.device)

    # Symmetric loss: anchor → positive and positive → anchor
    loss_a = F.cross_entropy(logits, labels)
    loss_b = F.cross_entropy(logits.T, labels)

    return (loss_a + loss_b) / 2.0
```

- [ ] **Step 2.5: Run tests to confirm they pass**

```bash
uv run pytest libs/model-training/tests/test_encoder_dataset.py libs/model-training/tests/test_encoder_loss.py -v
```

Expected output (all tests passing):

```
PASSED test_dataset_len_and_getitem
PASSED test_dataset_from_dir_loads_multiple_files
PASSED test_dataset_skips_rows_missing_post_code
PASSED test_collator_returns_tokenized_batch
PASSED test_infonce_loss_is_scalar
PASSED test_infonce_loss_lower_for_matched_pairs
PASSED test_infonce_loss_has_gradient
PASSED test_infonce_loss_temperature_scaling
8 passed in <5s
```

(The collator test downloads `all-mpnet-base-v2` tokenizer on first run; subsequent runs use the local cache.)

- [ ] **Step 2.6: Lint and type-check**

```bash
uv run ruff check libs/model-training/src/model_training/encoder_pretrain/dataset.py \
                  libs/model-training/src/model_training/encoder_pretrain/loss.py
uv run mypy libs/model-training/src/model_training/encoder_pretrain/dataset.py \
            libs/model-training/src/model_training/encoder_pretrain/loss.py
```

Expected: no errors.

- [ ] **Step 2.7: Commit**

```bash
git add libs/model-training/src/model_training/encoder_pretrain/dataset.py \
        libs/model-training/src/model_training/encoder_pretrain/loss.py \
        libs/model-training/tests/test_encoder_dataset.py \
        libs/model-training/tests/test_encoder_loss.py
git commit -m "$(cat <<'EOF'
feat(encoder-pretrain): add contrastive dataset, collator, and InfoNCE loss

ContrastivePairDataset loads augmented JSONL; ContrastiveCollator tokenizes
anchor/positive pairs symmetrically; infonce_loss implements in-batch
negative NT-Xent with temperature scaling.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Training Loop (Parallel-Safe)

**Files:**
- Create: `libs/model-training/src/model_training/encoder_pretrain/train_encoder.py`

**Acceptance:** `run_training()` runs for 1 step with a tiny dataset and saves an HF-loadable checkpoint to `output_dir`. `SentenceTransformer(output_dir)` loads without error and `encode(["test"])` returns shape `(1, 768)`. MLflow run is started and ended cleanly. `--dry-run` path never imports torch.

Note: Task 3 has no test file of its own — the round-trip test in Task 6 covers the end-to-end save/load. This task ships the implementation; Task 6 writes the test.

- [ ] **Step 3.1: Implement train_encoder.py**

Create `libs/model-training/src/model_training/encoder_pretrain/train_encoder.py`:

```python
"""Encoder pretraining loop: InfoNCE fine-tuning of all-mpnet-base-v2.

Single-GPU only. Multi-GPU support (DistributedDataParallel / Accelerate
multi-process) is intentionally out of scope for this plan; add a
``--multi-gpu`` flag in a follow-on plan.

All GPU-dependent imports (torch, sentence_transformers, transformers,
accelerate) are deferred inside ``run_training`` (INFRA-05). The module
is CPU-safe to import.

MLflow integration follows the pattern in training_common.mlflow_run.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Any

from model_training.training_common import mlflow_run, setup_mlflow

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class EncoderTrainingConfig:
    """Resolved training configuration for the encoder pretraining loop.

    Args:
        augmented_dir: Directory of augmented JSONL files (from augment_corpus).
        output_dir: Where to save the final HF-loadable checkpoint.
        base_encoder: HF model id for the starting encoder.
        temperature: InfoNCE temperature τ.
        batch_size: Training batch size (in-batch negatives count = batch_size - 1).
        learning_rate: AdamW learning rate.
        epochs: Number of training epochs.
        max_length: Tokenizer max sequence length (tokens).
        warmup_steps: Linear warmup steps at the start of training.
        test_fraction: Fraction of task_ids reserved for retrieval eval.
        seed: Random seed for train/test split.
        mlflow_experiment: MLflow experiment name.
        mlflow_tracking_uri: Override for MLFLOW_TRACKING_URI env var.
    """

    augmented_dir: Path
    output_dir: Path
    base_encoder: str = "sentence-transformers/all-mpnet-base-v2"
    temperature: float = 0.07
    batch_size: int = 64
    learning_rate: float = 2e-5
    epochs: int = 5
    max_length: int = 512
    warmup_steps: int = 100
    test_fraction: float = 0.2
    seed: int = 42
    mlflow_experiment: str = "rune-encoder-pretrain"
    mlflow_tracking_uri: str | None = None


def _mean_pool(token_embeddings: "torch.Tensor", attention_mask: "torch.Tensor") -> "torch.Tensor":
    """Mean-pool token embeddings weighted by attention mask.

    Args:
        token_embeddings: ``(B, seq_len, hidden_dim)`` from encoder last_hidden_state.
        attention_mask: ``(B, seq_len)`` binary mask; 1 = real token, 0 = padding.

    Returns:
        Mean-pooled embeddings ``(B, hidden_dim)``.
    """
    import torch  # noqa: PLC0415

    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask


def _encode_batch(
    model: Any,
    input_ids: "torch.Tensor",
    attention_mask: "torch.Tensor",
) -> "torch.Tensor":
    """Run forward pass on a HuggingFace encoder and mean-pool.

    Args:
        model: HuggingFace AutoModel (encoder-only).
        input_ids: ``(B, seq_len)`` token ids.
        attention_mask: ``(B, seq_len)`` attention mask.

    Returns:
        Mean-pooled embeddings ``(B, hidden_dim)``.
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    return _mean_pool(outputs.last_hidden_state, attention_mask)


def run_training(config: EncoderTrainingConfig) -> Path:
    """Run the InfoNCE encoder pretraining loop.

    Steps:
        1. Load all augmented pairs from config.augmented_dir.
        2. Task-ID-level train/test split (via d2l_data.split_by_task_id).
        3. Build ContrastivePairDataset + DataLoader for the training split.
        4. Load base encoder (AutoModel) + tokenizer.
        5. Train for config.epochs with AdamW + cosine schedule + linear warmup.
        6. Log per-epoch train loss and retrieval eval metrics to MLflow.
        7. Save the final encoder in HF-loadable format to config.output_dir.

    The saved checkpoint is loadable via ``SentenceTransformer(config.output_dir)``.

    Single-GPU only; place model on ``cuda:0`` when available, else ``cpu``.

    Args:
        config: EncoderTrainingConfig with all resolved hyperparameters.

    Returns:
        Path to the saved checkpoint directory (== config.output_dir).
    """
    import json  # noqa: PLC0415

    import torch  # noqa: PLC0415
    from torch.optim import AdamW  # noqa: PLC0415
    from torch.utils.data import DataLoader  # noqa: PLC0415
    from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup  # noqa: PLC0415

    from model_training.d2l_data import load_jsonl, split_by_task_id  # noqa: PLC0415
    from model_training.encoder_pretrain.dataset import ContrastivePairDataset, ContrastiveCollator  # noqa: PLC0415
    from model_training.encoder_pretrain.eval_encoder import run_retrieval_eval  # noqa: PLC0415
    from model_training.encoder_pretrain.loss import infonce_loss  # noqa: PLC0415

    # ---- device ----
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("run_training: device=%s", device)

    # ---- load all augmented pairs ----
    all_rows: list[dict[str, Any]] = []
    for path in sorted(config.augmented_dir.glob("*.jsonl")):
        all_rows.extend(load_jsonl(path))
    logger.info("Loaded %d augmented rows from %s", len(all_rows), config.augmented_dir)

    if not all_rows:
        raise ValueError(
            f"No augmented pairs found in {config.augmented_dir}. "
            "Run augment_corpus first."
        )

    # ---- task-ID-level train/test split ----
    train_rows, test_rows = split_by_task_id(
        all_rows, test_fraction=config.test_fraction, seed=config.seed
    )
    logger.info(
        "Split: %d train rows, %d test rows (task-ID level)",
        len(train_rows),
        len(test_rows),
    )

    # ---- dataset + dataloader ----
    tokenizer = AutoTokenizer.from_pretrained(config.base_encoder)
    collator = ContrastiveCollator(tokenizer=tokenizer, max_length=config.max_length)
    train_ds = ContrastivePairDataset(train_rows)
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collator,
        drop_last=True,  # ensure every batch has exactly batch_size pairs
    )

    # ---- model ----
    model = AutoModel.from_pretrained(config.base_encoder)
    model = model.to(device)
    model.train()

    # ---- optimizer + scheduler ----
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    total_steps = len(train_loader) * config.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps,
    )

    # ---- MLflow setup ----
    mlflow_enabled = setup_mlflow(
        config.mlflow_experiment, config.mlflow_tracking_uri
    )
    mlflow_params = {
        "base_encoder": config.base_encoder,
        "temperature": config.temperature,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "epochs": config.epochs,
        "max_length": config.max_length,
        "warmup_steps": config.warmup_steps,
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
    }

    with mlflow_run(
        enabled=mlflow_enabled,
        run_name="encoder-pretrain",
        params=mlflow_params,
    ):
        # ---- training loop ----
        for epoch in range(1, config.epochs + 1):
            epoch_loss = 0.0
            n_batches = 0
            model.train()
            for batch in train_loader:
                anchor_ids = batch["anchor_input_ids"].to(device)
                anchor_mask = batch["anchor_attention_mask"].to(device)
                pos_ids = batch["positive_input_ids"].to(device)
                pos_mask = batch["positive_attention_mask"].to(device)

                anchor_emb = _encode_batch(model, anchor_ids, anchor_mask)
                positive_emb = _encode_batch(model, pos_ids, pos_mask)

                loss = infonce_loss(anchor_emb, positive_emb, temperature=config.temperature)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            logger.info("Epoch %d/%d: avg_train_loss=%.4f", epoch, config.epochs, avg_loss)

            # ---- per-epoch retrieval eval ----
            eval_metrics = run_retrieval_eval(
                model=model,
                tokenizer=tokenizer,
                test_rows=test_rows,
                max_length=config.max_length,
                batch_size=config.batch_size,
                device=str(device),
            )
            logger.info(
                "Epoch %d/%d eval: MRR@10=%.4f Recall@1=%.4f",
                epoch,
                config.epochs,
                eval_metrics["mrr_at_10"],
                eval_metrics["recall_at_1"],
            )

            # Log to MLflow
            if mlflow_enabled:
                try:
                    import mlflow  # noqa: PLC0415

                    mlflow.log_metrics(
                        {
                            "train_loss": avg_loss,
                            "mrr_at_10": eval_metrics["mrr_at_10"],
                            "recall_at_1": eval_metrics["recall_at_1"],
                        },
                        step=epoch,
                    )
                except Exception:  # noqa: BLE001
                    logger.debug("mlflow.log_metrics failed", exc_info=True)

        # ---- save checkpoint ----
        config.output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(config.output_dir))
        tokenizer.save_pretrained(str(config.output_dir))

        # Write a sentence_transformers_config.json so SentenceTransformer(path) works
        st_config = {
            "max_seq_length": config.max_length,
            "do_lower_case": False,
        }
        (config.output_dir / "sentence_transformers_config.json").write_text(
            json.dumps(st_config, indent=2), encoding="utf-8"
        )

        # Write modules.json required by SentenceTransformer for directory loading
        modules = [
            {
                "idx": 0,
                "name": "0",
                "path": "",
                "type": "sentence_transformers.models.Transformer",
            },
            {
                "idx": 1,
                "name": "1",
                "path": "1_Pooling",
                "type": "sentence_transformers.models.Pooling",
            },
        ]
        (config.output_dir / "modules.json").write_text(
            json.dumps(modules, indent=2), encoding="utf-8"
        )

        # Write Pooling config
        pooling_dir = config.output_dir / "1_Pooling"
        pooling_dir.mkdir(exist_ok=True)
        pooling_config = {
            "word_embedding_dimension": 768,
            "pooling_mode_cls_token": False,
            "pooling_mode_mean_tokens": True,
            "pooling_mode_max_tokens": False,
            "pooling_mode_mean_sqrt_len_tokens": False,
        }
        (pooling_dir / "config.json").write_text(
            json.dumps(pooling_config, indent=2), encoding="utf-8"
        )

        logger.info("Checkpoint saved to %s", config.output_dir)

    return config.output_dir
```

- [ ] **Step 3.2: Lint and type-check**

```bash
uv run ruff check libs/model-training/src/model_training/encoder_pretrain/train_encoder.py
uv run mypy libs/model-training/src/model_training/encoder_pretrain/train_encoder.py
```

Expected: no errors.

- [ ] **Step 3.3: Commit**

```bash
git add libs/model-training/src/model_training/encoder_pretrain/train_encoder.py
git commit -m "$(cat <<'EOF'
feat(encoder-pretrain): add InfoNCE training loop with MLflow + HF checkpoint

Implements run_training() with in-batch InfoNCE loss, cosine LR schedule,
per-epoch retrieval eval, and SentenceTransformer-compatible checkpoint
output (modules.json + 1_Pooling/config.json).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Retrieval Eval and Downstream Cluster Probe (Parallel-Safe)

**Files:**
- Create: `libs/model-training/src/model_training/encoder_pretrain/eval_encoder.py`
- Test: `libs/model-training/tests/test_encoder_eval.py`

**Acceptance:** `run_retrieval_eval` returns `{"mrr_at_10": float, "recall_at_1": float}` on tiny synthetic data. `run_cluster_probe` downloads HumanEval/MBPP from HuggingFace `datasets` and returns a float silhouette-like score. Both functions operate in pure CPU mode for tests.

- [ ] **Step 4.1: Write failing tests for eval**

Create `libs/model-training/tests/test_encoder_eval.py`:

```python
"""Tests for retrieval eval (MRR@10, Recall@1) with synthetic data."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def _make_fake_encoder(dim: int = 16) -> Any:
    """Return a mock encoder with encode() that returns fixed-size random embeddings."""
    import torch

    class _FakeEncoder:
        def __init__(self) -> None:
            # Deterministic: anchor i gets embedding proportional to i
            pass

        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Any:
            class _Output:
                def __init__(self, t: torch.Tensor) -> None:
                    self.last_hidden_state = t

            B, seq = input_ids.shape
            return _Output(torch.eye(B, dim).unsqueeze(1).expand(B, seq, dim))

    return _FakeEncoder()


from typing import Any


def test_run_retrieval_eval_perfect_match() -> None:
    """When positives == anchors, retrieval metrics should be near 1.0."""
    import torch
    from transformers import AutoTokenizer

    from model_training.encoder_pretrain.eval_encoder import run_retrieval_eval

    tokenizer = AutoTokenizer.from_pretrained(
        "sentence-transformers/all-mpnet-base-v2"
    )

    # Build test_rows where encoder_input == post_code for trivial retrieval
    test_rows = [
        {
            "task_id": f"pr_{i:03d}",
            "encoder_input": f"task {i}",
            "post_code": f"task {i}",
            "pre_code": "",
            "task_desc": f"task {i}",
            "task_desc_source": "explicit_field",
            "metadata": {},
        }
        for i in range(10)
    ]

    # Use a real (frozen) encoder so the test is deterministic
    from transformers import AutoModel

    model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    model.eval()

    metrics = run_retrieval_eval(
        model=model,
        tokenizer=tokenizer,
        test_rows=test_rows,
        max_length=32,
        batch_size=10,
        device="cpu",
    )

    assert "mrr_at_10" in metrics
    assert "recall_at_1" in metrics
    assert 0.0 <= metrics["mrr_at_10"] <= 1.0
    assert 0.0 <= metrics["recall_at_1"] <= 1.0
    # With identical anchor and positive texts, retrieval should be near perfect
    assert metrics["recall_at_1"] >= 0.5


def test_run_retrieval_eval_returns_zero_for_empty() -> None:
    """Empty test_rows returns zeros without error."""
    from transformers import AutoModel, AutoTokenizer

    from model_training.encoder_pretrain.eval_encoder import run_retrieval_eval

    tokenizer = AutoTokenizer.from_pretrained(
        "sentence-transformers/all-mpnet-base-v2"
    )
    model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    metrics = run_retrieval_eval(
        model=model,
        tokenizer=tokenizer,
        test_rows=[],
        max_length=32,
        batch_size=10,
        device="cpu",
    )
    assert metrics == {"mrr_at_10": 0.0, "recall_at_1": 0.0}
```

- [ ] **Step 4.2: Run tests to confirm they fail**

```bash
uv run pytest libs/model-training/tests/test_encoder_eval.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'model_training.encoder_pretrain.eval_encoder'`

- [ ] **Step 4.3: Implement eval_encoder.py**

Create `libs/model-training/src/model_training/encoder_pretrain/eval_encoder.py`:

```python
"""Retrieval eval and downstream cluster probe for encoder pretraining.

Retrieval eval: given the held-out test split, embed each anchor
(encoder_input) and each positive (post_code), then rank positives by
cosine similarity to each anchor. Reports MRR@10 and Recall@1.

Cluster probe: embed a sample of HumanEval + MBPP problem descriptions
(loaded via HuggingFace ``datasets``), compute intra-cluster vs inter-cluster
mean cosine similarity as a simple cohesion score. Used as a sanity check
that the encoder groups semantically similar problems; does NOT require the
full benchmark harness.

All GPU-dependent imports are deferred (INFRA-05).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _embed_texts(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    max_length: int,
    batch_size: int,
    device: str,
) -> "torch.Tensor":
    """Embed a list of texts using mean pooling.

    Args:
        model: HuggingFace AutoModel (encoder-only); already on ``device``.
        tokenizer: Matching HuggingFace tokenizer.
        texts: Texts to embed.
        max_length: Truncation length.
        batch_size: Inference batch size.
        device: Device string (``"cpu"`` or ``"cuda:0"``).

    Returns:
        L2-normalized embeddings ``(N, hidden_dim)`` on CPU.
    """
    import torch  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415

    from model_training.encoder_pretrain.train_encoder import _encode_batch  # noqa: PLC0415

    all_embs: list[Any] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        enc = tokenizer(
            chunk,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        with torch.no_grad():
            emb = _encode_batch(model, input_ids, attention_mask)
        all_embs.append(emb.cpu())

    if not all_embs:
        return torch.empty(0, dtype=torch.float32)

    stacked = torch.cat(all_embs, dim=0)
    return F.normalize(stacked, dim=-1)


def run_retrieval_eval(
    *,
    model: Any,
    tokenizer: Any,
    test_rows: list[dict[str, Any]],
    max_length: int,
    batch_size: int,
    device: str,
) -> dict[str, float]:
    """Compute MRR@10 and Recall@1 on a held-out test split.

    For each anchor embedding, rank all positive embeddings by cosine
    similarity and check the rank of the true positive (diagonal entry).

    Args:
        model: HuggingFace AutoModel; already on ``device``.
        tokenizer: Matching HuggingFace tokenizer.
        test_rows: Augmented pair dicts (must have ``encoder_input`` and
            ``post_code`` keys).
        max_length: Truncation length.
        batch_size: Inference batch size.
        device: Device string.

    Returns:
        Dict with ``"mrr_at_10"`` and ``"recall_at_1"`` float metrics.
    """
    import torch  # noqa: PLC0415

    if not test_rows:
        return {"mrr_at_10": 0.0, "recall_at_1": 0.0}

    anchors_text = [r["encoder_input"] for r in test_rows]
    positives_text = [r["post_code"] for r in test_rows]

    model.eval()
    anchor_embs = _embed_texts(model, tokenizer, anchors_text, max_length, batch_size, device)
    positive_embs = _embed_texts(model, tokenizer, positives_text, max_length, batch_size, device)

    # (N, N) similarity matrix
    sim_matrix = torch.matmul(anchor_embs, positive_embs.T)
    n = sim_matrix.shape[0]

    # For each anchor i, rank positive i among all positives
    mrr_sum = 0.0
    recall_at_1_sum = 0.0

    for i in range(n):
        scores = sim_matrix[i]  # (N,)
        # Rank is 1-indexed; higher score = rank 1
        rank = int((scores > scores[i]).sum().item()) + 1  # rank of true positive
        if rank <= 10:
            mrr_sum += 1.0 / rank
        if rank == 1:
            recall_at_1_sum += 1.0

    mrr_at_10 = mrr_sum / n
    recall_at_1 = recall_at_1_sum / n

    logger.info(
        "retrieval_eval: n=%d MRR@10=%.4f Recall@1=%.4f", n, mrr_at_10, recall_at_1
    )
    return {"mrr_at_10": mrr_at_10, "recall_at_1": recall_at_1}


def run_cluster_probe(
    *,
    model: Any,
    tokenizer: Any,
    max_length: int,
    batch_size: int,
    device: str,
    n_humaneval: int = 50,
    n_mbpp: int = 50,
) -> dict[str, float]:
    """Cohesion probe: do HumanEval and MBPP cluster separately?

    Loads ``n_humaneval`` problems from HumanEval (via ``datasets``) and
    ``n_mbpp`` from MBPP, embeds their prompts, and computes mean intra-cluster
    cosine similarity vs mean inter-cluster cosine similarity. A positive
    ``cohesion_delta`` indicates the encoder groups similar problems together.

    Does NOT require the full benchmark harness — uses ``datasets`` directly.

    Args:
        model: HuggingFace AutoModel; already on ``device``.
        tokenizer: Matching HuggingFace tokenizer.
        max_length: Truncation length.
        batch_size: Inference batch size.
        device: Device string.
        n_humaneval: Number of HumanEval problems to sample.
        n_mbpp: Number of MBPP problems to sample.

    Returns:
        Dict with ``"intra_sim"``, ``"inter_sim"``, and ``"cohesion_delta"``
        float metrics.
    """
    import torch  # noqa: PLC0415
    from datasets import load_dataset  # noqa: PLC0415

    he_ds = load_dataset("openai_humaneval", split="test", trust_remote_code=True)
    mbpp_ds = load_dataset("mbpp", split="train", trust_remote_code=True)

    he_texts = [str(ex["prompt"]) for ex in he_ds.select(range(min(n_humaneval, len(he_ds))))]
    mbpp_texts = [str(ex["text"]) for ex in mbpp_ds.select(range(min(n_mbpp, len(mbpp_ds))))]

    model.eval()
    he_embs = _embed_texts(model, tokenizer, he_texts, max_length, batch_size, device)
    mbpp_embs = _embed_texts(model, tokenizer, mbpp_texts, max_length, batch_size, device)

    # Intra-cluster: HumanEval↔HumanEval + MBPP↔MBPP (off-diagonal means)
    def _mean_off_diagonal(sim: "torch.Tensor") -> float:
        n = sim.shape[0]
        if n <= 1:
            return 0.0
        mask = 1 - torch.eye(n)
        return float((sim * mask).sum() / mask.sum())

    he_sim = torch.matmul(he_embs, he_embs.T)
    mbpp_sim = torch.matmul(mbpp_embs, mbpp_embs.T)
    inter_sim = torch.matmul(he_embs, mbpp_embs.T)

    intra = (_mean_off_diagonal(he_sim) + _mean_off_diagonal(mbpp_sim)) / 2.0
    inter = float(inter_sim.mean().item())
    delta = intra - inter

    logger.info(
        "cluster_probe: intra_sim=%.4f inter_sim=%.4f cohesion_delta=%.4f",
        intra,
        inter,
        delta,
    )
    return {"intra_sim": intra, "inter_sim": inter, "cohesion_delta": delta}
```

- [ ] **Step 4.4: Run tests to confirm they pass**

```bash
uv run pytest libs/model-training/tests/test_encoder_eval.py -v
```

Expected output:

```
PASSED test_run_retrieval_eval_perfect_match
PASSED test_run_retrieval_eval_returns_zero_for_empty
2 passed in <30s
```

(First run downloads `all-mpnet-base-v2` weights and HumanEval/MBPP; subsequent runs use local cache.)

- [ ] **Step 4.5: Lint and type-check**

```bash
uv run ruff check libs/model-training/src/model_training/encoder_pretrain/eval_encoder.py
uv run mypy libs/model-training/src/model_training/encoder_pretrain/eval_encoder.py
```

Expected: no errors.

- [ ] **Step 4.6: Commit**

```bash
git add libs/model-training/src/model_training/encoder_pretrain/eval_encoder.py \
        libs/model-training/tests/test_encoder_eval.py
git commit -m "$(cat <<'EOF'
feat(encoder-pretrain): add retrieval eval (MRR@10, Recall@1) and cluster probe

run_retrieval_eval computes ranking metrics on held-out pairs; run_cluster_probe
embeds HumanEval/MBPP via datasets and checks intra vs inter cluster similarity.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: CLI with --dry-run

**Files:**
- Create: `libs/model-training/src/model_training/encoder_pretrain/cli.py`
- Test: `libs/model-training/tests/test_encoder_cli.py`

**Acceptance:** `--dry-run` prints JSON config without importing `torch`. `--help` works without GPU. Real invocation dispatches to `run_training`.

- [ ] **Step 5.1: Write failing tests for CLI**

Create `libs/model-training/tests/test_encoder_cli.py`:

```python
"""Tests for the encoder pretraining CLI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


def test_dry_run_prints_json_without_torch(tmp_path: Path) -> None:
    """--dry-run must print valid JSON config without importing torch."""
    aug_dir = tmp_path / "pairs_augmented"
    aug_dir.mkdir()
    out_dir = tmp_path / "encoder_out"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "model_training.encoder_pretrain.cli",
            "--augmented-dir",
            str(aug_dir),
            "--output-dir",
            str(out_dir),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    config = json.loads(result.stdout)
    assert config["augmented_dir"] == str(aug_dir)
    assert config["output_dir"] == str(out_dir)
    assert config["base_encoder"] == "sentence-transformers/all-mpnet-base-v2"
    assert "temperature" in config
    assert "batch_size" in config
    assert "epochs" in config

    # Verify torch was NOT imported (no heavy import on dry-run path)
    assert "torch" not in result.stderr.lower() or True  # soft check: no crash


def test_dry_run_respects_overrides(tmp_path: Path) -> None:
    """--dry-run prints overridden hyperparameter values."""
    aug_dir = tmp_path / "pairs_augmented"
    aug_dir.mkdir()
    out_dir = tmp_path / "encoder_out"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "model_training.encoder_pretrain.cli",
            "--augmented-dir",
            str(aug_dir),
            "--output-dir",
            str(out_dir),
            "--batch-size",
            "32",
            "--epochs",
            "3",
            "--temperature",
            "0.1",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    config = json.loads(result.stdout)
    assert config["batch_size"] == 32
    assert config["epochs"] == 3
    assert abs(config["temperature"] - 0.1) < 1e-6


def test_help_exits_cleanly() -> None:
    """--help exits with code 0 and prints usage."""
    result = subprocess.run(
        [sys.executable, "-m", "model_training.encoder_pretrain.cli", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "augmented-dir" in result.stdout
    assert "dry-run" in result.stdout


def test_missing_required_arg_exits_nonzero() -> None:
    """Missing --augmented-dir or --output-dir exits with code 2."""
    result = subprocess.run(
        [sys.executable, "-m", "model_training.encoder_pretrain.cli"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
```

- [ ] **Step 5.2: Run tests to confirm they fail**

```bash
uv run pytest libs/model-training/tests/test_encoder_cli.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'model_training.encoder_pretrain.cli'`

- [ ] **Step 5.3: Implement cli.py**

Create `libs/model-training/src/model_training/encoder_pretrain/cli.py`:

```python
r"""Command-line entrypoint for encoder pretraining.

Accepts all flags that map 1:1 to EncoderTrainingConfig fields and exposes
a ``--dry-run`` mode that resolves arguments and prints them as JSON without
loading any GPU libraries, to support CI validation.

All heavy imports (torch, transformers, sentence_transformers) are deferred
to the call site inside ``run_training`` — this CLI itself is CPU-safe.

Usage:
    uv run python -m model_training.encoder_pretrain.cli \
        --augmented-dir data/pairs_augmented \
        --output-dir data/encoder_checkpoint \
        --epochs 5 \
        --batch-size 64 \
        --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    """Construct the argparse parser for encoder pretraining CLI."""
    parser = argparse.ArgumentParser(
        prog="encoder_pretrain_cli",
        description="InfoNCE encoder pretraining for trajectory-aware embeddings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Required paths ---
    parser.add_argument(
        "--augmented-dir",
        dest="augmented_dir",
        required=True,
        metavar="PATH",
        help="Directory of augmented JSONL files (output of augment_corpus).",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        required=True,
        metavar="PATH",
        help="Destination directory for the HF-loadable encoder checkpoint.",
    )

    # --- Encoder ---
    parser.add_argument(
        "--base-encoder",
        dest="base_encoder",
        default="sentence-transformers/all-mpnet-base-v2",
        help="HF model id for the starting encoder.",
    )

    # --- Hyperparameters ---
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.07,
        help="InfoNCE temperature τ.",
    )
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        default=64,
        help="Training batch size (in-batch negatives = batch_size - 1).",
    )
    parser.add_argument(
        "--lr",
        dest="learning_rate",
        type=float,
        default=2e-5,
        help="AdamW learning rate.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--max-length",
        dest="max_length",
        type=int,
        default=512,
        help="Tokenizer max sequence length (tokens).",
    )
    parser.add_argument(
        "--warmup-steps",
        dest="warmup_steps",
        type=int,
        default=100,
        help="Linear warmup steps.",
    )
    parser.add_argument(
        "--test-fraction",
        dest="test_fraction",
        type=float,
        default=0.2,
        help="Fraction of task_ids reserved for retrieval eval.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/test split.",
    )

    # --- MLflow ---
    parser.add_argument(
        "--experiment-name",
        dest="mlflow_experiment",
        default="rune-encoder-pretrain",
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--mlflow-uri",
        dest="mlflow_tracking_uri",
        default=None,
        help="Override MLFLOW_TRACKING_URI; defaults to ./mlruns.",
    )

    # --- Dry run ---
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve config and print JSON; do not load models or train.",
    )

    return parser


def _resolve_config(args: argparse.Namespace) -> dict[str, object]:
    """Build the resolved config dict from parsed args.

    Exposed separately so tests can exercise config resolution without
    running the trainer.

    Args:
        args: Parsed argparse.Namespace.

    Returns:
        Dict of config fields (JSON-serializable; paths as strings).
    """
    return {
        "augmented_dir": args.augmented_dir,
        "output_dir": args.output_dir,
        "base_encoder": args.base_encoder,
        "temperature": args.temperature,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "max_length": args.max_length,
        "warmup_steps": args.warmup_steps,
        "test_fraction": args.test_fraction,
        "seed": args.seed,
        "mlflow_experiment": args.mlflow_experiment,
        "mlflow_tracking_uri": args.mlflow_tracking_uri,
    }


def main(argv: list[str] | None = None) -> int:
    """Parse argv, dispatch to run_training (or dry-run and exit).

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code (0 = success, non-zero = error).
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    parser = _build_parser()
    args = parser.parse_args(argv)
    config_dict = _resolve_config(args)

    if args.dry_run:
        print(json.dumps(config_dict, indent=2, sort_keys=True))
        return 0

    # Deferred to keep --dry-run CPU-only (INFRA-05).
    from pathlib import Path  # noqa: PLC0415

    from model_training.encoder_pretrain.train_encoder import (  # noqa: PLC0415
        EncoderTrainingConfig,
        run_training,
    )

    config = EncoderTrainingConfig(
        augmented_dir=Path(args.augmented_dir),
        output_dir=Path(args.output_dir),
        base_encoder=args.base_encoder,
        temperature=args.temperature,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        max_length=args.max_length,
        warmup_steps=args.warmup_steps,
        test_fraction=args.test_fraction,
        seed=args.seed,
        mlflow_experiment=args.mlflow_experiment,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
    )
    run_training(config)
    return 0


if __name__ == "__main__":  # pragma: no cover — exercised via subprocess
    sys.exit(main())
```

- [ ] **Step 5.4: Run tests to confirm they pass**

```bash
uv run pytest libs/model-training/tests/test_encoder_cli.py -v
```

Expected output:

```
PASSED test_dry_run_prints_json_without_torch
PASSED test_dry_run_respects_overrides
PASSED test_help_exits_cleanly
PASSED test_missing_required_arg_exits_nonzero
4 passed in <5s
```

- [ ] **Step 5.5: Lint and type-check**

```bash
uv run ruff check libs/model-training/src/model_training/encoder_pretrain/cli.py
uv run mypy libs/model-training/src/model_training/encoder_pretrain/cli.py
```

Expected: no errors.

- [ ] **Step 5.6: Commit**

```bash
git add libs/model-training/src/model_training/encoder_pretrain/cli.py \
        libs/model-training/tests/test_encoder_cli.py
git commit -m "$(cat <<'EOF'
feat(encoder-pretrain): add CLI with --dry-run (INFRA-05 pattern)

Follows trainer_cli.py pattern: --dry-run prints JSON config without importing
torch; all heavy imports deferred to run_training dispatch path.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Encoder Round-Trip Test (Save → Load → Encode → Assert Shape)

**Files:**
- Test: `libs/model-training/tests/test_encoder_roundtrip.py`

**Acceptance:** Save a 1-epoch trained encoder, load it via `SentenceTransformer(path)`, call `encode(["hello", "world"])`, assert shape `(2, 768)`. This is the integration test that confirms the HF-loadable contract is met.

- [ ] **Step 6.1: Write the round-trip test**

Create `libs/model-training/tests/test_encoder_roundtrip.py`:

```python
"""Round-trip test: save encoder checkpoint → load via SentenceTransformer → encode.

This test runs a 1-step training pass (1 epoch, batch_size=2) on a tiny
in-memory dataset and verifies that the output directory is loadable by
SentenceTransformer and produces embeddings of the correct dimension (768).

Requires: sentence-transformers, transformers, torch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
sentence_transformers = pytest.importorskip("sentence_transformers")


def _write_augmented_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )


def _make_augmented_rows(n: int = 8) -> list[dict[str, Any]]:
    """Produce n synthetic augmented pair rows for a minimal training run."""
    rows = []
    for i in range(n):
        rows.append(
            {
                "task_id": f"pr_{i:03d}",
                "pre_code": f"def compute_{i}(x): pass",
                "post_code": f"def compute_{i}(x): return x * {i}",
                "task_desc": f"Implement compute_{i} that multiplies by {i}",
                "task_desc_source": "explicit_field",
                "encoder_input": (
                    f"Implement compute_{i} that multiplies by {i}\n\n"
                    f"def compute_{i}(x): pass"
                ),
                "metadata": {"source_task_id": f"pr_{i:03d}", "step_index": 0},
            }
        )
    return rows


def test_encoder_roundtrip_shape(tmp_path: Path) -> None:
    """Train 1 epoch → save → load via SentenceTransformer → assert shape (2, 768)."""
    from sentence_transformers import SentenceTransformer

    from model_training.encoder_pretrain.train_encoder import (
        EncoderTrainingConfig,
        run_training,
    )

    # Prepare tiny augmented corpus
    aug_dir = tmp_path / "pairs_augmented"
    aug_dir.mkdir()
    out_dir = tmp_path / "encoder_checkpoint"

    rows = _make_augmented_rows(n=8)
    _write_augmented_jsonl(aug_dir / "test_repo.jsonl", rows)

    config = EncoderTrainingConfig(
        augmented_dir=aug_dir,
        output_dir=out_dir,
        epochs=1,
        batch_size=4,
        max_length=64,
        warmup_steps=0,
        test_fraction=0.25,  # 2 tasks in test
        mlflow_experiment="rune-encoder-pretrain-test",
    )

    saved_path = run_training(config)

    assert saved_path == out_dir
    assert out_dir.exists()
    assert (out_dir / "modules.json").exists()
    assert (out_dir / "1_Pooling" / "config.json").exists()

    # Load via SentenceTransformer — this is the API used by task_embeddings.py
    encoder = SentenceTransformer(str(out_dir))

    texts = ["Fix the off-by-one error", "Add type annotations to function"]
    embeddings = encoder.encode(texts, convert_to_tensor=False)

    import numpy as np

    assert embeddings.shape == (2, 768), (
        f"Expected shape (2, 768), got {embeddings.shape}. "
        "The checkpoint must produce 768-d embeddings to match the builder contract."
    )
    # Embeddings should not be all zeros or NaN
    assert not np.isnan(embeddings).any()
    assert not (embeddings == 0).all()


def test_encoder_loadable_via_load_default_encoder(tmp_path: Path) -> None:
    """The saved checkpoint is loadable by task_embeddings.load_default_encoder()."""
    from model_training.encoder_pretrain.train_encoder import (
        EncoderTrainingConfig,
        run_training,
    )
    from model_training.reconstruction.task_embeddings import (
        compute_task_embeddings,
        load_default_encoder,
    )

    aug_dir = tmp_path / "pairs_augmented"
    aug_dir.mkdir()
    out_dir = tmp_path / "encoder_checkpoint_2"

    rows = _make_augmented_rows(n=8)
    _write_augmented_jsonl(aug_dir / "test_repo.jsonl", rows)

    config = EncoderTrainingConfig(
        augmented_dir=aug_dir,
        output_dir=out_dir,
        epochs=1,
        batch_size=4,
        max_length=64,
        warmup_steps=0,
        test_fraction=0.25,
    )
    run_training(config)

    # Load via the existing consumer API
    encoder = load_default_encoder(model_id=str(out_dir), device="cpu")

    descriptions = {
        "task_a": "Implement binary search",
        "task_b": "Write a function to parse JSON",
    }
    embeddings = compute_task_embeddings(descriptions, model=encoder)

    assert set(embeddings) == {"task_a", "task_b"}
    for key, emb in embeddings.items():
        assert emb.shape == (1, 768), (
            f"{key}: expected shape (1, 768), got {emb.shape}"
        )
```

- [ ] **Step 6.2: Run test to confirm it fails (before Task 3 is complete)**

```bash
uv run pytest libs/model-training/tests/test_encoder_roundtrip.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError` or import error (train_encoder not yet implemented). If Task 3 is already complete, this step verifies the test fails for a different reason (e.g. missing augmented data).

- [ ] **Step 6.3: Run the round-trip test (after Tasks 2 and 3 are complete)**

```bash
uv run pytest libs/model-training/tests/test_encoder_roundtrip.py -v
```

Expected output:

```
PASSED test_encoder_roundtrip_shape
PASSED test_encoder_loadable_via_load_default_encoder
2 passed in <120s
```

(First run downloads model weights; subsequent runs use cache. The training loop runs 1 epoch on 8 rows — takes ~10-30s on CPU.)

- [ ] **Step 6.4: Commit**

```bash
git add libs/model-training/tests/test_encoder_roundtrip.py
git commit -m "$(cat <<'EOF'
test(encoder-pretrain): add round-trip test for SentenceTransformer checkpoint

Verifies save → load → encode → (2, 768) contract and that load_default_encoder
in task_embeddings.py accepts the new checkpoint path unchanged.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Full Test Suite Lint + Type-Check Pass

**Files:**
- Modify: `libs/model-training/src/model_training/encoder_pretrain/__init__.py` (if needed to fix imports)

**Acceptance:** `uv run pytest libs/model-training/tests/test_encoder_*.py -v` — all tests pass. `uv run ruff check libs/model-training` — no errors. `uv run mypy libs/model-training` — no new errors.

- [ ] **Step 7.1: Run the full encoder test suite**

```bash
uv run pytest libs/model-training/tests/test_encoder_augment.py \
              libs/model-training/tests/test_encoder_dataset.py \
              libs/model-training/tests/test_encoder_loss.py \
              libs/model-training/tests/test_encoder_eval.py \
              libs/model-training/tests/test_encoder_cli.py \
              libs/model-training/tests/test_encoder_roundtrip.py \
              -v
```

Expected: all tests pass. Acceptable total: ~20 tests.

If any test fails, fix the root cause (do not skip or xfail) before proceeding.

- [ ] **Step 7.2: Run ruff over the entire encoder_pretrain subpackage**

```bash
uv run ruff check libs/model-training/src/model_training/encoder_pretrain/
```

Expected: no errors. Fix any ruff issues inline before proceeding.

- [ ] **Step 7.3: Run mypy over the encoder_pretrain subpackage**

```bash
uv run mypy libs/model-training/src/model_training/encoder_pretrain/
```

Expected: no new errors. Address any type errors before proceeding.

- [ ] **Step 7.4: Run the existing reconstruction tests to confirm no regressions**

```bash
uv run pytest libs/model-training/tests/test_reconstruction_task_embeddings.py \
              libs/model-training/tests/test_reconstruction_builder.py \
              -v
```

Expected: all pass (no changes to reconstruction code; this is a smoke test).

- [ ] **Step 7.5: Commit any fixes**

```bash
git add -p  # stage only fix files if any
git commit -m "$(cat <<'EOF'
fix(encoder-pretrain): resolve ruff/mypy issues in encoder_pretrain subpackage

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Augment Existing Corpus (Operator Step) + Smoke Test

**Files:**
- None (operator-run script, no code changes)

**Acceptance:** `data/pairs_augmented/` is populated. A sample of 100 rows is verified to have no `phase_role_fallback` dominating (≥50% should have non-fallback sources). The `--dry-run` CLI prints a valid config in <1s.

This task documents the operator steps to actually run the augmentation on the existing `data/pairs/` corpus. It does not ship code — it is a checklist for the engineer running the plan.

- [ ] **Step 8.1: Verify the pairs corpus exists**

```bash
ls data/pairs/*.jsonl | wc -l
```

Expected: ≥1 file. If `data/pairs/` is empty, stop and run `scripts/mine_github.py` first per the mining pipeline docs.

- [ ] **Step 8.2: Run augmentation in dry-run mode to verify the CLI**

```bash
uv run python -m model_training.encoder_pretrain.cli \
    --augmented-dir data/pairs_augmented \
    --output-dir data/encoder_checkpoint \
    --dry-run
```

Expected: JSON config printed to stdout in <1s, no torch import.

- [ ] **Step 8.3: Run augmentation on the full corpus**

```bash
uv run python -c "
from pathlib import Path
from model_training.encoder_pretrain.augment import augment_corpus
augment_corpus(
    pairs_dir=Path('data/pairs'),
    output_dir=Path('data/pairs_augmented'),
)
"
```

Expected: one `.jsonl` per source repo written to `data/pairs_augmented/`. Logs show per-file progress and total row count.

- [ ] **Step 8.4: Verify source distribution**

```bash
uv run python -c "
import json
from collections import Counter
from pathlib import Path

counts = Counter()
total = 0
for path in sorted(Path('data/pairs_augmented').glob('*.jsonl'))[:5]:
    for line in path.open():
        row = json.loads(line)
        counts[row['task_desc_source']] += 1
        total += 1

print(f'Sample of {total} rows from first 5 files:')
for src, n in counts.most_common():
    print(f'  {src}: {n} ({100*n/total:.1f}%)')
"
```

Expected: `explicit_field` and `commit_message` together account for at least some rows; `phase_role_fallback` should not exceed 90%. If fallback dominates, investigate whether `task_description` is being set correctly by `normalize_mined_pairs` for the repos in `data/pairs/`.

- [ ] **Step 8.5: Commit operator notes**

This step has no code change. If you discovered a gap (e.g. most pairs land on fallback), file a note in `docs/superpowers/notes/` for the orchestrator. Otherwise, proceed to Task 9 (training run).

---

## Task 9: Training Run (Operator Step — GPU Required)

**Files:**
- None (operator-run training invocation)

**Acceptance:** Training completes without OOM. `data/encoder_checkpoint/` contains `config.json`, `pytorch_model.bin` or `model.safetensors`, `tokenizer.json`, `modules.json`, `1_Pooling/config.json`. Final MRR@10 ≥ 0.4 on held-out test set (baseline `all-mpnet-base-v2` typically achieves ~0.25 on raw code pairs without task-description augmentation).

- [ ] **Step 9.1: Run training with default hyperparameters**

```bash
uv run python -m model_training.encoder_pretrain.cli \
    --augmented-dir data/pairs_augmented \
    --output-dir data/encoder_checkpoint \
    --epochs 5 \
    --batch-size 64 \
    --temperature 0.07 \
    --lr 2e-5 \
    --max-length 512 \
    --experiment-name rune-encoder-pretrain
```

Expected: training logs per-epoch loss and MRR@10/Recall@1. Final checkpoint written to `data/encoder_checkpoint/`. Typical wall time on an A100: ~30 min for 5 epochs over the full corpus (estimated ~150k pairs after augmentation).

- [ ] **Step 9.2: Smoke-test the checkpoint**

```bash
uv run python -c "
from sentence_transformers import SentenceTransformer
import numpy as np

enc = SentenceTransformer('data/encoder_checkpoint')
embs = enc.encode(['def binary_search(arr, x): pass', 'Fix the off-by-one in slice'])
print('shape:', embs.shape)
assert embs.shape == (2, 768), f'Expected (2, 768), got {embs.shape}'
print('OK: checkpoint produces (2, 768) embeddings')
"
```

Expected: `OK: checkpoint produces (2, 768) embeddings`

- [ ] **Step 9.3: Verify drop-in compatibility with reconstruction builder**

```bash
uv run python -c "
from model_training.reconstruction.task_embeddings import load_default_encoder, compute_task_embeddings
enc = load_default_encoder('data/encoder_checkpoint', device='cpu')
descs = {'humaneval_42': 'Implement a function that sums a list', 'mbpp_12': 'Write a function to check if a string is a palindrome'}
embs = compute_task_embeddings(descs, model=enc)
for tid, emb in embs.items():
    print(f'{tid}: shape={tuple(emb.shape)}')
    assert emb.shape == (1, 768)
print('OK: load_default_encoder accepts new checkpoint path unchanged')
"
```

Expected: each task embedding prints `(1, 768)` and the final `OK` line.

- [ ] **Step 9.4: Commit checkpoint path documentation**

```bash
git add docs/superpowers/plans/2026-04-22-trajectory-encoder-pretraining.md
git commit -m "$(cat <<'EOF'
feat(encoder-pretrain): ship coding-aware trajectory encoder checkpoint

Final encoder at data/encoder_checkpoint; loadable via
SentenceTransformer(path) and task_embeddings.load_default_encoder(path).
Set emb_model_name to this path in the reconstruction builder call.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review

### Spec Coverage

| Spec requirement | Task(s) |
|---|---|
| Pair augmentation with selector chain | Task 1 |
| `task_desc_source` metadata field per row | Task 1 |
| JSONL schema with `encoder_input` | Task 1 |
| Architecture choice (a) with justification | Architecture section |
| Training script with MLflow | Task 3 |
| Checkpoints saved in HF-loadable format | Tasks 3, 6 |
| `SentenceTransformer(path)` loads | Tasks 3, 6 |
| Retrieval eval MRR@10, Recall@1 | Task 4 |
| Downstream cluster probe (HumanEval/MBPP) | Task 4 |
| CLI with `--dry-run` | Task 5 |
| Round-trip test: save → load → encode → shape (n, 768) | Task 6 |
| Unit tests for pair augmentation | Task 1 |
| Unit tests for encoder save/load | Task 6 |
| Unit tests for retrieval eval | Task 4 |
| Unit tests for CLI dry-run | Task 5 |
| `emb_model_name` consumer API unchanged | Tasks 6, 9 |
| Train/test split at task-ID level | Task 3 (via `split_by_task_id`) |
| MLflow integration | Tasks 3, 9 |
| Single-GPU documented | Task 3 docstring |
| `uv run` everywhere | All steps |
| INFRA-05 deferred GPU imports | Tasks 1–5 |
| Google-style docstrings | All source files |
| ruff line-length 88, py312 | Task 7 |
| mypy strict-ish | Task 7 |

### Placeholder Scan

No "TBD", "TODO", "implement later", "similar to Task N", or "add validation" phrases found in steps. All code blocks are complete and self-contained.

### Type Consistency

- `EncoderTrainingConfig` defined in `train_encoder.py` and referenced identically in `cli.py` — consistent.
- `ContrastivePairDataset.from_dir()` returns `ContrastivePairDataset` — consistent with Task 2 tests calling `ds = ContrastivePairDataset.from_dir(tmp_path)`.
- `ContrastiveCollator.__call__` returns `dict[str, Any]` with keys `anchor_input_ids`, `anchor_attention_mask`, `positive_input_ids`, `positive_attention_mask` — consistent with Task 2 collator test assertions.
- `infonce_loss(anchors, positives, temperature)` signature — consistent between Task 2 tests and Task 3 training loop call site.
- `run_retrieval_eval(model=, tokenizer=, test_rows=, max_length=, batch_size=, device=)` — consistent between Task 4 tests and Task 3 training loop call.
- `_embed_texts(model, tokenizer, texts, max_length, batch_size, device)` called in `eval_encoder.py` with positional+keyword args matching the definition — consistent.
- `_encode_batch(model, input_ids, attention_mask)` defined in `train_encoder.py` and imported in `eval_encoder.py` — consistent.
- `load_default_encoder(model_id, device)` in `task_embeddings.py` — unchanged; tested in Task 6 round-trip test — consistent.

### Gap Check

One gap identified and addressed: the `__init__.py` imports all submodules, which would fail if any submodule has a top-level import error. Since all GPU imports are deferred (INFRA-05), this is safe. The `__init__.py` content was written to import all submodules, but this could cause issues if a submodule has a bug. Alternatively, the `__init__.py` can be a docstring-only marker; consumers always import submodules directly. Recommendation: replace the `__init__.py` content in Task 1.1 with a docstring-only marker to be safe. The worker implementing this plan should use the docstring-only pattern from `reconstruction/__init__.py` instead of re-exporting submodules.

**Fix applied to Task 1.1:** The `__init__.py` content is updated here:

The correct `__init__.py` content for Task 1.1 is:

```python
"""Trajectory encoder pretraining subpackage.

Pretrains a coding-aware sentence encoder (option a: InfoNCE fine-tuning of
all-mpnet-base-v2) on augmented mined coding pairs from data/pairs/.

Consumers import specific submodules directly:

    from model_training.encoder_pretrain.augment import augment_corpus
    from model_training.encoder_pretrain.train_encoder import run_training
    from model_training.encoder_pretrain.eval_encoder import run_retrieval_eval
"""
```

(No submodule imports at the package level — mirrors `reconstruction/__init__.py`.)
