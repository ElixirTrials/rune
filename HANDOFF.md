# Quality-Weighted SFT Pipeline — Handoff

## Plan File
`/Users/noahdolevelixir/.claude/plans/twinkling-crafting-dawn.md`

## Branch
`fix/diff-loss-per-turn-alignment` (existing branch, all work uncommitted)

## What's Done

### Phase A: Quality scoring kernel + unroll integration (COMPLETE)
- **`libs/model-training/src/model_training/d2l_quality.py`** — NEW. `QualityWeightConfig` dataclass, `score_episode_quality()`, `score_external_quality()`, `classify_causal_link()`, `is_url_only()`. Pure Python, no GPU deps. 21 unit tests passing.
- **`libs/model-training/tests/test_d2l_quality.py`** — NEW. 21 tests covering all scoring paths, floor, config overrides.
- **`libs/model-training/src/model_training/d2l_data.py`** — MODIFIED. `unroll_trajectory_to_pairs()` now accepts `quality_config` kwarg and emits `quality_score` float on each pair + in `pair["metadata"]`. `pairs_to_chat_messages()` propagates `quality_score` in pre_post_records (single-turn: passthrough, multi-turn: mean across group).
- **`libs/model-training/tests/test_d2l_unroll.py`** — EXTENDED. 3 new tests: `test_unroll_has_quality_score`, `test_unroll_ep0_not_penalized_by_causal`, `test_unroll_url_only_feedback_hits_floor`. All passing.

### Phase B: Corpus audit (COMPLETE)
- **`scripts/audit_quality_scores.py`** — NEW. Diagnostic script scoring all 1,344 episodes across 11 repos. Results:
  - 56% score >= 0.6, 26% in 0.10-0.30, 2.6% at floor (0.05)
  - 61% entity overlap, 39% no overlap, 0% URL-only
  - 48% rich feedback, 26% moderate, 26% short
  - Spot checks validated: lowest = "build" (5 chars) + 5K diff, highest = specific review comments with identifier overlap
  - 38 false-negative causal cases (rich text, no overlap) — score 0.28, acceptable
  - **No threshold changes needed** — defaults discriminate well

### Phase C: Training pipeline integration (PARTIALLY COMPLETE)
- **`libs/model-training/src/model_training/trainer.py`** — MODIFIED. `_build_training_dataset()` now includes `"quality_score": pp.get("quality_score", 1.0)` in Dataset rows when `diff_aware_loss=True`.
- **`libs/model-training/src/model_training/diff_loss.py`** — MODIFIED. `DiffWeightedDataCollator.__call__()` pops `quality_score` from features (default 1.0), multiplies into per-token weights after hunk computation: `[wi * q for wi in w]`. Skips multiplication when q == 1.0 for perf.

### Phase C remaining:
- **Add 2 tests to `test_diff_loss.py`** in `TestDiffWeightedDataCollator` class (ends at line 551):
  1. `test_quality_score_multiplied_into_weights` — feature with quality_score=0.5 produces half the weights of quality_score=1.0
  2. `test_quality_score_absent_defaults_to_one` — missing quality_score leaves weights unchanged
- **Run lint + type check**: `uv run ruff check libs/model-training/src/model_training/d2l_quality.py d2l_data.py diff_loss.py trainer.py` and `uv run mypy` on same files
- **`quality_min_score` subsampling param** — NOT YET added to `_build_training_dataset` / `train_qlora` signatures (plan Step 4 optional filter). Low priority, can defer.

## Phase D: External data ingestion (NOT STARTED)

### Step 6: `d2l_external.py` (new module)
- `load_codereview_dataset(split, max_rows, streaming)` — deferred HF datasets import
- `codereview_row_to_pair(row, quality_config) -> dict | None` — converts `ronantakizawa/github-codereview` rows to pair schema with `## Task / ## Current Code / ## Review Feedback / ## Revision` sections
- `ingest_codereview_to_pairs(split, max_rows, quality_config, min_quality_score) -> list[dict]`
- Marks `metadata.source = "external_codereview"`
- Note: dataset has its own `quality_score` column — ignore it, recompute with our heuristic
- Schema: `before_code` (snippet), `reviewer_comment` (text), `after_code` (snippet), `repo_name`, `pr_number`, `file_path`, `comment_line`, `comment_type`, `quality_score` (theirs), `is_negative` (bool — filter these out)

### Step 7: `scripts/ingest_codereview.py` (new CLI)
- Calls `ingest_codereview_to_pairs()` + `save_jsonl()`
- Args: `--output`, `--max-rows`, `--min-quality`, `--split`

### Step 8: Tests for external ingestion
- `tests/test_d2l_external.py` — codereview_row_to_pair returns None for degenerate rows, produces correct section structure, quality_score in [floor, 1.0], metadata.source set

## Key Architecture Decisions
- Quality score computed at **unroll time**, baked into JSONL (not computed on-the-fly at collation)
- **Multiplicative composite**: `max(0.05, source * causal * feedback * proportionality)`
- Composes with existing hunk weights: `final_weight[token] = quality_score * hunk_weight[token]`
- Multi-turn conversations use **mean** of constituent episode scores
- `quality_score=1.0` is **identity** — backward compatible, existing behavior preserved when field absent
- No `WeightedRandomSampler` — loss weighting handles gradient modulation

## How to Resume
1. Read the plan: `/Users/noahdolevelixir/.claude/plans/twinkling-crafting-dawn.md`
2. Finish Phase C: add 2 collator tests + lint/type check
3. Do Phase D: create `d2l_external.py`, `scripts/ingest_codereview.py`, `test_d2l_external.py`
4. Run full test suite: `uv run pytest libs/model-training/tests/ -x`
5. Commit everything
