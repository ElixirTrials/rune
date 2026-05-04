# Handoff — Bulk Up Training Corpus

## Branch
`fix/diff-loss-per-turn-alignment`

## What's Done
- Quality-weighted SFT pipeline complete (scoring, collator integration, external ingestion)
- GraphQL mining refactor (10x fewer API calls, ~11 min for 25 repos at 50 PRs)
- 7,670 external codereview pairs ingested (`data/mined/external_codereview.unrolled.jsonl`)
- Mining re-run at 200 PRs/repo across 25 repos (in progress or complete)
- S3 synced to `s3://elixirtrials-949678234935-eu-west-2-artifacts/training-data/github-pairs/`
- 502 tests pass, ruff clean, mypy clean

## Bugs Fixed This Session
1. **quality_score silently dropped** — `_attach_assistant_masks` stripped `quality_score`
   column during pre-tokenization. Collator defaulted to 1.0 for all examples, making all
   quality scoring dead code. Fix: added `"quality_score"` to `preserve_columns`.
2. **test_mine_pr_trajectories broken by GraphQL refactor** — mock didn't include
   `fetch_pr_metadata_graphql` return value or commit author. Fixed mock fixture.
3. **test_search_quality_v2 referencing removed `_features_for_pr`** — rewrote to mock
   `search_and_score_prs_graphql` directly.

## Concatenation
```bash
# IMPORTANT: exclude all_unrolled.jsonl from glob to avoid self-inclusion
ls data/mined/*.unrolled.jsonl | grep -v all_unrolled | xargs cat > data/mined/all_unrolled.jsonl
```

## After bulk-up
1. Re-run audit: `uv run python scripts/audit_quality_scores.py`
2. Sync to S3: `aws s3 sync data/mined/ s3://elixirtrials-949678234935-eu-west-2-artifacts/training-data/github-pairs/ --include '*.jsonl'`
3. Train: `--dataset-path data/mined/all_unrolled.jsonl` with `--diff-aware-loss`

## Notes
- `instructions/mining_repos.json` now has `max_prs=200` (was 50)
- Do NOT store PATs in tracked files. Use `gh auth token` or `GITHUB_TOKEN` env var.
