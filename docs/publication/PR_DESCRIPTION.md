# docs(#52): publication task plan for TMLR submission

**Branch:** `docs/issue52-publication-plan` → base `chore/publication-cleanup`
**Type:** docs only, no code changes. **Venue committed:** TMLR.

## What this adds
Publication-readiness docs under `docs/publication/`, turning the v13 adversarial review into a tracked plan for getting `drafts/paper_v13.tex` submission-ready:

| File | Purpose |
|------|---------|
| `publication_task_plan.md` | **The execution tracker.** Phases 0–3, each task with owner / pre-registered gate / effort / dependency, plus GPU budget and a definition-of-done checklist. |
| `remediation_plan_FINAL.md` | The remediation plan the tracker draws from (v2, source-corrected). |
| `adversary_review.md` | Adversary red-team of the plan (verdict: *sound_with_revisions*, 12 critiques, all folded in). |
| `HANDOFF_v13_review.md` | Original 28-finding review (2 blocking, 9 major, 12 minor, 5 nits). |
| `v13_review_findings.json` | Structured findings with locations + recomputed statistics. |

## Why base off `chore/publication-cleanup`
The RepoBench keystone harness (`tools/_repobench_clamp_run.py`, `src/rune/bench/repobench.py`) lives on that branch, not `main`. Basing here means a single checkout on the GPU instance gives both the plan and the harness for Phase 1.

## What to do after pulling (GPU instance)
Phase 1 is the load-bearing work, ~2 GPU-hr on frozen c3 (no retraining):
1. **`a2_tail`** — place the identical oracle conditioning string (variant `use`, ~124 tok) at the prompt tail within W=768; report vs floor at matched cursor-code lengths.
2. **`a2_tail_filler`** — 124-tok neutral filler control to isolate the pointer's marginal effect.
3. **swap/mutation control** — *build first* (not yet in harness; design-spec §8 prescribes "port PR #57 §8"), then run on the keystone subset.

Pre-registered gates for each are in `publication_task_plan.md` §Phase 1. If only one run happens, it is `a2_tail` + filler.

## Key correction baked in
The adapter's episodic conditioning is **oracle-supplied** (`row.gold_snippet_index`, arm `episodic_use`). The keystone is honestly a **channel comparison under oracle conditioning**, not a retrieval demonstration — Setup must state this. The retracted "adapter internalizes retrieval" framing is removed.

## Two items awaiting the advisor (off critical path)
1. **ReasonCACHE Option A vs B** (finding M4) — recommend prose theorem-engagement (B) this round, KV/prefix arm (A) committed for next.
2. **Subagent model "Fable-5"** — not a resolvable roster id; needs a mapping or enablement. Nothing here delegates, so not blocking.
