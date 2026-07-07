# C0.1 — Corpus-trajectory lookup (Phase-0 / 2a-feasibility)

**Date:** 2026-07-07. **Task:** `publication_task_plan.md` row C0.1 (= `remediation_plan_FINAL.md`
Phase-0 "2a-feasibility"). CPU-only; no GPU touched; nothing committed.

## Decision

**disjoint_count = 120 (conservative, committed-corpus interpretation) → C2.1 is FEASIBLE**
(pre-registered rule: ≥ ~50 disjoint MBPP tasks with gold trajectories). Under every
defensible interpretation of the rule except one over-strict variant (see §5), the count
clears the bar by 2.4×.

## 1. What a "gold trajectory" is in this corpus

The +0.105 objective corpora are **not mined agent sessions**. Each row is a deterministic
render of the MBPP sanitized gold solution:
`context = render_training_format_trajectory(task=description)` (byte-identical scaffold),
`answer = reference = MBPP canonical code`, `test_code = the real assert suite`. Builders
(deleted in publication-cleanup, recovered from git at commit `db48504`):

- `tools/build_heldout_mbpp_recall_corpus.py` — pool = `google-research-datasets/mbpp`
  config `sanitized` split `test`, filtered by `_row` (has `test_list`, extractable
  `entry_point`, reference defines `def <entry>(` with a non-empty body), minus the 10
  cross-over pilot ids {11,12,14,16,17,18,19,20,56,57}; sorted by task number; first 40 →
  train, next 24 → held-out eval.
- `tools/build_scaling_train_corpora.py` — same pool minus eval ids; first N ∈ {80,160}
  (nested supersets of the 40).

**Verified reconstruction:** I re-fetched the sanitized test split (257 rows) via the HF
datasets-server API and re-applied the exact `_row` filter: 240 usable rows, 230 after
excluding the 10 cross-over ids. Sorting by task number and slicing reproduces the three
committed files **exactly** (set-identical to `mbpp_recall_train.jsonl`,
`mbpp_recall_heldout.jsonl`, and `mbpp_recall_train_160.jsonl`). So "task has a gold
trajectory" is fully characterized: it is a usable sanitized-test row, and the render is a
pure deterministic function of the MBPP record.

## 2. The sets

| Set | Size | Contents / provenance |
|---|---|---|
| Committed gold-trajectory tasks | **194** | `benchmarks/mbpp_recall_train_160.jsonl` (160; ⊇ 80 ⊇ 40, verified nested, sha256 `5711834e…`/`4dbf049b…`/`e60f0dd8…`) ∪ `benchmarks/mbpp_recall_heldout.jsonl` (24, sha256 `cae274bf…`) ∪ `configs/issue52_mbpp_body_crossover.jsonl` (10 pilot tasks). All three pairwise-disjointness checks pass. |
| Objective-grid **selection set** | **64** | train40 (c3's training corpus; MLflow exp 45 `issue52-phase1` run `fe72f9ddd69c4f7b8bd86b6b12372d47` param `corpus_path=…/mbpp_recall_train.jsonl`) ∪ heldout24 (the 24 tasks on which the c1–c4 grid was scored and c3 selected; Δlp_matched +0.105, CI [+0.033,+0.182], sign test 17/24 p=0.064 — `docs/issue52-experimentation-log.md` E-phase1). Both halves are consumed by the selection event. |
| **N=60 keystone set** | **60** | RepoBench v1.1 Python `cross_file_first` rows (`tianyang/repobench_python_v1.1`), 8k×30 + 32k×30, offset 100, W=768 (`tools/_repobench_clamp_run.py`; `docs/issue52-repobench-clamp-findings-2026-06-22.md`). **Not MBPP** — intersection with any MBPP task set is empty by construction. |

File hashes above match the artifact index in `docs/issue52-experimentation-log.md` §6.2
(train sha `e60f0dd8…`, heldout `cae274bf…`), tying the on-disk files to the logged runs.

## 3. The disjoint count

```
{committed gold-trajectory tasks}  −  (selection set ∪ keystone set)
= 194 − 64 (keystone contributes 0: RepoBench, not MBPP)
= 130
− 10 cross-over pilot tasks (pilot-2 trained-on-test; excluded for hygiene)
= 120
```

**disjoint_count = 120** — the tasks of `mbpp_recall_train_160.jsonl` minus
`mbpp_recall_train.jsonl`. All 120 rows are already committed in the repo (no new
trajectory generation needed, satisfying C2.1's "do NOT generate new trajectories" rule).

MBPP task numbers (mbpp/N) of the 120-task pool:
143, 145, 160, 161, 162, 164, 165, 166, 167, 168, 170, 171, 172, 222, 223, 224, 226, 227,
228, 229, 230, 234, 235, 237, 238, 239, 240, 242, 244, 245, 247, 249, 250, 251, 252, 253,
255, 256, 257, 259, 260, 261, 262, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274,
277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 290, 291, 292, 293, 294, 295, 296,
297, 299, 301, 304, 305, 306, 307, 308, 309, 310, 311, 388, 389, 390, 391, 393, 394, 395,
396, 397, 398, 399, 400, 401, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415,
417, 418, 419, 420, 421, 422, 424, 425, 426, 427, 428, 429.

(For reference — selection set: train40 = mbpp/{58, 59, 61–72, 74, 75, 77, 79, 80, 83, 84,
86–97, 99–105}; heldout24 = mbpp/{106, 108, 109, 113, 115–120, 123, 125–133, 135, 138,
141, 142}.)

## 4. Contamination status of the 120-task pool

The 120 tasks were **never used to train or select c3**: c3 was trained on train40 only
(MLflow run param), and the grid selection was scored on heldout24 only. Their only prior
use is as *training data for the separate n80/n160 scaling checkpoints* (E-G2-scale, MLflow
exp 47 `issue52-goal2-scaling`, runs `9812c7f2eb1349a2` [train_160] and `39a6f211ddfa4cca`
[train_80]) — checkpoints that play no role in the +0.105 claim. As an evaluation pool for
**frozen c3**, the 120 tasks are statistically clean: no selection event ever touched them.

## 5. Sensitivity of the count to interpretation

| Interpretation | Count | Clears ≥ ~50? |
|---|---|---|
| Committed trajectory tasks ∖ (selection ∪ keystone), incl. cross-over 10 | 130 | yes |
| **Same, excl. cross-over pilot tasks (reported headline)** | **120** | **yes** |
| Constructible pool (all 230 usable non-crossover sanitized-test rows) ∖ selection | 166 | yes |
| Over-strict: tasks never touched by *any* training of *any* checkpoint (166 − 120) | 46 | **no (marginal)** |

The over-strict variant is not what C2.1 requires — it would exclude tasks solely because
a *different* checkpoint (n80/n160) trained on them, which cannot bias an estimate made
with frozen c3 — but it is reported for transparency. If a reviewer insisted on it, the
46-task pool alone misses the ~50 bar; the correct response is the argument above, not new
trajectory generation.

## 6. Recommended C2.1 protocol (handed to the run task)

Re-estimate frozen-c3 matched-log-prob (vs warm-start) on the 120-task pool =
`mbpp_recall_train_160.jsonl` rows whose `task_id` ∉ `mbpp_recall_train.jsonl`; recompute
the across-task sign test at n=120 (binomial, as in E-phase1). Log to MLflow
`issue52-phase1`. Everything needed is already committed; no corpus building, no mining.

## 7. Evidence trail

- `benchmarks/mbpp_recall_train.jsonl` (40 rows, sha256 `e60f0dd85fad5114…`)
- `benchmarks/mbpp_recall_heldout.jsonl` (24 rows, sha256 `cae274bf1aed31c8…`)
- `benchmarks/mbpp_recall_train_80.jsonl` (80 rows, sha256 `4dbf049b59979ddf…`)
- `benchmarks/mbpp_recall_train_160.jsonl` (160 rows, sha256 `5711834e1ae90ffa…`)
- `benchmarks/mbpp_heldout_tasks.json` (same 24 ids as the heldout jsonl; verified)
- `configs/issue52_mbpp_body_crossover.jsonl` (10 pilot rows)
- Builders at git `db48504`: `tools/build_heldout_mbpp_recall_corpus.py`,
  `tools/build_scaling_train_corpora.py`
- MLflow (http://localhost:5000): exp 45 `issue52-phase1` run
  `fe72f9ddd69c4f7b8bd86b6b12372d47` (c3; `corpus_path` param); exp 47
  `issue52-goal2-scaling` runs `9812c7f2eb1349a2`, `39a6f211ddfa4cca` (corpus_path params
  name train_160/train_80); exp 46 `corpus-registry` run `ea4f3c43af3a4258…` (external
  codereview corpus only — contains no MBPP manifest)
- HF datasets-server fetch of `google-research-datasets/mbpp` / `sanitized` / `test`
  (257 rows; snapshots in session scratchpad `mbpp_test_{0,100,200}.json`)
- `docs/issue52-experimentation-log.md` (E-phase1, E-G2-scale, artifact index §6)
- `docs/issue52-repobench-clamp-findings-2026-06-22.md` +
  `tools/_repobench_clamp_run.py` (keystone set identity: RepoBench, not MBPP)

## 8. Caveats

1. The `issue52-repobench-clamp` / `issue52-repobench-template-hpo` experiments are **not
   present on this MLflow server** (localhost:5000 holds exps 0–77; the clamp runs are
   documented in `docs/issue52-repobench-clamp-findings-2026-06-22.md` and are planned to
   be (re-)logged by C1). The keystone set's identity was therefore established from the
   harness source + findings doc, not from a live MLflow run. This does not affect the
   disjoint count: the keystone set is RepoBench rows and cannot intersect MBPP ids.
2. The adversary review's "224 used" figure appears to be the (mis-remembered) sanitized
   *test-split size*, which is actually 257 rows (427 sanitized total; 240 usable after
   the builder's filters). The binding numbers used here were recomputed from source.
3. Exp 46 `corpus-registry` holds only the external_codereview corpus registration; there
   is no MBPP corpus manifest in MLflow. The committed benchmark files + their sha256
   match to the experiment log's artifact index serve as the manifest.
4. Environment note (unrelated to the result): `.venv` was found already broken (dangling
   interpreter symlink); a `uv run --no-sync` invocation recreated it empty. Before the
   next GPU run, re-provision with `uv sync --extra gpu` (never plain `uv sync`).
