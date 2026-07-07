# C2.1 prep — fresh-pool re-estimate of +0.105 (ready to launch)

**Date:** 2026-07-07. **Task:** `publication_task_plan.md` C2.1, gated FEASIBLE by C0.1
(`c01_corpus_lookup.md`: disjoint_count = 120 ≥ ~50). This note is the *prep* deliverable:
the run itself is GPU work (~1–2 GPU-hr) and has **not** been executed (GPU campaign in
progress). Everything CPU-verifiable has been dry-run.

## 1. What the original +0.105 estimate was (methodology of record)

Evidence: `docs/issue52-experimentation-log.md` §3.2 E-phase1; recovered
`docs/issue52-phase1-results-2026-06-04.md` (git `f827a13`, "Generalization significance"
section); MLflow exp 45 `issue52-phase1` run `fe72f9ddd69c4f7b8bd86b6b12372d47` (the c3
training run — `corpus_path=benchmarks/mbpp_recall_train.jsonl`, sha `e60f0dd8…`; the
probe numbers themselves were **post-hoc, never logged to MLflow** — see
`docs/mlflow-experiment-inventory-2026-06-09.md` row J).

- **Instrument:** `tools/_specificity_probe.py` (the "frozen E1 probe"), invoked by
  `tools/_phase1_orchestrate.py` per checkpoint as
  `probe --ckpt <ckpt> --corpus benchmarks/mbpp_recall_heldout.jsonl --out <dump>`.
  Per task it assembles the matched adapter from
  `render_training_format_trajectory(description)` (exact engine conditioning surface),
  hot-applies it functionally at `effective_scaling` = checkpoint `lora_alpha` (45.2548),
  and teacher-forces the committed `reference` solution (first 96 answer tokens) under
  bf16 + flash-attention-2, scoring mean gold log-prob per span
  (`scoring_core.mean_gold_logprob`, float32 log-softmax, float64 accumulation).
- **Matched-log-prob definition (the +0.105):** Δlp_matched = per-task
  `lp_m` (matched-adapter mean gold logprob, **absent** prompt regime — task description
  NOT in prompt — **body** span = answer tokens after the `def <entry>(…)` line),
  **c3 minus warm-start**, averaged across tasks. Numerically identical to the reported
  "Δ m-zero" (+0.635 vs +0.530): the shared base-model `lp_z` term cancels in the pairing.
- **Across-task sign test:** #positive per-task deltas, exact two-sided binomial at
  p=0.5 → 17/24 → **p = 0.064** (reproduced: 0.063915).
- **CI:** bootstrap 95% CI on the mean of per-task deltas → [+0.033, +0.182].
- **Checkpoints:** c3 = `c3_t07_lp2_lg1.pt` (exp-45 run `fe72f9…` artifact
  `checkpoints/checkpoint_step48.pt`), warm-start = Sakana doc-to-lora
  `qwen_4b_d2l/checkpoint-20000/pytorch_model.bin`.

**Why the re-estimate is needed:** heldout24 is the *selection set* — c3 was picked from
the c1–c4 grid on those same 24 tasks (winner's curse; `HANDOFF_v13_review.md` M1).

## 2. What was restored / built (this prep)

| Path | Provenance | Status |
|---|---|---|
| `tools/_specificity_probe.py` | restored **verbatim** from git `205fa3d` (orphan archive tip; deleted from main at `40bfcf3`/`51f9afe`, not part of `1cd5d60`) | ruff-clean; `--help` and all runtime imports dry-run OK against current `src/` |
| `tools/scoring_core.py` | restored verbatim from `1cd5d60~1` (unchanged since `cef27c7`, 2026-06-02 — the exact math the June runs used) | ruff-clean |
| `tools/_c21_fresh_pool_run.py` | **new** C2.1 runner (pool builder + probe orchestration + stats + MLflow) | ruff-clean; CPU paths fully dry-run |
| `benchmarks/mbpp_recall_fresh120.jsonl` | derived deterministically: raw `mbpp_recall_train_160.jsonl` lines (byte-identical, original order) whose `task_id` ∉ `mbpp_recall_train.jsonl`; inputs sha-pinned to `docs/publication/hashes.txt` | **n=120**, sha256 `6142c54b5c3560320bb0fee7661c8bf49f7f0f864297a82eed653512ff887507`; zero overlap with train40 / heldout24 / crossover10; ids == the C0.1 published 120-task list; all 120 references probe-scoreable (def-marker present) |
| `third_party/doc-to-lora/…/checkpoint-20000/pytorch_model.bin` | re-downloaded from HF `SakanaAI/doc-to-lora` (third_party/ had been wiped; gitignored) | sha256 `6438b46c828dd3b5f88f21add0f7f5cacc7994d47bf15eda266786a506044591` — matches the exp-56 `d2l_provenance.json` pin and the experiment-log §6.1 warm-start prefix |
| `/tmp/phase1/ckpt/c3_t07_lp2_lg1.pt` | already restored by C0.2 from S3 mlflow artifacts | sha256 `53e24af243a38dfbfad82f7293635bfc592922dd2058fefbbfa10714b5457a3f` verified — matches `hashes.txt` and the checkpoint-of-record |

`.gitignore` line `tools/_specificity_probe.py` (June scratch-tool hygiene, `40bfcf3`)
was commented out so the restored instrument is committable; `scoring_core.py` and the
new runner were never ignored. Working-tree state for the orchestrator to commit:
`benchmarks/mbpp_recall_fresh120.jsonl`, `docs/publication/c21_prep.md`,
`tools/_c21_fresh_pool_run.py`, `tools/_specificity_probe.py`, `tools/scoring_core.py`,
`.gitignore` (the pre-existing `.devcontainer/post-create.sh` modification is not from
this task).

The runner verifies both checkpoint sha256s **before** any forward pass, evaluates the
two frozen checkpoints only (no training step, no trajectory generation, no corpus
building — the probe is read-only over committed rows), and hard-fails if any pool task
is missing from either dump (no silent exclusions; `--allow-partial` to override
explicitly).

**API compatibility re-checked against current `src/`:** `effective_scaling` is still
Sakana-parity `lora_alpha` (not `alpha/r`) — same semantics as the June campaign, so the
post-`bce5f2fe` scaling fix does not break comparability; `_functional_lora`,
`load_hypernetwork` (mmap load — RAM-safe), `extract_activations_with_model`,
`render_training_format_trajectory`, `combine_lora` all present with unchanged signatures.

## 3. CPU dry-runs performed (all pass)

1. `--build-pool`: pool derived, count/overlap/id-list/byte-identity checks pass.
2. `--help` for both runner and probe; probe's deferred runtime imports resolve CPU-side.
3. `sign_test_p(17, 24) = 0.063915` — reproduces the documented p=0.064 exactly.
4. `bootstrap_ci` deterministic at fixed seed.
5. `--stats-only` end-to-end on synthetic 120-task dumps, logging params(14) /
   metrics(45) / artifacts(3) to a scratch MLflow experiment (verified via REST, then
   deleted). Incomplete-dump hard-fail path exercised.
6. No GPU touched; no model forward executed; nothing committed.

## 4. GPU launch (later, when the GPU is free)

```bash
cd /workspaces/rune-gpu && mkdir -p /tmp/c21 && free -g && nvidia-smi && \
nohup env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  uv run --no-sync python tools/_c21_fresh_pool_run.py \
    --experiment issue52-phase1 --workdir /tmp/c21 \
  > /tmp/c21/launch.log 2>&1 &
```

- Sequence: verify both ckpt sha256s → probe c3 on fresh120 (one model load, GPU-only,
  `offload_base` not involved) → probe warm-start (second load; sequential subprocesses so
  GPU memory is fully released between arms) → paired stats → one MLflow run
  `c21-fresh120-reestimate` in **`issue52-phase1`** (exp 45) with dumps + summary as
  artifacts. Logs: `/tmp/c21/probe_{c3,warm}.log`, results `/tmp/c21/c21_summary.json`.
- **Expected runtime:** ~1–2 GPU-hr (120 tasks × [1 hypernet assembly + 3 forwards × 2
  regimes] per arm, × 2 arms, + 2 base-model loads; 5× the heldout24 probe volume).
  Fits the plan's C2.1 budget.
- RAM note: base loads straight to GPU (`device_map={"": "cuda"}`), hypernet ckpt loads
  via mmap — the known-safe Phase-1 loading pattern for this ~15GB-RAM box.
- If `/tmp/phase1/ckpt/c3_t07_lp2_lg1.pt` is lost to a VM restart, re-restore:
  `aws s3 cp s3://elixirtrials-949678234935-us-east-1-artifacts/mlflow/artifacts/45/fe72f9ddd69c4f7b8bd86b6b12372d47/artifacts/checkpoints/checkpoint_step48.pt /tmp/phase1/ckpt/c3_t07_lp2_lg1.pt`
  (then the runner's sha check re-verifies).

## 5. Pre-registered analysis & gate (unchanged from the plan)

- **Headline statistic:** mean Δlp_matched (absent/body, c3 − warm) over the 120-task
  fresh pool; **sign test:** exact two-sided binomial on #positive (zeros dropped),
  n=120; **CI:** percentile bootstrap of the mean, **10,000 resamples, seed 0**
  (pre-registered here).
- **Gate:** sign-test **p < 0.05** → hand the de-biased number to the article side,
  strip the selection-bias caveat from §5.2/abstract. **p ≥ 0.05** → signal the
  prose-downgrade path (A-OBJ): report as suggestive-on-a-selected-configuration.
  **Do NOT generate new trajectories to chase p<0.05** (plan rule; none are needed —
  the pool is fully committed data).
- Secondary (logged, not gating): Δ m-zero cross-check (must ≈ headline), absent/sig
  (signature-retention context), present/body, absent/full.

## 6. Deviations from the original estimate (unavoidable, documented)

1. **Bootstrap resample count/seed:** the June CI was computed in session-scratchpad
   state that was not preserved; resamples=10000/seed=0 are pre-registered here. The
   point estimate and sign test — the gating statistic — are exactly method-identical.
2. **MLflow:** the original probe numbers were post-hoc and never logged; the
   re-estimate gets a first-class run in exp `issue52-phase1` (tagged `task=C2.1`,
   `original_run_id=fe72f9…`). This is an improvement, not a methodology change.
3. **Derangement partner permutation** (mismatch arm only): defined over the 120-row
   pool order instead of the 24-row heldout order — inherent to changing pools; the
   headline Δlp_matched and the m-zero cross-check do **not** depend on the mismatch arm.
4. **Warm-start checkpoint** restored from HF hub rather than the original local copy —
   byte-identical (sha256 match against the logged provenance record).
