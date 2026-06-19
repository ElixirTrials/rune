# Rune LCB-v6 durable benchmark — findings (2026-06-19)

Engine commit: `db48504` (HEAD). Checkpoint: c3 (`c3_t07_lp2_lg1.pt`, sha256 `53e24af2…`).
All runs durable in MLflow experiment **`issue52-lcb-durable`** (each: official metric +
config + engine_commit + checkpoint_sha256 + gens/grade/per-qid artifacts).

## 1. Setup (pinned, pre-registered)
- **Benchmark:** LiveCodeBench v6, functional-only (call-based, `starter_code` present),
  graded by the **official LCB harness** (`lcb_runner.codegen_metrics`, same grader for every arm).
- **Task set:** `test6.jsonl` (sha256 `bb4c364f…`, exact upstream match). Functional-49 =
  window [2025-02-01, 2025-05-01); extension = +14 functional from [2025-01-01, 2025-02-01)
  → pooled **N=63** (all 2025, same release; no contamination from earlier years).
- **Params:** timeout 6s, budget `--max-iters 24`, judge OFF, seed 0, temperature 0.3.
- **Arms:** base = single-shot zero-shot (no engine loop); c3 = escalate, adapter@1.0;
  scale0 = escalate, adapter-off (attribution).

## 2. Headline result (pooled N=63)
| Arm | pass@1 |
|---|---|
| base (single-shot) | **12/63 (19.0%)** |
| c3 (escalate, adapter@1.0) | **16/63 (25.4%)** |

**Uplift: +4 tasks (+6.3 pts), and c3 is a STRICT SUPERSET of base — 0 regressions.**
c3 solves all 12 base tasks plus `3739, 3793, 3809, 3832`.
(Functional-49 subset alone: base 9/49, c3 12/49, +3, strict superset — consistent.)

Per-arm breakdown (functional-49): base pass=9, tle=7, runtime=2, wrong=31; c3 pass=12,
tle=4, runtime=1, wrong=32.

## 3. Attribution (scaffold vs adapter) — pooled N=63
| Arm | pass@1 |
|---|---|
| base | 12/63 (19.0%) |
| scale0 (escalate, adapter-off) | 14/63 (22.2%) |
| c3 (escalate, adapter@1.0) | 16/63 (25.4%) |

- base→scale0 (iterative scaffold): +3 `{3753,3764,3832}` / −1 `{3785}` → **net +2**.
- scale0→c3 (adapter): +4 `{3739,3785,3793,3809}` / −2 `{3753,3764}` → **net +2 (with churn)**.

**Interpretation:** the +4 c3-over-base uplift splits roughly evenly — scaffold +2, adapter
+2 — but the adapter contribution carries task churn (gains 4, loses 2), so it is suggestive
rather than clean at this N. The adapter measurably changes generations (logit Δ=10.97 at
scaling 1.0, §4) but does not yet reliably convert them to additional correct solutions.
(functional-49 subset: base 9 / scale0 11 / c3 12 — scaffold +2, adapter +1; consistent.)

## 4. Adapter-off validity
Empirical probe (logits on a sample prompt):
- base vs adapter scaling=0: **max|Δ| = 0.0** → scale0 is exactly the raw base (adapter truly off).
- base vs adapter scaling=1.0: **max|Δ| = 10.97** → the adapter is genuinely active at 1.0.
So scale0 is a valid adapter-off baseline and the c3-vs-scale0 attribution is sound.

## 5. Statistics (context; omitted from the article per decision)
Paired comparison ⇒ McNemar exact. N=63: discordant 4 gained / 0 regressed → two-sided
p=0.125 (underpowered by design). Power analysis: 80% power for this effect rate needs
N≈129 (the recent-clean LCB-v6 functional pool is ~70). Headline is the **uplift + strict
superset**, not a significance claim.

## 6. Infrastructure / reproducibility (engine hardening in this PR)
- `wrapper.py`: `device_map` streaming model load — avoids the CPU-RAM thrash a plain
  `.to(device)` causes on a ~15GB host (load-time `folio_wait` hang). Numerically identical.
- `sandbox/executor.py`: per-process `RLIMIT_AS` cap (4GB) on untrusted code — a
  runaway/large-input solution raises MemoryError instead of OOM-killing the host
  (root cause of two VM crashes during grading).
- Grader (`tools/_lcb_grade.py`): same 4GB per-candidate cap + 2-way parallelism.
- Recovery: c3's 49-task generation survived both crashes and was re-graded from saved
  gens (no 7h re-run).

## 7. Pending (in progress)
- scale0 extension to N=63 (pooled attribution).
- **HumanEval+ (EvalPlus, 164 tasks)** corroboration — base vs c3 uplift on a 2nd
  function-level benchmark. `tools/_he_run.py`, graded by the hardened rune sandbox.
- **Mutated-spec pointer-vs-content control** on the c3 spec-absent solves (reference_a,
  scaling 0.627; c3 spec-absent = 14/24, NOT the 8/24 adapter-off floor). Decision rule:
  output tracks mutated spec = content; reproduces original memorized solution = pointer.
