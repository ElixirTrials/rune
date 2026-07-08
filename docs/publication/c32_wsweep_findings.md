# C3.2 recovery-vs-budget W-sweep — findings (2026-07-08)

Plan row C3.2 (`docs/publication/publication_task_plan.md`): re-run the N=60 keystone row
set at W ∈ {256, 512, 1536} and combine with the existing W=768 C1 campaign leg to produce
the recovery-vs-budget curve. Gate as pre-registered: *"Accept if Phase-1 GPU budget allows;
else decline explicitly."* **Realized: accepted and run** (~1.5 GPU-hr, within budget).
All eight arms of the C1 campaign (`docs/publication/c1_keystone_findings.md` §1) at every W.

**Provenance (pinned).**
- Harness: `tools/_repobench_clamp_run.py`. Sweep legs (256/512/1536) at engine commit
  `c4562db`, which includes the **tail-overhead guard**: rows whose header + conditioning +
  cursor overhead alone exceeds W are recorded as skipped (`tail_overhead_tokens>W`) and
  counted in `a2_tail_inapplicable`, instead of silently overflowing the budget. The W=768
  point is the C1 campaign run (engine commit `ee1a133`, **pre-guard** — see Limitations).
- Rows: RepoBench v1.1 Python (`tianyang/repobench_python_v1.1`), split `cross_file_first`,
  8k+32k × 30, offset=100, seed 0, temperature 0.0, max_new 48. N=60 — verified identical
  task_id set *and order* across all four legs; row-level fields (gold, ctx_tokens, …)
  identical across legs.
- Checkpoint: c3 (`c3_t07_lp2_lg1.pt`), sha256 `53e24af243a38dfbfad82f7293635bfc592922dd2058fefbbfa10714b5457a3f`.
- MLflow: experiment `issue52-repobench-clamp` (exp id 79), `MLFLOW_TRACKING_URI=http://localhost:5000`:

| W | run name | run id | trace (MLflow artifact = durable copy) | trace sha256 |
|---|---|---|---|---|
| 256 | `clamp-use-W256-8k_32k-n60-off100-seed0` | `ab4d331287774435abb0653967469551` | `rb_clamp_w256.json` | `ea22f847021d3326aaa3696cbd1a167d07c536e180732c2b22804c224158a50c` |
| 512 | `clamp-use-W512-8k_32k-n60-off100-seed0` | `3ba5333785ee4acfa19cc526cf00ca91` | `rb_clamp_w512.json` | `796c021291c2d685495d7db29f342a7bf3540458ada2aa0a15cc1dfeb3723755` |
| 768 | `clamp-use-W768-8k_32k-n60-off100-seed0` (C1) | `f37374906c5f4f5c972b8e7b8127089a` | `rb_clamp_c1_full.json` | `d0d6d6837cdaeafc8121ecfc6af0ee134e38608fca2fb10fec0662670a870a3a` |
| 1536 | `clamp-use-W1536-8k_32k-n60-off100-seed0` | `d3aae62a5c514acb94b1b2a6381d85a9` | `rb_clamp_w1536.json` | `287d0c96ae197121a7496022acc2dfae0fc4d060d407e67cc0c042442c4718c3` |

**Verification basis.** Every number below is from an independent adversarial re-verification
(session-scratchpad `verify_wsweep.py`, stdlib only, no harness imports): every prediction
string re-scored (recovered/EM/edit-similarity), Wilson CIs, five McNemar pairs, attributable
fraction, beyond-prompt counts, and inapplicable counts reimplemented from scratch; MLflow
params/metrics cross-checked via REST. **Result: 0 invariant violations, 0 discrepancies vs
the harness printouts.** Swap donors re-derived independently from row order and matched on
all rows at all legs.

## Limitations / accounting decisions (read first)

1. **No verifier-reported invariant failures.** All sweep-leg invariants pass: 60 rows
   (30/30 per level), unique task_ids, no error keys, scored tail rows within budget
   (`prompt_tokens ≤ W`, `prefix_tokens > 0`, filler token-matched to conditioning on every
   row), conditioning W-invariant for the adapter arms, swap donors admissible everywhere.
2. **The W=768 leg is the pre-guard C1 run.** Its `a2_tail`/`a2_tail_filler` include row
   `cross_file_first/6125`, whose conditioning (~2020 tokens) alone exceeds the budget
   (C1 findings, Limitations §1). A **guard-consistent W=768 variant** was computed by
   treating 6125's tail arms as inapplicable — exactly as the guard would: a2_tail 50/59,
   filler 5/59; every other arm and every McNemar unchanged (6125 is concordant-false on
   both arms of the episodic-vs-a2_tail pair). Both variants are reported below.
3. **Tail-guard skips at the sweep legs** (counted in `a2_tail_inapplicable`, reason string
   `tail_overhead_tokens>W`): W=256 skips 4 rows {2106 (~452 overhead), 6117 (~380),
   6125 (~2028), 6126 (~686)}; W=512 skips 2 {6125, 6126}; W=1536 skips 1 {6125}. Skip sets
   are nested (monotone in W) and consistent with per-row overhead. **Row 6125 is
   tail-inapplicable at every tested W** — its conditioning alone exceeds even 1536.
4. **Episodic 31/60 at 256/512/768 is a coincidence of counts, not row-identical:** verdict
   agreement 52/60 (256v512), 54/60 (256v768), 52/60 (512v768); exact pred-string identity
   lower still. Legitimate — the episodic arm's prompt *is* the floor prompt
   clamp(prefix, W), which changes with W, while its adapter conditioning is W-invariant
   (verified identical `cond_tokens` across all four legs).
5. **Gate framing correction.** The plan named this "the advantage-grows-as-the-budget-
   tightens curve." Realized: that shape holds for the **tail-channel advantage over the
   adapter** (gap grows as W shrinks), but **not** for episodic-over-floor, which is flat
   in W (§4). No spin: a2_tail is flat-high at every budget, and the tail-vs-adapter gap
   closes to non-significance at W=1536.

## 1. Cross-W recovery table (all 8 arms)

k/n, rate, Wilson 95% CI. Tail-arm denominators exclude guard-skipped (inapplicable) rows.

| arm | W=256 | W=512 | W=768 (C1 as-is) | W=768 (guard-consistent) | W=1536 |
|---|---|---|---|---|---|
| floor | 6/60 0.100 [0.047, 0.202] | 8/60 0.133 [0.069, 0.242] | 9/60 0.150 [0.081, 0.261] | 9/60 0.150 [0.081, 0.261] | 17/60 0.283 [0.185, 0.408] |
| a2_clamp | 6/60 0.100 [0.047, 0.202] | 11/60 0.183 [0.106, 0.299] | 11/60 0.183 [0.106, 0.299] | 11/60 0.183 [0.106, 0.299] | 21/60 0.350 [0.242, 0.476] |
| a2_full | 17/30 0.567 [0.392, 0.726] | 17/30 0.567 | 17/30 0.567 | 17/30 0.567 | 17/30 0.567 |
| episodic_use | 31/60 0.517 [0.393, 0.638] | 31/60 0.517 | 31/60 0.517 | 31/60 0.517 | 42/60 0.700 [0.575, 0.801] |
| dump_gf | 4/60 0.067 [0.026, 0.159] | 9/60 0.150 [0.081, 0.261] | 11/60 0.183 [0.106, 0.299] | 11/60 0.183 [0.106, 0.299] | 20/60 0.333 [0.227, 0.459] |
| **a2_tail** | **49/56 0.875 [0.764, 0.938]** | **50/58 0.862 [0.751, 0.928]** | **50/60 0.833 [0.720, 0.907]** | **50/59 0.847 [0.735, 0.918]** | **49/59 0.831 [0.715, 0.905]** |
| a2_tail_filler | 2/56 0.036 [0.010, 0.121] | 4/58 0.069 [0.027, 0.164] | 5/60 0.083 [0.036, 0.181] | 5/59 0.085 [0.037, 0.184] | 10/59 0.169 [0.095, 0.285] |
| swap | 2/60 0.033 [0.009, 0.114] | 3/60 0.050 [0.017, 0.137] | 6/60 0.100 [0.047, 0.202] | 6/60 0.100 [0.047, 0.202] | 13/60 0.217 [0.131, 0.336] |

`a2_full` is skipped at every W iff ctx_tokens > 12,000 — exactly the 30 32k rows,
identical pattern on all legs (the skip depends on context size, not W).

## 2. a2_full invariance check (determinism signal)

The 30 scored `a2_full` rows have **identical recovered verdicts AND identical raw
prediction strings across all four legs** (0 divergences) — 17/30 everywhere. The a2_full
forward does not depend on W, so this is a within-sweep bit-exactness check on the whole
stack (loader, adapter machinery untouched for this arm, greedy decode), analogous to the
270/270 June replication in the C1 findings §5. It passed.

## 3. Channel-gap curve — the figure's data

Primary pair: `episodic_use` (pointer in adapter) vs `a2_tail` (identical pointer in-prompt
at tail). Discordants are (episodic-only, a2_tail-only); exact McNemar, two-sided.

| W | n (pair) | gap (a2_tail − episodic) | discordants | McNemar p | episodic − floor gap | beyond-prompt (episodic) |
|---|---|---|---|---|---|---|
| 256 | 56 | +0.358 | (3, 21) | 2.77e-04 | 0.417 | 25 |
| 512 | 58 | +0.345 | (2, 21) | 6.60e-05 | 0.383 | 20 |
| 768 as-is | 60 | +0.317 | (1, 20) | 2.10e-05 | 0.367 | 22 |
| 768 guard-consistent | 59 | +0.331 | (1, 20) | 2.10e-05 | 0.367 | 22 |
| 1536 | 59 | +0.131 | (3, 10) | 9.23e-02 | 0.417 | 24 |

Floor-vs-episodic McNemar is p ≤ 3e-06 at every W; floor beats episodic on at most 2 rows
at any leg. "Beyond-prompt" = rows episodic recovers while both floor and a2_clamp fail
(gold identifier not recoverable from the clamped prompt) — flat in W: 25/20/22/24.

**Which N accounting the figure should use: guard-consistent at all four points.** Plot
tail-arm rates over the applicable-row denominators (n = 56/58/59/59), i.e. the W=768 point
as **a2_tail 50/59 = 0.847, gap +0.331** — the same rule the guarded harness applies at the
sweep legs, with row 6125 inapplicable everywhere. Caption must state: tail-inapplicable
rows (conditioning + fixed overhead alone > W) are excluded from tail-arm denominators
only; all other arms use n=60 (a2_full n=30, 8k only). The as-is 768 numbers (50/60, 0.833)
remain the numbers of record in `c1_keystone_findings.md`; the two variants differ only in
the 6125 row's denominator treatment and share every McNemar verdict.

## 4. What the curve shows (no spin)

1. **a2_tail is flat-high across a 6× budget range:** 0.875 → 0.862 → 0.847 → 0.831
   (guard-consistent), all CIs heavily overlapping. Once the ~124-token pointer fits, the
   tail channel's performance is essentially budget-insensitive.
2. **The tail-vs-adapter gap closes monotonically as W grows:** +0.358 → +0.345 → +0.331 →
   +0.131; discordants 21v3 → 21v2 → 20v1 → 10v3; McNemar significant at every W ≤ 768
   (p ≤ 2.8e-04) and **not significant at W=1536 (p=0.092)**. What this licenses: the
   in-prompt tail channel's advantage over the adapter is a *tight-budget* phenomenon.
   What it does **not** license: (a) claiming adapter parity at W=1536 — non-significance
   is not equivalence, and the discordants still favor a2_tail 10v3; (b) attributing the
   closure to the adapter improving — episodic's rise at 1536 tracks the longer prompt
   prefix, not the adapter (see 3).
3. **Episodic's lift over floor is stable, not budget-dependent:** gap 0.417 / 0.383 /
   0.367 / 0.417 across W, p ≤ 3e-06 everywhere. The episodic 0.517 → 0.700 jump at W=1536
   is **not** a conditioning effect — the adapter conditioning is W-invariant (verified);
   only the prompt (= floor prompt) grows — and it mirrors floor's own rise
   (0.150 → 0.283 as more current-file code fits). Adapter pointer and longer prefix
   **compose additively** (lift stays ~0.42) rather than becoming redundant.
4. **The swap confound control holds at every W:** swap ≤ floor in rate at all four legs
   (2, 3, 6, 13 vs floor 6, 8, 9, 17); McNemar swap-vs-floor discordants 0v4 / 1v6 / 3v6 /
   5v9, p = 0.125 / 0.125 / 0.508 / 0.424 — all n.s.; swap never significantly beats *or*
   trails floor. Attributable fraction (e−s)/(e−f) ≥ 1.0 at every leg: 1.160 / 1.217 /
   1.136 / 1.160 (n=60 each). The content-borne reading of the episodic effect is
   budget-robust.
5. **Legacy prompt-side arms scale with W as expected:** floor 0.100 → 0.283 and a2_clamp
   0.100 → 0.350 rise as more code/context fits; dump_gf (adapter dump conditioning over
   the floor prompt) tracks floor (0.067 → 0.333). Filler rises with W (0.036 → 0.169) but
   stays at/below floor at every leg — the pointer-content contrast (a2_tail − filler)
   remains ≥ 0.66 everywhere.

## 5. Reproduction

```
uv run --extra gpu python tools/_repobench_clamp_run.py \
  --levels 8k,32k --per-level 30 --offset 100 --window {256|512|1536} --seed 0 \
  --experiment issue52-repobench-clamp --out rb_clamp_w{W}.json
```
Engine commit `c4562db` (tail-overhead guard included). Scorer + loader:
`src/rune/bench/identifier_match.py`, `src/rune/bench/repobench.py`. Durable trace copies
are the MLflow (S3-backed) run artifacts listed in the provenance table; local scratchpad
copies verified byte-identical (sha256 match; recorded in `docs/publication/hashes.txt`).
Verifier: session-scratchpad `verify_wsweep.py` (raw output `verify_wsweep_out.json`).
Prior context: `docs/publication/c1_keystone_findings.md`,
`docs/issue52-repobench-clamp-findings-2026-06-22.md`.
