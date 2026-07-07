# C1 keystone campaign — findings (2026-07-07)

The load-bearing runs for the TMLR submission (plan `docs/publication/publication_task_plan.md`,
rows C1.1–C1.3): `a2_tail`, `a2_tail_filler`, and `swap` arms added to the N=60 RepoBench
clamp harness and run alongside the five legacy arms on frozen checkpoint c3.

**Provenance (pinned).**
- Harness: `tools/_repobench_clamp_run.py`. Engine commit `ee1a133` (the run's logged
  `engine_commit` param, verified against MLflow).
- Benchmark rows: RepoBench v1.1 Python (`tianyang/repobench_python_v1.1`), split
  `cross_file_first`, levels 8k+32k × 30, **offset=100**, seed 0, W=768, temperature 0.0,
  max_new 48. N=60, same row set as the June keystone run.
- Checkpoint: c3 (`c3_t07_lp2_lg1.pt`), sha256 `53e24af243a38dfbfad82f7293635bfc592922dd2058fefbbfa10714b5457a3f`,
  restored 2026-07-07 from the S3 MLflow artifact (see `hashes.txt`).
- MLflow: experiment **`issue52-repobench-clamp`** (new experiment id 79 after the tracking-DB
  restore), run `f37374906c5f4f5c972b8e7b8127089a`, run name
  `clamp-use-W768-8k_32k-n60-off100-seed0`, `MLFLOW_TRACKING_URI=http://localhost:5000`.
- Per-task trace: `rb_clamp_c1_full.json`, sha256
  `d0d6d6837cdaeafc8121ecfc6af0ee134e38608fca2fb10fec0662670a870a3a`. The **durable copy is
  the MLflow (S3-backed) run artifact**; the local scratchpad copy is byte-identical (same
  sha256). Hash recorded in `docs/publication/hashes.txt`.

**Verification basis.** Every number in this document is from an *independent* re-verification
of the trace, not the harness printout: recovered/EM/edit-similarity were re-derived from the
raw prediction strings with a reimplemented whole-token scorer (all 450 scored arm-rows match
the stored verdicts), and Wilson CIs, exact McNemar, and the attributable fraction were
reimplemented from scratch. Every harness-printed statistic reproduced exactly, including the
full-precision p-values behind the printed `p=0.0000` lines.

## Limitations (verifier-reported; read first)

1. **One real invariant violation — row `cross_file_first/6125` (32k).** The episodic
   conditioning string for this row is itself **2020 tokens** (not the assumed ~124), and the
   harness's `_assemble_tail_prompt` while-loop stops shrinking at budget=0 but never *skips*
   when the conditioning alone exceeds W. Result: `a2_tail` prompt = 2028 tokens and
   `a2_tail_filler` prompt = 2029 tokens, both > the 768-token budget, with
   `prefix_tokens=0` (current-file prefix fully displaced) — the W=768 within-budget-trade
   invariant is violated on that row for both arms. **Impact is nil in the favorable
   direction:** neither arm recovered on 6125; the two arms were symmetrically over-budget and
   token-matched (filler_tokens = 2020 = cond_tokens); excluding the row changes nothing
   (a2_tail 50/59; episodic-vs-a2_tail discordants still 1/20, p=2.10e-05) — if anything the
   row deflates a2_tail's denominator. **Recommendation for the paper:** report 6125 as
   budget-inapplicable in the N accounting. The harness guard has since been added
   (post-run): rows whose tail overhead alone exceeds W are now recorded as skipped
   (`tail_overhead_tokens>W`) and counted in a new `a2_tail_inapplicable` metric, so
   future runs (e.g. the C3.2 W-sweep) cannot silently overflow.
2. **C1.1 realized an outcome outside both pre-registered branches** (a2_tail *strictly beats*
   episodic with separated CIs). The gate's decision logic still resolves to the
   reframe/32k-headline side, but the reverse separation must be stated plainly — see §3.1.
3. All other invariants pass: 60 rows (30/30 per level), unique task_ids, no error keys,
   `a2_full` skipped exactly iff ctx_tokens > 12,000 (exactly the 30 32k rows), swap donors
   admissible on all 60 rows (no identity/substring leaks, donor occurrences ≥ 1),
   `swapped_recovered` re-verified from raw predictions, `recovers_beyond_prompt` flags
   internally consistent. swap-inapplicable rows: 0. One cosmetic discrepancy: an
   intermediate summary quoted the swap CI upper bound as 0.202 vs the log's 0.201 —
   recomputed 0.2013, rounding only.

## 1. Arms (per row; only the *delivery* of cross-file context varies)

- `floor` — no context; prompt = clamp(prefix, W).
- `a2_clamp` — raw context in prompt, clamped to W (front-loaded context evicted).
- `a2_full` — raw context in prompt at full window (ceiling; skipped when the forward exceeds
  12k tokens — the cost argument).
- `episodic_use` — oracle conditioning (episodic `use` template, anchor 0, ~124 tok) in the
  **adapter**; prompt = clamp(prefix, W).
- `dump_gf` — multi-file dump conditioning in the adapter, gold-first (legacy negative).
- `a2_tail` (**C1.1**) — the *identical* oracle conditioning string placed in-prompt at the
  **tail**, adjacent to the cursor, within W=768 (prefix shrinks to make room).
- `a2_tail_filler` (**C1.2**) — token-matched neutral filler in place of the pointer, same
  displacement, same budget.
- `swap` (**C1.3**) — episodic conditioning with the gold identifier renamed to an admissible
  donor identifier; adapter channel otherwise identical to `episodic_use`.

Metric: gold cross-file **identifier recovery** (whole-token match in the completion);
EM + edit-similarity secondary; no sandbox.

## 2. Headline — recovery, all 8 arms (N=60 unless noted)

| arm | recovered | rate | Wilson 95% CI |
|---|---|---|---|
| floor | 9/60 | 0.150 | [0.081, 0.261] |
| a2_clamp | 11/60 | 0.183 | [0.106, 0.299] |
| a2_full | 17/30 | 0.567 | [0.392, 0.726] — *32k skipped 30/30 (pre-registered, ctx > 12k)* |
| episodic_use | 31/60 | 0.517 | [0.393, 0.638] |
| dump_gf | 11/60 | 0.183 | [0.106, 0.299] |
| **a2_tail** | **50/60** | **0.833** | **[0.720, 0.907]** |
| a2_tail_filler | 5/60 | 0.083 | [0.036, 0.181] |
| swap | 6/60 | 0.100 | [0.047, 0.201] |

Per-level, key arms: a2_tail 24/30 (8k), 26/30 (32k); episodic_use 18/30 (8k), 13/30 (32k).
Beyond-prompt recoveries (gold identifier absent from the clamped prompt): episodic_use 22,
dump_gf 3. McNemar floor vs episodic_use: floor-only=1, episodic-only=23, p=2.98e-06 —
the June keystone effect replicates on the same rows (see §5).

## 3. Pre-registered gates — realized branches

### 3.1 C1.1 — `a2_tail` vs `episodic_use` (keystone framing gate)

**Gate as pre-registered** (plan C1.1): *"episodic beats a2_tail, separated CIs → keystone
strengthened (weight beats best prompt channel, info held fixed). a2_tail ≈ episodic
(overlapping CIs) → channel not decisive under oracle conditioning; the 32k-infeasible regime
becomes the headline. This gate outcome tells the article side which keystone framing to
write."*

**Realized numbers.** a2_tail 50/60 = 0.833 [0.720, 0.907] vs episodic_use 31/60 = 0.517
[0.393, 0.638]. CIs **separated in the direction the gate did not enumerate** (a2_tail lower
bound 0.720 > episodic upper bound 0.638). Paired McNemar, n=60: episodic-only=1,
a2_tail-only=20, exact p=2.10e-05. Per level: a2_tail 24/30 (8k) and 26/30 (32k) vs episodic
18/30 and 13/30. The single budget-violating row (6125, §Limitations) was recovered by
neither arm, so this reading is insensitive to it (excluding it: a2_tail 50/59, discordants
still 1/20, p=2.10e-05).

**Which branch obtained.** The **reframe branch** — episodic does **not** beat a2_tail — and
in a form *stronger* than pre-registered: **a2_tail decisively beats episodic_use with
separated CIs**, an ordering outside both enumerated branches. No spin: when the ~124-token
oracle pointer fits in the window, the in-prompt tail channel is the better delivery channel
than the adapter, information held fixed. The pre-registered "channel not decisive under
oracle conditioning" wording is therefore *too weak* for the realized data and must not be
used as-is.

**What it means for the article (A-KEY, A-ORACLE).** Per the gate's decision logic the
headline moves to the **32k-infeasible / constant-prompt regime**: raw context in-prompt
(`a2_full`) is infeasible on all 30/30 32k rows (ctx > 12k), while the adapter channel runs
at constant prompt length everywhere. The paper must additionally state, honestly, that
(a) a2_tail beats episodic with separated CIs, and (b) a2_tail itself remains *feasible* at
32k (26/30) precisely because the distilled ~124-token pointer fits any window — the
infeasibility argument applies to **raw context** (a2_full: 0/30 scored at 32k), not to the
oracle pointer. The Setup's oracle-conditioning wording (A-ORACLE) must make explicit that
the conditioning string is an oracle-distilled pointer held identical across the adapter and
tail channels, and that the keystone comparison is channel-of-delivery, not amount of
information.

### 3.2 C1.2 — pointer effect (`a2_tail` − `a2_tail_filler`)

**Gate as pre-registered** (plan C1.2): *"Report pointer effect = (a2_tail − a2_tail_filler)."*

**Realized numbers.** Pointer effect = 0.833 − 0.083 = **0.750** (50/60 vs 5/60). Paired
McNemar, n=60: a2_tail-only=45, filler-only=0, exact p=5.68e-14. CIs [0.720, 0.907] vs
[0.036, 0.181], widely separated. Filler 5/60 sits at/below the no-context floor 9/60
(filler CI [0.036, 0.181] overlaps floor CI [0.081, 0.261]).

**Which branch obtained.** The pointer effect is large and decisive: the a2_tail gain is
attributable to the **pointer content**, not to token displacement near the cursor —
token-matched filler at identical displacement lands at/below floor. This is the cleanest
paired contrast in the campaign (45 vs 0 discordants) and directly supports the A-KEY /
A-ORACLE wording that the oracle pointer carries the effect.

### 3.3 C1.3 — swap / mutation control (confound gate)

**Gate as pre-registered** (plan C1.3): *"`s`≈floor-CI → refutes frequency/output-bias
confound. between → report attributable fraction (e−s)/(e−f). `s`≈episodic-CI → keystone
compromised (signal article side)."*

**Realized numbers.** swap 6/60 = 0.100 [0.047, 0.201]; floor 9/60 = 0.150 [0.081, 0.261] —
each point estimate lies inside the other's CI; McNemar swap-vs-floor, n=60: swap-only=3,
floor-only=6, p=0.508 (indistinguishable). swap vs episodic_use: CIs separated (swap upper
0.201 < episodic lower 0.393); McNemar swap-only=1, episodic-only=26, p=4.17e-07.
Attributable fraction (e−s)/(e−f) on the full n=60 common support =
(0.5167 − 0.100)/(0.5167 − 0.150) = 25/22 = **1.136** (> 1 because swap is numerically below
floor, 6 < 9). Supporting signal: 12/60 swap rows recovered the **donor** identifier — the
adapter tracks the renamed content. swap-inapplicable rows: 0.

**Which branch obtained.** The **first** branch: s ≈ floor-CI — the frequency/output-bias
confound is **refuted**. Not the "between" branch, and not the "keystone compromised" branch.
The entire episodic−floor effect is attributable to the conditioning *content*: rename the
gold identifier in the conditioning and the episodic advantage vanishes to floor, while the
adapter demonstrably follows the renamed content (donor recoveries).

**What it means for the article (A-FORMAT).** The conditioning-format / confound wording can
state affirmatively that the episodic effect is content-borne, with the attributable-fraction
and donor-recovery numbers above; no compromise signal to send.

## 4. Pointer effect — summary line for the paper

**a2_tail − a2_tail_filler = 0.750** (50/60 vs 5/60; paired discordants 45 vs 0; exact
McNemar p=5.68e-14; n=60 including row 6125, where both arms were symmetrically over-budget
and neither recovered).

## 5. Reproducibility

**Legacy-arm determinism (replication of the June keystone after checkpoint restore).** The
verifier located the June legacy artifact (`rb_clamp_episodic_n60.json`, 2026-06-22) and
aligned all 60 task_ids against this run. Across the five legacy arms, all 270 scored
arm-rows (60×5 minus the 30 pre-registered a2_full skips) have **identical recovered
verdicts (270/270, zero diverging rows)**, and the a2_full skip pattern matches exactly.
Determinism is **bit-exact**: raw prediction strings identical on 270/270 arm-rows; per-arm
counts unchanged (floor 9/60, a2_clamp 11/60, a2_full 17/30, episodic_use 31/60, dump_gf
11/60). Greedy decode on frozen c3 reproduced the June keystone run token-for-token two
weeks later, after restoring the checkpoint from the S3 artifact.

**Known minor caveats (from the build review of the C1 arms):**
- (a) Swap replacement is case-sensitive whole-token, so module-path echoes of the gold
  identifier can survive in the conditioning. Conservative direction: it can only inflate
  swap recovery, i.e. it works *against* the confound refutation that was nevertheless
  obtained.
- (b) `a2_tail` keeps its header outside the clamp while `floor` clamps the whole string.
  This affects only the secondary matched-length comparison; `a2_tail_filler` shares
  a2_tail's construction, so the primary pointer-effect pair (§3.2/§4) is clean.
- (c) Tail assembly was reviewed as able to exceed W only if the *fixed overhead* alone
  exceeded W — which cannot trigger at W=768 with ~124-token conditioning. The review did
  not anticipate the conditioning string *itself* exceeding W; that unexamined case is
  exactly the row-6125 violation in Limitations §1.

**Reproduction.**
```
uv run --extra gpu python tools/_repobench_clamp_run.py \
  --levels 8k,32k --per-level 30 --offset 100 --window 768 --seed 0 \
  --experiment issue52-repobench-clamp --out rb_clamp_c1_full.json
```
Scorer + loader: `src/rune/bench/identifier_match.py`, `src/rune/bench/repobench.py`.
Prior findings and setup rationale: `docs/issue52-repobench-clamp-findings-2026-06-22.md`,
`docs/issue52-repobench-template-hpo-findings-2026-06-22.md`.
