# C4 Stage-1 — I0 symbol-reuse audit + I5 capacity go/no-go (2026-07-09)

Plan `docs/publication/c4_implementation_plan.md` (the C4 Stage-1 plan demanded by the
PR #60 specialist reviews). Two experiments on the frozen C1 keystone instrument:
the **I0 symbol-reuse audit** (do engine trajectories reuse, in round t≥2, symbols
introduced in round t−1?) and the **I5 capacity curve** (K∈{1,2,4,8} facts compiled
into ONE adapter, two build modes, vs K pointers in the prompt tail). Output: a numeric
**go/no-go** for the multi-round continuation benchmark (Stage 2). **No `src/` changes;**
`tools/_repobench_clamp_run.py` and `tools/_specificity_probe.py` byte-untouched.

**Realized gate: S1-GO at the pre-registered default margin M=+0.15 — but at the razor
edge, satisfied by build mode (a) only, and conditional on accepting an instrument whose
run-validity anchor soft-failed.** Both the go and the no-go reading are stated in §4
because the co-author margin sign-off is still pending. Stage 2 is not built by this plan
regardless.

**Provenance (pinned).**
- Harness: new `tools/` files (`_c4_fixture_audit.py` for I0; `_c4_capacity_run.py` +
  composition lib for I5), reusing the C1 scoring surface (`clamp._prefix`, `_gen_line`,
  `_score`, `_assemble_tail_prompt`, `_load_stratified`, `_wilson_ci`,
  `_two_sided_binom_p`, `_paired_discordants`). `_repobench_clamp_run.py` and
  `_specificity_probe.py` unchanged.
- Rows (I5): the C1 keystone rows — `_load_stratified(["8k","32k"], 30, offset=100)`,
  N=60, frozen order, `task_id_order_sha256 3de3ac2adac984f26e4ebd6b8766c083a308be233a5c5478c13d01bdcc478533`.
  C1-parity params: W=768, per-level 30, offset 100, seed 0, `variant="use"`, `anchor=0`,
  `scaling=0.91`, `max_new=48`.
- Rows (I0): six regenerated LCB-v6 sessions, qids 3748,3753,3754,3777,3799,3801, c3,
  seed 0, full mode.
- Checkpoint: c3 (`/tmp/phase1/ckpt/c3_t07_lp2_lg1.pt`), sha256
  `53e24af243a38dfbfad82f7293635bfc592922dd2058fefbbfa10714b5457a3f` — verified before any
  forward pass. Engine commit `534dae6`, stamped in MLflow.
- MLflow: experiment **`issue52-c4`** (exp id 81), `MLFLOW_TRACKING_URI=http://localhost:5000`.
  I0 audit run `b37b106052b14033987456a6efbbb174` (artifact path `audit/`); I5 sanity run
  `f6af67858efb43` (`c4-sanity-W768-K1,2,4,8-off100-seed0`); I5 capacity run
  `87420fb4383444c8aaf148c8b7017a4d` (`c4-capacity-W768-K1,2,4,8-off100-seed0`, artifacts
  `capacity_traces.json` + `stage1_gate.json`). MLflow rows are **not durable on this box**
  (DB-snapshot-loss precedent); every load-bearing number below is also an artifact whose
  sha256 is in `docs/publication/hashes.txt` (§6).

**Verification basis.** Every headline number is from an independent stdlib-only recompute
from the raw per-task artifacts (`/tmp/c4/*.json`, `session.jsonl`), importing nothing from
the harnesses: I0 counts via a hand-rolled de-fence + own AST walk + own prefix recovery
(`/tmp/c4/audit/_independent_recompute.py`); I5 recovery counts, Wilson CIs, McNemar
p-values, bundle-sign tests, K\*, and the gate boolean via a from-scratch verifier
(`/tmp/c4/verify_capacity.py`). **Result: 0 discrepancies — every headline number
reproduces to reported precision** (I0 counts matched the tool per-session, per-round, and
aggregate byte-for-byte; I5 matched the runner and MLflow on every count/CI/p/K\*/gate).
Artifact sha256s re-checked against `hashes.txt` and all match.

## Limitations / accounting decisions (read first)

1. **VM wipe + session regeneration.** The VM was wiped and rebuilt today; `/tmp/c4` and
   the plan-intended fresh `i0_sessions` were lost. Reconciled honestly: (a) the C1 anchor
   trace was re-fetched from MLflow run `f37374906c5f` (its floor/episodic/a2_tail =
   9/31/50 of 60 reconfirm the pinned C1 numbers, so the anchor is against a genuine C1
   artifact); (b) c3 sha re-verified == pinned before any forward pass; (c) the I0 audit
   ran over **already-regenerated** sessions from the same engine + checkpoint rather than a
   fresh GPU run — the audit is CPU AST over existing `session.jsonl`, no forward pass
   needed. See `docs/issue52-lcb-remediation-2026-07-09.md` for the session-regeneration
   campaign (flag-gated, adversarially verified, no task-specific logic).
2. **The gate is razor-edge.** Mode-(a) K=2 delta equals the pre-registered margin
   **to the digit** (18 vs 9 of 60 = +0.1500 == M=0.15); a single one of 60 rows flipping
   drops delta to +0.1333 < 0.15 and flips `go` to False on the delta≥M leg. The McNemar
   leg (p=0.0117) has comfortable margin; the delta≥M leg has **zero slack**. This is
   stated as a first-class limitation, not a footnote (§4).
3. **S1-ANCHOR-2 soft-fail.** The enlarged-rank capacity instrument agrees with the
   native-rank sanity leg on only **46/60** adapter_k1 predictions (< 55/60 soft
   threshold). Characterized as enlarged-rank GEMM-shape drift under greedy/temp-0 decode
   (the plan's only permitted source), netting **−2** recovery (29 capacity vs 31 sanity;
   two recovered→not flips, zero reverse), aggregate well within Wilson overlap. The gate
   is read **conditional on accepting the enlarged-rank instrument**; §5 records the drift
   direction (it does not inflate the razor-edge delta — see §4).
4. **I0 same-target structural caveat.** All 119 eligible I0 pairs are **same-target**
   within-subtask iteration (a function being repaired reuses its own symbols); **0/119**
   are cross-subtask accumulation pairs — a structural property of these six single-subtask
   LCB tasks. The 0.60 threshold is met literally, but these sessions cannot exercise the
   cross-subtask accumulate-K behavior Stage 2 ultimately cares about (§2).
5. **Tail infeasibility scored as failure** (pre-declared deviation from C1's exclusion
   guard): the linear prompt cost hitting the W wall *is the measured quantity*. Sensitivity
   views excluding infeasible rows are noted where they change the reading (K\*, §3).

## 1. Pre-registration (locked before any GPU run; verbatim from the plan)

**Science gate — S1-GO:** at K=2, for **at least one** build mode: paired McNemar
(`adapter_*_k2` vs `floor`, two-sided exact) p < 0.05 **AND** recovery-rate delta ≥ **M**.
**M is the co-author-set margin.** Proposed default **M = +0.15** (≈40% of the K=1
episodic−floor lift of +0.367); **pending sign-off**. Changing M after seeing results is
prohibited. **NO-GO** → Stage 2 is not built; the findings doc records the capacity ceiling;
the systems-eval redirect and the honest a2_tail reframe stand.

**Anchor gates (run-validity, before the science gate is read):** **S1-ANCHOR-1** — sanity
leg (native rank) `floor` and `adapter_k1` match the C1 run `f37374906c5f` trace
token-for-token, 60/60; failure → stop. **S1-ANCHOR-2** — capacity leg (enlarged rank)
`adapter_k1` prediction agreement with the sanity leg (expected ≈60/60; GEMM-shape drift the
only permitted source); agreement < 55/60 → investigate before reading gates.

**I0 threshold:** aggregate reuse fraction ≥ **0.60** over eligible rounds of the six
regenerated LCB sessions → natural trajectories suffice for Stage 2's consistency metric;
< 0.60 → synthetic accumulate-K task mandatory. Either way I5 proceeds.

**Pre-declared deviations from C1:** (1) tail infeasibility scored-as-failure not excluded;
(2) adapter arms at K>1 share one adapter across bundle rows; (3) capacity leg at enlarged
PEFT rank (anchored by S1-ANCHOR-2); (4) mode-(b) keeps ONE bias slice (bias is
conditioning-independent).

## 2. I0 result — symbol-reuse audit

Eligible round = each adjacent code-bearing-round pair (actions code/repair/integrate,
t≥2); reuse = introduced-symbols(prev) ∩ Load-context used-symbols(curr) nonempty
(Store-context rebindings excluded — `x=1` then `x=2` is not reuse). Four regenerated
session sets audited (i1 primary — closest to plan intent; i2–i4 robustness). Pooled =
with broken-tail prefix-parse recovery; strict = both adjacent payloads parse whole.

| set | flag stack | pooled r/e | pooled f | strict r/e | strict f |
|---|---|---|---|---|---|
| **i1** (PRIMARY) | waves 1–2 (ship gate + budget guards) | **20/27** | **0.741** | 18/27 | 0.667 |
| i2 | + wave 3 (repair_context_fix) | 19/26 | 0.731 | 17/26 | 0.654 |
| i3 | + wave 4 (cond-budget + concise), judge on | 31/31 | 1.000 | 29/31 | 0.935 |
| i4 | + no-preserve-logic replan, judge on | 35/35 | 1.000 | 33/35 | 0.943 |
| **aggregate** | all four sets | **105/119** | **0.882** | 97/119 | 0.815 |

i1 per-session pooled: 3748 8/8, 3753 0/0, 3754 1/8, 3777 1/1, 3799 6/6, 3801 4/4.

**Threshold verdict: ABOVE 0.60 on every set, both definitions** (lowest is i2 strict
0.654; primary i1 = 0.741 pooled and 0.667 strict). By the letter of the pre-registration,
natural trajectories suffice for Stage 2's consistency metric; the synthetic accumulate-K
task is **not mandated on I0 grounds**. I5 proceeds independently either way.

**Honest caveat (reported, not suppressed).** 100% of eligible pairs are same-target
within-subtask iteration; **0/119** are cross-subtask accumulation pairs (Limitations §4).
The numeric gate reads GO/above-threshold; the qualitative accumulate-K coverage is zero.
A co-author could reasonably still mandate the synthetic accumulate-K task on that basis —
a judgment call *above* the pre-registered numeric gate, which itself is met.

**OOD statement (plan Task 2 Step 4).** c3 was trained on `benchmarks/mbpp_recall_train.jsonl`
(exactly 40 MBPP tasks; MLflow exp 45 run `fe72f9ddd69c…`), conditioning surface
`render_training_format_trajectory` (## Task / ## Current Code / ## Review Feedback). The
audited sessions are LCB-v6 competitive rows — **OOD in task domain** (LiveCodeBench-v6 vs
MBPP) but **in-distribution in conditioning surface** (same renderer feeds the hypernet). An
OOD null ≠ a mechanism null: this is why the I5 capacity gate does **not** run on these OOD
sessions but on the C1 keystone RepoBench instrument, where c3's episodic capability is
already demonstrated (episodic recovery 0.517 vs a2_tail 0.833 / floor 0.150).

## 3. I5 result — capacity curve

All arms on the 60 C1 keystone rows (W=768, seed 0, `variant=use`, `anchor=0`,
`scaling=0.91`, `max_new=48`). k/n, rate, Wilson 95% CI. `floor` is K-independent =
**9/60 = 0.150 [0.081, 0.261]**, infeasible=0. Adapter arms infeasible=0 at all K.

| K | mode (a) regen-from-scratch | mode (b) incremental-delta | tail (K pointers in prompt) | infeas (tail) |
|---|---|---|---|---|
| 1 | adapter_k1 29/60 0.483 [0.362, 0.607] | (modes coincide at K=1) | 50/60 0.833 | 1 |
| 2 | **18/60 0.300 [0.199, 0.425]** | 12/60 0.200 [0.118, 0.318] | 38/60 0.633 | 2 |
| 4 | 13/60 0.217 [0.131, 0.336] | 4/60 0.067 [0.026, 0.159] | 20/60 0.333 | 8 |
| 8 | 9/56 0.161 [0.087, 0.278] | 0/56 0.000 [0, 0.064] | 3/56 0.054 | 24 |

(K=8 on 56 rows; the `60 mod 8 = 4` remainder dropped and logged.)

**Paired McNemar (adapter vs floor / vs tail; two-sided exact; discordants a_only/b_only).**

| pair | discordants | p | reading |
|---|---|---|---|
| adapter_k1 vs floor | 21/1 | 1.10e-5 | K=1 beats floor strongly |
| adapter_a_k2 vs floor | 10/1 | **0.011719** | mode (a) beats floor (gate leg) |
| adapter_b_k2 vs floor | 6/3 | 0.50781 | mode (b) n.s. vs floor |
| adapter_a_k2 vs tail_k2 | 5/25 | 3.25e-4 | tail beats mode (a) |
| adapter_b_k2 vs tail_k2 | 3/29 | 2.56e-6 | tail beats mode (b) |
| adapter_a_k4 vs floor | 4/0 | 0.125 | mode (a) n.s. vs floor at K=4 |
| adapter_b_k4 vs floor | 1/6 | 0.125 | (mode b trails, n.s.) |
| adapter_a_k8 vs floor | 2/1 | 1.0 | mode (a) ≈ floor at K=8 |
| adapter_b_k8 vs floor | 0/8 | **0.0078125** | mode (b) significantly **WORSE** than floor |

**Bundle-level sign test at K=2** (30 bundles, n_eff=9 after ties; reported because rows in
a bundle share an adapter and are not independent — row-level McNemar stays primary for C1
comparability): mode (a) pos 9 / neg 0, p=0.003906; mode (b) pos 6 / neg 3, p=0.50781.
**Same GO/NO-GO split as the row-level McNemar.**

**K\* crossover** (smallest K where the better adapter mode ≥ tail on paired rows,
infeasibility counted as tail failure): `kstar_a = 8`, `kstar_b = −1` (never). **The
`kstar_a=8` reading is an artifact of tail_k8 collapsing to 0.054 under infeasibility
(24/56 infeasible), not adapter competence** — both mode (a) and tail sit near floor at
K=8. Excluding infeasible rows, no crossover occurs at any tested K. Do not read `kstar_a=8`
as an adapter win.

**What the curve shows (no spin).** Both build modes fall far below tail at every K>1; the
capacity ceiling is steep and monotone. Mode (a) (regenerate) degrades gracefully
(0.483→0.300→0.217→0.161≈floor); mode (b) (sum-of-LoRA-deltas) collapses to 0.000 at K=8 and
is **significantly worse than floor there** (p=0.0078) — rank-stacked deltas do not compose
semantically at high K. K=1 reproduces the C1 loss (episodic 0.483–0.517 vs tail/a2_tail
0.833).

## 4. Realized gate

**Inputs at K=2** (all from raw `capacity_traces.json`, 60 paired rows, sha
`55248426…3afec371b`): floor 9/60=0.1500.
- **Mode (a)** adapter_a_k2 = 18/60 = 0.3000; delta vs floor = **+0.1500**; McNemar p =
  **0.011719**; passes (p<0.05 AND delta≥M=0.15) = **TRUE**.
- **Mode (b)** adapter_b_k2 = 12/60 = 0.2000; delta = +0.0500; McNemar p = 0.50781;
  passes = FALSE.
- `stage1_gate.json`: `go = TRUE` (satisfied solely by mode (a)).

Because the pre-registered margin is still the plan **default M=+0.15 with co-author
sign-off pending**, both readings are stated explicitly per the integrity rule rather than
inventing a margin:

- **GO reading (at M=+0.15, the pre-registered default).** Mode (a) satisfies both legs —
  McNemar p=0.0117 with comfortable margin, delta = +0.1500 = M exactly. **S1-GO.**
  Consequence as pre-registered: Stage 2 *may* be scoped (but is not built by this plan);
  build mode (a), not (b), is the only candidate (b fails at every K and is worse than floor
  at K=8).
- **NO-GO reading (the razor edge).** The delta≥M leg has **zero slack**: 18 vs 9 of 60 is
  +0.1500 to the digit, and a single row flip → +0.1333 < 0.15 → NO-GO. Any co-author margin
  strictly above +0.15 (e.g. one-row-stricter, ~0.167), or a decision to require *separated
  marginal CIs* rather than the paired McNemar, reads **NO-GO** — note mode (a)'s Wilson CI
  [0.199, 0.425] **overlaps** floor's [0.081, 0.261], so the go rests entirely on the paired
  test, not on separated marginal intervals. Under NO-GO, Stage 2 is not built; the capacity
  ceiling here is the record; the systems-eval redirect and the a2_tail reframe stand.

**Instrument caveat on reading the gate.** The go is conditional on accepting the
enlarged-rank instrument, which soft-failed S1-ANCHOR-2 (46/60, §5). The observed drift
direction is enlarged-rank **−2** recovery vs native at K=1 (29 vs 31), i.e. the instrument
*under*-counts — so the razor-edge delta is **not inflated** by GEMM drift (if anything the
native-rank mode-(a) K=2 recovery could be marginally higher). This makes the GO reading
robust to the *direction* of the observed drift, but the soft-fail still means the instrument
is not bit-clean and the co-author should weigh it before locking Stage 2.

**Recommendation to the gate owner (Task 6 does not set M).** Read this as **conditional
S1-GO**: the pre-registered default clears both legs, but with zero delta-slack and an
overlapping marginal CI, it is one row and one margin-decision away from NO-GO. The
defensible next step is a co-author margin sign-off *plus* either a native-rank re-run at K=2
(to retire the ANCHOR-2 soft-fail) or an explicit decision to accept the enlarged-rank
instrument, before any Stage 2 build.

## 5. Anchors and invariance

- **S1-ANCHOR-1 — PASS.** Sanity leg (native rank, engine `generate_adapter` path)
  reproduces C1 run `f37374906c5f` **token-for-token**: match_floor 60/60 exact, match
  adapter_k1 60/60 exact. Recovery reproduces the C1 pins exactly: floor 9/60=0.1500,
  adapter_k1 31/60=0.5167; C1 a2_tail pin 50/60=0.8333 confirmed in the fetched trace.
  Instrument valid; campaign proceeded.
- **S1-ANCHOR-2 — SOFT-FAIL (46/60, < 55/60).** Enlarged-rank vs native-rank adapter_k1
  agreement 46/60; the 14 disagreements are enlarged-rank GEMM-shape drift (identical
  generated prefixes diverging mid-line under greedy/temp-0 decode), netting −2 recovery (29
  vs 31; two recovered→not flips at tasks 2122 & 2125, zero reverse), aggregate within Wilson
  overlap. Flagged for the gate reading (§4); does not auto-block per the pre-registration
  ("investigate before reading gates").
- **Bias-invariance check — PASS.** Validates mode-(b)'s one-bias composition: 36/36 lora_A
  keys have identical bias rank-slices across two different conditionings, and 36/36 context
  slices differ. The bias is conditioning-independent as pre-declared (deviation 4).

## 6. Verification and provenance

**Independent recompute: 0 discrepancies.** I0 — a from-scratch script
(`/tmp/c4/audit/_independent_recompute.py`, hand-rolled regex de-fence + own AST walk + own
prefix recovery, imports nothing from the tool) reproduced the tool's counts EXACTLY,
per-session and aggregate, for all four sets; eligible denominators (27/26/31/35) and
introduced-symbol sets per round matched byte-for-byte. I5 — a stdlib-only verifier
(`/tmp/c4/verify_capacity.py`) matched the runner and MLflow on every recovery count, Wilson
CI, McNemar p, bundle-sign, K\*, and the gate boolean (spot-checked
recovery_adapter_a_k2=0.3, kstar_a=8, bundle_sign_pos=9). One I0 methodology note
(self-resolved, not a report defect): a first-pass recompute over-counted i1/3754 reuse by
treating Store-context reassignments as "used"; applying the documented Load-context rule
reproduced the tool's 1/8 exactly — the report's definition is the conservative/correct one.

**Scope not independently re-derived** (non-gate-bearing, low risk): the strict audit
fraction was verified at the pooled level and on all eligible denominators but the
strict-parse variant was not fully re-run; the I0 same-target/cross-target stratification
(0/119 cross-subtask) was taken from the tool. Neither affects the I0 threshold verdict
(every set > 0.60 on both definitions) nor the capacity gate.

**Artifacts (durable; sha256 in `docs/publication/hashes.txt`).**

| artifact | sha256 | MLflow (exp `issue52-c4`, id 81) |
|---|---|---|
| `capacity_traces.json` | `55248426ad9962cf0bce2c67b9b5a1fc3ced7fc3b45391f167955afbf3ec371b` | run `87420fb4383444c8aaf148c8b7017a4d` |
| `sanity_traces.json` | `b071ad3e92b667ac27afc0de234b125da9059799ccf683f1bd64b9960fb97eb0` | run `f6af67858efb43` |
| `c1_traces.json` (= `rb_clamp_c1_full.json`) | `d0d6d6837cdaeafc8121ecfc6af0ee134e38608fca2fb10fec0662670a870a3a` | exp 79 run `f37374906c5f` (anchor) |
| `i1_audit.json` (PRIMARY) | `56e68602ebf2f7a2c8ba14d1be8312c8153f2fa0c4e849414bbce7144dab07f3` | run `b37b106052b14033987456a6efbbb174` |
| `i2_audit.json` | `a39bacab58689aa21782a4f7b109b12ad6c9be2597298e35f70249e824060695` | run `b37b106052b14033987456a6efbbb174` |
| `i3_audit.json` | `e7dc62b7a93dcc6277659bb1dfac300c991061c0e67af8b5d3f6886cac269469` | run `b37b106052b14033987456a6efbbb174` |
| `i4_audit.json` | `687ecaaeff97bc34dec23c5f145301123fbc2a5b37c6c66324e16a26e03de7e2` | run `b37b106052b14033987456a6efbbb174` |
| `_independent_recompute.py` | `039ec2fea3a615d57cce7124242089cd79bcee4520359d25ed1b5af2b4ebb27a` | run `b37b106052b14033987456a6efbbb174` |

Checkpoint c3 sha256 `53e24af243a38dfbfad82f7293635bfc592922dd2058fefbbfa10714b5457a3f`
(re-verified today, pre-forward). Engine commit `534dae6`. `stage1_gate.json` (293 B) on
capacity run `87420fb4…`. Prior context: `docs/publication/c1_keystone_findings.md`,
`docs/publication/c32_wsweep_findings.md`, `docs/issue52-lcb-remediation-2026-07-09.md`,
`docs/issue52-lcb-failure-modes-2026-07-08.md`.
