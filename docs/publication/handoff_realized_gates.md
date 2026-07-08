# Consolidated handoff — realized gate outcomes for the article side (2026-07-08)

This is the handoff promised by `docs/publication/publication_task_plan.md` §"Handoffs to
the article side": one block per article item (A-KEY … A-REASONCACHE), each with the
realized numbers, the gate branch that obtained, a one-sentence instruction to the article
side, and the provenance pointer. All numbers below are independently verified (see the
"Verification basis" sections of the cited findings docs). Code-side work for the TMLR
submission is **complete**: C0.1, C0.2, C1.1–C1.3, C2.1, and C3.2 are run and verified;
C3.1 is explicitly declined.

---

## A-KEY — keystone headline

**Realized (C1.1, W=768, N=60):** a2_tail 50/60 = 0.833 [0.720, 0.907] vs episodic_use
31/60 = 0.517 [0.393, 0.638] — CIs separated in the direction *neither* pre-registered
branch enumerated (a2_tail beats episodic; McNemar p=2.10e-05, discordants 20v1). Per the
gate's decision logic the **reframe branch** obtains: the headline moves to the
**32k-infeasible / constant-prompt regime** — raw context in-prompt (`a2_full`) is
infeasible on 30/30 32k rows (ctx > 12k), while the adapter channel runs at constant prompt
length everywhere (episodic 13/30 at 32k where a2_full scores 0/30 attempted).

**What the W-sweep (C3.2) adds:** the tail-vs-adapter gap **closes monotonically as W
grows** — +0.358 (W=256) → +0.345 (512) → +0.331 (768, guard-consistent) → +0.131 (1536),
losing significance only at W=1536 (p=0.092, n.s.; discordants still 10v3 favoring tail);
a2_tail itself is **flat-high** across the 6× budget range (0.875 → 0.831); and at W=1536
episodic jumps to 0.700 mirroring floor's rise to 0.283 — adapter pointer and longer
prefix **compose additively** (lift over floor stable at ~0.37–0.42, p ≤ 3e-06 at every W)
rather than becoming redundant.

**Instruction:** write the keystone as the 32k-infeasible/constant-prompt headline, state
plainly that the in-prompt tail pointer beats the adapter with separated CIs at W ≤ 768 and
that this advantage is budget-dependent — closing to non-significance at W=1536 without
licensing an adapter-parity claim — and do not use the pre-registered "channel not decisive"
wording, which is too weak for the realized data.

**Provenance:** `docs/publication/c1_keystone_findings.md` §3.1;
`docs/publication/c32_wsweep_findings.md` §3–4; MLflow exp 79 `issue52-repobench-clamp`
runs `f37374906c5f4f5c972b8e7b8127089a` (768), `ab4d331287774435abb0653967469551` (256),
`3ba5333785ee4acfa19cc526cf00ca91` (512), `d3aae62a5c514acb94b1b2a6381d85a9` (1536).

## A-ORACLE — Setup oracle-conditioning wording

**Realized (C1.2 + C3.2):** the conditioning string is an oracle-distilled ~124-token
pointer held *identical* across the adapter and tail channels — the keystone comparison is
channel-of-delivery, information fixed. Pointer effect at W=768: a2_tail − a2_tail_filler
= 0.833 − 0.083 = **0.750** (50/60 vs 5/60; discordants 45v0; McNemar p=5.68e-14). The
W-sweep confirms the pointer *content* carries the effect at every budget: token-matched
filler at identical displacement sits **at/below floor at every W** (2/56, 4/58, 5/59,
10/59 vs floor 6/60, 8/60, 9/60, 17/60), and a2_tail − filler ≥ 0.66 everywhere.

**Instruction:** the Setup must state that the conditioning is an oracle-distilled pointer
(oracle wording explicit), identical across channels, and that the tail arm's gain is
attributable to pointer content, not token displacement near the cursor — supported at all
four budgets.

**Provenance:** `docs/publication/c1_keystone_findings.md` §3.2, §4;
`docs/publication/c32_wsweep_findings.md` §1, §4.5.

## A-OBJ — objective claim (de-biased number)

**Realized (C2.1, fresh pool n=120):** mean Δlp_matched = **+0.147430** (c3 −0.990289 vs
warm-start −1.137719), 100/20 positive per-task deltas, exact two-sided sign test
**p=5.508e-14**, percentile bootstrap 95% CI **[+0.109, +0.191]** (10,000 resamples,
seed 0). **Gate branch: p < 0.05** → the de-biased number replaces the selection-biased
+0.105 (p=0.064, n=24 heldout) and the selection-bias caveat is stripped.

**Caveat to carry (one sentence):** the 120-task pool was used as training data for the
*separate* n80/n160 scaling checkpoints but never for training or selecting c3; under an
over-strict criterion excluding tasks touched by any training of any checkpoint, only 46
tasks remain — reported for transparency, though it cannot bias an estimate made with
frozen c3.

**Instruction:** replace +0.105 / p=0.064 with +0.147 / p=5.5e-14 / n=120 (CI [+0.109,
+0.191]) in §5.2 and the abstract, strip the winner's-curse caveat, and carry the
strict-criterion sentence above.

**Provenance:** MLflow exp 45 `issue52-phase1` run `1769a1f8dedd43a789041536294c9825`;
pool `benchmarks/mbpp_recall_fresh120.jsonl` sha256 `6142c54b5c3560320bb0fee7661c8bf49f7f0f864297a82eed653512ff887507`;
methodology and pre-registration: `docs/publication/c21_prep.md`; feasibility and pool
derivation: `docs/publication/c01_corpus_lookup.md` (§5 for the 46-task strict variant).

## A-REPRO — reproducibility appendix

**Hash paste:** `docs/publication/hashes.txt` is the manifest to paste — c3 checkpoint
(`53e24af2…`), all MBPP corpus splits, the external_codereview splits, and the per-task
traces of every clamp run including the four W-sweep legs.

**MLflow DB-loss note (state in the appendix):** the tracking DB was restored from a
snapshot predating experiments 78–86; those runs' param/metric rows are lost, but their
per-task artifacts persist in S3 and are hashed in the manifest; the C1/C3.2 campaigns are
re-logged first-class in the new experiment id 79 (`issue52-repobench-clamp`).

**Bit-exact replication:** after restoring c3 from the S3 artifact, the five legacy arms
reproduced the June keystone run **token-for-token on 270/270 scored arm-rows** two weeks
later (greedy decode, frozen c3); the C3.2 a2_full check extends this — identical
predictions on all 30 scored rows across all four sweep legs.

**Row-6125 accounting:** row `cross_file_first/6125`'s conditioning (~2020 tokens) exceeds
every tested W; report it as budget-inapplicable in the tail-arm N accounting
(guard-consistent denominators 56/58/59/59 at W=256/512/768/1536). The harness guard
(engine `c4562db`) now records such rows as skipped (`tail_overhead_tokens>W`) in
`a2_tail_inapplicable`; the W=768 C1 leg predates the guard, and its guard-consistent
variant (a2_tail 50/59) changes no statistical conclusion.

**Instruction:** paste `hashes.txt` verbatim, include the DB-loss provenance note and the
270/270 bit-exact replication as the determinism evidence, and use guard-consistent N
accounting with row 6125 marked inapplicable in any tail-arm table or figure.

**Provenance:** `docs/publication/hashes.txt`; `docs/publication/c1_keystone_findings.md`
§5, Limitations §1; `docs/publication/c32_wsweep_findings.md` Limitations, §2–3.

## A-FORMAT — conditioning-format / confound wording

**Realized (C1.3 + C3.2):** the frequency/output-bias confound is **refuted** (first
pre-registered branch; no compromise signal). swap ≈ floor at **all four W**: 2/60, 3/60,
6/60, 13/60 vs floor 6/60, 8/60, 9/60, 17/60 — swap ≤ floor in rate everywhere, McNemar
p = 0.125 / 0.125 / 0.508 / 0.424 (all n.s., discordants 0v4 / 1v6 / 3v6 / 5v9), and swap
vs episodic is separated at W=768 (p=4.17e-07). Attributable fraction (e−s)/(e−f) =
**1.136** at W=768 (1.160 / 1.217 / 1.160 at 256/512/1536; ≥ 1.0 because swap sits
numerically below floor). Donor-recovery signal: **12/60** swap rows recovered the renamed
*donor* identifier — the adapter tracks the conditioning content.

**Instruction:** state affirmatively that the episodic effect is content-borne — renaming
the gold identifier in the conditioning collapses the episodic advantage to floor at every
tested budget while the adapter demonstrably follows the renamed content (12/60 donor
recoveries) — using the attributable-fraction and swap-vs-floor numbers above.

**Provenance:** `docs/publication/c1_keystone_findings.md` §3.3;
`docs/publication/c32_wsweep_findings.md` §4.4.

## A-REASONCACHE — ReasonCACHE comparator

**Realized: C3.1 explicitly DECLINED.** The pre-registered trigger — advisor election of
Option A (prefix-tuning / KV-injection arm) — was not met; no KV-injection harness was
built and no run occurred. The Option B prose comparison stands.

**Instruction:** keep the ReasonCACHE comparison as prose (Option B); do not cite any
Rune-side KV-injection or prefix-tuning numbers, as none exist.

**Provenance:** `docs/publication/publication_task_plan.md` row C3.1 (trigger condition).
