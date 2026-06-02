# Issue #52 — Deliverable 4 results: T0 + E1 (experimentation phase, 2026-06-02)

In-repo auditable record of the cheap-diagnostic phase (the chronological working log lives in
`instructions/scratchpad.md`, which is gitignored by repo convention; durable metrics + artifacts are
in the S3-backed MLflow store). Scoring rules were **frozen before any trained-checkpoint delta** in
`docs/issue52-predeclared-spec-T0-E1-E2-2026-06-02.md` (leakage rule). Research/diagnostic harnesses are
snapshotted on the `scratch/issue52-research-tools` orphan branch (not for merge).

## Attribution correction (load-bearing)
The deliverable-4 handoff's headline warm-start numbers (goal +2.30 / file +1.76 / diff +1.01 /
tail +2.01) are **GEMMA (gemma_demo)**, not qwen_4b_d2l. True qwen warm-start: goal +2.235 / file +1.596
/ diff +0.983; code-recall +2.597. There is **no qwen continuation/tail number on disk** — any body/tail
ceiling claim must come from a fresh run. The calibration ladder is re-anchored to the qwen rungs.

## T0 — paired significance of the feedback-swap smoke (MLflow `issue52-T0-paired`)
Controlled paired re-run, one process, `max_seq_length=768`, byte-identical 60 val rows, NaN-paired
shared denominator (n=60).

| | warm-start (arm1) | trained recipe-4b (arm2) |
|---|---|---|
| matched−swap | +0.0188 (frac 0.48) | +0.0691 (frac 0.65) |
| matched−zero | +0.089 | +0.495 |

Paired d = arm2−arm1 = **+0.0503**; bootstrap 95% CI **[+0.010, +0.096]** (excludes 0); sign test
**+37/−19, p = 0.022**; scatter broad (top-5 |d| = 43%). Reproduces the historical +0.0185/+0.0687 under
controlled conditions (the feared length-regime confound was not the driver; the old +0.0019 was the
scaler_B-collapse run).

**Verdict (frozen go/no-go): NULL/NO-GO.** The training gain is real and statistically significant
per-episode, but +0.069 is below the rung-1 body-recall threshold (+0.14) — significant but
magnitudinally trivial (~1–2% of real binding). Does **not** justify a long run on the unfiltered
external_codereview proxy corpus. Decision moves to E1/E2.

## E1 — capacity vs representation (MLflow `issue52-E1-capacity-vs-repr`)
Oracle per-episode LoRA vs hypernet-generated adapter, **matched rank r=8**, `down_proj` only,
`lora_alpha=8×45.2548` (alpha/r = the un-divided functional contract), scored on the **BODY span only**
(`[hi,len)`; signature span hardened to raise+exclude, never the contaminating (0,0) fallback), ABSENT
(hidden) regime, derangement negative. Oracle trains on the **same ABSENT surface it is scored on**
(train==score ⇒ a weak oracle is a true capacity limit, not a transfer artifact).

| | matched body `lp_m` | base `lp_z` | body m−mismatch (episode-specific) |
|---|---|---|---|
| Oracle r8 | −0.22 (overfit-PC pass) | −1.67 | **+21.7** |
| Hypernet | −1.00 | −1.65 | **+0.14** (frac 0.70) |
| Hypernet **signature** | −1.99 | | **+4.09** |

**Verdict: REPRESENTATION/OBJECTIVE wall, not capacity.** An r8 `down_proj` LoRA *can* hold
episode-specific body content (oracle, overfit-PC confirmed). The hypernet binds the **signature**
episode-specifically (+4.09) but the **body** only generically (+0.14 episode-specific; the +0.65 lift
over base is non-specific). The sig-vs-body asymmetry *within the same hypernet at the same rank* is the
decisive, oracle-independent evidence — the doc-Q&A compressor keeps answerable labels and discards the
verbatim body. **Lever = fine-tune/re-objective the hypernet, not raise rank.** Skip the r16/r32 sweep.

### Precision sensitivity (FP is not the explanation)
Re-ran the hypernet body+sig arm in **bf16** vs 4-bit: body m−mismatch **+0.137 (bf16) ≈ +0.141 (4-bit)**;
sig +3.84 ≈ +4.09; per-episode values track tightly (1/10 sign "flip", and that one is ≈0). Scoring uses
fp32 log_softmax + fp64 accumulation; the small +0.14 body margin is the **true** episode-specific binding,
not a 4-bit noise-floor artifact. The representation wall holds at both precisions.

### Precision regime correction
CLAUDE.md lists the base as `Qwen3.5-9B`; all issue-52 work uses **`Qwen3-4B-Instruct-2507`**. The engine
(`src/rune/model/wrapper.py`) loads that base in **bf16**, not 4-bit — the 4-bit nf4 in the eval probes was
a 9B-era leftover. The 4B base in bf16 (~8GB on a 23GB GPU) leaves ample headroom, so downstream work runs
in **bf16** (matches the engine, removes the quantization variable). bf16 is now the E1 body baseline
(+0.137).

## Next: E1 cross-over control (predeclared, frozen)
Does a tiny **contrastive** body-span hypernet fine-tune (matched-body-lp > derangement-partner-body-lp —
*not* CE) on the exact 10 facts move body m−mismatch toward the hypernet's own signature-binding level?
Bar (vs +4.09, not the oracle's +21.7): **+0.14 → ≥ +1.0 = reachable (fine-tune is the lever); stays
~+0.14 = architecture/conditioning attenuation.** This is a *trainability* probe (trained-on-test by
design), not product-generalization. Reuses `hypernet_distill`'s generation→apply→backprop plumbing
(scaler_B preserved). E2 (directionality) remains a separate axis, pending.

## Infra (this branch)
- `_save_checkpoint`: checkpoints upload via `mlflow.log_artifact` → verify present in the S3-backed store
  → delete the local staging copy (kept only on upload failure; never lost). No checkpoints on the tiny
  VM disk.
- `run_guarded.sh`: RAM watchdog + disk-free floor + pidfile registration.
- `instance_guard.sh`: standalone always-on RAM+disk daemon (kills registered guarded jobs on breach,
  never the session/MCP servers).
