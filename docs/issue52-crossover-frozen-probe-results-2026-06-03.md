# Issue #52 — Body-contrastive cross-over: frozen-probe go/no-go (2026-06-03, us-east)

Resolves the pending go/no-go from `instructions/handoff.md` §8 and
`instructions/next-steps-us-east-2026-06-03.md` (steps A.5–A.7). All numbers below are
from the **frozen E1 body-span probe** (`tools/_specificity_probe.py`, bf16), the only
valid cross-over instrument; the in-loop training readout is necessary-not-sufficient.

## TL;DR verdict — QUALIFIED / reachable-but-not-clean-GO

The body-derangement contrastive fine-tune **moves the predeclared primary metric hugely
and reproducibly** (absent/body m−mismatch **+0.137 → +1.026**, all 10 episodes up, sign
test +10/−0 p=0.002, bootstrap 95% CI [+0.558, +1.299]) — so the lever is
**gradient-reachable**. **But** the decomposition shows the gain is **~91% deranged-partner
suppression, ~8% matched-body rise** — the exact "margin movement, not matched rising"
pattern the handoff flagged as *not the desired signal*. Body **accessibility**
(`lp_matched` / m−zero) is essentially **flat**; what improved is body **discriminability**.

**Framing: this is objective misspecification, not hypernet brokenness or probe
invalidity** (GATE A/B + the `lp_zero`-identical decomposition make that hard to argue).

**Recommendation: iterate the pilot, do NOT launch the long run.** Re-anchor the objective
on **matched-body recall** with the derangement term demoted to a **guard**: optimize body
`m−zero` / mean `lp_matched` as the *primary* term (floor target), and keep the hinge as a
*constraint* that fires only when `lp_mismatch` rises or when an m−mismatch gain comes with
Δmatched ≤ 0. (Unguarded `m−zero` alone reopens the CE generic-boost confound — base is
fixed, so it lifts matched **and** mismatch together; the guard is what makes it safe.) The
signature span proves matched *can* rise under this gradient (`lp_matched` +0.875), so
positive body-binding is reachable in principle — the contrastive-hinge-alone objective just
isn't capturing it (suppression is the gradient path of least resistance, §mechanism).

## Gates (all passed — the result is trustworthy)

- **Env parity (migration gate): reproduced.** Fresh 30-step run `d16efade` reproduced
  `c401f0c0` to 4 decimals (loss 0.1039, lp_matched −0.7677, lp_mismatch −3.5975, lp_zero
  −1.5663, scaler_B/absmax 0.1768). The trained `checkpoint_step30.pt` sha256 came back at
  the recorded historical value (`d296a4e2…`); I report it but do **not** lean on
  bitwise-determinism-across-transformers-5.9 as a finding — the verdict rests on the probe
  numbers + GATE B, which stand regardless.
- **Reconstruction faithful (GATE A + GATE B).** `benchmarks/mbpp_phase0_iter.json` was
  rebuilt from the committed corpus with a **byte-exact round-trip**
  (`render_training_format_trajectory(desc) == corpus["context"]`, all 10).
  `tools/scoring_core.py` was already committed (intact; convention confirmed by the
  committed toy test). Acceptance: the fresh **warm-start** probe lands on history exactly
  — absent/body **+0.1370** (hist +0.137), absent/sig **+3.8365** (hist +3.8–4.09).
- **Retention.** Signature binding not traded away (absent/sig m−mismatch +3.84 → **+5.95**);
  matched−zero discipline > 0 (no scaler_B collapse; absmax constant 0.177).

## Primary results — absent regime (NIAH/hidden), the decisive regime

| span | warm-start m−mismatch | trained m−mismatch | warm m−zero | trained m−zero |
|---|---|---|---|---|
| full | +1.170 | +2.350 | +2.013 | +2.249 |
| **body** | **+0.137** | **+1.026** | **+0.840** | **+0.915** |
| sig  | +3.837 | +5.948 | +5.245 | +6.121 |

Per-episode body m−mismatch (warm → trained, all 10 positive):
`11 +0.03→+0.16 · 12 +0.43→+1.61 · 14 −0.13→+0.85 · 16 +0.62→+2.05 · 17 −0.31→+1.97 ·
18 −0.01→+0.91 · 19 +0.05→+0.30 · 20 −0.03→+0.39 · 56 +0.13→+0.89 · 57 +0.60→+1.12`.
Paired Δ mean +0.889, sign test **+10/−0 p=0.002**, bootstrap 95% CI **[+0.558, +1.299]**.

## The decisive decomposition (why this is not a clean GO)

`lp_zero` (no-adapter base) is identical across both probe runs, so Δ(m−zero) = Δ(lp_matched):

**absent / body** (mean abs logprobs):

| | lp_matched | lp_mismatch (deranged) | lp_zero (base) |
|---|---|---|---|
| warm-start | −1.026 | −1.163 | −1.866 |
| trained | −0.951 | −1.977 | −1.866 |
| **Δ** | **+0.075** | **−0.814** | 0.000 |

→ The +0.889 m−mismatch gain = **matched rose +0.075 + deranged fell +0.814**. The
adapter learned to **suppress the wrong episode's body**, not to **recall the right
episode's body** better.

**Accessibility stat-test (not eyeballed).** Per-episode Δ`lp_matched` (≡ Δm−zero, since
`lp_zero` is identical per episode): mean **+0.075**, sign test **+9/−1 p=0.021**, bootstrap
95% CI **[−0.212, +0.274] — spans zero** (dragged by mbpp/19, −1.075). So accessibility is
**directionally broad (9/10 up) but magnitude-negligible and not magnitude-significant** —
the body-recall improvement is real-in-sign but ~10× smaller than the suppression effect
that drives the headline metric.

**absent / sig** (contrast — here matched genuinely rises):

| | lp_matched | lp_mismatch | lp_zero |
|---|---|---|---|
| warm-start | −2.284 | −6.121 | −7.529 |
| trained | −1.409 | −7.356 | −7.529 |
| **Δ** | **+0.875** | **−1.236** | 0.000 |

→ Signature m−mismatch gain (+2.11) = matched +0.875 + deranged −1.236. Matched-rise is
real for the signature. Proof the gradient *can* raise matched — the body just doesn't.

## Interpretation vs the predeclared bar

The handoff predeclared (decision threshold, not truth boundary): "+0.14 → ≥ +1.0, **with
matched rising more than mismatch/zero** and signature retained = scale." The m−mismatch
cleared +1.0 with overwhelming significance, **but the matched-rising-more-than-mismatch
condition is not met** — matched held, mismatch fell. The handoff is explicit that "margin
movement alone" via "pushing the deranged partner down" is *not* the desired signal. So
this is **reachable-and-reproducible but not a clean GO**: the contrastive body objective,
as configured, yields **discriminability without accessibility**.

This is *not* a NULL (the objective bites massively, 10/10) and *not* the generic-boost
FAIL (matched and mismatch did not rise together — mismatch fell). It is a third,
informative outcome the predeclaration under-specified.

### Train-loop vs frozen-probe gap (don't iterate blind)
Handoff §8 trainer-sanity showed in-loop `lp_matched` rising and `lp_mismatch` falling (the
"desired shape"). The frozen probe shows the *opposite emphasis on body* (~91% suppression /
~8% matched-rise). Plausible causes (AI-engineer review): (a) the hinge is active mostly on
easy negatives during training; (b) subtle readout-token differences between the distill
forward and the probe's absent regime; (c) 30 steps overfit the *loss surface* without
transferring to the *probe geometry*. **Action for the next pilot: log BOTH the in-loop
readout AND a cheap frozen-probe snapshot (5–10 episodes) every N steps**, so the objective
is tuned against the metric that actually decides.

### Mechanism — why suppression wins (not a bug)
With the base frozen and `lp_zero` unchanged, the hypernet only shifts adapter-conditioned
logits. The hinge `margin − (lp_matched − lp_mismatch)` has two cheap fixes: pull
`lp_mismatch` down or pull `lp_matched` up. On a short body span the deranged context is a
*strong, stable negative*, while raising matched body competes with the existing warm-start
binding and spreads gradient over fewer high-surprisal tokens than the signature span —
consistent with sig matched +0.875 vs body +0.075. Suppression is the gradient path of least
resistance for this loss.

## Recommended next step (gated — iterate the pilot, not the long run)

1. **Re-anchor on matched-body recall, derangement as a guard** (ordered options):
   - *Primary + guard (recommended):* primary = body `m−zero` / mean `lp_matched` with a
     floor target; derangement hinge kept as a **constraint** that penalizes only when
     `lp_mismatch` rises or an m−mismatch gain comes with Δmatched ≤ 0. The guard is what
     makes m−zero-as-primary safe against the generic-boost confound.
   - *Asymmetric / two-phase hinge:* `max(0, margin − (lp_m − lp_n)) + λ·(−lp_m)`, λ tuned so
     pilot logs show **matched moving before mismatch collapses**.
   - *Trainer decomposition logging:* mirror the frozen-probe table in MLflow each step
     (Δ`lp_matched`, Δ`lp_mismatch`, fraction of the m−mismatch gain from each) so a future
     auto-VERDICT can't read GO off m−mismatch alone.
2. **Corpus note:** flag **mbpp/19** — it is the lone accessibility regression (−1.075) and a
   large share of the suppression-heavy win. If one episode drives the headline, corpus
   expansion may matter as much as loss shape.
3. Only if matched-body lp moves materially (toward the oracle's −0.22 / the +7.7 NIAH
   regime) → corpus-quality gate → short full run → HPO. The 10-row move alone does not
   launch the long run (handoff §3/§5).

### Next-pilot success criterion (predeclare now — freeze before looking at deltas)
The win is stated on **accessibility AND generation utility jointly** — recall is
instrumental, not the goal (recall is the Sakana *Q&A* objective; ours is code *generation*,
so verbatim-recall and next-step-quality can diverge):
1. **Δ`lp_matched` on absent/body is material with a paired bootstrap CI excluding zero**
   (this cross-over: +0.075, CI [−0.21, +0.27] — fails it), AND
2. **Δ`lp_mismatch` does not dominate** the m−mismatch gain (no suppression-only win), AND
3. **Generation co-primary (not just a gate):** xgrammar pass@1 / valid-code **not regressed in
   the absent regime** (the operative one), AND **present-regime spec-compliance not corrupted**
   — the *recitation-dominance canary*: if raising absent-recall makes the adapter override an
   in-prompt spec, the objective has pushed the base toward Q&A recitation and away from code
   generation. A recall win that drops generation in either regime is a **FAIL**.

A repeat of an m−mismatch-only headline (margin up, matched flat) is **not** a pass; nor is a
matched-recall win that degrades generation.

**Facet caveat (do not over-generalize the body cross-over):** the 10-MBPP-body probe is the
*continuation* facet, where correct-next-step ≈ the stored body, so recall and generation
coincide — it is the recall-*friendly* case. The **tried-and-failed/avoid** facet is the
opposite: the base must recall the failure but generate a *divergent* fix, so the target there
is **corrected code, not verbatim recall** (adapter makes the failure accessible, not
reproducible). That objective is designed when real failure-bearing trajectories exist.

### Scaling-path audit (the +0.137 is not an apply artifact — checked 2026-06-03)
Challenged whether the low body number is the historical `alpha/r` (8×-too-weak) bug.
Verified it is not: runtime `effective_scaling = lora_alpha = 45.2548` un-divided (the bug
was `alpha/r = 5.66`); rune's `_lora_delta` is byte-equivalent to ctx_to_lora's native
`lora_forward` (same einsum + `* scaling`); `use_bias=True` head bias included via
`combine_lora`; `_parity_engine_vs_functional.py` proved engine==functional. **Dispositive:**
sig (+3.84) and body (+0.137) are scored in one forward with one adapter application — a
global scaling error cannot lift one span while leaving the other flat, so the asymmetry is
hypernet *encoding*, not apply. (The +7-nat figure some notes recall is the **Gemma NIAH**
calibration — different base/checkpoint/fact regime — not a bound on qwen absent/body.)

### Gates NOT yet evaluated (deferred — name them on the experiment card before any HPO)
- **xgrammar pass@1 (generation-stability)** — NOT run. Promote to a **first-class pilot
  gate**: specificity-via-suppression risks *destructive interference* on non-matched
  content, which can improve probe margins while hurting matched-token structured generation.
- **Retention** (goal/file/diff/code-recall matched−mismatch) — NOT evaluated by this probe
  (covers only MBPP body/sig/full). Deferral is acceptable *only* because this verdict does
  not promote to the long run.

## Reproduction

- Parity run: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True tools/run_guarded.sh
  /tmp/xover.log tools/_distill_entry.py --config configs/issue52_body_crossover_4b.yaml
  --max-steps 30` → MLflow exp `issue52-body-crossover` (43), run `d16efade…`,
  artifact `checkpoints/checkpoint_step30.pt`.
- Reconstruct probe task list: `uv run python tools/_reconstruct_mbpp_phase0.py` (GATE A).
- Frozen probe: `tools/run_guarded.sh /tmp/probe_X.log tools/_specificity_probe.py
  [--ckpt <step30.pt>] --out /tmp/probe_X.jsonl`.
- Decision: `uv run python tools/_crossover_decision.py /tmp/probe_warmstart.jsonl
  /tmp/probe_trained.jsonl` (NB: its auto-VERDICT keys on m−mismatch≥+1.0 only and does
  **not** apply the matched-rise decomposition — read the decomposition table above, not
  the auto-line).

## Merge hygiene
All cross-over scaffolding stays REMOVE-BEFORE-MERGE (handoff §8 manifest). This is a
trained-on-test trainability probe; promote `body_derangement` only if a redesigned
objective demonstrates matched-body accessibility, with its own tests/docs.
