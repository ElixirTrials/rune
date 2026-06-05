# Goal-3 conclusions — corrected runner + flavor HPO (2026-06-05)

**Branch:** issue52-bf16-body-contrastive · **Checkpoint:** c3 (`c3_t07_lp2_lg1.pt`).
Overnight rerun of the Goal-3 experiments on the **corrected runner**, with an HPO to
completion. All runs: corrected runner = freeform codegen + public-example oracle +
(judge OFF — see below). MLflow-logged; sessions on disk; HPO study in
`optuna_bench_hpo.db` (resumable).

## The corrected runner (what changed this session)

Three runner defects made multi-turn repair untestable; all fixed + validated:

1. **Freeform codegen** — code was JSON-wrapped (`{"code": ...}`); the instruct model
   over-escaped newlines (`\n`→`\\n`), collapsing multi-line code to one line → phantom
   line-1 SyntaxError that poisoned the repair loop. Now code is emitted freeform,
   de-fenced (markdown-it) + ast-validated. Probe: freeform 3/3 compile vs JSON path
   stochastically collapsing.
2. **Public-example oracle** (`rune/engine/oracle.py`) — the only in-loop check was
   `run_in_sandbox(strip_self_tests(code))` = a bare `def` → exit 0, so logic errors never
   triggered `diagnose→repair` (only module-load crashes did). The oracle runs the spec's
   *public* doctest examples in-loop (no held-out leakage). `decode_string` now engages
   repair with a real traceback → accurate diagnosis.
3. **In-loop model-judge** — implemented (judge names a concrete failing input → flip to
   failing), but **validation exposed false positives**: it flagged a *correct* int_to_roman
   ("wrong on input 4") while its own reasoning concluded the code was fine (root cause:
   verdict emitted before reasoning; fixed by reordering `JudgeResult` reason→verdict, still
   unvalidated). It is also slow (~35 min / 4 hard tasks via spurious repair). **Decision:
   judge OFF for the HPO** — a false-positiving judge corrupts pass@1 (the HPO metric) and
   inflates runtime; it stays a separate, to-be-validated arm (per advisor).

## Flavor × adapter_scaling HPO (the "experiment we did earlier", clean rerun)

`configs/goal3_flavor_hpo.yaml`: prompt_mode ∈ {training_exact, reference_a/b/b1/c}
(spec-in-adapter) × adapter_scaling[0.1,1.5], c3, 16 trials (fresh), goal3_ref_pool_24,
0.70 tuning split. ~4.3h.

- **Best: `reference_a` @ adapter_scaling 0.627** → tuning pass@1 **0.588 (10/17)**,
  held-out **validation pass@1 0.571 (4/7)**.
- Best held since trial 2; no other flavor/scaling beat it. Spec-in-adapter plateaus
  ~0.57–0.59. (`reference_a` = plain `## Task` spec-in-adapter, closest to c3's training
  format — the training-faithful encoding wins among the flavors.)

## The decisive 3-way comparison (same tasks, corrected runner, judge off)

To separate "spec **in** the adapter" from "adapter **as added** memory", three arms on the
**same held-out 7** (`mbpp/115,133,118,119,106,113,135`) and on the full 24:

| arm | spec location | adapter | val7 (n=7) | full 24 (n=24) |
|---|---|---|---|---|
| scale0 | prompt | off | 0.714 (5/7) | **0.792 (19/24)** |
| HPO best `reference_a`@0.63 | **adapter only** | on | 0.571 (4/7) | ~0.583 (14/24) |
| c3-full | prompt **+** adapter | on (scale 1.0) | 0.857 (6/7) | **0.750 (18/24)** |

Paired (same tasks) c3-full vs scale0 on the 24: **+2** recovered (mbpp/115, 123),
**−3** lost (mbpp/125, 131, 141) → net **−1**.

## Conclusions

1. **Spec-in-adapter *instead of* the prompt costs accuracy** (0.583 vs 0.792 on 24; 4/7 vs
   5/7 held-out). The hypernetwork does not carry the full task spec as reliably as the
   prompt — consistent with the earlier single-turn findings. The reference/spec-in-adapter
   flavors are the *wrong* way to use the adapter for these tasks. **This is the robust
   result** (large gap, n=24).
2. **The adapter *added* to the prompt is net neutral here, NOT a win.** On the full 24 it
   is **slightly negative** (18 vs 19; +2 recovered, −3 lost — it perturbs the model both
   ways and trades tasks at scaling 1.0). The n=7 held-out slice looked positive (0.857 vs
   0.714) but that **reversed on n=24** — a concrete small-sample mirage. No clear additive
   benefit at scaling 1.0 on this easy pool; a scaling sweep for full-mode is the obvious
   next step (the HPO tuned scaling only for the spec-in-adapter flavors).
3. **Repair now genuinely engages** (the oracle gives a real in-loop signal; e.g.
   decode_string repairs on its public example), but this **easy, high-floor pool
   (scale0 0.79) cannot exercise the multi-turn repair-memory thesis** — most tasks pass on
   attempt 1, so few repair episodes occur. Testing whether the adapter *helps repair across
   turns* needs a harder slice defined as **tasks that fail the public example on attempt 1**.

## Caveats (honest)

- **Small n / single seed.** The held-out 3-way is 7 tasks (6 vs 5 vs 4 of 7 = one-task
  differences); point estimates, not significance. The 24-task numbers are the firmer ones.
- **c3-full used scaling 1.0 (untuned)**; `reference_a` used the HPO-tuned 0.627 — not a
  perfectly controlled scaling comparison. A scaling sweep for full-mode is the obvious
  follow-up.
- **Judge excluded** from all these numbers (false-positives + slow). Its reason-first fix is
  committed but unvalidated; it remains a separate arm.
- **Pool is easy** (simple MBPP); conclusions may not transfer to harder, repair-heavy tasks
  — which is exactly the slice the repair-memory thesis needs next.

## Next

1. Validate the reason-first judge in isolation (does it flag the calculate held-out gap
   *without* false-positiving correct code?) before any in-loop use.
2. Build the repair-triggering slice (attempt-1 public-example failures, disjoint from c3
   train) and run the multi-turn 3-arm (scale0 / warm / c3) to test the repair-memory thesis
   now that repair engages.
3. Scaling sweep for c3 prompt_mode=full (adapter-as-added-memory) to find its best strength.
