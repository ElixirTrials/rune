# Handoff: Phase-Benchmark Oracle Pivot

**Date:** 2026-04-22
**Branch:** `feat/training-upgrade`
**Prior session context:** exhausted at 97%; handing off cleanly.

---

## What just shipped (last session)

1. **Reconstruction dataset builder** — full T2L-compatible dataset producer under `libs/model-training/src/model_training/reconstruction/`. 8 tasks, 42/42 tests passing, ruff/mypy clean.
   - Plan: `docs/superpowers/plans/2026-04-22-reconstruction-dataset-builder.md` (commit `bdc3302`)
   - Implementation commits: `9f3ceec` → `8f64491` (8 feature commits)
   - Oracle-source-agnostic: reads any AdapterRegistry, regardless of how oracles were produced.

2. **Strategic pivot locked in via YAML amendment** — `docs/superpowers/specs/2026-04-22-phase-benchmark-pivot.yaml` (commit `afb9431`).
   - **Supersedes:** unbounded per-(task, step) oracle corpus.
   - **Replaces with:** bounded **5 phases × ~6 coding benchmarks = 30 oracles**, bootstrapped via **STaR-style self-distillation filtered by Pass@1**.
   - **Preserves episodic memory** as weight-space specialization per pipeline role.
   - **Makes Report_2 kill-switch natively measurable** (Pass@1 ≥ 5% on held-out benchmarks).

3. **Original PR 28 fit-assessment YAML** committed as historical reference (`78e580d`). This is the source document that the phase-benchmark-pivot amendment builds on top of.

---

## Where we left off

### Pending tasks (priority order)

| ID | Subject | Status | Notes |
|----|---------|--------|-------|
| #21 | Add `DIAGNOSE` to `PipelinePhase` enum | in_progress, code not written | See "Immediate next step" below. ~2 minutes. |
| #20 | Draft implementation plan for self-distillation phase corpus | in_progress, not started | Multi-week effort; writing-plans skill. See "Plan scope" below. |

### Immediate next step — Task #21 (tiny, low-risk)

Single-line change to `libs/shared/src/shared/rune_models.py`:

```python
class PipelinePhase(str, Enum):
    """Pipeline phases for the multi-phase swarm pipeline.

    Each phase uses a dedicated Jinja2 trajectory template and prompt template.
    """

    DECOMPOSE = "decompose"
    PLAN = "plan"
    CODE = "code"
    INTEGRATE = "integrate"
    DIAGNOSE = "diagnose"  # ADD THIS LINE
```

**TDD recipe (drop into `libs/shared/tests/test_rune_models.py` — was reverted from last session):**

```python
# Add to imports:
from shared.rune_models import AdapterRef, CodingSession, EvolMetrics, PipelinePhase


# --- PipelinePhase tests ---


def test_pipeline_phase_has_all_five_canonical_values() -> None:
    """PipelinePhase enumerates all 5 Rune pipeline phases (incl. DIAGNOSE)."""
    values = {p.value for p in PipelinePhase}
    assert values == {"decompose", "plan", "code", "integrate", "diagnose"}


def test_pipeline_phase_diagnose_round_trips_via_string() -> None:
    """PipelinePhase is a str-Enum — DIAGNOSE round-trips through its string value."""
    assert PipelinePhase("diagnose") is PipelinePhase.DIAGNOSE
    assert PipelinePhase.DIAGNOSE.value == "diagnose"
```

**Steps:**
1. Append the two tests + updated import to `libs/shared/tests/test_rune_models.py`.
2. Run: `uv run pytest libs/shared/tests/test_rune_models.py -v -k pipeline_phase` → expect 2 FAILED (ValueError: 'diagnose' is not a valid PipelinePhase).
3. Add `DIAGNOSE = "diagnose"` to the enum.
4. Re-run tests → expect 2 PASSED.
5. `uv run ruff check libs/shared/src/shared/rune_models.py libs/shared/tests/test_rune_models.py`
6. `uv run mypy libs/shared/src/shared/rune_models.py`
7. Commit:

```
feat(shared): add DIAGNOSE to PipelinePhase enum

5th phase (diagnose/repair) has existed in templates and scripts since
the two-step diagnose→repair design landed; now enumerated. Required
for phase-benchmark oracle pivot (see 2026-04-22-phase-benchmark-pivot.yaml).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
```

**Verified safe:** no existing code does attribute-access on the enum (`grep PipelinePhase\\.` returned zero matches); no test file enumerates values exhaustively. Adding a member is non-breaking.

---

### Big task — Task #20: plan for self-distillation phase corpus harness

**Scope (from pivot YAML):** build the upstream producer that fills `AdapterRegistry` with 30 oracles so the reconstruction dataset builder has something to consume.

**Work-breakdown at subsystem level** (each bullet = plausibly its own plan via writing-plans):

1. **Benchmark harness library** (`libs/evaluation/benchmarks/`). Unified Pass@1 runner for HumanEval, MBPP, APPS, BigCodeBench, DS-1000, LiveCodeBench, + held-out SWE-Bench-lite, CodeContests. Takes `(base_model_id, adapter_stack, problem_set)`, returns Pass@1. Used by both oracle validation AND hypernetwork eval. **~1–2 weeks.** Probably first to plan.

2. **Pipeline tracer** (`scripts/phase_corpus_producer.py` + support lib). Iterates benchmarks × problems, invokes `scripts.rune_runner.run_phased_pipeline` with base Qwen 3.5 9B + DeltaCoder, captures 5 phase-boundary outputs per run, persists raw traces as JSONL.

3. **Filter + bin stage**. Consumes raw traces + benchmark harness verdicts; keeps only Pass@1=1 runs; bins by `(phase, benchmark)`; writes one JSONL per bin formatted for `train_and_register`. Include diagnose-phase augmentation (initial-fail → repair-succeed runs only).

4. **Phase-QLoRA orchestrator**. Iterates bins, invokes the existing `trainer.train_and_register` with `task_type=f"{phase}_{benchmark}"` (e.g. `decompose_humaneval`). DeltaCoder warm-start, standard QLoRA recipe from PR 28 HPO winners. Each oracle registered to the `AdapterRegistry` the reconstruction dataset builder reads from.

5. **CLI entrypoint with dry-run** following the `trainer_cli.py` pattern (argparse, `--dry-run` prints resolved JSON without importing torch).

**Interfaces / contracts to encode in the plan:**

- Bin JSONL row schema:
  ```json
  {
    "phase": "decompose",
    "benchmark": "humaneval",
    "problem_id": "HumanEval/42",
    "problem_text": "...",
    "prior_phase_outputs": {"decompose": null, "plan": null, ...},
    "phase_output": "the decomposition the base model produced that led to a passing run",
    "end_to_end_passed": true
  }
  ```

- `task_type` format for registered oracles: `"{phase}_{benchmark}"` — consumed as-is by the reconstruction builder's `task_description_fn` callback (a one-line closure).

- Bootstrap model choice open question: Qwen 3.5 9B + DeltaCoder for all benchmarks OR escalate to Claude/GPT-4 for low-pass-rate benchmarks (APPS, LiveCodeBench < 10%). Leaning Qwen-only for round 1; re-evaluate after data counts per bin are observed. Flagged as `q1_bootstrap_model_choice` in the pivot YAML.

**Reuses (no changes needed):**
- `libs/model-training/src/model_training/reconstruction/*` (42 tests, just landed)
- `libs/model-training/src/model_training/trainer.py::train_and_register`
- `scripts/rune_runner.py::run_phased_pipeline` (5-phase orchestrator)
- `libs/adapter-registry/` (CRUD)
- Existing Jinja2 phase templates under `libs/shared/src/shared/templates/`

**Prereqs for this plan to execute:**
- Task #21 (DIAGNOSE enum) must land first
- Benchmark harness library (subsystem 1 above) should probably land first too — the filter stage needs `pass_at_1(final_code, problem)`

**Recommendation for how to write this plan:** Spawn `writing-plans` skill with scope = **benchmark harness library only** as the first sub-plan. It's the critical-path unblocker and is a clean, self-contained deliverable. After that lands, write the phase corpus producer plan with the benchmark harness as a dep.

---

## Key context for fresh session

### Terminal goal (from pivot YAML)

Sakana-style Doc-to-LoRA hypernetwork that emits **phase-specialized** LoRA adapters conditioned on `(phase_id, task_description, trajectory_state_so_far)`. Single forward pass through the hypernetwork per phase transition at inference. Phase ∈ {decompose, plan, code, integrate, diagnose}.

### Why the pivot (one-paragraph version)

Per-(task, step) oracles were unbounded, had fuzzy step semantics, no natively labeled training data, and no clear evaluation path. (5 phases × 6 benchmarks) = 30 oracles is bounded, has crisp phase semantics (each phase already has its own template/role in Rune), has a clean self-distillation data bootstrap (run base model end-to-end, keep Pass@1=1 traces, bin by phase), and makes the Report_2 kill-switch directly measurable (Pass@1 ≥ 5% on held-out benchmarks).

### Training mode chosen

`phase_benchmark_reconstruction_with_self_distillation`. Reconstruction loss only (no SFT second stage in the initial plan; can add later if needed).

### Must-read on entry

1. `docs/superpowers/specs/2026-04-22-phase-benchmark-pivot.yaml` — **the spec of record.**
2. `docs/superpowers/specs/2026-04-22-pr-28-training-upgrade-fit-assessment.yaml` — upstream assessment of PR 28 that the pivot amends.
3. `docs/superpowers/plans/2026-04-22-reconstruction-dataset-builder.md` — what just shipped; the consumer contract downstream.
4. `instructions/Report_2_LoRA_Fine_Tuning_Strategy.md` — Stage 1 QLoRA → Stage 2 hypernetwork; kill-switch.
5. `instructions/Training_Review2.md` — single-sample dataset risk, monolithic-head critique.

### Current repo state

- Branch `feat/training-upgrade`, clean working tree.
- Recent commits (top of history):
  ```
  78e580d docs(spec): PR 28 training-upgrade fit-assessment YAML
  afb9431 docs(spec): phase-benchmark oracle pivot supersedes per-(task,step) design
  8f64491 test(reconstruction): end-to-end shape + embedding + stats invariants
  5ba4ed6 feat(reconstruction): CLI with dry-run + warm-start/base aliases
  aabc5e8 feat(reconstruction): registry → manifest orchestrator
  7cb14f8 feat(reconstruction): task embedding computation + persistence
  6d76613 feat(reconstruction): registry filter for reconstruction candidates
  4262dae feat(reconstruction): across-corpus z-score statistics
  a9edf09 feat(reconstruction): PEFT state_dict → per-module A/B extraction
  9f3ceec feat(reconstruction): manifest dataclasses + JSON round-trip
  bdc3302 docs(plan): reconstruction dataset builder for T2L training
  ```

### User preferences / conventions (reminders)

- Always `uv run ...`, never bare `python`.
- Google-style docstrings; ruff line-length 88; mypy strict-ish; py312.
- Deferred GPU imports (INFRA-05): torch/safetensors/sentence_transformers inside function bodies so modules stay CPU-importable.
- Conventional Commits; `Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>`.
- Auto Mode was active last session — execute autonomously, prefer action over planning, ask only for destructive/irreversible actions.
- User directly requested parallel subagents for plan execution — use subagent-driven-development for the next big plan.

### Open design questions (from pivot YAML)

1. **Bootstrap model choice** — Qwen-only vs. escalate to Claude/GPT-4 for low-pass benchmarks. Lean: Qwen round 1.
2. **Trajectory embedding freshness** — online (recompute per phase) vs. cached. Lean: online.
3. **Phase embedding init** — learned, one-hot, or learned-from-text-embedding. Lean: learned-from-text.
4. **Diagnose data scarcity** — drop for thin bins vs. cross-benchmark pooling. Lean: drop for bins < 20 samples.

These are unresolved; surface for user decision before locking into the phase corpus plan.

---

## Suggested opening move for fresh session

1. Read this handoff + the pivot YAML.
2. Knock out Task #21 (DIAGNOSE enum) — 2 minutes, clean commit.
3. Ask user: "Which plan do you want first — benchmark harness library, or the phase corpus producer as a rollup that includes the harness?" Then invoke writing-plans on the chosen subsystem.
4. Before writing the plan, surface the 4 open design questions from the pivot YAML for user decisions, since they change the shape of the plan.
