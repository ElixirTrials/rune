# Plan Execution Handoff — 2026-04-22

**Read this first.** You are resuming execution of three committed plans that together deliver the phase-benchmark pivot spec (`docs/superpowers/specs/2026-04-22-phase-benchmark-pivot.yaml`). The user invoked `/superpowers:executing-plans` then course-corrected to **subagent-driven-development**. Auto mode is active — execute autonomously, minimize interruptions, prefer action over planning.

## Branch state

- **Branch:** `feat/training-upgrade` (do NOT work on `main`)
- **Latest commit:** `388ef18 docs(plan): phase corpus producer for 25-oracle self-distillation`
- **Three plans committed and ready to execute:**
  - `dbabe1f` → `docs/superpowers/plans/2026-04-22-trajectory-encoder-pretraining.md` (Plan B, 9 tasks)
  - `2461812` → `docs/superpowers/plans/2026-04-22-benchmark-harness-library.md` (Plan A, 10 tasks)
  - `388ef18` → `docs/superpowers/plans/2026-04-22-phase-corpus-producer.md` (Plan C, 11 tasks)
- `main` branch is untouched.

## Execution order (user-specified)

```
Step 1  →  A + B (parallel)  →  Step C
```

1. **Step 1 — `rune_runner.py --output-json` one-liner.** Inline. Trivial prerequisite for Plan C Task 2.
2. **Plan A + Plan B in parallel.** Use `superpowers:subagent-driven-development` — dispatch a fresh subagent per task; review between tasks. These plans touch disjoint code paths (`libs/evaluation/benchmarks/` vs `libs/model-training/encoder_pretrain/`), so they can run concurrently.
3. **Plan C after Plan A completes.** Plan C imports `from evaluation.benchmarks import run_benchmark`. Plan C's tasks 1–9 mock this import and can technically start earlier, but to avoid import-path churn, finish Plan A first.

## Required skill

**Invoke `superpowers:subagent-driven-development`** — NOT `executing-plans`. Reason: three large plans (2,800–4,100 lines each) + context budget concerns. Per-task subagent dispatch preserves the orchestrator's context. The user explicitly chose this approach.

## Step 1 — `rune_runner.py --output-json` (do this inline, not via subagent)

**Why:** Plan C Task 2's `run_pipeline_for_problem` shells out to `rune_runner.py --output-json <path>` to capture per-phase artifacts. The flag does not exist yet.

**Change:** Add an `--output-json PATH` argument to `scripts/rune_runner.py`. When set, serialize the return dict of `run_phased_pipeline` (or whatever the top-level pipeline function returns) to that path as JSON. One-line default `None`; conditional `json.dump` at the end of main.

**TDD steps:**

1. Write a test in `tests/test_rune_runner_output_json.py` that invokes `rune_runner.py --output-json /tmp/out.json --task "2+2"` via `subprocess.run`, reads `/tmp/out.json`, and asserts it contains the expected phase keys (`decompose`, `plan`, `code`, `integrate`).
2. Run the test — expect FAIL (flag doesn't exist).
3. Edit `scripts/rune_runner.py`: add argparse arg, serialize with `json.dumps(result, default=str, indent=2)` guarded by `if args.output_json:`.
4. Rerun test — expect PASS.
5. `uv run ruff check scripts/rune_runner.py tests/test_rune_runner_output_json.py`
6. Commit: `feat(runner): --output-json flag for phase corpus producer`

Keep this small. One commit. Then move to Step 2.

## Step 2 — Plan A + Plan B in parallel (subagent-driven)

Invoke `superpowers:subagent-driven-development` for each plan. The skill handles per-task subagent dispatch and two-stage review.

**Plan A dispatch brief (paste into skill invocation):**

```
Execute docs/superpowers/plans/2026-04-22-benchmark-harness-library.md via subagent-driven-development.
Branch: feat/training-upgrade. Use uv run for all Python. Commit per-task
with Conventional Commits + Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>.
5 open questions are flagged in the plan's Self-Review — resolve each the first
time it becomes relevant; do NOT re-open settled decisions.
```

**Plan B dispatch brief:**

```
Execute docs/superpowers/plans/2026-04-22-trajectory-encoder-pretraining.md via
subagent-driven-development. Branch: feat/training-upgrade. Use uv run. Commit
per-task with Conventional Commits + Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>.
HONOR the AMENDMENT 2026-04-22 block at the top — strict task_description field
only, no fallbacks, MIN_RETENTION_RATIO = 0.80.
```

Since the two plans touch disjoint files, you can either:
- Run them serially (simpler context management)
- Run them in parallel with two separate subagent-driven-development invocations (faster wall-clock; more context pressure on orchestrator)

The user said "(A+B) in parallel" so prefer parallel dispatch.

## Step 3 — Plan C (subagent-driven, after Plan A)

```
Execute docs/superpowers/plans/2026-04-22-phase-corpus-producer.md via
subagent-driven-development. Branch: feat/training-upgrade. Plan A is already
landed, so replace the "from evaluation.benchmarks import run_benchmark
# exact module path set by Plan A" comment with the actual import path confirmed
from the landed Plan A code. Commit per-task with Conventional Commits +
Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>.
```

## Hard constraints (do not re-debate)

These are locked by the spec and plan commits. Do NOT re-open in subagent runs:

- **Base model:** `Qwen/Qwen3.5-9B` (DeltaCoder warm-start ancestor; matches the landed reconstruction dataset builder and `libs/model-training/src/model_training/model_configs.py`. Note: `scripts/swarm.py:224` still defaults to `Qwen/Qwen2.5-Coder-7B` — that's the legacy swarm path and is not what this pivot targets.)
- **Warm-start:** DeltaCoder (`danielcherubini/Qwen3.5-DeltaCoder-9B`)
- **Oracle count:** 25 (24 per-(phase,benchmark) + 1 pooled diagnose)
- **Encoder architecture:** shared single encoder, input concatenation, `sentence-transformers/all-mpnet-base-v2` fine-tuned with InfoNCE (option (a))
- **Hypernetwork input dims:** phase_emb[64] + context_emb[768] = 832-dim
- **APPS subsampling:** N=500, seed=42, stratified by difficulty
- **LiveCodeBench:** pin `release_v4`
- **SWE-Bench-Lite:** data-loading + verdict scoring only; repo-checkout preflight is a follow-on plan
- **No fallbacks for `task_description`** (Plan B AMENDMENT block)
- **`diagnose_pooled`** is the literal bin key
- **`MIN_EXAMPLES_PER_BIN = 60`** for rationalization trigger
- **`MIN_RETENTION_RATIO = 0.80`** for Plan B corpus audit
- **Plan C subprocess-mode** invocation of `rune_runner.py` (not in-process)

## Rune conventions (enforce in every subagent)

- `uv run` for all Python (never bare `python`)
- Google-style docstrings
- ruff line-length 88, target py312
- mypy strict-ish
- INFRA-05 deferred GPU imports (torch/transformers/peft imports inside function bodies)
- Conventional Commits with `Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>`
- Tests never require GPU — fixtures + mocks only

## Open questions to watch for (from plan self-reviews)

These are noted in the plans themselves; resolve at the task that first touches them:

**Plan A:**
1. BigCodeBench `v0.1.2` — split name vs config name on HF
2. LiveCodeBench HF repo id may have moved since `livecodebench/code_generation_lite`
3. APPS strict I/O equality may undercount multi-valid-output problems
4. `asyncio` event loop per thread — verify `VLLMProvider` doesn't construct `httpx.AsyncClient` at `__init__` time
5. `datasets>=2.19.0` — reconcile with workspace upper bounds

**Plan C:**
1. APPS stratified sampling delegation to Plan A's `load_problems` — if Plan A doesn't implement stratification, Plan C wraps it
2. New `libs/corpus-producer/` uv-workspace package needs wiring into root `pyproject.toml` members list

## Reference docs

- `docs/superpowers/specs/2026-04-22-phase-benchmark-pivot.yaml` — spec of record
- `docs/superpowers/specs/2026-04-22-pr-28-training-upgrade-fit-assessment.yaml` — upstream assessment
- `docs/superpowers/handoffs/2026-04-22-phase-pivot-handoff.md` — prior session handoff (pre-plans)
- `instructions/Report_2_LoRA_Fine_Tuning_Strategy.md` — kill-switch semantics
- `CLAUDE.md` — Rune conventions

## Definition of done

- Step 1: committed with `feat(runner): --output-json flag …`
- Plan A: 10 task-commits landed; all tests green; `uv run pytest libs/evaluation/ -v` passes
- Plan B: 9 task-commits landed; all tests green; encoder checkpoint producible via CLI
- Plan C: 11 task-commits landed; all tests green; end-to-end integration test passes with mocked `run_benchmark`
- `feat/training-upgrade` is ready for `finishing-a-development-branch` skill

## Session-restart cheat sheet

After `/clear`, your first message should be:

```
Resume plan execution per docs/superpowers/handoffs/2026-04-22-plan-execution-handoff.md.
Branch: feat/training-upgrade. Start with Step 1 (rune_runner.py --output-json
one-liner, inline), then dispatch Plan A and Plan B in parallel via
superpowers:subagent-driven-development, then Plan C after A lands.
```
