# Episodic conditioning + bounded decomposition — design

**Date:** 2026-06-05 · **Issue:** #52 (adapter-as-memory) · **Branch:** issue52-bf16-body-contrastive

## Motivation

Diagnosis (`docs/issue52-goal3-hardtask-failure-2026-06-05.md`) showed the adapter
carries the full spec fine (+0.25 single-turn on easy tasks) but gives **no gain on hard
tasks**, for engine-structural reasons — not adapter capacity:

- The **plan prompt leaks the full spec** (`project_label`); the model is never focused on
  one sub-goal.
- The engine **stops at the single public example** (oracle) and ships code wrong on
  held-out cases.
- **JSON `decompose`/`plan`/`diagnose` truncate** on long hard-task output → re-plan loops →
  empty code (3/4 empty on the LiveCodeBench probe).
- **Over-decomposition** (`calculate` → 2 subtasks that never define the entry_point).

**Hypothesis (owner):** the multi-turn advantage is *focus* — point the model at the
specific step it must accomplish, not the whole goal. Put the goal + running state in an
**episodic adapter** that brings the *right context for the current step*; keep the prompt
thin. Each subtask becomes a full development cycle around its sub-goal.

## Core idea: invert prompt and adapter roles

| | today | this design |
|---|---|---|
| **prompt** | leaks full spec (plan) / mission ref (code) | **thin pointer to the current step's sub-goal** only |
| **adapter** | full spec at every step (redundant) | **episodic memory** — the *right context for this step* |

The full spec never appears verbatim in a prompt again.

## Components

### 1. Decompose — model-decides, bounded & verified (agent instruction, NOT regex)
- A strengthened **agent instruction** in the decompose prompt: default to **ONE subtask**
  for a single self-contained function; split only for genuinely separable components with
  dependencies; **cap at 3**. Each subtask must carry: `name`, `description`, a concrete
  **`acceptance_check`** (an example I/O or assert for that sub-goal), and **`builds`** (the
  named piece of the final `entry_point` it contributes). Decompose also emits a condensed
  **`overall_goal`** (one/two lines) for the episodic adapter.
- **Structural verification (this part is code, not the model's word):** at integration,
  AST-check that the `entry_point` is actually defined. The *count* and *content* are the
  model's call (per owner: "model decides, bounded + verified"); the *guard* is the
  instruction + the AST check, never a fragile regex.
- Single-function task ⇒ N=1: the sub-goal = the whole task, its check = the public
  example(s).
- Schema: `SubtaskSchema` gains `acceptance_check: str`, `builds: str`; `DecomposeResult`
  gains `overall_goal: str`.

### 2. Per-subtask dev cycle — focused prompt + episodic adapter
- **Prompt (thin):** "Implement `<sub-goal name>`: <one-line sub-goal>." / "Repair
  `<sub-goal>`." No full spec.
- **Adapter (episodic, context-appropriate)** — new `render_episode_adapter(step, ...)`:
  - *code / repair:* condensed `overall_goal` + the **current** sub-goal (`description` +
    `acceptance_check`) + current code + error. (Local episode — "focus the model down".)
  - *integration:* `overall_goal` + **all** subtasks' accepted code/results + integration
    error. ("Bring the right context to bear.")
- **Per-subtask signal (BOTH, per owner):**
  1. the sub-goal's **executable `acceptance_check`** (hard gate, via the oracle path), and
  2. the **fixed reason-first judge** (`JudgeResult` reordered) vs the sub-goal description.
  Either failing → repair; the adapter carries that failure episode forward.

### 3. Integration
Episodic adapter carries all subtasks' code; verify `entry_point` defined (AST) and passes
**all** the task's public examples (see #2 of roadmap).

### 4. Robust structured parsing (folded in — owner-approved)
The episodic `decompose` emits *more* structured output (subtasks + checks + overall_goal),
so it is *more* exposed to the truncation/re-plan loop. Fold the robustness fix in:
- Route `decompose`/`plan`/`diagnose` parsing through **`json_repair`** (as code-parse does)
  before pydantic validation; raise their `max_tokens`; on unrecoverable parse, degrade
  gracefully (e.g. fall back to N=1 whole-task) instead of re-plan-looping to empty.

## Data flow

```
decompose (robust parse) -> {overall_goal, subtasks:[{name, description, acceptance_check, builds}]}
  for each subtask (focused prompt + episodic adapter = overall_goal + this sub-goal + local code/err):
     code -> run acceptance_check (executable) AND judge(sub-goal) -> fail? repair (episode grows)
  integrate (episodic adapter = overall_goal + ALL subtasks) -> AST-verify entry_point + run all public examples
```

## Files touched
- `engine/parse.py` — schema fields; `json_repair` path for decompose/plan/diagnose; `JudgeResult` already reordered.
- `engine/graph.py` — `render_episode_adapter`; `state_to_ctx` stops leaking full spec into prompts; thin prompt rendering.
- `engine/policy.py` — decompose count/verify wiring (uses the instruction; AST verify hook).
- `engine/oracle.py` — per-subtask `acceptance_check` execution; integration runs ALL public examples.
- `templates/` — new thin focused `prompt_code`/`prompt_code_repair`; strengthened `prompt_decompose` (bounding instruction); `prompt_judge` (per sub-goal).
- `config.py` — `max_tokens` for plan/decompose; scaling stays tunable (roadmap #3).

## Roadmap (post-design, owner-sequenced)
1. **(folded in)** robust decompose/plan/diagnose parsing + decompose-guard instruction.
2. **Stronger in-loop signal:** feed **all** public examples (LCB problems have several), not just the first.
3. **Re-tune `adapter_scaling`** for hard tasks (sweep below 0.627).
4. Decompose guard — already the agent instruction in #1 (no regex).

## Verification (owner: smoke/probe each, report, record to scratchpad)
- **Unit (CPU, first):** robust parsing survives a truncated/garbled decompose/plan JSON;
  `render_episode_adapter` emits the right context per step (local for code, all-subtasks for
  integrate); decompose bound (≤3) + AST entry_point verification; oracle runs all public
  examples + per-subtask checks.
- **Tiny live slice (then):** 2 single-function tasks (expect N=1, clean cycle) + 1 genuinely
  multi-component task (expect N>1, per-subtask cycles + integration) — confirm end-to-end
  generation works (no empty code / re-plan loops) before any larger run or HPO.
- Each step reported and recorded to scratchpad.

## Out of scope (later)
- **Adapter mutation vs hotswap** (owner: after this hypothesis).
- The full LiveCodeBench run / HPO (gated on this redesign + #2/#3).

## Success criteria
- No empty-code/re-plan loops on hard/LCB tasks (robust parsing).
- Episodic conditioning renders the intended context per step (verified).
- On a hard slice, repair engages on real (executable + judge) signals; the held-out gap
  shrinks vs the oracle-stops-early baseline. This design targets the *mechanism*; the
  effect size is the empirical question the validation slice (then a re-run) answers — it is
  not pre-committed.
