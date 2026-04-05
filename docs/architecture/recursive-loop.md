# 5-Phase Coding Pipeline

## Overview

Rune's execution model is a 5-phase sequential pipeline: decompose, plan, code, integrate, and diagnose/repair. Phases 2 and 3 run as parallel swarms; phases 1 and 4 run single-agent; phase 5 activates conditionally when code fails. Within each phase, a generate-execute-reflect loop iterates until tests pass or max attempts are reached. Each phase is driven by a Jinja2 template, optionally enhanced by a hypernetwork-generated LoRA adapter.

For swarm orchestration details, see [Swarm Architecture](../swarm-architecture.md). For adapter storage, see [Adapter Storage](adapter-storage.md).

---

## Pipeline Architecture

```mermaid
flowchart LR
    Task([Task]) --> P1[Phase 1: DECOMPOSE\nsingle agent]
    P1 --> P2[Phase 2: PLAN\nparallel swarm]
    P2 --> P3[Phase 3: CODE\nparallel swarm + retries]
    P3 --> P4[Phase 4: INTEGRATE\nsingle agent]
    P4 --> P5{Phase 5: DIAGNOSE/REPAIR\nconditional}
    P5 -->|failure| P5
    P5 -->|success| Done([Integrated Code\n+ Adapters])
```

---

## Phase Details

### Phase 1: DECOMPOSE (single agent)

| Field | Description |
|-------|-------------|
| **Template** | `decompose.j2` (trajectory), `prompt_decompose.j2` (prompt) |
| **Input** | Project specification |
| **Output** | List of subtasks |
| **Adapter** | `decompose_adapter` (if available) |

Breaks a project specification into discrete subtasks. Runs once with the base model (+ optional adapter).

### Phase 2: PLAN (parallel swarm)

| Field | Description |
|-------|-------------|
| **Template** | `plan.j2` (trajectory), `prompt_plan.j2` (prompt) |
| **Input** | Project spec + one subtask per agent |
| **Output** | Architecture plan per subtask |
| **Adapter** | `plan_adapter` (if available) |

Each subtask is assigned to a swarm agent. All agents run in parallel. Each produces an architecture plan for its subtask.

### Phase 3: CODE (parallel swarm + retries)

| Field | Description |
|-------|-------------|
| **Template** | `code.j2` (first attempt), `code_retry.j2` (retries) |
| **Input** | Subtask + plan + prior code skeleton |
| **Output** | Working code per subtask |
| **Adapter** | `code_adapter` (if available), refreshed via H() between iterations |

The retry template (`code_retry.j2`) includes error summaries and attempt history. Between iterations, the hypernetwork H() can generate a fresh adapter from the accumulated trajectory. This is the primary phase where the recursive loop operates.

### Phase 4: INTEGRATE (single agent)

| Field | Description |
|-------|-------------|
| **Template** | `integrate.j2` (trajectory), `prompt_integrate.j2` (prompt) |
| **Input** | Project spec + all code outputs from Phase 3 |
| **Output** | Final integrated codebase |
| **Adapter** | `integrate_adapter` (if available) |

Merges all subtask code outputs into a coherent final codebase.

### Phase 5: DIAGNOSE/REPAIR (conditional)

| Field | Description |
|-------|-------------|
| **Template** | `diagnose.j2` (trajectory), `prompt_diagnose.j2` (prompt), `code_repair.j2` (trajectory), `prompt_code_repair.j2` (prompt) |
| **Input** | Failed code + error output from Phase 3 or Phase 4 |
| **Output** | Fixed code per subtask |
| **Adapter** | Domain adapter stays loaded; error context flows through prompt |

Activates when code fails during the retry loop. Uses a two-step pattern to avoid prompt-adapter tension where domain context and error details compete for model attention:

1. **Diagnose:** The error is placed in the prompt and the original code in the adapter trajectory. The model produces a concise fix instruction describing what went wrong and how to fix it.
2. **Repair:** The model's own diagnosis becomes the `fix_guidance` in the prompt, while domain context stays in the adapter. The model produces fixed code guided by its own analysis.

This separation keeps each inference call focused: diagnose concentrates on understanding the failure, repair concentrates on applying the fix with full domain context.

---

## Per-Phase Iteration

Each phase runs up to N iterations (configurable per phase via environment variables):

| Variable | Scope |
|----------|-------|
| `RUNE_MAX_PHASE_ITERATIONS` | Global default for all phases |
| `RUNE_MAX_ITERATIONS_DECOMPOSE` | Phase 1 override |
| `RUNE_MAX_ITERATIONS_PLAN` | Phase 2 override |
| `RUNE_MAX_ITERATIONS_CODE` | Phase 3 override |
| `RUNE_MAX_ITERATIONS_INTEGRATE` | Phase 4 override |
| `RUNE_MAX_ITERATIONS_REPAIR` | Phase 5 override (covers diagnose + repair + re-integrate steps) |

CLI flag `--max-phase-iterations` overrides the global env var. Hardcoded fallback is 5.

Within each iteration:
1. Render the Jinja2 trajectory template with current state
2. Optionally generate/refresh adapter via hypernetwork H()
3. Run inference (base model + adapter) to produce output
4. Execute output in sandbox, evaluate results
5. Score fitness; if passing or max iterations reached, stop

---

## Template System

All phase instructions flow through 18 Jinja2 templates in `libs/shared/src/shared/templates/`:

| Template | Phase | Purpose |
|----------|-------|---------|
| `decompose.j2` | 1 | Trajectory context for decomposition |
| `prompt_decompose.j2` | 1 | Model prompt for decomposition |
| `prompt_decompose_concise.j2` | 1 | Concise model prompt variant for decomposition |
| `plan.j2` | 2 | Trajectory context for planning |
| `prompt_plan.j2` | 2 | Model prompt for planning |
| `code.j2` | 3 | Trajectory context for first code attempt |
| `code_continue.j2` | 3 | Trajectory context for continuation |
| `code_retry.j2` | 3 | Trajectory context for retry (includes errors, history) |
| `prompt_code.j2` | 3 | Model prompt for code generation |
| `prompt_code_continue.j2` | 3 | Model prompt for code continuation |
| `prompt_code_retry.j2` | 3 | Model prompt for code retry |
| `integrate.j2` | 4 | Trajectory context for integration |
| `prompt_integrate.j2` | 4 | Model prompt for integration |
| `prompt_integrate_retry.j2` | 4 | Model prompt for integration retry |
| `diagnose.j2` | 5 | Trajectory context for failure diagnosis |
| `prompt_diagnose.j2` | 5 | Model prompt for diagnosis (error in prompt, code in adapter) |
| `code_repair.j2` | 5 | Trajectory context for targeted repair |
| `prompt_code_repair.j2` | 5 | Model prompt for repair (diagnosis becomes fix_guidance) |

Templates are rendered via `shared.template_loader.render_trajectory()` and `render_prompt()`.

---

## Sandbox

Code execution uses `shared.sandbox.SubprocessBackend` — a subprocess-based sandbox with configurable timeout. The sandbox runs agent-generated code in isolation and captures stdout, stderr, and exit code.

---

## Trajectory Schema

The trajectory flowing through the pipeline maps to `CodingSession` from `shared.rune_models`:

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | str | Unique session identifier |
| `task_description` | str | Human-readable task description |
| `task_type` | str | Task category (e.g. 'bug-fix', 'feature-impl') |
| `adapter_refs` | list[AdapterRef] | Adapters loaded during this session |
| `attempt_count` | int | Number of generate-execute-reflect cycles |
| `outcome` | str or None | 'success', 'exhausted', or None if in progress |

---

## Integration Points

- **Adapter Registry** ([Adapter Storage](adapter-storage.md)): Queried at phase start for adapter selection; written to after adapter generation
- **Inference Providers** (`libs/inference`): TransformersProvider, LlamaCppProvider, OllamaProvider, or VLLMProvider — selected via factory
- **Sandbox** (`shared.sandbox.SubprocessBackend`): Executes generated code
- **Hypernetwork** (`model_training.hypernetwork.DocToLoraHypernetwork`): Generates adapters from trajectories
- **Swarm** (`scripts/swarm.py`): Orchestrates parallel execution of Phases 2 and 3
