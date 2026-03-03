# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-02)

**Core value:** A local coding agent that learns from its own coding trajectories, building persistent parametric memory that scales independently of context window size.
**Current focus:** Milestone v2.0 — Phase 5.1 complete, ready for Phase 6

## Current Position

Phase: 5.1 of 7 (Template Artifact Cleanup) -- COMPLETE
Plan: 2 of 2 in current phase (all complete)
Status: Phase 5.1 complete, ready for Phase 6
Last activity: 2026-03-03 — Phase 5.1 Plan 02: template docs deleted, ElixirTrials references removed

Progress: [███████░░░] 69% (v1.0 complete; v2.0 phases 4-5.1 complete)

## Performance Metrics

**Velocity:**
- Total plans completed: 12
- Average duration: 3.4 min
- Total execution time: 0.72 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-readme | 1 | 6 min | 6 min |
| 02-implementation-plan | 2 | 6 min | 3 min |
| 03-architecture-docs | 1 | ~5 min | ~5 min |
| 04-cleanup | 3 | 11 min | 3.7 min |
| 05-foundation-libraries | 3 | 8 min | 2.7 min |
| 05.1-template-artifact-cleanup | 2 | 7 min | 3.5 min |

**Recent Trend:**
- Last 5 plans: 3 min, 3 min, 2 min, 4 min, 3 min
- Trend: Improving

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Milestone v1.0]: Documentation-first milestone — 3 phases (README, Implementation Plan, Architecture Docs) — COMPLETE
- [Milestone v2.0]: Restructure monorepo to match implementation plan component layout before writing logic
- [Roadmap v2.0]: 4 phases (4-7): Cleanup → Foundation Libraries → Service Scaffolds → Config & Quality Gate
- [Phase 4]: Remove agent-b-service before any other workspace changes to avoid lockfile conflicts
- [Phase 5]: adapter-registry is the hard gate — no service can be scaffolded until it exists and is importable
- [Phase 6]: lora-server is Dockerfile-only; do NOT add to uv workspace members
- [Phase 6]: LoraServerConfig must raise ValueError on tensor_parallel_size=2 (vLLM bug #21471)
- [Phase 7]: QA-01 through QA-04 are a terminal validation gate; all must pass before v2.0 is complete
- [Phase 04-cleanup]: Removed agent-b-service from Makefile typecheck target (not in plan but required for correctness)
- [Phase 04-cleanup]: Renamed agent-a-service to rune-agent; also updated Makefile and copilot-instructions glob patterns (not in plan but contained stale references)
- [Phase 04-cleanup]: Added Python-file filter to mypy glob pattern to skip TypeScript directories (events-ts, shared-ts)
- [Phase 04-cleanup]: Kept torch/transformers/peft in mypy ignore_missing_imports for future Phase 5 model-training lib
- [Phase 05-foundation-libraries]: Used explicit __tablename__ = "adapter_records" to avoid collision with shared.models Entity/Task tables
- [Phase 05-foundation-libraries]: Used openai AsyncOpenAI with custom base_url for vLLM (not direct vllm import)
- [Phase 05-foundation-libraries]: All GPU imports deferred behind TYPE_CHECKING guards in peft_utils.py
- [Phase 05.1-cleanup]: Removed apps/ directory entirely since hitl-ui was the only app
- [Phase 05.1-cleanup]: Removed apps/**/docs/** filter from CI since apps/ no longer exists
- [Phase 05.1-cleanup]: Updated Makefile help text to reflect Python-only tooling (no more tsc/vitest)
- [Phase 05.1-cleanup]: Removed repo_name/repo_url/edit_uri from mkdocs.yml (no public repo yet)

### Pending Todos

None.

### Blockers/Concerns

None yet.

## Session Continuity

Last session: 2026-03-03
Stopped at: Completed 05.1-02-PLAN.md (Phase 5.1 Template Artifact Cleanup - all plans complete)
Resume file: None
