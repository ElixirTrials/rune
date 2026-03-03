# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-02)

**Core value:** A local coding agent that learns from its own coding trajectories, building persistent parametric memory that scales independently of context window size.
**Current focus:** Milestone v2.0 — Phase 6 in progress (Service Scaffolds)

## Current Position

Phase: 6 of 7 (Service Scaffolds)
Plan: 3 of 4 in current phase (plans 01-03 complete)
Status: Phase 6 in progress
Last activity: 2026-03-03 — Phase 6 Plan 03: training-svc and evolution-svc scaffolds with 7 stub 501 endpoints

Progress: [████████░░] 82% (v1.0 complete; v2.0 phases 4-6.03 complete)

## Performance Metrics

**Velocity:**
- Total plans completed: 15
- Average duration: 3.1 min
- Total execution time: 0.83 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-readme | 1 | 6 min | 6 min |
| 02-implementation-plan | 2 | 6 min | 3 min |
| 03-architecture-docs | 1 | ~5 min | ~5 min |
| 04-cleanup | 3 | 11 min | 3.7 min |
| 05-foundation-libraries | 3 | 8 min | 2.7 min |
| 05.1-template-artifact-cleanup | 2 | 7 min | 3.5 min |
| 06-service-scaffolds | 3 | 7 min | 2.3 min |

**Recent Trend:**
- Last 5 plans: 4 min, 3 min, 2 min, 2 min, 3 min
- Trend: Stable

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
- [Phase 06-service-scaffolds]: lora-server is Dockerfile-only with no pyproject.toml, not in uv workspace members
- [Phase 06-service-scaffolds]: LoraServerConfig raises ValueError on TP=2 referencing vLLM bug #21471
- [Phase 06-service-scaffolds]: VLLMClient wraps AsyncOpenAI, not direct vllm import
- [Phase 06-service-scaffolds]: Health sidecar uses httpx for /ready check against vLLM on port 8000
- [Phase 06-service-scaffolds]: RuneState uses plain TypedDict without Annotated[..., add_messages] -- trajectory managed explicitly by nodes
- [Phase 06-service-scaffolds]: should_retry is fully implemented (not stubbed) with 3-way branching: tests_passed, attempts exhausted, retry
- [Phase 06-service-scaffolds]: Used APIRouter with no prefix and full endpoint paths to avoid nesting issues
- [Phase 06-service-scaffolds]: Followed api-service storage.py pattern with SQLite default and check_same_thread=False

### Pending Todos

None.

### Blockers/Concerns

None yet.

## Session Continuity

Last session: 2026-03-03
Stopped at: Completed 06-03-PLAN.md (Training & Evolution Service Scaffolds)
Resume file: None
