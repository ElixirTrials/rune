---
gsd_state_version: 1.0
milestone: v5.0
milestone_name: First Implementation
status: executing
stopped_at: Completed 20-02-PLAN.md
last_updated: "2026-03-05T16:11:55.454Z"
last_activity: "2026-03-05 — 20-02 complete: implemented all 4 node functions (generate_node, execute_node, reflect_node, save_trajectory_node), 11 behavior tests green, mypy+ruff clean, py.typed markers added to inference+model-training libs; 30 combined tests passing"
progress:
  total_phases: 5
  completed_phases: 3
  total_plans: 7
  completed_plans: 7
  percent: 10
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-05)

**Core value:** A local coding agent that learns from its own coding trajectories, building persistent parametric memory that scales independently of context window size.
**Current focus:** Phase 20 — Agent Loop (Plan 02 complete — phase done)

## Current Position

Phase: 20 of 22 (Agent Loop)
Plan: 02 complete (20-02-PLAN.md done — all Phase 20 plans complete)
Status: In progress — Phase 20 complete, ready for Phase 21
Last activity: 2026-03-05 — 20-02 complete: implemented all 4 node functions (generate_node, execute_node, reflect_node, save_trajectory_node), 11 behavior tests green, mypy+ruff clean, py.typed markers added to inference+model-training libs; 30 combined tests passing

Progress: [██░░░░░░░░] 10%

## Performance Metrics

**Velocity:**
- Total plans completed: 2 (v5.0)
- Average duration: 8.5 min
- Total execution time: 17 min

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 18-adapter-registry | 2 | 17 min | 8.5 min |
| 19-inference-provider-abstraction | 3 | 13 min | 4.3 min |

*Updated after each plan completion*
| Phase 19-inference-provider-abstraction P01 | 18 | 3 tasks | 8 files |
| Phase 19-inference-provider-abstraction P03 | 5 | 2 tasks | 10 files |
| Phase 20-agent-loop P01 | 3 | 2 tasks | 5 files |
| Phase 20-agent-loop P02 | 7 | 2 tasks | 4 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting v5.0:

- vllm/vllm-openai:v0.16.0 base image used; openai removed from pip install (bundled in vLLM image); VLLM_ALLOW_RUNTIME_LORA_UPDATING=True enables runtime adapter loading
- lora-server host port changed from 8000 to 8100 in docker-compose; container-internal port stays 8000 (startup.sh unchanged)
- rune_data:/data shared volume added to api and lora-server in docker-compose for cross-service SQLite access
- Hardware validation deferred — user will validate hardware separately; not a roadmap phase
- Provider abstraction added — InferenceProvider interface (INF-01) with VLLMProvider (INF-02), OllamaProvider (INF-03), factory (INF-04), per-step model/provider config (INF-07)
- AGENT-01 is backend-agnostic — calls InferenceProvider.generate(), not VLLMClient directly
- Build order: adapter-registry → inference abstraction + providers → agent loop + trajectory → QLoRA training → kill-switch gate
- INFRA-01/02/03 in Phase 19 (lora-server config + port fix); INFRA-04/05 in Phase 21 (GPU deps)
- evolution-svc explicitly out of scope for v5.0 (deferred to v6+)
- bfloat16 hardcoded in build_qlora_config (float16 causes silent NaN loss at 7B scale)
- max_loras=2, max_lora_rank=64, gpu_memory_utilization=0.80 to avoid VRAM OOM at startup
- Engine from sqlalchemy.engine not sqlmodel — SQLModel does not re-export Engine (discovered 18-01, confirmed 18-02)
- Session-per-method pattern with expire_on_commit=False + expunge() — required to prevent DetachedInstanceError when returning records from closed sessions (established 18-01)
- WAL test must use file-based engine (tmp_path), not :memory: — WAL has no practical effect on in-memory SQLite
- memory_engine fixture: create_engine("sqlite:///:memory:") with function scope gives per-test isolation at zero cleanup cost
- [Phase 19-01]: asyncio_mode=auto added to inference lib pyproject.toml — lib-scoped pytest overrides root config
- [Phase 19-01]: VLLMProvider.generate() passes adapter_id as model param — vLLM LoRA routing mechanism requires lora_name in the model field
- [Phase 19-01]: VLLM_BASE_URL default set to port 8100 (api-service owns port 8000)
- [Phase 19-03]: os.environ.get() used in factory.py instead of os.getenv() — mypy cannot narrow os.getenv(key, str_default) return type to str; explicit variable annotations resolve the issue
- [Phase 19-03]: Env var reads placed inside get_provider() function body (not module level) — allows monkeypatch.setenv() to work in tests; module-level reads captured at import time
- [Phase 20-agent-loop]: RUNE_TRAJECTORY_DIR read inside function body for monkeypatch testability (same pattern as Phase 19 factory.py)
- [Phase 20-agent-loop]: record_trajectory() extended with keyword-only task_description/task_type/adapter_ids args for save_trajectory_node compatibility
- [Phase 20-agent-loop]: format_for_sft() uses reversed(steps) to find last tests_passed=True step as assistant content
- [Phase 20-02]: RUNE_MODEL and RUNE_EXEC_TIMEOUT env vars read inside function bodies for monkeypatch testability — same pattern as Phase 19 factory.py
- [Phase 20-02]: py.typed markers added to inference and model-training libs — correct PEP 561 solution for mypy strict import-untyped compliance
- [Phase 20-02]: reflect_node uses list concatenation (state["trajectory"] + [step]) not .append() — LangGraph requires immutable state updates

### Pending Todos

None.

### Blockers/Concerns

- Phase 19: PP=2 + LoRA compatibility with Qwen2.5-Coder-7B is unverified empirically; fallback is PP=1/TP=1 (single GPU, 24GB VRAM)
- Phase 22: Sakana AI Doc-to-LoRA pre-trained weight availability and licensing unconfirmed as of 2026-03-05; plan-phase must verify before implementation

## Session Continuity

Last session: 2026-03-05T17:10:00.000Z
Stopped at: Completed 20-02-PLAN.md
Resume file: None
