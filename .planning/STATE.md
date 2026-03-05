---
gsd_state_version: 1.0
milestone: v5.0
milestone_name: First Implementation
status: executing
stopped_at: "Completed 21-01-PLAN.md"
last_updated: "2026-03-05T20:30:00Z"
last_activity: "2026-03-05 — 21-01 complete: implemented QLoRA training pipeline (peft_utils, config, trainer), train_qlora + train_and_register orchestrators, GPU deps in pyproject.toml, mypy overrides for bitsandbytes/trl; 23 tests pass, 2 xfail"
progress:
  total_phases: 5
  completed_phases: 3
  total_plans: 8
  completed_plans: 8
  percent: 12
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-05)

**Core value:** A local coding agent that learns from its own coding trajectories, building persistent parametric memory that scales independently of context window size.
**Current focus:** Phase 21 — QLoRA Training Pipeline (Plan 01 complete)

## Current Position

Phase: 21 of 22 (QLoRA Training Pipeline)
Plan: 01 complete (21-01-PLAN.md done)
Status: In progress — 21-01 complete, ready for Phase 21 next plans or Phase 22
Last activity: 2026-03-05 — 21-01 complete: implemented QLoRA training pipeline (peft_utils, config, trainer), train_qlora + train_and_register orchestrators, GPU deps in pyproject.toml, mypy overrides for bitsandbytes/trl; 23 tests pass, 2 xfail

Progress: [███░░░░░░░] 12%

## Performance Metrics

**Velocity:**
- Total plans completed: 8 (v5.0)
- Average duration: ~10 min
- Total execution time: ~42 min

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 18-adapter-registry | 2 | 17 min | 8.5 min |
| 19-inference-provider-abstraction | 3 | 13 min | 4.3 min |
| 20-agent-loop | 2 | 10 min | 5 min |
| 21-qlora-training-pipeline | 1 | 25 min | 25 min |

*Updated after each plan completion*
| Phase 19-inference-provider-abstraction P01 | 18 | 3 tasks | 8 files |
| Phase 19-inference-provider-abstraction P03 | 5 | 2 tasks | 10 files |
| Phase 20-agent-loop P01 | 3 | 2 tasks | 5 files |
| Phase 20-agent-loop P02 | 7 | 2 tasks | 4 files |
| Phase 21-qlora-training-pipeline P01 | 25 | 2 tasks | 9 files |

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
- [Phase 21-01]: bfloat16 compute dtype set in BitsAndBytesConfig, NOT LoraConfig — LoraConfig is purely about LoRA layer structure
- [Phase 21-01]: No modules_to_save in LoraConfig — would break vLLM adapter loading (include embed_tokens/lm_head in saved PEFT artifact)
- [Phase 21-01]: sys.modules injection pattern for mocking deferred GPU imports in CPU CI — unittest.mock.patch("peft.X") fails when peft is not installed
- [Phase 21-01]: RUNE_ADAPTER_DIR/RUNE_BASE_MODEL/RUNE_DATABASE_URL env vars read inside function bodies for monkeypatch testability

### Pending Todos

None.

### Blockers/Concerns

- Phase 19: PP=2 + LoRA compatibility with Qwen2.5-Coder-7B is unverified empirically; fallback is PP=1/TP=1 (single GPU, 24GB VRAM)
- Phase 22: Sakana AI Doc-to-LoRA pre-trained weight availability and licensing unconfirmed as of 2026-03-05; plan-phase must verify before implementation

## Session Continuity

Last session: 2026-03-05T20:30:00Z
Stopped at: Completed 21-01-PLAN.md
Resume file: .planning/phases/21-qlora-training-pipeline/21-01-SUMMARY.md
