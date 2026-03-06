---
gsd_state_version: 1.0
milestone: v5.0
milestone_name: First Implementation
status: executing
stopped_at: Completed 23-01-PLAN.md
last_updated: "2026-03-06T14:37:22Z"
last_activity: "2026-03-06 - Completed 23-01: Wire AdapterRegistry.store() into _run_hypernetwork_job, fix mypy model_training overrides, add integration test"
progress:
  total_phases: 6
  completed_phases: 5
  total_plans: 13
  completed_plans: 13
  percent: 92
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-05)

**Core value:** A local coding agent that learns from its own coding trajectories, building persistent parametric memory that scales independently of context window size.
**Current focus:** Phase 23 — Integration Fix & Quality Gate (Plan 01 complete)

## Current Position

Phase: 23 of 23 (Integration Fix & Quality Gate)
Plan: 01 complete (23-01-PLAN.md done) — hypernetwork registry wiring + mypy fix
Status: In progress — Phase 23 Plan 01 complete
Last activity: 2026-03-06 - Completed 23-01: Wire AdapterRegistry.store() into _run_hypernetwork_job, fix mypy model_training overrides, add integration test

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
| 21-qlora-training-pipeline | 2 | 37 min | 18.5 min |
| 22-kill-switch-gate | 1 | ~18 min | ~18 min |

*Updated after each plan completion*
| Phase 19-inference-provider-abstraction P01 | 18 | 3 tasks | 8 files |
| Phase 19-inference-provider-abstraction P03 | 5 | 2 tasks | 10 files |
| Phase 20-agent-loop P01 | 3 | 2 tasks | 5 files |
| Phase 20-agent-loop P02 | 7 | 2 tasks | 4 files |
| Phase 21-qlora-training-pipeline P01 | 25 | 2 tasks | 9 files |
| Phase 21-qlora-training-pipeline P02 | 12 | 2 tasks | 5 files |
| Phase 22-kill-switch-gate P01 | ~18 | 1 task | 3 files |
| Phase 22-kill-switch-gate P02 | 12 | 2 tasks | 5 files |
| Phase 22-kill-switch-gate P03 | 8 | 1 tasks | 2 files |
| Phase 23-integration-fix-quality-gate P01 | 6 | 2 tasks | 3 files |

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
- [Phase 21-02]: Deferred import of model_training.trainer inside _run_training_job body — prevents GPU lib import at startup in CPU-only environments (same INFRA-05 pattern)
- [Phase 21-02]: JOB_STORE as module-level dict — state is lost on restart, acceptable for single-user local MVP
- [Phase 21-02]: _run_training_job is a regular (non-async) function — FastAPI BackgroundTasks runs it in thread pool executor
- [Phase 21-02]: Mock pattern: patch training_svc.routers.training._run_training_job in tests to prevent model_training.trainer GPU import
- [Phase 22-01]: DocToLoraHypernetwork uses _LazyHypernetworkProxy — real nn.Module subclass built on first instantiation to keep module importable without torch (INFRA-05)
- [Phase 22-01]: save_hypernetwork_adapter: modules_to_save=None — vLLM rejects embed_tokens/lm_head in adapter artifacts (consistent with Phase 21-01 decision)
- [Phase 22-01]: Torch tests use hidden_dim=32, num_layers=1-2 to keep CPU CI fast — large linear weight_head (8192 x 3.67M) hangs indefinitely on CPU with default 7B params
- [Phase 22-kill-switch-gate]: run_humaneval_subset accepts Optional completions dict — None raises NotImplementedError; inference wiring deferred to orchestration level
- [Phase 22-kill-switch-gate]: data/ directory un-ignored via local .gitignore in evaluation/src/evaluation/ — root .gitignore has global data/ exclusion
- [Phase 22-kill-switch-gate]: sys.executable used in subprocess (not hardcoded python) — matches current venv Python in all execution environments
- [Phase 22-kill-switch-gate]: _run_hypernetwork_job uses first trajectory_id from request.trajectory_ids — single trajectory drives the hypernetwork forward pass
- [Phase 23-01]: Use training_svc.storage.engine (not RUNE_DATABASE_URL) in _run_hypernetwork_job — service-level function shares service DB, ensures QLoRA and hypernetwork adapters in same database
- [Phase 23-01]: model_training added to workspace-packages mypy override block (not GPU libs block) — it is a local workspace package, not a GPU-only library
- [Phase 23-01]: All new imports deferred inside _run_hypernetwork_job function body per INFRA-05 — allows patch("training_svc.storage.engine", ...) in tests

### Pending Todos

None.

### Blockers/Concerns

- Phase 22: Sakana AI Doc-to-LoRA pre-trained weight availability and licensing unconfirmed as of 2026-03-05; plan-phase must verify before implementation

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 1 | Remove all hardware requirements - make repo hardware-agnostic for any local setup | 2026-03-06 | 9ebaeb4 | [1-remove-all-hardware-requirements-make-re](./quick/1-remove-all-hardware-requirements-make-re/) |

## Session Continuity

Last session: 2026-03-06T14:37:22Z
Stopped at: Completed 23-01-PLAN.md
Resume file: None
