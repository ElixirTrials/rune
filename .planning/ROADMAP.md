# Roadmap: Rune

## Milestones

- [x] **v1.0 Documentation & Implementation Plan** - Phases 1-3 (shipped 2026-03-02)
- [ ] **v2.0 Repo Restructuring & Scaffold** - Phases 4-7 (in progress)

## Phases

<details>
<summary>v1.0 Documentation & Implementation Plan (Phases 1-3) - COMPLETE 2026-03-02</summary>

### Phase 1: README
**Goal**: A polished README.md exists that anyone can read to understand what Rune is, how it works, what hardware it needs, where the project stands, and which papers it builds on
**Depends on**: Nothing (first phase)
**Requirements**: DOC-01, DOC-02, DOC-03, DOC-04, DOC-05
**Success Criteria** (what must be TRUE):
  1. A reader new to Rune can explain the core innovation (parametric episodic memory via LoRA) and how it differs from Aider, Cursor, and Claude Code after reading the README
  2. The architecture overview section describes the recursive loop, hypernetwork, evolution operator, and adapter serving — with a diagram or structured visual representation of data flow
  3. The hardware requirements section lists specific prerequisites (dual RTX 4090, CUDA 12.8+, PyTorch nightly cu128, QLoRA) so a developer knows before cloning whether they can run Rune
  4. The current status section honestly states that no implementation exists, this is a research-stage project, and the core hypothesis (Doc-to-LoRA on coding trajectories) is unvalidated
  5. The references section cites Doc-to-LoRA (arXiv:2602.15902), S-LoRA (arXiv:2311.03285), and QLoRA (arXiv:2305.14314) with links
**Plans**: 1/1 plans complete

Plans:
- [x] 01-01-PLAN.md — Write complete README.md (vision, architecture diagrams, hardware, status, references)

### Phase 2: Implementation Plan
**Goal**: A detailed, phased implementation plan document exists that transforms the Rune architecture specification into actionable phases with pass/fail criteria, risk gates, and a concrete build order grounded in hardware reality
**Depends on**: Phase 1
**Requirements**: PLAN-01, PLAN-02, PLAN-03, PLAN-04, PLAN-05, PLAN-06
**Success Criteria** (what must be TRUE):
  1. Phase 0 of the implementation plan defines hardware and environment validation with specific pass/fail criteria (both GPUs recognized, PyTorch nightly cu128 forward+backward pass without segfault, vLLM serving Qwen2.5-Coder-7B via pipeline parallelism without TP+LoRA corruption)
  2. Phase 1 of the implementation plan is structured as a kill-switch gate: it defines a measurable success threshold (at minimum, 5% Pass@1 improvement on held-out HumanEval tasks) and states explicitly that no infrastructure is built if this gate fails
  3. Every subsequent phase in the plan states its deliverables, which previous phase it depends on, and the success criteria that mark it complete
  4. The plan explicitly addresses the three primary research risks: adapter composition interference (heterogeneous subspaces), hypernetwork mode collapse (mean-adapter collapse), and catastrophic forgetting (non-immutable adapter writes) — each with a mitigation strategy
  5. The plan reflects the hardware reality: dual RTX 4090 with CXL, pipeline parallelism (PP=2) not tensor parallelism, QLoRA as required for 24GB per-GPU constraint, with QLoRA introduced only after Phase 1 bfloat16 baseline passes
  6. The plan includes the recommended build order: adapter-registry → lora-server → model-training extensions → api-service extensions → rune-agent → evolution-svc → training-svc → hypernetwork
**Plans**: 2/2 plans complete

Plans:
- [x] 02-01-PLAN.md — Clean MkDocs config, replace landing page, write risk matrix and build order appendices
- [x] 02-02-PLAN.md — Write main implementation plan document (5 phases, kill-switch gate, executive summary)

### Phase 3: Architecture Docs
**Goal**: Internal architecture documentation exists covering the four core design decisions that implementation will depend on: the recursive code generation loop, how Rune maps to the existing monorepo, the adapter storage strategy, and the multi-GPU inference and training strategy
**Depends on**: Phase 2
**Requirements**: ARCH-01, ARCH-02, ARCH-03, ARCH-04
**Success Criteria** (what must be TRUE):
  1. The recursive code generation loop document shows explicit data flow: what enters each step (generate, execute, reflect, distill), what exits, what triggers a retry, and what terminates the loop
  2. The monorepo mapping document lists each new Rune service (rune-agent, lora-server, training-svc, evolution-svc, adapter-registry) alongside the existing monorepo service it extends or runs alongside, with integration points named
  3. The adapter storage strategy document specifies the filesystem path convention, the SQLite schema fields (at minimum: id, version, task_type, created_at, file_path, pass_rate), and the write-once enforcement policy
  4. The multi-GPU strategy document specifies pipeline parallelism (PP=2, TP=1) as the required configuration, explains why tensor parallelism is excluded (confirmed vLLM bug #21471, no NVLink on consumer Blackwell), and documents the GPU lease mechanism for coordinating concurrent training and serving
**Plans**: 1/1 plans complete

Plans:
- [x] 03-01-PLAN.md — Write 4 architecture documents (recursive-loop, monorepo-mapping, adapter-storage, multi-gpu-strategy) and update MkDocs nav

</details>

---

### v2.0 Repo Restructuring & Scaffold (In Progress)

**Milestone Goal:** Restructure the monorepo to match the Rune implementation plan's component layout. Remove template placeholders, create initial importable scaffolds for all planned services and libraries, and validate the full workspace builds and passes quality gates without GPU.

**Build constraint:** Each phase ends with `uv lock && uv sync` passing. Phases respect the import dependency DAG: cleanup unlocks rename → adapter-registry unlocks all services → services unlock final config sync.

## Phase Details

### Phase 4: Cleanup
**Goal**: The template placeholders are gone and the workspace is clean — agent-b-service deleted, agent-a-service renamed to rune-agent with all four name locations updated, root pyproject.toml free of unused template dependencies, and the Makefile typecheck target future-proofed with a glob pattern
**Depends on**: Phase 3 (v1.0 complete)
**Requirements**: CLN-01, CLN-02, CLN-03, CLN-04
**Success Criteria** (what must be TRUE):
  1. `services/agent-b-service` directory does not exist and no reference to it appears anywhere in the repository (pyproject.toml, Makefile, imports, docs); `uv lock && uv sync` passes after removal
  2. `services/rune-agent` exists where `services/agent-a-service` was, with the directory name, `src/rune_agent/` module, `pyproject.toml` project name, and hatch wheel packages all updated to `rune-agent`/`rune_agent`; `uv lock && uv sync` passes
  3. Root `pyproject.toml` contains no references to `google-cloud-aiplatform`, `vertexai`, `google-genai`, or `sentence-transformers`; GPU packages are not listed in the root dev dependencies
  4. `make typecheck` runs `uv run mypy services/*/src libs/*/src` (glob pattern) instead of a hardcoded list of component paths
**Plans**: 3 plans

Plans:
- [ ] 04-01-PLAN.md — Remove agent-b-service (grep references, remove from workspace, uv lock/sync, delete directory)
- [ ] 04-02-PLAN.md — Rename agent-a-service to rune-agent (git mv, update all four name locations, update root pyproject.toml references, uv lock/sync)
- [ ] 04-03-PLAN.md — Clean root pyproject.toml dependencies and update Makefile typecheck glob

### Phase 5: Foundation Libraries
**Goal**: All four library scaffolds exist as importable uv workspace members — adapter-registry first (the dependency root that all services import), followed by extensions to shared, model-training, and inference; every library passes a CPU-only importability test
**Depends on**: Phase 4
**Requirements**: LIB-01, LIB-02, LIB-03, LIB-04
**Success Criteria** (what must be TRUE):
  1. `libs/adapter-registry` is a uv workspace member; `from adapter_registry import AdapterRecord, AdapterRegistry, AdapterAlreadyExistsError, AdapterNotFoundError` succeeds without a GPU; stub CRUD methods (`store`, `retrieve_by_id`, `query_by_task_type`, `list_all`) raise `NotImplementedError` with descriptive messages; `uv lock && uv sync` passes after adding the member
  2. `libs/shared/src/shared/rune_models.py` exists and exports `CodingSession`, `AdapterRef`, and `EvolMetrics` as Pydantic models importable without GPU
  3. `libs/model-training` contains `peft_utils.py`, `trajectory.py`, and `config.py` with typed function signatures that raise `NotImplementedError`; no `torch`, `peft`, `bitsandbytes`, or `transformers` imports appear at module level
  4. `libs/inference` contains `adapter_loader.py` and the inference loader points at the vLLM OpenAI-compatible endpoint via the `openai` package (not a direct `vllm` import); importable without GPU
**Plans**: 3/3 plans complete

Plans:
- [x] 05-01-PLAN.md — Scaffold libs/adapter-registry (new workspace member: AdapterRecord, AdapterRegistry, exceptions, smoke test)
- [x] 05-02-PLAN.md — Extend libs/shared with rune_models.py (CodingSession, AdapterRef, EvolMetrics)
- [x] 05-03-PLAN.md — Extend libs/model-training with PEFT stubs and extend libs/inference with adapter_loader and vLLM client stub

### Phase 5.1: Template Artifact Cleanup
**Goal**: All remaining template artifacts are removed from libs/, apps/, docs/, and root — the three unused TypeScript/Python template libraries are deleted, the template HITL UI app is deleted, template-specific documentation and diagrams are removed, and the workspace configuration is updated to reflect the cleaned state
**Depends on**: Phase 5
**Requirements**: CLN-05, CLN-06, CLN-07, CLN-08
**Success Criteria** (what must be TRUE):
  1. `libs/data-pipeline`, `libs/events-ts`, and `libs/shared-ts` directories do not exist; no references to them appear in root `pyproject.toml`, `mkdocs.yml`, or any configuration file; `uv lock && uv sync` passes after removal
  2. `apps/hitl-ui` directory does not exist; no references to it appear in CI workflow, Makefile, docker-compose, scripts, or copilot-instructions
  3. Template-specific docs are removed: `docs/diagrams/agent-flow.md`, `docs/diagrams/hitl-flow.md`, `docs/diagrams/langgraph-architecture.md`, `docs/onboarding.md`, `docs/testing-guide.md`, `docs/components-overview.md`, and `PROJECT_OVERVIEW.md`; `mkdocs.yml` nav is updated to remove references to deleted files; `uv run mkdocs build` still passes
  4. No file in the repository references "ElixirTrials" in its content (excluding `.planning/` and `pyproject.toml` project name); `grep -r "ElixirTrials" . | grep -v .planning/ | grep -v pyproject.toml` returns zero matches
**Plans**: 2 plans

Plans:
- [x] 05.1-01-PLAN.md — Remove template libraries (data-pipeline, events-ts, shared-ts) and template app (hitl-ui) from workspace and filesystem
- [x] 05.1-02-PLAN.md — Remove template docs and diagrams, update mkdocs.yml nav, clean all ElixirTrials references

### Phase 6: Service Scaffolds
**Goal**: All five services exist in their correct forms — four as importable uv workspace members with FastAPI endpoints returning 501, one (lora-server) as a Dockerfile-only service not in the workspace — and the workspace configuration in root pyproject.toml is fully synchronized across all five required sections
**Depends on**: Phase 5.1
**Requirements**: SVC-01, SVC-02, SVC-03, SVC-04, SVC-05, CFG-01, CFG-02
**Success Criteria** (what must be TRUE):
  1. `services/lora-server` contains a Dockerfile, `startup.sh`, `config.yaml`, FastAPI health sidecar, and `VLLMClient` stub; it does NOT appear in the uv workspace members list; `LoraServerConfig` raises `ValueError` if `tensor_parallel_size=2`
  2. `services/training-svc` and `services/evolution-svc` are uv workspace members with FastAPI apps, stub endpoints returning HTTP 501, correct Pydantic request/response schemas, and SQLModel job tracking; both are importable without GPU
  3. `services/rune-agent` has a LangGraph `StateGraph(RuneState)` with generate, execute, reflect, and save_trajectory nodes; `should_retry` is implemented (not stubbed) and returns the correct branch for both retry and terminal cases; the graph compiles without error
  4. `services/api-service` exposes `/adapters` and `/sessions` router stubs returning HTTP 501, wired into `main.py` via `include_router`; `adapter-registry` is declared as a workspace dependency
  5. Root `pyproject.toml` workspace members list includes all new components (rune-agent, training-svc, evolution-svc, adapter-registry) and excludes removed ones (agent-a-service, agent-b-service); all five hardcoded path sections (workspace members, mypy overrides, pytest pythonpath, testpaths, coverage source) are consistent; `uv lock && uv sync` passes cleanly
**Plans**: 4 plans

Plans:
- [ ] 06-01-PLAN.md — Scaffold services/lora-server (Dockerfile-only: config.py with TP=2 ValueError, startup.sh, config.yaml, Dockerfile, health sidecar, VLLMClient stub)
- [ ] 06-02-PLAN.md — Rework services/rune-agent (RuneState TypedDict, 4-node StateGraph, should_retry implementation, updated exports)
- [ ] 06-03-PLAN.md — Scaffold services/training-svc and services/evolution-svc (workspace members: FastAPI 501 endpoints, Pydantic schemas, SQLModel job tracking, uv lock/sync)
- [ ] 06-04-PLAN.md — Extend api-service with /adapters and /sessions routers; synchronize root pyproject.toml across all five config sections; final uv lock/sync

### Phase 7: Configuration & Quality Gate
**Goal**: The full workspace passes every quality gate without GPU — pytest importability tests pass, no GPU imports exist at module level in any scaffold, ruff lints clean, all API stubs return 501, docker-compose includes lora-server, and mkdocs builds successfully
**Depends on**: Phase 6
**Requirements**: CFG-03, CFG-04, QA-01, QA-02, QA-03, QA-04
**Success Criteria** (what must be TRUE):
  1. `uv run pytest` passes across the full workspace; every new component (adapter-registry, rune-agent, training-svc, evolution-svc) has at least one test confirming the component is importable without a GPU present
  2. A scan of all scaffold modules finds zero occurrences of `import torch`, `import peft`, `import bitsandbytes`, `import transformers`, or `import vllm` at module level in any new or modified component
  3. `uv run ruff check .` passes cleanly across all new and modified components with zero errors or warnings
  4. Every API route stub in api-service, training-svc, and evolution-svc returns HTTP 501 (Not Implemented); no stub returns HTTP 200 with placeholder data
  5. `infra/docker-compose.yml` defines a `lora-server` service with GPU passthrough (`deploy.resources.reservations.devices`) and a volume mount for adapter storage; `mkdocs.yml` nav reflects the new component structure and `uv run mkdocs build` exits 0
**Plans**: 3 plans

Plans:
- [ ] 07-01-PLAN.md — Fix ruff lint, create importability tests for 3 components, fix xdist collision, verify QA-01 through QA-04
- [ ] 07-02-PLAN.md — Add lora-server service with GPU passthrough to docker-compose.yml (CFG-03)
- [ ] 07-03-PLAN.md — Create mkdocs.yml + docs for 3 components, update root nav, verify mkdocs build (CFG-04)

## Progress

**Execution Order:**
Phases execute in numeric order: 4 → 5 → 5.1 → 6 → 7

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. README | v1.0 | 1/1 | Complete | 2026-03-02 |
| 2. Implementation Plan | v1.0 | 2/2 | Complete | 2026-03-02 |
| 3. Architecture Docs | v1.0 | 1/1 | Complete | 2026-03-02 |
| 4. Cleanup | v2.0 | 3/3 | Complete | 2026-03-02 |
| 5. Foundation Libraries | v2.0 | 3/3 | Complete | 2026-03-02 |
| 5.1. Template Artifact Cleanup | v2.0 | 2/2 | Complete | 2026-03-03 |
| 6. Service Scaffolds | v2.0 | Complete    | 2026-03-03 | - |
| 7. Configuration & Quality Gate | v2.0 | 0/3 | Not started | - |
