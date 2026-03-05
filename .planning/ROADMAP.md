# Roadmap: Rune

## Milestones

- ✅ **v1.0 Documentation & Implementation Plan** — Phases 1-3 (shipped 2026-03-02)
- ✅ **v2.0 Repo Restructuring & Scaffold** — Phases 4-7 (shipped 2026-03-03)
- ✅ **v3.0 Scientific Article Documentation** — Phases 8-12 (shipped 2026-03-03)
- ✅ **v4.0 API Wireframes & TDD Foundation** — Phases 13-17 (shipped 2026-03-05)
- 🚧 **v5.0 First Implementation** — Phases 18-22 (in progress)

## Phases

<details>
<summary>✅ v1.0 Documentation & Implementation Plan (Phases 1-3) — SHIPPED 2026-03-02</summary>

- [x] Phase 1: README (1/1 plans) — completed 2026-03-02
- [x] Phase 2: Implementation Plan (2/2 plans) — completed 2026-03-02
- [x] Phase 3: Architecture Docs (1/1 plans) — completed 2026-03-02

</details>

<details>
<summary>✅ v2.0 Repo Restructuring & Scaffold (Phases 4-7) — SHIPPED 2026-03-03</summary>

- [x] Phase 4: Cleanup (3/3 plans) — completed 2026-03-02
- [x] Phase 5: Foundation Libraries (3/3 plans) — completed 2026-03-02
- [x] Phase 5.1: Template Artifact Cleanup (2/2 plans) — completed 2026-03-03
- [x] Phase 6: Service Scaffolds (4/4 plans) — completed 2026-03-03
- [x] Phase 7: Configuration & Quality Gate (3/3 plans) — completed 2026-03-03

</details>

<details>
<summary>✅ v3.0 Scientific Article Documentation (Phases 8-12) — SHIPPED 2026-03-03</summary>

- [x] Phase 8: MkDocs Infrastructure (2/2 plans) — completed 2026-03-03
- [x] Phase 9: References Skeleton + Background (2/2 plans) — completed 2026-03-03
- [x] Phase 10: Methods Section (1/1 plan) — completed 2026-03-03
- [x] Phase 11: Results & Discussion Outlines (2/2 plans) — completed 2026-03-03
- [x] Phase 12: Abstract + Quality Audit (2/2 plans) — completed 2026-03-03

</details>

<details>
<summary>✅ v4.0 API Wireframes & TDD Foundation (Phases 13-17) — SHIPPED 2026-03-05</summary>

- [x] Phase 13: Test Infrastructure (2/2 plans) — completed 2026-03-03
- [x] Phase 14: Core Library Wireframes (3/3 plans) — completed 2026-03-03
- [x] Phase 15: New & Reworked Library Wireframes (2/2 plans) — completed 2026-03-03
- [x] Phase 16: Service Wireframes (4/4 plans) — completed 2026-03-03
- [x] Phase 17: Quality Gate (2/2 plans) — completed 2026-03-04

</details>

### v5.0 First Implementation (In Progress)

**Milestone Goal:** Transform wireframe stubs into a working end-to-end system — adapter registry to provider-agnostic inference to agent loop to QLoRA training to Doc-to-LoRA kill-switch gate — running entirely on local hardware.

- [x] **Phase 18: Adapter Registry** — SQLite-backed CRUD hub that all other components read from and write to (2/2 plans complete)
- [ ] **Phase 19: Inference Provider Abstraction** — Abstract InferenceProvider interface with vLLM and Ollama implementations, provider factory, and lora-server configuration
- [ ] **Phase 20: Agent Loop** — Backend-agnostic generate → execute → reflect → save_trajectory cycle with trajectory persistence
- [ ] **Phase 21: QLoRA Training Pipeline** — Full gradient-descent training path from trajectory to PEFT adapter stored in registry, with training-svc HTTP dispatch
- [ ] **Phase 22: Kill-Switch Gate** — Doc-to-LoRA hypernetwork + evaluation lib measuring the 5% Pass@1 improvement threshold

## Phase Details

### Phase 18: Adapter Registry
**Goal**: Users can persist, retrieve, and query LoRA adapter metadata via a SQLite-backed registry that all other components share
**Depends on**: Nothing (first v5.0 phase)
**Requirements**: AREG-01, AREG-02, AREG-03, AREG-04, AREG-05
**Success Criteria** (what must be TRUE):
  1. User can call AdapterRegistry.store() with full metadata and retrieve the same record by ID with no data loss
  2. User can call AdapterRegistry.query_by_task_type() and receive only adapters matching the specified task type
  3. User can call AdapterRegistry.list_all() and receive all non-archived records (archived records excluded)
  4. Multiple FastAPI requests can read and write concurrently without deadlock (WAL mode confirmed via PRAGMA)
  5. AdapterRegistry constructor accepts an explicit engine parameter so all services share one SQLite file via DATABASE_URL
**Plans**: 2 plans

Plans:
- [x] 18-01-PLAN.md — Implement AdapterRegistry: constructor (Engine + WAL hook + create_all) and 4 CRUD methods — DONE 2026-03-05
- [x] 18-02-PLAN.md — Pivot tests from red to green: update conftest fixtures, replace NotImplementedError assertions, add WAL and concurrency integration tests — DONE 2026-03-05

### Phase 19: Inference Provider Abstraction
**Goal**: Users can generate completions and hot-load/unload LoRA adapters through a provider-agnostic interface, with vLLM and Ollama backends selectable by configuration and a correctly configured lora-server
**Depends on**: Phase 18
**Requirements**: INF-01, INF-02, INF-03, INF-04, INF-05, INF-06, INF-07, INFRA-01, INFRA-02, INFRA-03
**Success Criteria** (what must be TRUE):
  1. InferenceProvider interface exists with generate(), load_adapter(), unload_adapter(), and list_adapters() methods that both backends implement
  2. Provider factory returns a VLLMProvider or OllamaProvider instance based on an environment variable or config file with no code changes required
  3. User can call provider.generate() with an adapter name and receive adapter-conditioned output from the active backend
  4. User can load two adapters simultaneously via VLLMProvider and generate from each by name without collision
  5. Agent can configure different models or providers per step (e.g., VLLMProvider for generation, OllamaProvider for reflection) without changing agent graph structure
  6. lora-server starts without error with VLLM_ALLOW_RUNTIME_LORA_UPDATING=True set, and docker-compose runs all services without port conflict
**Plans**: 3 plans

Plans:
- [ ] 19-01-PLAN.md — InferenceProvider ABC + GenerationResult + VLLMProvider + OllamaProvider with full test coverage
- [ ] 19-02-PLAN.md — lora-server Dockerfile update (vLLM base image + runtime LoRA env) and docker-compose port conflict fix
- [ ] 19-03-PLAN.md — Provider factory with instance cache + per-step config + __init__.py exports + delete old stubs

### Phase 20: Agent Loop
**Goal**: Users can invoke the Rune agent on a coding task and observe a complete generate → execute → reflect → save_trajectory cycle through the InferenceProvider interface, with trajectory data persisted to disk
**Depends on**: Phase 18, Phase 19
**Requirements**: AGENT-01, AGENT-02, AGENT-03, AGENT-04, AGENT-05, AGENT-06, TRAIN-01, TRAIN-02
**Success Criteria** (what must be TRUE):
  1. Agent invocation on a coding task produces generated code from generate_node via InferenceProvider.generate() (backend-agnostic — same call path for vLLM or Ollama)
  2. execute_node runs the generated code in a subprocess sandbox and returns stdout, stderr, exit_code, and tests_passed within the timeout
  3. reflect_node accumulates attempt count and trajectory data without making any LLM call
  4. save_trajectory_node writes a structured JSON trajectory file (session_id, steps, outcome) readable after the session ends
  5. Agent loop closes end-to-end: retries on failure up to max_attempts, then terminates with either success or exhausted outcome; RuneState includes session_id field
**Plans**: TBD

### Phase 21: QLoRA Training Pipeline
**Goal**: Users can trigger end-to-end QLoRA training from a recorded trajectory, producing a PEFT safetensors adapter stored in the registry and accessible for loading via the inference provider
**Depends on**: Phase 18, Phase 20
**Requirements**: TRAIN-03, TRAIN-04, TRAIN-05, TRAIN-06, TRAIN-07, INFRA-04, INFRA-05
**Success Criteria** (what must be TRUE):
  1. User can call build_qlora_config() and receive a PEFT LoraConfig with NF4 quantization and bfloat16 compute dtype
  2. User can call apply_lora_adapter() to attach a LoRA adapter to a base model for inference or further training
  3. POST /train/lora with a session_id runs the full trajectory → SFT format → PEFT train → save safetensors → registry.store() pipeline and returns a job_id
  4. Trained adapter files are in standard PEFT safetensors format (no embed_tokens keys) compatible with vLLM dynamic loading via InferenceProvider.load_adapter()
  5. GPU imports (peft, bitsandbytes, transformers, trl, datasets) are deferred inside function bodies and do not cause ImportError in CPU-only CI
**Plans**: TBD

### Phase 22: Kill-Switch Gate
**Goal**: Users can measure whether Doc-to-LoRA-generated adapters improve Pass@1 on a HumanEval subset by at least 5% relative to the base model, validating the core hypothesis
**Depends on**: Phase 20, Phase 21
**Requirements**: DTOL-01, DTOL-02, DTOL-03, DTOL-04, EVAL-01, EVAL-02, EVAL-03
**Success Criteria** (what must be TRUE):
  1. DocToLoraHypernetwork exists in model-training lib, accepts a coding trajectory, and returns rank-8 LoRA adapter weights in a single forward pass under 1 second
  2. Generated adapter weights are in standard PEFT safetensors format and can be loaded into vLLM via InferenceProvider.load_adapter() without error
  3. User can call run_humaneval_subset() and receive pass/fail results for each task in the 20-task subset
  4. User can call calculate_pass_at_k() and receive a Pass@1 score for both baseline and adapter-enhanced configurations
  5. Kill-switch gate comparison prints a clear verdict: PASS (>=5% improvement) or FAIL (hypothesis rejected), with both baseline and adapter Pass@1 scores shown
**Plans**: TBD

## Progress

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. README | v1.0 | 1/1 | Complete | 2026-03-02 |
| 2. Implementation Plan | v1.0 | 2/2 | Complete | 2026-03-02 |
| 3. Architecture Docs | v1.0 | 1/1 | Complete | 2026-03-02 |
| 4. Cleanup | v2.0 | 3/3 | Complete | 2026-03-02 |
| 5. Foundation Libraries | v2.0 | 3/3 | Complete | 2026-03-02 |
| 5.1. Template Artifact Cleanup | v2.0 | 2/2 | Complete | 2026-03-03 |
| 6. Service Scaffolds | v2.0 | 4/4 | Complete | 2026-03-03 |
| 7. Configuration & Quality Gate | v2.0 | 3/3 | Complete | 2026-03-03 |
| 8. MkDocs Infrastructure | v3.0 | 2/2 | Complete | 2026-03-03 |
| 9. References Skeleton + Background | v3.0 | 2/2 | Complete | 2026-03-03 |
| 10. Methods Section | v3.0 | 1/1 | Complete | 2026-03-03 |
| 11. Results & Discussion Outlines | v3.0 | 2/2 | Complete | 2026-03-03 |
| 12. Abstract + Quality Audit | v3.0 | 2/2 | Complete | 2026-03-03 |
| 13. Test Infrastructure | v4.0 | 2/2 | Complete | 2026-03-03 |
| 14. Core Library Wireframes | v4.0 | 3/3 | Complete | 2026-03-03 |
| 15. New & Reworked Library Wireframes | v4.0 | 2/2 | Complete | 2026-03-03 |
| 16. Service Wireframes | v4.0 | 4/4 | Complete | 2026-03-03 |
| 17. Quality Gate | v4.0 | 2/2 | Complete | 2026-03-04 |
| 18. Adapter Registry | v5.0 | Complete    | 2026-03-05 | 2026-03-05 |
| 19. Inference Provider Abstraction | v5.0 | 0/3 | Planned | - |
| 20. Agent Loop | v5.0 | 0/TBD | Not started | - |
| 21. QLoRA Training Pipeline | v5.0 | 0/TBD | Not started | - |
| 22. Kill-Switch Gate | v5.0 | 0/TBD | Not started | - |

---
*Last updated: 2026-03-05 after Phase 19 planning (3 plans created in 2 waves)*
