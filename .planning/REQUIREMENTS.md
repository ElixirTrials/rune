# Requirements: Rune

**Defined:** 2026-03-02
**Core Value:** A local coding agent that learns from its own coding trajectories, building persistent parametric memory that scales independently of context window size.

## v1 Requirements (COMPLETE)

Requirements for milestone v1.0 — Documentation & Implementation Plan.

### Documentation

- [x] **DOC-01**: README.md covers project vision, core innovation (parametric episodic memory), and how it differs from existing coding agents
- [x] **DOC-02**: README.md includes architecture overview diagram (recursive loop, hypernetwork, evolution operator, adapter serving)
- [x] **DOC-03**: README.md includes hardware requirements and getting-started prerequisites
- [x] **DOC-04**: README.md includes current status section with honest assessment of research stage
- [x] **DOC-05**: README.md includes references to key papers (Doc-to-LoRA, S-LoRA, QLoRA)

### Implementation Plan

- [x] **PLAN-01**: Phase 0 defines hardware/environment validation with specific pass/fail criteria
- [x] **PLAN-02**: Phase 1 defines core hypothesis validation as kill-switch gate (Doc-to-LoRA on coding tasks)
- [x] **PLAN-03**: Each subsequent phase has clear deliverables, dependencies, and success criteria
- [x] **PLAN-04**: Plan addresses key research risks: adapter composition interference, hypernetwork mode collapse, catastrophic forgetting
- [x] **PLAN-05**: Plan reflects hardware reality: dual RTX 4090 + CXL, pipeline parallelism (not tensor), QLoRA required
- [x] **PLAN-06**: Plan includes suggested build order from architecture research (adapter-registry → lora-server → api-extensions → rune-agent → evolution → training)

### Architecture Documentation

- [x] **ARCH-01**: Document the recursive code generation loop with data flow
- [x] **ARCH-02**: Document how Rune components map to existing monorepo services
- [x] **ARCH-03**: Document adapter storage strategy (filesystem + SQLite metadata)
- [x] **ARCH-04**: Document multi-GPU strategy (pipeline parallelism, GPU scheduling for training vs serving)

## v2 Requirements

Requirements for milestone v2.0 — Repo Restructuring & Scaffold.

### Cleanup

- [x] **CLN-01**: Template placeholder service `agent-b-service` is removed from the workspace (directory deleted, all references in root pyproject.toml removed, uv lock/sync passes)
- [x] **CLN-02**: Template placeholder service `agent-a-service` is renamed to `rune-agent` with all four name locations updated (directory, src/ module, pyproject.toml name, hatch wheel packages)
- [x] **CLN-03**: Root pyproject.toml dev dependencies are cleaned of unused template packages (google-cloud-aiplatform, vertexai, google-genai, sentence-transformers) and GPU packages are moved out of root scope
- [x] **CLN-04**: Makefile typecheck target uses glob pattern (`services/*/src libs/*/src`) instead of hardcoded component list

### Template Artifact Cleanup

- [x] **CLN-05**: Template libraries `libs/data-pipeline`, `libs/events-ts`, and `libs/shared-ts` are removed from the workspace (directories deleted, all references in root pyproject.toml removed, uv lock/sync passes)
- [x] **CLN-06**: Template-specific documentation removed: `docs/diagrams/agent-flow.md`, `docs/diagrams/hitl-flow.md`, `docs/diagrams/langgraph-architecture.md`, `docs/onboarding.md`, `docs/testing-guide.md`, `docs/components-overview.md`, `PROJECT_OVERVIEW.md`; mkdocs.yml nav updated; mkdocs build passes
- [x] **CLN-07**: No file references "ElixirTrials" in content (excluding .planning/ and pyproject.toml project name)
- [x] **CLN-08**: Template app `apps/hitl-ui` is removed from the workspace (directory deleted, all references in CI, Makefile, docker-compose, scripts, and copilot-instructions removed)

### Scaffold — Libraries

- [ ] **LIB-01**: `libs/adapter-registry` exists as a uv workspace member with SQLModel `AdapterRecord` schema, stub CRUD methods (`store`, `retrieve_by_id`, `query_by_task_type`, `list_all`), and custom exceptions (`AdapterAlreadyExistsError`, `AdapterNotFoundError`)
- [ ] **LIB-02**: `libs/shared` is extended with `rune_models.py` containing Rune-specific Pydantic models (`CodingSession`, `AdapterRef`, `EvolMetrics`)
- [ ] **LIB-03**: `libs/model-training` is extended with stub modules (`peft_utils.py`, `trajectory.py`, `config.py`) containing typed function signatures that raise `NotImplementedError`
- [ ] **LIB-04**: `libs/inference` is extended with `adapter_loader.py` stub and updated loader pointing at vLLM OpenAI-compatible endpoint

### Scaffold — Services

- [x] **SVC-01**: `services/rune-agent` has a LangGraph `StateGraph` with `RuneState`, four nodes (generate, execute, reflect, save_trajectory), conditional edges, and a working `should_retry` function
- [x] **SVC-02**: `services/lora-server` exists as a Dockerfile-only service (NOT a uv workspace member) with startup script, config enforcing PP=2/TP=1, FastAPI health sidecar, and `VLLMClient` stub
- [x] **SVC-03**: `services/training-svc` exists as a uv workspace service with FastAPI endpoints (`/train/lora`, `/train/hypernetwork`, `/jobs/{job_id}`), request schemas, and `TrainingJob` SQLModel
- [x] **SVC-04**: `services/evolution-svc` exists as a uv workspace service with FastAPI endpoints (`/evaluate`, `/evolve`, `/promote`, `/prune`), evaluation schemas, and stub lifecycle manager
- [x] **SVC-05**: `services/api-service` is extended with `/adapters` and `/sessions` router stubs (501 responses) wired into main.py

### Configuration

- [x] **CFG-01**: Root pyproject.toml workspace members list includes all new components and excludes removed ones; all five hardcoded path sections (workspace members, mypy overrides, pytest pythonpath, testpaths, coverage source) are consistent
- [x] **CFG-02**: `uv lock && uv sync` passes cleanly with the final workspace configuration
- [x] **CFG-03**: `infra/docker-compose.yml` includes lora-server container definition with GPU passthrough and adapter volume mount
- [x] **CFG-04**: `mkdocs.yml` nav reflects the new component structure; `uv run mkdocs build` passes

### Quality

- [x] **QA-01**: Every new component has at least one test confirming importability without GPU; `uv run pytest` passes across the full workspace
- [x] **QA-02**: No GPU imports (`torch`, `peft`, `bitsandbytes`, `transformers`, `vllm`) appear at module level in any scaffold component
- [x] **QA-03**: `uv run ruff check .` passes cleanly across all new and modified components
- [x] **QA-04**: All API route stubs return HTTP 501 (Not Implemented), not 200 with fake data

## v3 Requirements

Requirements for milestone v3.0 — Scientific Article Documentation.

### Infrastructure

- [x] **INFRA-01**: `mkdocs.yml` has `pymdownx.arithmatex` enabled with `generic: true`; MathJax 3 CDN loaded via `extra_javascript`; `docs/assets/mathjax-config.js` configures TeX delimiters and `tags: "ams"` for auto-numbered equations; inline (`$...$`) and block (`$$...$$`) math renders correctly in `mkdocs serve`
- [x] **INFRA-02**: `markdown.extensions.footnotes` enabled in `mkdocs.yml`; footnotes render as numbered superscripts linking to endnotes; used for in-text citations pointing to `references.md` anchors
- [x] **INFRA-03**: `docs/assets/academic.css` provides academic styling classes: abstract block, author/affiliation header, figure captions via `attr_list`; loaded via `extra_css` in `mkdocs.yml`
- [x] **INFRA-04**: `docs/article/` directory exists with `index.md` landing page; root `mkdocs.yml` nav has "Scientific Article" section between "Appendices" and "Components"; `uv run mkdocs build` passes

### Article Content

- [ ] **ART-01**: Abstract section is fully written: states the problem (context window limitations for local coding agents), the approach (parametric episodic memory via composable LoRA adapters), the mechanism (Doc-to-LoRA hypernetwork with evolutionary selection), and the expected contribution; uses three-tier claim vocabulary (validated/expected/proposed); includes "Research Stage" disclaimer
- [x] **ART-02**: Background section is fully written covering: (a) context window problem and memory approaches taxonomy (token-space vs destructive weight-space vs composable weight-space), (b) LoRA/QLoRA foundations with math notation, (c) multi-adapter serving (S-LoRA), (d) hypernetwork architectures (Ha et al. 2016, Doc-to-LoRA, SHINE), (e) LoRA composition methods (TIES, LoRA Soups, adapter merging interference), (f) Programming by Backprop validation of code-as-abstraction, (g) episodic memory position (arXiv:2502.06975); cites all 6+ must-cite papers
- [x] **ART-03**: Methods section is fully written covering: (a) system architecture overview with recursive loop, (b) Doc-to-LoRA hypernetwork adaptation for coding trajectories, (c) adapter distillation pipeline with pseudocode, (d) evolution operator (consolidate/update/forget/merge) with fitness-based selection, (e) memory composition strategy (hierarchical adapter tree), (f) PBB-grounded training data strategy (code representations over prose), (g) System-2 reasoning loop with chain-of-thought; includes mathematical notation for LoRA operations and hypernetwork mapping
- [x] **ART-04**: Results section has detailed outline: kill-switch experiment design (Pass@1 on HumanEval), ablation structure (hypernetwork vs fine-tuning, single vs composed adapters, with vs without evolution), specific metrics (Pass@1, adapter quality, composition interference), baseline comparisons (vanilla model, RAG, fine-tuned), hardware configuration; all content marked as "Planned Experiments"
- [ ] **ART-05**: Discussion section has detailed outline: expected contributions analysis, limitations (pre-implementation status, hardware constraints, adapter interference risk, hypernetwork mode collapse), future work (QDoRA, cross-project transfer, online adaptation), broader implications for local AI agents; structured with subsection headers

### References & Quality

- [ ] **REF-01**: `docs/article/references.md` contains a formatted bibliography with HTML anchor targets for all cited papers; minimum 10 references including Doc-to-LoRA, S-LoRA, QLoRA, PBB, SHINE, "Episodic Memory is the Missing Piece", LoRA (Hu et al. 2021), HyperNetworks (Ha et al. 2016), LoRA Soups, and at least one LoRA composition interference paper
- [ ] **REF-02**: All in-text citations use footnotes linking to `references.md#anchor` targets; `mkdocs build` with `anchors: warn` produces zero broken citation warnings
- [ ] **REF-03**: `uv run mkdocs build` exits 0 with no warnings; all math renders correctly; all footnote links resolve; nav entry appears correctly

## Future Requirements (Deferred)

- Functional implementation of adapter storage write-once enforcement (deferred to implementation milestone)
- Alembic migration scripts for new SQLModel tables (deferred to implementation milestone)
- Docker sandbox for rune-agent code execution (deferred to implementation milestone)
- MLflow experiment tracking integration (deferred to implementation milestone)
- Dramatiq async job queue for training-svc (scaffold uses stub; full async deferred)

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| GPU-dependent code or training logic | Scaffold is CPU-only; GPU work deferred to implementation milestone |
| Real LLM calls or inference | Stubs only; implementation deferred |
| CI/CD pipeline updates beyond Makefile fix | Deferred until components have real code |
| Base model download or configuration | Deferred to Phase 0 implementation |
| Adapter composition logic (TIES/CAT) | Phase 4 implementation |
| Frontend feature implementation in hitl-ui | Minor structural update only in this milestone |
| Alembic migrations | Deferred to implementation; schemas defined but not migrated |

## Traceability

Which phases cover which requirements.

### v1 (COMPLETE)

| Requirement | Phase | Status |
|-------------|-------|--------|
| DOC-01 | Phase 1 | Complete |
| DOC-02 | Phase 1 | Complete |
| DOC-03 | Phase 1 | Complete |
| DOC-04 | Phase 1 | Complete |
| DOC-05 | Phase 1 | Complete |
| PLAN-01 | Phase 2 | Complete |
| PLAN-02 | Phase 2 | Complete |
| PLAN-03 | Phase 2 | Complete |
| PLAN-04 | Phase 2 | Complete |
| PLAN-05 | Phase 2 | Complete |
| PLAN-06 | Phase 2 | Complete |
| ARCH-01 | Phase 3 | Complete |
| ARCH-02 | Phase 3 | Complete |
| ARCH-03 | Phase 3 | Complete |
| ARCH-04 | Phase 3 | Complete |

### v2

| Requirement | Phase | Status |
|-------------|-------|--------|
| CLN-01 | Phase 4 | Complete |
| CLN-02 | Phase 4 | Complete |
| CLN-03 | Phase 4 | Complete |
| CLN-04 | Phase 4 | Complete |
| CLN-05 | Phase 5.1 | Complete |
| CLN-06 | Phase 5.1 | Complete |
| CLN-07 | Phase 5.1 | Complete |
| CLN-08 | Phase 5.1 | Complete |
| LIB-01 | Phase 5 | Pending |
| LIB-02 | Phase 5 | Pending |
| LIB-03 | Phase 5 | Pending |
| LIB-04 | Phase 5 | Pending |
| SVC-01 | Phase 6 | Complete |
| SVC-02 | Phase 6 | Complete |
| SVC-03 | Phase 6 | Complete |
| SVC-04 | Phase 6 | Complete |
| SVC-05 | Phase 6 | Complete |
| CFG-01 | Phase 6 | Complete |
| CFG-02 | Phase 6 | Complete |
| CFG-03 | Phase 7 | Complete |
| CFG-04 | Phase 7 | Complete |
| QA-01 | Phase 7 | Complete |
| QA-02 | Phase 7 | Complete |
| QA-03 | Phase 7 | Complete |
| QA-04 | Phase 7 | Complete |

**v2 Coverage:**
- v2 requirements: 25 total
- Mapped to phases: 25
- Unmapped: 0

### v3

| Requirement | Phase | Status |
|-------------|-------|--------|
| INFRA-01 | Phase 8 | Complete |
| INFRA-02 | Phase 8 | Complete |
| INFRA-03 | Phase 8 | Complete |
| INFRA-04 | Phase 8 | Complete |
| REF-01 | Phase 9 | Pending |
| ART-02 | Phase 9 | Complete |
| ART-03 | Phase 10 | Complete |
| ART-04 | Phase 11 | Complete |
| ART-05 | Phase 11 | Pending |
| ART-01 | Phase 12 | Pending |
| REF-02 | Phase 12 | Pending |
| REF-03 | Phase 12 | Pending |

**v3 Coverage:**
- v3 requirements: 12 total
- Mapped to phases: 12
- Unmapped: 0

---
*Requirements defined: 2026-03-02*
*Last updated: 2026-03-03 — Added v3 traceability (Phases 8-12)*
