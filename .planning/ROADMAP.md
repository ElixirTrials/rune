# Roadmap: Rune

## Milestones

- [x] **v1.0 Documentation & Implementation Plan** - Phases 1-3 (shipped 2026-03-02)
- [x] **v2.0 Repo Restructuring & Scaffold** - Phases 4-7 (shipped 2026-03-03)
- [x] **v3.0 Scientific Article Documentation** - Phases 8-12 (shipped 2026-03-03)
- [ ] **v4.0 API Wireframes & TDD Foundation** - Phases 13-17 (in progress)

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

<details>
<summary>v2.0 Repo Restructuring & Scaffold (Phases 4-7) - COMPLETE 2026-03-03</summary>

**Milestone Goal:** Restructure the monorepo to match the Rune implementation plan's component layout. Remove template placeholders, create initial importable scaffolds for all planned services and libraries, and validate the full workspace builds and passes quality gates without GPU.

**Build constraint:** Each phase ends with `uv lock && uv sync` passing. Phases respect the import dependency DAG: cleanup unlocks rename → adapter-registry unlocks all services → services unlock final config sync.

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
- [x] 04-01-PLAN.md — Remove agent-b-service (grep references, remove from workspace, uv lock/sync, delete directory)
- [x] 04-02-PLAN.md — Rename agent-a-service to rune-agent (git mv, update all four name locations, update root pyproject.toml references, uv lock/sync)
- [x] 04-03-PLAN.md — Clean root pyproject.toml dependencies and update Makefile typecheck glob

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
- [x] 06-01-PLAN.md — Scaffold services/lora-server (Dockerfile-only: config.py with TP=2 ValueError, startup.sh, config.yaml, Dockerfile, health sidecar, VLLMClient stub)
- [x] 06-02-PLAN.md — Rework services/rune-agent (RuneState TypedDict, 4-node StateGraph, should_retry implementation, updated exports)
- [x] 06-03-PLAN.md — Scaffold services/training-svc and services/evolution-svc (workspace members: FastAPI 501 endpoints, Pydantic schemas, SQLModel job tracking, uv lock/sync)
- [x] 06-04-PLAN.md — Extend api-service with /adapters and /sessions routers; synchronize root pyproject.toml across all five config sections; final uv lock/sync

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
- [x] 07-01-PLAN.md — Fix ruff lint, create importability tests for 3 components, fix xdist collision, verify QA-01 through QA-04
- [x] 07-02-PLAN.md — Add lora-server service with GPU passthrough to docker-compose.yml (CFG-03)
- [x] 07-03-PLAN.md — Create mkdocs.yml + docs for 3 components, update root nav, verify mkdocs build (CFG-04)

</details>

---

<details>
<summary>v3.0 Scientific Article Documentation (Phases 8-12) - COMPLETE 2026-03-03</summary>

**Milestone Goal:** Write a scientific-article style page for the MkDocs documentation site presenting the theoretical foundation and algorithmic design of Rune's parametric episodic memory approach. Infrastructure comes first (math rendering must exist before equations are written). Abstract is written last (summarizes what was actually written, not what was planned).

### Phase 8: MkDocs Infrastructure
**Goal**: Math rendering, citation footnotes, and academic styling are active in the MkDocs site; the article directory structure exists with nav integration; `uv run mkdocs build` passes — all of this confirmed before a single equation or citation is written
**Depends on**: Phase 7 (v2.0 complete)
**Requirements**: INFRA-01, INFRA-02, INFRA-03, INFRA-04
**Success Criteria** (what must be TRUE):
  1. Visiting `mkdocs serve` and navigating to any article page shows inline math (`\(...\)`) and block math (`\[...\]`) rendered as formatted equations — not raw LaTeX strings — in the browser
  2. A footnote reference `[^hu2021lora]` in any article page renders as a numbered superscript that links to the endnote at the bottom of the same page; the endnote links back to the footnote location
  3. The `docs/article/` directory exists and appears as a "Scientific Article" nav section in the MkDocs sidebar between "Appendices" and "Components"; `uv run mkdocs build` exits 0
  4. `docs/assets/academic.css` is loaded; an abstract block rendered with the `.abstract` class is visually distinguished from surrounding body text (bordered/shaded box)
**Plans**: 2/2 plans complete

Plans:
- [x] 08-01-PLAN.md — Configure arithmatex + MathJax 3 (mkdocs.yml extensions, mathjax-config.js, CDN extra_javascript)
- [x] 08-02-PLAN.md — Enable footnotes + attr_list + md_in_html, create academic.css, scaffold docs/article/ with stub index.md and nav entry

### Phase 9: References Skeleton + Background
**Goal**: A complete annotated bibliography exists with HTML anchor targets for all cited papers; the Background section is fully written covering the memory taxonomy, LoRA/QLoRA foundations, S-LoRA serving, hypernetwork architectures, LoRA composition methods, and concurrent work — with all in-text citations linking to bibliography anchors
**Depends on**: Phase 8
**Requirements**: REF-01, ART-02
**Success Criteria** (what must be TRUE):
  1. `docs/article/references.md` exists and contains formatted entries for all 10+ required papers (Doc-to-LoRA, S-LoRA, QLoRA, PBB, SHINE, Episodic Memory Missing Piece, LoRA Hu et al. 2021, HyperNetworks Ha et al. 2016, LoRA Soups, Inter-LoRA Orthogonality); each entry has an HTML `<a id="citekey">` anchor target
  2. The Background section establishes the three-category memory taxonomy (token-space vs destructive weight-space vs composable weight-space) and positions Rune in the composable weight-space category with citations to supporting papers
  3. Background covers LoRA mathematical foundations with at least one rendered equation (the rank decomposition `ΔW = BA`), QLoRA's 4-bit quantization relevance, S-LoRA's concurrent adapter loading, Ha et al. hypernetwork framing, and Doc-to-LoRA as the central prior work — with honest description of what Doc-to-LoRA validates vs what Rune proposes to extend
  4. The trajectory-as-distinct-input-modality argument appears in Background, distinguishing code trajectories from document inputs (Doc-to-LoRA), in-context examples (SHINE), and text instructions (Text-to-LoRA), with PBB cited as independent empirical support
**Plans**: 2/2 plans complete

Plans:
- [x] 09-01-PLAN.md — Create docs/article/references.md with 12 bibliography entries and HTML anchor targets; add to mkdocs.yml nav
- [x] 09-02-PLAN.md — Write docs/article/background.md with 8 subsections (memory taxonomy, LoRA/QLoRA/S-LoRA, hypernetworks, Doc-to-LoRA, composition methods, PBB, trajectory modality argument); add to mkdocs.yml nav

### Phase 10: Methods Section
**Goal**: The Methods section is fully written as the canonical algorithm specification for Rune — precise enough to guide Phase 1 implementation — with formal pseudocode for the recursive loop, LaTeX equations for LoRA operations and the hypernetwork mapping, and explicit "specified vs TBD" markers throughout
**Depends on**: Phase 9
**Requirements**: ART-03
**Success Criteria** (what must be TRUE):
  1. A formatted pseudocode block describes the generate-execute-reflect-save loop with explicit typed inputs and outputs at each step; the block is precise enough that an implementer can map it directly to the LangGraph StateGraph nodes in `services/rune-agent`
  2. The Methods section contains rendered LaTeX equations for the LoRA adapter update rule (`ΔW = BA`), the hypernetwork mapping from trajectory to adapter weights, and at least one equation representing the evolution operator fitness criterion
  3. The adapter hierarchy (project, domain, task levels with compositional accumulation) and the Evolution Operator (consolidate/update/forget/merge operations) are described with explicit contrast against LoRA Soups; design decisions are labeled as "specified," "ablation target," or "TBD" throughout
  4. The training data strategy section cites PBB (arXiv:2506.18777) as empirical grounding for training on code execution trajectories without input-output pairs; the serving architecture section frames PP=2 + QLoRA as the intended configuration pending Phase 0 empirical validation (not asserted as confirmed working)
**Plans**: 1/1 plan complete

Plans:
- [x] 10-01-PLAN.md — Write docs/article/methods.md (system architecture, hypernetwork adaptation, adapter distillation pseudocode, evolution operator, hierarchy, PBB-grounded training strategy, hardware section)

### Phase 11: Results & Discussion Outlines
**Goal**: Results and Discussion sections exist as detailed placeholder structures — the experimental design is specific enough to be a real research proposal (named benchmarks, explicit hypotheses, concrete metrics) while clearly marked as planned work; Discussion covers limitations, future directions, and broader implications
**Depends on**: Phase 10
**Requirements**: ART-04, ART-05
**Success Criteria** (what must be TRUE):
  1. The Results section contains a Phase 1 kill-switch experimental design with: named evaluation set (HumanEval subset, 20-30 tasks), explicit kill-switch hypothesis (Pass@1 improvement threshold), adapter diversity metrics (`||ΔW||` norms, cosine similarity), and baseline comparisons (vanilla model, RAG, fine-tuned); every results table and figure placeholder is labeled "Planned Experiments"
  2. The Results section includes a PBB-inspired evaluation criterion: a description of testing whether a generated adapter enables program evaluation on held-out inputs without those inputs in context, grounded explicitly in PBB's experimental methodology
  3. The Discussion section has subsection headers for: expected contributions, limitations (pre-implementation status, adapter interference risk, hypernetwork mode collapse, cold-start corpus size), four open research questions (procedural encoding, recursive refinement value, composition interference, cold-start minimum), and future work (QDoRA, cross-project transfer, online adaptation)
  4. A visible research-status admonition or disclaimer appears at the top of the article (or in the Discussion Limitations subsection) stating explicitly that the system is pre-implementation and no experiments have been run
**Plans**: 2/2 plans complete

Plans:
- [x] 11-01-PLAN.md — Write docs/article/results.md (experimental design proposal, kill-switch hypothesis, PBB-inspired metric, ablation structure, placeholder tables)
- [x] 11-02-PLAN.md — Write docs/article/discussion.md (contributions, limitations, open questions, future work, broader implications, research-status disclaimer)

### Phase 12: Abstract + Quality Audit
**Goal**: The abstract is written after all other sections exist and accurately summarizes what was written; all citation footnotes resolve to bibliography anchors; `uv run mkdocs build` exits 0 with no warnings; the article is complete and publication-ready within the MkDocs site
**Depends on**: Phase 11
**Requirements**: ART-01, REF-02, REF-03
**Success Criteria** (what must be TRUE):
  1. The Abstract section (approximately 250 words) states the problem (context window limitations for local coding agents), the approach (parametric episodic memory via composable LoRA adapters), the mechanism (Doc-to-LoRA hypernetwork with evolutionary selection), and the expected contribution — using three-tier claim vocabulary (validated/expected/proposed) throughout
  2. Every footnote citation in every article section (`[^citekey]`) renders as a numbered superscript that hyperlinks to a named anchor in `references.md`; clicking the link in a built site navigates to the correct bibliography entry
  3. `uv run mkdocs build` exits 0 with no warnings; the "Scientific Article" nav section appears correctly with all section pages (index, abstract, background, methods, results, discussion, references) reachable; all internal links resolve
  4. A claim-tier audit is complete: every sentence in the article containing a performance verb (achieves, improves, enables, demonstrates) is either backed by a citation or labeled with a claim tier (expected/proposed); no present-tense empirical claims appear without qualification
**Plans**: 2/2 plans complete

Plans:
- [x] 12-01-PLAN.md — Write docs/article/abstract.md and docs/article/index.md (cover page with linked TOC, author block, research-status admonition)
- [x] 12-02-PLAN.md — Claim-tier audit, citation link verification, final mkdocs build --strict gate

</details>

---

### v4.0 API Wireframes & TDD Foundation (In Progress)

**Milestone Goal:** Expand all service and library scaffolds with complete API wireframes — every public method has a Google-style docstring and raises NotImplementedError, every method has a failing test, and shared test fixture factories maximize DRY across the test suite.

**Build constraint:** Test infrastructure (Phase 13) must come first — all other phases depend on shared fixtures. Library wireframes (Phases 14-15) precede service wireframes (Phase 16) because services import from libs. Quality gate (Phase 17) is terminal.

**Branch:** feat/v4-wireframes

## Phase Details

### Phase 13: Test Infrastructure
**Goal**: Shared test fixture factories exist at the root and in every component conftest.py — `pytest` can discover and inject any shared factory fixture into any component test without duplication; the factory functions produce valid, typed domain objects covering all core Rune models
**Depends on**: Phase 12 (v3.0 complete)
**Requirements**: TINF-01, TINF-02
**Success Criteria** (what must be TRUE):
  1. A root-level `conftest.py` exports `make_adapter_record`, `make_coding_session`, `make_training_job`, `make_evolution_job`, `make_evol_metrics`, and `make_adapter_ref` as pytest fixtures; a test in any workspace component can use `make_adapter_record()` without importing it explicitly
  2. Each of the 11 components (6 libs + 5 services) has a `conftest.py`; service conftest files provide a `test_client` fixture returning a FastAPI `TestClient`; the inference component conftest provides a mock vLLM client fixture
  3. Calling any factory fixture with no arguments returns a valid object that passes Pydantic model validation; calling with keyword overrides (e.g., `make_adapter_record(task_type="code-gen")`) produces an object with the override applied
**Plans**: 2 plans

Plans:
- [x] 13-01-PLAN.md — Create root conftest.py with 6 shared factory fixtures (make_adapter_record, make_coding_session, make_training_job, make_evolution_job, make_evol_metrics, make_adapter_ref)
- [x] 13-02-PLAN.md — Create per-component conftest.py files for all 11 components with component-specific fixtures (TestClient for services, mock vLLM for inference)

### Phase 14: Core Library Wireframes
**Goal**: The four existing library scaffolds — adapter-registry, model-training, shared, and events-py — are upgraded to full API wireframes: every public method has a Google-style docstring with Args, Returns, Raises, and Example sections, and every method has a failing TDD test asserting expected signature and behavior
**Depends on**: Phase 13
**Requirements**: LIB-05, LIB-08, LIB-09, LIB-10
**Success Criteria** (what must be TRUE):
  1. All 4 CRUD methods in `libs/adapter-registry` (`store`, `retrieve_by_id`, `query_by_task_type`, `list_all`) have Google-style docstrings; each has a test that fails with `NotImplementedError` and asserts the expected return type signature
  2. All 9 functions across `libs/model-training` (`config.py`, `peft_utils.py`, `trajectory.py`) have Google-style docstrings; each has a test that calls the function and expects `NotImplementedError`
  3. `AdapterRef`, `CodingSession`, and `EvolMetrics` in `libs/shared` have Google-style class and field docstrings; tests validate field types, required vs optional fields, defaults, and round-trip JSON serialization
  4. `create_event` and associated models in `libs/events-py` have Google-style docstrings; tests cover edge cases: missing payload, invalid kind, and custom event_id override
**Plans**: 3 plans

Plans:
- [x] 14-01-PLAN.md — Upgrade libs/adapter-registry: Google-style docstrings on all 4 CRUD methods; write failing TDD tests for each
- [x] 14-02-PLAN.md — Upgrade libs/model-training: Google-style docstrings on all 9 functions; write failing TDD tests for each
- [x] 14-03-PLAN.md — Upgrade libs/shared models and libs/events-py: Google-style docstrings; write failing TDD tests for models and event edge cases

### Phase 15: New & Reworked Library Wireframes
**Goal**: The two libraries requiring deeper structural changes are completed — libs/inference has template-era code removed and replaced with Rune-specific wireframes; libs/evaluation is built from scratch with a complete 6-function public API; both have full Google-style docstrings and failing TDD tests
**Depends on**: Phase 14
**Requirements**: LIB-06, LIB-07
**Success Criteria** (what must be TRUE):
  1. `libs/inference` contains no references to Vertex AI or LangChain; `loaders.py` and `factory.py` from the template era are deleted; `adapter_loader.py` exports `load_adapter`, `unload_adapter`, and `list_loaded_adapters`; new `completion.py` exports `generate_completion`, `generate_with_adapter`, and `batch_generate`; all 6 functions have Google-style docstrings and raise `NotImplementedError`
  2. `libs/evaluation` is a uv workspace member with at least 6 public functions: `run_humaneval_subset`, `calculate_pass_at_k`, `score_adapter_quality`, `compare_adapters`, `test_generalization`, `evaluate_fitness`; each has a Google-style docstring with Args, Returns, Raises, and Example sections
  3. Every public function in both `libs/inference` and `libs/evaluation` has a failing TDD test; inference tests use the mock vLLM client fixture from the component conftest; evaluation tests verify expected return types and that `NotImplementedError` is raised
**Plans**: 2 plans

Plans:
- [ ] 15-01-PLAN.md — Rework libs/inference: remove template code (loaders.py, factory.py); expand adapter_loader.py; add completion.py; write failing TDD tests for all 6 functions
- [ ] 15-02-PLAN.md — Wireframe libs/evaluation from scratch: scaffold module, 6 public functions with Google-style docstrings and NotImplementedError; write failing TDD tests

### Phase 16: Service Wireframes
**Goal**: All five services have complete API wireframes — every endpoint handler, node function, and client method has a Google-style docstring with Args, Returns, Raises sections, and every function has a failing TDD test asserting the expected response schema and status code (for HTTP endpoints) or return type (for internal functions)
**Depends on**: Phase 15
**Requirements**: SVC-06, SVC-07, SVC-08, SVC-09, SVC-10
**Success Criteria** (what must be TRUE):
  1. All 6 endpoint functions in `services/api-service` (`list_adapters`, `get_adapter`, `create_adapter`, `list_sessions`, `get_session`, `create_session`) have Google-style docstrings; each has a TestClient test asserting the expected response schema structure and HTTP 501 status code
  2. All 4 endpoint functions in `services/evolution-svc` and all 3 in `services/training-svc` have Google-style docstrings; each has a TestClient test asserting the expected response shape and HTTP 501 status
  3. All 4 node functions and 2 graph functions in `services/rune-agent` (`generate_node`, `execute_node`, `reflect_node`, `save_trajectory_node`, `should_retry`, `create_graph`) have Google-style docstrings; each node function has a test asserting expected state key mutations; `should_retry` test covers both retry and terminal branches
  4. `check_vllm_ready` and all `VLLMClient` methods in `services/lora-server` have Google-style docstrings; each has a failing test (lora-server is Dockerfile-only, so tests run against the Python source directly, not via TestClient)
**Plans**: TBD

Plans:
- [ ] 16-01-PLAN.md — Upgrade services/api-service: Google-style docstrings on 6 endpoints; write failing TestClient tests for each
- [ ] 16-02-PLAN.md — Upgrade services/evolution-svc and services/training-svc: Google-style docstrings on all endpoints; write failing TestClient tests
- [ ] 16-03-PLAN.md — Upgrade services/rune-agent: Google-style docstrings on 4 nodes + 2 graph functions; write failing tests for node state mutations and should_retry branches
- [ ] 16-04-PLAN.md — Upgrade services/lora-server: Google-style docstrings on check_vllm_ready and VLLMClient methods; write failing tests (direct Python, no TestClient)

### Phase 17: Quality Gate
**Goal**: The TDD red phase is verified — every new test fails with the correct failure mode (NotImplementedError or assertion failure, never unexpected pass), every public method across all 11 components has a complete Google-style docstring, and the full workspace passes ruff and mypy cleanly
**Depends on**: Phase 16
**Requirements**: QA-05, QA-06, QA-07
**Success Criteria** (what must be TRUE):
  1. `uv run pytest` reports the expected pattern: all TDD wireframe tests fail (with `NotImplementedError` or assertion failure), zero tests fail unexpectedly, and zero new tests pass unexpectedly — the test suite is in a confirmed red phase
  2. A docstring coverage audit across all 11 components (6 libs + 5 services) finds zero public methods or functions missing a Google-style docstring with at minimum Args, Returns, and Raises sections
  3. `uv run ruff check .` exits 0 with no errors or warnings; `uv run mypy services/*/src libs/*/src` exits 0 with no type errors — all new wireframe code passes static analysis without exceptions or ignores
**Plans**: TBD

Plans:
- [ ] 17-01-PLAN.md — Run full pytest suite; verify all TDD tests fail with NotImplementedError or assertion failure; fix any unexpected passes; confirm red-phase pattern
- [ ] 17-02-PLAN.md — Docstring coverage audit across all 11 components; fix any missing or incomplete Google-style docstrings; run ruff + mypy gate to completion

## Progress

**Execution Order:**
Phases execute in numeric order: 13 → 14 → 15 → 16 → 17

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
| 13. Test Infrastructure | 1/2 | Complete    | 2026-03-03 | - |
| 14. Core Library Wireframes | v4.0 | Complete    | 2026-03-03 | 2026-03-03 |
| 15. New & Reworked Library Wireframes | 1/2 | In Progress|  | - |
| 16. Service Wireframes | v4.0 | 0/4 | Not started | - |
| 17. Quality Gate | v4.0 | 0/2 | Not started | - |
