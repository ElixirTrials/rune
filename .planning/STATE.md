# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-03)

**Core value:** A local coding agent that learns from its own coding trajectories, building persistent parametric memory that scales independently of context window size.
**Current focus:** Milestone v3.0 — Scientific Article Documentation

## Current Position

Phase: 11 — Results & Discussion
Plan: 01 of 02 (01 complete)
Status: Phase 11 plan 01 complete; ready for plan 02
Last activity: 2026-03-03 — Phase 11 plan 01 complete (results.md + mkdocs.yml nav)

Progress: [███████░░░] 70% of v3.0 (v1.0 complete; v2.0 complete; Phases 8-10 done; 11-01 done)

## Performance Metrics

**Velocity:**
- Total plans completed: 22
- Average duration: 3.1 min
- Total execution time: ~1.2 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-readme | 1 | 6 min | 6 min |
| 02-implementation-plan | 2 | 6 min | 3 min |
| 03-architecture-docs | 1 | ~5 min | ~5 min |
| 04-cleanup | 3 | 11 min | 3.7 min |
| 05-foundation-libraries | 3 | 8 min | 2.7 min |
| 05.1-template-artifact-cleanup | 2 | 7 min | 3.5 min |
| 06-service-scaffolds | 4 | 10 min | 2.5 min |
| 07-configuration-quality-gate | 3 | 7 min | 2.3 min |
| 08-mkdocs-infrastructure | 2 | 6 min | 3 min |
| 09-references-skeleton-background | 2 | 8 min | 4 min |
| 10-methods-section | 1 | 4 min | 4 min |
| 11-results-discussion | 1 | 5 min | 5 min |

**Recent Trend:**
- Last 5 plans: 2 min, 3 min, 3 min, 4 min, 5 min
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
- [Phase 07-01]: Used --import-mode=importlib in pytest addopts to resolve same-basename test file collision across workspace components with -n auto
- [Phase 07-02]: Used Compose Spec v3.8+ deploy.resources.reservations.devices (not deprecated runtime: nvidia) for GPU passthrough
- [Phase 07-02]: Health sidecar port 8001 used for healthcheck, not vLLM port 8000 directly
- [Phase 07-02]: start_period: 120s set to accommodate 60-120s vLLM 7B model load time
- [Phase 07-03]: lora-server excluded from mkdocs nav — Dockerfile-only with no public Python API (Phase 6 locked decision)
- [Phase 07-03]: Component nav entries ordered alphabetically within Components section
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
- [Phase 06-service-scaffolds]: Used APIRouter with prefix='/adapters' and prefix='/sessions' for router stub pattern
- [Phase 06-service-scaffolds]: Root pyproject.toml five-section sync completed (members, mypy, pythonpath, testpaths, coverage)
- [Milestone v3.0]: Scientific article documentation — 5 phases (8-12): Infrastructure → References+Background → Methods → Results+Discussion → Abstract+Audit
- [Roadmap v3.0]: Infrastructure (Phase 8) MUST precede all content writing — arithmatex misconfiguration is a silent fatal pitfall
- [Roadmap v3.0]: references.md (Phase 9) created before section files — anchors: warn setting validates citation links from first commit onward
- [Roadmap v3.0]: Abstract (Phase 12) written LAST — summarizes what was actually written, not what was planned
- [Roadmap v3.0]: Do NOT use $ delimiters for math — use \(...\) and \[...\] exclusively to avoid arithmatex dollar-sign collision with prose
- [Roadmap v3.0]: Do NOT use mkdocs-bibtex — archived November 2025, requires Pandoc system binary; use built-in footnotes extension instead
- [Roadmap v3.0]: Use MathJax 3 (not KaTeX) — ML theory paper requires \begin{align}, \mathbb, \mathcal which KaTeX handles inconsistently
- [Roadmap v3.0]: vLLM LoRA + PP=2 + QLoRA compatibility is MEDIUM-LOW confidence; Methods section must frame PP=2 as intended config pending Phase 0 validation, not confirmed working
- [Roadmap v3.0]: PBB (arXiv:2506.18777) is a must-cite cross-cutting citation — appears in Background, Methods, Results, and Experimental Design sections
- [Roadmap v3.0]: Three-tier claim vocabulary (validated/expected/proposed) must be established before writing Abstract to prevent overclaiming on an unbuilt system
- [Phase 08-01]: Dollar delimiters disabled explicitly via inline_syntax: [round], block_syntax: [square, begin] — prevents $5.00 prose false positives
- [Phase 08-01]: mathjax-config.js must appear before MathJax CDN in extra_javascript — window.MathJax read at startup; reversed order causes silent misconfiguration
- [Phase 08-01]: tags: "ams" placed inside tex block (not options) for AMS-style equation numbering
- [Phase 08-01]: ignoreHtmlClass: ".*" without trailing pipe — arithmatex official pattern; ".*|" is Material theme-specific
- [Phase 08-02]: attr_list + md_in_html enabled together — required for div.abstract markdown=1 pattern; neither extension useful alone for this use case
- [Phase 08-02]: academic.css uses CSS custom property fallbacks — var(--rune-accent, #22c55e) ensures correct rendering regardless of custom.css load order
- [Phase 08-02]: Scientific Article nav uses section-with-children (not direct page link) — extensible for Phase 9-12 child pages without nav restructuring
- [Phase 09-02]: All 12 footnotes defined in background.md (plan said 10, but content plan lists all 12 papers — 12 is correct and complete)
- [Phase 09-02]: Trajectory modality argument uses a 4-row comparison table for clarity over prose enumeration
- [Phase 09-02]: Doc-to-LoRA scope caveat and PBB in-context reliability caveat both stated explicitly in prose
- [Phase 10-01]: Used text fenced code block for pseudocode (not math block) to avoid LaTeX rendering conflicts
- [Phase 10-01]: Perceiver-based encoder labeled as ablation target (not specified) — alternatives to be compared in Phase 1
- [Phase 10-01]: 9 footnotes defined in methods.md (subset of background.md's 12 — excluded pink2025episodic, charakorn2025t2l, liu2026shine)
- [Phase 11-01]: Added inline citations for charakorn2026doc2lora, dettmers2023qlora, sheng2023slora to eliminate unreferenced footnote build warnings

### Pending Todos

None.

### Blockers/Concerns

None yet.

## Session Continuity

Last session: 2026-03-03
Stopped at: Completed 11-01-PLAN.md
Resume file: None
