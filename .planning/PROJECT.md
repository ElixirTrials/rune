# Rune

## What This Is

Rune is a local-first coding agent that uses LoRA weight space as episodic memory, enabling unbounded context for Small Language Models (<10B params). Instead of stuffing tokens into context windows, Rune encodes procedural knowledge — architecture specs, execution feedback, coding conventions — directly into composable LoRA adapters via a Doc-to-LoRA hypernetwork. Each coding session produces reusable adapters that persist across sessions and compose hierarchically.

## Core Value

A local coding agent that learns from its own coding trajectories, building persistent parametric memory that scales independently of context window size.

## Requirements

### Validated

- ✓ README.md documenting project vision, architecture, and getting-started guide — v1.0
- ✓ Detailed, phased implementation plan refined from specification into actionable phases — v1.0
- ✓ Architecture documentation covering recursive loop, hypernetwork, evolution operator, adapter serving — v1.0
- ✓ Repo restructured to match implementation plan component layout — v2.0
- ✓ All template placeholders removed, services renamed and scaffolded — v2.0
- ✓ Quality gates pass: pytest, ruff, GPU import scan, 501 stubs, docker-compose, mkdocs — v2.0
- ✓ Scientific article documenting theoretical foundation (abstract, background, methods, results, discussion) — v3.0
- ✓ Complete API wireframes for all 11 components with Google-style docstrings — v4.0
- ✓ Every public method/function raises NotImplementedError with descriptive messages — v4.0
- ✓ Failing tests for every public method (TDD red phase: 87 tests, 13 expected failures) — v4.0
- ✓ Shared test fixture factories (conftest.py) for DRY test setup across components — v4.0

### Active

- [ ] Hardware and environment validation (Phase 0: dual RTX 4090, CUDA, PyTorch nightly, vLLM)
- [ ] Core hypothesis validation (kill-switch gate: Doc-to-LoRA on coding tasks, 5% Pass@1 improvement)
- [ ] Functional implementation of adapter-registry CRUD with SQLite persistence
- [ ] Functional implementation of rune-agent recursive loop (generate, execute, reflect, save)
- [ ] vLLM serving with adapter hot-loading via lora-server
- [ ] Training pipeline for LoRA adapter distillation from coding trajectories

## Current Milestone: v5.0 First Implementation

**Goal:** Transform wireframe stubs into working implementations across all components — from adapter registry to agent loop to vLLM serving — making the full Rune system functional end-to-end.

**Target features:**
- Hardware and environment validation (dual RTX 4090, CUDA, vLLM)
- adapter-registry CRUD with SQLite persistence
- vLLM serving with adapter hot-loading (lora-server)
- Training pipeline for LoRA adapter distillation
- rune-agent recursive loop (generate, execute, reflect, save)
- Core hypothesis validation (Doc-to-LoRA kill-switch gate)

### Out of Scope

- Mobile app — local-first desktop/server only
- Cloud API dependencies for inference — must run entirely on local hardware
- Tensor parallelism — confirmed vLLM bug #21471 with TP+LoRA on consumer GPUs; PP=2 required
- Frontend beyond basic HITL UI — focus is on agent backend infrastructure

## Context

**Shipped through v4.0.** Monorepo with 4,052 lines of Python across 11 components (6 libraries + 5 services). All services have complete API wireframes with Google-style docstrings and TDD failing tests. Scientific article documents the theoretical foundation.

**Tech stack:** Python 3.12+, FastAPI, LangGraph, SQLModel, Pydantic, uv workspace, MkDocs with MathJax.

**Hardware target:** Dual RTX 4090 (24GB each, 48GB total VRAM) with CXL interconnect, AMD Threadripper 7960X. QLoRA required for 24GB per-GPU constraint. Pipeline parallelism (PP=2, TP=1).

**Research basis:** Doc-to-LoRA (arXiv:2602.15902), S-LoRA (arXiv:2311.03285), QLoRA (arXiv:2305.14314), PBB (arXiv:2506.18777).

**Known tech debt from v4.0:** Root conftest.py rootdir isolation requires factory fixture duplication; POST stubs in api-service lack request body schemas; evaluation/model-training libs not yet wired to their consumer services.

## Constraints

- **Local-first**: Must run entirely on local hardware. No cloud API dependencies for inference.
- **Hardware**: Dual RTX 4090 + CXL + Threadripper 7960X (48GB VRAM total)
- **Monorepo**: Build on existing template structure (uv workspace, FastAPI)
- **Python**: Primary implementation language, using uv for package management
- **Research stage**: No working implementation exists — plan includes validation checkpoints

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Build on existing monorepo template | Reuse FastAPI/LangGraph infrastructure | ✓ Good (v2.0) |
| Dual RTX 4090 + CXL as target hardware | 48GB VRAM total, QLoRA required for 24GB per-GPU | — Pending |
| Doc-to-LoRA hypernetwork approach | Instant adapter generation (<1s) vs minutes for fine-tuning | — Pending |
| Documentation-first milestone | Validate approach through thorough planning before implementation | ✓ Good (v1.0) |
| Restructure before implement | Align monorepo layout to implementation plan components | ✓ Good (v2.0) |
| Scientific article before implementation | Sharpen algorithm design before coding | ✓ Good (v3.0) |
| TDD wireframes before implementation | Red-phase tests define expected behavior before writing logic | ✓ Good (v4.0) |
| Factory fixtures as test-only (not runtime) | Avoid runtime factory pattern complexity | ✓ Good (v4.0) |
| Centralized ruff config in root pyproject.toml | Per-component configs broke with dotted-key format | ✓ Good (v4.0) |
| lora-server as Dockerfile-only (not uv workspace) | GPU-dependent service doesn't fit CPU-only workspace | ✓ Good (v2.0) |

---
*Last updated: 2026-03-05 after v5.0 milestone started*
