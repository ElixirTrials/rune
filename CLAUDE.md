# Rune

Local-first coding agent that encodes coding trajectories into LoRA adapters, building persistent weight-space episodic memory. 4-phase pipeline (decompose → plan → code → integrate), parallel swarm orchestration, Doc-to-LoRA hypernetwork, TIES/DARE merging, adapter registry with lineage tracking.

## Running Tests

```bash
uv sync --all-extras
uv run pytest                    # 301+ tests, ~1 min on CPU
uv run pytest -x                 # stop on first failure
uv run pytest tests/             # root-level integration tests only
uv run ruff check                # lint
uv run mypy                      # type check
```

## Key Entry Points

- `scripts/rune_runner.py` — Single pipeline run (4-phase: decompose → plan → code → integrate)
- `scripts/swarm.py` — Multi-agent swarm orchestrator (agents + training pool + evolution + watchdog)
- `scripts/e2e_test.py` — End-to-end test exercising full pipeline
- `scripts/swarm_workers.py` — Training pool manager (QLoRA in subprocess, vLLM sleep/wake)
- `scripts/swarm_evolution.py` — Evolution worker (TIES/DARE merge, pruning, lineage)

## Conventions

- **Docstrings:** Google style
- **Linting:** `ruff` (line-length 88, target py312)
- **Types:** `mypy` with strict-ish config
- **Deps:** `uv` for everything (sync, run, lock)
- **GPU imports:** Deferred inside function bodies (INFRA-05 pattern) — modules stay importable in CPU-only CI

## Architecture

- `scripts/` — Fat orchestrator layer; this is where the pipeline and swarm logic lives
- `libs/` — Reusable components (adapter-registry, inference, model-training, shared, evaluation, events-py)
- `services/` — FastAPI microservices (training-svc, evolution-svc, rune-agent, api-service)
- `docs/` — MkDocs documentation site

The scripts layer is the primary execution path. Services provide REST APIs but the swarm bypasses them for local execution.

## Important Files

- `libs/shared/src/shared/rune_models.py` — Cross-service data contracts (CodingSession, SwarmConfig, PipelinePhase, etc.)
- `libs/shared/src/shared/templates/*.j2` — Jinja2 templates for each pipeline phase
- `libs/model-training/src/model_training/hypernetwork.py` — DocToLoraHypernetwork (Perceiver-based)
- `libs/model-training/src/model_training/merging.py` — TIES/DARE adapter merging
- `libs/adapter-registry/src/adapter_registry/registry.py` — AdapterRegistry (SQLite CRUD)
- `libs/inference/src/inference/provider.py` — InferenceProvider ABC
- `libs/shared/src/shared/sandbox.py` — SubprocessBackend for code execution

## Template Editing

Pipeline phase templates live in `libs/shared/src/shared/templates/`:
- `decompose.j2` / `prompt_decompose.j2`
- `plan.j2` / `prompt_plan.j2`
- `code.j2` / `code_retry.j2` / `prompt_code.j2`
- `integrate.j2` / `prompt_integrate.j2`

Templates are rendered via `shared.template_loader.render_trajectory()` and `render_prompt()`.
