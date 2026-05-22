# Rune v2 Handoff

## Branch & Location
- **Branch**: `feat/rune-v2`
- **Worktree**: `/Users/noahdolevelixir/Code/rune-v2`
- **Source repo**: `/Users/noahdolevelixir/Code/rune` (original, on `feat/pipeline-speed-p4-p5`)
- **Latest commit**: `f8e7985` — docs: add MkDocs config, API reference stubs, and Google-style docstrings

## What's Built (18 commits, all green)
| Layer | Status | Key Files |
|---|---|---|
| Infrastructure | Complete | pyproject.toml, Makefile, CLAUDE.md, mkdocs.yml |
| Templates | Copied from v1 | src/rune/templates/*.j2 (15 files) |
| State types | Complete + tested | src/rune/engine/state.py |
| Config | Complete + tested | src/rune/config.py |
| Sandbox | Complete + tested | src/rune/sandbox/executor.py |
| Registry | Complete + tested | src/rune/registry/store.py |
| Parse | Complete + tested | src/rune/engine/parse.py |
| Policy | Complete + tested | src/rune/engine/policy.py |
| Interfaces | Complete + tested | src/rune/engine/interfaces.py |
| Graph (engine) | Complete + tested | src/rune/engine/graph.py |
| Model layer | Stubs (GPU-dependent) | src/rune/model/{inference,hypernetwork,adapter}.py |
| Training | Carried from v1 + gate tested | src/rune/training/{diff_loss,oracle_cache,config,gate}.py |
| CLI | Complete + tested | src/rune/cli.py |
| Benchmark | Stub | src/rune/bench/runner.py |
| Docs | MkDocs + Google docstrings | mkdocs.yml, docs/**, all src files |

## Quality Gate (all passing)
- `uv run ruff check .` — clean
- `uv run ruff format .` — clean
- `uv run mypy src/` — 0 errors, 24 files
- `uv run pytest tests/ -q` — 58 passed

## Next Steps (in order)

### 1. Fill PRODUCT.md
CLAUDE.md requires this before any non-trivial change. Currently it's the Template placeholder. Fill with Rune v2's actual product context.

### 2. Wire CLI to Engine (GPU required)
Connect `rune run` in cli.py to `create_engine()` from graph.py:
- Load base model + tokenizer via transformers
- Load hypernetwork checkpoint
- Create model wrapper that `step_node` expects (needs `.generate_adapter()`, `.hotswap_adapter()`, `.generate()`)
- Invoke engine with `engine.ainvoke(initial_state, config={"configurable": {"model": model}})`
- The `run` command currently raises `NotImplementedError`

### 3. Implement Benchmark Runner
Fill `run_benchmark()` in bench/runner.py:
- For each BenchTask, invoke engine with task description
- Extract generated code, run test_code against it
- Collect TaskResult, compute pass@1
- Wire to `rune bench` CLI command

### 4. Mode 2: Template Experimentation
- MBPP baseline → Pass@1 measurement
- Iterate on Jinja2 templates in src/rune/templates/
- Additional benchmarks beyond MBPP

## Key Architecture Decisions
- **Single-loop engine**: One `step_node` + `should_continue` conditional edge in LangGraph
- **Policy is deterministic**: `select_action()` is a pure function of state, no LLM calls
- **DAG layering**: `build_execution_layers()` uses `graphlib.TopologicalSorter` for parallel-safe subtask batching
- **Two-stage generation**: Free-form thinking (stop at `</think>`) → outlines Pydantic-constrained output (KV-cache continuation)
- **GPU imports deferred**: All torch/transformers/peft/outlines imports inside function bodies for CPU-only CI

## Reference Docs
- **Spec**: docs/superpowers/specs/2026-05-21-rune-v2-simplification-design.md
- **Plan**: docs/superpowers/plans/2026-05-22-rune-v2-implementation.md
- **Backup copies**: /tmp/rune-v2-docs/ (may not persist across reboots)
