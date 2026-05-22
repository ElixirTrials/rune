# Rune

Local-first coding agent that encodes coding trajectories into LoRA adapters via a perceiver hypernetwork. Single-loop LangGraph engine with four concerns: mine, train, run, benchmark.

## Read first
- **`PRODUCT.md`** — read before any non-trivial change. If missing or contains `<!-- TODO -->` stubs, stop and ask the user to fill it in.

## Stack
- `uv` Python 3.12 single package.
- Engine: LangGraph single-loop (`src/rune/engine/`).
- Model: outlines + PEFT + transformers (`src/rune/model/`).
- Training: oracle + hypernetwork distillation + DiffAwareSFTTrainer (`src/rune/training/`).
- Quality: ruff, mypy (strict), pytest.

## Hard rules
**Long-running ops** — never execute. Ask the user to run and log.
**Deploy / install** — never.
**GPU imports** — deferred inside function bodies (importable in CPU-only CI).

## Running Tests
```bash
uv sync
uv run pytest tests/unit/ -q       # fast, no GPU
uv run pytest tests/ -q             # all tests
uv run pytest tests/gpu/ -m gpu -q  # GPU only
uv run ruff check .                 # lint
uv run mypy src/                    # type check
```

## Style
- No preamble. No restating the question.
- Diff-style edits over rewrites.
- No emoji unless asked. No comments unless the *why* is non-obvious.
- Always use `uv run` to launch Python.

## Key Entry Points
- `src/rune/cli.py` — typer CLI: `rune run`, `rune train`, `rune mine`, `rune bench`
- `src/rune/engine/graph.py` — LangGraph StateGraph with single step_node
- `src/rune/engine/policy.py` — deterministic action selection + DAG layer grouping
- `src/rune/config.py` — PipelineConfig frozen dataclass
