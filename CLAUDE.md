# Rune

Local-first coding agent that encodes coding trajectories into LoRA adapters via a perceiver hypernetwork (ctx-to-lora / HyperLoRA). Single-loop LangGraph engine with four concerns: mine, train, run, benchmark.

## Read first
- **`PRODUCT.md`** — read before any non-trivial change. If missing or contains `<!-- TODO -->` stubs, stop and ask the user to fill it in.

## Stack
- `uv` Python 3.12 single package.
- Engine: LangGraph single-loop (`src/rune/engine/`).
- Model: xgrammar (structured output) + PEFT + transformers (`src/rune/model/`).
- Training: hypernetwork distillation via DiffAwareSFTTrainer (`src/rune/training/`).
- Base model: Qwen/Qwen3-4B-Instruct-2507 — the *instruct* variant, required so the pre-warmed Sakana doc-to-lora adapter (hypernet warm start) stays compatible. Single source of truth: `config.yaml` / `RUNE_BASE_MODEL` via `rune.config.load_rune_config()`; never hardcode a model id.
- Quality: ruff, mypy (strict), pytest.

## Hard rules
**GPU / long-running ops** — OK to run directly on this GPU instance (engine runs, smoke tests, training, benchmarks). Capture/log output; prefer background runs for multi-minute jobs.
**Deploy / install** — never.
**GPU imports** — deferred inside function bodies (importable in CPU-only CI).
**CPU RAM is tiny (~15GB).** Always check `free -g` before loading the base model + hypernet or setting `offload_base=True` — moving the base model (~8GB bf16 for the 4B-Instruct base) to CPU RAM can OOM-kill the VM. Prefer `offload_base=False` (base+hypernet fit comfortably in the 23GB GPU). When loading risks OOM, run under a RAM watchdog that kills the job before the VM dies.

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

## Architecture

### Engine loop (`src/rune/engine/`)
Single `step_node` in a LangGraph StateGraph. Each iteration: `select_action` → render Jinja2 templates → generate adapter via hypernetwork → hot-swap LoRA weights → model.generate → sandbox execution → parse output → update state. Loop terminates when budget exhausted or all subtasks integrated.

Action sequence: `decompose → plan → code → [diagnose → repair]* → integrate`.

Subtask dependencies form a DAG; independent subtasks in the same layer run in parallel.

### Hypernetwork (`src/rune/model/`)
`ctx-to-lora` HyperLoRA perceiver: trajectory text → base model activations → per-layer LoRA A/B weights → PEFT hot-swap. Continuation rounds use scaled adapter (`cont_multiplier ≈ 1.53` over base scaling).

### Training (`src/rune/training/`)
Three-stage pipeline: oracle QLoRA → hypernetwork distillation (DiffAwareSFTTrainer + KL+CE) → success gate. Oracle stage is currently a stub.

### Mining (`src/rune/mining/`)
Session scanner → trajectory extractor → per-action JSONL shards, keyed by `{action}_{benchmark}`.

## Key Entry Points
- `src/rune/cli.py` — typer CLI: `rune run`, `rune train`, `rune mine`, `rune bench`
- `src/rune/engine/graph.py` — LangGraph StateGraph with single step_node + continuation sub-loop
- `src/rune/engine/policy.py` — deterministic action selection + DAG layer grouping
- `src/rune/engine/state.py` — RunState TypedDict, Action/Subtask/Feedback dataclasses
- `src/rune/config.py` — PipelineConfig frozen dataclass
- `src/rune/model/wrapper.py` — ModelWrapper bridges engine to hypernetwork + inference
- `src/rune/model/hypernetwork.py` — HyperLoRA loader + adapter weight generation
- `src/rune/model/inference.py` — xgrammar-constrained generation with thinking phase
- `src/rune/training/orchestrator.py` — three-stage training pipeline
- `src/rune/bench/runner.py` — benchmark runner with pass@1 scoring
- `src/rune/bench/hpo.py` — Optuna HPO over engine params
