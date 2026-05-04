# Rune

Local-first coding agent that encodes coding trajectories into LoRA adapters, building persistent weight-space episodic memory. 5-phase pipeline (decompose → plan → code → integrate → diagnose/repair), parallel swarm orchestration, Doc-to-LoRA hypernetwork with Sakana perceiver, TIES/DARE merging, adapter registry with lineage tracking.

## Read first
- **`PRODUCT.md`** — read before any non-trivial change. If missing or contains `<!-- TODO -->` stubs, stop and ask the user to fill it in. Cite specific sections (`do-not-break:`, `out-of-scope:`, `regulatory-surface:`) when justifying decisions.
- **`.claude/skills-index.md`** — trigger table. Match the user's keywords to a skill, then load just that skill.

## Stack
- `uv` Python 3.12 workspace.
- Backend: FastAPI services under `services/` (`api-service`, `training-svc`, `evolution-svc`, `lora-server`, `rune-agent`). Agents use LangGraph.
- Libs: `libs/<name>` (Python). Common: `shared`, `events-py`, `adapter-registry`, `inference`, `model-training`, `corpus-producer`, `evaluation`.
- Infra: `infra/docker-compose.yml` (Postgres + MLflow + litestream). Scripts: `scripts/`. Docs: `docs/` + `mkdocs.yml`.
- Quality: ruff, mypy (strict-ish), pytest+xdist+cov.
- Cloud: AWS (ECS Fargate / RDS / S3 / Secrets Manager / EventBridge+SQS / SageMaker).

## Hard rules

**Git** — never run mutating git (`commit`, `push`, `merge`, `rebase`, `reset`, `checkout`, `branch`, `stash`, `tag`, `cherry-pick`, `revert`). Hook-enforced. Read-only git only when asked.

**Long-running ops** — never execute (hook-blocked). Ask the user to run and log. See `long-run-fallback` skill for the exact protocol.

**Deploy / install** — never. Hook-enforced. Includes installers, migrators, deployers — all user-managed.

**Migrations** (alembic) — never apply. Generate, read the SQL, flag risk. See `alembic-safe` skill.

**Model routing** — default **Opus**. For trivially-scoped, single-file, read-only work: switch to Sonnet (`/sonnet`) without being asked. State the switch. Skills `senior-architect`, `compliance-check`, `ux-pathway`, `senior-devops` require Opus (hook-enforced).

## Style
- No preamble. No restating the question. No closing summary unless asked.
- Glob/Grep before Read. Never open a file speculatively to "see what's in it".
- Diff-style edits over rewrites. Edit existing files; don't create new ones unless required.
- Plan-then-execute for any change touching 2+ files (`plan-first` skill).
- One focused question when uncertain. Not multiple speculative paths.
- No emoji unless asked. No comments unless the *why* is non-obvious.
- Don't read lockfiles, dist, build, caches, node_modules, .venv (hook-enforced as backup).

## Running Tests

```bash
uv sync --all-extras
uv run pytest                    # 776+ tests, ~30s on GPU
uv run pytest -x                 # stop on first failure
uv run pytest tests/             # root-level integration tests only
uv run ruff check                # lint
uv run mypy libs/ services/      # type check
```

## Key Entry Points

- `scripts/rune_runner.py` — Single pipeline run (5-phase: decompose → plan → code → integrate → diagnose/repair) with DAG-ordered code execution
- `scripts/swarm.py` — Multi-agent swarm orchestrator (agents + training pool + evolution + watchdog)
- `scripts/e2e_test.py` — End-to-end test exercising full pipeline
- `scripts/benchmark_challenging.py` — 3-task end-to-end benchmark
- `scripts/optimization/run_optimization.py` — Bayesian parameter optimization (Optuna)
- `scripts/optimization/run_training_hpo.py` — HPO overhaul (Optuna + Hyperband pruner, hunk-weighted metrics, 4-bit NF4 heldout eval)
- `scripts/experiment_harness.py` — Isolated adapter/prompt experiments (~15s/trial)
- `scripts/swarm_workers.py` — Training pool manager (QLoRA in subprocess, vLLM sleep/wake)
- `scripts/swarm_evolution.py` — Evolution worker (TIES/DARE merge, pruning, lineage)
- `scripts/train.sh` — Unified training CLI wrapper (threads warmup_ratio, LoRA overrides, NEFTune, diff-aware loss)
- `scripts/phase_corpus_producer.py` — 25-bin oracle corpus producer (GPU sharding via `--shard IDX/TOTAL --cuda-visible-devices`)
- `scripts/train_round2.py` — Round-2 oracle-teacher distillation training
- `scripts/evaluate_round2.py` — Strict success gate (≥4/6 benchmarks ≥2.0% Pass@1, no regression >1.0%); exits 0 PASS / 1 FAIL
- `scripts/validate_oracles.py` — Per-oracle validator (≥3% Pass@1 improvement over base)

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

- `libs/shared/src/shared/pipeline_config.py` — PipelineConfig frozen dataclass (adapter, generation, prompt, trajectory settings)
- `libs/shared/src/shared/blackboard.py` — Typed blackboard for DAG-ordered code phase (SubtaskArtifact, interface extraction, topological sort)
- `libs/shared/src/shared/rune_models.py` — Cross-service data contracts (CodingSession, SwarmConfig, PipelinePhase, etc.)
- `libs/shared/src/shared/templates/*.j2` — Jinja2 templates for each pipeline phase
- `libs/model-training/src/model_training/sakana_d2l.py` — Sakana Doc-to-LoRA adapter generation (HyperLoRA perceiver → PEFT adapter)
- `libs/model-training/src/model_training/hypernetwork.py` — DocToLoraHypernetwork (Perceiver-based)
- `libs/model-training/src/model_training/merging.py` — TIES/DARE adapter merging
- `libs/model-training/src/model_training/diff_loss.py` — `DiffAwareSFTTrainer` + `DiffWeightedDataCollator` (hunk-weighted token loss, identity fallback)
- `libs/model-training/src/model_training/kill_switch.py` — Kill-switch wiring (≥5% HumanEval Pass@1 regression trigger, k=5, 20–30 held-out tasks)
- `libs/model-training/src/model_training/training_common.py` — `mlflow_log_params` shared helper
- `libs/model-training/src/model_training/round2_config.py` — `Round2TrainConfig` (Pydantic, inherits `D2LTrainConfig`)
- `libs/model-training/src/model_training/oracle_cache.py` — `OracleAdapterCache` (LRU max 4, stores `LoraDict` tensor dicts), bin-key lookup, coverage audit
- `libs/model-training/src/model_training/round2_train.py` — Round-2 training loop (`apply_functional_lora`, KL+CE loss, `train_d2l_qwen3_round2`, `register_round2_adapter`)
- `libs/model-training/src/model_training/round2_gate.py` — `evaluate_round2_gate` strict success gate
- `libs/adapter-registry/src/adapter_registry/registry.py` — AdapterRegistry (SQLite CRUD); reserved `task_type="round2_hypernet"`, `generation=2`, `parent_ids=json.dumps(sorted(oracle_ids))`
- `libs/corpus-producer/src/corpus_producer/trainer_bridge.py` — Sets oracle adapter IDs (`oracle_<bin_key>`)
- `libs/corpus-producer/src/corpus_producer/s3_uploader.py` — S3 manifest upload (lazy boto3 import, graceful degradation)
- `libs/inference/src/inference/provider.py` — InferenceProvider ABC (with temperature/top_p/repetition_penalty)
- `libs/shared/src/shared/sandbox.py` — SubprocessBackend for code execution

## Pipeline Configuration

Configuration lives at `~/.rune/pipeline_config.json`, loaded by `shared.pipeline_config.load_config()`. Key settings (from Bayesian optimization):

- `adapter.scaling`: 0.16 — adapter influence strength (Sakana's 45.25x is too aggressive)
- `adapter.use_bias`: true — concatenate bias as extra rank dimensions
- `generation.temperature`: 0.25 — low temperature for consistent output
- `generation.max_tokens`: 1024 — sufficient for subtask code
- `generation.repetition_penalty`: 1.04 — mild anti-repetition
- `prompt.style`: skeleton — code skeleton prompts dominate (20/20 in top optimization trials)
- `trajectory.style`: prose — natural language trajectories work best for adapter encoding

Override via env vars: `RUNE_TEMPERATURE`, `RUNE_MAX_TOKENS`, `RUNE_REPETITION_PENALTY`, `RUNE_TOP_P`.

## Template Editing

Pipeline phase templates live in `libs/shared/src/shared/templates/`:
- `decompose.j2` / `prompt_decompose.j2` / `prompt_decompose_concise.j2`
- `plan.j2` / `prompt_plan.j2`
- `code.j2` / `code_retry.j2` / `code_continue.j2` / `prompt_code.j2`
- `integrate.j2` / `prompt_integrate.j2`
- `diagnose.j2` / `prompt_diagnose.j2` — Phase 5 failure diagnosis
- `code_repair.j2` / `prompt_code_repair.j2` — Targeted subtask repair

Prompts orient the model (subtask name, project label, format directive). Adapters carry domain context via trajectory templates. See `instructions/adapter-research-findings.md` for detailed design rationale.

Templates are rendered via `shared.template_loader.render_trajectory()` and `render_prompt()`.

## DAG-Ordered Code Phase

Subtasks execute in dependency order via a typed blackboard (`libs/shared/src/shared/blackboard.py`):
- Decompose phase outputs `[depends: 1, 2]` declarations parsed by `_parse_subtask_list`
- `build_execution_layers()` topologically sorts subtasks into layers
- Layer 0 (no deps) runs first, publishes interfaces to blackboard
- Layer N reads predecessor interfaces from blackboard via adapter trajectory
- Backward compatible: missing `[depends:]` puts all subtasks in layer 0 (parallel)

## Two-Step Diagnose→Repair

When code fails, the retry loop uses a two-step approach:
1. **Diagnose:** Error in prompt ("crashes with: NameError..."), code in adapter → model produces concise fix instruction
2. **Repair:** Model's own diagnosis becomes the fix_guidance in prompt, domain stays in adapter → produces fixed code
This avoids the prompt-adapter tension where domain context and error details compete for model attention.

## Adapter Research

Comprehensive findings documented in `instructions/adapter-research-findings.md`:
- Three bugs fixed in Sakana D2L → PEFT conversion (combine_lora, alpha scaling, module paths)
- Adapter scaling 0.16x is optimal (full 45.25x causes degenerate repetition)
- Skeleton prompts + prose trajectories is the winning combination
- 200-trial Bayesian optimization across 5 diverse coding tasks

## Workflow shortcuts
- `make help` — top-level commands.
- `scripts/create-service.sh --lang py|ts --lib|--app <name>` — scaffold new package (use via `new-package` skill).

## Do not
- Create skills, hooks, or modify settings without explicit user request.
- Commit `.claude/runs/` — ephemeral log dumps; gitignored.
- Treat `.github/copilot-instructions.md` as authoritative for Claude — it's for Copilot. This file is the source of truth for Claude behavior.
- Add backwards-compatibility shims, dead-code re-exports, or feature flags Claude wasn't asked for.

<!-- code-review-graph MCP tools -->
## MCP Tools: code-review-graph

**IMPORTANT: This project has a knowledge graph. ALWAYS use the
code-review-graph MCP tools BEFORE using Grep/Glob/Read to explore
the codebase.** The graph is faster, cheaper (fewer tokens), and gives
you structural context (callers, dependents, test coverage) that file
scanning cannot.

### When to use graph tools FIRST

- **Exploring code**: `semantic_search_nodes` or `query_graph` instead of Grep
- **Understanding impact**: `get_impact_radius` instead of manually tracing imports
- **Code review**: `detect_changes` + `get_review_context` instead of reading entire files
- **Finding relationships**: `query_graph` with callers_of/callees_of/imports_of/tests_for
- **Architecture questions**: `get_architecture_overview` + `list_communities`

Fall back to Grep/Glob/Read **only** when the graph doesn't cover what you need.

### Key Tools

| Tool | Use when |
|------|----------|
| `detect_changes` | Reviewing code changes — gives risk-scored analysis |
| `get_review_context` | Need source snippets for review — token-efficient |
| `get_impact_radius` | Understanding blast radius of a change |
| `get_affected_flows` | Finding which execution paths are impacted |
| `query_graph` | Tracing callers, callees, imports, tests, dependencies |
| `semantic_search_nodes` | Finding functions/classes by name or keyword |
| `get_architecture_overview` | Understanding high-level codebase structure |
| `refactor_tool` | Planning renames, finding dead code |

### Workflow

1. The graph auto-updates on file changes (via hooks).
2. Use `detect_changes` for code review.
3. Use `get_affected_flows` to understand impact.
4. Use `query_graph` pattern="tests_for" to check coverage.

<!-- code-review-graph MCP tools -->
## MCP Tools: code-review-graph

**IMPORTANT: This project has a knowledge graph. ALWAYS use the
code-review-graph MCP tools BEFORE using Grep/Glob/Read to explore
the codebase.** The graph is faster, cheaper (fewer tokens), and gives
you structural context (callers, dependents, test coverage) that file
scanning cannot.

### When to use graph tools FIRST

- **Exploring code**: `semantic_search_nodes` or `query_graph` instead of Grep
- **Understanding impact**: `get_impact_radius` instead of manually tracing imports
- **Code review**: `detect_changes` + `get_review_context` instead of reading entire files
- **Finding relationships**: `query_graph` with callers_of/callees_of/imports_of/tests_for
- **Architecture questions**: `get_architecture_overview` + `list_communities`

Fall back to Grep/Glob/Read **only** when the graph doesn't cover what you need.

### Key Tools

| Tool | Use when |
|------|----------|
| `detect_changes` | Reviewing code changes — gives risk-scored analysis |
| `get_review_context` | Need source snippets for review — token-efficient |
| `get_impact_radius` | Understanding blast radius of a change |
| `get_affected_flows` | Finding which execution paths are impacted |
| `query_graph` | Tracing callers, callees, imports, tests, dependencies |
| `semantic_search_nodes` | Finding functions/classes by name or keyword |
| `get_architecture_overview` | Understanding high-level codebase structure |
| `refactor_tool` | Planning renames, finding dead code |

### Workflow

1. The graph auto-updates on file changes (via hooks).
2. Use `detect_changes` for code review.
3. Use `get_affected_flows` to understand impact.
4. Use `query_graph` pattern="tests_for" to check coverage.
