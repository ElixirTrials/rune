# rune-agent

LangGraph state graph implementing the recursive code generation loop.

## Architecture

The agent defines a `StateGraph` with four nodes:

```
START → generate → execute → reflect → [should_retry] → generate (retry) OR save_trajectory → END
```

Two graph modes:
- **`create_graph()`** — Standard loop with retry logic (used standalone)
- **`create_single_iteration_graph()`** — Single iteration: generate → execute → reflect → END (used by `scripts/rune_runner.py`'s outer iteration loop, where the hypernetwork produces a new adapter between iterations)

## Key Files

| File | Purpose |
|------|---------|
| `graph.py` | Graph construction (`create_graph`, `create_single_iteration_graph`, `should_retry` router) |
| `nodes.py` | Node implementations (`generate_node`, `execute_node`, `reflect_node`, `save_trajectory_node`) |
| `state.py` | `RuneState` TypedDict defining the agent state |

## State

`RuneState` tracks: task description, generated code, execution results, test pass/fail, attempt count, max attempts, trajectory history.

## Retry Logic

`should_retry(state)` routes from reflect:
- Tests passed → `save_trajectory` (success)
- Attempts exhausted → `save_trajectory` (exhausted)
- Otherwise → `generate` (retry)

## Relationship to Pipeline

The outer 5-phase pipeline (`scripts/rune_runner.py`) uses `create_single_iteration_graph()` for each iteration within a phase. Between iterations, the hypernetwork can generate a fresh adapter. The agent graph handles a single generate-execute-reflect cycle; the pipeline handles phase sequencing (decompose → plan → code → integrate → diagnose/repair) and iteration.
