Based on a thorough read of the full spec, here is my peer review: 

***

## Peer Review: Rune v2 Simplified Single-Loop Architecture

**Overall verdict:** The simplification direction is sound and the architecture is coherent. The core loop maps cleanly to the stated paper abstraction. However, there are several logical and algorithmic issues that deserve attention before implementation.

***

## Critical Issues

### 1. Deadlock / Infinite Loop in Policy Logic

The `select_action` policy has a latent infinite loop. Consider a state where subtasks exist, none need planning (`needs_planning = False`), none need coding (`needs_coding = False`), and `all_coded` is also `False` (e.g., some subtasks are planned but have no code yet). The function falls through to `return ACTIONS["done"]` and signals completion — **silently abandoning work** — rather than looping. This is a logic gap, not a graceful termination. 

More precisely, the guards `needs_planning`, `needs_coding`, and `all_coded` are not mutually exhaustive. A subtask can be in the state "has a plan, has no code, no failure feedback," which satisfies none of the three conditions. The policy must either enforce a total ordering or raise an explicit error for unhandled states.

### 2. `generate_adapter_node` Signature Mismatch

`generate_adapter_node` calls `pool.generate_adapter(state["trajectory_text"])` and then `registry.register(adapter, parent=state["current_adapter"])` — but `registry.register()` requires five named arguments (`path`, `action`, `session_id`, `generation`) beyond `adapter_id` and `parent_id`. None of these are populated in the node snippet. This is an incomplete interface binding, not just a stub. 

### 3. `render_trajectory` and `render_prompt` are Identical

Both functions are defined as:
```python
def render_trajectory(template_name: str, **kwargs) -> str:
    return _env.get_template(f"{template_name}.j2").render(**kwargs)

def render_prompt(template_name: str, **kwargs) -> str:
    return _env.get_template(f"{template_name}.j2").render(**kwargs)
```
They are byte-for-byte identical. If there is no behavioral distinction now, this is dead duplication. If they are *meant* to diverge (e.g., trajectory templates truncate context, prompt templates inject system metadata), that divergence must be specified — otherwise one function should be deleted. 

***

## Moderate Issues

### 4. `all_coded` Semantics Are Ambiguous

`all_coded` is documented as "all subtasks have *passing* code". This means a subtask with code that **never executed** (e.g., `execute_node` skipped due to `executes_code=False` — which doesn't apply here, but consider network failures or budget exhaustion mid-run) would block `all_coded` forever. The definition of "passing" should be explicit: does it require `exit_code == 0`, or `tests_passed == test_count`, or simply non-empty `code` dict entry? 

### 5. Budget Decrement Without Granularity

`budget_remaining` is decremented in `update_state_node`, but no spec exists for *how much* per action. If all actions cost 1 step, `decompose` (one-time) consumes the same budget as `code_retry` (potentially many times). A flat budget can cause premature termination on large tasks or make HPO of `max_steps` unstable because the effective cost per task varies non-linearly with the number of subtasks. Consider a weighted budget or separate retry counter. 

### 6. Sync/Async Node Inconsistency

`generate_output_node` is `async def`, while all other nodes are synchronous. LangGraph supports mixed sync/async nodes, but this requires the graph to be invoked via `ainvoke()` consistently. The `should_continue` conditional edge and the `execute_node` are both synchronous — if `execute_node` wraps a blocking subprocess (sandboxed code execution with up to 30s timeout), running this inside an async LangGraph event loop will block the event loop unless it's wrapped with `asyncio.to_thread()` or similar. This is a classic async/sync contamination bug. 

### 7. XGrammar + Thinking Token Mutual Exclusion Not Enforced Structurally

The spec correctly notes that XGrammar and Qwen3 thinking tokens are mutually exclusive and states "the `ModelPool` enforces this". However, there is no structural enforcement in the `Action` dataclass — an `Action` with `output_schema != None` can be constructed without the caller knowing thinking must be disabled. This invariant should be enforced at the `generate()` call site with an explicit assertion or inside `Action.__post_init__`, not left as a documentation note. 

***

## Minor Issues

### 8. `RunConfig.from_trial` Silently Overrides HPO Parameters

```python
@classmethod
def from_trial(cls, trial, **overrides) -> RunConfig:
    return cls(
        adapter_scaling=trial.suggest_float(...),
        temperature=trial.suggest_float(...),
        **overrides,
    )
```
If `overrides` contains `adapter_scaling` or `temperature` (e.g., passed from `base_config.model_dump()`), Python will raise a `TypeError: duplicate keyword argument` at runtime. The `**overrides` dict must explicitly exclude the keys already set by the trial, or be applied first with trial suggestions taking precedence. 

### 9. `plan` Action Has `output_schema=None` but Consumes Structured Data

The `plan` action produces plan text stored `keyed by subtask name` in `plans: dict[str, str]`. Without a schema, parsing the subtask-keyed output from free-form LLM text is fragile and the extraction logic lives implicitly in `update_state_node`. This is the same structural problem that `diagnose` correctly solves with `DiagnoseResult`. Consider a `PlanResult` schema — or at minimum document the parsing contract explicitly. 

### 10. `AdapterRegistry.register` Signature in Node vs. Interface

The registry interface declares `register(adapter_id, *, path, parent_id, action, session_id, generation)` but `generate_adapter_node` calls `registry.register(adapter, parent=state["current_adapter"])` — using positional argument and `parent` instead of `parent_id`. This is a naming inconsistency that will raise a `TypeError` at runtime. 

***

## Design Observations (Non-Blocking)

- **Swarm/evolution removal is risky without explicit benchmarking.** The spec lists swarm orchestration and TIES/DARE merging as "cut," but these may carry performance on hard benchmarks. The risk section mentions `PRODUCT.md` but not a regression baseline. 
- **`RunState` grows unboundedly.** `trajectory: list[StepRecord]` and `code: dict[str, str]` accumulate all history with no eviction policy. For long runs (`max_steps=30`, many subtasks), this could exhaust context windows when `trajectory_text` is rendered. A truncation or summarization strategy should be specified. 
- **HPO surfaces are correctly separated.** Training HPO (minimize `eval_loss`) and engine HPO (maximize `pass_at_1`) with independent Optuna studies and MLflow nested runs is well-designed. 
- **The `done` action in `ACTIONS` registry is missing.** `select_action` returns `ACTIONS["done"]` but the `ACTIONS` dict shown only defines `decompose`, `plan`, `code`, `code_retry`, `integrate`, `diagnose`. This would raise a `KeyError`. 