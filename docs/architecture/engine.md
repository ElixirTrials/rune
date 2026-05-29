# Engine Architecture

The engine is a single-node LangGraph `StateGraph` (`rune.engine.graph.create_engine`): one
`step` node, entry point `step`, and a conditional self-edge `step → {continue: step,
done: END}`. Iteration is driven entirely by the edge condition; the node itself is stateless
beyond the `RunState` TypedDict (`rune.engine.state`) threaded through it.

## Per-iteration: `step_node`

Each `step_node` invocation performs one full cycle:

1. **Complexity gate.** If no subtasks exist yet and `_is_simple_task` matches (task under
   `_SIMPLE_WORD_LIMIT=200` words containing a single-unit signal such as "write a function"),
   a synthetic `_main` subtask is injected and its plan pre-seeded (`plans["_main"] =
   task`), so both decomposition and planning are skipped and the first action is `code`.
2. **Action selection.** `select_action` (`rune.engine.policy`) deterministically returns the
   list of `Action`s for this step — possibly several sibling actions targeting independent
   subtasks in the same execution layer. An empty list signals termination.
3. **Per-action generation** (sequential over the returned actions):
   - `state_to_ctx` builds the Jinja2 context (subtask, plan, existing code, error summary,
     diagnosis-derived `fix_guidance`, capped `repair_history` and `code_trajectory`).
   - `render_template` (`rune.engine.parse`) renders the action's `trajectory_template` and
     `prompt_template`.
   - `model.generate_adapter(trajectory_text)` synthesizes a LoRA adapter from the trajectory
     text via the HyperLoRA hypernetwork (`rune.model`).
   - `scale_lora_b` (`rune.model.adapter`) scales the adapter's `lora_B` weights by
     `adapter_scaling`, then `model.hotswap_adapter` swaps the scaled state dict into the base
     model.
   - `model.generate` runs xgrammar-constrained decoding (with a thinking phase) against the
     action's `output_schema`.
4. **Sandbox execution.** Code-bearing actions (`Action.executes_code`) have their output
   passed once through `extract_partial_code`; the extracted source is dispatched across all
   sibling code actions concurrently via `asyncio.gather` over
   `asyncio.to_thread(run_in_sandbox, …)` (`rune.sandbox.executor`), yielding per-subtask
   `Feedback`.
5. **Parse and state update.** `parse_output` is applied per action against an accumulating
   `running` snapshot (each sibling sees the prior sibling's applied change, avoiding
   stale-snapshot clobbering). The merged updates append `StepRecord`s to `trajectory`,
   advance `step`, and decrement `budget_remaining` by one.

## Deterministic action sequence

`select_action` realizes the sequence `decompose → plan → code → [diagnose → repair]* →
integrate` as a priority cascade over `RunState`:

- **No subtasks** → `decompose`.
- **Unplanned subtasks** → `plan` for every subtask in the first ready execution layer.
- **Failing subtasks** (`not code_passed[name]`) → for each subtask in the first ready layer
  whose dependencies all pass:
  - no code yet, or `retries ≥ MAX_REPAIRS (2)` → `code` (regenerate from scratch);
  - else if a diagnosis exists → `repair`;
  - else → `diagnose`.
- **All subtasks pass** → `integrate`; if integration already failed, `diagnose` then re-
  `integrate`; once `integrated_code` is set, return `[]` (done).

Repair budgeting: `MAX_REPAIRS=2`, `MAX_RETRIES=2·MAX_REPAIRS=4`. A subtask at `MAX_RETRIES`
is exhausted and dropped from selection. If all repairable work is exhausted and integration
is still failing, `select_action` returns `[]` to stop rather than burn the budget on a
non-converging `integrate ↔ diagnose` loop.

## DAG execution layers

`build_execution_layers` uses `graphlib.TopologicalSorter` over `Subtask.depends_on`. Edges
are restricted to known subtask names so a phantom dependency cannot inject a node or block
readiness. `get_ready` yields each topological layer; `select_action` always operates on
`layers[0]` (the current ready frontier) and emits one action per subtask in it, which the
caller dispatches as parallel siblings. A `CycleError` degrades gracefully: all subtasks are
treated as independent (empty dependency sets), so the run proceeds rather than deadlocking.

## Continuation sub-loop

When `model.generate` reports a truncated result for a `code`, `repair`, or `integrate`
action, `step_node` enters an in-iteration continuation loop to recover the cut-off code. Each
round re-renders the `code_continue` trajectory with the `accumulated_code` and a 4-line
`resume_tail`, regenerates an adapter scaled by `adapter_scaling · cont_multiplier (≈1.53)`,
and calls `model.generate_continuation` with the accumulated code as assistant prefix and a
code-only system prompt. The loop bounds itself by `cont_budget (5)`, exits early on
`degeneration_score > 0.5` (`rune.engine.continuation`), two consecutive empty rounds, a
`validate_syntax` pass, or a non-truncated round. The stitched code is re-wrapped as
`{"code": …}` for downstream parsing. See [Model](model.md) for adapter synthesis and
continuation scaling.

## Termination

`should_continue` returns `done` when `actions` is empty (policy signals no remaining work) or
`budget_remaining ≤ 0`, routing the conditional edge to `END`; otherwise `continue` re-enters
`step`.

See [API Reference](../api/engine/graph.md) for implementation details.
